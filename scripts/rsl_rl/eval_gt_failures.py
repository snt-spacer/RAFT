# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation driver for GroundTruth-trained policies under controlled thruster failures.

Iterates over k = 0, 1, ..., max_failures. For each k the environment is forced to sample
exactly k failed thrusters at every episode reset (drawn uniformly via the per-env RNG already
built into the task). Reports position-distance metrics and a success rate per k.

The GroundTruth env concatenates the real failure mask into the policy observation, so this
script works with any standard PPO actor — no RMA encoder is required.

Output files (written next to the checkpoint, or to --output_dir):
  eval_gt_failures.json         — per-k summary metrics
  eval_gt_failures_episodes.csv — one row per episode: k, episode_id, env_id, n_steps,
                                   success, min_pos_m, final_pos_m, is_partial
  eval_gt_failures_trajectories.csv — one row per step: k, episode_id, env_id, step,
                                       pos_dist_m, success

Typical usage::

    python scripts/rsl_rl/eval_gt_failures.py \\
        --task Isaaclab-RANSv2-GroundTruth-Position-v0 \\
        env.robot_name=CuboThrusterFailure env.task_name=GoToPositionRMA \\
        --checkpoint logs/rsl_rl/<experiment>/<run>/model_1000.pt \\
        --num_envs 512 \\
        --max_failures 4 \\
        --eval_episodes_per_env 5 \\
        --pos_tol 0.05 \\
        --success_steps 50 \\
        --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(
    description="Per-failure-count evaluation for GroundTruth-trained policies."
)
parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaaclab-RANSv2-GroundTruth-Position-v0",
    help="Gym task id.",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent cfg entry point.")
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")

# -- Checkpoint ---------------------------------------------------------------- #
parser.add_argument("--run", type=str, default=None, help="Run directory (used when --checkpoint is a filename).")
parser.add_argument("--checkpoint", type=str, default=None, help="Path or filename of the checkpoint to load.")

# -- Eval protocol ------------------------------------------------------------- #
parser.add_argument(
    "--max_failures",
    type=int,
    default=None,
    help="Maximum number of failed thrusters to test (defaults to robot cfg value).",
)
parser.add_argument(
    "--eval_episodes_per_env",
    type=int,
    default=5,
    help="Approximate number of full episodes per environment per failure level.",
)
parser.add_argument("--pos_tol", type=float, default=0.05, help="Position tolerance for success (m).")
parser.add_argument(
    "--success_steps",
    type=int,
    default=50,
    help="Consecutive in-tolerance steps required to count an episode as successful.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory for output files. Defaults to the checkpoint's directory.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after the simulator is up."""

import csv
import json
import os
import statistics
import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Isaaclab_RANSv2.tasks  # noqa: F401


def _resolve_checkpoint(experiment_name: str) -> str:
    checkpoint = args_cli.checkpoint
    if checkpoint is not None and os.path.isfile(checkpoint):
        return os.path.abspath(checkpoint)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))
    run_dir = args_cli.run if args_cli.run is not None else ".*"
    ckpt_pattern = checkpoint if checkpoint is not None else "model_.*.pt"
    return get_checkpoint_path(log_root, run_dir, ckpt_pattern)


def _force_reset_all(env) -> None:
    """Expire every env's episode so the next step triggers a full reset."""
    base = env.unwrapped
    base.episode_length_buf = torch.full_like(base.episode_length_buf, base.max_episode_length - 1)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    import importlib.metadata as metadata

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    resume_path = _resolve_checkpoint(agent_cfg.experiment_name)
    print(f"[Eval] Checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    tmp_log_dir = os.path.join(os.path.dirname(resume_path), "eval_gt_tmp")
    os.makedirs(tmp_log_dir, exist_ok=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=tmp_log_dir, device=agent_cfg.device)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=agent_cfg.device)

    base = env.unwrapped
    task = base.task_api
    if not hasattr(task, "force_failure_count"):
        raise RuntimeError(
            "Task does not expose force_failure_count — it must use ThrusterFailureTaskMixin "
            "(e.g. GoToPositionRMATask)."
        )

    robot_cfg = base.robot_api._robot_cfg
    max_k = (
        args_cli.max_failures
        if args_cli.max_failures is not None
        else getattr(robot_cfg, "failure_curriculum_max_failures", robot_cfg.num_thrusters)
    )
    max_k = min(max_k, robot_cfg.num_thrusters)

    max_ep_len = base.max_episode_length
    steps_per_k = int(max_ep_len * args_cli.eval_episodes_per_env)

    print(
        f"[Eval] num_envs={env.num_envs}  max_episode_length={max_ep_len}  "
        f"steps_per_k={steps_per_k}  k_range=0..{max_k}"
    )
    print(
        f"[Eval] success criteria: pos_dist < {args_cli.pos_tol} m sustained >= "
        f"{args_cli.success_steps} consecutive steps"
    )

    out_dir = args_cli.output_dir or os.path.dirname(resume_path)
    os.makedirs(out_dir, exist_ok=True)
    episodes_csv_path = os.path.join(out_dir, "eval_gt_failures_episodes.csv")
    trajectories_csv_path = os.path.join(out_dir, "eval_gt_failures_trajectories.csv")

    results: dict[int, dict] = {}
    global_episode_id = 0

    # Open CSV writers upfront so we can stream rows as they are collected.
    ep_csv_f = open(episodes_csv_path, "w", newline="")
    traj_csv_f = open(trajectories_csv_path, "w", newline="")
    ep_writer = csv.writer(ep_csv_f)
    traj_writer = csv.writer(traj_csv_f)
    ep_writer.writerow(["k", "episode_id", "env_id", "n_steps", "success", "min_pos_m", "final_pos_m", "is_partial"])
    traj_writer.writerow(["k", "episode_id", "env_id", "step", "pos_dist_m", "success"])

    for k in range(max_k + 1):
        task.force_failure_count(k)
        # Expire all episodes so they reset (and resample failure masks) on the next step.
        _force_reset_all(env)

        obs = env.get_observations()
        if isinstance(obs, tuple):
            obs = obs[0]

        # Warmup step: _force_reset_all sets episode_length_buf = max_ep_len - 1, so the first
        # env.step() triggers a mass timeout for every env simultaneously. Take one step here to
        # flush those forced resets before starting collection — otherwise all N envs fire as
        # "done" on step 0 with stale pre-reset positions, polluting the metrics.
        policy.reset()  # zero all RNN hidden states before a new collection block
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        if isinstance(obs, tuple):
            obs = obs[0]
        policy.reset()  # zero again: the warmup episode is not a real episode

        # Snapshot the current pos_dist as the baseline for the first collection step.
        # After the warmup, every env has genuinely reset with k failures and its new episode
        # has already begun (episode_length_buf == 1), so pos_dist is valid.
        pos_dist = task._position_dist.clone()

        # Per-env episode trackers.
        sustained = torch.zeros(env.num_envs, device=agent_cfg.device)
        episode_success = torch.zeros(env.num_envs, dtype=torch.bool, device=agent_cfg.device)
        ep_min_pos = torch.full((env.num_envs,), float("inf"), device=agent_cfg.device)

        # Per-env trajectory buffer: list of pos_dist values for the current episode.
        ep_traj: list[list[float]] = [[] for _ in range(env.num_envs)]

        completed_success: list[bool] = []
        completed_final_pos: list[float] = []
        completed_min_pos: list[float] = []

        for _step in range(steps_per_k):
            # Snapshot before stepping: Isaac Lab resets done envs *inside* env.step(), so
            # task._position_dist for done envs reflects the POST-RESET initial distance
            # (~2 m from a fresh spawn) rather than the last frame of the completed episode.
            prev_pos_dist = pos_dist.clone()

            with torch.inference_mode():
                actions = policy(obs)
                obs, _rew, dones, _extras = env.step(actions)
                # Mirror what PPO training does: zero the RNN hidden state for terminated envs
                # so the next episode starts with clean memory. Must stay inside inference_mode
                # because the hidden state tensor was created here and cannot be modified
                # in-place from outside an inference context.
                if (dones > 0).any():
                    policy.reset(dones)

            pos_dist = task._position_dist.clone()

            done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)

            # For done envs substitute the pre-step distance so that sustained/success/min
            # all reflect the true last frame rather than the post-reset spawn distance.
            effective_pos = pos_dist.clone()
            if done_ids.numel() > 0:
                effective_pos[done_ids] = prev_pos_dist[done_ids]

            in_tol = effective_pos < args_cli.pos_tol
            sustained = torch.where(in_tol, sustained + 1, torch.zeros_like(sustained))
            episode_success |= sustained >= args_cli.success_steps
            ep_min_pos = torch.minimum(ep_min_pos, effective_pos)

            # Append this step's pos_dist to each env's trajectory buffer.
            effective_pos_list = effective_pos.cpu().tolist()
            for i in range(env.num_envs):
                ep_traj[i].append(effective_pos_list[i])

            if done_ids.numel() > 0:
                done_ids_list = done_ids.cpu().tolist()
                ep_success_list = episode_success.cpu().tolist()
                ep_min_list = ep_min_pos.cpu().tolist()
                prev_pos_list = prev_pos_dist.cpu().tolist()

                for env_i in done_ids_list:
                    traj = ep_traj[env_i]
                    success_i = int(ep_success_list[env_i])
                    min_pos_i = ep_min_list[env_i]
                    final_pos_i = prev_pos_list[env_i]
                    n_steps_i = len(traj)

                    ep_writer.writerow([k, global_episode_id, env_i, n_steps_i, success_i, min_pos_i, final_pos_i, 0])
                    for step_i, pd_val in enumerate(traj):
                        traj_writer.writerow([k, global_episode_id, env_i, step_i, pd_val, success_i])

                    completed_success.append(bool(success_i))
                    completed_final_pos.append(final_pos_i)
                    completed_min_pos.append(min_pos_i)
                    global_episode_id += 1
                    ep_traj[env_i] = []

                episode_success[done_ids] = False
                sustained[done_ids] = 0.0
                ep_min_pos[done_ids] = float("inf")

        # Partial episodes: the collection loop ends right as episode 3 has just started
        # (1-2 steps in), because the warmup step consumed 1 step of episode 1's budget so
        # 399 + 400 = 799 steps are used out of 800. These fragments are logged to the CSV
        # for completeness but excluded from all metrics — they cannot possibly satisfy the
        # success criterion and would only dilute the reported success rate.
        active = (~torch.isinf(ep_min_pos)).nonzero(as_tuple=False).squeeze(-1)
        if active.numel() > 0:
            active_list = active.cpu().tolist()
            ep_success_list = episode_success.cpu().tolist()
            ep_min_list = ep_min_pos.cpu().tolist()
            pos_dist_list = pos_dist.cpu().tolist()

            for env_i in active_list:
                traj = ep_traj[env_i]
                success_i = int(ep_success_list[env_i])
                min_pos_i = ep_min_list[env_i]
                final_pos_i = pos_dist_list[env_i]
                n_steps_i = len(traj)

                # Write to CSV (is_partial=1) but do NOT add to the metric lists.
                ep_writer.writerow([k, global_episode_id, env_i, n_steps_i, success_i, min_pos_i, final_pos_i, 1])
                for step_i, pd_val in enumerate(traj):
                    traj_writer.writerow([k, global_episode_id, env_i, step_i, pd_val, success_i])
                global_episode_id += 1

        # completed_success/final_pos/min_pos contain only fully-terminated episodes.
        # Partial episodes are written to the CSV but deliberately excluded here.
        n = len(completed_success)
        n_success = sum(completed_success)
        success_rate = n_success / n if n > 0 else float("nan")

        min_pos_succ   = [v for s, v in zip(completed_success, completed_min_pos)   if s]
        min_pos_fail   = [v for s, v in zip(completed_success, completed_min_pos)   if not s]
        final_pos_succ = [v for s, v in zip(completed_success, completed_final_pos) if s]
        final_pos_fail = [v for s, v in zip(completed_success, completed_final_pos) if not s]

        def _stats(vals: list[float]) -> dict:
            if not vals:
                nan = float("nan")
                return {"min": nan, "mean": nan, "max": nan}
            return {"min": min(vals), "mean": statistics.fmean(vals), "max": max(vals)}

        results[k] = {
            "num_episodes_completed": n,
            "num_success": n_success,
            "success_rate": success_rate,
            "min_pos_success_m": _stats(min_pos_succ),
            "min_pos_fail_m":    _stats(min_pos_fail),
            "final_pos_success_m": _stats(final_pos_succ),
            "final_pos_fail_m":    _stats(final_pos_fail),
        }

        r = results[k]
        sr_str = f"{success_rate * 100:5.1f}%" if success_rate == success_rate else "  nan%"
        ok  = r["min_pos_success_m"]
        fail = r["min_pos_fail_m"]
        print(
            f"[Eval] k={k}  completed={n:5d}  success={sr_str}  "
            f"min_pos(ok)  min={ok['min']:.4f} avg={ok['mean']:.4f} max={ok['max']:.4f} m  |  "
            f"min_pos(fail)  min={fail['min']:.4f} avg={fail['mean']:.4f} max={fail['max']:.4f} m"
        )

    ep_csv_f.close()
    traj_csv_f.close()

    task.force_failure_count(None)

    summary = {
        "task": args_cli.task,
        "num_envs": env.num_envs,
        "checkpoint": resume_path,
        "seed": args_cli.seed,
        "pos_tol_m": args_cli.pos_tol,
        "success_steps": args_cli.success_steps,
        "max_episode_length": int(max_ep_len),
        "eval_episodes_per_env": args_cli.eval_episodes_per_env,
        "results": {str(k): v for k, v in results.items()},
    }

    json_path = os.path.join(out_dir, "eval_gt_failures.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Eval] JSON summary  -> {json_path}")
    print(f"[Eval] Episodes CSV  -> {episodes_csv_path}")
    print(f"[Eval] Trajectories  -> {trajectories_csv_path}")

    print("\n=== Success rate by thruster failure count (completed episodes only) ===")
    hdr = f"{'k':>3}  {'completed':>9}  {'success%':>9}  {'ok_min':>8}  {'ok_avg':>8}  {'ok_max':>8}  {'fail_min':>9}  {'fail_avg':>9}  {'fail_max':>9}"
    print(hdr)
    for k in sorted(results.keys()):
        r = results[k]
        sr = r["success_rate"]
        sr_str = f"{sr * 100:5.1f}%" if sr == sr else "   nan%"
        ok   = r["min_pos_success_m"]
        fail = r["min_pos_fail_m"]
        def _fmt(v): return f"{v:.4f}" if v == v else "   nan"
        print(
            f"{k:>3}  {r['num_episodes_completed']:>9}  {sr_str:>9}  "
            f"{_fmt(ok['min']):>8}  {_fmt(ok['mean']):>8}  {_fmt(ok['max']):>8}  "
            f"{_fmt(fail['min']):>9}  {_fmt(fail['mean']):>9}  {_fmt(fail['max']):>9}"
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
