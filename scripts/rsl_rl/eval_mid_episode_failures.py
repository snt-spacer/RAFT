"""E9 — Mid-episode failure injection evaluation.

Tests whether trained policies can adapt to failures that occur *during* an episode rather than
at reset time.  Episodes start healthy (k=0); at a configurable injection step, k thrusters are
suddenly degraded.  Only episodes that were alive at the injection moment contribute to the
post-injection SR — this isolates the adaptation ability from reset-time performance.

Works with any task that exposes ``DegradationStateTaskMixin`` (Observer, GroundTruth, Vanilla):

  - OBS:  history buffer accumulates the new failure signature; D_hat adapts gradually.
  - GT:   privileged obs (D_gt) is updated atomically at injection; perfect instant knowledge.
  - VAN:  no failure information; relies on the nominal policy alone.

Output:
  eval_mid_episode.json              — per-k metrics (pre- and post-injection SR)
  eval_mid_episode_episodes.csv      — one row per post-injection episode

Usage::

    python scripts/rsl_rl/eval_mid_episode_failures.py \\
        --task Isaaclab-RANSv2-Observer-Position-v0 \\
        --checkpoint logs/rsl_rl/Observer_GoToPosition/<run>/model_4999.pt \\
        --num_envs 512 --max_failures 4 --eval_episodes_per_env 10 \\
        --inject_step 100 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Mid-episode failure injection evaluation.")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--task", type=str, default="Isaaclab-RANSv2-Observer-Position-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--max_failures", type=int, default=4)
parser.add_argument("--eval_episodes_per_env", type=int, default=10)
parser.add_argument("--inject_step", type=int, default=100,
                    help="Step within the episode at which failures are injected (0-indexed).")
parser.add_argument("--pos_tol", type=float, default=0.05)
parser.add_argument("--success_steps", type=int, default=50)
parser.add_argument("--output_dir", type=str, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ────────────────────────────────────────────────────────

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
from isaaclab_tasks.utils.hydra import hydra_task_config

import Isaaclab_RANSv2.tasks  # noqa: F401


def _force_reset_all(env) -> None:
    base = env.unwrapped
    base.episode_length_buf = torch.full_like(base.episode_length_buf, base.max_episode_length - 1)


def _inject_failures(task, robot, k: int, num_envs: int, device: str) -> None:
    """Atomically inject k failures into all envs mid-episode.

    1. Calls the task's own degradation sampler so failure modes match training distribution.
    2. Pushes the new state to the robot (affects physics immediately).
    3. Updates task's D_gt tensors so GT policy sees the correct privileged obs.
    """
    all_ids = torch.arange(num_envs, device=device)
    task.force_failure_count(k)

    if hasattr(task, "_sample_degradation_state"):
        new_scales, new_offsets = task._sample_degradation_state(all_ids)
        robot.set_degradation_state(new_scales, new_offsets)
        if hasattr(task, "_degradation_scales") and task._degradation_scales is not None:
            task._degradation_scales.copy_(new_scales)
            task._degradation_offsets.copy_(new_offsets)
    elif hasattr(task, "_sample_failure_mask"):
        # Older binary-mask mixin fallback.
        mask = task._sample_failure_mask(all_ids)
        robot.set_failure_mask(mask, None)
        if hasattr(task, "_failure_mask") and task._failure_mask is not None:
            task._failure_mask.copy_(mask)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    import importlib.metadata as metadata

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    checkpoint = os.path.abspath(args_cli.checkpoint)
    print(f"[E9] Checkpoint:   {checkpoint}")
    print(f"[E9] Task:         {args_cli.task}")
    print(f"[E9] inject_step:  {args_cli.inject_step}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    tmp_log_dir = os.path.join(os.path.dirname(checkpoint), "e9_tmp")
    os.makedirs(tmp_log_dir, exist_ok=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=tmp_log_dir, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=agent_cfg.device)

    base = env.unwrapped
    task = base.task_api
    robot = base.robot_api
    device = agent_cfg.device
    num_envs = env.num_envs

    if not (hasattr(task, "_sample_degradation_state") or hasattr(task, "_sample_failure_mask")):
        raise RuntimeError("Task must use DegradationStateTaskMixin or ThrusterFailureTaskMixin.")

    robot_cfg = robot._robot_cfg
    max_k = min(args_cli.max_failures, robot_cfg.num_thrusters)
    max_ep_len = base.max_episode_length
    inject_step = args_cli.inject_step

    if inject_step >= max_ep_len:
        raise ValueError(f"inject_step={inject_step} >= max_episode_length={max_ep_len}.")

    post_steps_budget = int(max_ep_len * args_cli.eval_episodes_per_env)

    out_dir = args_cli.output_dir or os.path.dirname(checkpoint)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "eval_mid_episode.json")
    csv_path = os.path.join(out_dir, "eval_mid_episode_episodes.csv")

    print(f"[E9] num_envs={num_envs}  max_ep_len={max_ep_len}  inject_step={inject_step}")
    print(f"[E9] post_steps_budget per k: {post_steps_budget}")

    results: dict[int, dict] = {}

    ep_csv_f = open(csv_path, "w", newline="")
    ep_writer = csv.writer(ep_csv_f)
    ep_writer.writerow(["k", "episode_id", "env_id", "n_steps_post", "success", "min_pos_post_m", "final_pos_m"])
    global_ep_id = 0

    for k in range(1, max_k + 1):
        print(f"\n[E9] k={k} ──────────────────────────────────────────")

        # ── Phase 0: warm start — reset all envs healthy ─────────────────────
        task.force_failure_count(0)
        _force_reset_all(env)
        obs = env.get_observations()
        if isinstance(obs, tuple):
            obs = obs[0]
        policy.reset()
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        if isinstance(obs, tuple):
            obs = obs[0]
        policy.reset()

        # ── Phase 1: run inject_step steps with k=0 ──────────────────────────
        task.force_failure_count(0)
        step_in_ep = torch.zeros(num_envs, device=device, dtype=torch.long)

        for _s in range(inject_step):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _rew, dones, _extras = env.step(actions)
                if (dones > 0).any():
                    policy.reset(dones)
            if isinstance(obs, tuple):
                obs = obs[0]
            step_in_ep += 1
            step_in_ep = torch.where(dones > 0, torch.zeros_like(step_in_ep), step_in_ep)

        # ── Injection moment ─────────────────────────────────────────────────
        print(f"  Injecting k={k} failures at step {inject_step}...")
        _inject_failures(task, robot, k, num_envs, device)
        # From here, resets will also use k failures (so post-injection episodes start with k).
        policy.reset()  # clear RNN hidden state — new failure context begins

        # Mark which envs were alive (mid-episode) at injection.
        alive_at_injection = (step_in_ep > 0)
        print(f"  Envs alive at injection: {alive_at_injection.sum().item()} / {num_envs}")

        # ── Phase 2: run post-injection steps ────────────────────────────────
        pos_dist = task._position_dist.clone()
        sustained = torch.zeros(num_envs, device=device)
        episode_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        ep_min_pos = torch.full((num_envs,), float("inf"), device=device)
        steps_since_inject = torch.zeros(num_envs, device=device, dtype=torch.long)

        # Track which envs contributed a "post-injection" episode to results.
        contributed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # First post-injection episode: the one alive at injection, plus post-injection resets.
        post_successes: list[bool] = []
        post_min_pos: list[float] = []
        post_final_pos: list[float] = []
        post_env_ids: list[int] = []

        for _s in range(post_steps_budget):
            prev_pos_dist = pos_dist.clone()

            with torch.inference_mode():
                actions = policy(obs)
                obs, _rew, dones, _extras = env.step(actions)
                if (dones > 0).any():
                    policy.reset(dones)
            if isinstance(obs, tuple):
                obs = obs[0]

            pos_dist = task._position_dist.clone()
            effective_pos = pos_dist.clone()
            done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                effective_pos[done_ids] = prev_pos_dist[done_ids]

            in_tol = effective_pos < args_cli.pos_tol
            sustained = torch.where(in_tol, sustained + 1, torch.zeros_like(sustained))
            episode_success |= sustained >= args_cli.success_steps
            ep_min_pos = torch.minimum(ep_min_pos, effective_pos)
            steps_since_inject += 1

            if done_ids.numel() > 0:
                for env_id in done_ids.cpu().tolist():
                    post_successes.append(bool(episode_success[env_id].item()))
                    post_min_pos.append(float(ep_min_pos[env_id].item()))
                    post_final_pos.append(float(prev_pos_dist[env_id].item()))
                    post_env_ids.append(env_id)
                    ep_writer.writerow([
                        k, global_ep_id, env_id,
                        int(steps_since_inject[env_id].item()),
                        int(episode_success[env_id].item()),
                        float(ep_min_pos[env_id].item()),
                        float(prev_pos_dist[env_id].item()),
                    ])
                    global_ep_id += 1

                # Reset trackers for completed envs.
                episode_success[done_ids] = False
                ep_min_pos[done_ids] = float("inf")
                sustained[done_ids] = 0
                steps_since_inject[done_ids] = 0

        # Flush any still-running episodes as partial results.
        for env_id in range(num_envs):
            if ep_min_pos[env_id] < float("inf"):
                post_successes.append(bool(episode_success[env_id].item()))
                post_min_pos.append(float(ep_min_pos[env_id].item()))
                post_final_pos.append(float(pos_dist[env_id].item()))
                post_env_ids.append(env_id)

        sr = statistics.mean(post_successes) if post_successes else float("nan")
        n_eps = len(post_successes)
        results[k] = {
            "success_rate": sr,
            "n_episodes": n_eps,
            "inject_step": inject_step,
            "k": k,
        }
        print(f"  k={k}  SR_post_injection={sr*100:.1f}%  n_episodes={n_eps}")

    ep_csv_f.close()

    # ── Save JSON ────────────────────────────────────────────────────────────
    out = {
        "inject_step": inject_step,
        "max_episode_length": max_ep_len,
        "pos_tol": args_cli.pos_tol,
        "success_steps": args_cli.success_steps,
        "task": args_cli.task,
        "results": {str(k): v for k, v in results.items()},
    }
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[E9] Saved → {json_path}")

    print("\n[E9] ══════════  SUMMARY  ══════════")
    print(f"{'k':>4}  {'SR post-injection':>20}  {'n_eps':>8}")
    for k, v in results.items():
        print(f"{k:>4}  {v['success_rate']*100:>19.1f}%  {v['n_episodes']:>8}")


if __name__ == "__main__":
    main()
    simulation_app.close()
