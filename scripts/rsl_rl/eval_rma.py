# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation driver for RMA-trained policies on the GoToPoseRMA task.

Pins the per-env thruster failure count to a fixed value (``k = 0, 1, ..., max``) and reports a
success rate per ``k``. Success is measured as: position error within ``--pos_tol`` AND heading
error within ``--heading_tol``, sustained for at least ``--success_steps`` consecutive steps at
some point during the episode.

Two evaluation modes:

- **Phase 1 (default)**: the actor reads the *true* mask through ``mu``. Upper bound on
  performance — measures whether the privileged-information policy can compensate for failures.
- **Phase 2** (``--phase2_checkpoint <path>``): the actor's privileged latent ``z`` is overridden
  with ``phi(history)`` from a trained adaptation module. Mirrors deployment-time conditions.

Output: a JSON file with per-k metrics, plus a console summary table.

Typical usage:

    ./isaaclab.sh -p scripts/rsl_rl/eval_rma.py \\
        --task Isaaclab-RANSv2-RMA-v0 \\
        --num_envs 1024 \\
        --max_failures 4 \\
        --eval_episodes_per_env 4 \\
        --pos_tol 0.02 --heading_tol 0.01 --success_steps 50

    # With Phase 2 adaptation module:
    ./isaaclab.sh -p scripts/rsl_rl/eval_rma.py \\
        --task Isaaclab-RANSv2-RMA-v0 --num_envs 1024 \\
        --phase2_checkpoint logs/.../phase2/adapt_final.pt \\
        --history_length 50 --backbone conv
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Per-failure-count evaluation of an RMA policy.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel envs.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaaclab-RANSv2-RMA-v0",
    help=(
        "Gym task id. Use Isaaclab-RANSv2-RMA-Position-v0 for the position-only task; in that "
        "case --heading_tol is ignored."
    ),
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent cfg entry point.")
parser.add_argument("--seed", type=int, default=42, help="Env seed.")

# -- Phase-1 checkpoint --------------------------------------------------- #
parser.add_argument("--phase1_run", type=str, default=None, help="Phase-1 run directory.")
parser.add_argument("--phase1_checkpoint", type=str, default=None, help="Phase-1 checkpoint filename.")

# -- Phase-2 checkpoint (optional) --------------------------------------- #
parser.add_argument(
    "--phase2_checkpoint",
    type=str,
    default=None,
    help="Path to a Phase-2 adaptation-module checkpoint. If omitted, eval uses the true mask via mu.",
)
parser.add_argument("--history_length", type=int, default=50, help="History length for the adaptation module.")
parser.add_argument(
    "--backbone", type=str, default="conv", choices=["conv", "gru"], help="Adaptation-module backbone."
)
parser.add_argument("--gru_hidden_dim", type=int, default=128, help="GRU hidden dim (used when backbone=gru).")

# -- Eval protocol -------------------------------------------------------- #
parser.add_argument("--max_failures", type=int, default=None, help="Max k to evaluate (defaults to robot cfg).")
parser.add_argument(
    "--eval_episodes_per_env",
    type=int,
    default=2,
    help="Roughly how many full episodes per env per failure level (drives the run length).",
)
parser.add_argument("--pos_tol", type=float, default=0.02, help="Position tolerance for success (m).")
parser.add_argument("--heading_tol", type=float, default=0.01, help="Heading tolerance for success (rad).")
parser.add_argument(
    "--success_steps",
    type=int,
    default=50,
    help="Required number of consecutive in-tolerance steps to count an episode as successful.",
)
parser.add_argument(
    "--output", type=str, default=None, help="Where to dump the JSON summary. Defaults to <phase1_run>/eval_rma.json."
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import statistics
import torch
import gymnasium as gym

from rsl_rl.modules import RMAAdaptationModule, RMAHistoryBuffer
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Isaaclab_RANSv2.tasks  # noqa: F401


def _resolve_phase1_checkpoint(experiment_name: str) -> str:
    checkpoint = args_cli.phase1_checkpoint
    if checkpoint is not None and os.path.isfile(checkpoint):
        return os.path.abspath(checkpoint)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))
    run_dir = args_cli.phase1_run if args_cli.phase1_run is not None else ".*"
    checkpoint = checkpoint if checkpoint is not None else "model_.*.pt"
    return get_checkpoint_path(log_root_path, run_dir, checkpoint)


def _build_actor_forward(actor, adaptation_module, history_buf):
    """Return a callable ``f(obs) -> action`` that runs the actor either with the true mask
    (``mu``) or with the predicted latent from the adaptation module."""
    if adaptation_module is None:
        @torch.no_grad()
        def fwd(obs):
            return actor(obs)
        return fwd

    @torch.no_grad()
    def fwd(obs):
        # Replace mu(mask) with phi(history). We bypass actor.forward and call the body directly
        # so we don't have to monkey-patch state on the actor.
        z = adaptation_module(history_buf.obs, history_buf.actions)
        general = obs["policy"]
        general = actor.obs_normalizer(general)
        latent = torch.cat([general, z], dim=-1)
        out = actor.mlp(latent)
        if actor.distribution is not None:
            return actor.distribution.deterministic_output(out)
        return out

    return fwd


def _force_reset_all(env) -> None:
    """Force-reset every env on the next step boundary by maxing the episode-length buffer.

    Using ``env.reset()`` directly would also work but goes through the gym reset path; the
    simpler trick here is consistent with how the wrapper drives things.
    """
    base = env.unwrapped
    base.episode_length_buf = torch.full_like(base.episode_length_buf, base.max_episode_length - 1)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    # CLI overrides
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    import importlib.metadata as metadata

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    # locate Phase-1 checkpoint
    resume_path = _resolve_phase1_checkpoint(agent_cfg.experiment_name)
    print(f"[Eval] Phase-1 checkpoint: {resume_path}")

    # build env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # build runner with same architecture as Phase-1 and load weights
    tmp_log_dir = os.path.join(os.path.dirname(resume_path), "eval_tmp")
    os.makedirs(tmp_log_dir, exist_ok=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=tmp_log_dir, device=agent_cfg.device)
    runner.load(resume_path)
    actor = runner.alg.actor.to(agent_cfg.device)
    actor.eval()
    for p in actor.parameters():
        p.requires_grad = False

    if not hasattr(actor, "encode_privileged"):
        raise RuntimeError(
            "The loaded actor is not an RMAModel (no 'encode_privileged'). Phase-1 checkpoint "
            "must be trained with the RMA agent cfg."
        )

    # optional Phase-2 adaptation module
    adaptation_module = None
    history_buf = None
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs_dim = obs["policy"].shape[-1]
    action_dim = env.num_actions

    if args_cli.phase2_checkpoint is not None:
        backbone_kwargs = {}
        if args_cli.backbone == "gru":
            backbone_kwargs["hidden_dim"] = args_cli.gru_hidden_dim
        adaptation_module = RMAAdaptationModule(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=actor.latent_dim,
            history_length=args_cli.history_length,
            backbone=args_cli.backbone,
            backbone_kwargs=backbone_kwargs,
        ).to(agent_cfg.device)
        ckpt = torch.load(args_cli.phase2_checkpoint, map_location=agent_cfg.device, weights_only=False)
        adaptation_module.load_state_dict(ckpt["adaptation_module"])
        adaptation_module.eval()
        for p in adaptation_module.parameters():
            p.requires_grad = False
        history_buf = RMAHistoryBuffer(
            num_envs=env.num_envs,
            obs_dim=obs_dim,
            action_dim=action_dim,
            history_length=args_cli.history_length,
            device=agent_cfg.device,
        )
        print(f"[Eval] Loaded Phase-2 adaptation module from: {args_cli.phase2_checkpoint}")
    else:
        print("[Eval] No Phase-2 checkpoint — evaluating with the true mask via mu.")

    actor_forward = _build_actor_forward(actor, adaptation_module, history_buf)

    # task handle (for force_failure_count and metric accessors)
    base = env.unwrapped
    task = base.task_api
    if not hasattr(task, "force_failure_count"):
        raise RuntimeError(
            "Task does not expose force_failure_count — it must use the RMATaskMixin "
            "(GoToPoseRMATask, GoToPositionRMATask, ...)."
        )

    # Position-only tasks (e.g. GoToPositionRMA) don't track heading-to-target as a goal, so the
    # heading-tolerance gate doesn't apply.
    has_heading_goal = hasattr(task, "_heading_error")
    if not has_heading_goal:
        print("[Eval] Task has no heading goal — ignoring --heading_tol.")

    # robot config — figure out max k
    robot_cfg = base.robot_api._robot_cfg
    max_k = (
        args_cli.max_failures
        if args_cli.max_failures is not None
        else getattr(robot_cfg, "failure_curriculum_max_failures", robot_cfg.num_thrusters)
    )
    max_k = min(max_k, robot_cfg.num_thrusters)

    # how long to run per k
    max_ep_len = base.max_episode_length
    steps_per_k = int(max_ep_len * args_cli.eval_episodes_per_env)
    print(
        f"[Eval] num_envs={env.num_envs} max_episode_length={max_ep_len} "
        f"steps_per_k={steps_per_k} k_range=0..{max_k}"
    )
    if has_heading_goal:
        print(
            f"[Eval] success criteria: pos_dist < {args_cli.pos_tol} m AND |heading_err| < "
            f"{args_cli.heading_tol} rad sustained >= {args_cli.success_steps} steps"
        )
    else:
        print(
            f"[Eval] success criteria: pos_dist < {args_cli.pos_tol} m sustained >= "
            f"{args_cli.success_steps} steps"
        )

    results: dict[int, dict] = {}

    for k in range(max_k + 1):
        task.force_failure_count(k)
        # Force a reset on the next step so every env resamples the mask under the new k.
        _force_reset_all(env)
        # The wrapper's get_observations doesn't trigger reset; an explicit step does. Take a
        # zero/no-op step to flush. Use the actor's default action with whatever obs we have.
        obs = env.get_observations()
        if isinstance(obs, tuple):
            obs = obs[0]

        # Per-env trackers (reset at the start of each k).
        sustained = torch.zeros(env.num_envs, device=agent_cfg.device)
        episode_success = torch.zeros(env.num_envs, dtype=torch.bool, device=agent_cfg.device)
        if history_buf is not None:
            history_buf.reset()

        # Buffers we accumulate across full episodes that complete during this k's run.
        completed_success: list[bool] = []
        completed_final_pos: list[float] = []
        completed_final_heading: list[float] = []
        completed_min_pos: list[float] = []
        # Per-env running min within the current episode.
        ep_min_pos = torch.full((env.num_envs,), float("inf"), device=agent_cfg.device)

        for step in range(steps_per_k):
            action = actor_forward(obs)
            if history_buf is not None:
                history_buf.append(obs["policy"], action)

            obs, _rew, dones, _extras = env.step(action)

            # Pull live metrics off the task. After step, _position_dist (and _heading_error if
            # present) are up-to-date because get_observations runs at the end of step().
            pos_dist = task._position_dist
            in_tol = pos_dist < args_cli.pos_tol
            heading_err = (
                task._heading_error.abs() if has_heading_goal else torch.zeros_like(pos_dist)
            )
            if has_heading_goal:
                in_tol = in_tol & (heading_err < args_cli.heading_tol)
            sustained = torch.where(in_tol, sustained + 1, torch.zeros_like(sustained))
            episode_success |= sustained >= args_cli.success_steps
            ep_min_pos = torch.minimum(ep_min_pos, pos_dist)

            # On terminations, snapshot per-env outcomes and reset trackers for the new episode.
            done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                completed_success.extend(episode_success[done_ids].cpu().tolist())
                completed_final_pos.extend(pos_dist[done_ids].cpu().tolist())
                completed_final_heading.extend(heading_err[done_ids].cpu().tolist())
                completed_min_pos.extend(ep_min_pos[done_ids].cpu().tolist())

                episode_success[done_ids] = False
                sustained[done_ids] = 0
                ep_min_pos[done_ids] = float("inf")
                if history_buf is not None:
                    history_buf.reset(done_ids)

        # If we ran out of steps before some envs terminated, include their current state too.
        # Without this, very-good policies that hold the goal indefinitely are under-counted.
        active = (~torch.isinf(ep_min_pos)).nonzero(as_tuple=False).squeeze(-1)
        if active.numel() > 0:
            completed_success.extend(episode_success[active].cpu().tolist())
            completed_final_pos.extend(task._position_dist[active].cpu().tolist())
            if has_heading_goal:
                completed_final_heading.extend(task._heading_error.abs()[active].cpu().tolist())
            else:
                completed_final_heading.extend([float("nan")] * active.numel())
            completed_min_pos.extend(ep_min_pos[active].cpu().tolist())

        n = len(completed_success)
        success_rate = (sum(completed_success) / n) if n > 0 else float("nan")
        result = {
            "num_episodes": n,
            "success_rate": success_rate,
            "mean_final_pos_dist_m": statistics.fmean(completed_final_pos) if n else float("nan"),
            "mean_final_heading_err_rad": statistics.fmean(completed_final_heading) if n else float("nan"),
            "mean_min_pos_dist_m": statistics.fmean(completed_min_pos) if n else float("nan"),
        }
        results[k] = result
        print(
            f"[Eval] k={k}  n={n:5d}  success={success_rate * 100:5.1f}%  "
            f"final_pos={result['mean_final_pos_dist_m']:.3f} m  "
            f"final_hdg={result['mean_final_heading_err_rad']:.3f} rad  "
            f"min_pos={result['mean_min_pos_dist_m']:.3f} m"
        )

    # Restore default behaviour on the task in case someone keeps the env around.
    task.force_failure_count(None)

    summary = {
        "task": args_cli.task,
        "num_envs": env.num_envs,
        "phase1_checkpoint": resume_path,
        "phase2_checkpoint": args_cli.phase2_checkpoint,
        "history_length": args_cli.history_length if adaptation_module is not None else None,
        "backbone": args_cli.backbone if adaptation_module is not None else None,
        "pos_tol_m": args_cli.pos_tol,
        "heading_tol_rad": args_cli.heading_tol if has_heading_goal else None,
        "success_steps": args_cli.success_steps,
        "max_episode_length": int(max_ep_len),
        "uses_heading_goal": has_heading_goal,
        "results": {str(k): v for k, v in results.items()},
    }

    out_path = args_cli.output or os.path.join(os.path.dirname(resume_path), "eval_rma.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Eval] Wrote summary to: {out_path}")

    # Compact text table for quick scanning.
    print("\n=== Success rate by failure count ===")
    print(f"{'k':>3}  {'episodes':>8}  {'success%':>9}  {'final_pos':>10}  {'final_hdg':>10}")
    for k in sorted(results.keys()):
        r = results[k]
        sr = r["success_rate"]
        sr_str = f"{sr * 100:5.1f}" if sr == sr else "  nan"  # NaN check
        print(
            f"{k:>3}  {r['num_episodes']:>8}  {sr_str:>9}  "
            f"{r['mean_final_pos_dist_m']:>10.4f}  {r['mean_final_heading_err_rad']:>10.4f}"
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
