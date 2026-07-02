"""Severity-sweep evaluation for a trained Observer policy.

Evaluates a single checkpoint at a fixed failure count (k=1) and a pinned severity value,
sweeping the severity parameter across a caller-specified list.  Outputs one JSON file per
severity level, compatible with the format written by ``eval_gt_failures.py``.

Usage::

    python scripts/experiments/eval_severity.py \\
        --task Isaaclab-RANSv2-Observer-Position-v0 \\
        --checkpoint logs/rsl_rl/Observer_GoToPosition/<run>/model_4999.pt \\
        --mode deg \\
        --severities 0.9 0.7 0.5 0.3 0.1 \\
        --output_dir docs/results/e3_severity_sweep/deg \\
        --num_envs 512 --eval_episodes_per_env 10 \\
        --pos_tol 0.05 --success_steps 50 --headless

Mode options:
    deg   — continuous degradation; severity controls ``degradation_scale_range`` (scale=severity)
    stk   — stuck-on; severity controls ``stuck_offset_range`` (offset=severity)
"""

import argparse
import os
import sys

# cli_args.py lives next to the other rsl_rl scripts, not in this directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Fixed-severity evaluation for the Observer policy.")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--task", type=str, default="Isaaclab-RANSv2-Observer-Position-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--mode", type=str, choices=["deg", "stk"], required=True,
                    help="Failure mode to sweep: 'deg' (continuous) or 'stk' (stuck-on).")
parser.add_argument("--severities", type=float, nargs="+", required=True,
                    help="List of severity values to evaluate (scale or offset).")
parser.add_argument("--eval_episodes_per_env", type=int, default=10)
parser.add_argument("--pos_tol", type=float, default=0.05)
parser.add_argument("--success_steps", type=int, default=50)
parser.add_argument("--output_dir", type=str, default="docs/results/e3_severity_sweep")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
import importlib.metadata as metadata
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


def _run_episodes(env, policy, num_steps: int) -> list[dict]:
    """Roll out episodes for ``num_steps`` steps; return per-episode stats."""
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    task = env.unwrapped.task_api

    episodes: list[dict] = []

    obs = env.get_observations()

    # Warmup step: flush the mass timeouts triggered by the force-reset above.
    # Double reset mirrors eval_gt_failures.py: once before to zero RNN state, once after
    # so the warmup episode doesn't bleed into the real collection.
    policy.reset()
    with torch.inference_mode():
        obs, _, _, _ = env.step(policy(obs))
    policy.reset()
    pos_dist = task._position_dist.clone()

    sustained = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    ep_min_pos = torch.full((num_envs,), float("inf"), device=device)

    with torch.inference_mode():
        for _ in range(num_steps):
            prev_pos_dist = pos_dist.clone()  # snapshot BEFORE step

            actions = policy(obs)
            obs, _rew, dones, _extras = env.step(actions)

            done_mask = dones > 0
            if done_mask.any():
                policy.reset(dones)

            pos_dist = task._position_dist.clone()

            # Use pre-step position for envs that just finished (post-step reflects new episode)
            effective_pos = pos_dist.clone()
            done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                effective_pos[done_ids] = prev_pos_dist[done_ids]

            in_tol = effective_pos < args_cli.pos_tol
            sustained = torch.where(in_tol, sustained + 1.0, torch.zeros_like(sustained))
            episode_success |= sustained >= args_cli.success_steps  # latched
            ep_min_pos = torch.minimum(ep_min_pos, effective_pos)

            if done_ids.numel() > 0:
                for i in done_ids.tolist():
                    episodes.append({
                        "success": int(episode_success[i].item()),
                        "final_pos_m": effective_pos[i].item(),
                        "min_pos_m": ep_min_pos[i].item(),
                    })
                episode_success[done_ids] = False
                sustained[done_ids] = 0.0
                ep_min_pos[done_ids] = float("inf")

    return episodes


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    os.makedirs(args_cli.output_dir, exist_ok=True)

    # Build env once; we will modify the task cfg between severity sweeps
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    num_steps = int(args_cli.eval_episodes_per_env * env.unwrapped.max_episode_length)

    all_results = {}

    for severity in args_cli.severities:
        task = env.unwrapped.task_api
        # Pin failure count to k=1 and set the severity for this sweep step
        task.force_failure_count(1)
        if args_cli.mode == "deg":
            task._task_cfg.failure_mode_probs = (1.0, 0.0, 0.0)
            task._task_cfg.degradation_scale_range = (severity, severity)
        else:  # stk
            task._task_cfg.failure_mode_probs = (0.0, 0.0, 1.0)
            task._task_cfg.stuck_offset_range = (severity, severity)

        # Force reset all envs to pick up the new severity
        env.unwrapped.episode_length_buf = torch.full_like(
            env.unwrapped.episode_length_buf, env.unwrapped.max_episode_length - 1
        )

        episodes = _run_episodes(env, policy, num_steps)
        if not episodes:
            all_results[str(severity)] = {"success_rate": 0.0, "mean_final_pos_m": 0.0, "mean_min_pos_m": 0.0}
            continue

        sr = statistics.mean(e["success"] for e in episodes)
        fpe = statistics.mean(e["final_pos_m"] for e in episodes)
        mpe = statistics.mean(e["min_pos_m"] for e in episodes)
        all_results[str(severity)] = {
            "success_rate": sr,
            "mean_final_pos_m": fpe,
            "mean_min_pos_m": mpe,
            "num_episodes": len(episodes),
        }
        print(f"  severity={severity:.2f}  SR={sr*100:.1f}%  FPE={fpe*100:.2f}cm  MPE={mpe*100:.2f}cm")

    out_path = os.path.join(args_cli.output_dir, "severity_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
