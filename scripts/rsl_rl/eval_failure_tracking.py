"""Failure Tracking Evaluation — collect per-step (D_gt, D_hat/hidden) trajectories.

Records observation trajectories for three failure modes × four k values.
Supports ObserverActorModel (OBS / OBS-MSE) and RNNModel (RAFT / GRU variants).

  - OBS / OBS-MSE : records D_hat = actor.get_observer_output(obs) each step.
  - RAFT          : records GRU hidden state h_t = actor.rnn.hidden_state each step.

Outputs per (mode, k):
    <out_dir>/<MODE>_k<k>.npz   — arrays:
        single_dgt   (T, 16)    — env[0] D_gt  trajectory
        single_pred  (T, D)     — env[0] D_hat or h_t trajectory
        agg_dgt      (N, 16)    — aggregate D_gt  (subsampled, all envs)
        agg_pred     (N, D)     — aggregate D_hat or h_t (subsampled)
        reset_steps  (T,)       — bool mask: env[0] reset at this step

Usage (inside Isaac Lab container):
    python scripts/rsl_rl/eval_failure_tracking.py \\
        --task Isaaclab-RANSv2-Observer-Position-v0 \\
        --agent rsl_rl_cfg_entry_point \\
        --checkpoint /root/ws/docs/paper_checkpoints/OBS_MSE/seed_42.pt \\
        --method OBS_MSE \\
        --num_envs 128 \\
        --output_dir /root/ws/docs/results/e11_failure_tracking/OBS_MSE/seed_42 \\
        --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaaclab-RANSv2-Observer-Position-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--method", type=str, required=True,
                    help="Label for this checkpoint: OBS | OBS_MSE | RAFT")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import os
import numpy as np
import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import Isaaclab_RANSv2.tasks  # noqa: F401

# Failure modes: (DEG_prob, DEAD_prob, STK_prob)
MODES = {
    "DEG":  (1.0, 0.0, 0.0),
    "DEAD": (0.0, 1.0, 0.0),
    "STK":  (0.0, 0.0, 1.0),
}
KS = [1, 2, 3, 4]
AGG_SUBSAMPLE = 4   # record every Nth step for aggregate arrays


def _force_reset_all(env) -> None:
    base = env.unwrapped
    base.episode_length_buf = torch.full_like(base.episode_length_buf, base.max_episode_length - 1)


def _warmup(env, policy, device):
    """Trigger mass reset and return first obs of fresh episodes."""
    obs = env.get_observations()
    policy.reset()
    with torch.inference_mode():
        obs, _, _, _ = env.step(policy(obs))
    policy.reset()
    return obs


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    ckpt = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    device = agent_cfg.device
    tmp_dir = os.path.join(os.path.dirname(ckpt), "_ft_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=tmp_dir, device=device)
    runner.load(ckpt)

    policy = runner.get_inference_policy(device=device)
    actor  = runner.alg.actor

    is_observer = hasattr(actor, "get_observer_output")
    is_rnn      = hasattr(actor, "get_hidden_state") and not is_observer

    print(f"\n[FT] Method:     {args_cli.method}")
    print(f"[FT] Checkpoint: {ckpt}")
    print(f"[FT] Actor type: {type(actor).__name__}")
    print(f"[FT] is_observer={is_observer}  is_rnn={is_rnn}\n")

    task = env.unwrapped.task_api
    max_ep_len = env.unwrapped.max_episode_length

    out_dir = args_cli.output_dir or os.path.dirname(ckpt)
    os.makedirs(out_dir, exist_ok=True)

    for mode_name, probs in MODES.items():
        task._task_cfg.failure_mode_probs = probs

        for k in KS:
            print(f"  [{mode_name}] k={k} ...", end=" ", flush=True)
            task.force_failure_count(k)
            _force_reset_all(env)
            obs = _warmup(env, policy, device)

            traj_dgt   = []   # per-step, env[0] only
            traj_pred  = []   # per-step, env[0] only
            agg_dgt    = []   # subsampled, all envs
            agg_pred   = []   # subsampled, all envs
            reset_mask = []   # bool: did env[0] reset at this step?

            with torch.inference_mode():
                for step in range(max_ep_len):
                    # ── D_gt from current obs ─────────────────────────────
                    d_gt_all = obs["privileged"][:, :16].cpu().float()   # (N, 16)

                    # ── Prediction from current obs ───────────────────────
                    if is_observer:
                        pred_all = actor.get_observer_output(obs).cpu().float()  # (N, 16)
                    elif is_rnn:
                        if actor.rnn.hidden_state is not None:
                            pred_all = actor.rnn.hidden_state.squeeze(0).cpu().float()  # (N, H)
                        else:
                            H = actor.rnn.rnn.hidden_size
                            pred_all = torch.zeros(args_cli.num_envs, H)

                    # Step policy (updates RNN hidden state in-place for next iter)
                    actions = policy(obs)

                    # After policy step: record updated RNN hidden state (post obs_t)
                    if is_rnn and actor.rnn.hidden_state is not None:
                        pred_all = actor.rnn.hidden_state.squeeze(0).cpu().float()

                    obs, _, dones, _ = env.step(actions)
                    dones_cpu = (dones > 0).cpu()

                    if dones_cpu.any():
                        policy.reset(dones)

                    # ── Record env[0] trajectory ──────────────────────────
                    traj_dgt.append(d_gt_all[0].numpy())
                    traj_pred.append(pred_all[0].numpy())
                    reset_mask.append(bool(dones_cpu[0].item()))

                    # ── Aggregate (subsampled) ────────────────────────────
                    if step % AGG_SUBSAMPLE == 0:
                        agg_dgt.append(d_gt_all.numpy())
                        agg_pred.append(pred_all.numpy())

            # Stack and save
            np.savez_compressed(
                os.path.join(out_dir, f"{mode_name}_k{k}.npz"),
                single_dgt  = np.array(traj_dgt),    # (T, 16)
                single_pred = np.array(traj_pred),    # (T, D)
                agg_dgt     = np.concatenate(agg_dgt, axis=0),    # (M, 16)
                agg_pred    = np.concatenate(agg_pred, axis=0),   # (M, D)
                reset_steps = np.array(reset_mask),   # (T,) bool
            )
            print(f"saved ({mode_name}_k{k}.npz)")

        # Release failure count lock between modes
        task.force_failure_count(None)

    print(f"\n[FT] Done → {out_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
