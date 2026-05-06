# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase-2 training driver for RMA.

Loads a Phase-1 RSL-RL checkpoint (an :class:`rsl_rl.models.RMAModel` actor with its trained
encoder ``mu``), freezes it, and trains an :class:`rsl_rl.modules.RMAAdaptationModule` to
regress the privileged latent ``z = mu(e)`` from a recent history of ``(policy_obs, action)``
pairs.

Typical usage:

    ./isaaclab.sh -p scripts/rsl_rl/train_rma_phase2.py \
        --task Isaaclab-RANSv2-RMA-v0 \
        --num_envs 4096 \
        --phase1_run <run_dir_under_logs/rsl_rl/AutoEnvGen_PPO_RMA> \
        --phase1_checkpoint model_*.pt \
        --num_iterations 2000 \
        --history_length 50 \
        --backbone conv

The trained adaptation module is saved next to the Phase-1 run (default
``logs/rsl_rl/<experiment>/<run>/phase2/``).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Phase-2 RMA training (offline regression of mu(mask) from history).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task (e.g. Isaaclab-RANSv2-RMA-v0).")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Phase-1 agent config entry point (must use class_name=RMAModel for the actor).",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")

# -- Phase-1 checkpoint ---------------------------------------------------- #
parser.add_argument(
    "--phase1_run",
    type=str,
    default=None,
    help="Phase-1 run directory under logs/rsl_rl/<experiment>/. If omitted, takes the latest.",
)
parser.add_argument(
    "--phase1_checkpoint",
    type=str,
    default=None,
    help="Phase-1 checkpoint filename (e.g. model_5000.pt). Takes the latest if omitted.",
)

# -- Adaptation module architecture --------------------------------------- #
parser.add_argument("--history_length", type=int, default=50, help="Length of the (obs, action) history window.")
parser.add_argument(
    "--backbone", type=str, default="conv", choices=["conv", "gru"], help="Adaptation module backbone."
)
parser.add_argument(
    "--gru_hidden_dim", type=int, default=128, help="GRU hidden dim (only used when backbone=gru)."
)

# -- Training schedule ---------------------------------------------------- #
parser.add_argument("--num_iterations", type=int, default=2000, help="Phase-2 outer iterations.")
parser.add_argument("--steps_per_iter", type=int, default=8, help="Env steps collected per iteration.")
parser.add_argument(
    "--train_steps_per_iter", type=int, default=8, help="Adaptation-module SGD steps per iteration."
)
parser.add_argument("--batch_size", type=int, default=256, help="SGD minibatch size.")
parser.add_argument("--replay_capacity", type=int, default=200_000, help="Replay buffer capacity (samples).")
parser.add_argument("--learning_rate", type=float, default=5.0e-4, help="Adam learning rate.")
parser.add_argument(
    "--warmup_envsteps",
    type=int,
    default=0,
    help="Env steps to collect before any gradient updates (lets the replay fill up).",
)

# -- Saving --------------------------------------------------------------- #
parser.add_argument(
    "--save_dir",
    type=str,
    default=None,
    help="Override save directory. Defaults to <phase1_run_dir>/phase2/.",
)
parser.add_argument("--save_interval", type=int, default=200, help="Iterations between checkpoint dumps.")
parser.add_argument("--log_interval", type=int, default=10, help="Iterations between log lines.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch
import gymnasium as gym

from rsl_rl.algorithms import RMAPhase2Trainer
from rsl_rl.modules import RMAAdaptationModule
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
    """Locate the Phase-1 checkpoint under ``logs/rsl_rl/<experiment_name>/...``.

    If ``--phase1_checkpoint`` is already a full (absolute or relative) path to a ``.pt`` file,
    return it directly without invoking ``get_checkpoint_path``.
    """
    checkpoint = args_cli.phase1_checkpoint
    if checkpoint is not None and os.path.isfile(checkpoint):
        return os.path.abspath(checkpoint)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))
    run_dir = args_cli.phase1_run if args_cli.phase1_run is not None else ".*"
    checkpoint = checkpoint if checkpoint is not None else "model_.*.pt"
    return get_checkpoint_path(log_root_path, run_dir, checkpoint)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    # apply CLI overrides shared with Phase-1 train.py
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    import importlib.metadata as metadata

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    # locate Phase-1 checkpoint
    resume_path = _resolve_phase1_checkpoint(agent_cfg.experiment_name)
    print(f"[Phase2] Phase-1 checkpoint: {resume_path}")

    # build env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # build a runner with the same architecture as Phase-1, then load weights.
    # Use a temp log dir; we won't call runner.learn() so logging artifacts won't pile up.
    tmp_log_dir = os.path.join(os.path.dirname(resume_path), "phase2_tmp")
    os.makedirs(tmp_log_dir, exist_ok=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=tmp_log_dir, device=agent_cfg.device)
    runner.load(resume_path)
    actor = runner.alg.actor.to(agent_cfg.device)

    # sanity: actor must expose encode_privileged (i.e. be an RMAModel)
    if not hasattr(actor, "encode_privileged"):
        raise RuntimeError(
            "The loaded actor does not expose 'encode_privileged'. The Phase-1 checkpoint must "
            "have been trained with class_name=RMAModel (see rsl_rl_ppo-rma_cfg.py)."
        )

    # build the adaptation module sized to the actor's streams
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs_dim = obs["policy"].shape[-1]
    action_dim = env.num_actions
    latent_dim = actor.latent_dim
    print(
        f"[Phase2] obs_dim={obs_dim} action_dim={action_dim} latent_dim={latent_dim} "
        f"history_length={args_cli.history_length} backbone={args_cli.backbone}"
    )

    backbone_kwargs = {}
    if args_cli.backbone == "gru":
        backbone_kwargs["hidden_dim"] = args_cli.gru_hidden_dim

    adapt = RMAAdaptationModule(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        history_length=args_cli.history_length,
        backbone=args_cli.backbone,
        backbone_kwargs=backbone_kwargs,
    ).to(agent_cfg.device)

    # save dir
    save_dir = args_cli.save_dir or os.path.join(os.path.dirname(resume_path), "phase2")
    print(f"[Phase2] Saving adaptation module to: {save_dir}")

    # train
    trainer = RMAPhase2Trainer(
        env=env,
        actor=actor,
        adaptation_module=adapt,
        policy_obs_group="policy",
        batch_size=args_cli.batch_size,
        replay_capacity=args_cli.replay_capacity,
        learning_rate=args_cli.learning_rate,
        warmup_envsteps=args_cli.warmup_envsteps,
        device=agent_cfg.device,
    )

    trainer.learn(
        num_iterations=args_cli.num_iterations,
        steps_per_iter=args_cli.steps_per_iter,
        train_steps_per_iter=args_cli.train_steps_per_iter,
        log_interval=args_cli.log_interval,
        save_path=save_dir,
        save_interval=args_cli.save_interval,
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
