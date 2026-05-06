# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO + RMA Phase-1 runner config for the GoToPosition task.

Same model architecture and obs-group routing as the GoToPose RMA cfg — only the experiment
name changes so logs/wandb projects don't collide. The actor/critic auto-size to whatever obs
the env produces, so no other field needs to change between pose and position.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from .rsl_rl_ppo_rma_cfg_common import RslRlRMAModelCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 5000
    save_interval = 1000
    experiment_name = "AutoEnvGen_PPO_RMA_Position"
    logger = "wandb"
    wandb_project = "AutoEnvGen_PPO_RMA_Position"
    wandb_kwargs = {
        "project": "AutoEnvGen_PPO_RMA_Position",
        "entity": "spacer-rl",
        "group": "thruster-failure",
    }

    obs_groups = {
        "actor": ["policy"],
        "actor_privileged": ["privileged"],
        "critic": ["policy", "privileged"],
    }

    actor = RslRlRMAModelCfg(
        hidden_dims=[64, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        latent_dim=8,
        encoder_hidden_dims=[32,],
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256],
        activation="elu",
        obs_normalization=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
