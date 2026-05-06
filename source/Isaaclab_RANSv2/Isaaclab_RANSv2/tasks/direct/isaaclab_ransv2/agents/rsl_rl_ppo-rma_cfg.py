# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO + RMA Phase-1 runner config.

Trains the policy and the privileged-info encoder ``mu`` jointly with PPO. The actor reads
``"policy"`` for the body and ``"privileged"`` for the encoder; the critic gets both groups
concatenated (asymmetric critic — privileged at training time only).

Phase 2 (offline supervised regression of the privileged latent from history) is run as a
separate step using :class:`rsl_rl.modules.RMAAdaptationModule` after Phase-1 has converged.
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
    save_interval = 500
    experiment_name = "AutoEnvGen_PPO_RMA"
    logger = "wandb"
    wandb_project = "AutoEnvGen_PPO_RMA"
    wandb_kwargs = {
        "project": "AutoEnvGen_PPO_RMA",
        "entity": "spacer-rl",
        "group": "thruster-failure",
    }

    # The RMA actor reads the regular policy obs and a separate privileged group with the
    # thruster-health mask. The critic gets both (asymmetric: privileged is allowed at training).
    obs_groups = {
        "actor": ["policy"],
        "actor_privileged": ["privileged"],
        "critic": ["policy", "privileged"],
    }

    actor = RslRlRMAModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        latent_dim=8,
        encoder_hidden_dims=[32, 32],
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
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
