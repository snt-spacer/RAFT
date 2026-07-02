# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO config for the Vanilla MLP baseline (VAN).

Plain two-layer MLP actor + MLP critic, no history encoding, no Transformer.
Trained without failures (default CuboThrusterFailure has max_failures=0).
Uses only obs["policy"]; the history buffer in the env is ignored.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    max_iterations: int = 3000
    save_interval: int = 500
    experiment_name: str = "Vanilla_GoToPosition"
    logger: str = "wandb"
    wandb_project: str = "Vanilla_GoToPosition"
    wandb_kwargs: dict = {
        "project": "Observer_GoToPosition",
        "entity": "spacer-rl",
        "group": "observer-sysid",
    }

    obs_groups: dict = {
        "actor": ["policy"],
        "critic": ["policy"],
    }

    actor: RslRlMLPModelCfg = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )

    critic: RslRlMLPModelCfg = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
