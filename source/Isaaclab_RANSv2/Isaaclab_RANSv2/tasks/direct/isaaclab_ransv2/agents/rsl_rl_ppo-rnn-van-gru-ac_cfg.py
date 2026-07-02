# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GRU recurrent policy with asymmetric critic (GRU-AC) — E11 ablation.

Tests whether adding a privileged asymmetric critic (same as OBS) on top of
a recurrent actor closes the gap to OBS, isolating recurrence + better value
estimation from the full OBS design.

- **Actor**: GRU-RNN, sees only obs["policy"] (same as E10 GRU variants).
- **Critic**: MLP, sees obs["policy"] + obs["privileged"] (16-dim D_gt).

Trained on Observer-Position-v0 (same env and failure curriculum as OBS).

Two variants:
  GRU-64-AC  : rnn_hidden_dim=64
  GRU-256-AC : rnn_hidden_dim=256
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlRNNModelCfg


@configclass
class PPORunnerCfgGRU64AC(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    max_iterations: int = 5000
    save_interval: int = 500
    experiment_name: str = "VAN_GRU64_AC_GoToPosition"
    logger: str = "tensorboard"

    obs_groups: dict = {
        "actor":  ["policy"],
        "critic": ["policy", "privileged"],
    }

    actor: RslRlRNNModelCfg = RslRlRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        rnn_type="gru",
        rnn_hidden_dim=64,
        rnn_num_layers=1,
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


@configclass
class PPORunnerCfgGRU256AC(PPORunnerCfgGRU64AC):
    experiment_name: str = "VAN_GRU256_AC_GoToPosition"

    actor: RslRlRNNModelCfg = RslRlRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
