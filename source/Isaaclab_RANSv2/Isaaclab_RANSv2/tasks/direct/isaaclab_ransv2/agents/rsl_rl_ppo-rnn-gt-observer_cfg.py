# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GT-Observer: mixed-mode fault oracle that exposes D_gt to the actor.

Same architecture and training budget as RAFT (GRU-64-AC), but the actor receives
the full 16-dim ground-truth degradation vector D_gt = [s_0..s_7, delta_0..delta_7]
as part of its observation. Trained on Observer-Position-v0 (3-mode mixed failure
sampling: DEG / DEAD / STK) for an apples-to-apples per-mode upper bound.

- Actor : GRU-RNN, sees obs["policy"] + obs["privileged"]  (task obs + D_gt)
- Critic: MLP,    sees obs["policy"] + obs["privileged"]
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlRNNModelCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    max_iterations: int = 5000
    save_interval: int = 500
    experiment_name: str = "GT_OBS_LongTrain"
    logger: str = "tensorboard"

    obs_groups: dict = {
        "actor":  ["policy", "privileged"],
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
