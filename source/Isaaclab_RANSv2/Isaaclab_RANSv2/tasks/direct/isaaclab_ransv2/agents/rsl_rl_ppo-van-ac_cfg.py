# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO config for VAN-MLP with asymmetric critic (VAN-AC) — E11 ablation.

Isolates the contribution of the asymmetric critic independently of the
Observer head and MSE supervision:

- **Actor**: plain MLP, sees only obs["policy"] (same as VAN-MLP in E1).
  No Observer head, no history buffer, no D_hat augmentation.
- **Critic**: MLP, sees obs["policy"] + obs["privileged"] (16-dim D_gt).
  Identical asymmetric critic as in OBS.

Trained on Observer-Position-v0 to match OBS curriculum (CuboThrusterFailureTraining,
3-mode failure scheduler). At deployment the critic is discarded.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfgVANAC(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    max_iterations: int = 5000
    save_interval: int = 500
    experiment_name: str = "VAN_AC_GoToPosition"
    logger: str = "tensorboard"

    obs_groups: dict = {
        "actor":  ["policy"],
        "critic": ["policy", "privileged"],
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
