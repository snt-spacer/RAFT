# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO config for the Transformer-history encoder experiment.

Both actor and critic use :class:`rsl_rl.models.TransformerEncoderModel`:
  - ``obs["policy"]`` (15 dims) → obs normalizer
  - ``obs["history"]`` (64 × 12) → Transformer encoder → 64-dim latent ``z``
  - ``concat(norm_obs, z)`` (79 dims) → MLP body

The model is non-recurrent from PPO's perspective; the history buffer is
maintained by the env and stored verbatim in the rollout storage.
"""

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlTransformerEncoderModelCfg(RslRlMLPModelCfg):
    """Runner-side config for :class:`rsl_rl.models.TransformerEncoderModel`."""

    class_name: str = "TransformerEncoderModel"

    history_obs_group: str = "history"
    """Name of the obs-TensorDict key holding the history buffer."""

    history_len: int = 32
    """Number of history tokens (must match the env's ``history_len``)."""

    history_feat_dim: int = 12
    """Feature dim per token (must match the env's ``history_feat_dim``)."""

    latent_dim: int = 64
    """Dimension of the Transformer output latent ``z``."""

    d_model: int = 128
    """Transformer hidden dimension."""

    nhead: int = 4
    """Number of self-attention heads."""

    num_layers: int = 2
    """Number of Transformer encoder layers."""

    dim_feedforward: int = 256
    """FFN hidden dimension inside each Transformer layer."""


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 500
    experiment_name = "TransformerHistory_GoToPosition"
    logger = "wandb"
    wandb_project = "TransformerHistory_GoToPosition"
    wandb_kwargs = {
        "project": "TransformerHistory_GoToPosition",
        "entity": "spacer-rl",
        "group": "transformer-history",
    }

    # Only "policy" goes into obs_groups; "history" is read directly by the model
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }

    actor = RslRlTransformerEncoderModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        history_obs_group="history",
        history_len=64,
        history_feat_dim=12,
        latent_dim=64,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
    )

    critic = RslRlTransformerEncoderModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=False,
        history_obs_group="history",
        history_len=64,
        history_feat_dim=12,
        latent_dim=64,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
    )

    algorithm = RslRlPpoAlgorithmCfg(
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
