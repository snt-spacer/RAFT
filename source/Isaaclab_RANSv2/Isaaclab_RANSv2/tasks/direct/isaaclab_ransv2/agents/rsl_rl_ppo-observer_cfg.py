# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO + Observer (system-identification) runner config.

Architecture
------------
- **Observer MLP** reads ``obs["history"]`` (flattened ring buffer) and predicts the
  16-dim degradation state ``D_hat = [scales | offsets]`` via supervised MSE against
  the ground-truth ``obs["privileged"]``.
- **Policy MLP** reads ``obs["policy"]`` concatenated with ``D_hat.detach()`` and
  outputs actions via PPO.
- **Asymmetric Critic** reads ``obs["policy"] + obs["privileged"]`` directly (privileged
  info available at training time, discarded at deployment).

Combined loss::

    L = L_ppo + observer_loss_coef * L_mse

Key hyperparameters to sweep:
    - ``actor.observer_hidden_dims``  — Observer MLP capacity
    - ``actor.hidden_dims``           — Policy MLP capacity
    - ``algorithm.observer_loss_coef``— Weight of Observer MSE loss
    - ``num_steps_per_env``           — Rollout length (affects history buffer freshness)
    - env ``history_len``             — Context window for the Observer (set in ObserverEnvCfg)
"""

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlObserverActorModelCfg(RslRlMLPModelCfg):
    """Runner-side config for :class:`rsl_rl.models.ObserverActorModel`."""

    class_name: str = "ObserverActorModel"

    observer_hidden_dims: list[int] = MISSING
    """Hidden dimensions of the Observer MLP (history → D_hat)."""

    history_obs_group: str = "history"
    """Key in the obs TensorDict holding the history buffer."""

    degradation_dim: int = 16
    """Output dimension of the Observer (``2 * num_thrusters``).
    Must match ``GoToPositionObserverCfg.privileged_observation_space``."""


@configclass
class RslRlObserverPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm config with the Observer auxiliary loss coefficient."""

    class_name: str = "ObserverPPO"

    observer_loss_coef: float = 1.0
    """Weight λ for the Observer MSE loss: ``L = L_ppo + λ * L_mse``.
    Set to 0.0 to disable Observer training."""


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    max_iterations: int = 5000
    save_interval: int = 500
    experiment_name: str = "Observer_GoToPosition"
    logger: str = "wandb"
    wandb_project: str = "Observer_GoToPosition"
    wandb_kwargs: dict = {
        "project": "Observer_GoToPosition",
        "entity": "spacer-rl",
        "group": "observer-sysid",
    }

    # obs["history"]   → ObserverActorModel (read directly, not via obs_groups)
    # obs["policy"]    → Policy MLP base obs (via obs_groups["actor"])
    # obs["privileged"]→ Asymmetric critic + MSE target (via obs_groups["critic"])
    obs_groups: dict = {
        "actor":  ["policy"],
        "critic": ["policy", "privileged"],
    }

    actor: RslRlObserverActorModelCfg = RslRlObserverActorModelCfg(
        hidden_dims=[256, 128, 64],
        observer_hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        history_obs_group="history",
        degradation_dim=16,
    )

    critic: RslRlMLPModelCfg = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
    )

    algorithm: RslRlObserverPpoAlgorithmCfg = RslRlObserverPpoAlgorithmCfg(
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
        observer_loss_coef=1.0,
    )
