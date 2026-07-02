"""GRU recurrent policy for VAN ablation (E10).

Reads only instantaneous obs["policy"] — same input as VAN-MLP —
but uses a GRU hidden state instead of an MLP to implicitly accumulate
temporal context.  Trained with the same failure curriculum as OBS.

Two variants registered:
  VAN-GRU-64  : rnn_hidden_dim=64   (light)
  VAN-GRU-256 : rnn_hidden_dim=256  (matched to OBS capacity)
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlRNNModelCfg


@configclass
class PPORunnerCfgGRU64(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    max_iterations: int = 5000
    save_interval: int = 500
    experiment_name: str = "VAN_GRU64_GoToPosition"
    logger: str = "tensorboard"

    obs_groups: dict = {
        "actor": ["policy"],
        "critic": ["policy"],
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

    critic: RslRlRNNModelCfg = RslRlRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        rnn_type="gru",
        rnn_hidden_dim=64,
        rnn_num_layers=1,
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
class PPORunnerCfgGRU256(PPORunnerCfgGRU64):
    experiment_name: str = "VAN_GRU256_GoToPosition"
    wandb_project: str = "VAN_GRU256_GoToPosition"

    actor: RslRlRNNModelCfg = RslRlRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )

    critic: RslRlRNNModelCfg = RslRlRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
