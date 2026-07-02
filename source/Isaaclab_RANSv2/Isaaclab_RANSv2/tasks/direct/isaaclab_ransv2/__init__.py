# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from . import environments

##
# Register Gym environments.
##

gym.register(
    id="Isaaclab-RANSv2-AutoEnvGen-v0",
    entry_point=f"{environments.__name__}.auto_env_gen:AutoEnvGen",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{environments.__name__}.auto_env_gen_cfg:AutoEnvGenCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_ppo-discrete_cfg_entry_point": f"{agents.__name__}:rl_games_ppo-discrete_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "rsl_rl_rnn_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-rnn_cfg:PPORunnerCfg",
        "rsl_rl_beta_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-beta_cfg:PPORunnerCfg",
        "rsl_rl_hypernet_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-hypernet_cfg:PPORunnerCfg",
        "rsl_rl_hypernet_beta_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-hypernet-beta_cfg:PPORunnerCfg",
        "rsl_rl_hypernet_rnn_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-hypernet-rnn_cfg:PPORunnerCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ppo-discrete_cfg_entry_point": f"{agents.__name__}:skrl_ppo-discrete_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaaclab-RANSv2-RMA-v0",
    entry_point=f"{environments.__name__}.rma_env:RMAEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{environments.__name__}.rma_env_cfg:RMAEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-rma_cfg:PPORunnerCfg",
        "rsl_rl_rma_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-rma_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Isaaclab-RANSv2-RMA-Position-v0",
    entry_point=f"{environments.__name__}.rma_env:RMAEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{environments.__name__}.rma_position_env_cfg:RMAPositionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-rma-position_cfg:PPORunnerCfg",
        "rsl_rl_rma_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-rma-position_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Isaaclab-RANSv2-GroundTruth-Position-v0",
    entry_point=f"{environments.__name__}.ground_truth_env:GroundTruthEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{environments.__name__}.ground_truth_env_cfg:GroundTruthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-rnn-ground-truth-position_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Isaaclab-RANSv2-History-Position-v0",
    entry_point=f"{environments.__name__}.history_env:HistoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{environments.__name__}.history_env_cfg:HistoryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-transformer_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Isaaclab-RANSv2-Vanilla-Position-v0",
    entry_point=f"{environments.__name__}.history_env:HistoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{environments.__name__}.history_env_cfg:HistoryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-vanilla-position_cfg:PPORunnerCfg",
        "rsl_rl_rnn_gru64_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-rnn-van-gru_cfg:PPORunnerCfgGRU64",
        "rsl_rl_rnn_gru256_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-rnn-van-gru_cfg:PPORunnerCfgGRU256",
        "rsl_rl_rnn_lstm64_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-rnn-van-lstm_cfg:PPORunnerCfgLSTM64",
        "rsl_rl_rnn_lstm256_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-rnn-van-lstm_cfg:PPORunnerCfgLSTM256",
    },
)

gym.register(
    id="Isaaclab-RANSv2-Observer-Position-v0",
    entry_point=f"{environments.__name__}.observer_env:ObserverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{environments.__name__}.observer_env_cfg:ObserverEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-observer_cfg:PPORunnerCfg",
        # E11 AC ablation — baselines with asymmetric critic (critic sees policy + privileged)
        "rsl_rl_van_ac_cfg_entry_point":       f"{agents.__name__}.rsl_rl_ppo-van-ac_cfg:PPORunnerCfgVANAC",
        "rsl_rl_rnn_gru64_ac_cfg_entry_point":  f"{agents.__name__}.rsl_rl_ppo-rnn-van-gru-ac_cfg:PPORunnerCfgGRU64AC",
        "rsl_rl_rnn_gru256_ac_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-rnn-van-gru-ac_cfg:PPORunnerCfgGRU256AC",
        "rsl_rl_rnn_lstm64_ac_cfg_entry_point":  f"{agents.__name__}.rsl_rl_ppo-rnn-van-lstm-ac_cfg:PPORunnerCfgLSTM64AC",
        "rsl_rl_rnn_lstm256_ac_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo-rnn-van-lstm-ac_cfg:PPORunnerCfgLSTM256AC",
        # Mixed-mode GT oracle: actor sees D_gt directly (same env, 3-mode training).
        "rsl_rl_gt_observer_cfg_entry_point":   f"{agents.__name__}.rsl_rl_ppo-rnn-gt-observer_cfg:PPORunnerCfg",
    },
)