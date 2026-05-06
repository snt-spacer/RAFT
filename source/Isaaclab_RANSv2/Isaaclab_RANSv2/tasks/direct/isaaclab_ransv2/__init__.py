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

