# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env config for the RMA GoToPosition experiment (precursor to GoToPose RMA)."""

from isaaclab.utils import configclass

from .auto_env_gen_cfg import AutoEnvGenCfg


@configclass
class RMAPositionEnvCfg(AutoEnvGenCfg):
    robot_name = "CuboThrusterFailure"
    task_name = "GoToPositionRMA"

    privileged_observation_space: int = 8
