# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cubo thruster-failure config with NO curriculum (E7 GT-collapse investigation).

Sets failure_curriculum_max_failures=4 with warmup=0 and ramp=1, so the agent
sees up to k=4 failures from the very first training step.

Use with env.robot_name=CuboThrusterFailureNoCurr.
"""

from isaaclab.utils import configclass

from .cubo_thruster_failure_cfg import CuboThrusterFailureRobotCfg


@configclass
class CuboThrusterFailureNoCurrRobotCfg(CuboThrusterFailureRobotCfg):
    robot_name: str = "CuboThrusterFailureNoCurr"

    failure_curriculum_max_failures: int = 4
    failure_curriculum_warmup_steps: int = 0
    failure_curriculum_ramp_steps: int = 1
