# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cubo robot configuration for active-failure training experiments (E1–E6).

Sets ``failure_curriculum_max_failures=4`` (all 4 failing-thruster slots enabled)
with a 50 M-step warmup and a 200 M-step ramp — matching the training protocol in
``docs/experiments.md``.

Use ``robot_name="CuboThrusterFailureTraining"`` in the env cfg, or pass
``env.robot_name=CuboThrusterFailureTraining`` as a Hydra override at launch time.

``VAN`` (trained without failures) should keep the default
``robot_name="CuboThrusterFailure"`` whose ``failure_curriculum_max_failures=0``.
"""

from isaaclab.utils import configclass

from .cubo_thruster_failure_cfg import CuboThrusterFailureRobotCfg


@configclass
class CuboThrusterFailureTrainingRobotCfg(CuboThrusterFailureRobotCfg):
    robot_name: str = "CuboThrusterFailureTraining"

    failure_curriculum_max_failures: int = 4
    failure_curriculum_warmup_steps: int = 50_000_000
    failure_curriculum_ramp_steps: int = 200_000_000
