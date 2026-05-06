# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task config for the RMA GoToPose experiment.

Adds the privileged observation slot used by the Phase-1 encoder. The privileged size equals the
number of thrusters whose health is exposed (binary mask). Curriculum knobs live on the robot
config; this file only declares the obs space contract.
"""

from isaaclab.utils import configclass

from .go_to_pose_cfg import GoToPoseCfg


@configclass
class GoToPoseRMACfg(GoToPoseCfg):
    """GoToPose configuration with a privileged thruster-health observation."""

    privileged_observation_space: int = 8
    """Dimension of the privileged stream (per-thruster health mask). Must match
    ``CuboThrusterFailureRobotCfg.num_thrusters``."""
