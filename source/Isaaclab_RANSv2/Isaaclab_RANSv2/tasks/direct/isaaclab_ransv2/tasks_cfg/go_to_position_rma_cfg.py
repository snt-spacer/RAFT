# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task config for the RMA GoToPosition experiment.

Adds the privileged observation slot used by the Phase-1 encoder (size = number of thrusters).
"""

from isaaclab.utils import configclass

from .go_to_position_cfg import GoToPositionCfg


@configclass
class GoToPositionRMACfg(GoToPositionCfg):
    """GoToPosition configuration with a privileged thruster-health observation."""

    privileged_observation_space: int = 8
    """Dimension of the privileged stream (per-thruster health mask). Must match
    ``CuboThrusterFailureRobotCfg.num_thrusters``."""
