# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env config for the RMA experiment.

Defaults to ``CuboThrusterFailure`` + ``GoToPoseRMA``. Adds a top-level slot for the privileged
observation space size so the env can advertise it on ``observation_space`` (which Isaac Lab uses
to build the gym observation dict).
"""

from isaaclab.utils import configclass

from .auto_env_gen_cfg import AutoEnvGenCfg


@configclass
class RMAEnvCfg(AutoEnvGenCfg):
    robot_name = "CuboThrusterFailure"
    task_name = "GoToPoseRMA"

    privileged_observation_space: int = 8
    """Dimension of the ``"privileged"`` observation group surfaced for the RMA encoder."""
