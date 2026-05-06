# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env config for the Transformer-history adaptation experiment.

Uses ``CuboThrusterFailure`` (so physical thruster failures are sampled per
episode) with ``GoToPositionRMA`` (which draws the failure mask via the
mixin).  Unlike ``GroundTruthEnv``, the mask is *not* exposed to the policy;
instead the policy receives a 64-step proprioceptive history buffer that the
Transformer encoder uses to infer the robot dynamics online.
"""

from isaaclab.utils import configclass

from .auto_env_gen_cfg import AutoEnvGenCfg


@configclass
class HistoryEnvCfg(AutoEnvGenCfg):
    robot_name = "CuboThrusterFailure"
    task_name = "GoToPositionRMA"

    history_len: int = 64
    """Number of (body_vel, action) tokens kept in the history buffer."""

    history_feat_dim: int = 12
    """Feature dimension per token: lin_vel_x + lin_vel_y + ang_vel + 9 actions."""
