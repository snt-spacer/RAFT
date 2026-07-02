# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env config for the Observer system-identification experiment.

Inherits ``history_len`` and ``history_feat_dim`` from :class:`HistoryEnvCfg` so both are
exposed as first-class hyperparameters for sweeping.  The task is ``"GoToPositionObserver"``
which uses :class:`DegradationStateTaskMixin` for 3-mode failure sampling.
"""

from isaaclab.utils import configclass

from .history_env_cfg import HistoryEnvCfg


@configclass
class ObserverEnvCfg(HistoryEnvCfg):
    robot_name: str = "CuboThrusterFailureTraining"
    task_name: str = "GoToPositionObserver"

    history_len: int = 32
    """Number of (body_vel, action) tokens in the Observer's input buffer. Tune as needed."""

    history_feat_dim: int = 12
    """Feature dimension per token: lin_vel_x + lin_vel_y + ang_vel + 9 actions."""
