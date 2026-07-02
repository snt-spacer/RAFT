# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .go_to_position_observer_cfg import GoToPositionObserverCfg


@configclass
class GoToPositionObserverDEGCfg(GoToPositionObserverCfg):
    """Observer cfg with only continuous-degradation failures (mode isolation for E2/E4/E5)."""

    failure_mode_probs: tuple = (1.0, 0.0, 0.0)
