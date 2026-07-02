# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task config for the Observer (system-identification) GoToPosition experiment.

Extends :class:`GoToPositionRMACfg` with per-mode failure parameters read by
:class:`DegradationStateTaskMixin` and updates ``privileged_observation_space`` to 16
(scale + offset per thruster).
"""

from isaaclab.utils import configclass

from .go_to_position_rma_cfg import GoToPositionRMACfg


@configclass
class GoToPositionObserverCfg(GoToPositionRMACfg):
    """GoToPosition config for the Observer architecture with 3-mode thruster failure."""

    privileged_observation_space: int = 16
    """Dimension of D_gt: ``[s_0,…,s_7, o_0,…,o_7]`` for 8 thrusters."""

    failure_mode_probs: tuple = (0.33, 0.33, 0.34)
    """Probability of each failure mode: (continuous_degradation, binary_dead, stuck_on).
    Values must sum to 1."""

    degradation_scale_range: tuple = (0.0, 1.0)
    """Uniform sampling range ``[min, max]`` for the scale multiplier in continuous-degradation
    mode. A value of 1.0 means no degradation; 0.0 is a complete failure."""

    stuck_offset_range: tuple = (0.1, 1.0)
    """Uniform sampling range ``[min, max]`` for the stuck-on offset in stuck-on mode.
    Represents the constant thrust fraction a stuck thruster outputs."""
