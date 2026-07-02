# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GoToPosition variant for the Observer (system-identification) architecture.

Uses :class:`DegradationStateTaskMixin` for 3-mode thruster failure sampling and exposes a
16-dim ``D_gt = [scales | offsets]`` privileged observation for the Observer MLP's supervised
MSE loss.
"""

from __future__ import annotations

from .go_to_position import GoToPositionTask
from .degradation_state_task_mixin import DegradationStateTaskMixin


class GoToPositionObserverTask(DegradationStateTaskMixin, GoToPositionTask):
    """GoToPosition with multi-mode per-episode thruster degradation + privileged D_gt."""
    pass
