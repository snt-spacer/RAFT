# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GoToPosition variant for the RMA thruster-failure experiment.

A simpler precursor to :class:`GoToPoseRMATask` — no target heading, only a 2-D goal position.
Uses the same RMA mixin so the two tasks share their failure-mask machinery.
"""

from __future__ import annotations

from .go_to_position import GoToPositionTask
from ._thruster_failure_task_mixin import ThrusterFailureTaskMixin


class GoToPositionRMATask(ThrusterFailureTaskMixin, GoToPositionTask):
    """GoToPosition with per-episode thruster failures + privileged mask exposure."""
    pass
