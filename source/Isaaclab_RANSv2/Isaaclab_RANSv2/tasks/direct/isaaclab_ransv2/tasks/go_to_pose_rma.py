# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GoToPose variant for the RMA thruster-failure experiment.

The RMA-specific behaviour (per-episode failure-mask sampling, curriculum schedule, eval-mode
override, privileged-observation getter, and mask-stat logging) lives in
:class:`ThrusterFailureTaskMixin`. This file is just the trivial composition with :class:`GoToPoseTask`.
"""

from __future__ import annotations

from .go_to_pose import GoToPoseTask
from ._thruster_failure_task_mixin import ThrusterFailureTaskMixin


class GoToPoseRMATask(ThrusterFailureTaskMixin, GoToPoseTask):
    """GoToPose with per-episode thruster failures + privileged mask exposure."""
    pass
