# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cubo robot configuration for the RMA thruster-failure experiment.

Adds curriculum knobs that control how many thrusters can fail simultaneously and at what
training-step the curriculum reaches its maximum. Failure injection is per-episode (the mask is
sampled at reset and held constant for the rest of the episode).
"""

from isaaclab.utils import configclass

from .cubo_cfg import CuboRobotCfg


@configclass
class CuboThrusterFailureRobotCfg(CuboRobotCfg):
    """Cubo configuration with thruster-failure injection knobs."""

    robot_name: str = "CuboThrusterFailure"

    # --- Curriculum ---
    failure_curriculum_max_failures: int = 0
    """Final cap on the number of thrusters that may fail simultaneously."""

    failure_curriculum_warmup_steps: int = 0
    """Global step count below which no thrusters fail (curriculum stays at 0)."""

    failure_curriculum_ramp_steps: int = 50_000_000
    """Global step count over which the failure cap ramps from 0 to ``failure_curriculum_max_failures``."""

    failure_sampling: str = "uniform_capped"
    """How to sample the per-episode failure count.

    - ``"uniform_capped"``: sample ``k ~ Uniform{0, 1, ..., k_curr}``. Keeps the easy regime alive.
    - ``"fixed_cap"``: always set ``k = k_curr`` once the curriculum has ramped past warmup.
    """
