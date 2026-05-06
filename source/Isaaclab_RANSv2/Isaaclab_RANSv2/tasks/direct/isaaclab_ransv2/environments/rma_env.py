# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env wrapper for the RMA experiment.

Reuses :class:`AutoEnvGen` for scene/robot/task wiring and adds two responsibilities:

1. Surface the privileged thruster-health mask under the ``"privileged"`` obs group, in addition
   to the standard ``"policy"`` group. The Phase-1 RMA encoder reads the privileged group; the
   actor body reads only ``"policy"``.
2. Advance the task's curriculum step counter once per env step so ``GoToPoseRMATask`` can ramp
   the failure cap from 0 to ``max_failures`` over training.
"""

from __future__ import annotations

import torch

from .auto_env_gen import AutoEnvGen
from .rma_env_cfg import RMAEnvCfg


class RMAEnv(AutoEnvGen):
    cfg: RMAEnvCfg

    def _get_observations(self) -> dict:
        observations = super()._get_observations()
        if hasattr(self.task_api, "get_privileged_observations"):
            observations["privileged"] = self.task_api.get_privileged_observations()
        # Tick the curriculum counter once per env step (counted in env-frames so the schedule
        # length is independent of num_envs — adjust ``ramp_steps`` accordingly if you prefer
        # global frames).
        if hasattr(self.task_api, "update_global_step"):
            self.task_api.update_global_step(self.num_envs)
        return observations
