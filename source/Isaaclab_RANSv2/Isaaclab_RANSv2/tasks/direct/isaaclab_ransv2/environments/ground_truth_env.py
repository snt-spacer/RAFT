# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env wrapper for the RMA experiment.

Reuses :class:`AutoEnvGen` for scene/robot/task wiring and adds two responsibilities:

Concatenates the privileged thruster-health mask (if the task provides it) to the standard
``"policy"`` obs group.
"""

from __future__ import annotations

import torch

from .auto_env_gen import AutoEnvGen
from .ground_truth_env_cfg import GroundTruthEnvCfg


class GroundTruthEnv(AutoEnvGen):
    cfg: GroundTruthEnvCfg

    def _get_observations(self) -> dict:
        observations = super()._get_observations()
        if hasattr(self.task_api, "get_privileged_observations"):
            privileged_obs = self.task_api.get_privileged_observations()
            if "policy" in observations:
                observations["policy"] = torch.cat([observations["policy"], privileged_obs], dim=-1)
        # Tick the curriculum counter once per env step (counted in env-frames so the schedule
        # length is independent of num_envs — adjust ``ramp_steps`` accordingly if you prefer
        # global frames).
        if hasattr(self.task_api, "update_global_step"):
            self.task_api.update_global_step(self.num_envs)
        return observations
