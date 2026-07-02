# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env for the Observer (system-identification) architecture.

Builds on :class:`HistoryEnv` to add a separate ``"privileged"`` observation group containing
the ground-truth degradation state ``D_gt = [scales | offsets]`` (16-dim for 8 thrusters).

Observation groups produced:
  - ``obs["policy"]``    — standard task kinematics (unchanged, no privileged info mixed in).
  - ``obs["history"]``   — ring buffer of ``(body_vel, prev_action)`` tokens for the Observer.
  - ``obs["privileged"]``— D_gt provided by :meth:`DegradationStateTaskMixin.get_privileged_observations`.

The Observer MLP ingests ``obs["history"]`` and regresses toward ``obs["privileged"]``.
The Policy MLP ingests ``obs["policy"]`` concatenated with the detached Observer prediction.
The asymmetric Critic ingests ``obs["policy"]`` + ``obs["privileged"]`` directly.
"""

from __future__ import annotations

import torch

from .history_env import HistoryEnv
from .observer_env_cfg import ObserverEnvCfg


class ObserverEnv(HistoryEnv):
    cfg: ObserverEnvCfg

    def _get_observations(self) -> dict:
        observations = super()._get_observations()

        if hasattr(self.task_api, "get_privileged_observations"):
            observations["privileged"] = self.task_api.get_privileged_observations()

        if hasattr(self.task_api, "update_global_step"):
            self.task_api.update_global_step(self.num_envs)

        return observations
