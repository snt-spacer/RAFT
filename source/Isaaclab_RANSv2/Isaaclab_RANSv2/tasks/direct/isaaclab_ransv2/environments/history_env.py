# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env that exposes a proprioceptive history buffer for the Transformer encoder.

At every step the environment appends a token ``[lin_vel_x, lin_vel_y, ang_vel,
prev_action_0..8]`` (12 dims) to a ring buffer of length ``history_len`` (default
64) and surfaces the entire buffer as ``obs["history"]`` with shape
``(num_envs, history_len, 12)``.

The standard task observations remain in ``obs["policy"]`` (15 dims for
GoToPosition). The failure mask is *not* exposed — it is the Transformer's job
to infer which thrusters failed from the history of thrust commands vs. motion.

On episode reset the history buffer for the done envs is zeroed so the
Transformer starts with a blank slate.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .auto_env_gen import AutoEnvGen
from .history_env_cfg import HistoryEnvCfg


class HistoryEnv(AutoEnvGen):
    cfg: HistoryEnvCfg

    def __init__(self, cfg: HistoryEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._history_buf = torch.zeros(
            (self.num_envs, self.cfg.history_len, self.cfg.history_feat_dim),
            device=self.device,
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_history_token(self) -> torch.Tensor:
        """Build the (num_envs, 12) token for the *current* timestep.

        Token layout: [lin_vel_x (body), lin_vel_y (body), ang_vel_z (world),
                       prev_action_0..8]
        """
        lin_vel_xy = self.robot_api.root_com_lin_vel_b[:, :2]          # (N, 2)
        ang_vel_z = self.robot_api.root_com_ang_vel_w[:, -1].unsqueeze(-1)  # (N, 1)
        prev_actions = self.robot_api._previous_actions                  # (N, 9)
        return torch.cat([lin_vel_xy, ang_vel_z, prev_actions], dim=-1)  # (N, 12)

    # ------------------------------------------------------------------
    # DirectRLEnv overrides
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        observations = super()._get_observations()

        token = self._build_history_token()  # (N, 12)

        # Shift history left by one step (oldest falls off, newest appended)
        # Clone the source slice to avoid in-place overlap corruption.
        self._history_buf[:, :-1] = self._history_buf[:, 1:].clone()
        self._history_buf[:, -1] = token

        observations["history"] = self._history_buf
        return observations

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        super()._reset_idx(env_ids)
        if env_ids is None:
            self._history_buf.zero_()
        elif len(env_ids) > 0:
            self._history_buf[env_ids] = 0.0
