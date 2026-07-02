# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mixin that injects per-episode multi-mode thruster degradation.

Three failure modes per thruster (mutually exclusive, selected by configurable probabilities):

1. **Continuous degradation** — ``scale ~ Uniform[scale_min, scale_max]``, ``offset = 0``
2. **Binary dead**           — ``scale = 0``, ``offset = 0``
3. **Stuck-on**              — ``scale = 0``, ``offset ~ Uniform[offset_min, offset_max]``

Healthy thrusters always have ``scale = 1``, ``offset = 0``.

The ground-truth degradation vector ``D_gt`` is 16-dim for 8 thrusters:
``[s_0, …, s_7, o_0, …, o_7]`` — scales stacked before offsets.

Configuration fields read from ``self._task_cfg``:

- ``failure_mode_probs``       — 3-tuple of floats summing to 1 (degradation, binary_dead, stuck_on)
- ``degradation_scale_range``  — ``(min, max)`` for the continuous degradation multiplier
- ``stuck_offset_range``       — ``(min, max)`` for the stuck-on offset
- ``failure_curriculum_*``     — curriculum knobs still live on ``self._robot._robot_cfg``

Intended usage::

    class GoToPositionObserverTask(DegradationStateTaskMixin, GoToPositionTask):
        pass
"""

from __future__ import annotations

import torch


class DegradationStateTaskMixin:
    """Mixin that adds 3-mode degradation-state machinery to a :class:`TaskCore` subclass."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self._degradation_scales: torch.Tensor | None = None
        self._degradation_offsets: torch.Tensor | None = None
        self._global_step: int = 0
        self._forced_failure_count: int | None = None

    # ------------------------------------------------------------------ #
    # Curriculum (mirrors ThrusterFailureTaskMixin)                      #
    # ------------------------------------------------------------------ #

    def _current_max_failures(self) -> int:
        if self._forced_failure_count is not None:
            return self._forced_failure_count
        cfg = self._robot._robot_cfg
        warmup = getattr(cfg, "failure_curriculum_warmup_steps", 0)
        ramp = max(1, getattr(cfg, "failure_curriculum_ramp_steps", 1))
        max_k = getattr(cfg, "failure_curriculum_max_failures", 0)
        if self._global_step <= warmup:
            return 0
        progress = min(1.0, (self._global_step - warmup) / ramp)
        return int(round(progress * max_k))

    def update_global_step(self, num_envs_stepped: int) -> None:
        self._global_step += num_envs_stepped

    def force_failure_count(self, k: int | None) -> None:
        if k is not None and k < 0:
            raise ValueError(f"k must be >= 0 (got {k}).")
        self._forced_failure_count = k

    # ------------------------------------------------------------------ #
    # Degradation-state sampling                                         #
    # ------------------------------------------------------------------ #

    def _sample_degradation_state(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``(scales, offsets)`` for the given envs using the warp RNG.

        Returns:
            scales:  ``(len(env_ids), num_thrusters)`` — multipliers in ``[0, 1]``.
            offsets: ``(len(env_ids), num_thrusters)`` — stuck-on offsets in ``[0, 1]``.
        """
        cfg = self._robot._robot_cfg
        task_cfg = self._task_cfg
        num_thrusters = cfg.num_thrusters
        n = len(env_ids)
        k_max = self._current_max_failures()

        # All healthy when curriculum hasn't started yet.
        if k_max == 0:
            scales = torch.ones((n, num_thrusters), device=self._device, dtype=torch.float32)
            offsets = torch.zeros((n, num_thrusters), device=self._device, dtype=torch.float32)
            return scales, offsets

        # --- Step 1: choose how many thrusters fail per env ---
        sampling = (
            "fixed_cap"
            if self._forced_failure_count is not None
            else getattr(cfg, "failure_sampling", "uniform_capped")
        )
        if sampling == "fixed_cap":
            k_per_env = torch.full((n,), k_max, device=self._device, dtype=torch.long)
        else:
            k_per_env = (
                self._rng.sample_integer_torch(0, k_max + 1, 1, ids=env_ids).to(torch.long)
            )

        # --- Step 2: rank-based thruster selection ---
        scores = self._rng.sample_uniform_torch(0.0, 1.0, num_thrusters, ids=env_ids)
        sorted_idx = torch.argsort(scores, dim=-1)
        rank = torch.empty_like(sorted_idx)
        rank.scatter_(
            dim=-1,
            index=sorted_idx,
            src=torch.arange(num_thrusters, device=self._device).unsqueeze(0).expand(n, -1),
        )
        failed = rank < k_per_env.unsqueeze(-1)  # (n, num_thrusters) bool

        # --- Step 3: sample continuous values for all thrusters (masked later) ---
        scale_min, scale_max = getattr(task_cfg, "degradation_scale_range", (0.0, 1.0))
        offset_min, offset_max = getattr(task_cfg, "stuck_offset_range", (0.1, 1.0))

        raw_scales = self._rng.sample_uniform_torch(scale_min, scale_max, num_thrusters, ids=env_ids)
        raw_offsets = self._rng.sample_uniform_torch(offset_min, offset_max, num_thrusters, ids=env_ids)
        mode_scores = self._rng.sample_uniform_torch(0.0, 1.0, num_thrusters, ids=env_ids)

        # --- Step 4: assign failure modes ---
        probs = getattr(task_cfg, "failure_mode_probs", (0.33, 0.33, 0.34))
        p0 = float(probs[0])
        p01 = p0 + float(probs[1])

        is_degradation = mode_scores < p0
        is_binary_dead = (mode_scores >= p0) & (mode_scores < p01)
        is_stuck_on = mode_scores >= p01

        # Start from healthy state.
        scales = torch.ones((n, num_thrusters), device=self._device, dtype=torch.float32)
        offsets = torch.zeros((n, num_thrusters), device=self._device, dtype=torch.float32)

        # Continuous degradation: scale from range, offset stays 0.
        deg_mask = failed & is_degradation
        scales = torch.where(deg_mask, raw_scales, scales)

        # Binary dead: scale=0, offset stays 0.
        dead_mask = failed & is_binary_dead
        scales = torch.where(dead_mask, torch.zeros_like(scales), scales)

        # Stuck-on: scale=0, offset from range.
        stuck_mask = failed & is_stuck_on
        scales = torch.where(stuck_mask, torch.zeros_like(scales), scales)
        offsets = torch.where(stuck_mask, raw_offsets, offsets)

        return scales, offsets

    # ------------------------------------------------------------------ #
    # Hooks                                                              #
    # ------------------------------------------------------------------ #

    def run_setup(self, robot, envs_origin: torch.Tensor) -> None:  # type: ignore[override]
        super().run_setup(robot, envs_origin)
        num_thrusters = robot._robot_cfg.num_thrusters
        self._degradation_scales = torch.ones(
            (self._num_envs, num_thrusters), device=self._device, dtype=torch.float32
        )
        self._degradation_offsets = torch.zeros(
            (self._num_envs, num_thrusters), device=self._device, dtype=torch.float32
        )

    def reset(  # type: ignore[override]
        self,
        env_ids: torch.Tensor,
        gen_actions: torch.Tensor | None = None,
        env_seeds: torch.Tensor | None = None,
    ) -> None:
        super().reset(env_ids, gen_actions=gen_actions, env_seeds=env_seeds)
        if self._degradation_scales is None:
            return
        new_scales, new_offsets = self._sample_degradation_state(env_ids)
        self._degradation_scales[env_ids] = new_scales
        self._degradation_offsets[env_ids] = new_offsets
        if hasattr(self._robot, "set_degradation_state"):
            self._robot.set_degradation_state(new_scales, new_offsets, env_ids)

    # ------------------------------------------------------------------ #
    # Privileged observations                                            #
    # ------------------------------------------------------------------ #

    def get_privileged_observations(self) -> torch.Tensor:
        """Return ``D_gt = [scales | offsets]``, shape ``(num_envs, 2 * num_thrusters)``."""
        if self._degradation_scales is None:
            num_t = self._robot._robot_cfg.num_thrusters
            return torch.ones(
                (self._num_envs, 2 * num_t), device=self._device, dtype=torch.float32
            )
        return torch.cat([self._degradation_scales, self._degradation_offsets], dim=-1)

    # ------------------------------------------------------------------ #
    # Logging                                                            #
    # ------------------------------------------------------------------ #

    def create_logs(self) -> None:  # type: ignore[override]
        super().create_logs()
        self.scalar_logger.add_log("task_state", "AVG/degraded_thruster_count", "mean")
        self.scalar_logger.add_log("task_state", "AVG/curriculum_max_failures", "mean")

    def compute_rewards(self) -> torch.Tensor:  # type: ignore[override]
        rew = super().compute_rewards()
        if self._degradation_scales is not None:
            degraded_count = (self._degradation_scales < 1.0).float().sum(dim=-1)
            self.scalar_logger.log("task_state", "AVG/degraded_thruster_count", degraded_count)
            curr_max = torch.full_like(degraded_count, float(self._current_max_failures()))
            self.scalar_logger.log("task_state", "AVG/curriculum_max_failures", curr_max)
        return rew
