# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable mixin that injects per-episode thruster-failure sampling, a curriculum schedule, and
a privileged-observation getter into any :class:`TaskCore` subclass.

Intended usage::

    class GoToPoseRMATask(ThrusterFailureTaskMixin, GoToPoseTask):
        pass

    class GoToPositionRMATask(ThrusterFailureTaskMixin, GoToPositionTask):
        pass

The mixin assumes the underlying task uses a robot that exposes ``set_failure_mask`` (e.g.
:class:`CuboThrusterFailureRobot`). Failure injection is per-episode: a fresh mask is sampled at
reset and held constant for the rest of the episode. The mixin does not mutate the task's
observation tensor — privileged data is exposed only through :meth:`get_privileged_observations`,
which the env then surfaces as a separate ``"privileged"`` observation group.
"""

from __future__ import annotations

import torch


class ThrusterFailureTaskMixin:
    """Mixin that adds failure-mask machinery to a :class:`TaskCore` subclass."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self._failure_mask: torch.Tensor | None = None
        self._global_step: int = 0
        self._forced_failure_count: int | None = None

    # ------------------------------------------------------------------ #
    # Curriculum                                                         #
    # ------------------------------------------------------------------ #

    def _current_max_failures(self) -> int:
        """Compute the current cap on simultaneous failures.

        The eval-mode override (:meth:`force_failure_count`) takes precedence; otherwise the
        curriculum schedule on the robot config is consulted.
        """
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
        """Advance the curriculum counter. Called by the env once per env step."""
        self._global_step += num_envs_stepped

    def force_failure_count(self, k: int | None) -> None:
        """Force every env to receive exactly ``k`` failed thrusters at episode reset.

        This bypasses the curriculum and the configured sampling mode. Pass ``None`` to restore
        normal training behaviour.
        """
        if k is not None and k < 0:
            raise ValueError(f"k must be >= 0 (got {k}).")
        self._forced_failure_count = k

    # ------------------------------------------------------------------ #
    # Failure-mask sampling                                              #
    # ------------------------------------------------------------------ #

    def _sample_failure_mask(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Draw a fresh per-episode mask for the given envs (1 = healthy, 0 = failed).

        Uses the task's :class:`PerEnvSeededRNG` so failure draws are reproducible per env-id
        across runs (matching the rest of the task's randomization).
        """
        cfg = self._robot._robot_cfg
        num_thrusters = cfg.num_thrusters
        n = len(env_ids)
        k_max = self._current_max_failures()
        sampling = (
            "fixed_cap"
            if self._forced_failure_count is not None
            else getattr(cfg, "failure_sampling", "uniform_capped")
        )

        if k_max == 0 or sampling not in ("uniform_capped", "fixed_cap"):
            return torch.ones((n, num_thrusters), device=self._device, dtype=torch.float32)

        if sampling == "fixed_cap":
            k_per_env = torch.full((n,), k_max, device=self._device, dtype=torch.long)
        else:  # uniform_capped: k ~ Uniform{0..k_max}
            k_per_env = (
                self._rng.sample_integer_torch(0, k_max + 1, 1, ids=env_ids).to(torch.long)
            )

        scores = self._rng.sample_uniform_torch(0.0, 1.0, num_thrusters, ids=env_ids)
        sorted_idx = torch.argsort(scores, dim=-1)
        rank = torch.empty_like(sorted_idx)
        rank.scatter_(
            dim=-1,
            index=sorted_idx,
            src=torch.arange(num_thrusters, device=self._device).unsqueeze(0).expand(n, -1),
        )
        failed = rank < k_per_env.unsqueeze(-1)
        return (~failed).float()

    # ------------------------------------------------------------------ #
    # Hooks                                                              #
    # ------------------------------------------------------------------ #

    def run_setup(self, robot, envs_origin: torch.Tensor) -> None:  # type: ignore[override]
        super().run_setup(robot, envs_origin)
        num_thrusters = robot._robot_cfg.num_thrusters
        self._failure_mask = torch.ones(
            (self._num_envs, num_thrusters), device=self._device, dtype=torch.float32
        )

    def reset(  # type: ignore[override]
        self,
        env_ids: torch.Tensor,
        gen_actions: torch.Tensor | None = None,
        env_seeds: torch.Tensor | None = None,
    ) -> None:
        super().reset(env_ids, gen_actions=gen_actions, env_seeds=env_seeds)
        if self._failure_mask is None:
            return
        new_mask = self._sample_failure_mask(env_ids)
        self._failure_mask[env_ids] = new_mask
        if hasattr(self._robot, "set_failure_mask"):
            self._robot.set_failure_mask(new_mask, env_ids)

    # ------------------------------------------------------------------ #
    # Exposed observations                                               #
    # ------------------------------------------------------------------ #

    def get_privileged_observations(self) -> torch.Tensor:
        """Return the per-env thruster health mask, shape ``(num_envs, num_thrusters)``."""
        if self._failure_mask is None:
            return torch.ones(
                (self._num_envs, self._task_cfg.privileged_observation_space),
                device=self._device,
                dtype=torch.float32,
            )
        return self._failure_mask

    # ------------------------------------------------------------------ #
    # Logging                                                            #
    # ------------------------------------------------------------------ #

    def create_logs(self) -> None:  # type: ignore[override]
        super().create_logs()
        self.scalar_logger.add_log("task_state", "AVG/failure_mask_active_count", "mean")
        self.scalar_logger.add_log("task_state", "AVG/curriculum_max_failures", "mean")

    def compute_rewards(self) -> torch.Tensor:  # type: ignore[override]
        rew = super().compute_rewards()
        if self._failure_mask is not None:
            failed_count = (1.0 - self._failure_mask).sum(dim=-1)
            self.scalar_logger.log("task_state", "AVG/failure_mask_active_count", failed_count)
            curr_max = torch.full_like(failed_count, float(self._current_max_failures()))
            self.scalar_logger.log("task_state", "AVG/curriculum_max_failures", curr_max)
        return rew
