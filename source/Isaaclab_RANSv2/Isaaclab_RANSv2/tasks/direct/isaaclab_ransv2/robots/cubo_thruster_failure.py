# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cubo robot variant that injects per-episode thruster failures.

Degradation state: each thruster has a ``(scale, offset)`` pair.  The applied thrust is::

    applied[i] = scale[i] * commanded[i] + offset[i]

Healthy:               scale=1, offset=0  → identity
Binary dead:           scale=0, offset=0  → zero force
Continuous degradation: scale∈(0,1), offset=0
Stuck-on:              scale=0, offset∈(0,1]

The state is owned by the task mixin and pushed via :meth:`set_degradation_state`.
:meth:`set_failure_mask` is kept for backward-compatibility with the binary-mask mixin.
"""

from __future__ import annotations

import torch

from isaaclab.scene import InteractiveScene

from ..robots_cfg import CuboThrusterFailureRobotCfg

from .cubo import CuboRobot


class CuboThrusterFailureRobot(CuboRobot):
    """Cubo with per-environment affine thruster degradation applied post-policy."""

    def __init__(
        self,
        scene: InteractiveScene | None = None,
        robot_cfg: CuboThrusterFailureRobotCfg = CuboThrusterFailureRobotCfg(),
        robot_uid: int = 0,
        num_envs: int = 1,
        decimation: int = 6,
        device: str = "cuda",
    ) -> None:
        super().__init__(
            scene=scene,
            robot_cfg=robot_cfg,
            robot_uid=robot_uid,
            num_envs=num_envs,
            decimation=decimation,
            device=device,
        )

    def initialize_buffers(self, env_ids=None) -> None:
        super().initialize_buffers(env_ids)
        num_t = self._robot_cfg.num_thrusters
        if not hasattr(self, "_thruster_scales"):
            self._thruster_scales = torch.ones(
                (self._num_envs, num_t), device=self._device, dtype=torch.float32
            )
            self._thruster_offsets = torch.zeros(
                (self._num_envs, num_t), device=self._device, dtype=torch.float32
            )

    @property
    def thruster_failure_mask(self) -> torch.Tensor:
        """Backward-compatible view: returns the scale tensor (1=healthy, <1=degraded)."""
        return self._thruster_scales

    def set_degradation_state(
        self,
        scales: torch.Tensor,
        offsets: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Write scale+offset degradation state for the given envs.

        Args:
            scales:  Shape ``(len(env_ids), num_thrusters)``. Multiplier applied to commanded thrust.
            offsets: Shape ``(len(env_ids), num_thrusters)``. Constant offset added after scaling.
            env_ids: Env indices to update. ``None`` updates all envs.
        """
        if env_ids is None:
            self._thruster_scales.copy_(scales)
            self._thruster_offsets.copy_(offsets)
        else:
            self._thruster_scales[env_ids] = scales
            self._thruster_offsets[env_ids] = offsets

    def set_failure_mask(self, mask: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Backward-compatible binary mask: scale=mask, offset=0."""
        n = len(env_ids) if env_ids is not None else self._num_envs
        offsets = torch.zeros((n, self._robot_cfg.num_thrusters), device=self._device)
        self.set_degradation_state(mask, offsets, env_ids)

    def apply_actions(self) -> None:
        """Apply affine degradation then defer to the parent implementation."""
        # _thrust_action shape: (num_envs, num_thrusters, 3); thrust magnitude lives in [..., 2].
        self._thrust_action[:, :, 2] = (
            self._thrust_action[:, :, 2] * self._thruster_scales + self._thruster_offsets
        )
        super().apply_actions()
        
    def compute_rewards(self):
        # TODO: DT should be factored in?
        
        """
        Reward term reference for Cubo
        --------------------------------
        All terms are penalties (negative scales) applied on top of the task-level reward.
        
        THRUSTERS
          joint_acceleration    (rew_joint_accel_scale=-2.5e-6)
              Penalizes the sum of squared joint accelerations across all joints.
              Discourages high-frequency, jerky motion anywhere in the articulation.
        
          thruster_action_rate  (rew_action_rate_scale=-0.12/8, direct mode only)
              Penalizes the L1 change in thruster commands between consecutive steps.
              Sliced to thruster dims only so RW does not inflate this term.
              Encourages smooth, gradual thrust transitions rather than bang-bang control.
        
          thruster_effort       (rew_thruster_effort_scale=-0.01)
              Penalizes the total normalized thrust summed over all 8 thrusters.
              Captures sustained activation cost that action_rate misses (a robot
              holding constant thrust pays zero action_rate but non-zero effort).
        
        REACTION WHEEL
          rw_saturation         (rew_reaction_wheel_saturation_scale=-0.1)
              Penalizes omega_rw^2 (internal reaction wheel angular speed, squared).
              A saturated wheel cannot produce torque; this keeps speed well below the
              physical limit so heading authority is always available.
        
          rw_usage              (rew_reaction_wheel_usage_scale=-0.05)
              Penalizes |commanded_torque| — discourages spinning the wheel unnecessarily.
              Together with rw_saturation: the policy learns to use the wheel only when
              there is a heading error to correct and to desaturate it afterwards.
        """

        joint_accelerations = torch.sum(torch.square(self.joint_acc), dim=1)
        self.scalar_logger.log("robot_state", "AVG/joint_acceleration", joint_accelerations)
        self.scalar_logger.log("robot_reward", "AVG/joint_acceleration", joint_accelerations * self._robot_cfg.rew_joint_accel_scale)

        reward = joint_accelerations * self._robot_cfg.rew_joint_accel_scale
        # if self._robot_cfg.direct_thruster_control:
        #     # Slice only thruster dims so RW changes don't pollute this term
        #     thruster_action_rate = torch.sum(
        #         torch.abs(
        #             self._unaltered_actions[:, : self._robot_cfg.num_thrusters]
        #             - self._previous_unaltered_actions[:, : self._robot_cfg.num_thrusters]
        #         ),
        #         dim=1,
        #     )
        #     self.scalar_logger.log("robot_state", "AVG/thruster_action_rate", thruster_action_rate)
        #     self.scalar_logger.log("robot_reward", "AVG/thruster_action_rate", thruster_action_rate * self._robot_cfg.rew_action_rate_scale)
        #     reward = reward + thruster_action_rate * self._robot_cfg.rew_action_rate_scale

        # # --- Thruster effort: penalize sustained total thrust (fuel cost) ---
        # thruster_effort = torch.sum(torch.abs(self._thrust_action[:, :, 2]), dim=-1) / self._robot_cfg.max_thrust
        # self.scalar_logger.log("robot_state", "AVG/thruster_effort", thruster_effort)
        # self.scalar_logger.log("robot_reward", "AVG/thruster_effort", thruster_effort * self._robot_cfg.rew_thruster_effort_scale)
        # reward = reward + thruster_effort * self._robot_cfg.rew_thruster_effort_scale

        # if self._robot_cfg.has_reaction_wheel:
        #     # --- RW saturation: penalize high internal wheel speed (quadratic) ---
        #     rw_saturation = torch.square(self.omega_reation_wheel).squeeze(-1)
        #     self.scalar_logger.log("robot_state", "AVG/rw_saturation", rw_saturation)
        #     self.scalar_logger.log("robot_reward", "AVG/rw_saturation", rw_saturation * self._robot_cfg.rew_reaction_wheel_saturation_scale)
        #     reward = reward + rw_saturation * self._robot_cfg.rew_reaction_wheel_saturation_scale

        #     # --- RW usage: penalize commanded torque magnitude ---
        #     rw_usage = torch.abs(self._reaction_wheel_action).squeeze(-1)
        #     self.scalar_logger.log("robot_state", "AVG/rw_usage", rw_usage)
        #     self.scalar_logger.log("robot_reward", "AVG/rw_usage", rw_usage * self._robot_cfg.rew_reaction_wheel_usage_scale)
        #     reward = reward + rw_usage * self._robot_cfg.rew_reaction_wheel_usage_scale

        return torch.zeros_like(reward)
        return reward
