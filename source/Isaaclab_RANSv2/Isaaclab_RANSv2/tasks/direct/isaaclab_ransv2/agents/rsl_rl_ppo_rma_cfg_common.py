# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared model config for the RMA Phase-1 PPO agents (GoToPose + GoToPosition)."""

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg


@configclass
class RslRlRMAModelCfg(RslRlMLPModelCfg):
    """Runner-side config for :class:`rsl_rl.models.RMAModel`."""

    class_name: str = "RMAModel"
    """The model class name."""

    privileged_obs_set: str | None = None
    """Observation set name for the privileged stream. Defaults to ``f"{obs_set}_privileged"``."""

    latent_dim: int = 8
    """Output dimension of the privileged-info encoder ``mu``."""

    encoder_hidden_dims: list[int] = MISSING
    """Hidden dimensions of the privileged-info encoder ``mu``."""
