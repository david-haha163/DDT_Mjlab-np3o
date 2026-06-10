# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions for D1 velocity-tracking locomotion.

- ``terrain_levels_vel`` ported from IsaacLab.
- ``commands_vel`` follows the mjlab GO1 pattern (staged velocity range
  expansion).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains import TerrainEntity

from .commands import UniformVelocityCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
    """A single stage in the ``commands_vel`` curriculum.

    Only the dimensions listed are updated when the step threshold is reached;
    omitted keys keep their previous values.  ``step`` is measured in
    environment steps (``env.common_step_counter``).
    """

    step: int
    lin_vel_x: tuple[float, float] | None
    lin_vel_y: tuple[float, float] | None
    ang_vel_z: tuple[float, float] | None


def terrain_levels_vel(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`mjlab.terrains.TerrainEntity` class.

    Returns:
        A dict with per-terrain-type mean levels and summary stats.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]

    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    command = env.command_manager.get_command(command_name)
    assert command is not None

    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)

    levels = terrain.terrain_levels.float()
    max_level = max(terrain_generator.num_rows - 1, 1)
    return {
        "terrain_levels_mean": torch.mean(levels) / max_level * 10.0,
    }


def commands_vel(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
    """Staged velocity command curriculum — expand command ranges as training progresses.

    Each stage specifies a ``step`` threshold (measured in ``env.common_step_counter``)
    and the command range dimensions that should take effect from that point onward.
    Later stages override earlier ones when multiple thresholds are reached.

    Example stages (GO1-style, assuming ``num_steps_per_env=24``)::

        velocity_stages = [
            {"step": 0,        "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
            {"step": 5000*24,  "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
            {"step": 10000*24, "lin_vel_x": (-2.0, 3.0)},
        ]

    Returns:
        A dict logging the current range bounds for each dimension.
    """
    del env_ids  # Unused — curriculum is global, not per-env.

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None
    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

    for stage in velocity_stages:
        if env.common_step_counter >= stage["step"]:
            if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
                cfg.ranges.lin_vel_x = stage["lin_vel_x"]
            if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
                cfg.ranges.lin_vel_y = stage["lin_vel_y"]
            if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
                cfg.ranges.ang_vel_z = stage["ang_vel_z"]

    return {
        "command_max": torch.tensor(max(abs(cfg.ranges.lin_vel_x[0]), cfg.ranges.lin_vel_x[1])),
    }
