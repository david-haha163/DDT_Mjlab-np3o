# Copyright (c) 2024-2026
"""D1-specific termination functions."""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg


def root_state_nonfinite(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    tensors = [
        asset.data.root_link_pos_w,
        asset.data.root_link_quat_w,
        asset.data.root_link_lin_vel_w,
        asset.data.root_link_ang_vel_w,
        asset.data.joint_pos,
        asset.data.joint_vel,
    ]
    bad = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for x in tensors:
        bad |= ~torch.isfinite(x.reshape(env.num_envs, -1)).all(dim=1)
    return bad


def out_of_terrain_bounds(
    env,
    margin: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Truncate if robot leaves the generated terrain footprint.

    Returns all-false for non-generator terrains (e.g. plane).
    """
    terrain = env.scene.terrain
    if terrain is None or terrain.cfg.terrain_type != "generator":
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    terrain_generator = terrain.cfg.terrain_generator
    if terrain_generator is None or terrain.terrain_origins is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    root_xy_w = env.scene[asset_cfg.name].data.root_link_pos_w[:, :2]

    num_rows, num_cols = terrain.terrain_origins.shape[:2]
    half_x = 0.5 * (num_rows * terrain_generator.size[0]) + terrain_generator.border_width
    half_y = 0.5 * (num_cols * terrain_generator.size[1]) + terrain_generator.border_width
    limit_x = max(0.0, half_x - margin)
    limit_y = max(0.0, half_y - margin)

    return (root_xy_w[:, 0].abs() > limit_x) | (root_xy_w[:, 1].abs() > limit_y)
