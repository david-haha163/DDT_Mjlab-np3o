# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cost functions for NP3O constrained training, adapted for mjlab.

These costs intentionally do NOT read torque/velocity/position/default limits
from MJCF at runtime.  The D1 task config is the source of truth and passes
hard-coded pattern dictionaries into these functions.  Cost functions only
resolve joint/actuator ids and build cached tensors from those dictionaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import re

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRLEnv


# Pattern dictionaries are parsed once per asset/id-selection/device.  The cost
# functions are called every training step, so do not regex-match every step.
_PATTERN_TENSOR_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}


def _ids_to_list(ids: list[int] | slice, count: int) -> list[int]:
    """Return entity-local ids as a concrete list."""
    if isinstance(ids, slice):
        return list(range(count))[ids]
    return list(ids)


def _normalise_pattern_items(patterns: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Create a hashable cache key while preserving dict order."""
    items: list[tuple[str, Any]] = []
    for pattern, value in patterns.items():
        if isinstance(value, (tuple, list)):
            items.append((str(pattern), tuple(float(v) for v in value)))
        else:
            items.append((str(pattern), float(value)))
    return tuple(items)


def _cached_tensor_from_patterns(
    *,
    asset,
    ids: list[int] | slice,
    names: list[str],
    count: int,
    device: torch.device | str,
    patterns: dict[str, Any],
    value_dim: int,
    kind: str,
) -> torch.Tensor:
    """Build and cache a tensor by matching selected names against regex patterns.

    Args:
        asset: mjlab articulation entity, used only for cache identity.
        ids: Entity-local joint/actuator ids selected by SceneEntityCfg.
        names: ``asset.joint_names`` or ``asset.actuator_names``.
        count: Total number of joints/actuators in the asset.
        device: Torch device for the output tensor.
        patterns: Ordered regex-pattern dictionary.
        value_dim: 1 for scalar values, 2 for [low, high] ranges.
        kind: Human-readable label used in errors and cache separation.
    """
    ids_list = _ids_to_list(ids, count)
    patterns_key = _normalise_pattern_items(patterns)
    key = (kind, id(asset), tuple(ids_list), patterns_key, str(device), value_dim)

    cached = _PATTERN_TENSOR_CACHE.get(key)
    if cached is not None:
        return cached

    values: list[Any] = []
    for idx in ids_list:
        name = names[idx]
        matched_value = None
        for pattern, value in patterns.items():
            if re.fullmatch(pattern, name) or re.search(pattern, name):
                matched_value = value
                break
        if matched_value is None:
            raise ValueError(
                f"No {kind} pattern matched '{name}'. Add a rule for this name "
                "or restrict the SceneEntityCfg selection."
            )

        if value_dim == 1:
            values.append(float(matched_value))
        elif value_dim == 2:
            if not isinstance(matched_value, (tuple, list)) or len(matched_value) != 2:
                raise ValueError(
                    f"{kind} pattern for '{name}' must provide a (low, high) pair; "
                    f"got {matched_value!r}."
                )
            low, high = matched_value
            values.append((float(low), float(high)))
        else:
            raise ValueError(f"Unsupported value_dim={value_dim} for {kind}.")

    tensor = torch.tensor(values, device=device).unsqueeze(0)
    _PATTERN_TENSOR_CACHE[key] = tensor
    return tensor


def _gravity_gate(asset) -> torch.Tensor:
    """Same projected-gravity gate used by the IsaacLab reference costs."""
    return torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7


def cost_pos_limit(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_pos_limit_patterns: dict[str, tuple[float, float]] | None = None,
    soft_ratio: float = 1.0,
) -> torch.Tensor:
    """Sum of joint-position excursions past hard-coded soft limits."""
    asset_cfg.resolve(env.scene)
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]

    if joint_pos_limit_patterns is None:
        # Backward-compatible fallback; D1 configs should pass explicit patterns.
        soft_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    else:
        hard_limits = _cached_tensor_from_patterns(
            asset=asset,
            ids=asset_cfg.joint_ids,
            names=asset.joint_names,
            count=asset.num_joints,
            device=env.device,
            patterns=joint_pos_limit_patterns,
            value_dim=2,
            kind="joint_pos_limit",
        )
        center = 0.5 * (hard_limits[..., 0] + hard_limits[..., 1])
        half_range = 0.5 * (hard_limits[..., 1] - hard_limits[..., 0]) * soft_ratio
        soft_limits = torch.stack((center - half_range, center + half_range), dim=-1)

    out_low = -(q - soft_limits[..., 0]).clamp(max=0.0)
    out_high = (q - soft_limits[..., 1]).clamp(min=0.0)
    raw = torch.sum(out_low + out_high, dim=1)
    return raw * _gravity_gate(asset)


def cost_torque_limit(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    actuator_effort_limit_patterns: dict[str, float] | None = None,
    soft_ratio: float = 1.0,
) -> torch.Tensor:
    """Sum of |actuator force| above hard-coded effort limits."""
    asset_cfg.resolve(env.scene)
    asset = env.scene[asset_cfg.name]

    if actuator_effort_limit_patterns is None:
        raise ValueError(
            "cost_torque_limit now expects actuator_effort_limit_patterns from the task cfg."
        )

    tau = asset.data.actuator_force[:, asset_cfg.actuator_ids]

    limit = _cached_tensor_from_patterns(
        asset=asset,
        ids=asset_cfg.actuator_ids,
        names=asset.actuator_names,
        count=asset.num_actuators,
        device=env.device,
        patterns=actuator_effort_limit_patterns,
        value_dim=1,
        kind="actuator_effort_limit",
    ) * soft_ratio

    raw = torch.sum((tau.abs() - limit).clamp(min=0.0), dim=1)
    return raw * _gravity_gate(asset)


def cost_dof_vel_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_vel_limit_patterns: dict[str, float] | None = None,
    soft_ratio: float = 1.0,
) -> torch.Tensor:
    """Sum of |qdot| above hard-coded velocity limits, capped at 1 per joint."""
    asset_cfg.resolve(env.scene)
    asset = env.scene[asset_cfg.name]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]

    if joint_vel_limit_patterns is None:
        raise ValueError(
            "cost_dof_vel_limits now expects joint_vel_limit_patterns from the task cfg."
        )

    limit = _cached_tensor_from_patterns(
        asset=asset,
        ids=asset_cfg.joint_ids,
        names=asset.joint_names,
        count=asset.num_joints,
        device=env.device,
        patterns=joint_vel_limit_patterns,
        value_dim=1,
        kind="joint_vel_limit",
    ) * soft_ratio

    raw = torch.sum((qd.abs() - limit).clamp(min=0.0, max=1.0), dim=1)
    return raw * _gravity_gate(asset)


def cost_hip_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sum of squared hip-joint deviation from zero."""
    asset_cfg.resolve(env.scene)
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    raw = torch.sum(torch.square(q), dim=-1)
    return raw * _gravity_gate(asset)


def cost_default_joint(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    default_joint_pos_patterns: dict[str, float] | None = None,
) -> torch.Tensor:
    """Sum of absolute deviation from the task-defined default pose.

    D1 configs pass ``_D1_DEFAULT_JOINT_POS``.  The robot init-state uses the
    same dictionary, so reset pose and cost target cannot silently diverge.
    """
    asset_cfg.resolve(env.scene)
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]

    if default_joint_pos_patterns is None:
        # Backward-compatible fallback; D1 configs should pass explicit patterns.
        q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    else:
        q_default = _cached_tensor_from_patterns(
            asset=asset,
            ids=asset_cfg.joint_ids,
            names=asset.joint_names,
            count=asset.num_joints,
            device=env.device,
            patterns=default_joint_pos_patterns,
            value_dim=1,
            kind="default_joint_pos",
        ).expand_as(q)

    raw = torch.sum(torch.abs(q - q_default), dim=1)

    return raw * _gravity_gate(asset)
