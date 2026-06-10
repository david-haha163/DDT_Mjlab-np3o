# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Custom observation functions for D1 velocity-tracking locomotion.

Ported from ``ddt_lab.tasks.manager_based.locomotion.mdp.observations``,
adapted to mjlab APIs. Each function keeps the original logic and only
updates import paths and type hints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp import height_scan as _mjlab_height_scan
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def _cache(env: ManagerBasedRLEnv) -> dict:
    cache = getattr(env, "_d1_tensor_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_d1_tensor_cache", cache)
    return cache


def _patterns_key(patterns) -> tuple[str, ...]:
    if isinstance(patterns, str):
        return (patterns,)
    if isinstance(patterns, (list, tuple)):
        return tuple(str(p) for p in patterns)
    return (str(patterns),)


def _contact_body_ids(env: ManagerBasedRLEnv, sensor_name: str, body_names) -> torch.Tensor:
    import re

    sensor = env.scene.sensors[sensor_name]
    patterns = _patterns_key(body_names)
    key = ("contact_body_ids", sensor_name, patterns, int(sensor.cfg.num_slots), tuple(sensor.primary_names))
    cache = _cache(env)
    if key not in cache:
        body_ids: list[int] = []
        num_slots = int(sensor.cfg.num_slots)
        for i, name in enumerate(sensor.primary_names):
            if any(re.match(p, name) for p in patterns):
                start = i * num_slots
                body_ids.extend(range(start, start + num_slots))
        cache[key] = torch.tensor(body_ids, device=env.device, dtype=torch.long)
    return cache[key]


def _ones(env: ManagerBasedRLEnv, key: tuple, shape: tuple[int, ...]) -> torch.Tensor:
    cache = _cache(env)
    if key not in cache:
        cache[key] = torch.ones(*shape, device=env.device)
    return cache[key]


def _ids_to_list(ids, count: int) -> list[int]:
    if isinstance(ids, slice):
        return list(range(count))[ids]
    return list(ids)


def _normalise_value_patterns(patterns: dict) -> tuple:
    items = []
    for pattern, value in patterns.items():
        if isinstance(value, (tuple, list)):
            items.append((str(pattern), tuple(float(v) for v in value)))
        else:
            items.append((str(pattern), float(value)))
    return tuple(items)


def _cached_scalar_tensor_from_patterns(
    env: ManagerBasedRLEnv,
    *,
    asset: Entity,
    ids,
    names: list[str],
    count: int,
    patterns: dict[str, float],
    kind: str,
) -> torch.Tensor:
    import re

    ids_list = _ids_to_list(ids, count)
    key = (kind, id(asset), tuple(ids_list), _normalise_value_patterns(patterns), str(env.device))
    cache = _cache(env)
    if key in cache:
        return cache[key]

    values = []
    for idx in ids_list:
        name = names[idx]
        matched = None
        for pattern, value in patterns.items():
            if re.fullmatch(pattern, name) or re.search(pattern, name):
                matched = value
                break
        if matched is None:
            raise ValueError(f"No {kind} pattern matched '{name}'.")
        values.append(float(matched))
    cache[key] = torch.tensor(values, device=env.device, dtype=torch.float32).unsqueeze(0)
    return cache[key]


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    default_joint_pos_patterns: dict[str, float] | None = None,
) -> torch.Tensor:
    """Joint positions relative to the task-defined default pose, with wheels zeroed.

    When ``default_joint_pos_patterns`` is provided, observations use the same
    default pose as reset/init and ``cost_default_joint`` instead of relying on
    ``asset.data.default_joint_pos``.
    """
    asset: Entity = env.scene[asset_cfg.name]
    if default_joint_pos_patterns is None:
        default_pos = asset.data.default_joint_pos
    else:
        default_pos = _cached_scalar_tensor_from_patterns(
            env,
            asset=asset,
            ids=slice(None),
            names=asset.joint_names,
            count=asset.num_joints,
            patterns=default_joint_pos_patterns,
            kind="obs_default_joint_pos",
        ).expand_as(asset.data.joint_pos)
    joint_pos_rel = asset.data.joint_pos - default_pos
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    joint_pos_rel = joint_pos_rel[:, asset_cfg.joint_ids]
    return torch.nan_to_num(joint_pos_rel, nan=0.0, posinf=100.0, neginf=-100.0)


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


# ---------------------------------------------------------------------------
# Privileged observation terms (critic-only)
# Mirrors the reference ``LocomotionWithNP3O`` priv_latent subset that can be
# obtained from Isaac Lab's Articulation data without extra PhysX API calls.
# ---------------------------------------------------------------------------


def contact_state(
    env: ManagerBasedRLEnv,
    sensor_name: str = "contact_forces",
    body_names: str = ".*_foot",
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary foot-contact state, centred at zero: +0.5 in contact, -0.5 not.

    Tensor/cache optimized: contact body ids are resolved once instead of doing
    regex matching every observation step.
    """
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene.sensors[sensor_name]
    body_ids = _contact_body_ids(env, sensor_name, body_names)
    net_forces = sensor.data.force[:, body_ids, :]  # (B, K, 3)
    net_forces = torch.nan_to_num(net_forces, nan=0.0, posinf=0.0, neginf=0.0)
    in_contact = (net_forces.norm(dim=-1) > threshold).float()
    return in_contact - 0.5

def joint_kp_factor(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Randomised kp (joint stiffness) as a scale factor relative to default.

    Shape: ``(num_envs, num_joints)``.
    Mirrors reference ``kp_factor`` in priv_latent; values outside [0, 2]
    are clamped to avoid very large signals from near-zero defaults.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # In mjlab, actuator gains are accessed via the actuator, not EntityData.
    # For IdealPdActuator, stiffness/damping are stored as tensors on the
    # actuator instance. We default to 1.0 (no randomization active) when
    # actuator data is not directly queryable from EntityData.
    joint_ids = asset_cfg.joint_ids
    num_joints = asset.num_joints if isinstance(joint_ids, slice) else len(joint_ids)
    return _ones(env, ("joint_kp_factor", num_joints, env.num_envs), (env.num_envs, num_joints))


def joint_kd_factor(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Randomised kd (joint damping) as a scale factor relative to default.

    Shape: ``(num_envs, num_joints)``.
    Mirrors reference ``kd_factor`` in priv_latent.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # Same reasoning as joint_kp_factor — default to 1.0.
    joint_ids = asset_cfg.joint_ids
    num_joints = asset.num_joints if isinstance(joint_ids, slice) else len(joint_ids)
    return _ones(env, ("joint_kd_factor", num_joints, env.num_envs), (env.num_envs, num_joints))


# ---------------------------------------------------------------------------
# NP3O velocity commands (3D — excludes heading from policy observation).
# Mirrors IsaacLab's ``generated_commands`` which only exposes the three
# velocity components (lin_vel_x, lin_vel_y, ang_vel_z) to the actor.
# The heading is an internal controller state, not a policy input.
# ---------------------------------------------------------------------------


def velocity_commands_3d(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Velocity commands without the heading dimension (policy-safe).

    Shape: ``(num_envs, 3)`` — [lin_vel_x, lin_vel_y, ang_vel_z].
    """
    command = env.command_manager.get_command(command_name)
    assert command is not None
    # Command tensor is (num_envs, 4): [lin_vel_x, lin_vel_y, ang_vel_z, heading].
    # Drop the heading column — the actor must not see it directly.
    return command[:, :3]


def safe_height_scan(
    env: ManagerBasedRLEnv,
    sensor_name: str = "height_scanner",
    nan_value: float = 0.0,
    inf_value: float = 1.0,
) -> torch.Tensor:
    """Height scan with finite-value protection.

    On high rough terrains or near terrain borders, RayCastSensor can occasionally
    return NaN/Inf for missed or degenerate rays.  A single non-finite scanner
    value can poison the critic/normalizer and eventually the policy update.
    """
    x = _mjlab_height_scan(env, sensor_name=sensor_name)
    return torch.nan_to_num(x, nan=nan_value, posinf=inf_value, neginf=-inf_value)
