# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for D1 velocity-tracking locomotion.

Ported from ``ddt_lab.tasks.manager_based.locomotion.mdp.rewards``,
adapted to mjlab APIs. Every function keeps the original logic and
only updates field names and import paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    yaw_quat,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRLEnv
    from mjlab.sensor import ContactSensor, RayCastSensor


# ---------------------------------------------------------------------------
# Small tensor/cache helpers
# ---------------------------------------------------------------------------


def _upright_gate(asset: Entity) -> torch.Tensor:
    """Gate penalties/rewards out when the robot is tumbling."""
    return torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7


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


def _cached_tensor(env: ManagerBasedRLEnv, key: tuple, values, dtype=torch.float32) -> torch.Tensor:
    cache = _cache(env)
    if key not in cache:
        cache[key] = torch.tensor(values, device=env.device, dtype=dtype)
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


def _contact_body_ids(env: ManagerBasedRLEnv, sensor_name: str, body_names) -> torch.Tensor:
    """Resolve contact-sensor slot ids once, then reuse the GPU index tensor.

    Several reward/observation terms used to run regex matching and Python list
    construction every simulation step.  Body names and slot layout are static,
    so caching avoids repeated Python work and CPU->GPU tensor construction.
    """
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


def _vectors_in_base_frame(env: ManagerBasedRLEnv, asset: Entity, vectors_w: torch.Tensor) -> torch.Tensor:
    """Batch-transform vectors from world frame to base frame.

    Args:
        vectors_w: tensor with shape (num_envs, num_items, 3).
    """
    num_items = vectors_w.shape[1]
    root_quat = asset.data.root_link_quat_w[:, None, :].expand(-1, num_items, -1).reshape(-1, 4)
    return quat_apply_inverse(root_quat, vectors_w.reshape(-1, 3)).reshape(env.num_envs, num_items, 3)


# ---------------------------------------------------------------------------
# Velocity tracking (exponential kernel)
# ---------------------------------------------------------------------------


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_link_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= _upright_gate(env.scene["robot"])
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= _upright_gate(env.scene["robot"])
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_link_quat_w), asset.data.root_link_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= _upright_gate(env.scene["robot"])
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_link_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= _upright_gate(env.scene["robot"])
    return reward


# ---------------------------------------------------------------------------
# Root penalties
# ---------------------------------------------------------------------------


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_link_lin_vel_b[:, 2])
    reward *= _upright_gate(env.scene["robot"])
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
    reward *= _upright_gate(env.scene["robot"])
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= _upright_gate(env.scene["robot"])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCastSensor = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_link_pos_w[:, 2] - adjusted_target_height)
    reward *= _upright_gate(env.scene["robot"])
    return reward


# ---------------------------------------------------------------------------
# Joint penalties
# ---------------------------------------------------------------------------


def joint_torques_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint torques on the articulation using L2 squared kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.actuator_force[:, asset_cfg.actuator_ids]), dim=1)


def joint_vel_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_pos_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits."""
    asset: Entity = env.scene[asset_cfg.name]
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def joint_vel_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_ratio: float = 0.9,
    vel_limit: float = 20.0,
) -> torch.Tensor:
    """Penalize joint velocities if they exceed the soft velocity limit."""
    asset: Entity = env.scene[asset_cfg.name]
    limit = vel_limit * soft_ratio
    return torch.sum((asset.data.joint_vel[:, asset_cfg.joint_ids].abs() - limit).clamp(min=0.0), dim=1)


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.actuator_force[:, asset_cfg.actuator_ids]),
        dim=1,
    )
    return reward


# ---------------------------------------------------------------------------
# Action penalties
# ---------------------------------------------------------------------------


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    reward = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    reward *= _upright_gate(env.scene["robot"])
    return reward


# ---------------------------------------------------------------------------
# Contact sensor rewards
# ---------------------------------------------------------------------------


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_name: str = "contact_forces", body_names: str = "^(?!.*_foot).*") -> torch.Tensor:
    """Penalize undesired contacts as the number of violations above a threshold.

    Tensor/cache optimized: body-name regex matching is resolved once and kept as
    a GPU LongTensor on the environment.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
    body_ids = _contact_body_ids(env, sensor_name, body_names)
    force_history = contact_sensor.data.force_history  # (B, N, H, 3)
    is_contact = force_history[:, body_ids, :, :].norm(dim=-1).amax(dim=-1) > threshold
    reward = is_contact.sum(dim=1).float()
    reward *= _upright_gate(env.scene["robot"])
    return reward

def contact_forces(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Penalize large contact forces with cached contact body ids."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = _contact_body_ids(env, sensor_cfg.name, sensor_cfg.body_names)
    force_history = contact_sensor.data.force_history  # (B, N, H, 3)
    is_contact = force_history[:, body_ids, :, :].norm(dim=-1).amax(dim=-1) > threshold
    return is_contact.sum(dim=1).float()

def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def default_joint_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    default_joint_pos_patterns: dict[str, float] | None = None,
) -> torch.Tensor:
    """Penalize joint position deviation from the task-defined default pose."""
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    if default_joint_pos_patterns is None:
        q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    else:
        q_default = _cached_scalar_tensor_from_patterns(
            env,
            asset=asset,
            ids=asset_cfg.joint_ids,
            names=asset.joint_names,
            count=asset.num_joints,
            patterns=default_joint_pos_patterns,
            kind="rew_default_joint_pos",
        ).expand_as(q)
    reward = torch.sum(torch.square(q - q_default), dim=1)
    reward *= _upright_gate(asset)
    return reward


def hip_pos(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lateral_command_index: int = 1,
    command_threshold: float = 1.0e-6,
    default_joint_pos_patterns: dict[str, float] | None = None,
) -> torch.Tensor:
    """Penalize hip joint deviation from default when lateral command is ~0."""
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    flag = (cmd[:, lateral_command_index].abs() < command_threshold).float()

    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    if default_joint_pos_patterns is None:
        q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    else:
        q_default = _cached_scalar_tensor_from_patterns(
            env,
            asset=asset,
            ids=asset_cfg.joint_ids,
            names=asset.joint_names,
            count=asset.num_joints,
            patterns=default_joint_pos_patterns,
            kind="rew_hip_default_joint_pos",
        ).expand_as(q)
    reward = flag * torch.sum(torch.square(q - q_default), dim=1)
    reward *= _upright_gate(asset)
    return reward


# ---------------------------------------------------------------------------
# Stand-still and gait penalties
# ---------------------------------------------------------------------------


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    from mjlab.envs.mdp import (
        joint_deviation_l1,
    )

    reward = joint_deviation_l1(env, asset_cfg)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= _upright_gate(env.scene["robot"])
    return reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= _upright_gate(env.scene["robot"])
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids].abs()

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = _contact_body_ids(env, sensor_cfg.name, sensor_cfg.body_names)
    num_slots = int(contact_sensor.cfg.num_slots)
    in_air_per_slot = contact_sensor.compute_first_air(env.step_dt)[:, body_ids]
    in_air = in_air_per_slot.reshape(env.num_envs, -1, num_slots).any(dim=-1)

    running_reward = (in_air * joint_vel).sum(dim=1)
    standing_reward = joint_vel.sum(dim=1)
    return torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )

class GaitReward:
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        pass

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_link_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= _upright_gate(env.scene["robot"])
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


# ---------------------------------------------------------------------------
# Symmetry rewards
# ---------------------------------------------------------------------------


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= _upright_gate(env.scene["robot"])
    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= _upright_gate(env.scene["robot"])
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= _upright_gate(env.scene["robot"])
    return reward


# ---------------------------------------------------------------------------
# Feet / stepping rewards
# ---------------------------------------------------------------------------


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= _upright_gate(env.scene["robot"])
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= _upright_gate(env.scene["robot"])
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= _upright_gate(env.scene["robot"])
    return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= _upright_gate(env.scene["robot"])
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= _upright_gate(env.scene["robot"])
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = _contact_body_ids(env, sensor_cfg.name, sensor_cfg.body_names)
    forces = contact_sensor.data.force[:, body_ids, :]
    forces_z = forces[:, :, 2].abs()
    forces_xy = torch.linalg.norm(forces[:, :, :2], dim=2)
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= _upright_gate(env.scene["robot"])
    return reward

def feet_distance_y_exp(
    env: ManagerBasedRLEnv,
    stance_width: float = 0.40,
    std: float = 0.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    foot_rel_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[:, None, :]
    feet_b = _vectors_in_base_frame(env, asset, foot_rel_w)

    # asset_cfg.body_ids may be a slice after MJLab entity resolution, so do not use len().
    n_feet = feet_b.shape[1]
    side_sign = _cached_tensor(
        env,
        ("feet_distance_y_side_sign", n_feet),
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
    )
    desired_ys = (0.5 * float(stance_width)) * side_sign.unsqueeze(0)
    stance_diff = (desired_ys - feet_b[:, :, 1]).square()
    reward = torch.exp(-stance_diff.sum(dim=1) / (std**2))
    reward *= _upright_gate(env.scene["robot"])
    return reward

def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float = 0.40,
    stance_length: float = 0.45,
    std: float = 0.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    foot_rel_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[:, None, :]
    feet_b = _vectors_in_base_frame(env, asset, foot_rel_w)

    # Expected order: FL, FR, RL, RR.  Cache this 4x2 template once.
    desired_xy = _cached_tensor(
        env,
        ("feet_distance_xy_template", float(stance_width), float(stance_length)),
        [
            [0.5 * stance_length, 0.5 * stance_width],
            [0.5 * stance_length, -0.5 * stance_width],
            [-0.5 * stance_length, 0.5 * stance_width],
            [-0.5 * stance_length, -0.5 * stance_width],
        ],
    )
    stance_diff = (desired_xy.unsqueeze(0) - feet_b[:, :, :2]).square().sum(dim=2)
    reward = torch.exp(-stance_diff.sum(dim=1) / (std**2))
    reward *= _upright_gate(env.scene["robot"])
    return reward

def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: Entity = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= _upright_gate(env.scene["robot"])
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward swinging feet for clearing a specified base-frame height.

    Tensor optimized: transform all feet in one batched quaternion call instead
    of one Python loop per foot.
    """
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_rel_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[:, None, :]
    foot_vel_rel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_link_lin_vel_w[:, None, :]
    foot_pos_b = _vectors_in_base_frame(env, asset, foot_pos_rel_w)
    foot_vel_b = _vectors_in_base_frame(env, asset, foot_vel_rel_w)

    foot_z_target_error = (foot_pos_b[:, :, 2] - target_height).square()
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.linalg.norm(foot_vel_b[:, :, :2], dim=2))
    reward = (foot_z_target_error * foot_velocity_tanh).sum(dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= _upright_gate(env.scene["robot"])
    return reward

def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding using cached contact ids and batched frame transforms."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = _contact_body_ids(env, sensor_cfg.name, sensor_cfg.body_names)
    force_history = contact_sensor.data.force_history  # (B, N, H, 3)
    contacts = force_history[:, body_ids, :, :].norm(dim=-1).amax(dim=-1) > 1.0

    asset: Entity = env.scene[asset_cfg.name]
    foot_vel_rel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_link_lin_vel_w[:, None, :]
    foot_vel_b = _vectors_in_base_frame(env, asset, foot_vel_rel_w)
    foot_lateral_vel = torch.linalg.norm(foot_vel_b[:, :, :2], dim=2)
    reward = (foot_lateral_vel * contacts).sum(dim=1)
    reward *= _upright_gate(env.scene["robot"])
    return reward

