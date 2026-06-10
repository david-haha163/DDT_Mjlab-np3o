# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""D1H flat task config.

D1H flat environment — plane terrain, no height scanner, no terrain curriculum.
Inherits robot definitions from the rough config and only overrides terrain/MDP
aspects that differ between flat and rough.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
    base_ang_vel,
    base_lin_vel,
    flat_orientation_l2,
    joint_acc_l2,
    joint_torques_l2,
    joint_vel_rel,
    last_action,
    projected_gravity,
    reset_root_state_uniform,
    reset_joints_by_offset,
    apply_external_force_torque,
    push_by_setting_velocity,
    time_out,
)
from mjlab.envs.mdp.dr import pd_gains
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
    CurriculumTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import (
    TerrainEntityCfg,
    TerrainGeneratorCfg,
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
    HfPyramidSlopedTerrainCfg,
)
from mjlab.utils.nan_guard import NanGuardCfg
from mjlab.utils.noise.noise_cfg import UniformNoiseCfg

from ...cmdp.commands import UniformVelocityCommandCfg
from ...cmdp.curriculums import terrain_levels_vel
from ...cmdp.observations import (
    contact_state,
    joint_kp_factor,
    joint_kd_factor,
    joint_pos_rel_without_wheel,
    velocity_commands_3d,
    safe_height_scan,
)
from ...cmdp.rewards import (
    action_rate_l2,
    ang_vel_xy_l2,
    base_height_l2,
    default_joint_l2,
    lin_vel_z_l2,
    joint_mirror,
    stand_still,
    track_ang_vel_z_exp,
    track_lin_vel_xy_exp,
    undesired_contacts,
    upward,
)
from ...cmdp.terminations import out_of_terrain_bounds, root_state_nonfinite

# ===========================================================================
# 1. Robot (imported from rough config)
# ===========================================================================

from .d1h_rough_cfg import (
    get_d1h_cfg,
    _D1H_DEFAULT_JOINT_POS,
    _D1H_SOFT_JOINT_POS_LIMIT_FACTOR,
    _D1H_SOFT_VEL_LIMIT_FACTOR,
    _D1H_SOFT_EFFORT_LIMIT_FACTOR,
    _D1H_JOINT_POS_LIMITS,
    _D1H_JOINT_VEL_LIMITS,
    _D1H_ACTUATOR_EFFORT_LIMITS,
    LEG_JOINT_PATTERN,
    WHEEL_JOINT_PATTERN,
    LEG_ACTUATOR_PATTERN,
    WHEEL_ACTUATOR_PATTERN,
    _LEG_JOINT_CFG,
    _WHEEL_JOINT_CFG,
    _ALL_JOINT_CFG,
    _LEG_ACTUATOR_CFG,
    _ALL_ACTUATOR_CFG,
    _apply_default_joint_pos_target,
    _base_link_contact,
)


# ===========================================================================
# 2. Cost functions
# ===========================================================================

from ...cmdp.costs import (
    cost_default_joint,
    cost_dof_vel_limits,
    cost_pos_limit,
    cost_torque_limit,
)


def _d1h_flat_cost_terms() -> dict:
    """Return cost terms for D1H NP3O constrained training (flat)."""
    from np3o.algorithms.np3o.cost_manager import CostTermCfg

    return {
        "pos_limit": CostTermCfg(
            func=cost_pos_limit,
            scale=1.0, d_value=0.0, k_value=0.01,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERN),
                "joint_pos_limit_patterns": _D1H_JOINT_POS_LIMITS,
                "soft_ratio": _D1H_SOFT_JOINT_POS_LIMIT_FACTOR,
            },
        ),
        "torque_limit": CostTermCfg(
            func=cost_torque_limit,
            scale=1.0, d_value=0.0, k_value=0.01,
            params={
                "asset_cfg": SceneEntityCfg("robot", actuator_names=LEG_ACTUATOR_PATTERN),
                "actuator_effort_limit_patterns": _D1H_ACTUATOR_EFFORT_LIMITS,
                "soft_ratio": _D1H_SOFT_EFFORT_LIMIT_FACTOR,
            },
        ),
        "dof_vel_limits": CostTermCfg(
            func=cost_dof_vel_limits,
            scale=1.0, d_value=0.0, k_value=0.01,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERN),
                "joint_vel_limit_patterns": _D1H_JOINT_VEL_LIMITS,
                "soft_ratio": _D1H_SOFT_VEL_LIMIT_FACTOR,
            },
        ),
        "default_joint": CostTermCfg(
            func=cost_default_joint,
            scale=0.2, d_value=0.0, k_value=0.01,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERN),
                "default_joint_pos_patterns": _D1H_DEFAULT_JOINT_POS,
            },
        ),
    }


# ===========================================================================
# 3. Flat terrain
# ===========================================================================

_FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    num_rows=10,
    border_width=10.0,
    curriculum=True,
    sub_terrains={
        "random_uniform": HfRandomUniformTerrainCfg(
            proportion=0.4,
            noise_range=(0.0, 0.05),
        ),
        "waves": HfWaveTerrainCfg(
            proportion=0.2,
            amplitude_range=(0.0, 0.2),
            num_waves=2,
        ),
        "slopes": HfPyramidSlopedTerrainCfg(
            proportion=0.4,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
        ),
    },
)


# ===========================================================================
# 4. D1H flat environment
# ===========================================================================


def d1h_flat_env_cfg(
    num_envs: int = 4096,
    play: bool = False,
    flatten_policy_history: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the D1H velocity-tracking environment configuration for flat terrain.

    Args:
        num_envs: Number of parallel environments.
        play: If True, disable noise/randomization and reduce env count.
        flatten_policy_history: If True (standard PPO), flatten the policy history dim.
            If False (NP3O), keep 3D shape for the BarlowTwins encoder.
    """
    terrain_generator = _FLAT_TERRAIN_CFG
    has_terrain = True

    # ---- Scene ----
    terrain_cfg = TerrainEntityCfg(
        terrain_type="generator" if has_terrain else "plane",
        terrain_generator=terrain_generator,
        max_init_terrain_level=5 if has_terrain else None,
        env_spacing=2.5,
    )

    sensors: list = [
        ContactSensorCfg(
            name="contact_forces",
            primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
            fields=("found", "force"),
            track_air_time=True,
            history_length=2,
        ),
    ]

    if has_terrain:
        from mjlab.sensor import RayCastSensorCfg, GridPatternCfg, ObjRef
        sensors.append(
            RayCastSensorCfg(
                name="height_scanner",
                frame=ObjRef(type="body", name="base_link", entity="robot"),
                pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
                ray_alignment="yaw",
                max_distance=5.0,
                exclude_parent_body=True,
            ),
        )

    scene = SceneCfg(
        terrain=terrain_cfg,
        entities={"robot": get_d1h_cfg()},
        sensors=tuple(sensors),
        num_envs=num_envs if not play else 50,
        env_spacing=2.5,
    )

    # ---- Observations ----
    policy_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=base_ang_vel, params={"asset_cfg": _ALL_JOINT_CFG},
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2), clip=(-100.0, 100.0), scale=0.25,
        ),
        "projected_gravity": ObservationTermCfg(
            func=projected_gravity, params={"asset_cfg": _ALL_JOINT_CFG},
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05), clip=(-100.0, 100.0), scale=1.0,
        ),
        "velocity_commands": ObservationTermCfg(
            func=velocity_commands_3d, params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0), scale=(2.0, 2.0, 0.25),
        ),
        "joint_pos": ObservationTermCfg(
            func=joint_pos_rel_without_wheel,
            params={
                "asset_cfg": _ALL_JOINT_CFG,
                "wheel_asset_cfg": _WHEEL_JOINT_CFG,
                "default_joint_pos_patterns": _D1H_DEFAULT_JOINT_POS,
            },
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01), clip=(-100.0, 100.0), scale=1.0,
        ),
        "joint_vel": ObservationTermCfg(
            func=joint_vel_rel, params={"asset_cfg": _ALL_JOINT_CFG},
            noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5), clip=(-100.0, 100.0), scale=0.05,
        ),
        "actions": ObservationTermCfg(
            func=last_action, clip=(-100.0, 100.0), scale=1.0,
        ),
    }

    critic_terms = {
        "base_lin_vel": ObservationTermCfg(func=base_lin_vel, clip=(-100.0, 100.0), scale=2.0),
        "base_ang_vel": ObservationTermCfg(
            func=base_ang_vel, params={"asset_cfg": _ALL_JOINT_CFG}, clip=(-100.0, 100.0), scale=0.25,
        ),
        "projected_gravity": ObservationTermCfg(
            func=projected_gravity, params={"asset_cfg": _ALL_JOINT_CFG}, clip=(-100.0, 100.0), scale=1.0,
        ),
        "velocity_commands": ObservationTermCfg(
            func=velocity_commands_3d, params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0), scale=(2.0, 2.0, 0.25),
        ),
        "joint_pos": ObservationTermCfg(
            func=joint_pos_rel_without_wheel,
            params={
                "asset_cfg": _ALL_JOINT_CFG,
                "wheel_asset_cfg": _WHEEL_JOINT_CFG,
                "default_joint_pos_patterns": _D1H_DEFAULT_JOINT_POS,
            },
            clip=(-100.0, 100.0), scale=1.0,
        ),
        "joint_vel": ObservationTermCfg(
            func=joint_vel_rel, params={"asset_cfg": _ALL_JOINT_CFG}, clip=(-100.0, 100.0), scale=0.05,
        ),
        "actions": ObservationTermCfg(func=last_action, clip=(-100.0, 100.0), scale=1.0),
    }

    priv_terms = {
        "contact_state": ObservationTermCfg(
            func=contact_state, params={"sensor_name": "contact_forces", "body_names": ".*_foot"},
            clip=(-1.0, 1.0), scale=1.0,
        ),
        "joint_kp_factor": ObservationTermCfg(
            func=joint_kp_factor, params={"asset_cfg": _ALL_JOINT_CFG}, clip=(0.0, 2.0), scale=1.0,
        ),
        "joint_kd_factor": ObservationTermCfg(
            func=joint_kd_factor, params={"asset_cfg": _ALL_JOINT_CFG}, clip=(0.0, 2.0), scale=1.0,
        ),
    }

    scanner_terms = {}
    if has_terrain:
        scanner_terms["height_scan"] = ObservationTermCfg(
            func=safe_height_scan, params={"sensor_name": "height_scanner"},
            clip=(-1.0, 1.0), scale=1.0,
        )

    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms, enable_corruption=not play,
            concatenate_terms=True, history_length=10, flatten_history_dim=flatten_policy_history,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms, enable_corruption=False, concatenate_terms=True,
        ),
        "priv": ObservationGroupCfg(
            terms=priv_terms, enable_corruption=False, concatenate_terms=True,
        ),
    }
    if has_terrain:
        observations["scanner"] = ObservationGroupCfg(
            terms=scanner_terms, enable_corruption=False, concatenate_terms=True,
        )

    # ---- Actions ----
    _action_scale = 0.25
    _hip_scale_reduction = 0.5

    actions = {
        "left_leg_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"),
            scale={
                ".*_hip_joint": _action_scale * _hip_scale_reduction,
                ".*_thigh_joint": _action_scale,
                ".*_calf_joint": _action_scale,
            },
            clip={".*": (-100.0, 100.0)},
            preserve_order=True,
        ),
        "left_leg_vel": JointVelocityActionCfg(
            entity_name="robot",
            actuator_names=("FL_foot_joint",),
            scale=5.0,
            clip={".*": (-100.0, 100.0)},
            preserve_order=True,
        ),
        "right_leg_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"),
            scale={
                ".*_hip_joint": _action_scale * _hip_scale_reduction,
                ".*_thigh_joint": _action_scale,
                ".*_calf_joint": _action_scale,
            },
            clip={".*": (-100.0, 100.0)},
            preserve_order=True,
        ),
        "right_leg_vel": JointVelocityActionCfg(
            entity_name="robot",
            actuator_names=("FR_foot_joint",),
            scale=5.0,
            clip={".*": (-100.0, 100.0)},
            preserve_order=True,
        ),
    }

    # ---- Events ----
    events: dict[str, EventTermCfg] = {}

    if not play:
        events["randomize_actuator_gains"] = EventTermCfg(
            func=pd_gains, mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
                "kp_range": (0.8, 1.2), "kd_range": (0.8, 1.2),
                "operation": "scale", "distribution": "uniform",
            },
        )

    events["reset_base"] = EventTermCfg(
        func=reset_root_state_uniform, mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5),
                "roll": (-0.0, 0.0), "pitch": (-0, 0), "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
            "asset_cfg": _ALL_JOINT_CFG,
        },
    )

    events["reset_robot_joints"] = EventTermCfg(
        func=reset_joints_by_offset, mode="reset",
        params={
            "position_range": (-0.001, 0.001), "velocity_range": (-0.0, 0.0),
            "asset_cfg": _LEG_JOINT_CFG,
        },
    )

    events["apply_default_joint_pos_target"] = EventTermCfg(
        func=_apply_default_joint_pos_target, mode="reset",
        params={"asset_cfg": _ALL_JOINT_CFG},
    )

    if not play:
        events["base_external_force_torque"] = EventTermCfg(
            func=apply_external_force_torque, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "force_range": (-10.0, 10.0), "torque_range": (-10.0, 10.0),
            },
        )
        events["push_robot"] = EventTermCfg(
            func=push_by_setting_velocity, mode="interval", interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0)}},
        )

    # ---- Rewards ----
    std = math.sqrt(0.25)
    rewards = {
        "track_lin_vel_xy_exp": RewardTermCfg(
            func=track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": std},
        ),
        "track_ang_vel_z_exp": RewardTermCfg(
            func=track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": std},
        ),
        "lin_vel_z_l2": RewardTermCfg(func=lin_vel_z_l2, weight=-2.0),
        "ang_vel_xy_l2": RewardTermCfg(func=ang_vel_xy_l2, weight=-0.05),
        "flat_orientation_l2": RewardTermCfg(func=flat_orientation_l2, weight=-5.0),
        "base_height_l2": RewardTermCfg(func=base_height_l2, weight=-10.0, params={"target_height": 0.45}),
        "joint_torques_l2": RewardTermCfg(func=joint_torques_l2, weight=-1.0e-5, params={"asset_cfg": _LEG_ACTUATOR_CFG}),
        "joint_acc_l2": RewardTermCfg(func=joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": _ALL_JOINT_CFG}),
        "action_rate_l2": RewardTermCfg(func=action_rate_l2, weight=-0.01),
        "joint_mirror": RewardTermCfg(
            func=joint_mirror, weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "mirror_joints": [["FL_(hip|thigh|calf)_joint", "FR_(hip|thigh|calf)_joint"]],
            },
        ),
        "stand_still": RewardTermCfg(
            func=stand_still, weight=0.0,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_PATTERN),
            },
        ),
        "undesired_contacts": RewardTermCfg(
            func=undesired_contacts, weight=-1.0,
            params={"sensor_name": "contact_forces", "body_names": ".*_calf", "threshold": 1.0},
        ),
        "upward": RewardTermCfg(func=upward, weight=0.5),
        "default_joint_l2": RewardTermCfg(
            func=default_joint_l2, weight=-0.5,
            params={"asset_cfg": _LEG_JOINT_CFG, "default_joint_pos_patterns": _D1H_DEFAULT_JOINT_POS},
        ),
    }

    # ---- Terminations ----
    terminations = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
        "out_of_terrain_bounds": TerminationTermCfg(
            func=out_of_terrain_bounds, time_out=True,
        ),
        "root_state_nonfinite": TerminationTermCfg(func=root_state_nonfinite),
        "base_contact": TerminationTermCfg(
            func=_base_link_contact,
            params={"sensor_name": "contact_forces", "threshold": 1.0},
        ),
    }

    # ---- Commands ----
    commands = {
        "base_velocity": UniformVelocityCommandCfg(
            resampling_time_range=(10.0, 10.0),
            entity_name="robot",
            rel_standing_envs=0.02,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=False,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.0, 0.0),
                ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi),
            ),
        ),
    }

    # ---- Curriculum ----
    curriculum = {}
    if has_terrain:
        curriculum["terrain_levels"] = CurriculumTermCfg(
            func=terrain_levels_vel, params={"command_name": "base_velocity"},
        )

    # ---- Simulation ----
    sim = SimulationCfg(
        mujoco=MujocoCfg(timestep=0.005, iterations=80, ls_iterations=40, ccd_iterations=20),
        njmax=1500,
        contact_sensor_maxmatch=256,
        nan_guard=NanGuardCfg(enabled=False, output_dir="/tmp/mjlab/nan_dumps"),
    )

    cfg = ManagerBasedRlEnvCfg(
        scene=scene, observations=observations, actions=actions,
        events=events, rewards=rewards, terminations=terminations,
        commands=commands, curriculum=curriculum, sim=sim,
        decimation=4, episode_length_s=20.0,
    )

    # Flat terrain reset profile.
    cfg.events["reset_base"].params["pose_range"].update({
        "z": (0.0, 0.2),
        "roll": (-0.05, 0.05),
        "pitch": (-0.05, 0.05),
        "yaw": (-3.14, 3.14),
    })

    if play:
        cfg.scene.num_envs = 50
        cfg.scene.env_spacing = 2.5
        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("base_external_force_torque", None)
        cfg.events.pop("push_robot", None)
        cfg.events.pop("randomize_actuator_gains", None)
        cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_ROOT
        cfg.viewer.entity_name = "robot"

    # Attach cost terms so downstream code can read them from the cfg directly.
    cfg.cost_terms = _d1h_flat_cost_terms()

    return cfg


def d1h_flat_play_env_cfg(
    num_envs: int = 50,
    flatten_policy_history: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Convenience wrapper: D1H flat play config (noise/randomization disabled)."""
    return d1h_flat_env_cfg(num_envs=num_envs, play=True, flatten_policy_history=flatten_policy_history)
