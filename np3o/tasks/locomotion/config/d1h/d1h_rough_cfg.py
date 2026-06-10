# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""D1H rough task config.

D1H is a bipedal wheeled robot (2 legs × 4 joints: hip/thigh/calf/foot).
This config follows the TITA pattern.

Robot, terrain, observations, actions, events, rewards, terminations,
costs, and simulation all live here.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco

from mjlab.actuator import IdealPdActuatorCfg
from mjlab.actuator.dc_actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
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
from mjlab.sensor import ContactSensorCfg, ContactMatch, RayCastSensorCfg, GridPatternCfg, ObjRef
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
    TerrainEntityCfg,
    TerrainGeneratorCfg,
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
# 1. Robot
# ===========================================================================

_D1H_XML_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "assets" / "robots" / "d1h" / "robot.xml"


def _get_d1h_spec() -> mujoco.MjSpec:
    """Load the D1H robot MuJoCo spec from XML."""
    return mujoco.MjSpec.from_file(str(_D1H_XML_PATH))


# Actuator groups — mirrors the D1 actuator layout.
#
# Leg joints (hip, thigh, calf): DcMotor
#   effort_limit=60.0, saturation_effort=80.0, velocity_limit=20.0,
#   stiffness=60.0, damping=1.5
#
# Wheel joints (foot): IdealPd
#   effort_limit=12.0, stiffness=0.0, damping=0.5

_D1H_LEG_ACTUATOR = DcMotorActuatorCfg(
    target_names_expr=(".*_(hip|thigh|calf)_joint",),
    stiffness=60.0,
    damping=1.5,
    effort_limit=60.0,
    saturation_effort=80.0,
    velocity_limit=20.0,
)

_D1H_WHEEL_ACTUATOR = IdealPdActuatorCfg(
    target_names_expr=(".*_foot_joint",),
    stiffness=0.0,
    damping=0.5,
    effort_limit=12.0,
)

_D1H_SOFT_JOINT_POS_LIMIT_FACTOR = 0.9
_D1H_SOFT_VEL_LIMIT_FACTOR = 0.9
_D1H_SOFT_EFFORT_LIMIT_FACTOR = 0.9

# ---- regex patterns ----
# Joint space
LEG_JOINT_PATTERN = ".*_(hip|thigh|calf)_joint"
WHEEL_JOINT_PATTERN = ".*_foot_joint"
ALL_JOINT_PATTERN = ".*"
HIP_JOINT_PATTERN = ".*_hip_joint"

# Actuator space (matches XML actuator names: FL_hip, FR_thigh, …)
LEG_ACTUATOR_PATTERN = ".*_(hip|thigh|calf)"
WHEEL_ACTUATOR_PATTERN = ".*_foot"
ALL_ACTUATOR_PATTERN = ".*"

# ---- cached SceneEntityCfg objects ----
_LEG_JOINT_CFG = SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERN, preserve_order=True)
_WHEEL_JOINT_CFG = SceneEntityCfg("robot", joint_names=WHEEL_JOINT_PATTERN, preserve_order=True)
_ALL_JOINT_CFG = SceneEntityCfg("robot", joint_names=ALL_JOINT_PATTERN, preserve_order=True)
_LEG_ACTUATOR_CFG = SceneEntityCfg("robot", actuator_names=LEG_ACTUATOR_PATTERN, preserve_order=True)
_WHEEL_ACTUATOR_CFG = SceneEntityCfg("robot", actuator_names=WHEEL_ACTUATOR_PATTERN, preserve_order=True)
_ALL_ACTUATOR_CFG = SceneEntityCfg("robot", actuator_names=ALL_ACTUATOR_PATTERN, preserve_order=True)

# One source of truth for the standing/default pose.
_D1H_DEFAULT_JOINT_POS = {
    "FL_hip_joint": 0.0,
    "FR_hip_joint": -0.0,
    ".*_thigh_joint": 0.8,
    ".*_calf_joint": -1.5,
    ".*_foot_joint": 0.0,
}

# Hard-coded D1H limits matching the XML joint ranges.
_D1H_JOINT_POS_LIMITS = {
    HIP_JOINT_PATTERN: (-0.785398, 0.785398),
    ".*_thigh_joint": (-1.8326, 3.40339),
    ".*_calf_joint": (-2.775, -0.855),
    WHEEL_JOINT_PATTERN: (-1.0e6, 1.0e6),
}

_D1H_JOINT_VEL_LIMITS = {
    LEG_JOINT_PATTERN: 20.0,
    WHEEL_JOINT_PATTERN: 30.0,
}

_D1H_ACTUATOR_EFFORT_LIMITS = {
    LEG_ACTUATOR_PATTERN: 60.0,
    WHEEL_ACTUATOR_PATTERN: 12.0,
}

_D1H_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(_D1H_LEG_ACTUATOR, _D1H_WHEEL_ACTUATOR),
    soft_joint_pos_limit_factor=_D1H_SOFT_JOINT_POS_LIMIT_FACTOR,
)

_D1H_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.45),
    joint_pos=_D1H_DEFAULT_JOINT_POS,
    joint_vel={".*": 0.0},
)


def get_d1h_cfg() -> EntityCfg:
    """Return a fresh ``EntityCfg`` for the D1H robot."""
    return EntityCfg(
        spec_fn=_get_d1h_spec,
        articulation=_D1H_ARTICULATION,
        init_state=_D1H_INIT_STATE,
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


def _d1h_rough_cost_terms() -> dict:
    """Return cost terms for D1H NP3O constrained training.

    Four terms:
    - ``pos_limit``        — joint position limit violations (leg joints hip/thigh/calf)
    - ``torque_limit``     — joint torque limit violations (leg actuators hip/thigh/calf)
    - ``dof_vel_limits``   — joint velocity limit violations (leg joints)
    - ``default_joint``    — deviation from default pose (leg joints)
    """
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


# ---- reset helper: set PD position targets to current joint positions -----


def _apply_default_joint_pos_target(
    env, env_ids, asset_cfg: SceneEntityCfg = _ALL_JOINT_CFG,
) -> None:
    """Set PD position targets to the CURRENT joint positions during reset.

    mjlab zeroes ``joint_pos_target`` inside ``sim.reset()``.  If this event does
    not run afterwards, the position-servo actuators see a target of 0 rad
    (all joints fully extended) and saturation-limit forces are produced for
    joints whose default pose is far from zero (calf ≈ -1.5, thigh ≈ 0.8).

    We sync targets to *current* joint pos (not default) because earlier reset
    events (like ``reset_joints_by_offset``) may have randomized joint positions.
    Setting target = current pos ensures zero PD error → zero torque at reset,
    matching IsaacLab's ``write_joint_state_to_sim()`` behaviour.
    """
    asset_cfg.resolve(env.scene)
    asset = env.scene[asset_cfg.name]
    q_current = asset.data.joint_pos[:, asset_cfg.joint_ids]
    asset.set_joint_position_target(q_current, joint_ids=asset_cfg.joint_ids)


# ---- inline termination: base_link contact --------------------------------


def _base_link_contact(
    env,
    sensor_name: str = "contact_forces",
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate episode when the base_link contacts the ground."""
    from mjlab.sensor import ContactSensor
    import torch

    sensor: ContactSensor = env.scene.sensors[sensor_name]
    cache = getattr(env, "_d1h_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_d1h_cache", cache)
    if "base_link_contact_ids" not in cache:
        base_ids: list[int] = []
        num_slots = int(sensor.cfg.num_slots)
        for i, name in enumerate(sensor.primary_names):
            if "base_link" in name:
                start = i * num_slots
                base_ids.extend(range(start, start + num_slots))
        cache["base_link_contact_ids"] = torch.tensor(
            base_ids, device=env.device, dtype=torch.long,
        )
    body_ids = cache["base_link_contact_ids"]
    if len(body_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    net_forces = sensor.data.force[:, body_ids, :]
    net_forces = torch.nan_to_num(net_forces, nan=0.0, posinf=0.0, neginf=0.0)
    return net_forces.norm(dim=-1).max(dim=-1).values > threshold


# ===========================================================================
# 3. Terrain
# ===========================================================================

_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    curriculum=True,
    sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=0.2),
        "pyramid_stairs": BoxPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.2),
            step_width=0.3,
            platform_width=3.0,
        ),
        "pyramid_stairs_inv": BoxInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.2),
            step_width=0.3,
            platform_width=3.0,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 1.0),
            platform_width=2.0,
        ),
        "hf_pyramid_slope_inv": HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 1.0),
            platform_width=2.0,
            inverted=True,
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.1,
            noise_range=(0.02, 0.1),
            noise_step=0.02,
        ),
        "wave_terrain": HfWaveTerrainCfg(
            proportion=0.1,
            amplitude_range=(0.0, 0.2),
            num_waves=4,
        ),
    },
)


# ===========================================================================
# 4. D1H rough environment
# ===========================================================================


def d1h_rough_env_cfg(
    num_envs: int = 4096,
    play: bool = False,
    flatten_policy_history: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the D1H velocity-tracking environment configuration.

    Args:
        num_envs: Number of parallel environments.
        play: If True, disable noise/randomization and reduce env count.
        flatten_policy_history: If True (standard PPO), flatten the policy history dim.
            If False (NP3O), keep 3D shape for the BarlowTwins encoder.
    """
    terrain_generator = _ROUGH_TERRAIN_CFG
    has_terrain = True

    # ---- Scene ----
    terrain_cfg = TerrainEntityCfg(
        terrain_type="generator" if has_terrain else "plane",
        terrain_generator=terrain_generator,
        max_init_terrain_level=2 if has_terrain else None,
        env_spacing=3.0,
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
        env_spacing=3.0,
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
    # D1H is bipedal with 4 joints per leg: hip (pos), thigh (pos), calf (pos),
    # foot (vel/wheel).  Hip uses a reduced action scale (0.5×), matching D1.
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
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-3.14, 3.14),
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
            "position_range": (-0.5, 1.0), "velocity_range": (-0.0, 0.0),
            "asset_cfg": _LEG_JOINT_CFG,
        },
    )

    # CRITICAL: set PD position targets to the default joint pose so the
    # position-servo actuators do not fight to reach 0 rad during the
    # mj_forward() call that follows reset events.
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
        mujoco=MujocoCfg(timestep=0.005, iterations=120, ls_iterations=60, ccd_iterations=40),
        njmax=1500,
        contact_sensor_maxmatch=500,
        nan_guard=NanGuardCfg(enabled=False, output_dir="/tmp/mjlab/nan_dumps"),
    )

    cfg = ManagerBasedRlEnvCfg(
        scene=scene, observations=observations, actions=actions,
        events=events, rewards=rewards, terminations=terminations,
        commands=commands, curriculum=curriculum, sim=sim,
        decimation=4, episode_length_s=20.0,
    )

    # Rough terrain reset profile — higher spawn, wider roll/pitch tolerance.
    cfg.events["reset_base"].params["pose_range"].update({
        "z": (0.2, 0.5),
        "roll": (-0.5, 0.5),
        "pitch": (-0.5, 0.5),
        "yaw": (-3.14, 3.14),
    })

    if play:
        cfg.scene.num_envs = 50
        cfg.scene.env_spacing = 3.0
        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("base_external_force_torque", None)
        cfg.events.pop("push_robot", None)
        cfg.events.pop("randomize_actuator_gains", None)
        cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_ROOT
        cfg.viewer.entity_name = "robot"

    # Attach cost terms so downstream code can read them from the cfg directly.
    cfg.cost_terms = _d1h_rough_cost_terms()

    return cfg


def d1h_rough_play_env_cfg(
    num_envs: int = 50,
    flatten_policy_history: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Convenience wrapper: D1H rough play config (noise/randomization disabled)."""
    return d1h_rough_env_cfg(num_envs=num_envs, play=True, flatten_policy_history=flatten_policy_history)
