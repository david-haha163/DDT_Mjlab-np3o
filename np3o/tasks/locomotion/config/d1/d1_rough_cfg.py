# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""D1 rough task config.

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
    joint_pos_limits,
    joint_torques_l2,
    joint_vel_l2,
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
from ...cmdp.curriculums import commands_vel, terrain_levels_vel
from ...cmdp.observations import (
    contact_state,
    joint_kp_factor,
    joint_kd_factor,
    joint_pos_rel_without_wheel,
    velocity_commands_3d,
    safe_height_scan,
)
from ...cmdp.terminations import out_of_terrain_bounds, root_state_nonfinite
from ...cmdp.rewards import (
    action_rate_l2,
    ang_vel_xy_l2,
    base_height_l2,
    default_joint_l2,
    lin_vel_z_l2,
    track_ang_vel_z_exp,
    track_lin_vel_xy_exp,
    undesired_contacts,
    upward,
)

# ===========================================================================
# 1. Robot
# ===========================================================================

_D1_XML_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "assets" / "robots" / "d1" / "robot.xml"


def _get_d1_spec() -> mujoco.MjSpec:
    """Load the D1 robot MuJoCo spec from XML."""
    return mujoco.MjSpec.from_file(str(_D1_XML_PATH))

# Actuator groups — mirrors the IsaacLab ``DDT_D1_CFG.actuators`` layout.
#
# IsaacLab DCMotorCfg for legs:
#   effort_limit=60.0, saturation_effort=80.0, velocity_limit=20.0,
#   stiffness=60.0, damping=1.5, friction=0.0
# → mjlab DcMotorActuatorCfg(stiffness=60.0, damping=1.5,
#   effort_limit=60.0, saturation_effort=80.0, velocity_limit=20.0)
#
# IsaacLab ImplicitActuatorCfg for wheels:
#   effort_limit_sim=12.0, velocity_limit_sim=30.0, stiffness=0.0,
#   damping=0.5, friction=0.0
# → mjlab IdealPdActuatorCfg(stiffness=0.0, damping=0.5, effort_limit=12.0)

_D1_LEG_ACTUATOR = DcMotorActuatorCfg(
    target_names_expr=(".*_(hip|thigh|calf)_joint",),
    stiffness=60.0,
    damping=1.5,
    effort_limit=60.0,
    saturation_effort=80.0,
    velocity_limit=20.0,
)

_D1_WHEEL_ACTUATOR = IdealPdActuatorCfg(
    target_names_expr=(".*_foot_joint",),
    stiffness=0.0,
    damping=0.5,
    effort_limit=12.0,
)

_D1_SOFT_JOINT_POS_LIMIT_FACTOR = 0.9
_D1_SOFT_VEL_LIMIT_FACTOR = 0.9
_D1_SOFT_EFFORT_LIMIT_FACTOR = 0.9

# ---- regex patterns (ordered: more specific first) ----
# Joint space
LEG_JOINT_PATTERN = ".*_(hip|thigh|calf)_joint"
WHEEL_JOINT_PATTERN = ".*_foot_joint"
ALL_JOINT_PATTERN = ".*"
HIP_JOINT_PATTERN = ".*_hip_joint"
THIGH_CALF_JOINT_PATTERN = ".*_(thigh|calf)_joint"

# Actuator space (PD actuator names carry the joint name)
LEG_ACTUATOR_PATTERN = ".*_(hip|thigh|calf)_joint"
WHEEL_ACTUATOR_PATTERN = ".*_foot_joint"
ALL_ACTUATOR_PATTERN = ".*"

# ---- cached SceneEntityCfg objects ----
_LEG_JOINT_CFG = SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERN, preserve_order=True)
_WHEEL_JOINT_CFG = SceneEntityCfg("robot", joint_names=WHEEL_JOINT_PATTERN, preserve_order=True)
_ALL_JOINT_CFG = SceneEntityCfg("robot", joint_names=ALL_JOINT_PATTERN, preserve_order=True)
_LEG_ACTUATOR_CFG = SceneEntityCfg("robot", actuator_names=LEG_ACTUATOR_PATTERN, preserve_order=True)
_WHEEL_ACTUATOR_CFG = SceneEntityCfg("robot", actuator_names=WHEEL_ACTUATOR_PATTERN, preserve_order=True)
_ALL_ACTUATOR_CFG = SceneEntityCfg("robot", actuator_names=ALL_ACTUATOR_PATTERN, preserve_order=True)

# One source of truth for the standing/default pose.  Reset/init reuses this
# same dict, and cost_default_joint also receives this same dict.
_D1_DEFAULT_JOINT_POS = {
    ".*L_hip_joint": 0.0,
    ".*R_hip_joint": -0.0,
    "F.*_thigh_joint": 0.8,
    "R.*_thigh_joint": 0.8,
    ".*_calf_joint": -1.5,
    ".*_foot_joint": -1.5,
}

# Hard-coded D1 limits matching the patterns above.  These are no longer read
# by the cost functions from MJCF at runtime, which keeps training behaviour
# stable and explicit.
_D1_JOINT_POS_LIMITS = {
    HIP_JOINT_PATTERN: (-0.785398, 0.785398),
    ".*_thigh_joint": (-1.8326, 3.40339),
    ".*_calf_joint": (-2.775, -0.855),
    WHEEL_JOINT_PATTERN: (-1.0e6, 1.0e6),
}

_D1_JOINT_VEL_LIMITS = {
    LEG_JOINT_PATTERN: 20.0,
    WHEEL_JOINT_PATTERN: 30.0,
}

_D1_JOINT_EFFORT_LIMITS = {
    LEG_JOINT_PATTERN: 60.0,
    WHEEL_JOINT_PATTERN: 12.0,
}

_D1_ACTUATOR_EFFORT_LIMITS = {
    LEG_ACTUATOR_PATTERN: 60.0,
    WHEEL_ACTUATOR_PATTERN: 12.0,
}

_D1_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(_D1_LEG_ACTUATOR, _D1_WHEEL_ACTUATOR),
    soft_joint_pos_limit_factor=_D1_SOFT_JOINT_POS_LIMIT_FACTOR,
)

_D1_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.60),
    joint_pos=_D1_DEFAULT_JOINT_POS,
    joint_vel={".*": 0.0},
)


def get_d1_cfg() -> EntityCfg:
    """Return a fresh ``EntityCfg`` for the D1 robot."""
    return EntityCfg(
        spec_fn=_get_d1_spec,
        articulation=_D1_ARTICULATION,
        init_state=_D1_INIT_STATE,
    )


# ===========================================================================
# 2. Cost functions
# ===========================================================================

from ...cmdp.costs import (
    cost_default_joint,
    cost_dof_vel_limits,
    cost_hip_pos,
    cost_pos_limit,
    cost_torque_limit,
)


def _d1_rough_cost_terms() -> dict:
    """Return cost terms for D1 NP3O constrained training.

    Five terms matching the IsaacLab ``D1FlatCfg.costs``:

    - ``pos_limit``        — joint position limit violations (all leg joints)
    - ``torque_limit``     — joint torque limit violations (all leg actuators)
    - ``dof_vel_limits``   — joint velocity limit violations
    - ``hip_pos``          — hip joint deviation from zero
    - ``default_joint``    — thigh + calf deviation from default pose
    """
    from np3o.algorithms.np3o.cost_manager import CostTermCfg

    return {
        "pos_limit": CostTermCfg(
            func=cost_pos_limit,
            scale=1.0, d_value=0.0, k_value=0.01,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERN),
                "joint_pos_limit_patterns": _D1_JOINT_POS_LIMITS,
                "soft_ratio": _D1_SOFT_JOINT_POS_LIMIT_FACTOR,
            },
        ),
        "torque_limit": CostTermCfg(
            func=cost_torque_limit,
            scale=1.0, d_value=0.0, k_value=0.01,
            params={
                "asset_cfg": SceneEntityCfg("robot", actuator_names=LEG_ACTUATOR_PATTERN),
                "actuator_effort_limit_patterns": _D1_ACTUATOR_EFFORT_LIMITS,
                "soft_ratio": _D1_SOFT_EFFORT_LIMIT_FACTOR,
            },
        ),
        "dof_vel_limits": CostTermCfg(
            func=cost_dof_vel_limits,
            scale=1.0, d_value=0.0, k_value=0.01,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERN),
                "joint_vel_limit_patterns": _D1_JOINT_VEL_LIMITS,
                "soft_ratio": _D1_SOFT_VEL_LIMIT_FACTOR,
            },
        ),
        "hip_pos": CostTermCfg(
            func=cost_hip_pos,
            scale=2.0, d_value=0.0, k_value=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HIP_JOINT_PATTERN)},
        ),
        "default_joint": CostTermCfg(
            func=cost_default_joint,
            scale=0.2, d_value=0.0, k_value=0.01,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=THIGH_CALF_JOINT_PATTERN),
                "default_joint_pos_patterns": _D1_DEFAULT_JOINT_POS,
            },
        ),
    }


# ---- reset helper: set PD position targets to default joint pose ----------


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


# ===========================================================================
# 3. Environment constants
# ===========================================================================



# ---------------------------------------------------------------------------
# Rough terrain (7 sub-terrain types, matches go1_velocity ROUGH_TERRAINS_CFG)
# ---------------------------------------------------------------------------

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


def d1_rough_env_cfg(
    num_envs: int = 4096,
    play: bool = False,
    flatten_policy_history: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the D1 velocity-tracking environment configuration.

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
            # Two frames are enough for contact-state/undesired-contact rewards.
            # Keeping four frames for every body in every env noticeably increases
            # GPU memory on both flat and rough tasks.
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
        entities={"robot": get_d1_cfg()},
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
                "default_joint_pos_patterns": _D1_DEFAULT_JOINT_POS,
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
                "default_joint_pos_patterns": _D1_DEFAULT_JOINT_POS,
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

    # ---- Action scales (mirrors D1FlatCfg.control) ----
    _action_scale = 0.25
    _hip_scale_reduction = 0.5

    # ---- Actions ----
    actions = {}
    for leg in ("FL", "FR", "RL", "RR"):
        actions[f"{leg.lower()}_leg_pos"] = JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(f"{leg}_hip_joint", f"{leg}_thigh_joint", f"{leg}_calf_joint"),
            scale={
                ".*_hip_joint": _action_scale * _hip_scale_reduction,
                ".*_thigh_joint": _action_scale,
                ".*_calf_joint": _action_scale,
            },
            clip={".*": (-100.0, 100.0)},
            preserve_order=True,
        )
        actions[f"{leg.lower()}_foot_vel"] = JointVelocityActionCfg(
            entity_name="robot",
            actuator_names=(f"{leg}_foot_joint",),
            scale=5.0,
            clip={".*": (-100.0, 100.0)},
            preserve_order=True,
        )

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
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.2),
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
            # Large random offsets on all joints make the posture costs non-zero
            # from reset. Keep early training close to the nominal pose and do not
            # randomize wheel positions here.
            "position_range": (-0.1, 0.1), "velocity_range": (-0.0, 0.0),
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
            func=track_lin_vel_xy_exp, weight=2.0, params={"command_name": "base_velocity", "std": std},
        ),
        "track_ang_vel_z_exp": RewardTermCfg(
            func=track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": std},
        ),
        "lin_vel_z_l2": RewardTermCfg(func=lin_vel_z_l2, weight=-2.0),
        "ang_vel_xy_l2": RewardTermCfg(func=ang_vel_xy_l2, weight=-0.05),
        "flat_orientation_l2": RewardTermCfg(func=flat_orientation_l2, weight=-1.0),
        "base_height_l2": RewardTermCfg(func=base_height_l2, weight=-1.0, params={"target_height": 0.45}),
        "joint_torques_l2": RewardTermCfg(func=joint_torques_l2, weight=0.0, params={"asset_cfg": _LEG_ACTUATOR_CFG}),
        "joint_vel_l2": RewardTermCfg(func=joint_vel_l2, weight=0.0, params={"asset_cfg": _LEG_JOINT_CFG}),
        "joint_acc_l2": RewardTermCfg(func=joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": _ALL_JOINT_CFG}),
        "joint_pos_limits": RewardTermCfg(func=joint_pos_limits, weight=-0.0, params={"asset_cfg": _LEG_JOINT_CFG}),
        "action_rate_l2": RewardTermCfg(func=action_rate_l2, weight=-0.01),
        "undesired_contacts": RewardTermCfg(
            func=undesired_contacts, weight=-1.0,
            params={"sensor_name": "contact_forces", "body_names": "^(?!.*_foot).*", "threshold": 1.0},
        ),
        "upward": RewardTermCfg(func=upward, weight=0.5),
        "default_joint_l2": RewardTermCfg(
            func=default_joint_l2, weight=-0.5,
            params={"asset_cfg": _LEG_JOINT_CFG, "default_joint_pos_patterns": _D1_DEFAULT_JOINT_POS},
        ),
        "hip_pos": RewardTermCfg(
            func=default_joint_l2, weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
                "default_joint_pos_patterns": _D1_DEFAULT_JOINT_POS,
            },
        ),
    }

    # ---- Terminations ----
    terminations = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
        "out_of_terrain_bounds": TerminationTermCfg(
            func=out_of_terrain_bounds, time_out=True,
        ),
        "root_state_nonfinite": TerminationTermCfg(func=root_state_nonfinite),
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
                lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0),
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

    # curriculum["commands_vel"] = CurriculumTermCfg(
    #     func=commands_vel,
    #     params={
    #         "command_name": "base_velocity",
    #         "velocity_stages": [
    #             {"step": 0, "lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
    #             {"step": 2000 * 24, "lin_vel_x": (-2.0, 2.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-2.0, 2.0)},
    #             {"step": 4000 * 24, "lin_vel_x": (-3.0, 3.0), "lin_vel_y": (-1.5, 1.5), "ang_vel_z": (-3.0, 3.0)},
    #         ],
    #     },
    # )

    # ---- Simulation ----
    sim = SimulationCfg(
        # Rough terrain needs robust contact.  Use the 120/60/40 profile.
        mujoco=MujocoCfg(timestep=0.005, iterations=120, ls_iterations=60, ccd_iterations=40),
        njmax=1500,
        contact_sensor_maxmatch=500,
        # Keep MJLab's dump-heavy NanGuard off during normal training.  The
        # wrapper/normalizer still sanitize non-finite obs/costs.  Re-enable only
        # when actively debugging a NaN reproduction.
        nan_guard=NanGuardCfg(enabled=False, output_dir="/tmp/mjlab/nan_dumps"),
    )

    cfg = ManagerBasedRlEnvCfg(
        scene=scene, observations=observations, actions=actions,
        events=events, rewards=rewards, terminations=terminations,
        commands=commands, curriculum=curriculum, sim=sim,
        decimation=4, episode_length_s=20.0,
    )

    # Rough terrain reset profile — higher spawn to clear stairs/obstacles,
    # wider roll/pitch tolerance for uneven surfaces.
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
        # Track base_link instead of auto-detecting (which can pick a foot/wheel).
        cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_ROOT
        cfg.viewer.entity_name = "robot"

    # Attach cost terms so downstream code can read them from the cfg directly.
    cfg.cost_terms = _d1_rough_cost_terms()

    return cfg


def d1_rough_play_env_cfg(
    num_envs: int = 50,
    flatten_policy_history: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Convenience wrapper: D1 rough play config (noise/randomization disabled)."""
    return d1_rough_env_cfg(num_envs=num_envs, play=True, flatten_policy_history=flatten_policy_history)
