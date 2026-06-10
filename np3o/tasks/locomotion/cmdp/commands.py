# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Uniform velocity command for locomotion tasks.

mjlab does not ship a built-in ``UniformVelocityCommand``, so this module
implements one following the IsaacLab ``UniformVelocityCommandCfg`` semantics
exactly: sampled linear/angular velocity targets with optional heading
command, resampled at configurable intervals.

Behaviour ports ``isaaclab.envs.mdp.commands.velocity_command.UniformVelocityCommand``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for a uniform velocity command generator.

    Samples linear velocity (x, y), angular velocity (z), and optionally a
    heading direction uniformly from the configured ranges.
    """

    @dataclass
    class Ranges:
        """Min/max ranges for each command dimension."""

        lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        """Linear velocity x range [m/s]."""
        lin_vel_y: tuple[float, float] = (-1.0, 1.0)
        """Linear velocity y range [m/s]."""
        ang_vel_z: tuple[float, float] = (-1.0, 1.0)
        """Angular velocity z range [rad/s]."""
        heading: tuple[float, float] = (-math.pi, math.pi)
        """Heading range [rad]."""

    entity_name: str = "robot"
    """Name of the entity for which commands are generated."""

    ranges: Ranges = field(default_factory=Ranges)
    """Uniform sampling ranges for each command dimension."""

    rel_standing_envs: float = 0.02
    """Probability of sampling zero-velocity commands (standing)."""

    rel_heading_envs: float = 1.0
    """Probability of using heading-based commands instead of ang_vel_z."""

    heading_command: bool = False
    """Whether to use heading commands instead of body-frame ang_vel_z."""

    heading_control_stiffness: float = 0.5
    """Proportional gain for heading control. ``heading_error * stiffness → ang_vel_z``."""

    def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
        return UniformVelocityCommand(self, env)


class UniformVelocityCommand(CommandTerm):
    """Uniform velocity command generator for locomotion.

    On resample, draws ``lin_vel_x``, ``lin_vel_y``, ``ang_vel_z``, and
    ``heading`` uniformly from the configured ranges. When heading_command is
    enabled, ``ang_vel_z`` is computed from the heading error.
    """

    cfg: UniformVelocityCommandCfg

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.entity_name]

        # Command tensor: (num_envs, 4) — [lin_vel_x, lin_vel_y, ang_vel_z, heading]
        self._command = torch.zeros(self.num_envs, 4, device=self.device)

        # Track the heading when resampled so we can compute heading error.
        self._heading_target = torch.zeros(self.num_envs, device=self.device)

        # Flag: is this env using heading command or ang_vel_z?
        self._is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Flag: is this env standing (zero command)?
        self._is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Current command tensor of shape ``(num_envs, 4)``.

        Columns: ``[lin_vel_x, lin_vel_y, ang_vel_z, heading]``.
        """
        return self._command

    # ------------------------------------------------------------------
    # CommandTerm abstract methods
    # ------------------------------------------------------------------

    def _update_metrics(self) -> None:
        """Track mean commanded velocity for logging.

        Divide by ``max_command_step`` (GO1 pattern) so the accumulated metric
        equals the per-resample-interval mean rather than an ever-growing sum.
        """
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        if "mean_lin_vel" not in self.metrics:
            self.metrics["mean_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        if "mean_ang_vel" not in self.metrics:
            self.metrics["mean_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["mean_lin_vel"] += torch.norm(self._command[:, :2], dim=-1) / max_command_step
        self.metrics["mean_ang_vel"] += self._command[:, 2].abs() / max_command_step

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Sample new velocity commands for the given environments.

        Reads ranges directly from ``self.cfg.ranges`` at resample time so that
        curriculum-based range updates take effect immediately.
        """
        n = len(env_ids)
        r = torch.empty(n, device=self.device)

        # Decide which envs get zero (standing) commands.
        is_standing = r.uniform_(0.0, 1.0) < self.cfg.rel_standing_envs
        self._is_standing_env[env_ids] = is_standing

        # Decide which envs get heading-based vs body-rate commands.
        is_heading = r.uniform_(0.0, 1.0) < self.cfg.rel_heading_envs
        self._is_heading_env[env_ids] = is_heading

        # Sample lin_vel_x, lin_vel_y, ang_vel_z — read ranges live from cfg.
        self._command[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self._command[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self._command[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        # Sample heading.
        heading_range = self.cfg.ranges.heading
        if heading_range is not None:
            heading = r.uniform_(*heading_range)
        else:
            heading = torch.zeros(n, device=self.device)
        self._command[env_ids, 3] = heading
        self._heading_target[env_ids] = heading

        # Zero out standing envs.
        standing_env_ids = self._is_standing_env.nonzero(as_tuple=False).flatten()
        self._command[standing_env_ids] = 0.0

    def _update_command(self) -> None:
        """Update heading-based commands: convert heading error to ang_vel_z.

        When ``heading_command`` is enabled, the stored ``ang_vel_z`` is
        overwritten with a proportional controller on the heading error.
        """
        if not self.cfg.heading_command:
            return

        entity = self._env.scene[self.cfg.entity_name]
        # Compute current yaw from root orientation.
        # projected_gravity_b is (num_envs, 3) with gravity in body frame.
        # We get yaw from the root orientation quaternion.
        root_quat = entity.data.root_link_quat_w  # (num_envs, 4)  w,x,y,z
        # Yaw from quaternion: atan2(2*(qw*qz + qx*qy), 1 - 2*(qy² + qz²))
        w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        current_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        # Heading error wrapped to [-pi, pi].
        heading_error = torch.atan2(
            torch.sin(self._heading_target - current_yaw),
            torch.cos(self._heading_target - current_yaw),
        )

        # Only apply to heading-enabled envs that aren't standing.
        heading_envs = self._is_heading_env & ~self._is_standing_env
        if heading_envs.any():
            self._command[heading_envs, 2] = heading_error[heading_envs] * self.cfg.heading_control_stiffness
