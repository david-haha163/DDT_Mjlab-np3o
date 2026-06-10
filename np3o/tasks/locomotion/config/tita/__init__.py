# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TITA velocity-tracking task registration for mjlab."""

from mjlab.tasks.registry import register_mjlab_task

from np3o.algorithms.np3o.runner import MjlabNP3ORunner

from .tita_flat_cfg import tita_flat_env_cfg, tita_flat_play_env_cfg
from .tita_rough_cfg import tita_rough_env_cfg, tita_rough_play_env_cfg
from ...rl.rl_cfg import tita_np3o_runner_cfg

# ---------------------------------------------------------------------------
# Flat terrain
# ---------------------------------------------------------------------------

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-TITA",
    env_cfg=tita_flat_env_cfg(play=False),
    play_env_cfg=tita_flat_play_env_cfg(),
    rl_cfg=tita_np3o_runner_cfg(experiment_name="tita_flat", max_iterations=15000),
    runner_cls=MjlabNP3ORunner,
)

# ---------------------------------------------------------------------------
# Rough terrain
# ---------------------------------------------------------------------------

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-TITA",
    env_cfg=tita_rough_env_cfg(play=False),
    play_env_cfg=tita_rough_play_env_cfg(),
    rl_cfg=tita_np3o_runner_cfg(experiment_name="tita_rough", max_iterations=15000),
    runner_cls=MjlabNP3ORunner,
)
