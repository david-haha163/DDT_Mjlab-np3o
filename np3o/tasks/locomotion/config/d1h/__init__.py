# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""D1H velocity-tracking task registration for mjlab."""

from mjlab.tasks.registry import register_mjlab_task

from np3o.algorithms.np3o.runner import MjlabNP3ORunner

from .d1h_flat_cfg import d1h_flat_env_cfg, d1h_flat_play_env_cfg
from .d1h_rough_cfg import d1h_rough_env_cfg, d1h_rough_play_env_cfg
from ...rl.rl_cfg import d1h_np3o_runner_cfg

# ---------------------------------------------------------------------------
# Flat terrain
# ---------------------------------------------------------------------------

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-D1H",
    env_cfg=d1h_flat_env_cfg(play=False),
    play_env_cfg=d1h_flat_play_env_cfg(),
    rl_cfg=d1h_np3o_runner_cfg(experiment_name="d1h_flat", max_iterations=15000),
    runner_cls=MjlabNP3ORunner,
)

# ---------------------------------------------------------------------------
# Rough terrain
# ---------------------------------------------------------------------------

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-D1H",
    env_cfg=d1h_rough_env_cfg(play=False),
    play_env_cfg=d1h_rough_play_env_cfg(),
    rl_cfg=d1h_np3o_runner_cfg(experiment_name="d1h_rough", max_iterations=15000),
    runner_cls=MjlabNP3ORunner,
)
