# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared NP3O runner config for locomotion tasks (D1 + TITA)."""

from __future__ import annotations

from copy import deepcopy

_BASE_NP3O_RUNNER_CFG = {
    "runner": {
        "policy_class_name": "ActorCriticBarlowTwins",
        "algorithm_class_name": "NP3O",
        "runner_class_name": "MjlabNP3ORunner",
        "num_steps_per_env": 24,
        "max_iterations": 5000,
        "save_interval": 100,
        "experiment_name": "d1",
        "run_name": "",
        "resume": False,
        "load_run": ".*",
        "load_checkpoint": r"model_.*\.pt",
    },
    "algorithm": {
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.01,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 0.01,
        "cost_value_loss_coef": 0.1,
        "cost_viol_loss_coef": 0.1,
    },
    "policy": {
        "init_noise_std": 1.0,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "priv_encoder_dims": [],
        "scan_encoder_dims": [128, 64, 32],
        "activation": "elu",
        "imi_flag": True,
        "bt_window": 5,
        "bt_mlp_encoder_dims": [128, 64],
        "bt_latent_encoder_dims": [32, 16],
        "bt_projector_dims": [64],
    },
}


def d1_np3o_runner_cfg(
    experiment_name: str = "d1",
    max_iterations: int = 15000,
) -> dict:
    """Return a deep-copied NP3O runner config dict.

    Args:
        experiment_name: Name of the experiment (e.g. "d1_flat", "d1_rough").
        max_iterations: Maximum number of training iterations.
    """
    cfg = deepcopy(_BASE_NP3O_RUNNER_CFG)
    cfg["runner"]["experiment_name"] = experiment_name
    cfg["runner"]["max_iterations"] = max_iterations
    return cfg


def d1h_np3o_runner_cfg(
    experiment_name: str = "d1h",
    max_iterations: int = 15000,
) -> dict:
    """Return a deep-copied NP3O runner config dict for D1H.

    Args:
        experiment_name: Name of the experiment (e.g. "d1h_flat", "d1h_rough").
        max_iterations: Maximum number of training iterations.
    """
    cfg = deepcopy(_BASE_NP3O_RUNNER_CFG)
    cfg["runner"]["experiment_name"] = experiment_name
    cfg["runner"]["max_iterations"] = max_iterations
    return cfg


def tita_np3o_runner_cfg(
    experiment_name: str = "tita",
    max_iterations: int = 15000,
) -> dict:
    """Return a deep-copied NP3O runner config dict for TITA.

    Args:
        experiment_name: Name of the experiment (e.g. "tita_flat", "tita_rough").
        max_iterations: Maximum number of training iterations.
    """
    cfg = deepcopy(_BASE_NP3O_RUNNER_CFG)
    cfg["runner"]["experiment_name"] = experiment_name
    cfg["runner"]["max_iterations"] = max_iterations
    return cfg
