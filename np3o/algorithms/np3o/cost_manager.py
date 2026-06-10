# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cost manager for NP3O constrained training, adapted for mjlab.

Mjlab-adapted counterpart of the IsaacLab ``CostManager``. Uses mjlab's
``SceneEntityCfg`` and ``Entity`` data API instead of IsaacLab's
``Articulation`` / ``ManagerTermBaseCfg``.

Episode cost logging follows the same convention as mjlab's ``RewardManager``:
accumulate ``val * env.step_dt`` and report ``sum / max_episode_length_s``.
This keeps costs numerically comparable to rewards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

import torch
from prettytable import PrettyTable

from mjlab.managers.scene_entity_config import SceneEntityCfg


@dataclass
class CostTermCfg:
    """Configuration for a single cost term.

    Mirrors the IsaacLab ``CostTermCfg`` but as a plain dataclass (mjlab does
    not use ``@configclass``).
    """

    func: Callable
    """Cost function: ``func(env, **params) -> Tensor (num_envs,)``."""

    scale: float = 1.0
    """Multiplied with the raw cost value before storage / loss computation."""

    d_value: float = 0.0
    """Per-term safety budget. ``cost_violation = (1-γ)*(cost_returns - d_value) + ...``"""

    k_value: float = 0.01
    """Initial Lagrangian multiplier for this term. Updated by ``NP3O.update_k_value``."""

    params: dict[str, Any] | None = None
    """Extra keyword arguments passed to ``func``. SceneEntityCfg values are
    auto-resolved before the first ``compute()`` call."""


class CostManager:
    """Computes cost vectors per step. Each cost term is a ``CostTermCfg``.

    Attributes read by the NP3O wrapper / runner:

    * ``num_costs``        — int
    * ``active_terms``     — list[str]
    * ``k_values``         — ``(1, num_costs)``
    * ``d_values_tensor``  — ``(1, 1, num_costs)`` (broadcasts over T, num_envs)

    Episode logging mirrors ``RewardManager``: per-step values are multiplied
    by ``env.step_dt`` before accumulation, and the episode log divides by
    ``max_episode_length_s``.  This keeps costs in the same 0.x range as
    rewards and matches the IsaacGym reference scale.
    """

    def __init__(self, cost_terms: dict[str, CostTermCfg], env, device: str):
        self._env = env
        self._device = device

        self._terms: list[tuple[str, CostTermCfg]] = []
        for name, term in cost_terms.items():
            # Resolve SceneEntityCfg params (same pattern as RewardManager).
            params = dict(term.params) if term.params else {}
            for value in params.values():
                if isinstance(value, SceneEntityCfg):
                    value.resolve(env.scene)
            # Create a resolved copy so the original dataclass is not mutated.
            resolved = CostTermCfg(
                func=term.func,
                scale=term.scale,
                d_value=term.d_value,
                k_value=term.k_value,
                params=params,
            )
            self._terms.append((name, resolved))

        if not self._terms:
            raise ValueError(
                "CostManager requires at least one cost term; "
                "pass cost_terms=None to the wrapper to run without costs."
            )

        self.num_costs: int = len(self._terms)
        self.active_terms: list[str] = [n for n, _ in self._terms]

        k_init = [term.k_value for _, term in self._terms]
        d_init = [term.d_value for _, term in self._terms]
        self.k_values = torch.tensor(k_init, device=device).view(1, -1)
        self.d_values_tensor = torch.tensor(d_init, device=device).view(1, 1, -1)

        # per-episode accumulators (one scalar per env, per term)
        self._episode_sums: dict[str, torch.Tensor] = {
            name: torch.zeros(env.num_envs, device=device) for name in self.active_terms
        }

        print("[INFO] Cost Manager: ", self)

    def __str__(self) -> str:
        msg = f"<CostManager> contains {self.num_costs} active terms.\n"
        table = PrettyTable()
        table.title = "Active Cost Terms"
        table.field_names = ["Index", "Name", "Scale", "d_value", "k_value"]
        table.align["Name"] = "l"
        for align_col in ("Scale", "d_value", "k_value"):
            table.align[align_col] = "r"
        for idx, (name, term) in enumerate(self._terms):
            table.add_row([idx, name, f"{term.scale:.4g}", f"{term.d_value:.4g}", f"{term.k_value:.4g}"])
        msg += table.get_string()
        msg += "\n"
        return msg

    @torch.no_grad()
    def compute(self) -> torch.Tensor:
        """Return ``(num_envs, num_costs)`` per-step costs (clamped to >=0).

        Side-effect: accumulates ``val * env.step_dt`` into
        ``self._episode_sums``, matching the RewardManager convention.
        """
        dt = self._env.step_dt
        cols = []
        for name, term in self._terms:
            val = term.func(self._env, **term.params) * term.scale * dt
            # Do not allow a single bad simulator state to poison NP3O storage.
            val = torch.nan_to_num(val, nan=0.0, posinf=1.0, neginf=0.0).clamp_min_(0.0)
            self._episode_sums[name] += val
            self._episode_sums[name] = torch.nan_to_num(self._episode_sums[name], nan=0.0, posinf=0.0, neginf=0.0)
            cols.append(val.unsqueeze(-1))
        return torch.cat(cols, dim=-1)

    @torch.no_grad()
    def log_episode(self, env_ids: torch.Tensor, max_episode_length_s: float) -> dict[str, torch.Tensor]:
        """Pop and return per-term mean cost over the episodes that just ended.

        Reports ``sum / max_episode_length_s``, matching the RewardManager
        convention so costs and rewards share the same numerical scale.
        """
        log: dict[str, torch.Tensor] = {}
        for name in self.active_terms:
            sums = self._episode_sums[name]
            log[f"Episode_Cost/{name}"] = torch.nan_to_num(
                sums[env_ids].mean() / max_episode_length_s, nan=0.0, posinf=0.0, neginf=0.0
            )
            sums[env_ids] = 0.0
        return log
