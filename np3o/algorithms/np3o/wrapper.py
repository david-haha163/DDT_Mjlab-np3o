# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mjlab-native NP3O environment wrapper.

Bridges mjlab's ``ManagerBasedRlEnv`` to the NP3O algorithm's expected
``VecEnv``-style API:

* ``get_observations()`` → ``(B, history_len, n_proprio)``  (policy ObsGroup)
* ``get_privileged_observations()`` → ``(B, n_critic)``  (critic + priv + scan)
* ``step(actions)`` → ``(policy_obs, critic_obs, rewards, costs, dones, infos)``

Env requirements:

* ObservationsCfg.policy: ``history_length=N``, ``flatten_history_dim=False`` so
  the policy obs has shape ``(B, N, D)``.
* ObservationsCfg.critic.base_lin_vel must be the first term (3 dims) — used as
  the BT vel supervision target.
"""

from __future__ import annotations

import torch
from types import SimpleNamespace

from .cost_manager import CostManager, CostTermCfg


class MjlabNP3OWrapper:
    """Bridges mjlab's ManagerBasedRlEnv to the reference NP3O VecEnv API."""

    def __init__(
        self,
        env,
        cost_terms: dict[str, CostTermCfg] | None = None,
        num_costs: int = 1,
        device: str | None = None,
        use_action_filter: bool = False,
        action_filter_alpha: float = 0.8,
    ):
        self.env = env
        self.unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        self.device = device or str(self.unwrapped.device)

        # ---- discover dims from the env's observation manager
        obs_dict = self.unwrapped.observation_manager.compute()
        if "policy" not in obs_dict:
            raise KeyError("policy ObsGroup is required")
        if "critic" not in obs_dict:
            raise KeyError("critic ObsGroup is required for NP3O privileged obs")

        policy_obs = obs_dict["policy"]
        critic_obs = obs_dict["critic"]

        if policy_obs.dim() != 3:
            raise ValueError(
                f"policy obs must be 3D (B, T, D); got {tuple(policy_obs.shape)}. "
                "Set ObservationsCfg.policy.history_length=N + flatten_history_dim=False."
            )
        if critic_obs.dim() != 2:
            raise ValueError(f"critic obs must be 2D (B, D); got shape {tuple(critic_obs.shape)}")

        self.history_len = int(policy_obs.shape[1])
        self.n_proprio = int(policy_obs.shape[-1])

        # Each segment is its own ObsGroup — dims are inferred from shape.
        priv_obs = obs_dict.get("priv")
        scan_obs = obs_dict.get("scanner")

        self.n_priv_latent = int(priv_obs.shape[-1]) if priv_obs is not None and priv_obs.shape[-1] > 0 else 0
        self.n_scan = int(scan_obs.shape[-1]) if scan_obs is not None and scan_obs.shape[-1] > 0 else 0
        # Full critic dim = shared prop + priv + scan.
        self.n_critic = int(critic_obs.shape[-1]) + self.n_priv_latent + self.n_scan

        self.num_envs = int(self.unwrapped.num_envs)
        self.num_actions = int(self.unwrapped.action_manager.total_action_dim)
        self.max_episode_length = float(self.unwrapped.max_episode_length)

        # ---- cost setup ------------------------------------------------------
        if cost_terms is not None:
            self.cost_manager = CostManager(cost_terms, self.unwrapped, self.device)
            self.num_costs = self.cost_manager.num_costs
            self.cost_k_values = self.cost_manager.k_values
            self.cost_d_values_tensor = self.cost_manager.d_values_tensor
        else:
            self.cost_manager = None
            self.num_costs = max(num_costs, 1)
            self.cost_k_values = torch.zeros(1, self.num_costs, device=self.device)
            self.cost_d_values_tensor = torch.zeros(1, 1, self.num_costs, device=self.device)

        self.use_action_filter = use_action_filter
        self.action_filter_alpha = action_filter_alpha
        self.last_actions: torch.Tensor | None = None
        self.policy_obs_shape = (self.history_len, self.n_proprio)
        self.critic_obs_shape = (self.n_critic,)

        # cfg.env namespace (mirrors what actor_critic / runner expect).
        # Include viewer config from the raw env so mjlab's viewer can use it.
        viewer_cfg = getattr(env.cfg, "viewer", None)
        self.cfg = SimpleNamespace(
            env=SimpleNamespace(
                n_proprio=self.n_proprio,
                n_critic=self.n_critic,
                n_priv_latent=self.n_priv_latent,
                n_scan=self.n_scan,
                history_len=self.history_len,
            ),
            viewer=viewer_cfg,
        )

    # ------------------------------------------------------------ properties
    @property
    def episode_length_buf(self):
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.unwrapped.episode_length_buf = value

    # ----------------------------------------------------------------- API

    def _concat_critic_obs(self, obs_dict) -> torch.Tensor:
        """Concatenate critic + priv + scanner ObsGroups into one flat tensor."""
        parts = [obs_dict["critic"].to(self.device)]
        if self.n_priv_latent > 0 and "priv" in obs_dict:
            parts.append(obs_dict["priv"].to(self.device))
        if self.n_scan > 0 and "scanner" in obs_dict:
            parts.append(obs_dict["scanner"].to(self.device))
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def get_observations(self) -> torch.Tensor:
        return self.unwrapped.observation_manager.compute()["policy"].to(self.device)

    def get_privileged_observations(self) -> torch.Tensor:
        return self._concat_critic_obs(self.unwrapped.observation_manager.compute())

    def reset(self, env_ids=None):
        obs_dict, _ = self.env.reset()
        return obs_dict["policy"].to(self.device)

    @staticmethod
    def _finite_env_mask(x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        return torch.isfinite(flat).all(dim=1)

    @staticmethod
    def _safe_tensor(x: torch.Tensor, limit: float = 100.0) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=limit, neginf=-limit).clamp(-limit, limit)

    def _try_reset_bad_envs(self, bad_envs: torch.Tensor):
        bad_ids = bad_envs.nonzero(as_tuple=False).squeeze(-1)
        if bad_ids.numel() == 0:
            return None
        try:
            return self.env.reset(env_ids=bad_ids)
        except TypeError:
            try:
                return self.env.reset(bad_ids)
            except Exception:
                return None
        except Exception:
            return None

    def step(self, actions: torch.Tensor):
        # ---- Low-pass action filter (EMA, mirrors LocomotionWithNP3O) ----
        if self.use_action_filter:
            if self.last_actions is None:
                self.last_actions = actions.clone()
            else:
                actions = self.last_actions * (1.0 - self.action_filter_alpha) + actions * self.action_filter_alpha
            self.last_actions = actions.clone()

        obs_dict, rewards, terminated, truncated, extras = self.env.step(actions.to(self.unwrapped.device))

        policy_obs = obs_dict["policy"].to(self.device)
        critic_obs = self._concat_critic_obs(obs_dict)
        rewards = rewards.to(self.device)
        dones = (terminated | truncated).to(self.device)

        costs = self._compute_costs(obs_dict)

        # Finite-value guard. This does not replace fixing the underlying
        # terrain/contact issue, but prevents one bad env from corrupting the
        # normalizer, actor std, and optimizer state.
        finite_policy = self._finite_env_mask(policy_obs)
        finite_critic = self._finite_env_mask(critic_obs)
        finite_rewards = torch.isfinite(rewards).reshape(rewards.shape[0], -1).all(dim=1)
        finite_costs = self._finite_env_mask(costs)
        bad_envs = ~(finite_policy & finite_critic & finite_rewards & finite_costs)

        if bad_envs.any():
            # Try to reset bad envs if mjlab exposes env_ids reset. If not, at
            # least sanitize tensors and mark these rollouts as done.
            reset_out = self._try_reset_bad_envs(bad_envs)
            if reset_out is not None:
                try:
                    reset_obs_dict = reset_out[0]
                    if isinstance(reset_obs_dict, dict) and "policy" in reset_obs_dict:
                        reset_policy = reset_obs_dict["policy"].to(self.device)
                        if reset_policy.shape[0] == self.num_envs:
                            policy_obs = reset_policy
                            critic_obs = self._concat_critic_obs(reset_obs_dict)
                        elif reset_policy.shape[0] == int(bad_envs.sum()):
                            policy_obs[bad_envs] = reset_policy
                            critic_obs[bad_envs] = self._concat_critic_obs(reset_obs_dict)
                except Exception:
                    pass
            policy_obs = self._safe_tensor(policy_obs, limit=100.0)
            critic_obs = self._safe_tensor(critic_obs, limit=100.0)
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
            costs = torch.nan_to_num(costs, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
            dones = dones | bad_envs

        # Log per-episode cost averages when episodes end.
        if self.cost_manager is not None:
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                cost_log = self.cost_manager.log_episode(done_ids, self.unwrapped.max_episode_length_s)
                log_dict = extras.setdefault("log", {})
                log_dict.update(cost_log)

        infos = dict(extras)
        infos["time_outs"] = truncated.to(self.device)

        return policy_obs, critic_obs, rewards, costs, dones, infos

    def _compute_costs(self, obs_dict) -> torch.Tensor:
        if self.cost_manager is not None:
            return self.cost_manager.compute().to(self.device)
        return torch.zeros(self.num_envs, self.num_costs, device=self.device)
