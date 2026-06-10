# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""EmpiricalNormalization (verbatim port from LocomotionWithNP3O)."""

import torch
import torch.nn as nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-2, until=None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.count = 0

    @torch.jit.unused
    def _ensure_mutable_buffers(self):
        """Convert buffers created/updated under inference_mode back to normal tensors.

        Some rollout/evaluation paths call the normalizer while PyTorch inference
        mode is enabled. If a buffer is assigned there, it becomes an inference
        tensor, and a later in-place running-stat update outside inference mode
        raises: "Inplace update to inference tensor outside InferenceMode".
        """
        # ``torch.inference_mode(False)`` is needed if this method itself is
        # called from an inference-mode context; otherwise clone() would also
        # create inference tensors.
        with torch.inference_mode(False):
            for name in ("_mean", "_var", "_std"):
                buf = getattr(self, name)
                if hasattr(buf, "is_inference") and buf.is_inference():
                    setattr(self, name, buf.clone().detach())

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        # A single NaN/Inf observation can permanently poison the running
        # statistics and then all network outputs. Sanitize before updating and
        # before returning normalized values.
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        if self.training:
            self.update(x)
        y = (x - self._mean) / (self._std + self.eps)
        return torch.nan_to_num(y, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)

    @torch.jit.unused
    def update(self, x):
        if self.until is not None and self.count >= self.until:
            return
        count_x = x.shape[0]
        if count_x == 0:
            return

        # Running statistics are buffers, not trainable parameters. Keep the
        # whole update out of autograd and avoid assigning inference tensors to
        # buffers. Use copy_ into existing buffers after making them mutable.
        with torch.no_grad():
            self._ensure_mutable_buffers()

            x = x.detach()
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
            var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
            mean_x = torch.mean(x, dim=0, keepdim=True)
            var_x = torch.nan_to_num(var_x, nan=1.0, posinf=1.0e6, neginf=1.0).clamp_min(0.0)
            mean_x = torch.nan_to_num(mean_x, nan=0.0, posinf=1.0e6, neginf=-1.0e6)

            new_count = self.count + count_x
            rate = count_x / new_count

            delta_mean = mean_x - self._mean
            new_mean = self._mean + rate * delta_mean
            new_var = self._var + rate * (var_x - self._var + delta_mean * (mean_x - new_mean))

            new_mean = torch.nan_to_num(new_mean, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
            new_var = torch.nan_to_num(new_var, nan=1.0, posinf=1.0e6, neginf=1.0).clamp_min(1.0e-6)
            new_std = torch.sqrt(new_var)

            self._mean.copy_(new_mean)
            self._var.copy_(new_var)
            self._std.copy_(new_std)
            self.count = new_count

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean
