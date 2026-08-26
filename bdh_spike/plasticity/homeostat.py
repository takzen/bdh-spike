"""Adaptive membrane-threshold regulation (Stage 4 homeostat).

Keeps network activity in a healthy band by adapting the firing threshold
``V_th`` online, purely locally (no gradients, no global error signal):

    r[t]     = β·r[t-1] + (1-β)·mean_batch(S[t])         (firing-rate EMA)
    V_th[t]  = clip(V_th[t-1] + η·(r[t] − r_target), V_min, V_max)

Hyperactivity ("seizure-like" runaway discharge, r > r_target) raises the
threshold; silence (sleep, r < r_target) lowers it back toward excitability.

Two operating modes:

* per-channel — ``V_th ∈ Rᶜ`` adapts each neuron independently;
* scalar — ``V_th ∈ R`` follows the population rate; designed for direct
  injection into a :class:`~bdh_spike.core.neuron.BDHSpikeCell` ``v_th``
  buffer via :meth:`apply_to`.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["AdaptiveThreshold"]


class AdaptiveThreshold(nn.Module):
    """Gradient-free homeostatic controller for the spiking threshold.

    Args:
        num_channels: Channel count ``C`` for per-channel thresholds, or
            ``None`` for a single scalar threshold shared across the layer.
        target_rate: Target mean firing rate ``r_target`` in ``[0, 1]``.
        eta: Adaptation strength ``η`` (threshold units per unit rate-error).
        rate_beta: EMA smoothing factor ``β`` of the measured firing rate.
        v_th_init: Initial threshold ``V_th``.
        v_min: Lower clamp — prevents threshold collapse into tonic bursting.
        v_max: Upper clamp — prevents permanent quiescence.
    """

    v_th: torch.Tensor
    rate: torch.Tensor

    def __init__(
        self,
        num_channels: int | None = None,
        target_rate: float = 0.10,
        eta: float = 0.05,
        rate_beta: float = 0.9,
        v_th_init: float = 1.0,
        v_min: float = 0.05,
        v_max: float = 5.0,
    ) -> None:
        super().__init__()
        if not 0.0 < v_min < v_max:
            raise ValueError(f"Require 0 < v_min < v_max, got ({v_min}, {v_max})")
        self.num_channels = None if num_channels is None else int(num_channels)
        self.target_rate = float(target_rate)
        self.eta = float(eta)
        self.rate_beta = float(rate_beta)
        self.v_th_init = float(v_th_init)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

        shape = () if num_channels is None else (int(num_channels),)
        # Buffers: adaptable online by design, invisible to autograd.
        self.register_buffer("v_th", torch.full(shape, float(v_th_init)))
        self.register_buffer("rate", torch.zeros(shape))

    def reset(self) -> None:
        """Restore initial threshold and clear the rate estimate."""
        with torch.no_grad():
            self.v_th.fill_(self.v_th_init)
            self.rate.zero_()

    @property
    def is_scalar(self) -> bool:
        """Whether a single population-wide threshold is maintained."""
        return self.num_channels is None

    @torch.no_grad()
    def step(self, spikes: torch.Tensor) -> torch.Tensor:
        """Adapt ``V_th`` from one observation of spikes.

        Args:
            spikes: Binary spikes ``[B, C]`` (per-channel mode) or ``[B, C]``
                collapsed to population rate (scalar mode); ``[C]`` also OK.

        Returns:
            The updated threshold buffer ``self.v_th`` (read-only view).
        """
        if spikes.dim() not in (1, 2):
            raise ValueError(f"Expected spikes of shape [B, C] or [C], got {tuple(spikes.shape)}")
        if not self.is_scalar and spikes.shape[-1] != self.num_channels:
            raise ValueError(f"Expected C={self.num_channels}, got {spikes.shape[-1]}")

        s = spikes.to(dtype=self.rate.dtype)
        batch_rate = s.mean() if self.is_scalar else s.mean(dim=0)

        # Firing-rate EMA, then threshold nudge proportional to rate error.
        self.rate.mul_(self.rate_beta).add_(batch_rate, alpha=1.0 - self.rate_beta)
        self.v_th.add_(self.eta * (self.rate - self.target_rate))
        self.v_th.clamp_(self.v_min, self.v_max)
        return self.v_th

    @torch.no_grad()
    def observe_sequence(self, spike_seq: torch.Tensor) -> torch.Tensor:
        """Run :meth:`step` over a canonical ``[T, B, C]`` spike train."""
        if spike_seq.dim() != 3:
            raise ValueError(
                f"Expected spike train of shape [T, B, C], got {tuple(spike_seq.shape)}"
            )
        for t in range(spike_seq.shape[0]):
            self.step(spike_seq[t])
        return self.v_th

    @torch.no_grad()
    def apply_to(self, cell: nn.Module) -> None:
        """Copy the scalar threshold into a ``BDHSpikeCell`` ``v_th`` buffer.

        Only valid in scalar mode — the PLIF cell carries a single shared
        threshold buffer by design.
        """
        if not self.is_scalar:
            raise TypeError(
                "apply_to requires a scalar-mode AdaptiveThreshold "
                "(construct with num_channels=None)"
            )
        if not hasattr(cell, "v_th") or not isinstance(cell.v_th, torch.Tensor):
            raise TypeError("target module must expose a tensor buffer named 'v_th'")
        cell.v_th.copy_(self.v_th)

    def extra_repr(self) -> str:
        mode = "scalar" if self.is_scalar else f"C={self.num_channels}"
        return (
            f"{mode}, target_rate={self.target_rate}, eta={self.eta}, "
            f"rate_beta={self.rate_beta}, v_th=[{self.v_min}, {self.v_max}]"
        )
