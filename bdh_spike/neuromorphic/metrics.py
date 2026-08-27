"""Neuromorphic energy metrics: SOPs vs FLOPs and spike-sparsity tracking.

Implements the project's mathematical reference card::

    SOPs = Σ_t nnz(S_in[t]) × FanOut

A Synaptic Operation (SOP) is one accumulate (AC) event triggered by a single
pre-synaptic spike — the dominant cost on neuromorphic hardware. The dense
equivalent on a von Neumann GPU/CPU is a full MAC sweep every time-step
(``FanIn × FanOut × T × B`` MACs ≈ ``2×`` that in FLOPs). Sparsity is reported
as the fraction of silent neuron-states, target ≥ 85–95 %.
"""

from __future__ import annotations

import torch

__all__ = [
    "SOPSMeter",
    "SpikeSparsityTracker",
    "calculate_sparsity",
    "flops_dense",
    "sops_count",
]


def calculate_sparsity(spikes: torch.Tensor) -> float:
    """Fraction of silent states: ``1 − mean(S)`` for any binary spike tensor.

    Args:
        spikes: Binary-valued tensor (any shape), values in ``{0, 1}``.

    Returns:
        Sparsity in ``[0, 1]`` — the higher, the more computationally sparse.
    """
    if spikes.numel() == 0:
        raise ValueError("cannot compute sparsity of an empty tensor")
    return 1.0 - float(spikes.detach().float().mean())


def sops_count(spike_train: torch.Tensor, fan_out: int) -> int:
    """Synaptic Operations triggered by a spike train (reference-card formula).

    Args:
        spike_train: Binary input train of shape ``[T, ..., fan_in]`` with the
            channel dimension last.
        fan_out: Fan-out of the synapse layer consuming these spikes.

    Returns:
        ``Σ_t nnz(S_in[t]) × FanOut`` — total AC events.
    """
    if spike_train.dim() < 2:
        raise ValueError("spike_train must have shape [T, ..., fan_in]")
    active = int(torch.count_nonzero(spike_train.detach()))
    return active * int(fan_out)


def flops_dense(fan_in: int, fan_out: int, steps: int = 1, batch: int = 1) -> int:
    """FLOPs of the equivalent dense (non-spiking) linear sweep per inference.

    Dense hardware multiplies every weight against every activation each
    step: ``T·B·FanIn·FanOut`` MACs ≈ ``2·T·B·FanIn·FanOut`` FLOPs.
    """
    macs = int(steps) * int(batch) * int(fan_in) * int(fan_out)
    return 2 * macs


@torch.no_grad()
def _mean_rate(spikes: torch.Tensor) -> float:
    return float(spikes.detach().float().mean())


class SpikeSparsityTracker:
    """Running per-update sparsity statistics across forward calls.

    Accumulates EMA-smoothed sparsity so long benchmark runs can log a stable
    sparsity signal without storing whole spike trains.

    Args:
        momentum: EMA momentum for the running estimate (higher = smoother).
    """

    def __init__(self, momentum: float = 0.9) -> None:
        self.momentum = float(momentum)
        self._running: float | None = None
        self.updates = 0
        self.min_seen = 1.0
        self.max_seen = 0.0

    def update(self, spikes: torch.Tensor) -> float:
        """Feed one binary spike tensor; returns its instantaneous sparsity."""
        s = calculate_sparsity(spikes)
        self.updates += 1
        self.min_seen = min(self.min_seen, s)
        self.max_seen = max(self.max_seen, s)
        if self._running is None or self.momentum <= 0.0:
            self._running = s
        else:
            self._running = self.momentum * self._running + (1.0 - self.momentum) * s
        return s

    @property
    def running(self) -> float:
        """EMA-smoothed sparsity estimate (0.0 before the first update)."""
        return 0.0 if self._running is None else self._running

    def report(self) -> dict[str, float]:
        """Summary dictionary suitable for logging."""
        return {
            "sparsity_ema": self.running,
            "sparsity_min": self.min_seen,
            "sparsity_max": self.max_seen,
            "updates": float(self.updates),
        }


class SOPSMeter:
    """Accumulates synaptic operations and compares them to dense FLOPs.

    Usage: register each synaptic layer once (fan-in/fan-out known up front),
    then feed its observed input spike trains during evaluation.

    Args:
        dense_flops_per_mac: FLOPs charged per multiply-accumulate (default 2).
    """

    def __init__(self, dense_flops_per_mac: int = 2) -> None:
        self.dense_flops_per_mac = int(dense_flops_per_mac)
        self.total_sops = 0
        self.total_dense_macs = 0

    def register_layer(self, fan_in: int, fan_out: int, steps: int = 1, batch: int = 1) -> None:
        """Book-keep the dense-equivalent cost of one synaptic layer."""
        self.total_dense_macs += int(steps) * int(batch) * int(fan_in) * int(fan_out)

    def update(self, spike_train: torch.Tensor, fan_out: int) -> int:
        """Accumulate SOPs from one observed input spike train."""
        sops = sops_count(spike_train, fan_out)
        self.total_sops += sops
        return sops

    def report(self) -> dict[str, float]:
        """SOPs vs dense FLOPs summary with the achieved speed-up ratio."""
        dense_flops = self.dense_flops_per_mac * self.total_dense_macs
        ratio = (dense_flops / self.total_sops) if self.total_sops > 0 else float("inf")
        return {
            "sops": float(self.total_sops),
            "dense_macs": float(self.total_dense_macs),
            "dense_flops": float(dense_flops),
            "flops_per_sop_ratio": ratio,
        }
