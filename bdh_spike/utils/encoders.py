"""Spike encoders: raw continuous signals → temporal spike trains (Stage 5).

Converts real-world values into the discrete event domain required by every
core module. All encoders emit **binary-valued float tensors** ``S(t) ∈ {0,1}``
in the canonical temporal layout ``[T, *input_shape]`` (time-steps first).

* :func:`rate_encode` — stochastic Bernoulli firing, probability ∝ intensity;
  highest information fidelity, naturally sparse for weak inputs.
* :func:`latency_encode` — time-to-first-spike coding; strongest value fires
  earliest, zero intensity never fires; maximally sparse (≤ 1 spike/unit).
* :func:`delta_encode` — event-driven ON/OFF coding of signal *changes*
  (event-camera / DVS semantics); silent under static conditions.
"""

from __future__ import annotations

import torch

__all__ = ["delta_encode", "latency_encode", "rate_encode"]


def _check_finite(x: torch.Tensor, name: str) -> None:
    if not torch.isfinite(x.detach()).all():
        raise ValueError(f"{name} contains non-finite values")


def rate_encode(x: torch.Tensor, num_steps: int, gain: float = 1.0) -> torch.Tensor:
    """Rate encoding: Poisson-style Bernoulli spike train.

    Args:
        x: Non-negative intensities, any shape (typically ``[B, C]``).
        num_steps: Number of temporal steps ``T`` of the produced train.
        gain: Multiplicative scaling of intensities into ``[0, 1]`` firing
            probabilities before clamping.

    Returns:
        Binary spikes ``[T, *x.shape]``, expected firing rate ``≈ gain·x``.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be ≥ 1, got {num_steps}")
    _check_finite(x, "rate_encode input")
    if float(x.detach().min()) < 0.0:
        raise ValueError("rate_encode requires non-negative intensities")

    prob = (x * gain).clamp_(0.0, 1.0)
    noise = torch.rand(num_steps, *x.shape, device=x.device, dtype=x.dtype)
    return (noise < prob).to(dtype=torch.float)


def latency_encode(
    x: torch.Tensor, num_steps: int, normalize: bool = True, eps: float = 1e-8
) -> torch.Tensor:
    """Latency encoding: one spike per unit, earlier for stronger intensity.

    Spike time ``t_s = ⌊(1 − level)·T⌋`` where ``level = x / max(x)`` — the
    maximum fires at ``t = 0``, vanishing values never fire.

    Args:
        x: Non-negative intensities, any shape.
        num_steps: Temporal window ``T``.
        normalize: Scale by the global tensor maximum.
        eps: Numerical guard for the normalization denominator.

    Returns:
        Binary spikes ``[T, *x.shape]`` with at most one spike per element.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be ≥ 1, got {num_steps}")
    _check_finite(x, "latency_encode input")
    if float(x.detach().min()) < 0.0:
        raise ValueError("latency_encode requires non-negative intensities")

    level = x
    if normalize:
        level = x / (x.max() + eps)
    level = level.clamp(0.0, 1.0)

    first_spike = ((1.0 - level) * num_steps).long().clamp(max=num_steps - 1)
    steps = torch.arange(num_steps, device=x.device).view(num_steps, *([1] * x.dim()))
    fired = level > 0.0
    return ((steps == first_spike.unsqueeze(0)) & fired).to(dtype=torch.float)


def delta_encode(x: torch.Tensor, threshold: float = 0.1, polarity: str = "on") -> torch.Tensor:
    """Delta encoding: spike only when the signal *changes* (DVS semantics).

    Args:
        x: Signal trajectory ``[T, ...]`` in canonical temporal layout.
        threshold: Minimal absolute change that triggers an event.
        polarity: ``"on"`` spikes on increases, ``"off"`` on decreases.

    Returns:
        Binary events ``[T, *rest]`` (same rank as ``x``; ``T`` preserved by
        padding a silent step at the start).
    """
    if polarity not in {"on", "off"}:
        raise ValueError(f"polarity must be 'on' or 'off', got {polarity!r}")
    if x.dim() < 2 or x.shape[0] < 2:
        raise ValueError("delta_encode expects a temporal trajectory [T≥2, ...]")
    _check_finite(x, "delta_encode input")

    diff = x[1:] - x[:-1]
    events = (diff > threshold).float() if polarity == "on" else (-diff > threshold).float()
    lead = torch.zeros_like(events[:1])
    return torch.cat([lead, events], dim=0)
