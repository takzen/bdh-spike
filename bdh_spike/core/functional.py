"""Surrogate gradients and math primitives for the BDH-Spike core.

The forward pass is fully binary (spikes in {0, 1}); all differentiability on
regular GPUs is achieved exclusively through surrogate gradients applied to the
spiking threshold crossing.
"""

from __future__ import annotations

import torch


class FastSigmoidSurrogate(torch.autograd.Function):
    """Heaviside spike trigger with a fast-sigmoid backward pass.

    Forward:  S(x) = Θ(x - V_th)                ∈ {0, 1}
    Backward: σ'(x) = 1 / (1 + k·|x - V_th|)^2   (fast sigmoid surrogate)

    The slope ``k`` controls how wide the gradient window around the threshold
    is. Default ``k = 25`` matches the project's mathematical reference card.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, x: torch.Tensor, threshold: torch.Tensor, slope: float
    ) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        ctx.slope = slope
        return (x >= threshold).to(dtype=x.dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, None, None]:
        (x,) = ctx.saved_tensors
        k = ctx.slope
        surrogate = 1.0 / (1.0 + k * (x - ctx.threshold).abs()).pow(2)
        return grad_output * surrogate, None, None


def spike_fn(
    x: torch.Tensor, threshold: torch.Tensor | float = 1.0, slope: float = 25.0
) -> torch.Tensor:
    """Emit binary spikes with a fast-sigmoid surrogate gradient.

    Args:
        x: Continuous membrane potential (or any pre-threshold signal).
        threshold: Firing threshold ``V_th`` (tensor or scalar).
        slope: Surrogate slope ``k`` (default 25).

    Returns:
        Binary spikes ``S ∈ {0, 1}`` of the same shape/dtype/device as ``x``.
    """
    if not isinstance(threshold, torch.Tensor):
        threshold = torch.tensor(threshold, dtype=x.dtype, device=x.device)
    return FastSigmoidSurrogate.apply(x, threshold, slope)
