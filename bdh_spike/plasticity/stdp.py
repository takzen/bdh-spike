"""Dual-weight plasticity engine: online 3-factor Hebbian STDP (Stage 4).

Implements the project's dual-weight doctrine:

* ``W_slow`` — structural weights, trained offline by BPTT through the
  fast-sigmoid surrogate. A regular ``nn.Parameter`` participating in the
  autograd graph.
* ``W_fast`` — episodic weights, updated **online during inference** by local
  Spike-Timing-Dependent Plasticity under ``torch.no_grad()``. Registered as a
  buffer: never listed in ``module.parameters()`` and unable to leak gradients
  into PyTorch's autograd graph.

Local 3-factor Hebbian rule (synapse i→j, per time-step Δt = 1):

    A_pre[t]  = A_pre[t-1]·e^{-Δt/τ_pre}  + S_pre[t]     (eligibility trace)
    A_post[t] = A_post[t-1]·e^{-Δt/τ_post} + S_post[t]

    ΔW[t] = η₊·M·(S_post[t] ⊗ A_pre[t])                  (LTP: pre before post)
          − η₋·M·(S_pre[t]  ⊗ A_post[t])                 (LTD: post before pre)

``M`` is the third (modulatory) factor gating how much eligible change is
written into ``W_fast`` each step. Synaptic transmission fuses both weight
sets: ``I_syn = (W_slow + W_fast)ᵀ · S_in`` — accumulate-only friendly.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["DualWeightLinear", "STDPState"]


@dataclass
class STDPState:
    """Encapsulated eligibility traces for :class:`DualWeightLinear`.

    Attributes:
        a_pre: Pre-synaptic eligibility trace, shape ``[B, fan_in]``.
        a_post: Post-synaptic eligibility trace, shape ``[B, fan_out]``.
    """

    a_pre: torch.Tensor
    a_post: torch.Tensor


class DualWeightLinear(nn.Module):
    """Dual-weight synaptic layer: slow BPTT weights + fast online-STDP weights.

    Args:
        fan_in: Input feature count ``C_in``.
        fan_out: Output feature count ``C_out``.
        tau_pre: Decay time-constant ``τ_pre`` of the pre-synaptic trace.
        tau_post: Decay time-constant ``τ_post`` of the post-synaptic trace.
        ltp_rate: Potentiation rate ``η₊`` (pre-before-post coincidences).
        ltd_rate: Depression rate ``η₋`` (post-before-pre coincidences).
        bias: If ``True``, add a (slow-only) trainable bias.
        w_fast_limit: Symmetric saturation bound ``|W_fast| ≤ limit``.
    """

    w_fast: torch.Tensor

    def __init__(
        self,
        fan_in: int,
        fan_out: int,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        ltp_rate: float = 0.005,
        ltd_rate: float = 0.005,
        bias: bool = False,
        w_fast_limit: float = 1.0,
    ) -> None:
        super().__init__()
        self.fan_in = int(fan_in)
        self.fan_out = int(fan_out)
        self.tau_pre = float(tau_pre)
        self.tau_post = float(tau_post)
        self.ltp_rate = float(ltp_rate)
        self.ltd_rate = float(ltd_rate)
        self.w_fast_limit = float(w_fast_limit)

        # Slow structural weights: the ONLY autograd participant here.
        self.w_slow = nn.Parameter(torch.empty(fan_out, fan_in))
        nn.init.kaiming_uniform_(self.w_slow, a=math.sqrt(5.0))

        # Fast episodic weights: grad-free buffer, shaped online by STDP.
        self.register_buffer("w_fast", torch.zeros(fan_out, fan_in))

        if bias:
            self.bias = nn.Parameter(torch.zeros(fan_out))
        else:
            self.register_parameter("bias", None)

    @property
    def decay_pre(self) -> float:
        """Per-step pre-trace decay ``e^{-Δt/τ_pre}`` (Δt = 1)."""
        return math.exp(-1.0 / self.tau_pre)

    @property
    def decay_post(self) -> float:
        """Per-step post-trace decay ``e^{-Δt/τ_post}`` (Δt = 1)."""
        return math.exp(-1.0 / self.tau_post)

    def effective_weight(self) -> torch.Tensor:
        """Fused synaptic kernel ``W_slow + W_fast``, shape ``[C_out, C_in]``."""
        return self.w_slow + self.w_fast

    def init_hidden(self, batch_size: int) -> STDPState:
        """Explicitly zero-initialize the eligibility traces for a batch."""
        device = self.w_fast.device
        return STDPState(
            a_pre=torch.zeros(int(batch_size), self.fan_in, device=device),
            a_post=torch.zeros(int(batch_size), self.fan_out, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate binary spikes (or currents) through fused weights.

        Args:
            x: Input of shape ``[B, C_in]`` or canonical ``[T, B, C_in]``.

        Returns:
            Synaptic currents of matching rank with trailing dim ``C_out``.
        """
        if x.dim() not in (2, 3):
            raise ValueError(
                f"Expected input of shape [B, C_in] or [T, B, C_in], got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.fan_in:
            raise ValueError(f"Expected fan_in={self.fan_in}, got {x.shape[-1]}")
        y = x @ self.effective_weight().transpose(0, 1)
        if self.bias is not None:
            y = y + self.bias
        return y

    # ------------------------------------------------------------ #
    # Online plasticity (strictly grad-free)                       #
    # ------------------------------------------------------------ #
    @torch.no_grad()
    def plastic_step(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        state: STDPState,
        modulator: float | torch.Tensor = 1.0,
    ) -> STDPState:
        """One online 3-factor Hebbian/STDP update of ``W_fast``.

        Args:
            pre_spikes: Binary pre-synaptic spikes ``[B, C_in]``.
            post_spikes: Binary post-synaptic spikes ``[B, C_out]``.
            state: Current eligibility traces from ``init_hidden`` / prior step.
            modulator: Third factor ``M`` — scalar or per-sample ``[B]``;
                gates the magnitude of the applied update.

        Returns:
            Updated :class:`STDPState` carrying the decayed+charged traces.
        """
        if pre_spikes.dim() != 2 or pre_spikes.shape[-1] != self.fan_in:
            raise ValueError(
                f"Expected pre_spikes of shape [B, {self.fan_in}], got {tuple(pre_spikes.shape)}"
            )
        if post_spikes.dim() != 2 or post_spikes.shape[-1] != self.fan_out:
            raise ValueError(
                f"Expected post_spikes of shape [B, {self.fan_out}], got {tuple(post_spikes.shape)}"
            )
        if pre_spikes.shape[0] != post_spikes.shape[0]:
            raise ValueError("pre/post batch size mismatch")

        # Exponential trace decay + charge (canonical STDP trace recursion).
        a_pre = state.a_pre * self.decay_pre + pre_spikes
        a_post = state.a_post * self.decay_post + post_spikes

        batch = pre_spikes.shape[0]
        m = torch.as_tensor(modulator, dtype=a_pre.dtype, device=a_pre.device)
        if m.dim() == 1:  # per-sample modulation → broadcast over channels
            m = m.unsqueeze(1)

        # LTP: post firing while pre-trace is hot (causal). LTD: anti-causal.
        ltp = ((m * post_spikes).transpose(0, 1) @ a_pre) / batch
        ltd = (a_post.transpose(0, 1) @ (m * pre_spikes)) / batch
        self.w_fast.add_(self.ltp_rate * ltp - self.ltd_rate * ltd)
        self.w_fast.clamp_(-self.w_fast_limit, self.w_fast_limit)

        return STDPState(a_pre=a_pre, a_post=a_post)

    @torch.no_grad()
    def run_episode(
        self,
        pre_seq: torch.Tensor,
        post_seq: torch.Tensor,
        state: STDPState | None = None,
        modulators: float | torch.Tensor | None = None,
    ) -> STDPState:
        """Stream an entire ``[T, B, ...]`` episode of online STDP updates.

        Args:
            pre_seq: Binary pre-synaptic spike train ``[T, B, C_in]``.
            post_seq: Binary post-synaptic spike train ``[T, B, C_out]``.
            state: Optional initial traces; lazily zero-initialized if omitted.
            modulators: Optional scalar, ``[T]`` or ``[T, B]`` third factor.

        Returns:
            Final :class:`STDPState` after the last time-step.
        """
        if pre_seq.dim() != 3 or post_seq.dim() != 3:
            raise ValueError("run_episode expects [T, B, C] spike trains")
        if pre_seq.shape[0] != post_seq.shape[0]:
            raise ValueError("pre/post temporal length mismatch")

        if state is None:
            state = self.init_hidden(pre_seq.shape[1])
        for updated in self.iterate_plastic_steps(pre_seq, post_seq, state, modulators):
            state = updated
        return state

    def iterate_plastic_steps(
        self,
        pre_seq: torch.Tensor,
        post_seq: torch.Tensor,
        state: STDPState,
        modulators: float | torch.Tensor | None = None,
    ) -> Iterator[STDPState]:
        """Streaming helper: yield the :class:`STDPState` after each step."""
        for t in range(pre_seq.shape[0]):
            # Index only true temporal modulation schedules ([T] or [T, B]);
            # 0-dim / single-element tensors act as constant scalars.
            if (
                isinstance(modulators, torch.Tensor)
                and modulators.dim() > 0
                and modulators.shape[0] == pre_seq.shape[0]
            ):
                mod_t: float | torch.Tensor = modulators[t]
            elif modulators is not None:
                mod_t = modulators
            else:
                mod_t = 1.0
            state = self.plastic_step(pre_seq[t], post_seq[t], state, modulator=mod_t)
            yield state

    def extra_repr(self) -> str:
        return (
            f"fan_in={self.fan_in}, fan_out={self.fan_out}, "
            f"tau_pre={self.tau_pre}, tau_post={self.tau_post}, "
            f"ltp={self.ltp_rate}, ltd={self.ltd_rate}, "
            f"w_fast_limit={self.w_fast_limit}"
        )
