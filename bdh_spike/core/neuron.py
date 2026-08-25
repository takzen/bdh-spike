"""BDH-PLIF neuron: Parametric LIF with lateral Baby-Dragon-Hatchling coupling.

Implements the canonical membrane dynamics from the project reference card:

    V[t]   = β · V[t-1] + I_syn[t] + M_BDH[t-1] − S[t-1] · V_th
    S[t]   = Θ(V[t] − V_th)
    M_BDH[t] = tanh(ρ · M_BDH[t-1] + g · S[t])

All emitted activations are binary spikes S(t) ∈ {0, 1}; differentiability is
provided solely by the fast-sigmoid surrogate gradient. Temporal tensors follow
the canonical [T, B, C] layout everywhere.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn

from bdh_spike.core.functional import spike_fn


@dataclass
class BDHState:
    """Encapsulated neuron state for :class:`BDHSpikeCell`.

    Attributes:
        mem: Membrane potential, shape ``[B, C]``.
        m_bdh: BDH recurrent coupling vector, shape ``[B, C]``.
        prev_spikes: Spikes emitted at the previous time-step, shape ``[B, C]``.
    """

    mem: torch.Tensor
    m_bdh: torch.Tensor
    prev_spikes: torch.Tensor


class BDHSpikeCell(nn.Module):
    """Single BDH-PLIF spiking neuron cell.

    Args:
        num_channels: Number of channels/features ``C`` of the input.
        beta_init: Initial membrane decay factor β ∈ (0, 1).
        v_th: Initial firing threshold ``V_th`` (registered as a buffer so the
            Stage-4 homeostat can adapt it online without touching parameters).
        surrogate_slope: Fast-sigmoid surrogate slope ``k`` (default 25).
        bdh_coupling: Strength ``g`` of the BDH recurrent coupling.
        bdh_decay: Decay factor ρ of the BDH coupling vector ``m_bdh``.
    """

    def __init__(
        self,
        num_channels: int,
        beta_init: float = 0.9,
        v_th: float = 1.0,
        surrogate_slope: float = 25.0,
        bdh_coupling: float = 0.5,
        bdh_decay: float = 0.8,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.surrogate_slope = float(surrogate_slope)
        self.bdh_coupling = float(bdh_coupling)
        self.bdh_decay = float(bdh_decay)

        # β constrained to (0, 1) via sigmoid on a raw learnable parameter.
        beta_logit_init = torch.full(
            (num_channels,), float(torch.logit(torch.tensor(float(beta_init))))
        )
        self.beta_logit = nn.Parameter(beta_logit_init)
        # Threshold as a buffer: adaptable by the homeostat, excluded from grads.
        self.register_buffer("v_th", torch.tensor(float(v_th)))

    @property
    def beta(self) -> torch.Tensor:
        """Effective per-channel membrane decay β ∈ (0, 1), shape ``[C]``."""
        return torch.sigmoid(self.beta_logit)

    def init_hidden(self, batch_size: int) -> BDHState:
        """Explicitly initialize (mem, m_bdh, prev_spikes) state for a batch."""
        device = self.v_th.device
        shape = (int(batch_size), self.num_channels)
        return BDHState(
            mem=torch.zeros(shape, device=device),
            m_bdh=torch.zeros(shape, device=device),
            prev_spikes=torch.zeros(shape, device=device),
        )

    def single_step(self, x: torch.Tensor, state: BDHState) -> tuple[torch.Tensor, BDHState]:
        """Advance one time-step.

        Args:
            x: Synaptic input current ``I_syn[t]``, shape ``[B, C]``.
            state: Current neuron state.

        Returns:
            Tuple ``(spikes, new_state)`` where ``spikes`` is binary ``[B, C]``.
        """
        # Membrane integration: leaky decay + input + BDH coupling − prior reset.
        mem = self.beta * state.mem + x + state.m_bdh - state.prev_spikes * self.v_th

        # Spike trigger with surrogate gradient (binary in {0, 1}).
        spikes = spike_fn(mem - self.v_th, threshold=self.v_th, slope=self.surrogate_slope)

        # Hard reset: membrane forced to 0 wherever a spike was emitted.
        mem = mem * (1.0 - spikes)

        # Nonlinear BDH recurrence: past coupling decays, fresh spikes excite.
        m_bdh = torch.tanh(self.bdh_decay * state.m_bdh + self.bdh_coupling * spikes)

        return spikes, BDHState(mem=mem, m_bdh=m_bdh, prev_spikes=spikes)

    def forward(
        self, x: torch.Tensor, state: BDHState | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, BDHState]:
        """Run the cell over one step ``[B, C]`` or a full sequence ``[T, B, C]``.

        Args:
            x: Input tensor of shape ``[B, C]`` (single step) or ``[T, B, C]``
                (temporal sequence, canonical layout).
            state: Optional initial state; lazily zero-initialized if omitted.

        Returns:
            For a single step: binary spikes ``[B, C]``.
            For a sequence: tuple ``(spikes_seq, final_state)`` where
            ``spikes_seq`` has shape ``[T, B, C]``.
        """
        if x.dim() == 2:
            if state is None:
                state = self.init_hidden(x.shape[0])
            spikes, _ = self.single_step(x, state)
            return spikes

        if x.dim() != 3:
            raise ValueError(f"Expected input of shape [B, C] or [T, B, C], got {tuple(x.shape)}")

        if state is None:
            state = self.init_hidden(x.shape[1])
        states: list[torch.Tensor] = []
        for t in range(x.shape[0]):
            spikes, state = self.single_step(x[t], state)
            states.append(spikes)
        return torch.stack(states, dim=0), state

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(channels={self.num_channels}, "
            f"v_th={float(self.v_th):.3f}, slope={self.surrogate_slope}, "
            f"coupling={self.bdh_coupling}, decay={self.bdh_decay})"
        )


def iterate_steps(
    cell: BDHSpikeCell, inputs: torch.Tensor, state: BDHState
) -> Iterator[tuple[torch.Tensor, BDHState]]:
    """Streaming helper: yield ``(spikes, state)`` per step over a ``[T, B, C]`` tensor."""
    for t in range(inputs.shape[0]):
        spikes, state = cell.single_step(inputs[t], state)
        yield spikes, state
