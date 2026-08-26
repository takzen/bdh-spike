"""Recurrent sequence model for streaming / continuous data (Stage 5).

Fuses the full Stage-2..4 stack into one streaming block::

    S_in ──► DualWeightLinear ──► BDH-PLIF Cell ──► S_out
                (W_slow+W_fast)       (V[t], M_BDH)

Optional online extras (all strictly gradient-free):

* 3-factor STDP writing episodic structure into ``W_fast`` during inference;
* scalar homeostat keeping the shared ``V_th`` inside a healthy firing band.

Temporal invariant: canonical ``[T, B, C]`` everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from bdh_spike.core.neuron import BDHSpikeCell, BDHState
from bdh_spike.plasticity.homeostat import AdaptiveThreshold
from bdh_spike.plasticity.stdp import DualWeightLinear, STDPState

__all__ = ["BDHSpikeSeq", "SeqState"]


@dataclass
class SeqState:
    """Joint streaming state: PLIF membrane + eligibility traces."""

    cell: BDHState
    stdp: STDPState


class BDHSpikeSeq(nn.Module):
    """Streaming BDH-Spike block for continuous data sequences.

    Args:
        fan_in: Input feature count ``C_in``.
        fan_out: Output feature count ``C_out``.
        tau_pre / tau_post: STDP trace decay constants.
        ltp_rate / ltd_rate: Potentiation / depression rates η₊ / η₋.
        beta_init: Initial membrane decay β of the PLIF cell.
        v_th: Initial firing threshold ``V_th``.
        surrogate_slope: Fast-sigmoid surrogate slope ``k``.
        bdh_coupling / bdh_decay: BDH lateral coupling strength ``g`` / decay ρ.
        learn_fast: If ``True``, update ``W_fast`` online via STDP during
            :meth:`forward` (inference-time continual adaptation).
        homeostasis: If ``True``, adapt the cell's ``V_th`` from observed
            output activity after each sequence.
        target_rate: Homeostatic target firing rate (if ``homeostasis``).
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
        beta_init: float = 0.9,
        v_th: float = 1.0,
        surrogate_slope: float = 25.0,
        bdh_coupling: float = 0.5,
        bdh_decay: float = 0.8,
        learn_fast: bool = False,
        homeostasis: bool = False,
        target_rate: float = 0.10,
    ) -> None:
        super().__init__()
        self.fan_in = int(fan_in)
        self.fan_out = int(fan_out)
        self.learn_fast = bool(learn_fast)

        self.synapse = DualWeightLinear(
            fan_in=fan_in,
            fan_out=fan_out,
            tau_pre=tau_pre,
            tau_post=tau_post,
            ltp_rate=ltp_rate,
            ltd_rate=ltd_rate,
        )
        self.cell = BDHSpikeCell(
            num_channels=fan_out,
            beta_init=beta_init,
            v_th=v_th,
            surrogate_slope=surrogate_slope,
            bdh_coupling=bdh_coupling,
            bdh_decay=bdh_decay,
        )
        self.homeostat = (
            AdaptiveThreshold(num_channels=None, target_rate=target_rate) if homeostasis else None
        )

    def init_hidden(self, batch_size: int) -> SeqState:
        """Zero-initialize the full joint state for a batch."""
        return SeqState(
            cell=self.cell.init_hidden(batch_size), stdp=self.synapse.init_hidden(batch_size)
        )

    def step(self, x_t: torch.Tensor, state: SeqState) -> tuple[torch.Tensor, SeqState]:
        """Advance one streaming step; returns ``(spikes [B, C_out], state)``."""
        current = self.synapse(x_t)
        spikes, cell_state = self.cell.single_step(current, state.cell)
        if self.learn_fast:
            stdp_state = self.synapse.plastic_step(x_t, spikes, state.stdp)
        else:
            stdp_state = state.stdp
        return spikes, SeqState(cell=cell_state, stdp=stdp_state)

    def forward(
        self, x: torch.Tensor, state: SeqState | None = None
    ) -> tuple[torch.Tensor, SeqState]:
        """Run a ``[T, B, C_in]`` sequence; returns ``(spikes [T,B,C_out], final state)``."""
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape [T, B, C_in], got {tuple(x.shape)}")
        if x.shape[-1] != self.fan_in:
            raise ValueError(f"Expected fan_in={self.fan_in}, got {x.shape[-1]}")

        if state is None:
            state = self.init_hidden(x.shape[1])
        traces: list[torch.Tensor] = []
        for t in range(x.shape[0]):
            spikes, state = self.step(x[t], state)
            traces.append(spikes)
        spikes_seq = torch.stack(traces, dim=0)

        if self.homeostat is not None:
            self.homeostat.observe_sequence(spikes_seq)
            self.homeostat.apply_to(self.cell)
        return spikes_seq, state
