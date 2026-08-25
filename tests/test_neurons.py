"""Unit tests for the BDH-PLIF neuron (Stage 2).

Covers: [T, B, C] temporal shape invariant, binary spike domain, hard membrane
reset correctness, BDH recurrent coupling, surrogate-gradient flow and state
initialization via ``init_hidden``.
"""

from __future__ import annotations

import pytest
import torch

from bdh_spike.core import BDHSpikeCell, BDHState, iterate_steps, spike_fn

T, B, C = 10, 4, 8


@pytest.fixture()
def cell() -> BDHSpikeCell:
    torch.manual_seed(0)
    return BDHSpikeCell(num_channels=C)


class TestBinarySpikeDomain:
    def test_output_is_binary(self, cell: BDHSpikeCell) -> None:
        x = torch.randn(T, B, C)
        spikes, _ = cell(x)
        assert set(spikes.unique().tolist()).issubset({0.0, 1.0})

    def test_spike_fn_forward_is_heaviside(self) -> None:
        x = torch.tensor([-1.0, 0.99, 1.0, 2.0])
        s = spike_fn(x)
        assert torch.equal(s, torch.tensor([0.0, 0.0, 1.0, 1.0]))


class TestTemporalShape:
    def test_sequence_shape_tbc(self, cell: BDHSpikeCell) -> None:
        x = torch.randn(T, B, C)
        spikes, state = cell(x)
        assert spikes.shape == (T, B, C), "canonical [T, B, C] layout violated"
        assert state.mem.shape == (B, C)
        assert state.m_bdh.shape == (B, C)

    def test_single_step_shape_bc(self, cell: BDHSpikeCell) -> None:
        x = torch.randn(B, C)
        spikes = cell(x)  # type: ignore[arg-type]
        assert spikes.shape == (B, C)

    def test_invalid_rank_raises(self, cell: BDHSpikeCell) -> None:
        with pytest.raises(ValueError):
            cell(torch.randn(T, B, C, 3))  # type: ignore[arg-type]

    def test_looped_single_step_matches_sequence_call(self, cell: BDHSpikeCell) -> None:
        """Unrolled single steps must reproduce the batched sequence forward."""
        torch.manual_seed(42)
        x = torch.randn(T, B, C)

        state = cell.init_hidden(B)
        manual = []
        for t in range(T):
            spikes, state = cell.single_step(x[t], state)
            manual.append(spikes)
        manual_seq = torch.stack(manual, dim=0)

        seq_out, final_state = cell(x.clone(), cell.init_hidden(B))
        assert torch.equal(manual_seq, seq_out)
        assert torch.equal(manual[-1], final_state.prev_spikes)


class TestMembraneReset:
    def test_hard_reset_zeroes_membrane_on_spike(self, cell: BDHSpikeCell) -> None:
        """After a spike the membrane must be exactly 0 (hard reset)."""
        strong_input = torch.full((B, C), 5.0)
        spikes, state = cell.single_step(strong_input, cell.init_hidden(B))
        assert spikes.all(), "strong input must drive every unit over threshold"
        assert torch.equal(state.mem, torch.zeros_like(state.mem))

    def test_prev_spike_subtracted_next_step(self) -> None:
        """V[t] includes the − S[t-1]·V_th term from the reference equation."""
        beta_fixed = BDHSpikeCell(num_channels=1, beta_init=0.5)
        # Keep m_bdh out of the way by disabling coupling for an exact check.
        beta_fixed.bdh_coupling = 0.0

        state = beta_fixed.init_hidden(1)
        _, state = beta_fixed.single_step(torch.full((1, 1), 3.0), state)
        assert state.prev_spikes.item() == 1.0

        mem_before = state.mem.item()  # == 0 after hard reset
        _, next_state = beta_fixed.single_step(torch.zeros(1, 1), state)
        expected = 0.5 * mem_before + 0.0 + state.m_bdh.item() - float(beta_fixed.v_th)
        assert next_state.mem.item() == pytest.approx(expected, abs=1e-6)
        assert next_state.mem.item() < 0.0, "refractory effect must push V below 0"


class TestBDHCoupling:
    def test_m_bdh_activates_after_spikes(self, cell: BDHSpikeCell) -> None:
        x = torch.full((T, B, C), 4.0)
        _, state = cell(x)
        assert (state.m_bdh > 0).all(), "spikes must excite the BDH coupling vector"

    def test_m_bdh_is_bounded_nonlinear(self, cell: BDHSpikeCell) -> None:
        x = torch.full((T, B, C), 100.0)
        _, state = cell(x)
        assert (state.m_bdh.abs() <= 1.0).all(), "tanh keeps m_bdh in [-1, 1]"


class TestSurrogateGradient:
    def test_gradient_flows_to_learnable_beta(self, cell: BDHSpikeCell) -> None:
        x = torch.randn(T, B, C, requires_grad=True)
        spikes, _ = cell(x)
        loss = spikes.sum()
        loss.backward()

        assert cell.beta_logit.grad is not None
        assert cell.beta_logit.grad.abs().sum() > 0, "surrogate gradient vanished entirely"
        assert x.grad is not None

    def test_surrogate_matches_reference_formula(self) -> None:
        k, v_th = 25.0, 1.0
        x = torch.tensor([0.999, 1.001])
        x.requires_grad_(True)
        spike_fn(x, threshold=torch.tensor(v_th), slope=k).sum().backward()

        for xi, gi in zip(x.detach(), x.grad, strict=True):
            expected = 1.0 / (1.0 + k * abs(float(xi) - v_th)) ** 2
            assert gi.item() == pytest.approx(expected, rel=1e-5)

    def test_no_grad_pollution_from_v_th_buffer(self, cell: BDHSpikeCell) -> None:
        assert not cell.v_th.requires_grad, "V_th buffer must stay grad-free"
        assert len(list(cell.parameters())) == 1, "only β should be trainable"


class TestStateInitialization:
    def test_init_hidden_shapes_and_zeros(self, cell: BDHSpikeCell) -> None:
        state = cell.init_hidden(B)
        assert isinstance(state, BDHState)
        for field in (state.mem, state.m_bdh, state.prev_spikes):
            assert field.shape == (B, C)
            assert torch.count_nonzero(field) == 0

    def test_explicit_state_persists_across_calls(self, cell: BDHSpikeCell) -> None:
        """Two consecutive calls with explicit state must differ from a fresh one."""
        x = torch.full((B, C), 2.0)
        state = cell.init_hidden(B)
        _, state = cell.single_step(x, state)
        _, continued = cell.single_step(torch.zeros(B, C), state)
        _, fresh = cell.single_step(torch.zeros(B, C), cell.init_hidden(B))
        assert not torch.equal(continued.mem, fresh.mem)

    def test_iterate_steps_streaming_helper(self, cell: BDHSpikeCell) -> None:
        x = torch.randn(T, B, C)
        state = cell.init_hidden(B)
        collected = []
        for step_idx, (spikes_t, state_t) in enumerate(iterate_steps(cell, x, state)):
            assert spikes_t.shape == (B, C)
            collected.append(spikes_t)
            state = state_t
        assert step_idx == T - 1
        assert len(collected) == T


class TestNumericalStability:
    def test_no_nans_under_long_strong_drive(self, cell: BDHSpikeCell) -> None:
        x = torch.full((64, B, C), 10.0)
        spikes, state = cell(x)
        assert torch.isfinite(spikes).all()
        assert torch.isfinite(state.mem).all()
        assert torch.isfinite(state.m_bdh).all()

    def test_deterministic_given_seed(self, cell: BDHSpikeCell) -> None:
        torch.manual_seed(7)
        x1 = torch.randn(T, B, C)
        torch.manual_seed(7)
        x2 = torch.randn(T, B, C)
        out1, _ = cell(x1)
        out2, _ = cell(x2)
        assert torch.equal(out1, out2)
