"""Unit tests for Stage-6 energy metrics: sparsity, SOPs vs FLOPs, trackers."""

from __future__ import annotations

import pytest
import torch

from bdh_spike.neuromorphic import (
    SOPSMeter,
    SpikeSparsityTracker,
    calculate_sparsity,
    flops_dense,
    sops_count,
)

T, B, Cin, Cout = 8, 4, 6, 5


class TestSparsity:
    def test_all_silent_is_full_sparsity(self) -> None:
        assert calculate_sparsity(torch.zeros(T, B, Cin)) == pytest.approx(1.0)

    def test_all_active_is_zero_sparsity(self) -> None:
        assert calculate_sparsity(torch.ones(T, B, Cin)) == pytest.approx(0.0)

    def test_mixed_fraction(self) -> None:
        s = torch.zeros(10)
        s[:3] = 1.0
        assert calculate_sparsity(s) == pytest.approx(0.7)

    def test_empty_tensor_raises(self) -> None:
        with pytest.raises(ValueError):
            calculate_sparsity(torch.tensor([]))

    def test_no_grad_pollution(self) -> None:
        s = torch.rand(4, 4, requires_grad=True)
        calculate_sparsity(s)
        assert not s.grad_fn  # mean under no_grad must not build a graph


class TestSOPsCount:
    def test_reference_card_formula(self) -> None:
        """SOPs = Σ_t nnz(S_in[t]) × FanOut (AGENTS.md math reference)."""
        train = torch.zeros(T, B, Cin)
        train[0, 0, 0] = 1.0
        train[3, 2, 5] = 1.0
        train[3, 2, 1] = 1.0
        assert sops_count(train, fan_out=Cout) == 3 * Cout

    def test_dense_train_scales_linearly(self) -> None:
        dense = torch.ones(T, B, Cin)
        total_active = T * B * Cin
        assert sops_count(dense, fan_out=Cout) == total_active * Cout

    def test_rejects_non_temporal(self) -> None:
        with pytest.raises(ValueError):
            sops_count(torch.ones(Cin), fan_out=Cout)


class TestFlopsDense:
    def test_dense_equivalent_two_flops_per_mac(self) -> None:
        assert flops_dense(fan_in=Cin, fan_out=Cout, steps=T, batch=B) == 2 * T * B * Cin * Cout

    def test_defaults_single_step_batch(self) -> None:
        assert flops_dense(fan_in=3, fan_out=4) == 24


class TestSpikeSparsityTracker:
    def test_ema_converges_to_constant_input(self) -> None:
        tracker = SpikeSparsityTracker(momentum=0.9)
        sparse = torch.zeros(100)
        sparse[:5] = 1.0  # 95% sparsity
        for _ in range(50):
            tracker.update(sparse)
        rep = tracker.report()
        assert rep["sparsity_ema"] == pytest.approx(0.95, abs=1e-6)
        assert rep["sparsity_min"] == pytest.approx(0.95)
        assert rep["sparsity_max"] == pytest.approx(0.95)
        assert rep["updates"] == 50.0

    def test_min_max_envelope(self) -> None:
        tracker = SpikeSparsityTracker()
        tracker.update(torch.zeros(10))
        tracker.update(torch.ones(10))
        rep = tracker.report()
        assert rep["sparsity_min"] == pytest.approx(0.0)
        assert rep["sparsity_max"] == pytest.approx(1.0)

    def test_running_before_first_update(self) -> None:
        assert SpikeSparsityTracker().running == 0.0


class TestSOPSMeter:
    def test_report_and_ratio(self) -> None:
        meter = SOPSMeter()
        meter.register_layer(fan_in=Cin, fan_out=Cout, steps=T, batch=B)
        train = torch.zeros(T, B, Cin)
        train[:, :, :2] = 1.0  # half the inputs active
        meter.update(train, fan_out=Cout)

        rep = meter.report()
        expected_sops = T * B * 2 * Cout  # two active channels per element
        assert rep["sops"] == expected_sops
        assert rep["dense_macs"] == T * B * Cin * Cout
        assert rep["dense_flops"] == 2 * T * B * Cin * Cout
        # 2/6 channels active × 2 FLOPs/MAC → exactly 6× fewer ops than dense.
        assert rep["flops_per_sop_ratio"] == pytest.approx(6.0)

    def test_accumulates_across_updates(self) -> None:
        meter = SOPSMeter()
        spike = torch.ones(1, 1, 1)
        meter.update(spike, fan_out=4)
        meter.update(spike, fan_out=4)
        assert meter.report()["sops"] == 8

    def test_inf_ratio_before_any_activity(self) -> None:
        meter = SOPSMeter()
        assert meter.report()["flops_per_sop_ratio"] == float("inf")
