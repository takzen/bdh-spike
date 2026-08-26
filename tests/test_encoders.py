"""Unit tests for Stage-5 spike encoders (rate / latency / delta).

Covers: canonical [T, ...] layout, binary spike domain, firing-rate fidelity,
first-spike timing semantics, event-driven change detection and input
validation.
"""

from __future__ import annotations

import pytest
import torch

from bdh_spike.utils import delta_encode, latency_encode, rate_encode

T, B, C = 16, 4, 8


class TestRateEncoding:
    def test_output_shape_and_binary_domain(self) -> None:
        x = torch.rand(B, C)
        s = rate_encode(x, num_steps=T)
        assert s.shape == (T, B, C)
        assert set(s.unique().tolist()).issubset({0.0, 1.0})

    def test_expected_firing_rate_tracks_intensity(self) -> None:
        torch.manual_seed(0)
        x = torch.tensor([[0.1], [0.9]])  # two intensities
        s = rate_encode(x, num_steps=2000)
        rates = s.mean(dim=(0, 2))  # mean over T and the singleton channel
        assert rates[0] == pytest.approx(0.1, abs=0.03)
        assert rates[1] == pytest.approx(0.9, abs=0.03)
        assert rates[0] < rates[1]

    def test_zero_input_never_fires(self) -> None:
        s = rate_encode(torch.zeros(B, C), num_steps=T)
        assert torch.count_nonzero(s) == 0

    def test_gain_scales_firing_probability(self) -> None:
        torch.manual_seed(1)
        x = torch.full((B, C), 0.5)
        low = rate_encode(x, num_steps=1000, gain=0.5).mean()
        high = rate_encode(x, num_steps=1000, gain=1.0).mean()
        assert low < high

    def test_rejects_negative_intensities(self) -> None:
        with pytest.raises(ValueError):
            rate_encode(torch.tensor([[-0.5]]), num_steps=T)

    def test_rejects_invalid_num_steps(self) -> None:
        with pytest.raises(ValueError):
            rate_encode(torch.rand(B, C), num_steps=0)


class TestLatencyEncoding:
    def test_output_shape_and_binary_domain(self) -> None:
        x = torch.rand(B, C)
        s = latency_encode(x, num_steps=T)
        assert s.shape == (T, B, C)
        assert set(s.unique().tolist()).issubset({0.0, 1.0})

    def test_at_most_one_spike_per_unit(self) -> None:
        x = torch.rand(B, C)
        s = latency_encode(x, num_steps=T)
        per_unit = s.sum(dim=0)
        assert (per_unit <= 1.0).all(), "time-to-first-spike must be single-event"

    def test_strongest_value_fires_first(self) -> None:
        x = torch.zeros(B, C)
        x[:, 0] = 10.0  # only channel 0 is non-zero → fires at t = 0
        s = latency_encode(x, num_steps=T)
        first = s.argmax(dim=0)
        assert (first[:, 0] == 0).all(), "max intensity must spike at t=0"
        assert torch.count_nonzero(s[:, :, 1:]) == 0, "zero intensity never fires"

    def test_weaker_values_fire_later(self) -> None:
        x = torch.tensor([[1.0], [0.25]])
        s = latency_encode(x, num_steps=8)
        t_strong = s[:, 0, 0].argmax()
        t_weak = s[:, 1, 0].argmax()
        assert t_weak > t_strong

    def test_rejects_negative_intensities(self) -> None:
        with pytest.raises(ValueError):
            latency_encode(torch.tensor([[-1.0]]), num_steps=T)


class TestDeltaEncoding:
    def test_static_signal_is_silent(self) -> None:
        x = torch.full((T, B, C), 0.7)
        s = delta_encode(x, threshold=0.1)
        assert s.shape == (T, B, C)
        assert torch.count_nonzero(s) == 0

    def test_step_increase_spikes_exactly_once(self) -> None:
        x = torch.cat([torch.zeros(4, B, 1), torch.ones(4, B, 1)], dim=0)
        s = delta_encode(x, threshold=0.5, polarity="on")
        assert torch.count_nonzero(s) == B
        assert s[4].sum() == B, "ON events land at the change step"
        assert torch.count_nonzero(s[:4]) == 0

    def test_off_polarity_detects_decreases(self) -> None:
        x = torch.cat([torch.ones(4, B, 1), torch.zeros(4, B, 1)], dim=0)
        on = delta_encode(x, threshold=0.5, polarity="on")
        off = delta_encode(x, threshold=0.5, polarity="off")
        assert torch.count_nonzero(on) == 0
        assert off[4].sum() == B

    def test_subthreshold_change_is_ignored(self) -> None:
        x = torch.linspace(0.0, 0.05, T).view(T, 1, 1).expand(T, B, C)
        s = delta_encode(x, threshold=0.1)
        assert torch.count_nonzero(s) == 0

    def test_rejects_non_temporal_input(self) -> None:
        with pytest.raises(ValueError):
            delta_encode(torch.ones(5))  # rank-1: no feature axis
        with pytest.raises(ValueError):
            delta_encode(torch.ones(1, B, C))  # single time-step

    def test_rejects_unknown_polarity(self) -> None:
        with pytest.raises(ValueError):
            delta_encode(torch.ones(T, B, C), polarity="sideways")
