"""Unit tests for Stage-4 plasticity: dual-weight STDP engine + homeostat.

Covers: W_slow/W_fast separation (parameter vs buffer), LTP/LTD timing
causality, exponential trace decay, third-factor modulation, W_fast clamping,
strict no-gradient-leak guarantees, and adaptive V_th regulation including
BDHSpikeCell integration.
"""

from __future__ import annotations

import math

import pytest
import torch

from bdh_spike.core import BDHSpikeCell
from bdh_spike.plasticity import AdaptiveThreshold, DualWeightLinear, STDPState

B, Cin, Cout, T = 4, 6, 5, 8


@pytest.fixture()
def layer() -> DualWeightLinear:
    torch.manual_seed(0)
    return DualWeightLinear(fan_in=Cin, fan_out=Cout)


class TestDualWeightSeparation:
    def test_w_slow_is_parameter_w_fast_is_buffer(self, layer: DualWeightLinear) -> None:
        assert isinstance(layer.w_slow, torch.nn.Parameter)
        assert "w_slow" in dict(layer.named_parameters())
        # W_fast must NEVER be trainable (AGENTS.md rule B).
        param_names = {n for n, _ in layer.named_parameters()}
        assert "w_fast" not in param_names
        assert not layer.w_fast.requires_grad

    def test_initial_w_fast_is_zero(self, layer: DualWeightLinear) -> None:
        assert torch.count_nonzero(layer.w_fast) == 0

    @pytest.mark.parametrize("shape", [(B, Cin), (T, B, Cin)])
    def test_transmission_shapes(self, layer: DualWeightLinear, shape: tuple[int, ...]) -> None:
        x = torch.randn(*shape)
        y = layer(x)
        assert y.shape == (*shape[:-1], Cout)

    def test_fused_kernel_transmission(self, layer: DualWeightLinear) -> None:
        """Transmission must use the fused kernel (W_slow + W_fast)ᵀ·x."""
        x = torch.randn(T, B, Cin)
        with torch.no_grad():
            layer.w_fast.normal_(0.0, 0.1)
        expected = x @ (layer.w_slow + layer.w_fast).transpose(0, 1)
        assert torch.allclose(layer(x), expected)

    def test_invalid_inputs_raise(self, layer: DualWeightLinear) -> None:
        with pytest.raises(ValueError):
            layer(torch.randn(T, T, B, Cin))
        with pytest.raises(ValueError):
            layer(torch.randn(B, Cout))

    def test_init_hidden_traces(self, layer: DualWeightLinear) -> None:
        state = layer.init_hidden(B)
        assert isinstance(state, STDPState)
        assert state.a_pre.shape == (B, Cin)
        assert state.a_post.shape == (B, Cout)
        assert torch.count_nonzero(state.a_pre) == 0
        assert torch.count_nonzero(state.a_post) == 0


class TestSTDPRule:
    @staticmethod
    def _pair(pre_t: int, post_t: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Two-step episode where every sample's pre/post unit fires once."""
        pre = torch.zeros(2, B, Cin)
        post = torch.zeros(2, B, Cout)
        pre[pre_t, :, 0] = 1.0
        post[post_t, :, 0] = 1.0
        return pre, post

    def test_ltp_when_pre_precedes_post(self) -> None:
        """Causal pairing (pre fires before post) potentiates W_fast."""
        net = DualWeightLinear(fan_in=Cin, fan_out=Cout, ltp_rate=0.05, ltd_rate=0.0)
        pre, post = self._pair(pre_t=0, post_t=1)
        net.run_episode(pre, post)

        expected = 0.05 * net.decay_pre  # trace decays once between steps
        assert net.w_fast[0, 0].item() == pytest.approx(expected, rel=1e-6)
        mask = torch.ones_like(net.w_fast, dtype=torch.bool)
        mask[0, 0] = False
        assert torch.count_nonzero(net.w_fast[mask]) == 0, "off-path synapses untouched"

    def test_ltd_when_post_precedes_pre(self) -> None:
        """Anti-causal pairing (post fired earlier) depresses W_fast."""
        net = DualWeightLinear(fan_in=Cin, fan_out=Cout, ltp_rate=0.0, ltd_rate=0.04)
        pre, post = self._pair(pre_t=1, post_t=0)
        net.run_episode(pre, post)

        expected = -0.04 * net.decay_post
        assert net.w_fast[0, 0].item() == pytest.approx(expected, rel=1e-6)

    def test_trace_decay_is_exponential(self) -> None:
        """A_pre must follow A[t] = A[t-1]·e^{-Δt/τ} + S[t] exactly."""
        net = DualWeightLinear(fan_in=Cin, fan_out=Cout, tau_pre=1.0)
        decay = math.exp(-1.0 / 1.0)
        pre = torch.zeros(T, B, Cin)
        pre[0, :, 3] = 1.0  # single spike at t=0 on channel 3
        post = torch.zeros(T, B, Cout)
        state = net.run_episode(pre, post)

        assert state.a_pre[0, 3].item() == pytest.approx(decay ** (T - 1), rel=1e-6)

    def test_zero_modulator_freezes_weights(self, layer: DualWeightLinear) -> None:
        pre = (torch.rand(T, B, Cin) > 0.5).float()
        post = (torch.rand(T, B, Cout) > 0.5).float()
        layer.run_episode(pre, post, modulators=0.0)
        assert torch.count_nonzero(layer.w_fast) == 0

    def test_modulator_scales_update_linearly(self) -> None:
        pre, post = self._pair(pre_t=0, post_t=1)

        net_a = DualWeightLinear(fan_in=Cin, fan_out=Cout, ltp_rate=0.05, ltd_rate=0.0)
        net_a.run_episode(pre, post, modulators=1.0)
        net_b = DualWeightLinear(fan_in=Cin, fan_out=Cout, ltp_rate=0.05, ltd_rate=0.0)
        net_b.run_episode(pre, post, modulators=torch.tensor([2.0]))

        assert net_b.w_fast[0, 0].item() == pytest.approx(2.0 * net_a.w_fast[0, 0].item())

    def test_w_fast_clamped_within_bounds(self) -> None:
        net = DualWeightLinear(
            fan_in=Cin, fan_out=Cout, ltp_rate=0.5, ltd_rate=0.0, w_fast_limit=1.0
        )
        pre = torch.ones(T, B, Cin)
        post = torch.ones(T, B, Cout)
        net.run_episode(pre, post)
        assert (net.w_fast.abs() <= net.w_fast_limit + 1e-7).all()
        assert net.w_fast.min().item() == pytest.approx(net.w_fast_limit), "saturates at bound"

    def test_run_episode_matches_manual_steps(self) -> None:
        """Unrolled plastic_step calls must equal the batched run_episode."""
        torch.manual_seed(3)
        pre = (torch.rand(T, B, Cin) > 0.6).float()
        post = (torch.rand(T, B, Cout) > 0.7).float()

        manual = DualWeightLinear(fan_in=Cin, fan_out=Cout)
        state = manual.init_hidden(B)
        for t in range(T):
            state = manual.plastic_step(pre[t], post[t], state)
        manual_w_fast = manual.w_fast.clone()

        batched = DualWeightLinear(fan_in=Cin, fan_out=Cout)
        with torch.no_grad():
            batched.w_slow.copy_(manual.w_slow)
        final_state = batched.run_episode(pre, post)

        assert torch.allclose(batched.w_fast, manual_w_fast)
        assert torch.allclose(final_state.a_pre, state.a_pre)
        assert torch.allclose(final_state.a_post, state.a_post)


class TestNoGradientLeak:
    def test_forward_gradients_flow_only_to_w_slow(self, layer: DualWeightLinear) -> None:
        x = torch.randn(B, Cin, requires_grad=True)
        loss = layer(x).sum()
        loss.backward()
        assert layer.w_slow.grad is not None
        assert layer.w_fast.grad is None, "W_fast must stay outside the autograd graph"

    def test_online_update_is_grad_free(self, layer: DualWeightLinear) -> None:
        """STDP must not create grad-tracking tensors even under enabled grad."""
        pre = (torch.rand(B, Cin) > 0.5).float()
        post = (torch.rand(B, Cout) > 0.5).float()
        state = layer.init_hidden(B)
        new_state = layer.plastic_step(pre, post, state)

        assert layer.w_fast.grad is None
        assert layer.w_slow.grad is None
        for trace in (new_state.a_pre, new_state.a_post):
            assert not trace.requires_grad


class TestModulatorFactor:
    """Dedicated §2-PLAN2 audit tests for the third factor ``M``.

    Documented semantics (must stay true): ``M`` defaults to 1.0, gates the
    magnitude of ΔW linearly, ``M == 0`` freezes ``W_fast``, per-sample
    ``[B]`` tensors broadcast over channels. ``M`` has no data/reward
    dependence anywhere in the codebase — it exists purely as an interface.
    """

    @staticmethod
    def _pair() -> tuple[torch.Tensor, torch.Tensor]:
        pre = torch.zeros(2, B, Cin)
        post = torch.zeros(2, B, Cout)
        pre[0, :, 0] = 1.0
        post[1, :, 0] = 1.0
        return pre, post

    def test_default_modulator_is_one(self, layer: DualWeightLinear) -> None:
        import inspect

        default = inspect.signature(DualWeightLinear.plastic_step).parameters["modulator"].default
        assert default == 1.0, "M must default to exactly 1.0"

    def test_unit_m_equals_no_explicit_m(self) -> None:
        pre, post = self._pair()
        net_a = DualWeightLinear(fan_in=Cin, fan_out=Cout, ltp_rate=0.05, ltd_rate=0.0)
        net_b = DualWeightLinear(fan_in=Cin, fan_out=Cout, ltp_rate=0.05, ltd_rate=0.0)
        net_a.run_episode(pre, post, modulators=1.0)
        net_b.run_episode(pre, post)  # default path
        assert torch.equal(net_a.w_fast, net_b.w_fast)

    def test_per_sample_tensor_broadcasts_over_channels(self) -> None:
        pre, post = self._pair()
        net = DualWeightLinear(fan_in=Cin, fan_out=Cout, ltp_rate=0.05, ltd_rate=0.0)
        m = torch.full((B,), 2.0)  # [B] per-sample modulation
        net.run_episode(pre, post, modulators=m)
        # Same total write as scalar 2.0 (all samples share the value).
        expected = 0.05 * net.decay_pre * 2.0
        assert net.w_fast[0, 0].item() == pytest.approx(expected, rel=1e-6)

    def test_zero_freezes_w_fast_during_training_stream(self, layer: DualWeightLinear) -> None:
        pre = (torch.rand(T, B, Cin) > 0.5).float()
        post = (torch.rand(T, B, Cout) > 0.5).float()
        before = layer.w_fast.clone()
        for t in range(T):
            layer.plastic_step(pre[t], post[t], layer.init_hidden(B), modulator=0.0)
        assert torch.equal(layer.w_fast, before)

    def test_negative_modulator_inverts_update_sign(self) -> None:
        """Documented edge semantics: negative M flips potentiation↔depression."""
        pre, post = self._pair()
        pos = DualWeightLinear(fan_in=Cin, fan_out=Cout, ltp_rate=0.05, ltd_rate=0.05)
        neg = DualWeightLinear(fan_in=Cin, fan_out=Cout, ltp_rate=0.05, ltd_rate=0.05)
        pos.run_episode(pre, post, modulators=1.0)
        neg.run_episode(pre, post, modulators=-1.0)
        assert torch.equal(pos.w_fast, -neg.w_fast)

    def test_m_has_no_hidden_reward_or_label_dependence(self) -> None:
        """Static audit: the only M entry point is the explicit argument."""
        import inspect

        src = inspect.getsource(DualWeightLinear)
        forbidden = ("reward", "label", "target", "loss")
        assert not any(word in src.lower() for word in forbidden), (
            "M must remain an explicit interface parameter"
        )


class TestAdaptiveThresholdHomeostat:
    @pytest.fixture()
    def homeo(self) -> AdaptiveThreshold:
        return AdaptiveThreshold(num_channels=None, target_rate=0.1, eta=0.05, v_th_init=1.0)

    def test_hyperactivity_raises_threshold(self, homeo: AdaptiveThreshold) -> None:
        spikes = torch.ones(B, Cin)
        for _ in range(20):
            homeo.step(spikes)
        assert float(homeo.v_th) > 1.0, "seizure-like firing must raise V_th"

    def test_silence_lowers_threshold(self, homeo: AdaptiveThreshold) -> None:
        spikes = torch.zeros(B, Cin)
        for _ in range(50):
            homeo.step(spikes)
        assert float(homeo.v_th) < 1.0, "silence must lower V_th toward excitability"

    def test_threshold_stays_within_bounds(self) -> None:
        homeo = AdaptiveThreshold(
            num_channels=None, target_rate=0.5, eta=10.0, v_min=0.05, v_max=5.0
        )
        for _ in range(100):
            homeo.step(torch.zeros(B, Cin))
            homeo.step(torch.ones(B, Cin))
        assert 0.05 <= float(homeo.v_th) <= 5.0

    def test_rate_ema_update(self) -> None:
        homeo = AdaptiveThreshold(num_channels=None, rate_beta=0.5)
        homeo.step(torch.ones(2, 4))
        assert float(homeo.rate) == pytest.approx(0.5)

    def test_per_channel_mode_shapes_and_independence(self) -> None:
        homeo = AdaptiveThreshold(num_channels=Cin, target_rate=0.5, rate_beta=0.0)
        spikes = torch.zeros(B, Cin)
        spikes[:, 0] = 1.0  # only channel 0 is active
        homeo.step(spikes)
        assert homeo.v_th.shape == (Cin,)
        assert float(homeo.v_th[0]) > float(homeo.v_th[1]), "active channel gets higher V_th"

    def test_buffers_require_no_grad(self, homeo: AdaptiveThreshold) -> None:
        assert not homeo.v_th.requires_grad
        assert not homeo.rate.requires_grad
        homeo.step(torch.ones(2, 2))
        assert not homeo.v_th.requires_grad

    def test_reset_restores_initial_state(self, homeo: AdaptiveThreshold) -> None:
        homeo.step(torch.ones(2, 4))
        homeo.reset()
        assert float(homeo.v_th) == pytest.approx(1.0)
        assert float(homeo.rate) == pytest.approx(0.0)

    def test_apply_to_bdh_spike_cell(self) -> None:
        """Integration: runaway cell activity must raise the shared V_th buffer."""
        torch.manual_seed(0)
        cell = BDHSpikeCell(num_channels=Cin, v_th=1.0)
        homeo = AdaptiveThreshold(target_rate=0.01, rate_beta=0.5, eta=0.1)

        x = torch.full((T, B, Cin), 5.0)
        spikes_seq, _ = cell(x)
        homeo.observe_sequence(spikes_seq)
        before = float(cell.v_th)
        homeo.apply_to(cell)

        assert float(cell.v_th) > before, "hyperactive cell must get a raised threshold"
        assert not cell.v_th.requires_grad

    def test_apply_to_rejects_per_channel_mode(self) -> None:
        homeo = AdaptiveThreshold(num_channels=Cin)
        cell = BDHSpikeCell(num_channels=Cin)
        with pytest.raises(TypeError):
            homeo.apply_to(cell)

    def test_observe_sequence_validates_rank(self, homeo: AdaptiveThreshold) -> None:
        with pytest.raises(ValueError):
            homeo.observe_sequence(torch.ones(B, Cin))
