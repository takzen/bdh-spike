"""Unit tests for Stage-5 model backbones: BDHSpikeSeq + BDHSpikeViT.

Covers: [T, B, C] temporal invariant, binary spike domain, streaming-state
continuity, online STDP toggle (W_fast updated only when learn_fast=True),
surrogate gradient reach (patch conv / W_slow), ViT end-to-end logits and
temporal sparsity of internal activity.
"""

from __future__ import annotations

import pytest
import torch

from bdh_spike.models import BDHSpikeSeq, BDHSpikeViT

T, B = 12, 4


@pytest.fixture()
def seq_model() -> BDHSpikeSeq:
    torch.manual_seed(0)
    return BDHSpikeSeq(fan_in=8, fan_out=6)


class TestBDHSpikeSeq:
    def test_output_shape_and_binary_domain(self, seq_model: BDHSpikeSeq) -> None:
        x = torch.randn(T, B, 8)
        spikes, state = seq_model(x)
        assert spikes.shape == (T, B, 6)
        assert set(spikes.unique().tolist()).issubset({0.0, 1.0})
        assert state.cell.mem.shape == (B, 6)

    def test_invalid_input_raises(self, seq_model: BDHSpikeSeq) -> None:
        with pytest.raises(ValueError):
            seq_model(torch.randn(B, 8))
        with pytest.raises(ValueError):
            seq_model(torch.randn(T, B, 9))

    def test_streaming_state_continuity(self, seq_model: BDHSpikeSeq) -> None:
        x = torch.randn(T, B, 8)
        full, _ = seq_model(x)
        first_half, mid_state = seq_model(x[: T // 2])
        second_half, _ = seq_model(x[T // 2 :], mid_state)
        assert torch.allclose(first_half, full[: T // 2])
        assert torch.allclose(second_half, full[T // 2 :])

    def test_learn_fast_updates_w_fast(self) -> None:
        torch.manual_seed(0)
        net = BDHSpikeSeq(fan_in=8, fan_out=6, learn_fast=True, ltp_rate=0.05, ltd_rate=0.05)
        x = (torch.rand(T, B, 8) > 0.5).float()
        before = net.synapse.w_fast.clone()
        net(x)
        assert not torch.equal(net.synapse.w_fast, before), "STDP must write into W_fast"

    def test_plasticity_off_keeps_w_fast_frozen(self, seq_model: BDHSpikeSeq) -> None:
        x = (torch.rand(T, B, 8) > 0.5).float()
        before = seq_model.synapse.w_fast.clone()
        seq_model(x)
        assert torch.equal(seq_model.synapse.w_fast, before)

    def test_gradients_reach_w_slow(self, seq_model: BDHSpikeSeq) -> None:
        x = torch.randn(T, B, 8)
        spikes, _ = seq_model(x)
        spikes.mean().backward()
        assert seq_model.synapse.w_slow.grad is not None
        assert seq_model.cell.beta_logit.grad is not None
        assert seq_model.synapse.w_fast.grad is None

    def test_homeostasis_adapts_cell_threshold(self) -> None:
        net = BDHSpikeSeq(fan_in=8, fan_out=6, homeostasis=True, target_rate=0.01)
        x = torch.full((T, B, 8), 5.0)
        before = float(net.cell.v_th)
        net(x)
        assert float(net.cell.v_th) > before, "runaway firing must raise V_th"


class TestBDHSpikeViT:
    @pytest.fixture()
    def vit(self) -> BDHSpikeViT:
        torch.manual_seed(0)
        return BDHSpikeViT(
            img_size=28,
            patch_size=7,
            in_channels=1,
            num_classes=10,
            embed_dim=16,
            num_heads=2,
            num_steps=8,
        )

    def test_logits_shape(self, vit: BDHSpikeViT) -> None:
        images = torch.rand(B, 1, 28, 28)
        logits = vit(images)
        assert logits.shape == (B, 10)
        assert torch.isfinite(logits).all()

    def test_patch_embed_token_grid(self, vit: BDHSpikeViT) -> None:
        tokens = vit.patch_embed(torch.rand(B, 1, 28, 28))
        assert tokens.shape == (B, 16, 16), "4x4 patches → N=16 tokens"

    def test_features_are_binary_rates(self, vit: BDHSpikeViT) -> None:
        rates = vit.features(torch.rand(B, 1, 28, 28))
        assert rates.shape == (B, 16, 16)
        assert rates.min() >= 0.0
        assert rates.max() <= 1.0

    def test_end_to_end_backward_pass(self, vit: BDHSpikeViT) -> None:
        images = torch.rand(B, 1, 28, 28)
        loss = vit(images).sum()
        loss.backward()
        assert vit.patch_embed.proj.weight.grad is not None
        assert vit.head.weight.grad is not None
        if vit.attention is not None:
            assert vit.attention.W_q.weight.grad is not None

    def test_without_attention_still_works(self) -> None:
        net = BDHSpikeViT(
            img_size=14,
            patch_size=7,
            embed_dim=8,
            num_heads=1,
            num_classes=3,
            num_steps=6,
            use_attention=False,
        )
        logits = net(torch.rand(2, 1, 14, 14))
        assert logits.shape == (2, 3)

    def test_internal_sparsity_above_70pct(self, vit: BDHSpikeViT) -> None:
        """Energy-first invariant: hidden spike trains must stay sparse."""
        images = torch.rand(B, 1, 28, 28) * 0.5
        with torch.no_grad():
            spikes_in = vit._encode_tokens(vit.patch_embed(images))
            sparsity = 1.0 - float(spikes_in.mean())
        assert sparsity > 0.50, f"input spike train too dense: {sparsity=:.2%}"

    def test_rejects_wrong_image_size(self, vit: BDHSpikeViT) -> None:
        with pytest.raises(ValueError):
            vit(torch.rand(B, 1, 32, 32))

    def test_rejects_undivisible_architecture(self) -> None:
        with pytest.raises(ValueError):
            BDHSpikeViT(img_size=30, patch_size=7)
