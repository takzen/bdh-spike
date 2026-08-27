"""Unit tests for Spike-Driven BDH Attention (Stage 3).

Covers: [T, B, N, C] temporal shape invariant, binary spike domain,
softmax-free associative masking (Q∧K), surrogate-gradient flow into the
synaptic projections, strict float-free guarantees of the ``bitwise`` path
(via ``TorchDispatchMode``), cross-mode parity and the memory footprint of
the association matrix vs a classic float-softmax attention map.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from bdh_spike.core import SpikeDrivenAttention, associate_spikes

T, B, N, C, H = 6, 3, 10, 16, 2


def make_attention(mode: str = "surrogate", heads: int = H) -> SpikeDrivenAttention:
    torch.manual_seed(0)
    return SpikeDrivenAttention(embed_dim=C, num_heads=heads, mode=mode)


def drive(scale: float = 2.0) -> torch.Tensor:
    """Input strong enough that a healthy fraction of units crosses V_th."""
    g = torch.Generator().manual_seed(123)
    return scale * torch.randn(T, B, N, C, generator=g)


class FloatOpCatcher(TorchDispatchMode):
    """Records every aten op that materializes a floating-point tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.offenders: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        out = func(*args, **(kwargs or {}))
        candidates = list(out) if isinstance(out, (list, tuple)) else [out]
        for item in candidates:
            if isinstance(item, torch.Tensor) and item.is_floating_point():
                self.offenders.append(str(func))
        return out


class TestBinarySpikeDomain:
    @pytest.mark.parametrize("mode", ["surrogate", "bitwise"])
    def test_output_is_binary(self, mode: str) -> None:
        att = make_attention(mode)
        y = att(drive())
        assert set(y.unique().tolist()).issubset({0.0, 1.0})

    def test_bitwise_mask_is_boolean(self) -> None:
        att = make_attention("bitwise")
        mask = att.attention_mask(drive())
        assert mask.dtype == torch.bool
        assert mask.shape == (T, B, H, N, N)

    def test_associate_spikes_rejects_non_bool(self) -> None:
        with pytest.raises(TypeError):
            associate_spikes(torch.zeros(2, 2), torch.zeros(2, 2))

    def test_associate_spikes_counts_overlaps(self) -> None:
        q = torch.tensor([[1.0, 0.0], [1.0, 1.0]]).bool()
        k = torch.tensor([[1.0, 1.0]]).bool()
        counts = associate_spikes(q, k)
        assert counts.tolist() == [[1], [2]]
        assert not counts.is_floating_point(), "counts must live in integer domain"


class TestTemporalShape:
    @pytest.mark.parametrize("mode", ["surrogate", "bitwise"])
    def test_sequence_shape_tbnc(self, mode: str) -> None:
        att = make_attention(mode)
        y = att(drive())
        assert y.shape == (T, B, N, C), "canonical [T, B, N, C] layout violated"

    def test_invalid_rank_raises(self) -> None:
        with pytest.raises(ValueError):
            make_attention()(torch.randn(B, N, C))

    def test_wrong_channel_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            make_attention()(torch.randn(T, B, N, C + 1))

    def test_embed_dim_must_divide_heads(self) -> None:
        with pytest.raises(ValueError):
            SpikeDrivenAttention(embed_dim=C, num_heads=5)

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            SpikeDrivenAttention(embed_dim=C, mode="softmax")


class TestSoftmaxFreeAssociativeMasking:
    def test_no_softmax_call_anywhere_in_module(self) -> None:
        import bdh_spike.core.attention as attention_module

        source = inspect.getsource(attention_module)
        assert "torch.softmax" not in source, "Softmax must never be called"
        assert "F.softmax" not in source, "Softmax must never be called"
        assert ".softmax" not in source.replace("'surrogate'", "").replace('"bitwise"', "")

    def test_mask_matches_manual_and_counts(self) -> None:
        att = make_attention("surrogate")
        x = drive()
        q_s, k_s, _ = att._encode(x)
        expected = (associate_spikes(q_s.bool(), k_s.bool()) >= float(att.attn_th)).to(torch.float)
        assert torch.equal(att.attention_mask(x), expected)

    def test_mask_is_sparse(self) -> None:
        mask = make_attention("surrogate").attention_mask(drive())
        density = mask.mean().item()
        assert 0.0 < density < 1.0, "associative masking must prune, not saturate"


class TestBitwiseFloatFreePath:
    def test_no_float_ops_after_encoding(self) -> None:
        att = make_attention("bitwise")
        q_s, k_s, v_s = att._encode(drive())
        with FloatOpCatcher() as catcher:
            att._attend_bitwise(q_s, k_s, v_s)
        assert not catcher.offenders, (
            f"floating-point ops leaked into the bitwise path: {catcher.offenders}"
        )

    def test_forward_values_identical_across_modes(self) -> None:
        soft, bitw = make_attention("surrogate"), make_attention("bitwise")
        bitw.load_state_dict(soft.state_dict())
        x = drive()
        assert torch.equal(soft(x), bitw(x)), "bitwise graph must match surrogate forward"


class TestSurrogateGradientFlow:
    def test_gradients_reach_all_projections(self) -> None:
        att = make_attention("surrogate")
        x = drive().requires_grad_(True)
        att(x).sum().backward()

        for name in ("W_q", "W_k", "W_v"):
            grad = getattr(att, name).weight.grad
            assert grad is not None, f"no gradient reached {name}"
            assert grad.abs().sum() > 0, f"gradient vanished entirely at {name}"
        assert x.grad is not None

    def test_thresholds_stay_grad_free(self) -> None:
        att = make_attention("surrogate")
        for buf in (att.in_v_th, att.attn_th, att.out_v_th):
            assert not buf.requires_grad
        trainable = {n for n, _ in att.named_parameters()}
        assert trainable == {"W_q.weight", "W_k.weight", "W_v.weight"}


class TestMemoryOverhead:
    def test_association_map_beats_float_softmax_footprint(self) -> None:
        Tm, Bm, Hm, Nm = 2, 2, 2, 32
        att = SpikeDrivenAttention(embed_dim=16, num_heads=Hm, mode="bitwise")
        g = torch.Generator().manual_seed(7)
        mask = att.attention_mask(2.0 * torch.randn(Tm, Bm, Nm, 16, generator=g))

        ours_bytes = mask.numel() * mask.element_size()  # bool: 1 B/entry
        softmax_baseline = Tm * Bm * Hm * Nm * Nm * 4  # float32 Softmax(QKᵀ/√d)
        assert ours_bytes == softmax_baseline / 4, (
            f"association map ({ours_bytes}B) must be exactly 25% of "
            f"float softmax ({softmax_baseline}B)"
        )


class TestDeterminism:
    def test_deterministic_given_seed(self) -> None:
        att = make_attention()
        g = torch.Generator().manual_seed(11)
        x1 = 2.0 * torch.randn(T, B, N, C, generator=g)
        g = torch.Generator().manual_seed(11)
        x2 = 2.0 * torch.randn(T, B, N, C, generator=g)
        assert torch.equal(att(x1), att(x2))


class TestAttentionThreshold:
    """Dedicated tests for attention threshold theta_a (Plan 2, Section 3)."""

    def test_threshold_buffer_storage_and_no_grad(self) -> None:
        att = SpikeDrivenAttention(embed_dim=C, num_heads=H, attn_threshold=2)
        assert att.attn_th.item() == 2.0
        assert not att.attn_th.requires_grad
        assert "attn_th" in dict(att.named_buffers())

    def test_threshold_selectivity(self) -> None:
        """Higher theta_a requires more overlapping spike channels to fire."""
        att1 = SpikeDrivenAttention(embed_dim=16, num_heads=1, attn_threshold=1, mode="bitwise")
        att2 = SpikeDrivenAttention(embed_dim=16, num_heads=1, attn_threshold=3, mode="bitwise")

        # Construct specific Q and K with known overlap count = 2
        q = torch.zeros(1, 1, 1, 1, 16, dtype=torch.bool)
        k = torch.zeros(1, 1, 1, 1, 16, dtype=torch.bool)
        v = torch.ones(1, 1, 1, 1, 16, dtype=torch.bool)
        q[..., :2] = True
        k[..., :2] = True  # overlap = 2

        # In att1 (theta_a=1 <= 2), mask fires
        _, a1 = att1._attend_bitwise(q, k, v)
        assert a1.item() is True

        # In att2 (theta_a=3 > 2), mask does not fire
        _, a2 = att2._attend_bitwise(q, k, v)
        assert a2.item() is False

    def test_surrogate_and_bitwise_parity_under_custom_threshold(self) -> None:
        for th in [1, 2, 4]:
            att_s = SpikeDrivenAttention(
                embed_dim=C, num_heads=H, attn_threshold=th, mode="surrogate"
            )
            att_b = SpikeDrivenAttention(
                embed_dim=C, num_heads=H, attn_threshold=th, mode="bitwise"
            )
            att_b.W_q.weight.data.copy_(att_s.W_q.weight.data)
            att_b.W_k.weight.data.copy_(att_s.W_k.weight.data)
            att_b.W_v.weight.data.copy_(att_s.W_v.weight.data)
            x = drive()
            assert torch.equal(att_s(x), att_b(x))
