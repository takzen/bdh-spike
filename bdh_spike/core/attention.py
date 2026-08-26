"""Spike-Driven BDH Attention: softmax-free attention on binary spikes.

Replaces the classic float ``Softmax(Q·Kᵀ/√d)·V`` pipeline with sparse
associative masking on binary spikes:

    Q_s = Θ(X·W_q − V_th)     K_s = Θ(X·W_k − V_th)     V_s = Θ(X·W_v − V_th)
    A   = Q_s ∧ K_s           (associative AND-mask — no Softmax, no √d scaling)
    Y   = Θ(A · V_s − V_out)  (output spiking neuron re-binarizes the result)

Every value past the synaptic projections lives in the discrete spike domain.
Temporal tensors keep the canonical [T, B, N, C] layout (time-steps first).

Two execution modes:

* ``mode="surrogate"`` — BPTT-trainable. Every forward value is a binary
  {0, 1} spike, but tensors are stored as float so the fast-sigmoid
  surrogate can carry gradients back into the synaptic projections
  (mirroring how the PLIF neuron keeps a float membrane behind binary
  spikes).
* ``mode="bitwise"``   — inference / neuromorphic deployment. A strictly
  boolean-integer graph (AND-masking + exact uint8/int16 accumulation):
  zero floating-point tensors are materialized after the input projections,
  so the module maps 1:1 onto POPCNT / AC event hardware (Loihi-style).
"""

from __future__ import annotations

import torch
from torch import nn

from bdh_spike.core.functional import spike_fn

__all__ = ["SpikeDrivenAttention", "associate_spikes"]

# Association-count accumulator: uint8 is exact for head widths up to 255
# (max possible overlap == head_dim), int16 covers anything larger.
_UINT8_MAX_HEAD_DIM = 255


def _acc_dtype(head_dim: int) -> torch.dtype:
    """Integer accumulator dtype for exact association counting."""
    return torch.uint8 if head_dim <= _UINT8_MAX_HEAD_DIM else torch.int16


def associate_spikes(q_spikes: torch.Tensor, k_spikes: torch.Tensor) -> torch.Tensor:
    """Associative overlap counts between binary query/key spike tensors.

    Computes ``counts[n, m] = Σ_c Q_s[n,c] ∧ K_s[m,c]`` using integer
    accumulate (AC) semantics — no multiplies, no Softmax, no √d scaling.

    Args:
        q_spikes: Binary queries ``[..., N, Cc]``.
        k_spikes: Binary keys    ``[..., M, Cc]``.

    Returns:
        Integer overlap counts ``[..., N, M]`` (uint8 or int16, never float).
    """
    if q_spikes.dtype != torch.bool or k_spikes.dtype != torch.bool:
        raise TypeError("associate_spikes expects boolean spike tensors")
    if q_spikes.shape[-1] != k_spikes.shape[-1]:
        raise ValueError(f"Q/K channel mismatch: {q_spikes.shape[-1]} != {k_spikes.shape[-1]}")
    dtype = _acc_dtype(q_spikes.shape[-1])
    return q_spikes.to(dtype) @ k_spikes.to(dtype).transpose(-1, -2)


class SpikeDrivenAttention(nn.Module):
    """Multiply-free spiking self-attention (softmax-free BDH attention).

    Args:
        embed_dim: Token embedding size ``C`` (must be divisible by heads).
        num_heads: Number of parallel attention heads.
        in_v_th: Firing threshold of the Q/K/V input spiking neurons.
        attn_threshold: Minimal Q∧K channel-overlap that opens a mask entry.
        out_v_th: Threshold of the output spiking neuron.
        surrogate_slope: Fast-sigmoid surrogate slope ``k`` (default 25).
        mode: ``"surrogate"`` (BPTT-trainable) or ``"bitwise"`` (float-free
            inference graph).
    """

    in_v_th: torch.Tensor
    attn_th: torch.Tensor
    out_v_th: torch.Tensor

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        in_v_th: float = 1.0,
        attn_threshold: int = 1,
        out_v_th: float = 1.0,
        surrogate_slope: float = 25.0,
        mode: str = "surrogate",
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} not divisible by num_heads={num_heads}")
        if mode not in {"surrogate", "bitwise"}:
            raise ValueError(f"mode must be 'surrogate' or 'bitwise', got {mode!r}")

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.surrogate_slope = float(surrogate_slope)
        self.mode = mode

        # Synaptic projections (float weights are structural W_slow territory;
        # every activation they feed is binarized immediately afterwards).
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # Thresholds as buffers: homeostat-adaptable, excluded from grads.
        self.register_buffer("in_v_th", torch.tensor(float(in_v_th)))
        self.register_buffer("attn_th", torch.tensor(float(attn_threshold)))
        self.register_buffer("out_v_th", torch.tensor(float(out_v_th)))

    @property
    def head_dim_is_uint8_safe(self) -> bool:
        return self.head_dim <= _UINT8_MAX_HEAD_DIM

    # ------------------------------------------------------------------ #
    # Encoding: synaptic current -> binary spikes (surrogate junction).  #
    # ------------------------------------------------------------------ #
    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project and binarize tokens into Q/K/V spike tensors ``[T,B,H,N,Cc]``.

        Returns float {0, 1} tensors so the fast-sigmoid surrogate can carry
        gradients back into ``W_q/W_k/W_v``; values are strictly binary.
        """
        q = spike_fn(self.W_q(x), threshold=self.in_v_th, slope=self.surrogate_slope)
        k = spike_fn(self.W_k(x), threshold=self.in_v_th, slope=self.surrogate_slope)
        v = spike_fn(self.W_v(x), threshold=self.in_v_th, slope=self.surrogate_slope)

        T, B, N, _ = x.shape
        q = q.view(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = k.view(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = v.view(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        return q, k, v

    def _merge_heads(self, y: torch.Tensor, batch: int, steps: int, tokens: int) -> torch.Tensor:
        """Merge heads ``[T,B,H,N,Cc] -> [T,B,N,C]`` preserving [T, B, ...]."""
        return y.permute(0, 1, 3, 2, 4).reshape(steps, batch, tokens, self.embed_dim)

    # ------------------------------------------------------------------ #
    # Surrogate path: binary-valued floats + surrogate junctions.        #
    # ------------------------------------------------------------------ #
    def _attend_surrogate(
        self, q_s: torch.Tensor, k_s: torch.Tensor, v_s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Forward values stay binary {0,1} end-to-end; the float dtype exists
        # solely to carry fast-sigmoid gradients back to W_q/W_k/W_v.
        counts = q_s @ k_s.transpose(-1, -2)  # overlap counts of binary spikes
        a = spike_fn(counts, threshold=self.attn_th, slope=self.surrogate_slope)
        out_pot = a @ v_s
        y = spike_fn(out_pot, threshold=self.out_v_th, slope=self.surrogate_slope)
        return y, a

    # ------------------------------------------------------------------ #
    # Bitwise path: strictly boolean-integer graph, zero float tensors.  #
    # ------------------------------------------------------------------ #
    def _attend_bitwise(
        self, q_s: torch.Tensor, k_s: torch.Tensor, v_s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts = associate_spikes(q_s.bool(), k_s.bool())
        a = counts >= self.attn_th.to(counts.dtype)  # boolean AND-mask
        acc = _acc_dtype(self.head_dim)
        out_counts = a.to(acc) @ v_s.bool().to(acc)
        y = out_counts >= self.out_v_th.to(out_counts.dtype)  # boolean spikes
        return y, a

    def attention_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Binary associative mask ``A`` of shape ``[T, B, H, N, N]``.

        Boolean in ``bitwise`` mode; float {0,1} (surrogate-carrying) in
        ``surrogate`` mode.
        """
        q_s, k_s, _ = self._encode(x)
        _, a = self._dispatch_attend(q_s, k_s, q_s)
        return a

    def _dispatch_attend(
        self, q_s: torch.Tensor, k_s: torch.Tensor, v_s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "bitwise":
            return self._attend_bitwise(q_s, k_s, v_s)
        return self._attend_surrogate(q_s, k_s, v_s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run spike-driven attention over a token sequence.

        Args:
            x: Input tokens in canonical temporal layout ``[T, B, N, C]``.

        Returns:
            Binary output spikes ``[T, B, N, C]`` with values in {0, 1}.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected input of shape [T, B, N, C], got {tuple(x.shape)}")
        T, B, N, C = x.shape
        if C != self.embed_dim:
            raise ValueError(f"Expected channel dim C={self.embed_dim}, got {C}")

        q_s, k_s, v_s = self._encode(x)
        y, _ = self._dispatch_attend(q_s, k_s, v_s)
        return self._merge_heads(y.to(torch.float), batch=B, steps=T, tokens=N)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(embed_dim={self.embed_dim}, "
            f"heads={self.num_heads}, mode={self.mode!r}, "
            f"in_v_th={float(self.in_v_th):.3f}, attn_th={float(self.attn_th):.1f}, "
            f"out_v_th={float(self.out_v_th):.3f})"
        )
