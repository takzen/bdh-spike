"""Spiking Vision-BDH backbone (Stage 5).

End-to-end multiply-frugal vision pipeline assembling every prior stage::

    image [B,C,H,W]
      ──► PatchEmbed (conv, W_slow)          → token currents [B,N,E]
      ──► rate_encode                        → spike train   [T,B,N,E]
      ──► SpikeDrivenAttention (softmax-free)→ attention out [T,B,N,E]
      ──► BDH-PLIF cell over time            → output spikes [T,B,N,E]
      ──► temporal mean (firing rates)       → rates         [B,N,E]
      ──► mean-pool over tokens + Linear head → logits       [B,num_classes]

Every activation between the patch projection and the classifier head lives in
the binary spike domain; gradients reach the conv/projection weights only via
the fast-sigmoid surrogate junctions.
"""

from __future__ import annotations

import torch
from torch import nn

from bdh_spike.core.attention import SpikeDrivenAttention
from bdh_spike.core.functional import spike_fn
from bdh_spike.core.neuron import BDHSpikeCell

__all__ = ["BDHSpikeViT", "PatchEmbed"]


class PatchEmbed(nn.Module):
    """Convolutional patch projection: image grid → token current vectors.

    Args:
        in_channels: Input image channels ``C_img``.
        embed_dim: Token embedding size ``E``.
        patch_size: Square patch side ``P`` (image sides must be divisible).
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 64, patch_size: int = 7) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )

    def num_patches(self, img_size: int) -> int:
        """Number of tokens for a square image of side ``img_size``."""
        if img_size % self.patch_size != 0:
            raise ValueError(f"img_size={img_size} not divisible by patch_size={self.patch_size}")
        return (img_size // self.patch_size) ** 2

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``[B, C_img, H, W] → [B, N, E]`` token currents."""
        if images.dim() != 4:
            raise ValueError(f"Expected images of shape [B, C, H, W], got {tuple(images.shape)}")
        tokens = self.proj(images)
        b, e, gh, gw = tokens.shape
        return tokens.reshape(b, e, gh * gw).transpose(1, 2)


class BDHSpikeViT(nn.Module):
    """Vision-BDH-Spike: spiking transformer-style classifier.

    Args:
        img_size: Square input image side.
        patch_size: Square patch side.
        in_channels: Input image channels.
        num_classes: Classifier head outputs.
        embed_dim: Token embedding size (must be divisible by ``num_heads``).
        num_heads: Spike-driven attention heads.
        num_steps: Temporal window ``T`` for rate encoding and PLIF dynamics.
        use_attention: Toggle the softmax-free attention stage.
        attn_threshold / mode: Forwarded to :class:`SpikeDrivenAttention`.
        beta_init / v_th / surrogate_slope: PLIF neuron hyper-parameters.
    """

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        num_classes: int = 10,
        embed_dim: int = 64,
        num_heads: int = 2,
        num_steps: int = 16,
        use_attention: bool = True,
        attn_threshold: int = 1,
        mode: str = "surrogate",
        beta_init: float = 0.9,
        v_th: float = 1.0,
        surrogate_slope: float = 25.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} not divisible by num_heads={num_heads}")
        if img_size % patch_size != 0:
            raise ValueError(f"img_size={img_size} not divisible by patch_size={patch_size}")

        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.num_steps = int(num_steps)
        self.use_attention = bool(use_attention)

        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        self.attention = (
            SpikeDrivenAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_threshold=attn_threshold,
                mode=mode,
            )
            if use_attention
            else None
        )
        self.cell = BDHSpikeCell(
            num_channels=embed_dim,
            beta_init=beta_init,
            v_th=v_th,
            surrogate_slope=surrogate_slope,
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def _encode_tokens(self, currents: torch.Tensor) -> torch.Tensor:
        """Convert signed token currents into a binary ``[T, B, N, E]`` train.

        Signed conv outputs are min-max normalized per sample into ``[0, 1]``
        firing levels, held constant across the window (steady-rate code), and
        binarized through the fast-sigmoid surrogate junction so BPTT gradients
        still reach the patch-projection weights.
        """
        t = self.num_steps
        lo = currents.amin(dim=(1, 2), keepdim=True)
        hi = currents.amax(dim=(1, 2), keepdim=True)
        level = (currents - lo) / (hi - lo + 1e-8)
        drive = level.unsqueeze(0).expand(t, *level.shape)
        return spike_fn(drive - 0.5, threshold=0.5, slope=self.cell.surrogate_slope)

    def features(self, images: torch.Tensor) -> torch.Tensor:
        """Spike firing-rate features ``[B, N, E]`` (pre-pooling readout)."""
        currents = self.patch_embed(images)  # [B, N, E]
        spikes_in = self._encode_tokens(currents)  # [T, B, N, E] binary

        hidden = spikes_in
        if self.attention is not None:
            hidden = self.attention(hidden)  # [T, B, N, E] binary

        t, b, n, c = hidden.shape
        out, _ = self.cell(hidden.reshape(t, b * n, c))  # PLIF over folded batch
        return out.mean(dim=0).view(b, n, c)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Classify a batch of images.

        Args:
            images: ``[B, C, H, W]`` with ``H == W == img_size``.

        Returns:
            Class logits ``[B, num_classes]``.
        """
        if (
            images.dim() != 4
            or images.shape[-1] != self.img_size
            or images.shape[-2] != self.img_size
        ):
            raise ValueError(
                f"Expected images of shape [B, C, {self.img_size}, {self.img_size}], "
                f"got {tuple(images.shape)}"
            )
        rates = self.features(images)  # [B, N, E]
        pooled = rates.mean(dim=1)  # token mean-pool → [B, E]
        return self.head(pooled)
