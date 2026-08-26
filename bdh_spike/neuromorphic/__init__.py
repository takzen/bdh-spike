"""Neuromorphic deployment utilities: Loihi/Lava export and energy metrics."""

from bdh_spike.neuromorphic.metrics import (
    SOPSMeter,
    SpikeSparsityTracker,
    calculate_sparsity,
    flops_dense,
    sops_count,
)

__all__ = [
    "SOPSMeter",
    "SpikeSparsityTracker",
    "calculate_sparsity",
    "flops_dense",
    "sops_count",
]
