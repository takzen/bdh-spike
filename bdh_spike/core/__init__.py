"""Core BDH-Spike primitives: neurons, surrogate gradients and spike-driven attention."""

from bdh_spike.core.attention import SpikeDrivenAttention, associate_spikes
from bdh_spike.core.functional import FastSigmoidSurrogate, spike_fn
from bdh_spike.core.neuron import BDHSpikeCell, BDHState, iterate_steps

__all__ = [
    "BDHSpikeCell",
    "BDHState",
    "FastSigmoidSurrogate",
    "SpikeDrivenAttention",
    "associate_spikes",
    "iterate_steps",
    "spike_fn",
]
