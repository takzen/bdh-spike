"""Core BDH-Spike primitives: neurons, surrogate gradients and spike-driven attention."""

from bdh_spike.core.functional import FastSigmoidSurrogate, spike_fn
from bdh_spike.core.neuron import BDHSpikeCell, BDHState, iterate_steps

__all__ = [
    "BDHSpikeCell",
    "BDHState",
    "FastSigmoidSurrogate",
    "iterate_steps",
    "spike_fn",
]
