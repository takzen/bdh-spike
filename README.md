# ⚡ BDH-Spike

> **Native Spiking Neural Network (SNN) implementation of the Baby Dragon Hatchling (BDH) architecture** — built for Neuromorphic & Edge AI with Continual Learning.

`BDH-Spike` bridges the theoretical bio-physical properties of the **Baby Dragon Hatchling (BDH)** architecture with **discrete event-driven Spiking Neural Networks**. No floating-point activations inside the core layers. No global backpropagation during inference. Just spikes.

---

## 🔥 Core Philosophy

| Invariant | Rule |
|---|---|
| **Binary Spike Domain** | All activations in core layers are discrete spikes $S(t) \in \{0, 1\}$. No intermediate continuous floats inside spike-driven modules. |
| **Dual-Weight Plasticity** | $W_{\text{slow}}$ (structural) trained offline via BPTT + surrogate gradients; $W_{\text{fast}}$ (episodic/Hebbian) updated online via local STDP — zero backprop at inference. |
| **Temporal Canonical Shape** | All temporal tensors maintain $[T, B, C, \dots]$ — *(Time-steps, Batch, Channels)*. Never $[B, T, C]$. |
| **Energy Sparsity First** | Target temporal sparsity of **85–95%** silent states. AC and bitwise ops instead of MACs wherever possible. |

---

## 🧮 Mathematical Reference Card

**BDH-PLIF Membrane Decay**

$$V[t] = \beta V[t-1] + I_{\text{syn}}[t] + M_{\text{BDH}}[t-1] - S[t-1]\,V_{\text{th}}$$

**Spike Trigger**

$$S[t] = \Theta\big(V[t] - V_{\text{th}}\big)$$

**Fast Sigmoid Surrogate Gradient** (default slope $k = 25$)

$$\sigma'(x) = \frac{1}{(1 + k|x - V_{\text{th}}|)^2}$$

**STDP Trace Decay** (3-factor Hebbian learning, fully local)

$$A_{\text{pre}}[t] = A_{\text{pre}}[t-1] \cdot e^{-\Delta t / \tau_{\text{pre}}} + S_{\text{pre}}[t]$$

---

## ⚙️ SOPs vs FLOPs

Spiking networks replace dense Multiply-Accumulate (MAC) streams with sparse Accumulate (AC) events. Synaptic Operations scale with the *number of active spikes*, not with tensor density:

$$\text{SOPs} = \sum_{t=1}^{T} \text{nnz}(S_{\text{in}}[t]) \times \text{FanOut}$$

| Metric | Dense ANN | BDH-Spike |
|---|---|---|
| Core operation | MAC (32-bit float) | AC / POPCNT (bitwise) |
| Activation cost | Every neuron, every step | Only emitting neurons (~5–15%) |
| Memory traffic | Full weight matrix per step | Active rows only |
| Inference learning | ❌ Frozen weights | ✅ Online STDP ($W_{\text{fast}}$) |
| Target hardware | GPU / TPU | Neuromorphic (Loihi 2) / Edge MCU |

*(Benchmark numbers land here as stages 6–7 of the roadmap complete.)*

---

## 📦 Installation

Requirements: **Python 3.13+** · managed with [uv](https://docs.astral.sh/uv/) · PyTorch 2.x

```bash
# clone & create the environment from the lockfile
git clone https://github.com/takzen/bdh-spike.git && cd bdh-spike
uv sync --dev
```

Optional — compile custom CUDA kernels (fused PLIF dynamics, bitwise POPCNT matmul):

```bash
uv pip install -e . --no-build-isolation --config-settings "--build-option=--cuda"
```

---

## 🚀 Quick Start

> ⏳ **Work in progress.** The repository is currently at the environment-scaffolding stage.
> First runnable neurons (`bdh_spike/core/neuron.py`) arrive with Stage 2 of the roadmap below.

---

## 🗺️ Roadmap

- [x] **Stage 1** — Repository & environment initialization (uv, Python 3.13, pyproject)
- [ ] **Stage 2** — Neuromorphic core: `BDHSpikeCell` (PLIF decay, hard reset, fast-sigmoid surrogate, BDH recurrent coupling)
- [ ] **Stage 3** — Spike-driven attention: softmax-free associative masking on binary spikes
- [ ] **Stage 4** — Dual plasticity engine: online STDP + homeostatic threshold adaptation (anti catastrophic-forgetting)
- [ ] **Stage 5** — Model assembly: Vision-BDH-Spike backbone + streaming sequence model + spike encoders
- [ ] **Stage 6** — Energy metrics & benchmarks: SOPs vs FLOPs tracker, N-MNIST eval, continual-learning split-task test
- [ ] **Stage 7** — Diagnostics: spike raster plots, membrane potential traces, telemetry HUD
- [ ] **Stage 8** — Documentation, full test pass, public release

---

## 🧪 Testing & Quality Gates

```bash
uv run pytest tests/ -v --durations=10   # unit tests must pass 100%
uv run ruff check . --fix && uv run ruff format .   # lint & format
```

Every completed stage additionally verifies a **> 70% spike sparsity** assertion via `bdh_spike.neuromorphic.metrics.calculate_sparsity`.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![uv](https://img.shields.io/badge/Managed%20with-uv-DE5FE9)
![snnTorch](https://img.shields.io/badge/snnTorch-SNN%20ecosystem-8A2BE2)
![SpikingJelly](https://img.shields.io/badge/SpikingJelly-SNN%20ecosystem-4B0082)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)

---

_Generated for TakzenAI // Sovereign Neuromorphic Architecture._
