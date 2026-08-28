# ⚡ BDH-Spike

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22139050.svg)](https://doi.org/10.5281/zenodo.22139050)

> **Native Spiking Neural Network (SNN) implementation of the Baby Dragon Hatchling (BDH) architecture** — built for Neuromorphic & Edge AI with Continual Learning.

`BDH-Spike` bridges the theoretical bio-physical properties of the **Baby Dragon Hatchling (BDH)** architecture with **discrete event-driven Spiking Neural Networks**. No floating-point activations inside the core layers. No global backpropagation during inference. Just spikes.

## 🛠️ Tech Stack

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22139050.svg)](https://doi.org/10.5281/zenodo.22139050)
![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![uv](https://img.shields.io/badge/Managed%20with-uv-DE5FE9)
![snnTorch](https://img.shields.io/badge/snnTorch-SNN%20ecosystem-8A2BE2)
![SpikingJelly](https://img.shields.io/badge/SpikingJelly-SNN%20ecosystem-4B0082)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

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

**Measured multi-seed ablation study** (`python -m benchmarks.ablation_study --seeds 0 1 2`, $N=3$ seeds, mean $\pm$ std):

| Ablation Variant | Configuration | Sparsity (%) | SOPs / sample | FLOPs/SOP Ratio | CL Forgetting $F \downarrow$ |
|---|---|:---:|:---:|:---:|:---:|
| **A: Full BDH-Spike** | $g=0.5, W_{\text{fast}}=\text{ON}, V_{\text{th}}=\text{adapt}$ | **$83.7 \pm 3.0\%$** | $2.00\text{M} \pm 0.06\text{M}$ | **$10.03 \pm 0.29\times$** | **$0.317 \pm 0.076$** |
| **B: No BDH Coupling** | $g=0.0, W_{\text{fast}}=\text{ON}, V_{\text{th}}=\text{adapt}$ | $82.5 \pm 7.5\%$ | $2.00\text{M} \pm 0.06\text{M}$ | $10.03 \pm 0.29\times$ | $0.387 \pm 0.200$ |
| **C: No $W_{\text{fast}}$** | $g=0.5, W_{\text{fast}}=\text{OFF}, V_{\text{th}}=\text{adapt}$ | $83.7 \pm 3.0\%$ | $2.00\text{M} \pm 0.06\text{M}$ | $10.03 \pm 0.29\times$ | $0.285 \pm 0.214$ |
| **D: No Homeostasis** | $g=0.5, W_{\text{fast}}=\text{ON}, V_{\text{th}}=\text{const}$ | $83.4 \pm 0.3\%$ | $2.00\text{M} \pm 0.06\text{M}$ | $10.03 \pm 0.29\times$ | $0.530 \pm 0.084$ |
| **E: Bitwise vs Surrogate** | Float-free vs surrogate graph | — | — | — | **$100\%$ exact match** |

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

## ⏱️ Run in 3 Lines

```bash
git clone https://github.com/takzen/bdh-spike.git && cd bdh-spike && uv sync --dev
uv run pytest tests/ -q                                   # 123 tests, 100% green
uv run python -m benchmarks.n_mnist_eval --dataset synthetic   # energy report
```

---

## 🚀 Quick Start

```python
import torch
from bdh_spike.core import BDHSpikeCell

cell = BDHSpikeCell(num_channels=64)
spikes, state = cell(torch.randn(16, 4, 64))  # [T, B, C] -> binary spikes [T, B, C]
```

Every emitted activation is a discrete spike $S(t) \in \{0, 1\}$; learning flows through the fast-sigmoid surrogate gradient only.

Spike-driven attention — no Softmax, just associative masking on binary spikes:

```python
import torch
from bdh_spike.core import SpikeDrivenAttention

attn = SpikeDrivenAttention(embed_dim=64, num_heads=4)
y = attn(torch.randn(16, 4, 32, 64))  # [T, B, N, C] -> binary spikes [T, B, N, C]

# Deployment: strictly boolean-integer graph — zero float tensors after encoding.
hw = SpikeDrivenAttention(embed_dim=64, num_heads=4, mode="bitwise")
```

Dual-weight plasticity — structural weights by BPTT, episodic weights online by local STDP (never touching autograd):

```python
from bdh_spike.plasticity import DualWeightLinear

syn = DualWeightLinear(fan_in=64, fan_out=32)
current = syn(spikes_in)  # differentiable via W_slow
state = syn.plastic_step(spikes_in, spikes_out)  # grad-free W_fast update
```

Homeostatic threshold adaptation keeps firing in the healthy band:

```python
from bdh_spike.plasticity import AdaptiveThreshold

homeo = AdaptiveThreshold(target_rate=0.10)
homeo.observe_sequence(output_spikes)  # seizure → V_th ↑ ; silence → V_th ↓
homeo.apply_to(cell)  # inject into BDHSpikeCell.v_th
```

Terminal telemetry & figures:

```python
from bdh_spike.utils import TelemetryRecorder, ascii_raster

print(ascii_raster(spikes))  # text raster: █ = spike, · = silence
rec = TelemetryRecorder(fan_out=32)
rec.update(spikes)
rec.render()  # terminal HUD
rec.dump("telemetry.json")  # web-dashboard export
```

---

## 🗺️ Roadmap

- [x] **Stage 1** — Repository & environment initialization (uv, Python 3.13, pyproject)
- [x] **Stage 2** — Neuromorphic core: `BDHSpikeCell` (PLIF decay, hard reset, fast-sigmoid surrogate, BDH recurrent coupling)
- [x] **Stage 3** — Spike-driven attention: softmax-free associative masking on binary spikes
- [x] **Stage 4** — Dual plasticity engine: online STDP + homeostatic threshold adaptation (continual learning stability)
- [x] **Stage 5** — Model assembly: Vision-BDH-Spike backbone + streaming sequence model + spike encoders
- [x] **Stage 6** — Energy metrics & benchmarks: SOPs vs FLOPs tracker, N-MNIST eval, continual-learning split-task test
- [x] **Stage 7** — Diagnostics: spike raster plots, membrane potential traces, telemetry HUD
- [x] **Stage 8** — Documentation, full test pass, public release

---

## 🐉 Continual Learning & Plasticity

Split-task benchmark (`python -m benchmarks.ablation_study --seeds 0 1 2`, 5 sequential tasks on a shared hidden layer). Structural weights are optimized per task; episodic `W_fast` adapts online via local STDP while each task stream flows through without global autograd.

Ablation results demonstrate that **homeostatic threshold adaptation ($V_{\text{th}}$)** provides the primary stabilization against catastrophic forgetting ($F = 0.317 \pm 0.076$ vs $0.530 \pm 0.084$ without homeostasis), with BDH coupling contributing to reduced variance across seeds.

---

## 🧪 Testing & Quality Gates

```bash
uv run pytest tests/ -v --durations=10   # unit tests must pass 100%
uv run ruff check . --fix && uv run ruff format .   # lint & format
```

Every completed stage additionally verifies a **> 70% spike sparsity** assertion via `bdh_spike.neuromorphic.metrics.calculate_sparsity`.

---

## 📄 License

Released under the [MIT License](LICENSE).

---

**Krzysztof Pika**
