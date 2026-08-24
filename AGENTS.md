# AGENTS.md // BDH-Spike Operational & Architectural Protocol

> **Target Audience:** Autonomous AI coding agents (Claude Code, Cursor, Aider, OpenHands) and human contributors.
> **Repository Scope:** Native Spiking Neural Network (SNN) implementation of the Baby Dragon Hatchling (BDH) architecture for Neuromorphic & Edge AI with Continual Learning.

---

## 1. System Mission & Core Philosophy

`BDH-Spike` bridges the theoretical bio-physical properties of the **Baby Dragon Hatchling (BDH)** architecture with **discrete event-driven Spiking Neural Networks (SNNs)**.

### Fundamental Invariants

1. **Binary Spike Domain:** All activations within core layers must be discrete spikes: $S(t) \in \{0, 1\}$. Do not introduce intermediate continuous float activations inside spike-driven modules.
2. **Dual-Weight Plasticity:**
   - **$W_{\text{slow}}$ (Structural Weights):** Optimized offline via BPTT with **Surrogate Gradients** (`fast_sigmoid` / `atan`).
   - **$W_{\text{fast}}$ (Episodic / Hebbian Weights):** Updated online during inference via local **STDP (3-factor Hebbian Plasticity)** with zero global backpropagation.
3. **Temporal Dimension Invariant:** All temporal sequence tensors **MUST** maintain the canonical shape:

   $$\text{Shape: } [T, B, C, \dots] \quad (\text{Time-steps}, \text{Batch}, \text{Channels/Features})$$

   _Agents must not switch to $[B, T, C]$ without explicit transpose layers._

4. **Energy Sparsity First:** Algorithms must maximize temporal sparsity ($S(t) \approx 0$ for 85–95% of states). Minimize MAC (Multiply-Accumulate) operations in favor of AC (Accumulate) and bitwise operations.

---

## 2. Directory Layout & Module Responsibilities

```text
bdh-spike/
├── AGENTS.md                  # This protocol
├── pyproject.toml             # Build system & metadata
├── csrc/                      # High-performance C++/CUDA extensions
│   ├── bdh_plif_kernel.cu     # Fused PLIF membrane dynamics kernel
│   └── spike_matmul.cu        # Bitwise boolean POPCNT matrix multiplication
├── bdh_spike/
│   ├── __init__.py
│   ├── core/
│   │   ├── neuron.py          # BDH-PLIF (Parametric LIF with lateral BDH coupling)
│   │   ├── functional.py      # Surrogate gradient definitions & math primitives
│   │   └── attention.py       # Spike-Driven BDH Attention (Softmax-free)
│   ├── plasticity/
│   │   ├── stdp.py            # Local Spike-Timing-Dependent Plasticity engine
│   │   └── homeostat.py       # Adaptive membrane threshold regulation (V_th dynamic)
│   ├── models/
│   │   ├── bdh_spike_vit.py   # Spiking Vision-BDH backbone
│   │   └── bdh_spike_seq.py   # Recurrent sequence model for streaming data
│   ├── neuromorphic/
│   │   ├── lava_export.py     # Intel Loihi 2 / Lava mapping routines
│   │   └── metrics.py         # SOPs (Synaptic Operations) vs FLOPs calculator
│   └── utils/
│       ├── encoders.py        # Rate / Latency / Delta spike encoding
│       └── visualizer.py      # Spike raster plot & membrane potential traces
├── benchmarks/
│   ├── n_mnist_eval.py        # Event camera benchmark
│   └── continual_learning.py  # Split-task catastrophic forgetting test
└── tests/
    ├── test_neurons.py        # Unit tests for membrane decay and resets
    ├── test_plasticity.py     # STDP weight update verification
    └── test_cuda_parity.py    # PyTorch vs CUDA kernel numerical parity
```

---

## 3. Tech Stack & Environment

- **Language:** Python 3.10+ & C++17 / CUDA 12.x
- **Deep Learning Framework:** PyTorch >= 2.2
- **SNN Ecosystem:** `snnTorch`, `SpikingJelly`
- **Testing & Quality:** `pytest`, `pytest-cov`, `ruff`, `mypy`

### Environment Setup Commands

```bash
# Clone & install dependencies in editable mode with dev tools
pip install -e ".[dev]"

# Optional: compile custom CUDA kernels
pip install -e . --no-build-isolation --config-settings "--build-option=--cuda"
```

---

## 4. Agent Rules of Engagement

When generating or refactoring code, agents MUST follow these strict rules:

### A. Implementing Neurons & Layers

- Always provide a surrogate gradient on the spiking threshold to ensure backward pass differentiability.
- Always handle membrane reset cleanly (either `reset_by_subtraction` or `hard_reset` to `0.0`).
- Maintain neuron state encapsulation: states (`mem`, `m_bdh`) must be initialized with an explicit helper method (e.g., `init_hidden(batch_size)`).

### B. Modifying Plasticity & Learning Rules

- Never calculate gradients (`requires_grad=True`) for online STDP updates. STDP runs with `torch.no_grad()`.
- STDP trace updates must follow exponential decay:

  $$A_{\text{pre}}[t] = A_{\text{pre}}[t-1] \cdot e^{-\Delta t / \tau_{\text{pre}}} + S_{\text{pre}}[t]$$

### C. Testing & Verification Requirements

Before declaring any task complete, verify:

- **Unit Tests:** `pytest tests/` passes with 100% success.
- **Sparsity Check:** Run `bdh_spike.neuromorphic.metrics.calculate_sparsity(output_spikes)` and assert sparsity $> 70\%$.
- **No Gradient Leak:** Ensure $W_{\text{fast}}$ updates do not pollute PyTorch's grad graph.
- **Code Quality:** Format with `ruff check . --fix` and `ruff format .`.

---

## 5. Standard CLI Operations

### Run Full Test Suite

```bash
pytest tests/ -v --durations=10
```

### Run Neuromorphic Energy & Sparsity Benchmark

```bash
python -m benchmarks.n_mnist_eval --epochs 5 --time-steps 16 --device cuda
```

### Run Continual Learning Catastrophic Forgetting Test

```bash
python -m benchmarks.continual_learning --tasks 5 --eval-method forgetting_ratio
```

---

## 6. Mathematical Reference Card

**LIF Decay:**

$$V[t] = \beta V[t-1] + I_{\text{syn}}[t] + M_{\text{BDH}}[t-1] - S[t-1]\,V_{\text{th}}$$

**Spike Trigger:**

$$S[t] = \Theta\big(V[t] - V_{\text{th}}\big)$$

**Fast Sigmoid Surrogate** (default slope $k = 25$):

$$\sigma'(x) = \frac{1}{(1 + k|x - V_{\text{th}}|)^2}$$

**SOPs Calculation:**

$$\text{SOPs} = \sum_{t=1}^{T} \text{nnz}(S_{\text{in}}[t]) \times \text{FanOut}$$

---

_Generated for TakzenAI // Sovereign Neuromorphic Architecture._
