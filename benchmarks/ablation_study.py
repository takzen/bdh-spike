"""Ablation study suite for BDH-Spike (Plan 2, Sections 4, 5, 6, 15).

Evaluates 5 architectural variants under identical seeds and hyperparameter configurations:
  Variant A: Full Model (BDH-Spike with lateral coupling, W_fast STDP, homeostasis)
  Variant B: No BDH Coupling (PLIF without m_BDH lateral coupling, g=0)
  Variant C: No W_fast (BPTT-only structural learning, W_fast adaptation disabled)
  Variant D: No Homeostasis (fixed static V_th threshold)
  Variant E: Bitwise Attention vs Surrogate Attention (inference graph mode comparison)

Measures:
  - Event Stream Classification: Accuracy (%), Spike Sparsity (%), SOPs, Dense FLOPs, FLOPs/SOP Ratio
  - Split-Task Continual Learning: Initial Acc (%), Final Acc (%), Forgetting Ratio F
  - Multi-seed aggregation: Mean ± Std over seeds (default seeds: [0, 1, 2])
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from bdh_spike.models import BDHSpikeSeq, BDHSpikeViT
from bdh_spike.neuromorphic.metrics import SOPSMeter, SpikeSparsityTracker, calculate_sparsity
from benchmarks.continual_learning import CLModel, forgetting_ratio, make_tasks
from benchmarks.n_mnist_eval import EventClassifier, evaluate, make_loader, synthetic_events


@dataclass
class ExperimentConfig:
    """Central reproducible configuration for benchmarks & ablation studies."""

    # General
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Event classification benchmark
    event_epochs: int = 5
    event_time_steps: int = 16
    event_batch_size: int = 64
    event_lr: float = 2e-3
    event_num_classes: int = 10
    event_n_train: int = 1024
    event_n_test: int = 256

    # Continual learning benchmark
    cl_tasks: int = 5
    cl_classes_per_task: int = 2
    cl_fan_in: int = 32
    cl_hidden: int = 16
    cl_samples_per_class: int = 128
    cl_epochs_per_task: int = 25
    cl_lr: float = 0.03
    cl_time_steps: int = 4

    # Neuron & Plasticity defaults
    beta_init: float = 0.9
    v_th: float = 1.0
    surrogate_slope: float = 25.0
    bdh_coupling: float = 0.5
    bdh_decay: float = 0.8
    ltp_rate: float = 0.005
    ltd_rate: float = 0.005
    target_rate: float = 0.10


def run_event_benchmark(
    config: ExperimentConfig,
    seed: int,
    bdh_coupling: float = 0.5,
    homeostasis: bool = False,
) -> dict[str, float]:
    """Train and evaluate on event streams with specified neuron dynamics."""
    torch.manual_seed(seed)
    device = torch.device(config.device)

    train_ds = synthetic_events(
        config.event_n_train, config.event_num_classes, config.event_time_steps, seed=seed + 1
    )
    test_ds = synthetic_events(
        config.event_n_test, config.event_num_classes, config.event_time_steps, seed=seed + 2
    )
    train_ld = make_loader(train_ds, config.event_batch_size, shuffle=True)
    test_ld = make_loader(test_ds, config.event_batch_size, shuffle=False)

    model = EventClassifier(config.event_time_steps, config.event_num_classes).to(device)
    # Apply ablation overrides to the sequence block cell
    model.seq.cell.bdh_coupling = bdh_coupling
    if homeostasis:
        from bdh_spike.plasticity.homeostat import AdaptiveThreshold

        model.seq.homeostat = AdaptiveThreshold(num_channels=None, target_rate=config.target_rate)

    opt = torch.optim.Adam(model.parameters(), lr=config.event_lr)

    for _ in range(config.event_epochs):
        model.train()
        for events, labels in train_ld:
            events, labels = events.to(device), labels.to(device)
            loss = nn.functional.cross_entropy(model(events), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

    acc, sops_meter, tracker = evaluate(model, test_ld, device)
    rep_e = sops_meter.report()
    rep_s = tracker.report()

    return {
        "accuracy": acc,
        "sparsity": rep_s["sparsity_ema"],
        "sops": rep_e["sops"],
        "dense_flops": rep_e["dense_flops"],
        "flops_per_sop": rep_e["flops_per_sop_ratio"],
    }


def _drive(x: torch.Tensor) -> torch.Tensor:
    peak = x.abs().amax(dim=(0, 2), keepdim=True)
    return x / (peak + 1e-8)


@torch.no_grad()
def _eval_cl_task(model: CLModel, loader: DataLoader, seen_classes: int) -> float:
    was_learning = model.learn_fast
    model.eval()
    correct = total = 0
    for x, y in loader:
        logits = model(_drive(x.transpose(0, 1)))
        preds = logits[:, :seen_classes].argmax(dim=1)
        correct += int((preds == y).sum())
        total += y.numel()
    model.learn_fast = was_learning
    return correct / max(total, 1)


def run_cl_benchmark(
    config: ExperimentConfig,
    seed: int,
    learn_fast: bool = True,
    bdh_coupling: float = 0.5,
    homeostasis: bool = False,
) -> dict[str, float]:
    """Run split-task continual learning schedule."""
    torch.manual_seed(seed)
    num_classes = config.cl_tasks * config.cl_classes_per_task
    train_lds, test_lds, _ = make_tasks(
        config.cl_tasks,
        config.cl_classes_per_task,
        config.cl_fan_in,
        n_train=config.cl_samples_per_class,
        n_test=64,
        seed=seed,
        time_steps=config.cl_time_steps,
    )

    model = CLModel(
        fan_in=config.cl_fan_in, hidden_dim=config.cl_hidden, num_classes=num_classes
    )
    # Apply ablation overrides
    model.hidden_blk.cell.bdh_coupling = bdh_coupling
    model.readout.cell.bdh_coupling = bdh_coupling
    if homeostasis:
        from bdh_spike.plasticity.homeostat import AdaptiveThreshold

        model.hidden_blk.homeostat = AdaptiveThreshold(
            num_channels=None, target_rate=config.target_rate
        )
        model.readout.homeostat = AdaptiveThreshold(
            num_channels=None, target_rate=config.target_rate
        )

    opt = torch.optim.Adam(model.parameters(), lr=config.cl_lr)

    history: list[list[float]] = [[] for _ in range(len(train_lds))]
    for k, train_ld in enumerate(train_lds):
        for _ in range(config.cl_epochs_per_task):
            model.train()
            model.learn_fast = False
            for x, y in train_ld:
                logits = model(_drive(x.transpose(0, 1)))
                loss = nn.functional.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

        seen = (k + 1) * (num_classes // len(train_lds))
        model.learn_fast = learn_fast
        for j in range(k + 1):
            history[j].append(_eval_cl_task(model, test_lds[j], seen))
        model.learn_fast = False

    fr = forgetting_ratio(history)
    init_acc = history[0][0] if history and history[0] else 0.0
    final_acc_t0 = history[0][-1] if history and history[0] else 0.0
    final_avg_acc = sum(h[-1] for h in history if h) / len(history) if history else 0.0

    return {
        "forgetting_ratio": fr,
        "task0_initial_acc": init_acc,
        "task0_final_acc": final_acc_t0,
        "final_avg_acc": final_avg_acc,
    }


def run_attention_mode_comparison(
    config: ExperimentConfig,
    seed: int,
) -> dict[str, Any]:
    """Compare bitwise vs surrogate attention modes."""
    torch.manual_seed(seed)
    T, B, N, C = 8, 4, 16, 32
    x = torch.randn(T, B, N, C)

    vit_surrogate = BDHSpikeViT(
        img_size=28,
        patch_size=7,
        in_channels=1,
        num_classes=10,
        embed_dim=16,
        num_heads=2,
        num_steps=8,
        mode="surrogate",
    )
    vit_bitwise = BDHSpikeViT(
        img_size=28,
        patch_size=7,
        in_channels=1,
        num_classes=10,
        embed_dim=16,
        num_heads=2,
        num_steps=8,
        mode="bitwise",
    )
    vit_bitwise.load_state_dict(vit_surrogate.state_dict())

    test_imgs = torch.rand(B, 1, 28, 28)
    with torch.no_grad():
        out_surr = vit_surrogate(test_imgs)
        out_bitw = vit_bitwise(test_imgs)
        exact_match = torch.equal(out_surr, out_bitw)
        max_diff = float((out_surr - out_bitw).abs().max())

    return {
        "exact_match": exact_match,
        "max_abs_diff": max_diff,
    }


def stats(values: list[float]) -> tuple[float, float]:
    """Compute mean and sample standard deviation."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var)


def run_full_ablation_study(
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Run all ablation variants across all configured seeds."""
    variants = {
        "A_full_model": {
            "name": "A: Full BDH-Spike",
            "bdh_coupling": 0.5,
            "learn_fast": True,
            "homeostasis": True,
        },
        "B_no_bdh_coupling": {
            "name": "B: No BDH Coupling (g=0)",
            "bdh_coupling": 0.0,
            "learn_fast": True,
            "homeostasis": True,
        },
        "C_no_w_fast": {
            "name": "C: No W_fast (BPTT-only)",
            "bdh_coupling": 0.5,
            "learn_fast": False,
            "homeostasis": True,
        },
        "D_no_homeostasis": {
            "name": "D: No Homeostasis (fixed V_th)",
            "bdh_coupling": 0.5,
            "learn_fast": True,
            "homeostasis": False,
        },
    }

    results: dict[str, Any] = {"config": asdict(config), "variants": {}}

    print("=" * 80)
    print("BDH-SPIKE ABLATION STUDY SUITE")
    print(f"Seeds: {config.seeds} | Device: {config.device}")
    print("=" * 80)

    for var_id, var_params in variants.items():
        print(f"\n--- Running Variant {var_params['name']} ---")
        event_metrics: dict[str, list[float]] = {
            "accuracy": [],
            "sparsity": [],
            "sops": [],
            "dense_flops": [],
            "flops_per_sop": [],
        }
        cl_metrics: dict[str, list[float]] = {
            "forgetting_ratio": [],
            "task0_initial_acc": [],
            "task0_final_acc": [],
            "final_avg_acc": [],
        }

        for seed in config.seeds:
            print(f"  [Seed {seed}] Running Event Stream & Continual Learning benchmarks...")
            ev_res = run_event_benchmark(
                config,
                seed,
                bdh_coupling=var_params["bdh_coupling"],
                homeostasis=var_params["homeostasis"],
            )
            for k, v in ev_res.items():
                event_metrics[k].append(v)

            cl_res = run_cl_benchmark(
                config,
                seed,
                learn_fast=var_params["learn_fast"],
                bdh_coupling=var_params["bdh_coupling"],
                homeostasis=var_params["homeostasis"],
            )
            for k, v in cl_res.items():
                cl_metrics[k].append(v)

        # Aggregate stats
        aggregated: dict[str, Any] = {"params": var_params, "raw": {}, "stats": {}}
        for k, vals in event_metrics.items():
            aggregated["raw"][f"event_{k}"] = vals
            m, s = stats(vals)
            aggregated["stats"][f"event_{k}"] = {"mean": m, "std": s}

        for k, vals in cl_metrics.items():
            aggregated["raw"][f"cl_{k}"] = vals
            m, s = stats(vals)
            aggregated["stats"][f"cl_{k}"] = {"mean": m, "std": s}

        results["variants"][var_id] = aggregated

        print(
            f"  -> Event Acc: {aggregated['stats']['event_accuracy']['mean']:.2%} ± {aggregated['stats']['event_accuracy']['std']:.2%}"
        )
        print(
            f"  -> Sparsity:  {aggregated['stats']['event_sparsity']['mean']:.2%} ± {aggregated['stats']['event_sparsity']['std']:.2%}"
        )
        print(
            f"  -> Forgetting F: {aggregated['stats']['cl_forgetting_ratio']['mean']:.3f} ± {aggregated['stats']['cl_forgetting_ratio']['std']:.3f}"
        )

    # Variant E: Bitwise vs Surrogate Attention parity check
    print("\n--- Running Variant E: Bitwise vs Surrogate Attention Parity ---")
    att_results = [run_attention_mode_comparison(config, s) for s in config.seeds]
    all_exact = all(r["exact_match"] for r in att_results)
    results["variant_E_attention"] = {
        "exact_match_all_seeds": all_exact,
        "seed_results": att_results,
    }
    print(f"  -> Bitwise/Surrogate exact match across all seeds: {all_exact}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="BDH-Spike Ablation Study")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=str, default="audits/ablation_results.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--cl-epochs", type=int, default=25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = ExperimentConfig(
        seeds=args.seeds,
        event_epochs=args.epochs,
        cl_epochs_per_task=args.cl_epochs,
        device=args.device,
    )

    results = run_full_ablation_study(config)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Done] Results successfully saved to {out_path}")


if __name__ == "__main__":
    main()
