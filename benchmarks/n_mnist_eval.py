"""Event-dataset benchmark: N-MNIST / DVS-Gesture / synthetic (Stage 6).

Evaluates the BDH-Spike stack on neuromorphic event streams and reports
accuracy together with energy-first metrics (SOPs vs dense FLOPs, temporal
sparsity) using :mod:`bdh_spike.neuromorphic.metrics`.

Examples::

    # fully self-contained smoke run (no download needed)
    python benchmarks/n_mnist_eval.py --synthetic --epochs 3

    # AGENTS.md canonical invocation
    python -m benchmarks.n_mnist_eval --epochs 5 --time-steps 16 --device cuda

Real datasets require the SpikingJelly raw archives::

    from spikingjelly.datasets.n_mnist import NMIST   # manual download first
"""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from bdh_spike.models import BDHSpikeSeq
from bdh_spike.neuromorphic.metrics import SOPSMeter, SpikeSparsityTracker


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
def synthetic_events(n_samples: int, num_classes: int, time_steps: int, seed: int = 0) -> Dataset:
    """Class-prototyped random event streams ``[T, P=2, H=7, W=7]``."""

    class _Synthetic(Dataset):
        def __init__(self, n: int) -> None:
            self.n = n
            g = torch.Generator().manual_seed(seed * 1000)
            self.prototypes = torch.randn(num_classes, 2, 7, 7, generator=g)

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, idx: int):
            label = idx % num_classes
            noise = torch.randn(2, 7, 7) * 0.35
            frame = (self.prototypes[label] + noise).clamp(-1.5, 1.5)
            return frame.unsqueeze(0).repeat(time_steps, 1, 1, 1), label

    return _Synthetic(n_samples)


def real_event_dataset(name: str, root: str, time_steps: int, train: bool) -> Dataset:
    """Load N-MNIST / DVS-Gesture via SpikingJelly (archives must exist)."""
    try:
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
        from spikingjelly.datasets.n_mnist import NMIST
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"SpikingJelly is required for {name}: pip install spikingjelly ({exc})"
        ) from exc
    cls = NMIST if name == "n-mnist" else DVS128Gesture
    try:
        return cls(
            root=root,
            train=train,
            data_type="frame",
            frames_number=time_steps,
            split_by="number",
        )
    except Exception as exc:  # pragma: no cover - depends on local files
        raise SystemExit(
            f"Could not materialize {name} frames under {root!r}: {exc}\n"
            "Download the raw archive first (see spikingjelly docs) or use --synthetic."
        ) from exc


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# --------------------------------------------------------------------------- #
# Model wrapper                                                               #
# --------------------------------------------------------------------------- #
class EventClassifier(nn.Module):
    """Spatial-pool events → BDH-Spike seq block → rate-readout logits."""

    def __init__(self, time_steps: int, num_classes: int = 10) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.seq = BDHSpikeSeq(fan_in=2 * 7 * 7, fan_out=num_classes)

    def _pool_events(self, events: torch.Tensor) -> torch.Tensor:
        """``[T, B, P, H, W] → [T, B, fan_in]`` pooled event currents."""
        t, b = events.shape[0], events.shape[1]
        x = self.pool(events.reshape(t * b, *events.shape[2:]))  # pool 4D
        return x.reshape(t, b, -1)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """Batch-first event clip ``[B, T, P, H, W]`` → rate logits ``[B, classes]``."""
        pooled = self._pool_events(events.transpose(0, 1))  # → [T, B, fan_in]
        spikes, _ = self.seq(pooled)
        return spikes.mean(dim=0) * 10.0  # rate readout ≈ logits scale


# --------------------------------------------------------------------------- #
# Eval / train loops                                                          #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(
    model: EventClassifier, loader: DataLoader, device: torch.device
) -> tuple[float, SOPSMeter, SpikeSparsityTracker]:
    model.eval()
    correct = total = 0
    sops = SOPSMeter()
    tracker = SpikeSparsityTracker()
    fan_in, fan_out, steps = model.seq.fan_in, model.seq.fan_out, model.time_steps
    for events, labels in loader:
        events, labels = events.to(device), labels.to(device)
        rates = model(events)
        correct += int((rates.argmax(dim=1) == labels).sum())
        total += labels.numel()

        pooled = model._pool_events(events.transpose(0, 1))
        sops.register_layer(fan_in=fan_in, fan_out=fan_out, steps=steps, batch=total)
        sops.update((pooled > 0.0).float(), fan_out=fan_out)
        hidden, _ = model.seq(pooled)
        tracker.update(hidden)
    return correct / max(total, 1), sops, tracker


def main() -> None:
    parser = argparse.ArgumentParser(description="BDH-Spike event benchmark")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--time-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dataset", choices=["synthetic", "n-mnist", "dvs-gesture"], default="synthetic"
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--num-classes", type=int, default=10)
    args = parser.parse_args()
    device = torch.device(args.device)

    if args.dataset == "synthetic":
        train_ds = synthetic_events(1024, args.num_classes, args.time_steps, seed=1)
        test_ds = synthetic_events(256, args.num_classes, args.time_steps, seed=2)
    else:
        train_ds = real_event_dataset(args.dataset, args.data_dir, args.time_steps, train=True)
        test_ds = real_event_dataset(args.dataset, args.data_dir, args.time_steps, train=False)
    train_ld = make_loader(train_ds, args.batch_size, shuffle=True)
    test_ld = make_loader(test_ds, args.batch_size, shuffle=False)

    torch.manual_seed(0)
    model = EventClassifier(args.time_steps, args.num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for events, labels in train_ld:
            events, labels = events.to(device), labels.to(device)
            loss = nn.functional.cross_entropy(model(events), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.detach())
        acc, sops, tracker = evaluate(model, test_ld, device)
        rep_e, rep_s = sops.report(), tracker.report()
        print(
            f"[epoch {epoch:02d}] loss={running / len(train_ld):.4f} acc={acc:.2%} | "
            f"sparsity_ema={rep_s['sparsity_ema']:.2%} | "
            f"SOPs={rep_e['sops']:.3g} dense_FLOPs={rep_e['dense_flops']:.3g} "
            f"(FLOPs/SOP ×{rep_e['flops_per_sop_ratio']:.1f})"
        )


if __name__ == "__main__":
    main()
