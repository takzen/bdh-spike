"""Split-task continual-learning benchmark (Stage 6).

Demonstrates the dual-weight doctrine against catastrophic forgetting:

* **Run A (`bptt-only`)** — structural weights ``W_slow`` are fine-tuned by
  BPTT/SGD on each new task; nothing else adapts. Classic recipe: old-task
  accuracy decays as new tasks overwrite features.
* **Run B (`dual-weight`)** — identical SGD schedule **plus** local 3-factor
  STDP adapting episodic ``W_fast`` while each task's stream flows through at
  inference time. Gradient-free and scoped to the fused kernel, so old-task
  knowledge in ``W_slow`` is untouched and every evaluated task gets freshly
  hebbian-reinforced before scoring.

Metrics per run: final accuracy per task and ``forgetting_ratio`` —
``mean_{k'<K}(max_j acc[k'][j] − acc[k'][final])`` (0 = no forgetting).

Example::

    python -m benchmarks.continual_learning --tasks 5 --eval-method forgetting_ratio
"""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bdh_spike.models import BDHSpikeSeq


def make_tasks(
    num_tasks: int,
    classes_per_task: int,
    fan_in: int,
    n_train: int,
    n_test: int,
    seed: int = 0,
    time_steps: int = 4,
) -> tuple[list[DataLoader], list[DataLoader], int]:
    """Synthetic split-task streams: class-prototyped gaussian blobs.

    Each static prototype is held constant over a ``time_steps`` window so the
    PLIF cell integrates it in the canonical ``[T, B, C]`` layout.
    """
    g = torch.Generator().manual_seed(seed)
    total = num_tasks * classes_per_task
    prototypes = torch.randn(total, fan_in, generator=g) * 2.0

    def build(n: int, offset: int) -> DataLoader:
        xs, ys = [], []
        for c in range(classes_per_task):
            base = prototypes[offset + c]
            x = base.unsqueeze(0) + torch.randn(n, fan_in, generator=g) * 0.5
            xs.append(x.unsqueeze(1).repeat(1, time_steps, 1))  # [N, T, C]
            ys.append(torch.full((n,), offset + c))
        return DataLoader(TensorDataset(torch.cat(xs), torch.cat(ys)), batch_size=64, shuffle=True)

    train = [build(n_train, k * classes_per_task) for k in range(num_tasks)]
    test = [build(n_test, k * classes_per_task) for k in range(num_tasks)]
    return train, test, total


def _drive(x: torch.Tensor) -> torch.Tensor:
    """Normalize a ``[T, B, C]`` window into a threshold-active drive."""
    peak = x.abs().amax(dim=(0, 2), keepdim=True)
    return x / (peak + 1e-8)


class CLModel(nn.Module):
    """Two-block spiking MLP: shared hidden features → class readout.

    The hidden block is *shared across all tasks* — exactly the condition
    under which sequential SGD causes catastrophic forgetting.
    """

    def __init__(self, fan_in: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.hidden_blk = BDHSpikeSeq(fan_in=fan_in, fan_out=hidden_dim)
        self.readout = BDHSpikeSeq(fan_in=hidden_dim, fan_out=num_classes)

    @property
    def learn_fast(self) -> bool:
        return self.readout.learn_fast

    @learn_fast.setter
    def learn_fast(self, flag: bool) -> None:
        self.hidden_blk.learn_fast = flag
        self.readout.learn_fast = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[T, B, C_in]`` → rate logits ``[B, num_classes]``."""
        h, _ = self.hidden_blk(x)
        s, _ = self.readout(h)
        return s.mean(dim=0) * 10.0


@torch.no_grad()
def evaluate_task(model: CLModel, loader: DataLoader, seen_classes: int) -> float:
    """Accuracy on one task's stream; argmax restricted to classes seen so far."""
    was_learning = model.learn_fast
    model.eval()
    correct = total = 0
    for x, y in loader:
        logits = model(_drive(x.transpose(0, 1)))  # [B,T,C] → [T,B,C]
        preds = logits[:, :seen_classes].argmax(dim=1)
        correct += int((preds == y).sum())
        total += y.numel()
    model.learn_fast = was_learning
    return correct / max(total, 1)


def forgetting_ratio(acc_history: list[list[float]]) -> float:
    """Mean peak-to-final accuracy drop over all tasks except the last.

    ``acc_history[k]`` holds task-k accuracies measured after each later
    training step (chronological); the drop is ``max − final`` per task.
    """
    if len(acc_history) < 2:
        return 0.0
    drops = []
    for k in range(len(acc_history) - 1):
        curve = acc_history[k]
        if not curve:
            continue
        drops.append(max(curve) - curve[-1])
    return sum(drops) / len(drops)


def run_schedule(
    learn_fast: bool,
    train_loaders: list[DataLoader],
    test_loaders: list[DataLoader],
    fan_in: int,
    num_classes: int,
    epochs_per_task: int,
    lr: float,
    seed: int,
    hidden_dim: int = 16,
) -> list[list[float]]:
    """Sequentially learn tasks; returns accuracy history ``[task][eval]``.

    Both schedules train the shared hidden block identically (SGD/BPTT per
    task). When ``learn_fast`` is set, episodic ``W_fast`` additionally adapts
    online via local STDP while each task's stream flows through at evaluation
    time — gradient-free, so structural weights never move and every evaluated
    task gets freshly hebbian-reinforced before scoring.
    """
    torch.manual_seed(seed)
    model = CLModel(fan_in=fan_in, hidden_dim=hidden_dim, num_classes=num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[list[float]] = [[] for _ in range(len(train_loaders))]
    for k, train_ld in enumerate(train_loaders):
        for _ in range(epochs_per_task):
            model.train()
            model.learn_fast = False  # structural learning only
            for x, y in train_ld:
                logits = model(_drive(x.transpose(0, 1)))  # [B,T,C] → [T,B,C]
                loss = nn.functional.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

        seen = (k + 1) * (num_classes // len(train_loaders))
        model.learn_fast = learn_fast  # episodic STDP at test time (if enabled)
        for j in range(k + 1):
            history[j].append(evaluate_task(model, test_loaders[j], seen))
        model.learn_fast = False
    return history


def print_history(name: str, history: list[list[float]], num_tasks: int) -> None:
    """Pretty-print the accuracy matrix: rows = evaluation point, cols = task."""
    header = "after task | " + " | ".join(f"t{j}" for j in range(num_tasks))
    print(header)
    for step in range(num_tasks):
        cells = " | ".join(
            f"{history[j][step - j]:.2f}"
            if step < len(history) and len(history[j]) > step - j
            else "  -"
            for j in range(step + 1)
        )
        print(f"     {step}      {cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BDH-Spike continual learning benchmark")
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--classes-per-task", type=int, default=2)
    parser.add_argument("--fan-in", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--samples-per-class", type=int, default=128)
    parser.add_argument("--epochs-per-task", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-method",
        choices=["forgetting_ratio"],
        default="forgetting_ratio",
    )
    args = parser.parse_args()

    num_classes = args.tasks * args.classes_per_task
    train_lds, test_lds, _ = make_tasks(
        args.tasks,
        args.classes_per_task,
        args.fan_in,
        n_train=args.samples_per_class,
        n_test=64,
        seed=args.seed,
    )

    results: dict[str, tuple[list[list[float]], float]] = {}
    for name, lf in (("bptt-only", False), ("dual-weight", True)):
        hist = run_schedule(
            learn_fast=lf,
            train_loaders=train_lds,
            test_loaders=test_lds,
            fan_in=args.fan_in,
            num_classes=num_classes,
            epochs_per_task=args.epochs_per_task,
            lr=args.lr,
            seed=args.seed,
            hidden_dim=args.hidden,
        )
        results[name] = (hist, forgetting_ratio(hist))

    print("\n=== Continual learning report ===")
    for name, (hist, fr) in results.items():
        print(f"\n[{name}] forgetting_ratio = {fr:.3f}")
        print_history(name, hist, args.tasks)

    a_fr = results["bptt-only"][1]
    b_fr = results["dual-weight"][1]
    verdict = (
        "dual-weight lower forgetting in this run (verify across seeds)"
        if b_fr < a_fr
        else "no benefit observed in this run (verify across seeds)"
    )
    print(f"\nverdict: dual-weight ({b_fr:.3f}) vs bptt-only ({a_fr:.3f}) → {verdict}")


if __name__ == "__main__":
    main()
