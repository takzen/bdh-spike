"""Spike visualization & telemetry: rasters, membrane traces, terminal HUD (Stage 7).

Two complementary views on network activity:

* Publication-grade figures (matplotlib): :func:`plot_raster` renders binary
  spike trains as event scatter plots; :func:`plot_membrane` overlays the
  continuous membrane potential, the firing threshold and the emitted spikes
  of a single unit.
* Zero-dependency terminal telemetry: :func:`ascii_raster` draws a text
  raster, and :class:`TelemetryRecorder` accumulates live firing statistics
  (rate / sparsity / SOPs), rendering a terminal HUD or exporting JSON lines
  ready for any web dashboard.

All temporal tensors follow the canonical ``[T, B, C]`` layout.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

from bdh_spike.core.neuron import BDHSpikeCell, BDHState
from bdh_spike.neuromorphic.metrics import calculate_sparsity, sops_count

__all__ = [
    "TelemetryRecorder",
    "ascii_raster",
    "plot_membrane",
    "plot_raster",
    "record_dynamics",
]


# --------------------------------------------------------------------------- #
# Recording                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class DynamicsRecord:
    """Full single-cell dynamics captured over a sequence.

    Attributes:
        spikes: Binary spikes ``[T, B, C]``.
        mem: Membrane potential *after* each step's reset ``[T, B, C]``.
        m_bdh: BDH coupling vector after each step ``[T, B, C]``.
    """

    spikes: torch.Tensor
    mem: torch.Tensor
    m_bdh: torch.Tensor


def record_dynamics(
    cell: BDHSpikeCell, x: torch.Tensor, state: BDHState | None = None
) -> Iterator[DynamicsRecord]:
    """Yield a :class:`DynamicsRecord` per batch element of a ``[T, B, C]`` run.

    Records the membrane potential and BDH coupling at every step so that
    :func:`plot_membrane` can draw exact traces afterwards.
    """
    if state is None:
        state = cell.init_hidden(x.shape[1])
    spikes_t, mem_t, mbdh_t = [], [], []
    for step in range(x.shape[0]):
        current = x[step]
        pre_mem = cell.beta * state.mem + current + state.m_bdh - state.prev_spikes * cell.v_th
        spikes, state = cell.single_step(current, state)
        spikes_t.append(spikes)
        mem_t.append(pre_mem)
        mbdh_t.append(state.m_bdh)
    record = DynamicsRecord(
        spikes=torch.stack(spikes_t),
        mem=torch.stack(mem_t),
        m_bdh=torch.stack(mbdh_t),
    )
    for b in range(record.spikes.shape[1]):
        yield DynamicsRecord(
            spikes=record.spikes[:, b], mem=record.mem[:, b], m_bdh=record.m_bdh[:, b]
        )


# --------------------------------------------------------------------------- #
# Matplotlib figures                                                          #
# --------------------------------------------------------------------------- #
def _plt():
    """Import matplotlib lazily and force a headless-friendly backend check."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "matplotlib is required for figure export; "
            "use ascii_raster / TelemetryRecorder for terminal-only output"
        ) from exc
    return plt


def plot_raster(
    spikes: torch.Tensor,
    batch_index: int = 0,
    ax=None,
    title: str = "Spike raster",
    color: str = "#2a9d8f",
):
    """Event-scatter raster of one sample: spikes ``[T, B, C]`` → dots ``(t, unit)``.

    Args:
        spikes: Binary spike train, canonical ``[T, B, C]``.
        batch_index: Which batch element to draw.
        ax: Optional existing matplotlib axes.
        title: Axes title.
        color: Marker color.

    Returns:
        ``(fig, ax)`` matplotlib handles.
    """
    plt = _plt()
    s = spikes.detach()[:, batch_index, :]  # [T, C]
    times, units = torch.nonzero(s > 0, as_tuple=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(times.numpy(), units.numpy(), s=6, c=color, marker="|")
    ax.set_xlabel("time-step")
    ax.set_ylabel("unit")
    ax.set_ylim(-0.5, s.shape[1] - 0.5)
    ax.invert_yaxis()
    ax.set_title(f"{title} — sparsity {calculate_sparsity(s):.1%}")
    return ax


def plot_membrane(
    record: DynamicsRecord,
    v_th: float = 1.0,
    unit: int = 0,
    ax=None,
    title: str = "Membrane potential",
):
    """Membrane trace of one unit with threshold line and spike markers.

    Args:
        record: A per-sample :class:`DynamicsRecord` from :func:`record_dynamics`.
        v_th: Firing threshold used by the cell.
        unit: Channel index to draw.
        ax: Optional existing matplotlib axes.
        title: Axes title.

    Returns:
        ``(fig, ax)`` matplotlib handles.
    """
    plt = _plt()
    mem = record.mem[:, unit].detach().numpy()
    spikes = record.spikes[:, unit].detach()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    ax.plot(mem, lw=1.4, label="V[t]")
    ax.axhline(v_th, color="crimson", ls="--", lw=1.0, label="V_th")
    fire_steps = torch.nonzero(spikes > 0, as_tuple=True)[0].numpy()
    ax.scatter(fire_steps, mem[fire_steps], c="crimson", zorder=3, s=18, label="spike")
    ax.set_xlabel("time-step")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    return ax


# --------------------------------------------------------------------------- #
# Terminal HUD & JSON telemetry                                               #
# --------------------------------------------------------------------------- #
def ascii_raster(spikes: torch.Tensor, batch_index: int = 0, max_steps: int = 80) -> str:
    """Text-mode spike raster: rows = units, columns = (downsampled) time.

    Args:
        spikes: Binary spike train ``[T, B, C]``.
        batch_index: Which batch element to draw.
        max_steps: Maximum number of time columns (longer trains are
            max-pooled down so no event is lost).

    Returns:
        Multi-line string; ``'█'`` = spike, ``'·'`` = silence.
    """
    s = (spikes.detach()[:, batch_index, :] > 0).to(torch.uint8)  # [T, C]
    t, c = s.shape
    if t > max_steps:
        pad = (-t) % max_steps
        if pad:
            s = torch.cat([s, torch.zeros(pad, c, dtype=s.dtype)], dim=0)
        s = s.view(-1, max_steps, c).max(dim=0).values
        t = max_steps
    rows = []
    for unit in range(c):
        rows.append("".join("█" if s[step, unit] else "·" for step in range(t)))
    return "\n".join(rows)


@dataclass
class TelemetryRecorder:
    """Live firing telemetry for terminal HUDs or JSON export.

    Accumulates per-``update`` statistics over binary spike tensors and can
    render them as a compact terminal HUD or dump JSON lines for a web UI.

    Args:
        fan_out: Optional fan-out of the consuming synapse layer; when set,
            SOPs are accumulated on every :meth:`update`.
    """

    fan_out: int | None = None
    steps: int = 0
    total_elements: int = 0
    total_spikes: int = 0
    total_sops: int = 0
    history: list[dict[str, float]] = field(default_factory=list)

    def update(self, spikes: torch.Tensor, fan_out: int | None = None) -> dict[str, float]:
        """Feed one binary spike tensor ``[T, B, C]``; returns its metrics."""
        fo = self.fan_out if fan_out is None else fan_out
        rate = float(spikes.detach().float().mean())
        entry = {
            "step": float(self.steps),
            "rate": rate,
            "sparsity": calculate_sparsity(spikes),
        }
        if fo is not None:
            entry["sops"] = float(sops_count(spikes.detach(), fan_out=fo))
            self.total_sops += int(entry["sops"])
        self.steps += int(spikes.shape[0])
        self.total_elements += spikes.numel()
        self.total_spikes += int(torch.count_nonzero(spikes.detach()))
        self.history.append(entry)
        return entry

    @property
    def overall_rate(self) -> float:
        return self.total_spikes / max(self.total_elements, 1)

    def hud_lines(self) -> list[str]:
        """Render the current state as terminal-HUD lines."""
        lines = [
            "╔══════════ BDH-SPIKE TELEMETRY ══════════╗",
            f"║ steps observed : {self.steps:>8d}              ║",
            f"║ mean fire rate : {self.overall_rate:>7.2%}              ║",
            f"║ total spikes   : {self.total_spikes:>8d}              ║",
        ]
        if self.total_sops:
            lines.append(f"║ total SOPs     : {self.total_sops:>8.3e}          ║")
        if self.history:
            last = self.history[-1]
            bar = "▁" * round(last["sparsity"] * 40) + "░"
            lines.append(f"║ last sparsity  : {last['sparsity']:>7.2%}  {bar[:41]}")
        lines.append("╚══════════════════════════════════════════╝")
        return lines

    def render(self) -> None:
        """Print the HUD to stdout."""
        print("\n".join(self.hud_lines()))

    def json_line(self) -> str:
        """One JSON line summarizing all history so far (web-export ready)."""
        payload = {
            "steps": self.steps,
            "total_spikes": self.total_spikes,
            "total_sops": self.total_sops,
            "overall_rate": round(self.overall_rate, 6),
            "history": self.history,
        }
        return json.dumps(payload)

    def dump(self, path: str) -> None:
        """Write the full telemetry payload as a single JSON document."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.json_line())
