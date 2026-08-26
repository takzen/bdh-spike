"""Unit tests for Stage-7 visualization & telemetry utilities."""

from __future__ import annotations

import json

import pytest
import torch

from bdh_spike.core import BDHSpikeCell
from bdh_spike.utils.visualizer import (
    TelemetryRecorder,
    ascii_raster,
    plot_membrane,
    plot_raster,
    record_dynamics,
)

T, B, C = 12, 3, 6


@pytest.fixture()
def cell() -> BDHSpikeCell:
    torch.manual_seed(0)
    return BDHSpikeCell(num_channels=C)


@pytest.fixture()
def spikes(cell: BDHSpikeCell) -> torch.Tensor:
    x = torch.full((T, B, C), 2.0)
    out, _ = cell(x)
    return out


class TestRecordDynamics:
    def test_shapes_and_binary_domain(self, cell: BDHSpikeCell) -> None:
        x = torch.full((T, B, C), 2.0)
        records = list(record_dynamics(cell, x))
        assert len(records) == B
        rec = records[0]
        assert rec.spikes.shape == (T, C)
        assert rec.mem.shape == (T, C)
        assert rec.m_bdh.shape == (T, C)
        assert set(rec.spikes.unique().tolist()).issubset({0.0, 1.0})

    def test_spikes_match_cell_forward(self, cell: BDHSpikeCell) -> None:
        torch.manual_seed(1)
        x = torch.randn(T, B, C)
        expected, _ = cell(x.clone())
        records = list(record_dynamics(cell, x))
        for b in range(B):
            assert torch.equal(records[b].spikes, expected[:, b])


class TestAsciiRaster:
    def test_marks_and_silence_symbols(self, spikes: torch.Tensor) -> None:
        art = ascii_raster(spikes)
        lines = art.split("\n")
        assert len(lines) == C
        assert all(len(line) == T for line in lines)
        assert set("".join(lines)) <= {"█", "·"}

    def test_all_active_vs_all_silent(self) -> None:
        active = ascii_raster(torch.ones(4, 1, 2))
        silent = ascii_raster(torch.zeros(4, 1, 2))
        assert active == ("████\n" * 2).strip("\n") or set(active) == {"█", "\n"}
        assert set(silent) == {"·", "\n"}

    def test_downsampling_preserves_width(self, spikes: torch.Tensor) -> None:
        art = ascii_raster(spikes, max_steps=5)
        lines = art.split("\n")
        assert all(len(line) == 5 for line in lines)


class TestTelemetryRecorder:
    def test_update_metrics_math(self) -> None:
        rec = TelemetryRecorder(fan_out=4)
        s = torch.zeros(4, 2, 5)
        s[:, :, 0] = 1.0  # rate = 0.2
        entry = rec.update(s)
        assert entry["rate"] == pytest.approx(0.2)
        assert entry["sparsity"] == pytest.approx(0.8)
        assert entry["sops"] == float(4 * 2 * 1 * 4)
        assert rec.total_sops == 32
        assert rec.steps == 4

    def test_accumulation_and_overall_rate(self) -> None:
        rec = TelemetryRecorder()
        rec.update(torch.zeros(2, 1, 4))
        rec.update(torch.ones(2, 1, 4))
        assert rec.overall_rate == pytest.approx(0.5)
        assert rec.total_spikes == 8
        assert len(rec.history) == 2

    def test_hud_lines_render_summary(self, spikes: torch.Tensor) -> None:
        rec = TelemetryRecorder(fan_out=5)
        rec.update(spikes)
        hud = "\n".join(rec.hud_lines())
        assert "BDH-SPIKE TELEMETRY" in hud
        assert "SOPs" in hud
        assert "fire rate" in hud

    def test_json_export_roundtrip(self, spikes: torch.Tensor, tmp_path) -> None:
        rec = TelemetryRecorder(fan_out=5)
        rec.update(spikes)
        path = tmp_path / "telemetry.json"
        rec.dump(str(path))
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["steps"] == T
        assert payload["total_spikes"] == int(spikes.sum())
        assert len(payload["history"]) == 1
        assert json.loads(rec.json_line())["overall_rate"] == pytest.approx(
            float(spikes.detach().float().mean())
        )


class TestMatplotlibFigures:
    @pytest.fixture(autouse=True)
    def _agg_backend(self):
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")

    def test_plot_raster_returns_axes(self, spikes: torch.Tensor) -> None:
        ax = plot_raster(spikes)
        assert ax is not None
        assert "sparsity" in ax.get_title()

    def test_plot_membrane_marks_threshold(self, cell: BDHSpikeCell) -> None:
        x = torch.full((T, 1, C), 2.0)
        record = next(iter(record_dynamics(cell, x)))
        ax = plot_membrane(record, v_th=float(cell.v_th))
        assert ax is not None
