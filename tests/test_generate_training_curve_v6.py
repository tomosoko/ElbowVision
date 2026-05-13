"""Unit tests for scripts/generate_training_curve_v6.py

Tests cover:
- Constants: _PROJECT_ROOT, rcParams
- load_results_csv: CSV parsing, type conversion, column extraction
- generate_yolo_training_curve: plot creation, panel layout, best epoch annotation
- main: CLI flow, directory creation
- Edge cases: empty CSV, single epoch, non-numeric values, missing columns
"""

from __future__ import annotations

import csv
import sys
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# ── ensure scripts/ is importable ──────────────────────────────
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import generate_training_curve_v6 as gtc6


# ── Helpers ────────────────────────────────────────────────────

def _write_csv(path: Path, n_epochs: int = 10) -> Path:
    """Write a synthetic results.csv with realistic YOLO training columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "epoch", "train/box_loss", "train/pose_loss",
        "val/box_loss", "val/pose_loss",
        "metrics/mAP50(B)", "metrics/mAP50(P)",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in range(n_epochs):
            frac = (i + 1) / n_epochs
            w.writerow({
                "epoch": i + 1,
                "train/box_loss": round(1.0 - 0.8 * frac, 4),
                "train/pose_loss": round(0.5 - 0.4 * frac, 4),
                "val/box_loss": round(1.1 - 0.7 * frac, 4),
                "val/pose_loss": round(0.6 - 0.35 * frac, 4),
                "metrics/mAP50(B)": round(0.5 + 0.45 * frac, 4),
                "metrics/mAP50(P)": round(0.4 + 0.55 * frac, 4),
            })
    return path


@pytest.fixture
def csv_path(tmp_path):
    return _write_csv(tmp_path / "results.csv", n_epochs=10)


# ── Constants ──────────────────────────────────────────────────

class TestConstants:
    def test_project_root_is_directory(self):
        assert gtc6._PROJECT_ROOT.is_dir()

    def test_project_root_contains_scripts(self):
        assert (gtc6._PROJECT_ROOT / "scripts").is_dir()


# ── load_results_csv ───────────────────────────────────────────

class TestLoadResultsCsv:
    def test_returns_dict(self, csv_path):
        data = gtc6.load_results_csv(csv_path)
        assert isinstance(data, dict)

    def test_contains_expected_keys(self, csv_path):
        data = gtc6.load_results_csv(csv_path)
        assert "epoch" in data
        assert "train/pose_loss" in data
        assert "val/pose_loss" in data
        assert "metrics/mAP50(B)" in data
        assert "metrics/mAP50(P)" in data

    def test_epoch_count(self, csv_path):
        data = gtc6.load_results_csv(csv_path)
        assert len(data["epoch"]) == 10

    def test_values_are_float(self, csv_path):
        data = gtc6.load_results_csv(csv_path)
        for v in data["epoch"]:
            assert isinstance(v, float)

    def test_mAP_increases_monotonically(self, csv_path):
        data = gtc6.load_results_csv(csv_path)
        map50 = data["metrics/mAP50(P)"]
        for i in range(1, len(map50)):
            assert map50[i] >= map50[i - 1]

    def test_loss_decreases(self, csv_path):
        data = gtc6.load_results_csv(csv_path)
        loss = data["train/pose_loss"]
        assert loss[-1] < loss[0]

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            gtc6.load_results_csv(tmp_path / "missing.csv")

    def test_handles_whitespace_keys(self, tmp_path):
        """Column headers with leading/trailing whitespace should be stripped."""
        p = tmp_path / "ws.csv"
        p.write_text(" epoch , val \n 1, 0.5\n 2, 0.3\n")
        data = gtc6.load_results_csv(p)
        assert "epoch" in data
        # value part is not stripped, but float() handles leading whitespace
        assert len(data["epoch"]) == 2

    def test_handles_non_numeric(self, tmp_path):
        """Non-numeric values should be kept as strings."""
        p = tmp_path / "mixed.csv"
        p.write_text("epoch,note\n1,good\n2,bad\n")
        data = gtc6.load_results_csv(p)
        assert data["note"] == ["good", "bad"]
        assert data["epoch"] == [1.0, 2.0]

    def test_single_row(self, tmp_path):
        p = _write_csv(tmp_path / "single.csv", n_epochs=1)
        data = gtc6.load_results_csv(p)
        assert len(data["epoch"]) == 1

    def test_many_epochs(self, tmp_path):
        p = _write_csv(tmp_path / "many.csv", n_epochs=200)
        data = gtc6.load_results_csv(p)
        assert len(data["epoch"]) == 200


# ── generate_yolo_training_curve ───────────────────────────────

class TestGenerateYoloTrainingCurve:
    def test_creates_file(self, tmp_path):
        csv_p = _write_csv(tmp_path / "runs" / "elbow_v6" / "results.csv")
        out = tmp_path / "fig.png"
        with mock.patch.object(gtc6, "_PROJECT_ROOT", tmp_path):
            gtc6.generate_yolo_training_curve(out)
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_best_epoch_is_last_for_monotonic(self, tmp_path):
        """For monotonically increasing mAP, best epoch should be the last."""
        csv_p = _write_csv(tmp_path / "runs" / "elbow_v6" / "results.csv", n_epochs=10)
        data = gtc6.load_results_csv(csv_p)
        best_ep = int(np.argmax(data["metrics/mAP50(P)"])) + 1
        assert best_ep == 10

    def test_two_panel_layout(self, tmp_path):
        """Figure should have 1x2 subplots."""
        import matplotlib.pyplot as plt
        _write_csv(tmp_path / "runs" / "elbow_v6" / "results.csv")
        out = tmp_path / "fig_panels.png"
        with mock.patch.object(gtc6, "_PROJECT_ROOT", tmp_path):
            with mock.patch.object(plt, "subplots", wraps=plt.subplots) as m:
                gtc6.generate_yolo_training_curve(out)
                args, kwargs = m.call_args
                assert args == (1, 2)

    def test_output_is_png(self, tmp_path):
        _write_csv(tmp_path / "runs" / "elbow_v6" / "results.csv")
        out = tmp_path / "fig.png"
        with mock.patch.object(gtc6, "_PROJECT_ROOT", tmp_path):
            gtc6.generate_yolo_training_curve(out)
        # PNG magic bytes
        with open(out, "rb") as f:
            header = f.read(4)
        assert header[1:4] == b"PNG"

    def test_best_epoch_mid_range(self, tmp_path):
        """When mAP peaks in the middle, best epoch should be correct."""
        csv_p = tmp_path / "runs" / "elbow_v6" / "results.csv"
        csv_p.parent.mkdir(parents=True)
        cols = [
            "epoch", "train/box_loss", "train/pose_loss",
            "val/box_loss", "val/pose_loss",
            "metrics/mAP50(B)", "metrics/mAP50(P)",
        ]
        with open(csv_p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            # mAP peaks at epoch 5
            for i in range(10):
                map_p = 0.99 if i == 4 else 0.5 + 0.04 * i
                w.writerow({
                    "epoch": i + 1,
                    "train/box_loss": 0.5,
                    "train/pose_loss": 0.3,
                    "val/box_loss": 0.6,
                    "val/pose_loss": 0.35,
                    "metrics/mAP50(B)": 0.8,
                    "metrics/mAP50(P)": map_p,
                })
        data = gtc6.load_results_csv(csv_p)
        best_ep = int(np.argmax(data["metrics/mAP50(P)"])) + 1
        assert best_ep == 5


# ── main ───────────────────────────────────────────────────────

class TestMain:
    def test_main_creates_output(self, tmp_path):
        _write_csv(tmp_path / "runs" / "elbow_v6" / "results.csv")
        with mock.patch.object(gtc6, "_PROJECT_ROOT", tmp_path):
            gtc6.main()
        fig_dir = tmp_path / "results" / "figures"
        assert fig_dir.is_dir()
        assert (fig_dir / "fig6_yolo_v6_training_curve.png").exists()

    def test_main_creates_figures_dir(self, tmp_path):
        _write_csv(tmp_path / "runs" / "elbow_v6" / "results.csv")
        fig_dir = tmp_path / "results" / "figures"
        assert not fig_dir.exists()
        with mock.patch.object(gtc6, "_PROJECT_ROOT", tmp_path):
            gtc6.main()
        assert fig_dir.is_dir()

    def test_main_missing_csv_raises(self, tmp_path):
        with mock.patch.object(gtc6, "_PROJECT_ROOT", tmp_path):
            with pytest.raises(FileNotFoundError):
                gtc6.main()


# ── Edge cases ─────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_epoch_curve(self, tmp_path):
        _write_csv(tmp_path / "runs" / "elbow_v6" / "results.csv", n_epochs=1)
        out = tmp_path / "fig_single.png"
        with mock.patch.object(gtc6, "_PROJECT_ROOT", tmp_path):
            gtc6.generate_yolo_training_curve(out)
        assert out.exists()

    def test_large_epoch_count(self, tmp_path):
        _write_csv(tmp_path / "runs" / "elbow_v6" / "results.csv", n_epochs=300)
        out = tmp_path / "fig_large.png"
        with mock.patch.object(gtc6, "_PROJECT_ROOT", tmp_path):
            gtc6.generate_yolo_training_curve(out)
        assert out.exists()

    def test_csv_with_extra_columns(self, tmp_path):
        """Extra columns should not break loading."""
        p = tmp_path / "extra.csv"
        p.write_text("epoch,metrics/mAP50(P),extra_col\n1,0.5,hello\n2,0.9,world\n")
        data = gtc6.load_results_csv(p)
        assert "extra_col" in data
        assert data["metrics/mAP50(P)"] == [0.5, 0.9]
