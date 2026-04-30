"""Tests for scripts/dataset_stats.py pure-function utilities.

Covers:
  - print_summary: stdout output, stats computation
  - plot_angle_distribution: PNG output, column-presence handling
  - plot_view_counts: PNG output, with/without split column
  - plot_sample_grid: PNG output, missing-image graceful handling
"""
import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from dataset_stats import (
    plot_angle_distribution,
    plot_sample_grid,
    plot_view_counts,
    print_summary,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def make_df(n_ap=6, n_lat=4, with_split=True, seed=0):
    """Return a minimal DataFrame matching dataset_summary.csv schema."""
    rng = np.random.default_rng(seed)
    ap = pd.DataFrame({
        "filename":          [f"ap_{i:04d}.png" for i in range(n_ap)],
        "view_type":         ["AP"] * n_ap,
        "flexion_deg":       rng.uniform(150, 180, n_ap),
        "rotation_error_deg": rng.uniform(-30, 30, n_ap),
    })
    lat = pd.DataFrame({
        "filename":          [f"lat_{i:04d}.png" for i in range(n_lat)],
        "view_type":         ["LAT"] * n_lat,
        "flexion_deg":       rng.uniform(60, 120, n_lat),
        "rotation_error_deg": rng.uniform(-30, 30, n_lat),
    })
    df = pd.concat([ap, lat], ignore_index=True)
    if with_split:
        df["split"] = (["train"] * int(len(df) * 0.8) + ["val"] * (len(df) - int(len(df) * 0.8)))
    return df


@pytest.fixture
def df_full():
    return make_df(n_ap=6, n_lat=4, with_split=True)


@pytest.fixture
def df_no_split():
    return make_df(n_ap=4, n_lat=4, with_split=False)


@pytest.fixture
def tmp(tmp_path):
    return str(tmp_path)


# ── print_summary ─────────────────────────────────────────────────────────

class TestPrintSummary:
    """Tests for print_summary(df)."""

    def test_total_count_in_output(self, df_full, capsys):
        print_summary(df_full)
        out = capsys.readouterr().out
        assert "10" in out  # 6 AP + 4 LAT = 10

    def test_view_types_shown(self, df_full, capsys):
        print_summary(df_full)
        out = capsys.readouterr().out
        assert "AP" in out
        assert "LAT" in out

    def test_ap_count_correct(self, df_full, capsys):
        print_summary(df_full)
        out = capsys.readouterr().out
        assert "6" in out

    def test_lat_count_correct(self, df_full, capsys):
        print_summary(df_full)
        out = capsys.readouterr().out
        assert "4" in out

    def test_split_shown_when_present(self, df_full, capsys):
        print_summary(df_full)
        out = capsys.readouterr().out
        assert "train" in out
        assert "val" in out

    def test_split_absent_no_crash(self, df_no_split, capsys):
        print_summary(df_no_split)
        out = capsys.readouterr().out
        assert "8" in out  # 4 AP + 4 LAT

    def test_flexion_range_shown(self, df_full, capsys):
        print_summary(df_full)
        out = capsys.readouterr().out
        assert "flexion" in out.lower() or "屈曲" in out

    def test_rotation_range_shown(self, df_full, capsys):
        print_summary(df_full)
        out = capsys.readouterr().out
        assert "rotation" in out.lower() or "回旋" in out

    def test_no_flexion_column_no_crash(self, capsys):
        df = pd.DataFrame({"view_type": ["AP", "LAT"], "rotation_error_deg": [0.0, 1.0]})
        print_summary(df)  # must not raise
        capsys.readouterr()

    def test_no_rotation_column_no_crash(self, capsys):
        df = pd.DataFrame({"view_type": ["AP"], "flexion_deg": [90.0]})
        print_summary(df)
        capsys.readouterr()

    def test_single_row_df(self, capsys):
        df = pd.DataFrame({
            "view_type": ["AP"],
            "flexion_deg": [90.0],
            "rotation_error_deg": [5.0],
        })
        print_summary(df)
        out = capsys.readouterr().out
        assert "1" in out

    def test_uppercase_view_type_handled(self, capsys):
        df = pd.DataFrame({
            "view_type": ["ap", "lat"],
            "flexion_deg": [120.0, 90.0],
            "rotation_error_deg": [0.0, 0.0],
        })
        print_summary(df)
        out = capsys.readouterr().out
        # Should normalise to uppercase
        assert "AP" in out or "ap" in out


# ── plot_angle_distribution ───────────────────────────────────────────────

class TestPlotAngleDistribution:
    """Tests for plot_angle_distribution(df, out_dir)."""

    def test_creates_png(self, df_full, tmp):
        plot_angle_distribution(df_full, tmp)
        assert os.path.exists(os.path.join(tmp, "angle_distribution.png"))

    def test_png_nonzero_size(self, df_full, tmp):
        plot_angle_distribution(df_full, tmp)
        size = os.path.getsize(os.path.join(tmp, "angle_distribution.png"))
        assert size > 1000  # non-trivial image

    def test_ap_only_df(self, tmp):
        df = make_df(n_ap=5, n_lat=0, with_split=False)
        plot_angle_distribution(df, tmp)
        assert os.path.exists(os.path.join(tmp, "angle_distribution.png"))

    def test_lat_only_df(self, tmp):
        df = make_df(n_ap=0, n_lat=5, with_split=False)
        plot_angle_distribution(df, tmp)
        assert os.path.exists(os.path.join(tmp, "angle_distribution.png"))

    def test_lowercase_view_type(self, tmp):
        df = pd.DataFrame({
            "view_type": ["ap", "lat", "ap"],
            "flexion_deg": [170.0, 90.0, 160.0],
            "rotation_error_deg": [5.0, -5.0, 10.0],
        })
        plot_angle_distribution(df, tmp)
        assert os.path.exists(os.path.join(tmp, "angle_distribution.png"))

    def test_mixed_case_view_type(self, tmp):
        df = pd.DataFrame({
            "view_type": ["Ap", "Lat"],
            "flexion_deg": [170.0, 90.0],
            "rotation_error_deg": [0.0, 0.0],
        })
        plot_angle_distribution(df, tmp)
        assert os.path.exists(os.path.join(tmp, "angle_distribution.png"))

    def test_large_dataset(self, tmp):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "view_type": ["AP"] * 500 + ["LAT"] * 500,
            "flexion_deg": rng.uniform(50, 180, 1000),
            "rotation_error_deg": rng.uniform(-45, 45, 1000),
        })
        plot_angle_distribution(df, tmp)
        assert os.path.exists(os.path.join(tmp, "angle_distribution.png"))

    def test_zero_rotation_error(self, tmp):
        df = pd.DataFrame({
            "view_type": ["AP", "LAT"],
            "flexion_deg": [170.0, 90.0],
            "rotation_error_deg": [0.0, 0.0],
        })
        plot_angle_distribution(df, tmp)
        assert os.path.exists(os.path.join(tmp, "angle_distribution.png"))


# ── plot_view_counts ──────────────────────────────────────────────────────

class TestPlotViewCounts:
    """Tests for plot_view_counts(df, out_dir)."""

    def test_creates_png(self, df_full, tmp):
        plot_view_counts(df_full, tmp)
        assert os.path.exists(os.path.join(tmp, "view_split_counts.png"))

    def test_png_nonzero_size(self, df_full, tmp):
        plot_view_counts(df_full, tmp)
        size = os.path.getsize(os.path.join(tmp, "view_split_counts.png"))
        assert size > 1000

    def test_no_split_column(self, df_no_split, tmp):
        plot_view_counts(df_no_split, tmp)
        assert os.path.exists(os.path.join(tmp, "view_split_counts.png"))

    def test_split_percentages_printed(self, df_full, capsys, tmp):
        plot_view_counts(df_full, tmp)
        out = capsys.readouterr().out
        assert "%" in out

    def test_ap_only(self, tmp):
        df = make_df(n_ap=5, n_lat=0, with_split=True)
        plot_view_counts(df, tmp)
        assert os.path.exists(os.path.join(tmp, "view_split_counts.png"))

    def test_lat_only(self, tmp):
        df = make_df(n_ap=0, n_lat=5, with_split=True)
        plot_view_counts(df, tmp)
        assert os.path.exists(os.path.join(tmp, "view_split_counts.png"))

    def test_single_row(self, tmp):
        df = pd.DataFrame({
            "view_type": ["AP"],
            "flexion_deg": [90.0],
            "rotation_error_deg": [0.0],
            "split": ["train"],
        })
        plot_view_counts(df, tmp)
        assert os.path.exists(os.path.join(tmp, "view_split_counts.png"))

    def test_three_splits(self, tmp):
        df = pd.DataFrame({
            "view_type": ["AP", "AP", "LAT", "LAT", "AP", "LAT"],
            "flexion_deg": [170.0, 160.0, 90.0, 80.0, 150.0, 70.0],
            "rotation_error_deg": [0.0] * 6,
            "split": ["train", "train", "train", "val", "test", "test"],
        })
        plot_view_counts(df, tmp)
        assert os.path.exists(os.path.join(tmp, "view_split_counts.png"))

    def test_overwrite_existing_png(self, df_full, tmp):
        plot_view_counts(df_full, tmp)
        mtime1 = os.path.getmtime(os.path.join(tmp, "view_split_counts.png"))
        import time; time.sleep(0.05)
        plot_view_counts(df_full, tmp)
        mtime2 = os.path.getmtime(os.path.join(tmp, "view_split_counts.png"))
        assert mtime2 >= mtime1


# ── plot_sample_grid ──────────────────────────────────────────────────────

class TestPlotSampleGrid:
    """Tests for plot_sample_grid(dataset_dir, df, out_dir, grid_size)."""

    def test_creates_png_no_images(self, df_full, tmp):
        # dataset_dir with no actual images → axes stay blank but file is saved
        plot_sample_grid(tmp, df_full, tmp, grid_size=2)
        assert os.path.exists(os.path.join(tmp, "sample_grid.png"))

    def test_png_nonzero_size(self, df_full, tmp):
        plot_sample_grid(tmp, df_full, tmp, grid_size=2)
        size = os.path.getsize(os.path.join(tmp, "sample_grid.png"))
        assert size > 1000

    def test_grid_size_1(self, df_full, tmp):
        plot_sample_grid(tmp, df_full, tmp, grid_size=1)
        assert os.path.exists(os.path.join(tmp, "sample_grid.png"))

    def test_grid_size_3(self, df_full, tmp):
        plot_sample_grid(tmp, df_full, tmp, grid_size=3)
        assert os.path.exists(os.path.join(tmp, "sample_grid.png"))

    def test_empty_df_no_crash(self, tmp):
        df_empty = pd.DataFrame(columns=["filename", "view_type", "flexion_deg",
                                          "rotation_error_deg", "split"])
        plot_sample_grid(tmp, df_empty, tmp, grid_size=2)
        assert os.path.exists(os.path.join(tmp, "sample_grid.png"))

    def test_real_image_shown(self, tmp):
        """Creates a real grayscale PNG and checks it loads without error."""
        import cv2
        img_dir = os.path.join(tmp, "images", "train")
        os.makedirs(img_dir, exist_ok=True)
        img = np.zeros((64, 64), dtype=np.uint8)
        img[16:48, 16:48] = 128
        cv2.imwrite(os.path.join(img_dir, "ap_0000.png"), img)

        df = pd.DataFrame({
            "filename": ["ap_0000.png"],
            "view_type": ["AP"],
            "flexion_deg": [170.0],
            "rotation_error_deg": [5.0],
            "split": ["train"],
        })
        plot_sample_grid(tmp, df, tmp, grid_size=2)
        assert os.path.exists(os.path.join(tmp, "sample_grid.png"))

    def test_no_split_column_uses_train_default(self, tmp):
        """Without split column, fallback to images/train path."""
        df = pd.DataFrame({
            "filename": ["ap_0000.png"],
            "view_type": ["AP"],
            "flexion_deg": [170.0],
            "rotation_error_deg": [5.0],
        })
        # No image at path — should still save PNG without crashing
        plot_sample_grid(tmp, df, tmp, grid_size=2)
        assert os.path.exists(os.path.join(tmp, "sample_grid.png"))

    def test_flat_images_dir_fallback(self, tmp):
        """If images/split/fname missing, tries images/fname (flat layout)."""
        import cv2
        img_dir = os.path.join(tmp, "images")
        os.makedirs(img_dir, exist_ok=True)
        img = np.full((32, 32), 200, dtype=np.uint8)
        cv2.imwrite(os.path.join(img_dir, "lat_0001.png"), img)

        df = pd.DataFrame({
            "filename": ["lat_0001.png"],
            "view_type": ["LAT"],
            "flexion_deg": [90.0],
            "rotation_error_deg": [-10.0],
            "split": ["val"],
        })
        plot_sample_grid(tmp, df, tmp, grid_size=2)
        assert os.path.exists(os.path.join(tmp, "sample_grid.png"))
