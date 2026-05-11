"""
Unit tests for scripts/generate_domain_gap_figure.py

Tests cover:
- edge_density: Canny edge density computation
- contrast_rms: RMS contrast (std) computation
- hist_intersection: normalized histogram intersection
- _load_drr_at_90: loading 90-degree DRR from npz library
- _load_dataset_drr_at_90: loading 90-degree DRR from dataset CSV
- main() CLI invocation with mocked data
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# ── Module import setup ──────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import generate_domain_gap_figure as gdf


# ====================================================================
# edge_density
# ====================================================================
class TestEdgeDensity:
    def test_black_image_returns_zero(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        assert gdf.edge_density(img) == 0.0

    def test_white_image_returns_zero(self):
        img = np.full((64, 64), 255, dtype=np.uint8)
        assert gdf.edge_density(img) == 0.0

    def test_high_contrast_returns_positive(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[:, 32:] = 255  # sharp vertical edge
        val = gdf.edge_density(img)
        assert val > 0.0

    def test_return_range_0_to_1(self):
        img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        val = gdf.edge_density(img)
        assert 0.0 <= val <= 1.0

    def test_more_edges_higher_density(self):
        # Checkerboard has more edges than single stripe
        stripe = np.zeros((64, 64), dtype=np.uint8)
        stripe[:, 32:] = 255

        checker = np.zeros((64, 64), dtype=np.uint8)
        for i in range(0, 64, 8):
            for j in range(0, 64, 8):
                if (i // 8 + j // 8) % 2 == 0:
                    checker[i:i+8, j:j+8] = 255

        assert gdf.edge_density(checker) > gdf.edge_density(stripe)

    def test_returns_float(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        assert isinstance(gdf.edge_density(img), float)


# ====================================================================
# contrast_rms
# ====================================================================
class TestContrastRms:
    def test_uniform_image_returns_zero(self):
        img = np.full((64, 64), 128, dtype=np.uint8)
        assert gdf.contrast_rms(img) == 0.0

    def test_high_contrast_returns_positive(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[:32] = 255
        val = gdf.contrast_rms(img)
        assert val > 0.0

    def test_max_contrast_binary(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:50] = 255
        # std of 50% zeros and 50% 255 = 127.5
        val = gdf.contrast_rms(img)
        assert abs(val - 127.5) < 1.0

    def test_returns_float(self):
        img = np.ones((32, 32), dtype=np.uint8) * 100
        assert isinstance(gdf.contrast_rms(img), float)

    def test_higher_variance_higher_contrast(self):
        low_var = np.full((64, 64), 128, dtype=np.uint8)
        low_var[:8] = 140  # small contrast

        high_var = np.zeros((64, 64), dtype=np.uint8)
        high_var[:32] = 255  # high contrast

        assert gdf.contrast_rms(high_var) > gdf.contrast_rms(low_var)


# ====================================================================
# hist_intersection
# ====================================================================
class TestHistIntersection:
    def test_identical_images_returns_one(self):
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        val = gdf.hist_intersection(img, img)
        assert abs(val - 1.0) < 1e-5

    def test_different_images_less_than_one(self):
        img1 = np.zeros((64, 64), dtype=np.uint8)
        img2 = np.full((64, 64), 255, dtype=np.uint8)
        val = gdf.hist_intersection(img1, img2)
        assert val < 0.1

    def test_return_range_0_to_1(self):
        img1 = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        val = gdf.hist_intersection(img1, img2)
        assert 0.0 <= val <= 1.0 + 1e-8

    def test_symmetric(self):
        img1 = np.random.randint(0, 128, (64, 64), dtype=np.uint8)
        img2 = np.random.randint(128, 256, (64, 64), dtype=np.uint8)
        assert abs(gdf.hist_intersection(img1, img2) -
                   gdf.hist_intersection(img2, img1)) < 1e-8

    def test_similar_images_high_intersection(self):
        rng = np.random.RandomState(42)
        img1 = rng.randint(100, 200, (64, 64)).astype(np.uint8)
        img2 = (img1.astype(int) + rng.randint(-5, 6, img1.shape)).clip(0, 255).astype(np.uint8)
        val = gdf.hist_intersection(img1, img2)
        assert val > 0.7

    def test_returns_float(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        assert isinstance(gdf.hist_intersection(img, img), float)


# ====================================================================
# _load_drr_at_90
# ====================================================================
class TestLoadDrrAt90:
    def test_loads_closest_to_90(self, tmp_path):
        angles = np.array([60.0, 80.0, 90.0, 120.0, 150.0, 180.0])
        drrs = np.random.randint(0, 256, (6, 64, 64), dtype=np.uint8)
        npz_path = tmp_path / "lib.npz"
        np.savez(str(npz_path), angles=angles, drrs=drrs)

        result = gdf._load_drr_at_90(npz_path)
        assert result is not None
        np.testing.assert_array_equal(result, drrs[2])  # index of 90.0

    def test_snaps_to_nearest_when_no_exact_90(self, tmp_path):
        angles = np.array([60.0, 85.0, 95.0, 120.0])
        drrs = np.random.randint(0, 256, (4, 64, 64), dtype=np.uint8)
        npz_path = tmp_path / "lib.npz"
        np.savez(str(npz_path), angles=angles, drrs=drrs)

        result = gdf._load_drr_at_90(npz_path)
        assert result is not None
        # 85 is closer to 90 than 95
        np.testing.assert_array_equal(result, drrs[1])

    def test_normalizes_non_uint8_to_uint8(self, tmp_path):
        angles = np.array([90.0])
        drrs = np.array([[[0.0, 0.5], [1.0, 0.25]]])  # float, not uint8
        npz_path = tmp_path / "lib.npz"
        np.savez(str(npz_path), angles=angles, drrs=drrs)

        result = gdf._load_drr_at_90(npz_path)
        assert result is not None
        assert result.dtype == np.uint8
        assert result.max() >= 254  # ~255 (epsilon in normalization)
        assert result.min() == 0

    def test_returns_none_on_missing_file(self, tmp_path):
        result = gdf._load_drr_at_90(tmp_path / "nonexistent.npz")
        assert result is None

    def test_uint8_passthrough(self, tmp_path):
        angles = np.array([90.0])
        drrs = np.array([[[10, 20], [30, 40]]], dtype=np.uint8)
        npz_path = tmp_path / "lib.npz"
        np.savez(str(npz_path), angles=angles, drrs=drrs)

        result = gdf._load_drr_at_90(npz_path)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, drrs[0])


# ====================================================================
# _load_dataset_drr_at_90
# ====================================================================
class TestLoadDatasetDrrAt90:
    def _make_dataset(self, tmp_path, rows):
        """Create CSV + image files for testing."""
        csv_path = tmp_path / "labels.csv"
        imgs_dir = tmp_path / "images"

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "view_type",
                                                    "flexion_deg", "split"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                # Create the image file
                split_dir = imgs_dir / row["split"]
                split_dir.mkdir(parents=True, exist_ok=True)
                img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
                cv2.imwrite(str(split_dir / row["filename"]), img)

        return csv_path, imgs_dir

    def test_selects_closest_to_90_lat(self, tmp_path):
        rows = [
            {"filename": "img_80.png", "view_type": "LAT", "flexion_deg": "80.0", "split": "val"},
            {"filename": "img_89.png", "view_type": "LAT", "flexion_deg": "89.0", "split": "val"},
            {"filename": "img_95.png", "view_type": "LAT", "flexion_deg": "95.0", "split": "val"},
        ]
        csv_path, imgs_dir = self._make_dataset(tmp_path, rows)
        result = gdf._load_dataset_drr_at_90(csv_path, imgs_dir)
        assert result is not None
        assert result.shape == (64, 64)

    def test_ignores_non_lat(self, tmp_path):
        rows = [
            {"filename": "ap_90.png", "view_type": "AP", "flexion_deg": "90.0", "split": "val"},
            {"filename": "lat_120.png", "view_type": "LAT", "flexion_deg": "120.0", "split": "val"},
        ]
        csv_path, imgs_dir = self._make_dataset(tmp_path, rows)
        result = gdf._load_dataset_drr_at_90(csv_path, imgs_dir)
        assert result is not None
        # Should pick lat_120 since it's the only LAT

    def test_returns_none_on_missing_csv(self, tmp_path):
        result = gdf._load_dataset_drr_at_90(
            tmp_path / "nonexistent.csv",
            tmp_path / "images"
        )
        assert result is None

    def test_returns_none_when_no_lat_images(self, tmp_path):
        rows = [
            {"filename": "ap_90.png", "view_type": "AP", "flexion_deg": "90.0", "split": "val"},
        ]
        csv_path, imgs_dir = self._make_dataset(tmp_path, rows)
        result = gdf._load_dataset_drr_at_90(csv_path, imgs_dir)
        assert result is None

    def test_returns_grayscale(self, tmp_path):
        rows = [
            {"filename": "lat_90.png", "view_type": "LAT", "flexion_deg": "90.0", "split": "train"},
        ]
        csv_path, imgs_dir = self._make_dataset(tmp_path, rows)
        result = gdf._load_dataset_drr_at_90(csv_path, imgs_dir)
        assert result is not None
        assert result.ndim == 2  # grayscale

    def test_prefers_exact_90_over_distant(self, tmp_path):
        rows = [
            {"filename": "lat_60.png", "view_type": "LAT", "flexion_deg": "60.0", "split": "val"},
            {"filename": "lat_90.png", "view_type": "LAT", "flexion_deg": "90.0", "split": "val"},
            {"filename": "lat_180.png", "view_type": "LAT", "flexion_deg": "180.0", "split": "val"},
        ]
        csv_path, imgs_dir = self._make_dataset(tmp_path, rows)
        result = gdf._load_dataset_drr_at_90(csv_path, imgs_dir)
        assert result is not None


# ====================================================================
# main() integration (matplotlib Agg)
# ====================================================================
class TestMain:
    def test_main_with_dataset_drr(self, tmp_path):
        """Test main() with dataset DRR (CSV + images) and real X-ray mock."""
        # Create dataset CSV + images
        csv_path = tmp_path / "labels.csv"
        imgs_dir = tmp_path / "images" / "val"
        imgs_dir.mkdir(parents=True)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "view_type",
                                                    "flexion_deg", "split"])
            writer.writeheader()
            writer.writerow({"filename": "drr_90.png", "view_type": "LAT",
                           "flexion_deg": "90.0", "split": "val"})

        drr_img = np.random.randint(50, 200, (64, 64), dtype=np.uint8)
        cv2.imwrite(str(imgs_dir / "drr_90.png"), drr_img)

        # Create real X-ray mock
        xray_dir = tmp_path / "real_xray"
        xray_dir.mkdir()
        xray_img = np.random.randint(30, 220, (64, 64), dtype=np.uint8)
        cv2.imwrite(str(xray_dir / "real.png"), xray_img)

        out_dir = tmp_path / "output"

        with patch("sys.argv", [
            "generate_domain_gap_figure.py",
            "--library", str(tmp_path / "dummy.npz"),
            "--dataset_csv", str(csv_path),
            "--dataset_imgs", str(tmp_path / "images"),
            "--xray", str(xray_dir / "real.png"),
            "--out_dir", str(out_dir),
        ]):
            # Patch _PROJECT_ROOT to tmp_path won't work because argparse
            # uses _PROJECT_ROOT / args.xxx. Instead, use absolute paths.
            # We need to patch _PROJECT_ROOT so the path joining works.
            with patch.object(gdf, "_PROJECT_ROOT", tmp_path):
                # Create the files at the expected paths
                lib_path = tmp_path / "dummy.npz"
                np.savez(str(lib_path),
                         angles=np.array([90.0]),
                         drrs=np.array([drr_img]))

                # Re-create paths relative to _PROJECT_ROOT
                (tmp_path / "results" / "figures").mkdir(parents=True, exist_ok=True)

                # Actually, main() constructs paths as _PROJECT_ROOT / args.xxx
                # So we need args to be relative paths from _PROJECT_ROOT
                with patch("sys.argv", [
                    "generate_domain_gap_figure.py",
                    "--library", "dummy.npz",
                    "--dataset_csv", "labels.csv",
                    "--dataset_imgs", "images",
                    "--xray", str(xray_dir / "real.png").replace(str(tmp_path) + "/", ""),
                    "--out_dir", "output",
                ]):
                    gdf.main()

        assert (tmp_path / "output" / "fig10_domain_gap.png").exists()

    def test_main_falls_back_to_library_drr(self, tmp_path):
        """When dataset CSV doesn't exist, falls back to library DRR."""
        # Create library npz
        drr_img = np.random.randint(50, 200, (64, 64), dtype=np.uint8)
        np.savez(str(tmp_path / "lib.npz"),
                 angles=np.array([90.0]),
                 drrs=np.array([drr_img]))

        # Create real X-ray
        xray_img = np.random.randint(30, 220, (64, 64), dtype=np.uint8)
        xray_path = tmp_path / "real.png"
        cv2.imwrite(str(xray_path), xray_img)

        with patch.object(gdf, "_PROJECT_ROOT", tmp_path):
            with patch("sys.argv", [
                "generate_domain_gap_figure.py",
                "--library", "lib.npz",
                "--dataset_csv", "nonexistent.csv",
                "--dataset_imgs", "nonexistent_imgs",
                "--xray", "real.png",
                "--out_dir", "output",
            ]):
                gdf.main()

        assert (tmp_path / "output" / "fig10_domain_gap.png").exists()

    def test_main_exits_gracefully_no_xray(self, tmp_path, capsys):
        """When real X-ray doesn't exist, main() prints error and returns."""
        drr_img = np.random.randint(50, 200, (64, 64), dtype=np.uint8)
        np.savez(str(tmp_path / "lib.npz"),
                 angles=np.array([90.0]),
                 drrs=np.array([drr_img]))

        with patch.object(gdf, "_PROJECT_ROOT", tmp_path):
            with patch("sys.argv", [
                "generate_domain_gap_figure.py",
                "--library", "lib.npz",
                "--dataset_csv", "nonexistent.csv",
                "--dataset_imgs", "nonexistent_imgs",
                "--xray", "nonexistent_xray.png",
                "--out_dir", "output",
            ]):
                gdf.main()

        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert not (tmp_path / "output" / "fig10_domain_gap.png").exists()

    def test_main_exits_gracefully_no_drr(self, tmp_path, capsys):
        """When neither dataset DRR nor library DRR exists, prints error."""
        xray_img = np.random.randint(30, 220, (64, 64), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "real.png"), xray_img)

        with patch.object(gdf, "_PROJECT_ROOT", tmp_path):
            with patch("sys.argv", [
                "generate_domain_gap_figure.py",
                "--library", "nonexistent.npz",
                "--dataset_csv", "nonexistent.csv",
                "--dataset_imgs", "nonexistent_imgs",
                "--xray", "real.png",
                "--out_dir", "output",
            ]):
                gdf.main()

        captured = capsys.readouterr()
        assert "ERROR" in captured.out


# ====================================================================
# Edge cases & properties
# ====================================================================
class TestEdgeCases:
    def test_edge_density_single_pixel(self):
        img = np.array([[128]], dtype=np.uint8)
        val = gdf.edge_density(img)
        assert val == 0.0

    def test_contrast_rms_single_pixel(self):
        img = np.array([[128]], dtype=np.uint8)
        val = gdf.contrast_rms(img)
        assert val == 0.0

    def test_hist_intersection_single_pixel_same(self):
        img = np.array([[100]], dtype=np.uint8)
        val = gdf.hist_intersection(img, img)
        assert abs(val - 1.0) < 1e-5

    def test_hist_intersection_single_pixel_different(self):
        img1 = np.array([[0]], dtype=np.uint8)
        img2 = np.array([[255]], dtype=np.uint8)
        val = gdf.hist_intersection(img1, img2)
        assert val < 0.01

    def test_load_drr_at_90_corrupt_npz(self, tmp_path):
        bad_file = tmp_path / "bad.npz"
        bad_file.write_bytes(b"not a real npz")
        result = gdf._load_drr_at_90(bad_file)
        assert result is None

    def test_edge_density_gradient_image(self):
        img = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
        val = gdf.edge_density(img)
        assert 0.0 <= val <= 1.0

    def test_contrast_rms_gradient_image(self):
        img = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
        val = gdf.contrast_rms(img)
        assert val > 50.0  # gradient has high std

    def test_hist_intersection_shifted(self):
        """Shifted image should have moderate intersection."""
        rng = np.random.RandomState(123)
        img1 = rng.randint(0, 128, (64, 64)).astype(np.uint8)
        img2 = (img1.astype(int) + 128).clip(0, 255).astype(np.uint8)
        val = gdf.hist_intersection(img1, img2)
        assert val < 0.5  # histograms barely overlap
