"""Tests for scripts/drr_utils.py pure-function utilities.

Covers functions that operate solely on numpy arrays and do NOT require
CT volume data or elbow_synth:
  - resize_to_match
  - compute_dice
  - compute_ssim
  - compute_all_metrics
  - histogram_match
  - load_real_xray
"""
import sys
import os

import numpy as np
import pytest

# Make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from drr_utils import (
    resize_to_match,
    compute_dice,
    compute_ssim,
    compute_all_metrics,
    histogram_match,
    load_real_xray,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def make_gray(h=64, w=64, fill=128, dtype=np.uint8):
    return np.full((h, w), fill, dtype=dtype)


def make_circle(h=64, w=64, r=20, fill=200, dtype=np.uint8):
    """Return a uint8 image with a filled circle centred in the frame."""
    img = np.zeros((h, w), dtype=dtype)
    cy, cx = h // 2, w // 2
    for y in range(h):
        for x in range(w):
            if (y - cy) ** 2 + (x - cx) ** 2 < r ** 2:
                img[y, x] = fill
    return img


# ── resize_to_match ───────────────────────────────────────────────────────

class TestResizeToMatch:
    def test_same_shape_returns_original(self):
        img = make_gray(32, 32)
        target = make_gray(32, 32, fill=0)
        result = resize_to_match(img, target)
        assert result is img, "Should return the same object when shapes match"

    def test_different_shape_returns_resized(self):
        img = make_gray(32, 32)
        target = make_gray(64, 64, fill=0)
        result = resize_to_match(img, target)
        assert result.shape == target.shape

    def test_height_only_differs(self):
        img = make_gray(16, 64)
        target = make_gray(32, 64, fill=0)
        result = resize_to_match(img, target)
        assert result.shape == (32, 64)

    def test_width_only_differs(self):
        img = make_gray(64, 16)
        target = make_gray(64, 32, fill=0)
        result = resize_to_match(img, target)
        assert result.shape == (64, 32)


# ── compute_dice ──────────────────────────────────────────────────────────

class TestComputeDice:
    def test_identical_images_returns_one(self):
        img = make_circle()
        dice = compute_dice(img, img)
        assert 0.99 <= dice <= 1.0, f"Expected ~1.0, got {dice}"

    def test_no_overlap_returns_near_zero(self):
        # All-zero pred vs bright circle gt → no overlap after Otsu
        pred = np.zeros((64, 64), dtype=np.uint8)
        gt = make_circle(fill=200)
        dice = compute_dice(pred, gt)
        assert dice < 0.05, f"Expected ~0, got {dice}"

    def test_range_is_zero_to_one(self):
        a = make_circle(fill=200)
        b = make_circle(r=10, fill=200)
        dice = compute_dice(a, b)
        assert 0.0 <= dice <= 1.0

    def test_fixed_method(self):
        img = make_circle(fill=200)
        dice = compute_dice(img, img, method='fixed')
        assert 0.99 <= dice <= 1.0

    def test_mismatched_shapes_handled(self):
        pred = make_circle(h=32, w=32, fill=200)
        gt = make_circle(h=64, w=64, fill=200)
        dice = compute_dice(pred, gt)
        assert 0.0 <= dice <= 1.0

    def test_symmetry(self):
        a = make_circle(r=15, fill=200)
        b = make_circle(r=10, fill=200)
        assert abs(compute_dice(a, b) - compute_dice(b, a)) < 1e-6


# ── compute_ssim ──────────────────────────────────────────────────────────

class TestComputeSsim:
    def test_identical_images_returns_one(self):
        img = make_circle()
        val = compute_ssim(img, img)
        assert abs(val - 1.0) < 1e-4, f"Expected ~1.0, got {val}"

    def test_range_is_minus_one_to_one(self):
        a = make_circle(fill=200)
        b = make_gray(fill=10)
        val = compute_ssim(a, b)
        assert -1.0 <= val <= 1.0

    def test_mismatched_shapes_handled(self):
        pred = make_circle(h=32, w=32, fill=200)
        gt = make_circle(h=64, w=64, fill=200)
        val = compute_ssim(pred, gt)
        assert -1.0 <= val <= 1.0

    def test_high_similarity_close_images(self):
        base = make_circle(fill=200)
        noise = (base.astype(np.int16) + np.random.randint(-5, 5, base.shape)).clip(0, 255).astype(np.uint8)
        val = compute_ssim(noise, base)
        assert val > 0.8, f"Similar images should have high SSIM, got {val}"


# ── compute_all_metrics ───────────────────────────────────────────────────

class TestComputeAllMetrics:
    def test_returns_two_floats(self):
        img = make_circle()
        result = compute_all_metrics(img, img)
        assert len(result) == 2
        s, d = result
        assert isinstance(s, float)
        assert isinstance(d, float)

    def test_identical_images_both_near_one(self):
        img = make_circle()
        s, d = compute_all_metrics(img, img)
        assert abs(s - 1.0) < 1e-4
        assert d >= 0.99

    def test_ssim_first_dice_second(self):
        img = make_circle()
        s, d = compute_all_metrics(img, img)
        # SSIM for identical images is exactly 1; Dice is ≥ 0.99
        assert s >= d * 0.99


# ── histogram_match ───────────────────────────────────────────────────────

class TestHistogramMatch:
    def test_output_shape_matches_source(self):
        source = make_gray(32, 32, fill=80)
        template = make_gray(64, 64, fill=200)
        result = histogram_match(source, template)
        assert result.shape == source.shape

    def test_output_dtype_is_uint8(self):
        source = make_gray(32, 32, fill=80)
        template = make_gray(32, 32, fill=200)
        result = histogram_match(source, template)
        assert result.dtype == np.uint8

    def test_flat_source_maps_to_flat_template(self):
        """All-80 source matched to all-200 template → all-200 output."""
        source = make_gray(32, 32, fill=80)
        template = make_gray(32, 32, fill=200)
        result = histogram_match(source, template)
        assert np.all(result == 200)

    def test_identical_source_and_template(self):
        """Matching a source to itself should return the source unchanged."""
        img = make_circle(fill=180)
        result = histogram_match(img, img)
        np.testing.assert_array_equal(result, img)

    def test_values_in_valid_range(self):
        source = make_circle(fill=100)
        template = make_circle(fill=200)
        result = histogram_match(source, template)
        assert result.min() >= 0
        assert result.max() <= 255


# ── load_real_xray ────────────────────────────────────────────────────────

class TestLoadRealXray:
    def test_missing_paths_returns_none(self, tmp_path):
        result = load_real_xray(
            paths=[str(tmp_path / "nonexistent.png")],
            verbose=False,
        )
        assert result is None

    def test_empty_paths_returns_none(self):
        result = load_real_xray(paths=[], verbose=False)
        assert result is None

    def test_valid_image_is_loaded(self, tmp_path):
        import cv2
        img = make_circle(fill=180)
        path = str(tmp_path / "test.png")
        cv2.imwrite(path, img)
        result = load_real_xray(paths=[path], verbose=False)
        assert result is not None
        assert result.ndim == 2

    def test_second_path_used_as_fallback(self, tmp_path):
        import cv2
        img = make_gray(fill=50)
        bad_path = str(tmp_path / "missing.png")
        good_path = str(tmp_path / "real.png")
        cv2.imwrite(good_path, img)
        result = load_real_xray(paths=[bad_path, good_path], verbose=False)
        assert result is not None
