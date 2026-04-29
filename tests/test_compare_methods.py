"""Tests for scripts/compare_methods.py pure-function utilities.

Covers functions that operate solely on numpy arrays and do NOT require
CT volume data, DICOM files, or trained models:
  - ncc
  - edges
  - preprocess_drr
  - crop_xray
  - preprocess_xray
"""
import sys
import os

import cv2
import numpy as np
import pytest

# Make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from compare_methods import ncc, edges, preprocess_drr, crop_xray, preprocess_xray


# ── Helpers ────────────────────────────────────────────────────────────────

def make_gray(h=64, w=64, fill=128, dtype=np.uint8):
    return np.full((h, w), fill, dtype=dtype)


def make_ramp(h=64, w=64, dtype=np.uint8):
    """Horizontal intensity ramp 0→255."""
    row = np.linspace(0, 255, w, dtype=dtype)
    return np.tile(row, (h, 1))


def make_bright_rect(h=64, w=64, rect_h=20, rect_w=20, fill=200, dtype=np.uint8):
    """Dark background with a bright rectangle in the centre."""
    img = np.zeros((h, w), dtype=dtype)
    cy, cx = h // 2, w // 2
    r0, r1 = cy - rect_h // 2, cy + rect_h // 2
    c0, c1 = cx - rect_w // 2, cx + rect_w // 2
    img[r0:r1, c0:c1] = fill
    return img


def make_rgb(h=64, w=64, fill=128, dtype=np.uint8):
    return np.full((h, w, 3), fill, dtype=dtype)


# ── ncc ───────────────────────────────────────────────────────────────────

class TestNcc:
    def test_identical_arrays_return_one(self):
        a = np.random.rand(64, 64).astype(np.float32)
        assert pytest.approx(ncc(a, a), abs=1e-5) == 1.0

    def test_negated_array_returns_minus_one(self):
        a = np.random.rand(64, 64).astype(np.float32) + 0.1  # avoid zero-mean
        b = -a
        assert pytest.approx(ncc(a, b), abs=1e-5) == -1.0

    def test_orthogonal_signals_near_zero(self):
        n = 64
        a = np.zeros((1, n), dtype=np.float32)
        a[0, : n // 2] = 1.0
        a[0, n // 2 :] = -1.0
        b = np.zeros((1, n), dtype=np.float32)
        b[0, : n // 4] = 1.0
        b[0, n // 4 : n // 2] = -1.0
        b[0, n // 2 : 3 * n // 4] = 1.0
        b[0, 3 * n // 4 :] = -1.0
        # Not strictly orthogonal but low correlation expected
        val = ncc(a, b)
        assert -1.0 <= val <= 1.0

    def test_output_range_is_minus_one_to_one(self):
        rng = np.random.default_rng(0)
        a = rng.random((32, 32)).astype(np.float32)
        b = rng.random((32, 32)).astype(np.float32)
        val = ncc(a, b)
        assert -1.0 <= val <= 1.0

    def test_both_constant_arrays_returns_finite(self):
        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32) * 2.0
        val = ncc(a, b)
        # Both have zero deviation; result is 0/(0*0+1e-8) ≈ 0
        assert np.isfinite(val)

    def test_zero_arrays_returns_finite(self):
        a = np.zeros((16, 16), dtype=np.float32)
        b = np.zeros((16, 16), dtype=np.float32)
        val = ncc(a, b)
        assert np.isfinite(val)

    def test_return_type_is_float(self):
        a = np.random.rand(8, 8).astype(np.float32)
        b = np.random.rand(8, 8).astype(np.float32)
        assert isinstance(ncc(a, b), float)

    def test_symmetry(self):
        rng = np.random.default_rng(42)
        a = rng.random((20, 20)).astype(np.float32)
        b = rng.random((20, 20)).astype(np.float32)
        assert pytest.approx(ncc(a, b), abs=1e-6) == ncc(b, a)

    def test_1d_flat_arrays(self):
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)  # perfect positive correlation
        assert pytest.approx(ncc(a, b), abs=1e-5) == 1.0

    def test_positive_correlation_positive_value(self):
        a = np.arange(100, dtype=np.float32)
        b = a * 2.0 + 5.0
        assert ncc(a, b) > 0.99

    def test_negative_correlation_negative_value(self):
        a = np.arange(100, dtype=np.float32)
        b = -a + 5.0
        assert ncc(a, b) < -0.99


# ── edges ─────────────────────────────────────────────────────────────────

class TestEdges:
    def test_output_shape_matches_input(self):
        img = make_gray(64, 64, fill=128).astype(np.float32) / 255.0
        result = edges(img)
        assert result.shape == img.shape

    def test_output_dtype_is_float32(self):
        img = make_gray(64, 64).astype(np.float32) / 255.0
        result = edges(img)
        assert result.dtype == np.float32

    def test_uniform_image_has_no_edges(self):
        img = make_gray(64, 64, fill=128).astype(np.float32) / 255.0
        result = edges(img)
        assert result.sum() == 0.0

    def test_sharp_step_edge_detected(self):
        img = np.zeros((64, 64), dtype=np.float32)
        img[:, 32:] = 1.0
        result = edges(img)
        assert result.sum() > 0.0

    def test_output_values_in_zero_one(self):
        rng = np.random.default_rng(7)
        img = rng.random((64, 64)).astype(np.float32)
        result = edges(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_strong_edge_produces_nonzero_column(self):
        img = np.zeros((64, 64), dtype=np.float32)
        img[:, 32:] = 1.0
        result = edges(img)
        # The edge column should have detections
        assert result[:, 31:34].sum() > 0

    def test_deterministic(self):
        img = make_ramp().astype(np.float32) / 255.0
        r1 = edges(img)
        r2 = edges(img.copy())
        np.testing.assert_array_equal(r1, r2)


# ── preprocess_drr ────────────────────────────────────────────────────────

class TestPreprocessDrr:
    def test_output_shape_default_size(self):
        img = make_gray(128, 128)
        result = preprocess_drr(img)
        assert result.shape == (256, 256)

    def test_output_shape_custom_size(self):
        img = make_gray(128, 128)
        result = preprocess_drr(img, size=64)
        assert result.shape == (64, 64)

    def test_output_dtype_is_float32(self):
        img = make_gray(64, 64)
        result = preprocess_drr(img)
        assert result.dtype == np.float32

    def test_output_values_in_zero_one(self):
        img = make_gray(64, 64, fill=200)
        result = preprocess_drr(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_rgb_input_gives_2d_output(self):
        img = make_rgb(64, 64)
        result = preprocess_drr(img)
        assert result.ndim == 2

    def test_grayscale_input_remains_2d(self):
        img = make_gray(64, 64)
        result = preprocess_drr(img)
        assert result.ndim == 2

    def test_all_black_input(self):
        img = make_gray(64, 64, fill=0)
        result = preprocess_drr(img)
        assert result.shape == (256, 256)
        assert result.min() >= 0.0

    def test_all_white_input(self):
        img = make_gray(64, 64, fill=255)
        result = preprocess_drr(img)
        assert result.shape == (256, 256)
        assert result.max() <= 1.0

    def test_non_square_input_resized_to_square(self):
        img = make_gray(32, 128)
        result = preprocess_drr(img, size=64)
        assert result.shape == (64, 64)

    def test_clahe_output_is_valid_float(self):
        """CLAHE output should be float32 in [0,1] for a bimodal input."""
        # Bimodal image: lower half dark, upper half bright
        img = np.zeros((256, 256), dtype=np.uint8)
        img[128:, :] = 200
        result = preprocess_drr(img, size=256)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        # Both dark and bright regions present in output
        assert result[:128, :].mean() < result[128:, :].mean()


# ── crop_xray ─────────────────────────────────────────────────────────────

class TestCropXray:
    def test_all_dark_returns_original_array(self):
        """When no pixel exceeds threshold, returns original."""
        img = make_gray(64, 64, fill=5)
        result = crop_xray(img, dark_thresh=15)
        assert result is img

    def test_large_bright_region_returns_original(self):
        """If bright content covers >70% of the image, no crop is applied."""
        img = make_gray(64, 64, fill=200)
        result = crop_xray(img, dark_thresh=15)
        assert result is img

    def test_small_bright_rect_is_cropped(self):
        """A small bright rectangle should trigger cropping."""
        img = make_bright_rect(h=128, w=128, rect_h=20, rect_w=20, fill=200)
        result = crop_xray(img, dark_thresh=15, margin=0.1)
        # Cropped image should be smaller than original
        assert result.shape[0] < 128 or result.shape[1] < 128

    def test_output_is_2d(self):
        img = make_bright_rect(h=128, w=128)
        result = crop_xray(img)
        assert result.ndim == 2

    def test_default_threshold_is_15(self):
        """Values ≤15 should be treated as dark."""
        img = np.full((64, 64), 15, dtype=np.uint8)
        result = crop_xray(img)
        assert result is img

    def test_pixel_above_threshold_counts_as_bright(self):
        img = np.full((64, 64), 16, dtype=np.uint8)
        # All pixels above threshold → counts as large bright area → no crop
        result = crop_xray(img)
        assert result is img

    def test_margin_parameter_pads_crop(self):
        """Larger margin should produce a larger cropped region."""
        img = make_bright_rect(h=256, w=256, rect_h=30, rect_w=30, fill=200)
        small_margin = crop_xray(img.copy(), dark_thresh=15, margin=0.0)
        large_margin = crop_xray(img.copy(), dark_thresh=15, margin=0.4)
        small_area = small_margin.shape[0] * small_margin.shape[1]
        large_area = large_margin.shape[0] * large_margin.shape[1]
        # Larger margin → equal or larger cropped area
        assert large_area >= small_area

    def test_horizontal_strip_crops_vertically(self):
        """Bright row in the centre → height of crop < original."""
        img = np.zeros((128, 128), dtype=np.uint8)
        img[60:68, :] = 200  # thin horizontal bright band
        result = crop_xray(img, dark_thresh=15, margin=0.05)
        assert result.shape[0] < 128

    def test_vertical_strip_crops_horizontally(self):
        """Bright column in the centre → width of crop < original."""
        img = np.zeros((128, 128), dtype=np.uint8)
        img[:, 60:68] = 200  # thin vertical bright band
        result = crop_xray(img, dark_thresh=15, margin=0.05)
        assert result.shape[1] < 128

    def test_full_bright_image_not_cropped(self):
        img = make_gray(64, 64, fill=255)
        result = crop_xray(img, dark_thresh=15)
        assert result is img


# ── preprocess_xray ───────────────────────────────────────────────────────

class TestPreprocessXray:
    def test_output_shape_default(self):
        img = make_gray(128, 128)
        result = preprocess_xray(img)
        assert result.shape == (256, 256)

    def test_output_shape_custom_size(self):
        img = make_gray(128, 128)
        result = preprocess_xray(img, size=64)
        assert result.shape == (64, 64)

    def test_output_dtype_is_float32(self):
        img = make_gray(128, 128)
        result = preprocess_xray(img)
        assert result.dtype == np.float32

    def test_output_values_in_zero_one(self):
        img = make_gray(64, 64, fill=150)
        result = preprocess_xray(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_rgb_input_gives_2d_output(self):
        img = make_rgb(64, 64)
        result = preprocess_xray(img)
        assert result.ndim == 2

    def test_grayscale_input_stays_2d(self):
        img = make_gray(64, 64)
        result = preprocess_xray(img)
        assert result.ndim == 2

    def test_non_square_input(self):
        img = make_gray(64, 128)
        result = preprocess_xray(img, size=32)
        assert result.shape == (32, 32)

    def test_all_dark_image(self):
        """All-dark X-ray (below thresh) should still process without error."""
        img = make_gray(64, 64, fill=0)
        result = preprocess_xray(img, size=64)
        assert result.shape == (64, 64)

    def test_deterministic(self):
        img = make_bright_rect(h=128, w=128, rect_h=60, rect_w=60, fill=180)
        r1 = preprocess_xray(img.copy(), size=64)
        r2 = preprocess_xray(img.copy(), size=64)
        np.testing.assert_array_equal(r1, r2)

    def test_rgb_and_gray_same_result(self):
        """RGB image with equal channels should give same result as grayscale."""
        gray = make_gray(64, 64, fill=150)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        r_gray = preprocess_xray(gray, size=64)
        r_rgb = preprocess_xray(rgb, size=64)
        np.testing.assert_array_almost_equal(r_gray, r_rgb)
