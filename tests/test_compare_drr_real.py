"""Tests for pure functions in scripts/compare_drr_real.py.

Covers:
  extract_bone_region, compute_edge, compute_ssim,
  compute_fft_profile, histogram_intersection, compute_contrast_ratio
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import target module
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import compare_drr_real as _m  # noqa: E402

extract_bone_region = _m.extract_bone_region
compute_edge = _m.compute_edge
compute_ssim = _m.compute_ssim
compute_fft_profile = _m.compute_fft_profile
histogram_intersection = _m.histogram_intersection
compute_contrast_ratio = _m.compute_contrast_ratio


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def uniform_gray() -> np.ndarray:
    return np.full((256, 256), 128, dtype=np.uint8)


@pytest.fixture()
def high_contrast() -> np.ndarray:
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(img, (128, 128), 60, 200, -1)
    return img


@pytest.fixture()
def two_region() -> np.ndarray:
    img = np.zeros((128, 128), dtype=np.uint8)
    img[:, :64] = 50
    img[:, 64:] = 200
    return img


# ===========================================================================
# TestExtractBoneRegion
# ===========================================================================


class TestExtractBoneRegion:
    def test_output_shape_preserved(self, high_contrast):
        mask = extract_bone_region(high_contrast)
        assert mask.shape == high_contrast.shape

    def test_output_dtype_uint8(self, high_contrast):
        mask = extract_bone_region(high_contrast)
        assert mask.dtype == np.uint8

    def test_binary_values_only(self, high_contrast):
        mask = extract_bone_region(high_contrast)
        unique = set(np.unique(mask))
        assert unique.issubset({0, 255})

    def test_bright_circle_detected_as_bone(self, high_contrast):
        mask = extract_bone_region(high_contrast)
        # Center of circle should be foreground
        assert mask[128, 128] == 255

    def test_dark_background_is_zero(self, high_contrast):
        mask = extract_bone_region(high_contrast)
        # Corner pixels should be background
        assert mask[0, 0] == 0
        assert mask[255, 255] == 0

    def test_two_region_separates_correctly(self, two_region):
        mask = extract_bone_region(two_region)
        # Bright right half should be foreground, dark left half background
        assert mask[64, 100] == 255
        assert mask[64, 10] == 0

    def test_small_image_returns_correct_shape(self):
        img = np.array([[0, 0, 200, 200],
                        [0, 0, 200, 200]], dtype=np.uint8)
        mask = extract_bone_region(img)
        assert mask.shape == (2, 4)

    def test_all_same_value_produces_valid_mask(self, uniform_gray):
        # Otsu on uniform image: may yield all-0 or all-255, both valid
        mask = extract_bone_region(uniform_gray)
        assert mask.dtype == np.uint8
        unique = set(np.unique(mask))
        assert unique.issubset({0, 255})


# ===========================================================================
# TestComputeEdge
# ===========================================================================


class TestComputeEdge:
    def test_output_shape_preserved(self, high_contrast):
        edges = compute_edge(high_contrast)
        assert edges.shape == high_contrast.shape

    def test_output_dtype_uint8(self, high_contrast):
        edges = compute_edge(high_contrast)
        assert edges.dtype == np.uint8

    def test_binary_values_only(self, high_contrast):
        edges = compute_edge(high_contrast)
        unique = set(np.unique(edges))
        assert unique.issubset({0, 255})

    def test_uniform_image_no_edges(self, uniform_gray):
        edges = compute_edge(uniform_gray)
        assert np.all(edges == 0)

    def test_circle_has_edges(self, high_contrast):
        edges = compute_edge(high_contrast)
        assert edges.max() == 255
        assert np.sum(edges > 0) > 0

    def test_step_edge_detected(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[:, 32:] = 255
        edges = compute_edge(img)
        # Some pixels along the step should be detected
        assert np.sum(edges > 0) > 0

    def test_rectangular_input(self):
        img = np.zeros((128, 64), dtype=np.uint8)
        cv2.circle(img, (32, 64), 20, 200, -1)
        edges = compute_edge(img)
        assert edges.shape == (128, 64)
        assert edges.dtype == np.uint8

    def test_all_white_no_edges(self):
        img = np.full((64, 64), 255, dtype=np.uint8)
        edges = compute_edge(img)
        assert np.all(edges == 0)


# ===========================================================================
# TestComputeSSIM
# ===========================================================================


class TestComputeSSIM:
    def test_identical_images_return_one(self, high_contrast):
        val = compute_ssim(high_contrast, high_contrast)
        assert abs(val - 1.0) < 1e-5

    def test_returns_float(self, high_contrast):
        val = compute_ssim(high_contrast, high_contrast)
        assert isinstance(val, float)

    def test_range_lower_bound(self, high_contrast):
        inverted = 255 - high_contrast
        val = compute_ssim(high_contrast, inverted)
        assert val >= -1.0

    def test_range_upper_bound(self, high_contrast):
        val = compute_ssim(high_contrast, high_contrast)
        assert val <= 1.0 + 1e-9

    def test_symmetry(self, high_contrast, uniform_gray):
        a = compute_ssim(high_contrast, uniform_gray)
        b = compute_ssim(uniform_gray, high_contrast)
        assert abs(a - b) < 1e-6

    def test_inverted_less_than_one(self, high_contrast):
        inverted = 255 - high_contrast
        val = compute_ssim(high_contrast, inverted)
        assert val < 1.0

    def test_noisy_less_than_identical(self, high_contrast):
        rng = np.random.default_rng(0)
        noise = rng.integers(-20, 20, high_contrast.shape, dtype=np.int16)
        noisy = np.clip(high_contrast.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        ssim_noisy = compute_ssim(high_contrast, noisy)
        ssim_identical = compute_ssim(high_contrast, high_contrast)
        assert ssim_noisy < ssim_identical

    def test_uniform_identical_returns_one(self, uniform_gray):
        val = compute_ssim(uniform_gray, uniform_gray)
        assert abs(val - 1.0) < 1e-5

    def test_checkerboard_vs_uniform_less_than_one(self, uniform_gray):
        checker = np.indices((256, 256)).sum(axis=0) % 2
        checker = (checker * 255).astype(np.uint8)
        val = compute_ssim(checker, uniform_gray)
        assert val < 1.0


# ===========================================================================
# TestComputeFFTProfile
# ===========================================================================


class TestComputeFFTProfile:
    def test_returns_ndarray(self, uniform_gray):
        profile = compute_fft_profile(uniform_gray)
        assert isinstance(profile, np.ndarray)

    def test_length_for_256x256(self, uniform_gray):
        profile = compute_fft_profile(uniform_gray)
        assert len(profile) == 128  # min(256//2, 256//2)

    def test_length_for_128x128(self):
        img = np.zeros((128, 128), dtype=np.uint8)
        profile = compute_fft_profile(img)
        assert len(profile) == 64  # min(64, 64)

    def test_non_negative_values(self, high_contrast):
        profile = compute_fft_profile(high_contrast)
        assert np.all(profile >= 0)

    def test_dc_dominates_for_uniform(self, uniform_gray):
        profile = compute_fft_profile(uniform_gray)
        # DC component (index 0) should be dominant for a uniform image
        assert profile[0] == profile.max()

    def test_non_square_uses_min_dim_half(self):
        img = np.zeros((128, 64), dtype=np.uint8)
        profile = compute_fft_profile(img)
        # max_r = min(128//2, 64//2) = min(64, 32) = 32
        assert len(profile) == 32

    def test_different_images_give_different_profiles(self, uniform_gray, high_contrast):
        p1 = compute_fft_profile(uniform_gray)
        p2 = compute_fft_profile(high_contrast)
        assert not np.allclose(p1, p2)

    def test_flat_image_only_dc(self):
        img = np.full((128, 128), 100, dtype=np.uint8)
        profile = compute_fft_profile(img)
        # All power should be at DC (index 0); higher frequencies near zero
        assert profile[0] >= profile[1:].max()


# ===========================================================================
# TestHistogramIntersection
# ===========================================================================


class TestHistogramIntersection:
    def test_identical_returns_one(self):
        h = np.array([10, 20, 30, 40], dtype=np.float64)
        val = histogram_intersection(h, h)
        assert abs(val - 1.0) < 1e-9

    def test_disjoint_returns_zero(self):
        h1 = np.array([10, 0, 0, 0], dtype=np.float64)
        h2 = np.array([0, 0, 0, 10], dtype=np.float64)
        val = histogram_intersection(h1, h2)
        assert abs(val) < 1e-9

    def test_partial_overlap_between_zero_and_one(self):
        h1 = np.array([5, 5, 0, 0], dtype=np.float64)
        h2 = np.array([0, 5, 5, 0], dtype=np.float64)
        val = histogram_intersection(h1, h2)
        assert 0.0 < val < 1.0

    def test_symmetry(self):
        h1 = np.array([3, 7, 2, 8], dtype=np.float64)
        h2 = np.array([1, 5, 9, 5], dtype=np.float64)
        assert abs(histogram_intersection(h1, h2) - histogram_intersection(h2, h1)) < 1e-9

    def test_result_in_range_0_1(self):
        rng = np.random.default_rng(42)
        h1 = rng.random(32)
        h2 = rng.random(32)
        val = histogram_intersection(h1, h2)
        assert 0.0 <= val <= 1.0

    def test_all_zero_inputs(self):
        h = np.zeros(8, dtype=np.float64)
        # With epsilon denominator, result should be 0.0 (no intersection)
        val = histogram_intersection(h, h)
        assert 0.0 <= val <= 1.0

    def test_single_bin_both_same_returns_one(self):
        # The epsilon denominator (1e-8) causes result ≈0.99999999, not exactly 1.0
        h1 = np.array([5.0])
        h2 = np.array([5.0])
        val = histogram_intersection(h1, h2)
        assert abs(val - 1.0) < 1e-6

    def test_one_sided_heavy_overlap_math(self):
        # h1=[10,0,0,0] → h1n=[1,0,0,0]
        # h2=[5,5,5,5]  → h2n=[0.25,0.25,0.25,0.25]
        # min=[0.25,0,0,0] → sum=0.25
        h1 = np.array([10, 0, 0, 0], dtype=np.float64)
        h2 = np.array([5, 5, 5, 5], dtype=np.float64)
        val = histogram_intersection(h1, h2)
        assert abs(val - 0.25) < 1e-9

    def test_half_non_overlap(self):
        # h1=[1,0] → h1n=[1,0]; h2=[0,1] → h2n=[0,1]
        # min=[0,0] → sum=0.0
        h1 = np.array([1.0, 0.0])
        h2 = np.array([0.0, 1.0])
        val = histogram_intersection(h1, h2)
        assert abs(val - 0.0) < 1e-9


# ===========================================================================
# TestComputeContrastRatio
# ===========================================================================


class TestComputeContrastRatio:
    def test_returns_float(self, high_contrast):
        val = compute_contrast_ratio(high_contrast)
        assert isinstance(val, float)

    def test_high_contrast_circle_greater_than_one(self, high_contrast):
        # Bright circle on dark background → bone_mean > bg_mean → ratio > 1
        val = compute_contrast_ratio(high_contrast)
        assert val > 1.0

    def test_all_zeros_returns_zero(self):
        img = np.zeros((128, 128), dtype=np.uint8)
        val = compute_contrast_ratio(img)
        assert val == 0.0

    def test_non_negative(self, high_contrast):
        val = compute_contrast_ratio(high_contrast)
        assert val >= 0.0

    def test_two_region_greater_than_one(self, two_region):
        val = compute_contrast_ratio(two_region)
        assert val > 1.0

    def test_large_bright_blob(self):
        img = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(img, (128, 128), 100, 240, -1)
        val = compute_contrast_ratio(img)
        assert val > 1.0

    def test_uses_extract_bone_region_logic(self, high_contrast):
        # Verify that bone pixels (foreground) have higher mean than background
        mask = extract_bone_region(high_contrast)
        bone_pixels = high_contrast[mask == 255]
        bg_pixels = high_contrast[mask == 0]
        if bone_pixels.size > 0 and bg_pixels.size > 0:
            expected_ratio = float(bone_pixels.mean()) / max(float(bg_pixels.mean()), 1.0)
            val = compute_contrast_ratio(high_contrast)
            assert abs(val - expected_ratio) < 1e-5

    def test_inverted_circle_ratio_behavior(self):
        # Dark circle on bright background → after Otsu, bright bg becomes bone
        img = np.full((256, 256), 200, dtype=np.uint8)
        cv2.circle(img, (128, 128), 60, 30, -1)
        val = compute_contrast_ratio(img)
        # Result should still be non-negative
        assert val >= 0.0
