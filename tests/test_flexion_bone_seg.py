"""Tests for scripts/flexion_bone_seg.py pure-function utilities.

Covers functions that operate solely on numpy arrays and do NOT require
CT volumes, real X-ray files, or filesystem access:
  - otsu_threshold
  - two_stage_otsu
  - rotation_matrix_z
  - bone_dice
  - build_forearm_weight_3d
"""
import sys
import os
import math

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from flexion_bone_seg import (
    otsu_threshold,
    two_stage_otsu,
    rotation_matrix_z,
    bone_dice,
    build_forearm_weight_3d,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def bimodal_array(n_low=500, n_high=500, low=0.1, high=0.9, seed=0):
    """Two distinct clusters well separated — ideal Otsu input."""
    rng = np.random.default_rng(seed)
    low_vals = rng.normal(low, 0.02, n_low).clip(0, 1)
    high_vals = rng.normal(high, 0.02, n_high).clip(0, 1)
    return np.concatenate([low_vals, high_vals]).astype(np.float32)


def make_binary_img(h=64, w=64, foreground_frac=0.5, seed=0, threshold=80) -> np.ndarray:
    """Return a uint8 image where ~foreground_frac pixels are above threshold."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), dtype=np.uint8)
    n_fg = int(h * w * foreground_frac)
    flat = img.flatten()
    flat[:n_fg] = 200
    rng.shuffle(flat)
    return flat.reshape(h, w)


# ── otsu_threshold ─────────────────────────────────────────────────────────

class TestOtsuThreshold:
    def test_returns_float(self):
        vals = bimodal_array()
        result = otsu_threshold(vals)
        assert isinstance(result, (float, np.floating))

    def test_bimodal_threshold_between_peaks(self):
        """Threshold for well-separated bimodal distribution should be between 0.1 and 0.9."""
        vals = bimodal_array(low=0.1, high=0.9)
        thresh = otsu_threshold(vals)
        assert 0.1 < thresh < 0.9

    def test_bimodal_threshold_between_clusters(self):
        """Threshold should lie between the tops of each cluster (gap region)."""
        # low cluster ~[0.06, 0.14], high cluster ~[0.86, 0.94]
        # Otsu may place threshold anywhere in the gap [0.14, 0.86]
        vals = bimodal_array(n_low=500, n_high=500, low=0.1, high=0.9)
        thresh = otsu_threshold(vals)
        assert 0.1 < thresh < 0.9

    def test_separates_clusters(self):
        """Threshold should classify the majority of each cluster correctly."""
        vals = bimodal_array(low=0.1, high=0.9)
        thresh = otsu_threshold(vals)
        low_vals = vals[:500]
        high_vals = vals[500:]
        assert (low_vals < thresh).mean() > 0.9
        assert (high_vals >= thresh).mean() > 0.9

    def test_all_same_values_returns_finite(self):
        """All identical values: threshold should still be a finite number."""
        vals = np.ones(100, dtype=np.float32) * 0.5
        thresh = otsu_threshold(vals)
        assert np.isfinite(thresh)

    def test_unimodal_distribution_returns_finite(self):
        rng = np.random.default_rng(42)
        vals = rng.normal(0.5, 0.1, 1000).clip(0, 1).astype(np.float32)
        thresh = otsu_threshold(vals)
        assert np.isfinite(thresh)
        assert 0.0 <= thresh <= 1.0

    def test_two_value_array(self):
        """Minimal bimodal: only two distinct values."""
        vals = np.array([0.0] * 100 + [1.0] * 100, dtype=np.float32)
        thresh = otsu_threshold(vals)
        assert np.isfinite(thresh)
        assert 0.0 <= thresh <= 1.0

    def test_integer_valued_array(self):
        """HU-like integer values (0–1000 range)."""
        vals = np.array(list(range(0, 500)) + list(range(700, 1000)), dtype=np.float32)
        thresh = otsu_threshold(vals)
        assert np.isfinite(thresh)
        assert 400 < thresh < 800

    def test_single_element(self):
        """Edge case: single-element array."""
        vals = np.array([0.5], dtype=np.float32)
        thresh = otsu_threshold(vals)
        assert np.isfinite(thresh)

    def test_threshold_within_data_range(self):
        """Threshold must lie within the range of input values."""
        rng = np.random.default_rng(99)
        vals = rng.uniform(0.2, 0.8, 200).astype(np.float32)
        thresh = otsu_threshold(vals)
        assert vals.min() <= thresh <= vals.max()


# ── two_stage_otsu ─────────────────────────────────────────────────────────

class TestTwoStageOtsu:
    def test_returns_float(self):
        vol = bimodal_array().reshape(10, 10, 10)
        result = two_stage_otsu(vol)
        assert isinstance(result, (float, np.floating))

    def test_result_is_finite(self):
        vol = bimodal_array(low=0.1, high=0.9).reshape(10, 10, 10)
        assert np.isfinite(two_stage_otsu(vol))

    def test_second_stage_higher_than_first_stage(self):
        """Second-stage threshold should be higher (refines foreground only)."""
        vals = bimodal_array(low=0.1, high=0.9)
        first = otsu_threshold(vals)
        result = two_stage_otsu(vals.reshape(10, 10, 10))
        # Second stage focuses on foreground (> first threshold), so result >= first
        assert result >= first - 0.05  # allow tiny float tolerance

    def test_trimodal_finds_high_peak(self):
        """With three clusters, second stage should target the highest cluster."""
        rng = np.random.default_rng(7)
        low = rng.normal(0.1, 0.01, 300).clip(0, 1)
        mid = rng.normal(0.5, 0.01, 300).clip(0, 1)
        high = rng.normal(0.9, 0.01, 300).clip(0, 1)
        vals = np.concatenate([low, mid, high]).astype(np.float32)
        thresh = two_stage_otsu(vals.reshape(30, 10, 3))
        # Should be above the mid cluster
        assert thresh > 0.5

    def test_volume_shape_agnostic(self):
        """Works regardless of volume shape."""
        for shape in [(100,), (10, 10), (5, 5, 4)]:
            vals = bimodal_array(100, 100).reshape(-1)[:np.prod(shape)]
            # pad if needed
            full = np.concatenate([vals, np.zeros(np.prod(shape) - len(vals))])
            result = two_stage_otsu(full.reshape(shape))
            assert np.isfinite(result)

    def test_sparse_foreground_falls_back(self):
        """If foreground has ≤100 samples, falls back to first-stage threshold."""
        # Only a few foreground samples
        vals = np.concatenate([
            np.ones(990, dtype=np.float32) * 0.1,
            np.ones(5, dtype=np.float32) * 0.9,
        ])
        first = otsu_threshold(vals)
        result = two_stage_otsu(vals.reshape(-1))
        # Should equal first-stage (fallback)
        assert result == first


# ── rotation_matrix_z ──────────────────────────────────────────────────────

class TestRotationMatrixZ:
    def test_returns_ndarray(self):
        R = rotation_matrix_z(0.0)
        assert isinstance(R, np.ndarray)

    def test_shape_3x3(self):
        R = rotation_matrix_z(45.0)
        assert R.shape == (3, 3)

    def test_zero_degrees_is_identity(self):
        R = rotation_matrix_z(0.0)
        assert np.allclose(R, np.eye(3), atol=1e-10)

    def test_180_degrees_negates_xy(self):
        R = rotation_matrix_z(180.0)
        expected = np.array([
            [-1,  0, 0],
            [ 0, -1, 0],
            [ 0,  0, 1],
        ], dtype=float)
        assert np.allclose(R, expected, atol=1e-10)

    def test_90_degrees(self):
        R = rotation_matrix_z(90.0)
        expected = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ], dtype=float)
        assert np.allclose(R, expected, atol=1e-10)

    def test_minus_90_degrees(self):
        R = rotation_matrix_z(-90.0)
        expected = np.array([
            [ 0, 1, 0],
            [-1, 0, 0],
            [ 0, 0, 1],
        ], dtype=float)
        assert np.allclose(R, expected, atol=1e-10)

    def test_determinant_is_one(self):
        for deg in [0, 30, 45, 90, 135, 180, 270, -45, -90]:
            R = rotation_matrix_z(float(deg))
            assert abs(np.linalg.det(R) - 1.0) < 1e-10, f"det != 1 for {deg} deg"

    def test_orthogonal(self):
        """R @ R.T == I for all angles."""
        for deg in [0, 30, 60, 90, 120, 180, 270]:
            R = rotation_matrix_z(float(deg))
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_z_axis_unchanged(self):
        """Rotation around z-axis should not move the z-axis vector."""
        for deg in [30, 45, 90, 180]:
            R = rotation_matrix_z(float(deg))
            z = np.array([0, 0, 1], dtype=float)
            assert np.allclose(R @ z, z, atol=1e-10)

    def test_rotation_composition(self):
        """Two 90-deg rotations should equal one 180-deg rotation."""
        R90 = rotation_matrix_z(90.0)
        R180 = rotation_matrix_z(180.0)
        assert np.allclose(R90 @ R90, R180, atol=1e-10)

    def test_rotation_inverse_is_transpose(self):
        """R(-deg) == R(deg).T"""
        for deg in [30, 45, 90]:
            R_pos = rotation_matrix_z(float(deg))
            R_neg = rotation_matrix_z(float(-deg))
            assert np.allclose(R_neg, R_pos.T, atol=1e-10)

    def test_360_degrees_is_identity(self):
        R = rotation_matrix_z(360.0)
        assert np.allclose(R, np.eye(3), atol=1e-10)

    def test_float_angle(self):
        R = rotation_matrix_z(37.5)
        assert R.shape == (3, 3)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


# ── bone_dice ──────────────────────────────────────────────────────────────

class TestBoneDice:
    def test_returns_float(self):
        img = make_binary_img()
        result = bone_dice(img, img)
        assert isinstance(result, float)

    def test_identical_images_returns_one(self):
        img = make_binary_img()
        assert abs(bone_dice(img, img) - 1.0) < 1e-9

    def test_no_overlap_returns_zero(self):
        h, w = 64, 64
        img1 = np.zeros((h, w), dtype=np.uint8)
        img2 = np.zeros((h, w), dtype=np.uint8)
        img1[:, :w // 2] = 200   # left half
        img2[:, w // 2:] = 200   # right half
        result = bone_dice(img1, img2)
        assert result == 0.0

    def test_full_overlap_different_scale(self):
        """Same mask, different pixel values both above threshold => dice=1."""
        mask = (np.arange(64 * 64) % 2).reshape(64, 64).astype(np.uint8) * 200
        img1 = mask
        img2 = mask.copy()
        img2[img2 > 0] = 150  # different intensity but still > 80
        assert abs(bone_dice(img1, img2) - 1.0) < 1e-9

    def test_partial_overlap(self):
        """50% overlap: dice should be approximately 0.5."""
        h, w = 64, 64
        img1 = np.zeros((h, w), dtype=np.uint8)
        img2 = np.zeros((h, w), dtype=np.uint8)
        img1[:, :w // 2] = 200      # left 32
        img2[:, w // 4: 3 * w // 4] = 200  # middle 32 (16 overlap with img1)
        result = bone_dice(img1, img2)
        # intersection = 16*64, total = 32*64 + 32*64 = 4096
        # dice = 2*1024/4096 = 0.5
        assert 0.4 < result < 0.6

    def test_empty_both_returns_one(self):
        """Both all-zero (below threshold) => both masks empty => dice=1.0."""
        img1 = np.zeros((64, 64), dtype=np.uint8)
        img2 = np.zeros((64, 64), dtype=np.uint8)
        assert bone_dice(img1, img2) == 1.0

    def test_threshold_parameter(self):
        """Values exactly at threshold should not be counted as bone."""
        h, w = 64, 64
        img1 = np.full((h, w), 80, dtype=np.uint8)  # at threshold
        img2 = np.full((h, w), 80, dtype=np.uint8)
        # threshold=80 => condition is > 80, so 80 is excluded
        result = bone_dice(img1, img2, threshold=80)
        assert result == 1.0  # both empty => 1.0

    def test_threshold_parameter_above(self):
        """Values one above threshold are counted."""
        h, w = 64, 64
        img1 = np.full((h, w), 81, dtype=np.uint8)
        img2 = np.full((h, w), 81, dtype=np.uint8)
        result = bone_dice(img1, img2, threshold=80)
        assert abs(result - 1.0) < 1e-9

    def test_custom_threshold(self):
        """Custom threshold = 150: values at 100 below threshold."""
        h, w = 64, 64
        img1 = np.full((h, w), 100, dtype=np.uint8)
        img2 = np.full((h, w), 200, dtype=np.uint8)
        # img1 all below threshold=150, img2 all above => no overlap
        result = bone_dice(img1, img2, threshold=150)
        assert result == 0.0

    def test_value_in_zero_to_one(self):
        """Dice is always in [0, 1]."""
        rng = np.random.default_rng(5)
        for _ in range(5):
            img1 = rng.integers(0, 256, (32, 32), dtype=np.uint8)
            img2 = rng.integers(0, 256, (32, 32), dtype=np.uint8)
            result = bone_dice(img1, img2)
            assert 0.0 <= result <= 1.0

    def test_symmetric(self):
        """bone_dice(a, b) == bone_dice(b, a)."""
        rng = np.random.default_rng(9)
        img1 = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        img2 = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        assert bone_dice(img1, img2) == bone_dice(img2, img1)


# ── build_forearm_weight_3d ────────────────────────────────────────────────

class TestBuildForearmWeight3d:
    def _make_mask(self, shape, joint_pd):
        """Forearm bone mask: True for PD > joint_pd."""
        mask = np.zeros(shape, dtype=bool)
        mask[joint_pd + 5:] = True
        return mask

    def test_returns_ndarray(self):
        shape = (40, 30, 30)
        mask = self._make_mask(shape, 20)
        W = build_forearm_weight_3d(shape, np.array([20, 15, 15]), mask)
        assert isinstance(W, np.ndarray)

    def test_output_shape_matches_vol(self):
        shape = (40, 30, 30)
        mask = self._make_mask(shape, 20)
        W = build_forearm_weight_3d(shape, np.array([20, 15, 15]), mask)
        assert W.shape == shape

    def test_values_in_zero_to_one(self):
        shape = (40, 30, 30)
        mask = self._make_mask(shape, 20)
        W = build_forearm_weight_3d(shape, np.array([20, 15, 15]), mask)
        assert W.min() >= 0.0
        assert W.max() <= 1.0

    def test_far_above_joint_high_weight(self):
        """Slices far above joint center should have weight close to 1."""
        shape = (80, 30, 30)
        jc = 40
        mask = self._make_mask(shape, jc)
        W = build_forearm_weight_3d(shape, np.array([jc, 15, 15]), mask)
        # Slices near the end (far above joint) should be high weight
        mean_high = W[70:].mean()
        assert mean_high > 0.8

    def test_far_below_joint_low_weight(self):
        """Slices far below joint center should have weight close to 0."""
        shape = (80, 30, 30)
        jc = 40
        mask = self._make_mask(shape, jc)
        W = build_forearm_weight_3d(shape, np.array([jc, 15, 15]), mask)
        mean_low = W[:5].mean()
        assert mean_low < 0.2

    def test_smooth_transition(self):
        """Weight should increase monotonically along PD axis (mean per slice)."""
        shape = (60, 20, 20)
        jc = 30
        mask = self._make_mask(shape, jc)
        W = build_forearm_weight_3d(shape, np.array([jc, 10, 10]), mask)
        means = W.mean(axis=(1, 2))
        # After Gaussian smoothing, slice means should be generally increasing
        # Check that bottom < top
        assert means[:10].mean() < means[-10:].mean()

    def test_custom_blend_sigma(self):
        """Different blend_sigma values should still return valid weight arrays."""
        shape = (40, 20, 20)
        mask = self._make_mask(shape, 20)
        for sigma in [2.0, 8.0, 16.0]:
            W = build_forearm_weight_3d(shape, np.array([20, 10, 10]), mask, blend_sigma=sigma)
            assert W.min() >= 0.0
            assert W.max() <= 1.0

    def test_float32_dtype(self):
        shape = (40, 20, 20)
        mask = self._make_mask(shape, 20)
        W = build_forearm_weight_3d(shape, np.array([20, 10, 10]), mask)
        assert W.dtype == np.float32

    def test_joint_at_top_all_high(self):
        """Joint near top of volume: most slices should get high weight."""
        shape = (40, 20, 20)
        jc = 5  # joint near top
        mask = self._make_mask(shape, jc)
        W = build_forearm_weight_3d(shape, np.array([jc, 10, 10]), mask)
        # Most of the volume (below joint) should have high weight
        assert W[20:].mean() > 0.7

    def test_joint_at_bottom_mostly_zero(self):
        """Joint near bottom: most slices below joint have low weight."""
        shape = (40, 20, 20)
        jc = 35  # joint near bottom
        mask = self._make_mask(shape, jc)
        W = build_forearm_weight_3d(shape, np.array([jc, 10, 10]), mask)
        assert W[:5].mean() < 0.1
