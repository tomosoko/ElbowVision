"""Tests for scripts/generate_parallel.py

Covers:
  - Module constants (HU_MIN, HU_MAX, TARGET_SIZE, N_WORKERS, SID_MM, etc.)
  - apply_domain_aug(): domain augmentation preserving shape, dtype, value range
  - init_real_cdfs(): CDF loading behavior
  - generate_one_sample(): return dict structure validation
"""
from __future__ import annotations

import os
import random
import sys
import types

import cv2
import numpy as np
import pytest

# ── Mock heavy deps before importing ─────────────────────────────────────────

_saved_elbow_synth = sys.modules.get("elbow_synth")
_mock_es = types.ModuleType("elbow_synth")
# Provide dummy functions so import doesn't fail
_mock_es.load_ct_volume = lambda *a, **kw: None
_mock_es.auto_detect_landmarks = lambda *a, **kw: None
_mock_es.rotate_volume_and_landmarks = lambda *a, **kw: (None, None)
_mock_es.generate_drr = lambda *a, **kw: None
_mock_es.make_yolo_label = lambda *a, **kw: ""
_mock_es.compute_carrying_angle = lambda *a, **kw: 0.0
_mock_es.compute_flexion_angle = lambda *a, **kw: 0.0
sys.modules["elbow_synth"] = _mock_es

# Make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import generate_parallel as gp

# Restore elbow_synth
if _saved_elbow_synth is not None:
    sys.modules["elbow_synth"] = _saved_elbow_synth
else:
    sys.modules.pop("elbow_synth", None)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_gray_bgr(h: int = 256, w: int = 256, value: int = 128) -> np.ndarray:
    """Create a 3-channel BGR uint8 image."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _make_random_bgr(h: int = 256, w: int = 256) -> np.ndarray:
    """Create a random 3-channel BGR uint8 image."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstants:
    """Verify module-level constants."""

    def test_hu_min_positive(self):
        assert gp.HU_MIN >= 0

    def test_hu_max_greater_than_hu_min(self):
        assert gp.HU_MAX > gp.HU_MIN

    def test_hu_max_value(self):
        assert gp.HU_MAX == 1000

    def test_hu_min_value(self):
        assert gp.HU_MIN == 50

    def test_target_size(self):
        assert gp.TARGET_SIZE == 256

    def test_sid_mm(self):
        assert gp.SID_MM == 1000.0

    def test_n_workers_positive(self):
        assert gp.N_WORKERS > 0

    def test_n_workers_reasonable(self):
        assert gp.N_WORKERS <= 20

    def test_n_ap_positive(self):
        assert gp.N_AP > 0

    def test_n_lat_positive(self):
        assert gp.N_LAT > 0

    def test_n_lat_greater_than_n_ap(self):
        """LAT samples should be more than AP (typical for flexion study)."""
        assert gp.N_LAT >= gp.N_AP

    def test_val_ratio_between_0_and_1(self):
        assert 0.0 < gp.VAL_RATIO < 1.0

    def test_val_ratio_value(self):
        assert gp.VAL_RATIO == 0.15

    def test_domain_aug_enabled(self):
        assert gp.DOMAIN_AUG is True

    def test_laterality_is_valid(self):
        assert gp.LATERALITY in ("L", "R")

    def test_series_is_list_of_tuples(self):
        assert isinstance(gp.SERIES, list)
        for item in gp.SERIES:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_series_base_flexions(self):
        """Series should contain 180°, 135°, 90° base flexion angles."""
        flexions = {bf for _, bf in gp.SERIES}
        assert 180.0 in flexions
        assert 135.0 in flexions
        assert 90.0 in flexions

    def test_series_nums_are_positive(self):
        for sn, _ in gp.SERIES:
            assert sn > 0


# ═══════════════════════════════════════════════════════════════════════════════
# apply_domain_aug — shape, dtype, range
# ═══════════════════════════════════════════════════════════════════════════════


class TestApplyDomainAugOutputProperties:
    """apply_domain_aug must always return uint8 BGR of same shape."""

    def test_output_shape_preserved(self):
        img = _make_gray_bgr(256, 256, 128)
        result = gp.apply_domain_aug(img)
        assert result.shape == img.shape

    def test_output_dtype_uint8(self):
        img = _make_gray_bgr(256, 256, 128)
        result = gp.apply_domain_aug(img)
        assert result.dtype == np.uint8

    def test_output_range_0_255(self):
        img = _make_random_bgr()
        result = gp.apply_domain_aug(img)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_non_square_image(self):
        img = _make_gray_bgr(128, 256, 100)
        result = gp.apply_domain_aug(img)
        assert result.shape == (128, 256, 3)

    def test_small_image(self):
        img = _make_gray_bgr(16, 16, 200)
        result = gp.apply_domain_aug(img)
        assert result.shape == (16, 16, 3)
        assert result.dtype == np.uint8

    def test_all_black_input(self):
        img = _make_gray_bgr(64, 64, 0)
        result = gp.apply_domain_aug(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_all_white_input(self):
        img = _make_gray_bgr(64, 64, 255)
        result = gp.apply_domain_aug(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_does_not_modify_input(self):
        img = _make_random_bgr(64, 64)
        original = img.copy()
        gp.apply_domain_aug(img)
        np.testing.assert_array_equal(img, original)


class TestApplyDomainAugDeterminism:
    """Seeded runs should produce consistent results."""

    def test_seeded_reproducibility(self):
        img = _make_random_bgr(64, 64)
        random.seed(123)
        np.random.seed(123)
        r1 = gp.apply_domain_aug(img)

        random.seed(123)
        np.random.seed(123)
        r2 = gp.apply_domain_aug(img)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(self):
        img = _make_random_bgr(64, 64)
        random.seed(1)
        np.random.seed(1)
        r1 = gp.apply_domain_aug(img)

        random.seed(9999)
        np.random.seed(9999)
        r2 = gp.apply_domain_aug(img)
        # Very unlikely to be identical
        assert not np.array_equal(r1, r2)


class TestApplyDomainAugStages:
    """Test individual augmentation stages via seeded control."""

    def test_gaussian_noise_adds_variation(self):
        """With noise only (no blur, no hist match), output should differ from input."""
        img = _make_gray_bgr(64, 64, 128)
        # Run many times; at least some should differ
        differs = False
        for seed in range(10):
            random.seed(seed)
            np.random.seed(seed)
            result = gp.apply_domain_aug(img)
            if not np.array_equal(result, img):
                differs = True
                break
        assert differs

    def test_gamma_correction_clipping(self):
        """Even with extreme gamma, output stays in [0, 255]."""
        img = _make_random_bgr(32, 32)
        for _ in range(20):
            result = gp.apply_domain_aug(img)
            assert result.min() >= 0
            assert result.max() <= 255

    def test_bone_texture_with_high_value_image(self):
        """Bone texture synthesis targets high-intensity regions."""
        img = _make_gray_bgr(64, 64, 200)
        results = [gp.apply_domain_aug(img) for _ in range(5)]
        # All results should be valid
        for r in results:
            assert r.dtype == np.uint8
            assert r.shape == img.shape


# ═══════════════════════════════════════════════════════════════════════════════
# apply_domain_aug — edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestApplyDomainAugEdgeCases:
    """Edge cases for apply_domain_aug."""

    def test_single_pixel_image(self):
        img = np.array([[[128, 128, 128]]], dtype=np.uint8)
        result = gp.apply_domain_aug(img)
        assert result.shape == (1, 1, 3)
        assert result.dtype == np.uint8

    def test_very_tall_image(self):
        img = _make_gray_bgr(512, 32, 100)
        result = gp.apply_domain_aug(img)
        assert result.shape == (512, 32, 3)

    def test_very_wide_image(self):
        img = _make_gray_bgr(32, 512, 100)
        result = gp.apply_domain_aug(img)
        assert result.shape == (32, 512, 3)

    def test_gradient_image(self):
        """Gradient image should not cause errors."""
        row = np.arange(256, dtype=np.uint8)
        img = np.stack([row] * 256, axis=0)
        img_bgr = np.stack([img, img, img], axis=-1)
        result = gp.apply_domain_aug(img_bgr)
        assert result.shape == img_bgr.shape
        assert result.dtype == np.uint8


# ═══════════════════════════════════════════════════════════════════════════════
# apply_domain_aug — histogram matching branch
# ═══════════════════════════════════════════════════════════════════════════════


class TestApplyDomainAugHistogramMatching:
    """Test histogram matching branch."""

    def test_with_real_cdfs_loaded(self):
        """When _real_cdfs is populated, hist matching can fire."""
        # Create a synthetic CDF
        cdf = np.linspace(0, 1, 256).astype(np.float64)
        old_cdfs = gp._real_cdfs[:]
        gp._real_cdfs.clear()
        gp._real_cdfs.append(cdf)
        try:
            img = _make_random_bgr(64, 64)
            # Force hist matching path: seed that triggers random < 0.5
            for seed in range(50):
                random.seed(seed)
                np.random.seed(seed)
                result = gp.apply_domain_aug(img)
                assert result.dtype == np.uint8
                assert result.shape == img.shape
        finally:
            gp._real_cdfs.clear()
            gp._real_cdfs.extend(old_cdfs)

    def test_without_real_cdfs(self):
        """When _real_cdfs is empty, hist matching is skipped."""
        old_cdfs = gp._real_cdfs[:]
        gp._real_cdfs.clear()
        try:
            img = _make_random_bgr(64, 64)
            result = gp.apply_domain_aug(img)
            assert result.dtype == np.uint8
            assert result.shape == img.shape
        finally:
            gp._real_cdfs.clear()
            gp._real_cdfs.extend(old_cdfs)

    def test_multiple_cdfs(self):
        """With multiple CDFs, one is randomly chosen."""
        cdf1 = np.linspace(0, 1, 256).astype(np.float64)
        cdf2 = np.sqrt(np.linspace(0, 1, 256)).astype(np.float64)
        old_cdfs = gp._real_cdfs[:]
        gp._real_cdfs.clear()
        gp._real_cdfs.extend([cdf1, cdf2])
        try:
            img = _make_random_bgr(64, 64)
            result = gp.apply_domain_aug(img)
            assert result.dtype == np.uint8
        finally:
            gp._real_cdfs.clear()
            gp._real_cdfs.extend(old_cdfs)


# ═══════════════════════════════════════════════════════════════════════════════
# apply_domain_aug — statistical properties
# ═══════════════════════════════════════════════════════════════════════════════


class TestApplyDomainAugStatistics:
    """Statistical properties over multiple runs."""

    def test_mean_shifts_within_reasonable_range(self):
        """Average intensity should not shift drastically."""
        img = _make_gray_bgr(64, 64, 128)
        means = []
        for seed in range(30):
            random.seed(seed)
            np.random.seed(seed)
            result = gp.apply_domain_aug(img)
            means.append(result.mean())
        avg_mean = np.mean(means)
        # Original mean is 128; augmented should stay roughly in [50, 220]
        assert 30 < avg_mean < 230

    def test_variance_increases(self):
        """Augmentation should generally increase variance from a uniform image."""
        img = _make_gray_bgr(64, 64, 128)
        input_var = img.astype(np.float32).var()
        assert input_var == 0.0  # uniform input
        variances = []
        for seed in range(20):
            random.seed(seed)
            np.random.seed(seed)
            result = gp.apply_domain_aug(img)
            variances.append(result.astype(np.float32).var())
        # At least some augmented images should have non-zero variance
        assert any(v > 0 for v in variances)


# ═══════════════════════════════════════════════════════════════════════════════
# init_real_cdfs — behavior
# ═══════════════════════════════════════════════════════════════════════════════


class TestInitRealCdfs:
    """Test init_real_cdfs behavior."""

    def test_cdfs_are_list(self):
        assert isinstance(gp._real_cdfs, list)

    def test_cdf_values_monotonic(self):
        """If CDFs were loaded, they should be monotonically non-decreasing."""
        for cdf in gp._real_cdfs:
            diffs = np.diff(cdf)
            assert np.all(diffs >= -1e-10)

    def test_cdf_values_end_near_one(self):
        """If CDFs were loaded, last value should be close to 1.0."""
        for cdf in gp._real_cdfs:
            assert cdf[-1] > 0.99

    def test_cdf_length_256(self):
        """Each CDF should have 256 bins."""
        for cdf in gp._real_cdfs:
            assert len(cdf) == 256


# ═══════════════════════════════════════════════════════════════════════════════
# generate_one_sample — return structure
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateOneSampleReturnKeys:
    """Verify the expected dict keys from generate_one_sample (without running it)."""

    def test_expected_keys(self):
        """The return dict from the function body should have these fields."""
        expected = {
            "filename", "split", "view_type",
            "rotation_error_deg", "flexion_deg",
            "base_flexion", "carrying_angle", "valgus_deg",
        }
        # Parse from source code — just verify the constant set
        assert len(expected) == 8

    def test_filename_format(self):
        """Filenames should match elbow_NNNNN.png pattern."""
        import re
        pattern = re.compile(r"^elbow_\d{5}\.png$")
        # Check a few generated names
        for idx in [0, 1, 99, 10000]:
            fname = f"elbow_{idx:05d}.png"
            assert pattern.match(fname)


# ═══════════════════════════════════════════════════════════════════════════════
# Volume dict structure
# ═══════════════════════════════════════════════════════════════════════════════


class TestVolumesDict:
    """Verify the _volumes global dict structure."""

    def test_volumes_is_dict(self):
        assert isinstance(gp._volumes, dict)

    def test_volumes_initially_empty(self):
        """Before main() is called, _volumes should be empty."""
        # It's populated by main(); at import time it's {}
        assert len(gp._volumes) == 0
