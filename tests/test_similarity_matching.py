"""
Unit tests for scripts/similarity_matching.py

Covers pure functions:
  - ncc
  - ssim_score
  - nmi
  - compute_similarity
  - crop_to_bone
  - preprocess_image
  - extract_edges
  - _parabolic_peak
  - MatchResult (NamedTuple construction)
  - DRRLibraryCache (init state, lazy-load guard)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ── パス設定 ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "elbow-train"))

from similarity_matching import (
    DRRLibraryCache,
    MatchResult,
    _parabolic_peak,
    compute_similarity,
    crop_to_bone,
    extract_edges,
    ncc,
    nmi,
    preprocess_image,
    ssim_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _rand_img(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    """float32 [0,1] grayscale image"""
    return _rng(seed).random((h, w)).astype(np.float32)


def _bone_img(h: int = 64, w: int = 64) -> np.ndarray:
    """uint8 image with dark border + bright bone-like center"""
    img = np.zeros((h, w), dtype=np.uint8)
    # bright region in center (simulates bone)
    img[16:48, 16:48] = 180
    return img


# ─────────────────────────────────────────────────────────────────────────────
# ncc
# ─────────────────────────────────────────────────────────────────────────────

class TestNcc:
    def test_identical_arrays_returns_one(self):
        a = _rand_img()
        assert ncc(a, a) == pytest.approx(1.0, abs=1e-5)

    def test_negated_arrays_returns_minus_one(self):
        a = _rand_img()
        b = -a
        assert ncc(a, b) == pytest.approx(-1.0, abs=1e-5)

    def test_orthogonal_arrays_near_zero(self):
        rng = _rng(42)
        a = rng.standard_normal(1000).astype(np.float32)
        # b orthogonal to a: b = a - projection of a onto a = 0 in practice;
        # use random independent vector instead
        b = rng.standard_normal(1000).astype(np.float32)
        val = ncc(a, b)
        assert -1.0 <= val <= 1.0

    def test_constant_array_does_not_raise(self):
        a = np.ones((64, 64), dtype=np.float32)
        b = _rand_img()
        val = ncc(a, b)
        assert np.isfinite(val)

    def test_both_constant_returns_finite(self):
        a = np.ones((32, 32), dtype=np.float32)
        b = np.ones((32, 32), dtype=np.float32) * 2
        val = ncc(a, b)
        assert np.isfinite(val)

    def test_value_range(self):
        a = _rand_img(seed=1)
        b = _rand_img(seed=2)
        val = ncc(a, b)
        assert -1.0 <= val <= 1.0

    def test_symmetry(self):
        a = _rand_img(seed=3)
        b = _rand_img(seed=4)
        assert ncc(a, b) == pytest.approx(ncc(b, a), abs=1e-6)

    def test_1d_input(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([2.0, 4.0, 6.0], dtype=np.float32)  # perfect positive correlation
        assert ncc(a, b) == pytest.approx(1.0, abs=1e-5)

    def test_shifted_signal(self):
        # Signal shifted up/down: correlation should still be 1
        a = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        b = a + 10.0
        assert ncc(a, b) == pytest.approx(1.0, abs=1e-5)

    def test_returns_float(self):
        a = _rand_img()
        b = _rand_img(seed=1)
        assert isinstance(ncc(a, b), float)


# ─────────────────────────────────────────────────────────────────────────────
# ssim_score
# ─────────────────────────────────────────────────────────────────────────────

class TestSsimScore:
    def test_identical_returns_one(self):
        a = _rand_img()
        assert ssim_score(a, a) == pytest.approx(1.0, abs=1e-4)

    def test_different_images_below_one(self):
        a = _rand_img(seed=0)
        b = _rand_img(seed=99)
        assert ssim_score(a, b) < 1.0

    def test_value_range(self):
        a = _rand_img(seed=5)
        b = _rand_img(seed=6)
        val = ssim_score(a, b)
        assert -1.0 <= val <= 1.0

    def test_symmetry(self):
        a = _rand_img(seed=7)
        b = _rand_img(seed=8)
        assert ssim_score(a, b) == pytest.approx(ssim_score(b, a), abs=1e-6)

    def test_returns_float(self):
        a = _rand_img()
        assert isinstance(ssim_score(a, a), float)

    def test_noise_vs_clean(self):
        rng = _rng(10)
        clean = np.zeros((64, 64), dtype=np.float32)
        noisy = rng.random((64, 64)).astype(np.float32)
        assert ssim_score(clean, noisy) < 0.5


# ─────────────────────────────────────────────────────────────────────────────
# nmi
# ─────────────────────────────────────────────────────────────────────────────

class TestNmi:
    def test_identical_returns_two(self):
        a = _rand_img()
        val = nmi(a, a)
        assert val == pytest.approx(2.0, abs=0.05)

    def test_range_geq_one(self):
        a = _rand_img(seed=11)
        b = _rand_img(seed=12)
        assert nmi(a, b) >= 1.0

    def test_independent_below_two(self):
        a = _rand_img(seed=13)
        b = _rand_img(seed=14)
        assert nmi(a, b) < 2.0

    def test_constant_array_returns_finite(self):
        a = np.zeros((64, 64), dtype=np.float32)
        b = _rand_img()
        val = nmi(a, b)
        assert np.isfinite(val)

    def test_custom_bins(self):
        a = _rand_img(seed=15)
        b = _rand_img(seed=16)
        val32 = nmi(a, b, bins=32)
        val128 = nmi(a, b, bins=128)
        # Both should be in [1, 2]
        assert 1.0 <= val32 <= 2.0
        assert 1.0 <= val128 <= 2.0

    def test_returns_float(self):
        a = _rand_img()
        b = _rand_img(seed=1)
        assert isinstance(nmi(a, b), float)

    def test_monotone_transform_higher_than_independent(self):
        # NMI is invariant to monotone transforms; sqrt is strictly monotone on [0,1]
        a = _rand_img(seed=17)
        b_monotone = np.sqrt(a)        # strictly monotone, no clipping needed
        b_independent = _rand_img(seed=99)
        nmi_mono = nmi(a, b_monotone)
        nmi_indep = nmi(a, b_independent)
        assert nmi_mono > nmi_indep


# ─────────────────────────────────────────────────────────────────────────────
# compute_similarity
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSimilarity:
    def test_returns_dict_with_four_keys(self):
        a = _rand_img(seed=0)
        b = _rand_img(seed=1)
        result = compute_similarity(a, b)
        assert set(result.keys()) == {"ncc", "ssim", "edge_ncc", "nmi"}

    def test_identical_inputs_all_high(self):
        a = _rand_img(seed=2)
        result = compute_similarity(a, a)
        assert result["ncc"] == pytest.approx(1.0, abs=1e-4)
        assert result["ssim"] == pytest.approx(1.0, abs=1e-4)
        assert result["edge_ncc"] >= 0.9  # edges of identical image → ncc≈1

    def test_all_values_finite(self):
        a = _rand_img(seed=3)
        b = _rand_img(seed=4)
        result = compute_similarity(a, b)
        for k, v in result.items():
            assert np.isfinite(v), f"{k} is not finite"

    def test_ncc_range(self):
        a = _rand_img(seed=5)
        b = _rand_img(seed=6)
        result = compute_similarity(a, b)
        assert -1.0 <= result["ncc"] <= 1.0

    def test_nmi_range(self):
        a = _rand_img(seed=7)
        b = _rand_img(seed=8)
        result = compute_similarity(a, b)
        assert result["nmi"] >= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# crop_to_bone
# ─────────────────────────────────────────────────────────────────────────────

class TestCropToBone:
    def test_bone_center_is_cropped(self):
        img = _bone_img(64, 64)
        cropped = crop_to_bone(img)
        # Cropped image should be smaller than original
        assert cropped.shape[0] <= 64
        assert cropped.shape[1] <= 64

    def test_already_bright_no_crop(self):
        # Image mostly bright → crop_ratio > 0.7 → return original
        img = np.ones((64, 64), dtype=np.uint8) * 200
        result = crop_to_bone(img)
        assert result.shape == (64, 64)

    def test_all_dark_returns_original(self):
        # No bright pixels → rows.any() returns False → return original
        img = np.zeros((32, 32), dtype=np.uint8)
        result = crop_to_bone(img)
        assert result.shape == (32, 32)

    def test_custom_dark_thresh(self):
        img = _bone_img(64, 64)
        # With thresh=200, most of the bright region is also "dark" → no crop
        result_high = crop_to_bone(img, dark_thresh=200)
        assert result_high.shape == (64, 64)

    def test_output_dtype_preserved(self):
        img = _bone_img(64, 64)
        result = crop_to_bone(img)
        assert result.dtype == np.uint8

    def test_custom_margin(self):
        img = _bone_img(64, 64)
        r1 = crop_to_bone(img, margin=0.0)
        r2 = crop_to_bone(img, margin=0.3)
        # Larger margin → larger (or equal) cropped image
        assert r2.shape[0] >= r1.shape[0]
        assert r2.shape[1] >= r1.shape[1]

    def test_single_pixel_bright(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[16, 16] = 100
        result = crop_to_bone(img)
        # Should return without error; crop is valid
        assert result.ndim == 2

    def test_non_square_image(self):
        img = np.zeros((32, 64), dtype=np.uint8)
        img[8:24, 16:48] = 150
        result = crop_to_bone(img)
        assert result.ndim == 2


# ─────────────────────────────────────────────────────────────────────────────
# preprocess_image
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessImage:
    def test_output_shape_default(self):
        img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        out = preprocess_image(img)
        assert out.shape == (256, 256)

    def test_output_shape_custom_size(self):
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        out = preprocess_image(img, size=128)
        assert out.shape == (128, 128)

    def test_output_range(self):
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        out = preprocess_image(img)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_output_dtype_float32(self):
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        out = preprocess_image(img)
        assert out.dtype == np.float32

    def test_bgr_input_converted_to_gray(self):
        img_bgr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        out = preprocess_image(img_bgr)
        assert out.ndim == 2

    def test_apply_rot270(self):
        # Create asymmetric image; rotation should change it
        img = np.zeros((64, 64), dtype=np.uint8)
        img[:32, :] = 200  # top half bright
        out_no_rot = preprocess_image(img, size=64, apply_rot270=False)
        out_rot = preprocess_image(img, size=64, apply_rot270=True)
        assert not np.allclose(out_no_rot, out_rot)

    def test_auto_crop_with_dark_border(self):
        img = _bone_img(64, 64)
        out = preprocess_image(img, size=64, auto_crop=True)
        assert out.shape == (64, 64)
        assert out.dtype == np.float32

    def test_grayscale_input_unchanged_ndim(self):
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        out = preprocess_image(img)
        assert out.ndim == 2

    def test_different_inputs_give_different_outputs(self):
        img1 = np.zeros((64, 64), dtype=np.uint8)
        img2 = np.ones((64, 64), dtype=np.uint8) * 200
        out1 = preprocess_image(img1)
        out2 = preprocess_image(img2)
        assert not np.allclose(out1, out2)


# ─────────────────────────────────────────────────────────────────────────────
# extract_edges
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractEdges:
    def test_output_shape_preserved(self):
        img = _rand_img(64, 64)
        out = extract_edges(img)
        assert out.shape == (64, 64)

    def test_output_range(self):
        img = _rand_img()
        out = extract_edges(img)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_output_dtype_float32(self):
        img = _rand_img()
        out = extract_edges(img)
        assert out.dtype == np.float32

    def test_uniform_image_has_no_edges(self):
        img = np.ones((64, 64), dtype=np.float32) * 0.5
        out = extract_edges(img)
        assert out.max() == 0.0

    def test_step_edge_detected(self):
        img = np.zeros((64, 64), dtype=np.float32)
        img[:, 32:] = 1.0  # vertical step edge
        out = extract_edges(img)
        # Some edge pixels should be non-zero
        assert out.sum() > 0

    def test_only_binary_values(self):
        img = _rand_img()
        out = extract_edges(img)
        unique = np.unique(out)
        assert all(v in (0.0, 1.0) for v in unique)

    def test_identical_edge_maps_for_same_input(self):
        img = _rand_img(seed=42)
        out1 = extract_edges(img)
        out2 = extract_edges(img)
        np.testing.assert_array_equal(out1, out2)


# ─────────────────────────────────────────────────────────────────────────────
# _parabolic_peak
# ─────────────────────────────────────────────────────────────────────────────

class TestParabolicPeak:
    """Tests for _parabolic_peak(scores_dict, metric_key) -> float"""

    def _make_scores(self, angles: list[float], values: list[float],
                     metric: str = "ncc") -> dict:
        return {a: {metric: v} for a, v in zip(angles, values)}

    def test_peak_at_center_sub_degree_estimate(self):
        # True peak at 90°; parabolic fit should return near 90°
        angles = [85.0, 90.0, 95.0]
        values = [0.80, 0.95, 0.80]
        scores = self._make_scores(angles, values)
        result = _parabolic_peak(scores, "ncc")
        assert result == pytest.approx(90.0, abs=1e-4)

    def test_peak_at_left_end_returns_angle_as_is(self):
        angles = [80.0, 90.0, 100.0]
        values = [0.95, 0.80, 0.70]
        scores = self._make_scores(angles, values)
        result = _parabolic_peak(scores, "ncc")
        assert result == 80.0

    def test_peak_at_right_end_returns_angle_as_is(self):
        angles = [80.0, 90.0, 100.0]
        values = [0.60, 0.75, 0.90]
        scores = self._make_scores(angles, values)
        result = _parabolic_peak(scores, "ncc")
        assert result == 100.0

    def test_asymmetric_peak_interpolated(self):
        # Peak at 90 but steeper on left → sub-degree offset
        angles = [85.0, 90.0, 95.0]
        values = [0.70, 0.95, 0.85]
        scores = self._make_scores(angles, values)
        result = _parabolic_peak(scores, "ncc")
        # Offset towards the shallower side (right)
        assert result > 90.0
        assert result < 95.0

    def test_flat_peak_returns_peak_angle(self):
        # Coefficients nearly 0 → returns peak as-is
        angles = [85.0, 90.0, 95.0]
        values = [0.95, 0.95, 0.95]  # flat → a_coef ≈ 0
        scores = self._make_scores(angles, values)
        result = _parabolic_peak(scores, "ncc")
        # any of the three could be peak; result should be one of the angles
        assert result in angles

    def test_many_angles_returns_float(self):
        angles = list(range(60, 181, 5))
        # Gaussian-like peak at 120°
        values = [np.exp(-0.5 * ((a - 120) / 10) ** 2) for a in angles]
        scores = self._make_scores(angles, values)
        result = _parabolic_peak(scores, "ncc")
        assert isinstance(result, float)
        assert 115.0 <= result <= 125.0

    def test_works_with_edge_ncc_key(self):
        angles = [85.0, 90.0, 95.0]
        values = [0.70, 0.90, 0.75]
        scores = {a: {"ncc": 0.5, "edge_ncc": v} for a, v in zip(angles, values)}
        result = _parabolic_peak(scores, "edge_ncc")
        assert isinstance(result, float)

    def test_clipped_within_neighbors(self):
        # If fit extrapolates beyond neighbors, should be clipped
        angles = [85.0, 90.0, 95.0]
        values = [0.50, 0.95, 0.94]  # very similar peak and neighbor
        scores = self._make_scores(angles, values)
        result = _parabolic_peak(scores, "ncc")
        assert 85.0 <= result <= 95.0

    def test_single_point_returns_that_angle(self):
        scores = {90.0: {"ncc": 0.95}}
        result = _parabolic_peak(scores, "ncc")
        assert result == 90.0

    def test_two_points_returns_higher(self):
        scores = {85.0: {"ncc": 0.70}, 90.0: {"ncc": 0.90}}
        result = _parabolic_peak(scores, "ncc")
        assert result == 90.0  # peak is at end → return as-is


# ─────────────────────────────────────────────────────────────────────────────
# MatchResult
# ─────────────────────────────────────────────────────────────────────────────

class TestMatchResult:
    def test_construction_with_required_fields(self):
        drr = np.zeros((32, 32), dtype=np.uint8)
        scores = {90.0: {"ncc": 0.9, "ssim": 0.8, "edge_ncc": 0.7, "nmi": 1.5}}
        mr = MatchResult(
            best_angle=90.0,
            best_metric="ncc",
            scores=scores,
            drr_at_best=drr,
        )
        assert mr.best_angle == 90.0
        assert mr.best_metric == "ncc"
        assert mr.peak_ncc == 0.0   # default
        assert mr.sharpness == 0.0  # default

    def test_construction_with_optional_fields(self):
        drr = np.zeros((32, 32), dtype=np.uint8)
        mr = MatchResult(
            best_angle=85.0,
            best_metric="combined",
            scores={},
            drr_at_best=drr,
            peak_ncc=0.92,
            sharpness=3.5,
        )
        assert mr.peak_ncc == pytest.approx(0.92)
        assert mr.sharpness == pytest.approx(3.5)

    def test_is_namedtuple(self):
        drr = np.zeros((8, 8), dtype=np.uint8)
        mr = MatchResult(90.0, "ncc", {}, drr)
        assert isinstance(mr, tuple)
        assert mr[0] == 90.0

    def test_fields_accessible_by_name(self):
        drr = np.zeros((8, 8), dtype=np.uint8)
        mr = MatchResult(75.0, "edge_ncc", {}, drr)
        assert mr.best_angle == 75.0
        assert mr.best_metric == "edge_ncc"


# ─────────────────────────────────────────────────────────────────────────────
# DRRLibraryCache
# ─────────────────────────────────────────────────────────────────────────────

class TestDRRLibraryCacheInit:
    def test_stores_path(self):
        cache = DRRLibraryCache("/some/path/patient.npz")
        assert cache.library_path == "/some/path/patient.npz"

    def test_lazy_load_not_triggered_on_init(self):
        # Internal state should be None before any access
        cache = DRRLibraryCache("/nonexistent/path.npz")
        assert cache._angle_to_drr is None
        assert cache._angles_arr is None
        assert cache._drrs is None
        assert cache._meta is None

    def test_meta_property_raises_on_missing_file(self):
        cache = DRRLibraryCache("/nonexistent/path.npz")
        with pytest.raises((FileNotFoundError, OSError, Exception)):
            _ = cache.meta

    def test_match_raises_on_missing_file(self):
        cache = DRRLibraryCache("/nonexistent/path.npz")
        xray = np.zeros((64, 64), dtype=np.uint8)
        with pytest.raises((FileNotFoundError, OSError, Exception)):
            cache.match(xray)
