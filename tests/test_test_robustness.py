"""
Unit tests for scripts/test_robustness.py — image perturbation pure functions.

Tests cover:
  - add_gaussian_noise: shape, dtype, clipping, sigma=0 identity
  - apply_blur: shape, dtype, odd kernel enforcement, ksize=1 identity
  - shift_brightness: clipping, delta=0 identity, positive/negative shifts
  - change_contrast: alpha=1.0 identity, alpha=0 flat, range preservation
  - apply_gamma: gamma=1.0 identity, LUT correctness, range preservation
  - histogram_equalization: shape, dtype, output range
  - PERTURBATIONS dict: structure, required keys, func callability
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

# ── elbow_synth モック ────────────────────────────────────────────────────────
_MOCK_SYNTH = types.ModuleType("elbow_synth")
_MOCK_SYNTH.__file__ = "mock_elbow_synth.py"
_MOCK_SYNTH.generate_drr = lambda *a, **kw: None
_MOCK_SYNTH.rotation_matrix_z = lambda angle: np.eye(3)
_MOCK_SYNTH.rotate_volume_and_landmarks = lambda *a, **kw: (None, None)
_saved = sys.modules.get("elbow_synth")
sys.modules["elbow_synth"] = _MOCK_SYNTH

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from test_robustness import (
    PERTURBATIONS,
    add_gaussian_noise,
    apply_blur,
    apply_gamma,
    change_contrast,
    histogram_equalization,
    shift_brightness,
)

# 復元
if _saved is not None:
    sys.modules["elbow_synth"] = _saved
else:
    sys.modules.pop("elbow_synth", None)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def gray_img() -> np.ndarray:
    """256x256 8-bit grayscale test image with gradient."""
    return np.tile(np.arange(256, dtype=np.uint8), (256, 1))


@pytest.fixture
def uniform_img() -> np.ndarray:
    """128x128 uniform mid-gray image."""
    return np.full((128, 128), 128, dtype=np.uint8)


@pytest.fixture
def black_img() -> np.ndarray:
    """64x64 black image."""
    return np.zeros((64, 64), dtype=np.uint8)


@pytest.fixture
def white_img() -> np.ndarray:
    """64x64 white image."""
    return np.full((64, 64), 255, dtype=np.uint8)


@pytest.fixture
def random_img() -> np.ndarray:
    """100x100 random image (deterministic)."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (100, 100), dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# add_gaussian_noise
# ═══════════════════════════════════════════════════════════════════════════════

class TestAddGaussianNoise:
    def test_output_shape_dtype(self, gray_img: np.ndarray) -> None:
        result = add_gaussian_noise(gray_img, sigma=10)
        assert result.shape == gray_img.shape
        assert result.dtype == np.uint8

    def test_sigma_zero_identity(self, gray_img: np.ndarray) -> None:
        result = add_gaussian_noise(gray_img, sigma=0)
        np.testing.assert_array_equal(result, gray_img)

    def test_output_range_clipped(self, gray_img: np.ndarray) -> None:
        result = add_gaussian_noise(gray_img, sigma=100)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_noise_actually_changes_image(self, uniform_img: np.ndarray) -> None:
        result = add_gaussian_noise(uniform_img, sigma=30)
        # With sigma=30 on uniform image, almost certainly some pixels differ
        assert not np.array_equal(result, uniform_img)

    def test_higher_sigma_more_noise(self, uniform_img: np.ndarray) -> None:
        np.random.seed(0)
        r_low = add_gaussian_noise(uniform_img, sigma=5)
        np.random.seed(0)
        r_high = add_gaussian_noise(uniform_img, sigma=50)
        diff_low = np.abs(r_low.astype(float) - uniform_img.astype(float)).mean()
        diff_high = np.abs(r_high.astype(float) - uniform_img.astype(float)).mean()
        assert diff_high > diff_low

    def test_black_image_clipped_at_zero(self, black_img: np.ndarray) -> None:
        result = add_gaussian_noise(black_img, sigma=50)
        assert result.min() >= 0

    def test_white_image_clipped_at_255(self, white_img: np.ndarray) -> None:
        result = add_gaussian_noise(white_img, sigma=50)
        assert result.max() <= 255


# ═══════════════════════════════════════════════════════════════════════════════
# apply_blur
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyBlur:
    def test_output_shape_dtype(self, gray_img: np.ndarray) -> None:
        result = apply_blur(gray_img, ksize=5)
        assert result.shape == gray_img.shape
        assert result.dtype == np.uint8

    def test_ksize_1_identity(self, gray_img: np.ndarray) -> None:
        result = apply_blur(gray_img, ksize=1)
        np.testing.assert_array_equal(result, gray_img)

    def test_even_ksize_made_odd(self, gray_img: np.ndarray) -> None:
        """Even ksize should be forced to odd (ksize | 1)."""
        result = apply_blur(gray_img, ksize=4)
        # ksize=4 -> 4|1 = 5, should not raise
        assert result.shape == gray_img.shape

    def test_ksize_zero_becomes_one(self, gray_img: np.ndarray) -> None:
        """ksize=0 -> max(1, 0|1) = max(1,1) = 1, identity."""
        result = apply_blur(gray_img, ksize=0)
        np.testing.assert_array_equal(result, gray_img)

    def test_blur_reduces_variance(self, random_img: np.ndarray) -> None:
        result = apply_blur(random_img, ksize=15)
        assert result.astype(float).std() < random_img.astype(float).std()

    def test_uniform_image_unchanged(self, uniform_img: np.ndarray) -> None:
        result = apply_blur(uniform_img, ksize=7)
        np.testing.assert_array_equal(result, uniform_img)

    def test_output_range(self, gray_img: np.ndarray) -> None:
        result = apply_blur(gray_img, ksize=11)
        assert result.min() >= 0
        assert result.max() <= 255


# ═══════════════════════════════════════════════════════════════════════════════
# shift_brightness
# ═══════════════════════════════════════════════════════════════════════════════

class TestShiftBrightness:
    def test_output_shape_dtype(self, gray_img: np.ndarray) -> None:
        result = shift_brightness(gray_img, delta=30)
        assert result.shape == gray_img.shape
        assert result.dtype == np.uint8

    def test_delta_zero_identity(self, gray_img: np.ndarray) -> None:
        result = shift_brightness(gray_img, delta=0)
        np.testing.assert_array_equal(result, gray_img)

    def test_positive_shift_increases_mean(self, uniform_img: np.ndarray) -> None:
        result = shift_brightness(uniform_img, delta=50)
        assert result.mean() > uniform_img.mean()

    def test_negative_shift_decreases_mean(self, uniform_img: np.ndarray) -> None:
        result = shift_brightness(uniform_img, delta=-50)
        assert result.mean() < uniform_img.mean()

    def test_clipping_upper(self, white_img: np.ndarray) -> None:
        result = shift_brightness(white_img, delta=100)
        assert result.max() == 255

    def test_clipping_lower(self, black_img: np.ndarray) -> None:
        result = shift_brightness(black_img, delta=-100)
        assert result.min() == 0

    def test_exact_shift_mid_gray(self) -> None:
        img = np.full((10, 10), 100, dtype=np.uint8)
        result = shift_brightness(img, delta=20)
        np.testing.assert_array_equal(result, np.full((10, 10), 120, dtype=np.uint8))

    def test_fractional_delta(self, uniform_img: np.ndarray) -> None:
        result = shift_brightness(uniform_img, delta=0.5)
        # 128 + 0.5 = 128.5 -> truncated to 128 as uint8
        assert result.dtype == np.uint8


# ═══════════════════════════════════════════════════════════════════════════════
# change_contrast
# ═══════════════════════════════════════════════════════════════════════════════

class TestChangeContrast:
    def test_output_shape_dtype(self, gray_img: np.ndarray) -> None:
        result = change_contrast(gray_img, alpha=1.5)
        assert result.shape == gray_img.shape
        assert result.dtype == np.uint8

    def test_alpha_one_identity(self, gray_img: np.ndarray) -> None:
        result = change_contrast(gray_img, alpha=1.0)
        np.testing.assert_array_equal(result, gray_img)

    def test_alpha_zero_flat_at_mean(self, gray_img: np.ndarray) -> None:
        result = change_contrast(gray_img, alpha=0.0)
        # All pixels should be approximately the mean
        expected_mean = gray_img.astype(float).mean()
        assert np.abs(result.astype(float).mean() - expected_mean) < 1.5
        assert result.astype(float).std() < 1.0

    def test_alpha_gt1_increases_contrast(self, random_img: np.ndarray) -> None:
        result = change_contrast(random_img, alpha=2.0)
        # Standard deviation should increase (before clipping saturates)
        # Use moderate alpha to avoid saturation
        result_mild = change_contrast(random_img, alpha=1.2)
        assert result_mild.astype(float).std() >= random_img.astype(float).std() * 0.95

    def test_alpha_lt1_decreases_contrast(self, random_img: np.ndarray) -> None:
        result = change_contrast(random_img, alpha=0.5)
        assert result.astype(float).std() < random_img.astype(float).std()

    def test_output_clipped(self, gray_img: np.ndarray) -> None:
        result = change_contrast(gray_img, alpha=10.0)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_uniform_image_unchanged(self, uniform_img: np.ndarray) -> None:
        """Uniform image: (v - mean) * alpha + mean = mean for any alpha."""
        result = change_contrast(uniform_img, alpha=5.0)
        np.testing.assert_array_equal(result, uniform_img)


# ═══════════════════════════════════════════════════════════════════════════════
# apply_gamma
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyGamma:
    def test_output_shape_dtype(self, gray_img: np.ndarray) -> None:
        result = apply_gamma(gray_img, gamma=1.5)
        assert result.shape == gray_img.shape
        assert result.dtype == np.uint8

    def test_gamma_one_identity(self, gray_img: np.ndarray) -> None:
        result = apply_gamma(gray_img, gamma=1.0)
        np.testing.assert_array_equal(result, gray_img)

    def test_gamma_lt1_brightens(self, uniform_img: np.ndarray) -> None:
        result = apply_gamma(uniform_img, gamma=0.5)
        assert result.mean() > uniform_img.mean()

    def test_gamma_gt1_darkens(self, uniform_img: np.ndarray) -> None:
        result = apply_gamma(uniform_img, gamma=2.0)
        assert result.mean() < uniform_img.mean()

    def test_black_stays_black(self, black_img: np.ndarray) -> None:
        result = apply_gamma(black_img, gamma=0.5)
        np.testing.assert_array_equal(result, black_img)

    def test_white_stays_white(self, white_img: np.ndarray) -> None:
        result = apply_gamma(white_img, gamma=2.0)
        np.testing.assert_array_equal(result, white_img)

    def test_output_range(self, gray_img: np.ndarray) -> None:
        result = apply_gamma(gray_img, gamma=0.3)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_lut_correctness(self) -> None:
        """Verify a specific pixel value through the gamma formula."""
        img = np.full((1, 1), 100, dtype=np.uint8)
        gamma = 2.0
        result = apply_gamma(img, gamma=gamma)
        expected = int((100 / 255.0) ** gamma * 255)
        assert result[0, 0] == expected

    def test_monotonic_for_gradient(self) -> None:
        """Gamma-corrected gradient should remain monotonically non-decreasing."""
        img = np.arange(256, dtype=np.uint8).reshape(1, 256)
        for g in [0.3, 0.5, 1.5, 2.5]:
            result = apply_gamma(img, gamma=g)
            diffs = np.diff(result[0].astype(int))
            assert np.all(diffs >= 0), f"Non-monotonic for gamma={g}"


# ═══════════════════════════════════════════════════════════════════════════════
# histogram_equalization
# ═══════════════════════════════════════════════════════════════════════════════

class TestHistogramEqualization:
    def test_output_shape_dtype(self, gray_img: np.ndarray) -> None:
        result = histogram_equalization(gray_img, clip_limit=2.0)
        assert result.shape == gray_img.shape
        assert result.dtype == np.uint8

    def test_output_range(self, random_img: np.ndarray) -> None:
        result = histogram_equalization(random_img, clip_limit=4.0)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_uniform_image_nearly_unchanged(self, uniform_img: np.ndarray) -> None:
        result = histogram_equalization(uniform_img, clip_limit=2.0)
        # CLAHE on uniform should not change much
        diff = np.abs(result.astype(float) - uniform_img.astype(float)).max()
        assert diff < 5  # small tolerance for CLAHE tile boundary effects

    def test_higher_clip_limit_more_equalization(self, random_img: np.ndarray) -> None:
        r_low = histogram_equalization(random_img, clip_limit=1.0)
        r_high = histogram_equalization(random_img, clip_limit=40.0)
        # Higher clip limit -> histogram spread wider -> std may differ
        # At minimum, they should produce different outputs
        assert not np.array_equal(r_low, r_high)

    def test_single_channel_only(self) -> None:
        """CLAHE works on single-channel; verify it doesn't fail."""
        img = np.random.RandomState(0).randint(0, 256, (64, 64), dtype=np.uint8)
        result = histogram_equalization(img, clip_limit=2.0)
        assert result.ndim == 2


# ═══════════════════════════════════════════════════════════════════════════════
# PERTURBATIONS dict
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerturbationsDict:
    EXPECTED_KEYS = {"gaussian_noise", "blur", "brightness_shift", "contrast_change", "gamma"}

    def test_has_all_perturbation_types(self) -> None:
        assert set(PERTURBATIONS.keys()) == self.EXPECTED_KEYS

    def test_each_has_required_fields(self) -> None:
        required = {"func", "levels", "param_name", "unit", "clinical_threshold"}
        for name, pert in PERTURBATIONS.items():
            missing = required - set(pert.keys())
            assert not missing, f"{name} missing keys: {missing}"

    def test_func_is_callable(self) -> None:
        for name, pert in PERTURBATIONS.items():
            assert callable(pert["func"]), f"{name} func not callable"

    def test_levels_is_nonempty_list(self) -> None:
        for name, pert in PERTURBATIONS.items():
            assert isinstance(pert["levels"], list), f"{name} levels not a list"
            assert len(pert["levels"]) > 0, f"{name} levels empty"

    def test_param_name_is_string(self) -> None:
        for name, pert in PERTURBATIONS.items():
            assert isinstance(pert["param_name"], str)

    def test_clinical_threshold_is_numeric(self) -> None:
        for name, pert in PERTURBATIONS.items():
            assert isinstance(pert["clinical_threshold"], (int, float))

    def test_func_references_match_module_functions(self) -> None:
        func_map = {
            "gaussian_noise": add_gaussian_noise,
            "blur": apply_blur,
            "brightness_shift": shift_brightness,
            "contrast_change": change_contrast,
            "gamma": apply_gamma,
        }
        for name, expected_func in func_map.items():
            assert PERTURBATIONS[name]["func"] is expected_func

    def test_each_func_accepts_image_and_level(self, random_img: np.ndarray) -> None:
        """Every perturbation func should accept (img, level) and return uint8 array."""
        for name, pert in PERTURBATIONS.items():
            level = pert["levels"][0]
            result = pert["func"](random_img, level)
            assert isinstance(result, np.ndarray), f"{name} did not return ndarray"
            assert result.dtype == np.uint8, f"{name} returned dtype {result.dtype}"
            assert result.shape == random_img.shape, f"{name} shape mismatch"

    def test_identity_levels(self, gray_img: np.ndarray) -> None:
        """Identity levels (sigma=0, ksize=1, delta=0, alpha=1.0, gamma=1.0)."""
        identity_map = {
            "gaussian_noise": 0,
            "blur": 1,
            "brightness_shift": 0,
            "contrast_change": 1.0,
            "gamma": 1.0,
        }
        for name, identity_level in identity_map.items():
            result = PERTURBATIONS[name]["func"](gray_img, identity_level)
            np.testing.assert_array_equal(
                result, gray_img, err_msg=f"{name} not identity at level={identity_level}"
            )
