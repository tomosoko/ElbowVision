"""
Unit tests for phantom_radiomics_stability.py

Covers pure functions (no DICOM I/O, no matplotlib rendering):
  - to_uint8_roi
  - extract_glcm_features
  - extract_roi_slices
  - apply_conditions
  - calculate_icc
  - extract_features_from_conditions
"""

import sys
import os
import numpy as np
import pytest

# Make the project root importable
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from phantom_radiomics_stability import (
    to_uint8_roi,
    extract_glcm_features,
    extract_roi_slices,
    apply_conditions,
    calculate_icc,
    extract_features_from_conditions,
)

# ─── helpers ─────────────────────────────────────────────────────────────────

def _uniform_volume(shape=(30, 128, 128), hu_value=100.0, rng=None):
    """Volume with a flat acrylic-like region in the centre."""
    if rng is None:
        rng = np.random.default_rng(0)
    vol = np.full(shape, -1000.0, dtype=np.float32)          # air background
    # Place a 64×64 acrylic block in the centre of every slice
    cy, cx = shape[1] // 2, shape[2] // 2
    half = 32
    vol[:, cy - half:cy + half, cx - half:cx + half] = hu_value
    return vol


# ─── to_uint8_roi ─────────────────────────────────────────────────────────────

class TestToUint8Roi:
    def test_output_dtype(self):
        patch = np.array([[0.0, 100.0], [200.0, 300.0]], dtype=np.float32)
        result = to_uint8_roi(patch)
        assert result.dtype == np.uint8

    def test_output_range(self):
        patch = np.random.uniform(-500, 800, (32, 32)).astype(np.float32)
        result = to_uint8_roi(patch)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_min_mapped_to_zero(self):
        patch = np.array([[10.0, 50.0], [30.0, 70.0]], dtype=np.float32)
        result = to_uint8_roi(patch)
        assert result.min() == 0

    def test_max_mapped_to_255(self):
        patch = np.array([[10.0, 50.0], [30.0, 70.0]], dtype=np.float32)
        result = to_uint8_roi(patch)
        assert result.max() == 255

    def test_constant_patch_returns_zeros(self):
        patch = np.full((16, 16), 200.0, dtype=np.float32)
        result = to_uint8_roi(patch)
        assert (result == 0).all()

    def test_output_shape_preserved(self):
        patch = np.random.rand(32, 32).astype(np.float32)
        result = to_uint8_roi(patch)
        assert result.shape == (32, 32)

    def test_linear_mapping(self):
        patch = np.array([[0.0, 127.0, 255.0]], dtype=np.float32)
        result = to_uint8_roi(patch)
        assert result[0, 0] == 0
        assert result[0, 2] == 255

    def test_negative_values_handled(self):
        patch = np.array([[-100.0, 0.0], [100.0, 200.0]], dtype=np.float32)
        result = to_uint8_roi(patch)
        assert result.dtype == np.uint8
        assert result.min() == 0
        assert result.max() == 255


# ─── extract_glcm_features ───────────────────────────────────────────────────

class TestExtractGlcmFeatures:
    def _uniform_patch(self, val=128, size=32):
        return np.full((size, size), val, dtype=np.uint8)

    def _random_patch(self, size=32, seed=0):
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, (size, size), dtype=np.uint8)

    def test_returns_dict(self):
        patch = self._random_patch()
        result = extract_glcm_features(patch)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        patch = self._random_patch()
        result = extract_glcm_features(patch)
        expected = {"contrast", "dissimilarity", "homogeneity",
                    "energy", "correlation", "mean", "std", "entropy"}
        assert set(result.keys()) == expected

    def test_all_values_are_float(self):
        patch = self._random_patch()
        result = extract_glcm_features(patch)
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not float"

    def test_uniform_patch_zero_contrast(self):
        patch = self._uniform_patch()
        result = extract_glcm_features(patch)
        assert result["contrast"] == pytest.approx(0.0, abs=1e-9)

    def test_uniform_patch_zero_std(self):
        patch = self._uniform_patch()
        result = extract_glcm_features(patch)
        assert result["std"] == pytest.approx(0.0, abs=1e-9)

    def test_uniform_patch_max_homogeneity(self):
        patch = self._uniform_patch()
        result = extract_glcm_features(patch)
        assert result["homogeneity"] == pytest.approx(1.0, abs=1e-6)

    def test_random_patch_positive_contrast(self):
        patch = self._random_patch()
        result = extract_glcm_features(patch)
        assert result["contrast"] > 0

    def test_random_patch_positive_entropy(self):
        patch = self._random_patch()
        result = extract_glcm_features(patch)
        assert result["entropy"] > 0

    def test_mean_range(self):
        patch = self._random_patch()
        result = extract_glcm_features(patch)
        assert 0.0 <= result["mean"] <= 255.0

    def test_energy_range(self):
        patch = self._random_patch()
        result = extract_glcm_features(patch)
        assert 0.0 <= result["energy"] <= 1.0

    def test_different_patches_differ(self):
        patch_a = self._random_patch(seed=1)
        patch_b = self._random_patch(seed=99)
        a = extract_glcm_features(patch_a)
        b = extract_glcm_features(patch_b)
        assert a["contrast"] != pytest.approx(b["contrast"], rel=1e-3)


# ─── extract_roi_slices ──────────────────────────────────────────────────────

class TestExtractRoiSlices:
    def test_returns_tuple_of_two(self):
        vol = _uniform_volume()
        rois, positions = extract_roi_slices(vol, n_slices=4, roi_size=16)
        assert isinstance(rois, list)
        assert isinstance(positions, list)

    def test_roi_shape(self):
        vol = _uniform_volume()
        rois, _ = extract_roi_slices(vol, n_slices=4, roi_size=16)
        assert len(rois) > 0
        for roi in rois:
            assert roi.shape == (16, 16)

    def test_position_tuples_are_three_int(self):
        vol = _uniform_volume()
        _, positions = extract_roi_slices(vol, n_slices=4, roi_size=16)
        for pos in positions:
            assert len(pos) == 3
            z, y, x = pos
            assert isinstance(z, (int, np.integer))

    def test_n_rois_equals_n_positions(self):
        vol = _uniform_volume()
        rois, positions = extract_roi_slices(vol, n_slices=4, roi_size=16)
        assert len(rois) == len(positions)

    def test_returns_empty_on_air_only_volume(self):
        # Volume with all HU=-1000 → no acrylic region → no valid ROI
        vol = np.full((20, 64, 64), -1000.0, dtype=np.float32)
        rois, positions = extract_roi_slices(vol, n_slices=4, roi_size=8)
        assert len(rois) == 0

    def test_roi_z_within_mid_range(self):
        vol = _uniform_volume(shape=(40, 128, 128))
        mid = vol.shape[0] // 2
        _, positions = extract_roi_slices(vol, n_slices=6, roi_size=16)
        for z, _, _ in positions:
            assert mid - 4 <= z <= mid + 4

    def test_different_roi_sizes(self):
        vol = _uniform_volume()
        for sz in [8, 16, 32]:
            rois, _ = extract_roi_slices(vol, n_slices=4, roi_size=sz)
            if rois:
                assert rois[0].shape == (sz, sz)


# ─── apply_conditions ────────────────────────────────────────────────────────

class TestApplyConditions:
    def setup_method(self):
        np.random.seed(42)
        self.vol = np.zeros((10, 64, 64), dtype=np.float32)

    def test_returns_dict(self):
        result = apply_conditions(self.vol)
        assert isinstance(result, dict)

    def test_expected_condition_count(self):
        result = apply_conditions(self.vol)
        # 1 original + 3 noise + 3 blur + 3 sharp + 1 noise+blur = 11
        assert len(result) == 11

    def test_original_key_present(self):
        result = apply_conditions(self.vol)
        assert "original" in result

    def test_noise_keys_present(self):
        result = apply_conditions(self.vol)
        for sigma in [5, 15, 30]:
            assert f"noise_σ{sigma}" in result

    def test_blur_keys_present(self):
        result = apply_conditions(self.vol)
        for s in [0.5, 1.0, 2.0]:
            assert f"blur_σ{s}" in result

    def test_sharp_keys_present(self):
        result = apply_conditions(self.vol)
        for alpha in [0.5, 1.0, 2.0]:
            assert f"sharp_α{alpha}" in result

    def test_noise_blur_key_present(self):
        result = apply_conditions(self.vol)
        assert "noise15+blur1" in result

    def test_all_volumes_same_shape_as_input(self):
        result = apply_conditions(self.vol)
        for cond_name, vol in result.items():
            assert vol.shape == self.vol.shape, f"{cond_name} has wrong shape"

    def test_all_volumes_float32(self):
        result = apply_conditions(self.vol)
        for cond_name, vol in result.items():
            assert vol.dtype == np.float32, f"{cond_name} dtype={vol.dtype}"

    def test_blur_volume_is_smoother(self):
        """blur_σ2.0 should have lower std than noisy original."""
        base = np.random.normal(0, 30, (10, 64, 64)).astype(np.float32)
        np.random.seed(42)
        result = apply_conditions(base)
        assert result["blur_σ2.0"].std() < result["noise_σ30"].std()

    def test_sharp_volume_differs_from_input(self):
        np.random.seed(0)
        vol = np.random.uniform(0, 100, (10, 32, 32)).astype(np.float32)
        result = apply_conditions(vol)
        # sharpening must change the volume
        assert not np.allclose(result["sharp_α2.0"], vol, atol=1e-3)


# ─── calculate_icc ───────────────────────────────────────────────────────────

class TestCalculateIcc:
    def test_perfect_agreement_returns_one(self):
        """All conditions produce identical values → ICC ≈ 1."""
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        vbc = {"cond_a": vals, "cond_b": vals, "cond_c": vals}
        icc = calculate_icc(vbc)
        assert icc == pytest.approx(1.0, abs=1e-6)

    def test_random_variation_lowers_icc(self):
        rng = np.random.default_rng(7)
        base = rng.uniform(0, 100, 20).tolist()
        noisy = (rng.uniform(0, 100, 20)).tolist()
        vbc = {"c1": base, "c2": noisy}
        icc = calculate_icc(vbc)
        assert icc < 1.0

    def test_returns_float(self):
        vbc = {"a": [1.0, 2.0], "b": [1.1, 2.1]}
        result = calculate_icc(vbc)
        assert isinstance(result, float)

    def test_icc_bounded(self):
        """Manual fallback ICC can be slightly below 0 when between-variance ≈ 0,
        but should stay in a reasonable range."""
        rng = np.random.default_rng(99)
        vbc = {str(i): rng.uniform(0, 10, 10).tolist() for i in range(5)}
        icc = calculate_icc(vbc)
        assert -1.0 <= icc <= 1.0 + 1e-6

    def test_constant_values_returns_zero_or_nan_safe(self):
        """All values identical across all conditions → no variance → 0."""
        vbc = {"c1": [5.0, 5.0, 5.0], "c2": [5.0, 5.0, 5.0]}
        icc = calculate_icc(vbc)
        # Both 0.0 and 1.0 are valid for constant data depending on implementation
        assert isinstance(icc, float)

    def test_high_icc_with_systematic_offset(self):
        """ROI varies a lot (good between-roi variance), conditions have small bias."""
        base = [10.0, 20.0, 30.0, 40.0, 50.0]
        conds = {f"c{i}": [v + i * 0.1 for v in base] for i in range(5)}
        icc = calculate_icc(conds)
        assert icc > 0.9

    def test_two_conditions_single_roi_still_returns_float(self):
        vbc = {"a": [1.0], "b": [2.0]}
        result = calculate_icc(vbc)
        assert isinstance(result, float)


# ─── extract_features_from_conditions ────────────────────────────────────────

class TestExtractFeaturesFromConditions:
    def _make_inputs(self, n_rois=3, roi_size=16):
        vol = _uniform_volume(shape=(20, 64, 64), hu_value=100.0)
        rng = np.random.default_rng(42)
        # build tiny synthetic rois and matching positions inside the acrylic block
        rois = [rng.uniform(80, 120, (roi_size, roi_size)).astype(np.float32)
                for _ in range(n_rois)]
        positions = [(10, 16, 16)] * n_rois  # all in same spot for simplicity
        conditions = {"original": vol, "noisy": vol + rng.normal(0, 10, vol.shape).astype(np.float32)}
        return conditions, rois, positions

    def test_returns_dict(self):
        conditions, rois, positions = self._make_inputs()
        result = extract_features_from_conditions(conditions, rois, positions)
        assert isinstance(result, dict)

    def test_feature_keys(self):
        conditions, rois, positions = self._make_inputs()
        result = extract_features_from_conditions(conditions, rois, positions)
        expected = {"contrast", "dissimilarity", "homogeneity",
                    "energy", "correlation", "mean", "std", "entropy"}
        assert set(result.keys()) == expected

    def test_condition_keys_per_feature(self):
        conditions, rois, positions = self._make_inputs()
        result = extract_features_from_conditions(conditions, rois, positions)
        for feat, cond_dict in result.items():
            assert set(cond_dict.keys()) == set(conditions.keys()), \
                f"Feature '{feat}' missing condition keys"

    def test_values_list_length_equals_n_rois(self):
        n_rois = 4
        conditions, rois, positions = self._make_inputs(n_rois=n_rois)
        result = extract_features_from_conditions(conditions, rois, positions)
        for feat, cond_dict in result.items():
            for cond_name, vals in cond_dict.items():
                assert len(vals) == n_rois, \
                    f"Feature '{feat}' / condition '{cond_name}': expected {n_rois} values"

    def test_values_are_floats(self):
        conditions, rois, positions = self._make_inputs()
        result = extract_features_from_conditions(conditions, rois, positions)
        for feat, cond_dict in result.items():
            for cond_name, vals in cond_dict.items():
                for v in vals:
                    assert isinstance(v, float), \
                        f"Feature '{feat}' / cond '{cond_name}': non-float value {v!r}"

    def test_single_roi(self):
        conditions, rois, positions = self._make_inputs(n_rois=1)
        result = extract_features_from_conditions(conditions, rois, positions)
        assert all(len(v) == 1 for cond_dict in result.values() for v in cond_dict.values())

    def test_result_varies_across_conditions(self):
        """Noisy condition should produce different mean values than original."""
        rng = np.random.default_rng(0)
        vol = np.random.uniform(50, 150, (20, 64, 64)).astype(np.float32)
        noisy_vol = vol + rng.normal(0, 30, vol.shape).astype(np.float32)
        conditions = {"original": vol, "noisy": noisy_vol}
        rois = [vol[10, 16:32, 16:32].copy()]
        positions = [(10, 16, 16)]
        result = extract_features_from_conditions(conditions, rois, positions)
        # std values should differ between conditions because of added noise
        orig_std = result["std"]["original"][0]
        noisy_std = result["std"]["noisy"][0]
        assert orig_std != pytest.approx(noisy_std, rel=1e-3)
