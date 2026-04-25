"""
Unit tests for phantom_seg_quality.py

Uses synthetic numpy volumes to avoid requiring real DICOM data.
Covers: sigma_to_mas_equivalent, make_gt_mask, make_degraded,
        seg_threshold, seg_otsu, seg_adaptive, calc_dice, calc_hd95,
        calc_metrics, SegResult, evaluate_condition, save_report
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

# Import module from top-level of ElbowVision
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phantom_seg_quality import (
    SegResult,
    calc_dice,
    calc_hd95,
    calc_metrics,
    evaluate_condition,
    make_degraded,
    make_gt_mask,
    save_report,
    seg_adaptive,
    seg_otsu,
    seg_threshold,
    sigma_to_mas_equivalent,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_bone_volume(shape=(20, 64, 64), hu_bone=400, hu_bg=-200):
    """Synthetic volume with a central bone-like cylinder (HU > 200)."""
    vol = np.full(shape, hu_bg, dtype=np.float32)
    cx, cy = shape[2] // 2, shape[1] // 2
    r = shape[1] // 6
    for z in range(shape[0]):
        Y, X = np.ogrid[:shape[1], :shape[2]]
        mask = (X - cx) ** 2 + (Y - cy) ** 2 < r ** 2
        vol[z][mask] = hu_bone
    return vol


def _make_binary_sphere(shape=(10, 32, 32), r=5):
    """Binary sphere mask centred in the volume."""
    mask = np.zeros(shape, dtype=np.uint8)
    cx, cy, cz = shape[2] // 2, shape[1] // 2, shape[0] // 2
    for z in range(shape[0]):
        Y, X = np.ogrid[:shape[1], :shape[2]]
        mask[z] = ((X - cx) ** 2 + (Y - cy) ** 2 + (z - cz) ** 2 < r ** 2).astype(np.uint8)
    return mask


# ─── sigma_to_mas_equivalent ──────────────────────────────────────────────────

class TestSigmaToMasEquivalent:
    def test_zero_sigma_returns_ref_mas(self):
        assert sigma_to_mas_equivalent(0) == 200.0

    def test_negative_sigma_returns_ref_mas(self):
        assert sigma_to_mas_equivalent(-5) == 200.0

    def test_ref_sigma_returns_ref_mas(self):
        result = sigma_to_mas_equivalent(5.0, ref_sigma=5.0, ref_mas=200.0)
        assert abs(result - 200.0) < 1e-6

    def test_double_sigma_halves_mAs_squared(self):
        # σ doubled → mAs = ref_mAs × (ref_σ/σ)² = ref_mAs × (1/2)² = ref_mAs/4
        result = sigma_to_mas_equivalent(10.0, ref_sigma=5.0, ref_mas=200.0)
        assert abs(result - 50.0) < 1e-6

    def test_half_sigma_quadruples_mAs(self):
        result = sigma_to_mas_equivalent(2.5, ref_sigma=5.0, ref_mas=200.0)
        assert abs(result - 800.0) < 1e-6

    def test_higher_noise_gives_lower_dose(self):
        low_noise = sigma_to_mas_equivalent(5.0)
        high_noise = sigma_to_mas_equivalent(20.0)
        assert high_noise < low_noise

    def test_custom_ref_params(self):
        result = sigma_to_mas_equivalent(10.0, ref_sigma=10.0, ref_mas=100.0)
        assert abs(result - 100.0) < 1e-6

    def test_return_type_is_float(self):
        result = sigma_to_mas_equivalent(5.0)
        assert isinstance(result, float)


# ─── make_gt_mask ─────────────────────────────────────────────────────────────

class TestMakeGtMask:
    def test_all_background_returns_empty_mask(self):
        vol = np.full((10, 32, 32), -500.0, dtype=np.float32)
        mask = make_gt_mask(vol)
        assert mask.sum() == 0

    def test_high_hu_region_is_detected(self):
        vol = _make_bone_volume(shape=(10, 32, 32))
        mask = make_gt_mask(vol, hu_min=200)
        assert mask.sum() > 0

    def test_mask_dtype_is_uint8(self):
        vol = _make_bone_volume(shape=(10, 32, 32))
        mask = make_gt_mask(vol)
        assert mask.dtype == np.uint8

    def test_mask_shape_matches_volume(self):
        shape = (15, 40, 50)
        vol = _make_bone_volume(shape=shape)
        mask = make_gt_mask(vol)
        assert mask.shape == shape

    def test_small_regions_filtered_out(self):
        # Single isolated voxel (< 50 voxels) should be removed
        vol = np.full((10, 32, 32), -500.0, dtype=np.float32)
        vol[5, 16, 16] = 500.0  # only 1 voxel — below 50 voxel threshold
        mask = make_gt_mask(vol, hu_min=200)
        assert mask.sum() == 0

    def test_large_bone_region_preserved(self):
        vol = _make_bone_volume(shape=(20, 64, 64))
        mask = make_gt_mask(vol, hu_min=200)
        # The cylinder has many voxels — should be preserved
        assert mask.sum() >= 50

    def test_hu_min_threshold_respected(self):
        vol = np.full((10, 32, 32), 150.0, dtype=np.float32)
        # All voxels = 150 HU < default hu_min=200
        mask = make_gt_mask(vol, hu_min=200)
        assert mask.sum() == 0

    def test_values_are_zero_or_one(self):
        vol = _make_bone_volume(shape=(10, 32, 32))
        mask = make_gt_mask(vol)
        assert set(np.unique(mask)).issubset({0, 1})


# ─── make_degraded ────────────────────────────────────────────────────────────

class TestMakeDegraded:
    def setup_method(self):
        self.vol = _make_bone_volume(shape=(10, 32, 32))

    def test_returns_list(self):
        conditions = make_degraded(self.vol, seed=42)
        assert isinstance(conditions, list)

    def test_contains_noise_conditions(self):
        conditions = make_degraded(self.vol, seed=42)
        types = [c["type"] for c in conditions]
        assert "noise" in types

    def test_contains_blur_conditions(self):
        conditions = make_degraded(self.vol, seed=42)
        types = [c["type"] for c in conditions]
        assert "blur" in types

    def test_contains_combo_conditions(self):
        conditions = make_degraded(self.vol, seed=42)
        types = [c["type"] for c in conditions]
        assert "combo" in types

    def test_each_condition_has_required_keys(self):
        conditions = make_degraded(self.vol, seed=42)
        required = {"type", "label", "param", "mas_eq", "volume"}
        for c in conditions:
            assert required.issubset(c.keys())

    def test_degraded_volume_shapes_match_input(self):
        conditions = make_degraded(self.vol, seed=42)
        for c in conditions:
            assert c["volume"].shape == self.vol.shape

    def test_same_seed_is_deterministic(self):
        c1 = make_degraded(self.vol, seed=42)
        c2 = make_degraded(self.vol, seed=42)
        for a, b in zip(c1, c2):
            np.testing.assert_array_equal(a["volume"], b["volume"])

    def test_different_seeds_give_different_noise(self):
        conditions_a = make_degraded(self.vol, seed=0)
        conditions_b = make_degraded(self.vol, seed=99)
        noise_a = [c for c in conditions_a if c["type"] == "noise" and c["param"] > 0]
        noise_b = [c for c in conditions_b if c["type"] == "noise" and c["param"] > 0]
        # At least one noise condition should differ
        diffs = [not np.array_equal(a["volume"], b["volume"])
                 for a, b in zip(noise_a, noise_b)]
        assert any(diffs)

    def test_blur_condition_volume_differs_from_original(self):
        conditions = make_degraded(self.vol, seed=42)
        blur_conds = [c for c in conditions if c["type"] == "blur"]
        assert len(blur_conds) > 0
        # Blurred volume should differ from original (for non-zero sigma)
        vol_blur = blur_conds[-1]["volume"]
        assert not np.array_equal(vol_blur, self.vol)


# ─── seg_threshold ────────────────────────────────────────────────────────────

class TestSegThreshold:
    def test_output_dtype_is_uint8(self):
        vol = _make_bone_volume(shape=(10, 32, 32))
        result = seg_threshold(vol)
        assert result.dtype == np.uint8

    def test_output_shape_matches_input(self):
        vol = _make_bone_volume(shape=(10, 32, 32))
        result = seg_threshold(vol)
        assert result.shape == vol.shape

    def test_all_background_gives_empty_mask(self):
        vol = np.full((10, 32, 32), -500.0, dtype=np.float32)
        result = seg_threshold(vol)
        assert result.sum() == 0

    def test_high_hu_volume_detected(self):
        vol = _make_bone_volume(shape=(10, 32, 32))
        result = seg_threshold(vol, hu_min=200)
        assert result.sum() > 0

    def test_values_are_binary(self):
        vol = _make_bone_volume(shape=(10, 32, 32))
        result = seg_threshold(vol)
        assert set(np.unique(result)).issubset({0, 1})

    def test_custom_hu_min(self):
        vol = np.full((10, 32, 32), 350.0, dtype=np.float32)
        # All voxels = 350 HU — above hu_min=300
        result_low = seg_threshold(vol, hu_min=300)
        # All voxels = 350 HU — below hu_min=400
        result_high = seg_threshold(vol, hu_min=400)
        assert result_low.sum() > 0
        assert result_high.sum() == 0


# ─── seg_otsu ─────────────────────────────────────────────────────────────────

class TestSegOtsu:
    def test_output_dtype_is_uint8(self):
        vol = _make_bone_volume(shape=(10, 32, 32))
        result = seg_otsu(vol)
        assert result.dtype == np.uint8

    def test_output_shape_matches_input(self):
        vol = _make_bone_volume(shape=(10, 32, 32))
        result = seg_otsu(vol)
        assert result.shape == vol.shape

    def test_values_are_binary(self):
        vol = _make_bone_volume(shape=(10, 32, 32))
        result = seg_otsu(vol)
        assert set(np.unique(result)).issubset({0, 1})

    def test_detects_bright_region(self):
        vol = _make_bone_volume(shape=(10, 64, 64), hu_bone=1000, hu_bg=-1000)
        result = seg_otsu(vol)
        assert result.sum() > 0


# ─── seg_adaptive ─────────────────────────────────────────────────────────────

class TestSegAdaptive:
    def test_output_dtype_is_uint8(self):
        vol = _make_bone_volume(shape=(10, 32, 32))
        result = seg_adaptive(vol)
        assert result.dtype == np.uint8

    def test_output_shape_matches_input(self):
        vol = _make_bone_volume(shape=(10, 32, 32))
        result = seg_adaptive(vol)
        assert result.shape == vol.shape

    def test_values_are_binary(self):
        vol = _make_bone_volume(shape=(10, 32, 32))
        result = seg_adaptive(vol)
        assert set(np.unique(result)).issubset({0, 1})


# ─── calc_dice ────────────────────────────────────────────────────────────────

class TestCalcDice:
    def test_perfect_overlap_is_one(self):
        mask = _make_binary_sphere(shape=(10, 32, 32), r=5)
        assert abs(calc_dice(mask, mask) - 1.0) < 1e-6

    def test_no_overlap_is_zero(self):
        pred = np.zeros((10, 32, 32), dtype=np.uint8)
        pred[:, :16, :] = 1
        gt = np.zeros((10, 32, 32), dtype=np.uint8)
        gt[:, 16:, :] = 1
        assert calc_dice(pred, gt) == 0.0

    def test_both_empty_is_one(self):
        pred = np.zeros((5, 10, 10), dtype=np.uint8)
        gt = np.zeros((5, 10, 10), dtype=np.uint8)
        assert calc_dice(pred, gt) == 1.0

    def test_partial_overlap_between_zero_and_one(self):
        pred = np.zeros((10, 32, 32), dtype=np.uint8)
        pred[:, 10:20, 10:20] = 1
        gt = np.zeros((10, 32, 32), dtype=np.uint8)
        gt[:, 15:25, 15:25] = 1
        result = calc_dice(pred, gt)
        assert 0.0 < result < 1.0

    def test_known_value(self):
        # pred = [1,1,0,0], gt = [1,0,1,0] → inter=1, denom=2+2=4 → dice=0.5
        pred = np.array([[[1, 1, 0, 0]]], dtype=np.uint8)
        gt   = np.array([[[1, 0, 1, 0]]], dtype=np.uint8)
        assert abs(calc_dice(pred, gt) - 0.5) < 1e-6

    def test_return_type_is_float(self):
        mask = _make_binary_sphere(shape=(5, 16, 16), r=3)
        assert isinstance(calc_dice(mask, mask), float)


# ─── calc_hd95 ────────────────────────────────────────────────────────────────

class TestCalcHd95:
    def test_identical_masks_low_distance(self):
        mask = _make_binary_sphere(shape=(10, 32, 32), r=5)
        result = calc_hd95(mask, mask)
        assert result < 1.0  # identical → should be ~0

    def test_empty_pred_returns_inf(self):
        pred = np.zeros((10, 32, 32), dtype=np.uint8)
        gt = _make_binary_sphere(shape=(10, 32, 32), r=5)
        result = calc_hd95(pred, gt)
        assert result == float("inf")

    def test_empty_gt_returns_inf(self):
        pred = _make_binary_sphere(shape=(10, 32, 32), r=5)
        gt = np.zeros((10, 32, 32), dtype=np.uint8)
        result = calc_hd95(pred, gt)
        assert result == float("inf")

    def test_larger_shift_gives_larger_distance(self):
        gt = _make_binary_sphere(shape=(10, 64, 64), r=5)
        pred_near = np.roll(gt, shift=2, axis=2)
        pred_far  = np.roll(gt, shift=10, axis=2)
        hd_near = calc_hd95(pred_near, gt)
        hd_far  = calc_hd95(pred_far,  gt)
        assert hd_far > hd_near

    def test_return_type_is_float(self):
        mask = _make_binary_sphere(shape=(10, 32, 32), r=5)
        assert isinstance(calc_hd95(mask, mask), float)


# ─── calc_metrics ─────────────────────────────────────────────────────────────

class TestCalcMetrics:
    def test_returns_dict_with_required_keys(self):
        mask = _make_binary_sphere(shape=(10, 32, 32), r=5)
        result = calc_metrics(mask, mask)
        assert {"dice", "hd95", "precision", "recall", "vol_sim"}.issubset(result.keys())

    def test_perfect_prediction_all_ones(self):
        mask = _make_binary_sphere(shape=(10, 32, 32), r=5)
        result = calc_metrics(mask, mask)
        assert abs(result["dice"] - 1.0) < 1e-6
        assert abs(result["precision"] - 1.0) < 1e-6
        assert abs(result["recall"] - 1.0) < 1e-6
        assert abs(result["vol_sim"] - 1.0) < 1e-5

    def test_no_prediction_gives_zero_precision(self):
        pred = np.zeros((10, 32, 32), dtype=np.uint8)
        gt = _make_binary_sphere(shape=(10, 32, 32), r=5)
        result = calc_metrics(pred, gt)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_vol_sim_is_between_zero_and_one(self):
        pred = _make_binary_sphere(shape=(10, 32, 32), r=3)
        gt   = _make_binary_sphere(shape=(10, 32, 32), r=5)
        result = calc_metrics(pred, gt)
        assert 0.0 <= result["vol_sim"] <= 1.0

    def test_all_values_are_floats(self):
        mask = _make_binary_sphere(shape=(10, 32, 32), r=5)
        result = calc_metrics(mask, mask)
        for v in result.values():
            assert isinstance(v, float)


# ─── SegResult ────────────────────────────────────────────────────────────────

class TestSegResult:
    def test_can_be_instantiated(self):
        r = SegResult(
            cond_type="noise", cond_label="σ=10", param=10.0,
            method="HU-Threshold", dice_mean=0.95, dice_std=0.01,
            hd95_mean=3.2, hd95_std=0.5, precision_mean=0.96,
            recall_mean=0.94, vs_mean=0.97,
        )
        assert r.cond_type == "noise"
        assert r.method == "HU-Threshold"
        assert r.dice_mean == 0.95

    def test_field_access(self):
        r = SegResult(
            cond_type="blur", cond_label="Gauss σ=1.0", param=1.0,
            method="Otsu", dice_mean=0.88, dice_std=0.0,
            hd95_mean=5.0, hd95_std=0.0, precision_mean=0.9,
            recall_mean=0.86, vs_mean=0.92,
        )
        assert r.param == 1.0
        assert r.vs_mean == 0.92


# ─── evaluate_condition ───────────────────────────────────────────────────────

class TestEvaluateCondition:
    def setup_method(self):
        self.vol = _make_bone_volume(shape=(10, 32, 32))
        self.gt = make_gt_mask(self.vol, hu_min=200)

    def _noise_cond(self, sigma=0):
        return {
            "type": "noise",
            "label": f"σ={sigma} HU",
            "param": float(sigma),
            "mas_eq": 9999 if sigma == 0 else sigma_to_mas_equivalent(sigma),
            "volume": self.vol.copy(),
        }

    def _blur_cond(self, sigma=1.0):
        from scipy.ndimage import gaussian_filter
        return {
            "type": "blur",
            "label": f"Gauss σ={sigma}",
            "param": float(sigma),
            "mas_eq": None,
            "volume": gaussian_filter(self.vol.astype(np.float32), sigma=sigma),
        }

    def test_returns_seg_result(self):
        cond = self._noise_cond(sigma=0)
        result = evaluate_condition(self.vol, self.gt, cond, "HU-Threshold", seg_threshold, n_trials=1)
        assert isinstance(result, SegResult)

    def test_result_fields_populated(self):
        cond = self._noise_cond(sigma=0)
        result = evaluate_condition(self.vol, self.gt, cond, "HU-Threshold", seg_threshold, n_trials=1)
        assert result.method == "HU-Threshold"
        assert result.cond_type == "noise"
        assert 0.0 <= result.dice_mean <= 1.0

    def test_blur_condition_uses_fixed_volume(self):
        cond = self._blur_cond(sigma=1.0)
        r1 = evaluate_condition(self.vol, self.gt, cond, "Otsu", seg_otsu, n_trials=1)
        r2 = evaluate_condition(self.vol, self.gt, cond, "Otsu", seg_otsu, n_trials=1)
        # Deterministic: same result both times
        assert r1.dice_mean == r2.dice_mean

    def test_high_noise_lowers_dice(self):
        cond_clean = self._noise_cond(sigma=0)
        cond_noisy = self._noise_cond(sigma=80)
        r_clean = evaluate_condition(self.vol, self.gt, cond_clean, "HU-Threshold", seg_threshold, n_trials=1)
        r_noisy = evaluate_condition(self.vol, self.gt, cond_noisy, "HU-Threshold", seg_threshold, n_trials=3)
        # Dice with heavy noise should be <= clean dice
        assert r_noisy.dice_mean <= r_clean.dice_mean + 0.1


# ─── save_report ──────────────────────────────────────────────────────────────

class TestSaveReport:
    def _make_results(self):
        return [
            SegResult("noise", "σ=0 HU",     0.0,  "HU-Threshold", 0.98, 0.0,  2.1, 0.0,  0.98, 0.97, 0.99),
            SegResult("noise", "σ=10 HU",    10.0, "HU-Threshold", 0.92, 0.02, 3.5, 0.3,  0.93, 0.91, 0.95),
            SegResult("noise", "σ=60 HU",    60.0, "HU-Threshold", 0.70, 0.05, 8.0, 1.0,  0.72, 0.68, 0.75),
            SegResult("blur",  "Gauss σ=1.0", 1.0, "HU-Threshold", 0.95, 0.0,  2.5, 0.0,  0.96, 0.94, 0.97),
            SegResult("noise", "σ=0 HU",      0.0, "Otsu",         0.85, 0.0,  4.0, 0.0,  0.86, 0.84, 0.88),
            SegResult("noise", "σ=10 HU",    10.0, "Otsu",         0.80, 0.03, 5.0, 0.5,  0.81, 0.79, 0.82),
            SegResult("noise", "σ=60 HU",    60.0, "Otsu",         0.60, 0.08, 12.0, 2.0, 0.62, 0.58, 0.63),
            SegResult("noise", "σ=0 HU",      0.0, "Adaptive",     0.88, 0.0,  3.5, 0.0,  0.89, 0.87, 0.90),
            SegResult("noise", "σ=10 HU",    10.0, "Adaptive",     0.83, 0.02, 4.5, 0.4,  0.84, 0.82, 0.86),
            SegResult("noise", "σ=60 HU",    60.0, "Adaptive",     0.65, 0.06, 10.0, 1.5, 0.66, 0.64, 0.68),
        ]

    def test_creates_file(self):
        results = self._make_results()
        gt = _make_binary_sphere(shape=(10, 32, 32), r=5)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name
        try:
            save_report(results, gt, path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_report_contains_method_names(self):
        results = self._make_results()
        gt = _make_binary_sphere(shape=(10, 32, 32), r=5)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="r") as f:
            path = f.name
        try:
            save_report(results, gt, path)
            content = open(path, encoding="utf-8").read()
            assert "HU-Threshold" in content
        finally:
            os.unlink(path)

    def test_report_contains_noise_section(self):
        results = self._make_results()
        gt = _make_binary_sphere(shape=(10, 32, 32), r=5)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name
        try:
            save_report(results, gt, path)
            content = open(path, encoding="utf-8").read()
            assert "NOISE LEVEL" in content
        finally:
            os.unlink(path)

    def test_report_contains_dice_values(self):
        results = self._make_results()
        gt = _make_binary_sphere(shape=(10, 32, 32), r=5)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name
        try:
            save_report(results, gt, path)
            content = open(path, encoding="utf-8").read()
            assert "Dice=" in content
        finally:
            os.unlink(path)

    def test_report_grade_labels(self):
        results = self._make_results()
        gt = _make_binary_sphere(shape=(10, 32, 32), r=5)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name
        try:
            save_report(results, gt, path)
            content = open(path, encoding="utf-8").read()
            # At least one grade label should appear
            assert any(g in content for g in ("EXCELLENT", "ACCEPTABLE", "POOR"))
        finally:
            os.unlink(path)
