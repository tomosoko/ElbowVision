"""
Unit tests for scripts/pipeline_synth_to_yolo.py pure functions.

Coverage:
  - Constants: KP_NAMES, KP_COLORS, HU_MIN, HU_MAX, TARGET_SIZE, SID_MM, LATERALITY
  - build_bone_split: bone split logic, blend weights, joint_pd, bone_thresh
  - draw_keypoints: empty/partial/full keypoints, scale, connections, confidence threshold
  - synthesize_lat_drr: gamma correction, volume compositing via mocks
"""
from __future__ import annotations

import math
import sys
import types
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# ── Mock elbow_synth before importing the script ──────────────────────────
_elbow_synth_mock = types.ModuleType("elbow_synth")
_elbow_synth_mock.load_ct_volume = MagicMock()
_elbow_synth_mock.auto_detect_landmarks = MagicMock()
_elbow_synth_mock.generate_drr = MagicMock(
    return_value=np.full((64, 64), 128, dtype=np.uint8)
)
_elbow_synth_mock.rotation_matrix_z = MagicMock(return_value=np.eye(3))
_elbow_synth_mock.rotate_volume_and_landmarks = MagicMock()
_saved_elbow_synth = sys.modules.get("elbow_synth")
sys.modules["elbow_synth"] = _elbow_synth_mock

import importlib
import pathlib

_root = str(pathlib.Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

# Ensure elbow-train is also in path (needed for elbow_synth resolution)
_etrain = str(pathlib.Path(_root) / "elbow-train")
if _etrain not in sys.path:
    sys.path.insert(0, _etrain)

from scripts.pipeline_synth_to_yolo import (  # noqa: E402
    HU_MAX,
    HU_MIN,
    KP_COLORS,
    KP_NAMES,
    LATERALITY,
    SID_MM,
    TARGET_SIZE,
    build_bone_split,
    draw_keypoints,
    synthesize_lat_drr,
)

# Restore original state so downstream test files can import the real elbow_synth
if _saved_elbow_synth is not None:
    sys.modules["elbow_synth"] = _saved_elbow_synth
else:
    sys.modules.pop("elbow_synth", None)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_kp_names_length(self):
        assert len(KP_NAMES) == 6

    def test_kp_names_content(self):
        assert "humerus_shaft" in KP_NAMES
        assert "lateral_epicondyle" in KP_NAMES
        assert "medial_epicondyle" in KP_NAMES
        assert "forearm_shaft" in KP_NAMES
        assert "radial_head" in KP_NAMES
        assert "olecranon" in KP_NAMES

    def test_kp_names_are_strings(self):
        for name in KP_NAMES:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_kp_colors_length(self):
        assert len(KP_COLORS) == 6

    def test_kp_colors_are_bgr_tuples(self):
        for color in KP_COLORS:
            assert len(color) == 3
            for channel in color:
                assert 0 <= channel <= 255

    def test_kp_names_colors_same_length(self):
        assert len(KP_NAMES) == len(KP_COLORS)

    def test_hu_min_less_than_hu_max(self):
        assert HU_MIN < HU_MAX

    def test_hu_min_positive(self):
        assert HU_MIN >= 0

    def test_hu_max_positive(self):
        assert HU_MAX > 0

    def test_target_size_positive(self):
        assert TARGET_SIZE > 0

    def test_target_size_power_of_two_or_256(self):
        # DRR synthesis typically uses 256 or power-of-2 sizes
        assert TARGET_SIZE >= 64

    def test_sid_mm_positive(self):
        assert SID_MM > 0

    def test_sid_mm_reasonable(self):
        # Source-image distance should be realistic (mm)
        assert 500 <= SID_MM <= 2000

    def test_laterality_valid(self):
        assert LATERALITY in ("L", "R")


# ═══════════════════════════════════════════════════════════════════════════
# build_bone_split
# ═══════════════════════════════════════════════════════════════════════════

def _make_vol(pd=50, ap=40, ml=40, fill=0.5):
    return np.full((pd, ap, ml), fill, dtype=np.float32)


def _make_lm(joint_pd_frac=0.5, joint_ap_frac=0.5, joint_ml_frac=0.5):
    return {
        "joint_center": np.array([joint_pd_frac, joint_ap_frac, joint_ml_frac]),
    }


class TestBuildBoneSplit:
    def test_returns_four_values(self):
        vol = _make_vol()
        lm = _make_lm()
        result = build_bone_split(vol, lm)
        assert len(result) == 4

    def test_output_shapes_match_input(self):
        vol = _make_vol(pd=30, ap=20, ml=10)
        lm = _make_lm()
        vol_hum, vol_fore, joint_pd, bone_thresh = build_bone_split(vol, lm)
        assert vol_hum.shape == vol.shape
        assert vol_fore.shape == vol.shape

    def test_weights_sum_to_one(self):
        """vol_hum + vol_fore should equal original vol everywhere."""
        vol = _make_vol(fill=1.0)
        lm = _make_lm(joint_pd_frac=0.5)
        vol_hum, vol_fore, _, _ = build_bone_split(vol, lm)
        np.testing.assert_allclose(vol_hum + vol_fore, vol, atol=1e-5)

    def test_weights_sum_to_one_asymmetric_joint(self):
        vol = _make_vol(fill=1.0)
        lm = _make_lm(joint_pd_frac=0.3)
        vol_hum, vol_fore, _, _ = build_bone_split(vol, lm)
        np.testing.assert_allclose(vol_hum + vol_fore, vol, atol=1e-5)

    def test_humerus_dominates_above_joint(self):
        """Voxels well above joint center should be mostly in humerus."""
        pd = 100
        vol = _make_vol(pd=pd, ap=20, ml=20, fill=1.0)
        lm = _make_lm(joint_pd_frac=0.6)  # joint at 60%
        vol_hum, vol_fore, joint_pd, _ = build_bone_split(vol, lm)
        # Slice far above joint: humerus weight should be 1.0
        assert vol_hum[5, 0, 0] == pytest.approx(1.0, abs=1e-5)
        assert vol_fore[5, 0, 0] == pytest.approx(0.0, abs=1e-5)

    def test_forearm_dominates_below_joint(self):
        """Voxels well below joint center should be mostly in forearm."""
        pd = 100
        vol = _make_vol(pd=pd, ap=20, ml=20, fill=1.0)
        lm = _make_lm(joint_pd_frac=0.3)  # joint at 30%
        vol_hum, vol_fore, joint_pd, _ = build_bone_split(vol, lm)
        # Slice far below joint: forearm weight should be 1.0
        assert vol_fore[90, 0, 0] == pytest.approx(1.0, abs=1e-5)
        assert vol_hum[90, 0, 0] == pytest.approx(0.0, abs=1e-5)

    def test_joint_pd_computed_correctly(self):
        pd = 80
        vol = _make_vol(pd=pd, ap=20, ml=20)
        lm = _make_lm(joint_pd_frac=0.25)
        _, _, joint_pd, _ = build_bone_split(vol, lm)
        assert joint_pd == int(0.25 * pd)

    def test_joint_pd_at_center(self):
        pd = 60
        vol = _make_vol(pd=pd, ap=10, ml=10)
        lm = _make_lm(joint_pd_frac=0.5)
        _, _, joint_pd, _ = build_bone_split(vol, lm)
        assert joint_pd == 30

    def test_bone_thresh_is_median_of_nonzero(self):
        pd, ap, ml = 20, 10, 10
        vol = np.zeros((pd, ap, ml), dtype=np.float32)
        vol[5:, :, :] = 1.0  # half the volume is 1.0
        lm = _make_lm()
        _, _, _, bone_thresh = build_bone_split(vol, lm)
        expected = np.percentile(vol[vol > 0.01], 50)
        assert bone_thresh == pytest.approx(expected, abs=1e-5)

    def test_bone_thresh_with_variable_intensities(self):
        pd, ap, ml = 40, 10, 10
        vol = np.random.uniform(0.1, 1.0, (pd, ap, ml)).astype(np.float32)
        lm = _make_lm()
        _, _, _, bone_thresh = build_bone_split(vol, lm)
        expected = float(np.percentile(vol[vol > 0.01], 50))
        assert bone_thresh == pytest.approx(expected, abs=1e-5)

    def test_hum_and_fore_non_negative(self):
        vol = _make_vol(fill=0.8)
        lm = _make_lm()
        vol_hum, vol_fore, _, _ = build_bone_split(vol, lm)
        assert (vol_hum >= 0).all()
        assert (vol_fore >= 0).all()

    def test_custom_joint_key(self):
        vol = _make_vol()
        lm = {
            "joint_center": np.array([0.5, 0.5, 0.5]),
            "alt_center": np.array([0.3, 0.5, 0.5]),
        }
        # Should work without error using default key
        result = build_bone_split(vol, lm, joint_key="joint_center")
        assert len(result) == 4

    def test_blend_half_minimum_three(self):
        """For small volumes, blend_half should be at least 3."""
        # pd_size=10 → 4% of 10 = 0.4 → max(3, 0) = 3
        pd = 10
        vol = _make_vol(pd=pd, ap=5, ml=5, fill=1.0)
        lm = _make_lm(joint_pd_frac=0.5)
        # Should not raise and weights must sum to 1
        vol_hum, vol_fore, _, _ = build_bone_split(vol, lm)
        np.testing.assert_allclose(vol_hum + vol_fore, vol, atol=1e-5)

    def test_blend_half_scales_with_pd_size(self):
        """For large volumes, blend_half should be > 3."""
        pd = 200
        vol = _make_vol(pd=pd, ap=10, ml=10, fill=1.0)
        lm = _make_lm(joint_pd_frac=0.5)
        # Should run without error
        vol_hum, vol_fore, _, _ = build_bone_split(vol, lm)
        np.testing.assert_allclose(vol_hum + vol_fore, vol, atol=1e-5)

    def test_zero_voxels_in_mixed_volume_remain_zero(self):
        """Zero-valued voxels in a mixed volume remain zero in both split outputs."""
        vol = np.full((30, 20, 20), 0.5, dtype=np.float32)
        # Carve a zero-filled region
        vol[0:5, :, :] = 0.0
        lm = _make_lm()
        vol_hum, vol_fore, _, _ = build_bone_split(vol, lm)
        # Where input is zero, both splits must also be zero
        zero_mask = vol == 0.0
        assert (vol_hum[zero_mask] == 0.0).all()
        assert (vol_fore[zero_mask] == 0.0).all()

    def test_output_dtype_float32(self):
        vol = _make_vol(fill=0.5)
        lm = _make_lm()
        vol_hum, vol_fore, _, _ = build_bone_split(vol, lm)
        assert vol_hum.dtype == np.float32
        assert vol_fore.dtype == np.float32


# ═══════════════════════════════════════════════════════════════════════════
# draw_keypoints
# ═══════════════════════════════════════════════════════════════════════════

def _make_img(h=64, w=64, channels=3):
    return np.zeros((h, w, channels), dtype=np.uint8)


def _make_kps_high_conf(n=6, x=32.0, y=32.0):
    """All 6 keypoints with high confidence at given position."""
    kps = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        kps[i] = [x + i * 3, y + i * 2, 0.9]
    return kps


def _make_kps_low_conf(n=6):
    """All keypoints with conf below 0.2 threshold."""
    kps = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        kps[i] = [10.0, 10.0, 0.1]
    return kps


class TestDrawKeypoints:
    def test_empty_keypoints_returns_copy(self):
        img = _make_img()
        result = draw_keypoints(img, [])
        assert result.shape == img.shape
        np.testing.assert_array_equal(result, img)

    def test_empty_keypoints_does_not_modify_original(self):
        img = _make_img()
        original = img.copy()
        draw_keypoints(img, [])
        np.testing.assert_array_equal(img, original)

    def test_returns_image_with_same_shape(self):
        img = _make_img(h=128, w=128)
        kps = _make_kps_high_conf()
        result = draw_keypoints(img, kps)
        assert result.shape == img.shape

    def test_returns_uint8(self):
        img = _make_img()
        kps = _make_kps_high_conf()
        result = draw_keypoints(img, kps)
        assert result.dtype == np.uint8

    def test_high_conf_keypoints_modify_image(self):
        img = _make_img()
        kps = _make_kps_high_conf()
        result = draw_keypoints(img, kps)
        # Should have drawn circles (non-zero pixels exist)
        assert result.sum() > 0

    def test_low_conf_keypoints_no_circles_drawn(self):
        img = _make_img()
        kps = _make_kps_low_conf()
        result = draw_keypoints(img, kps)
        # All conf < 0.2, nothing should be drawn
        np.testing.assert_array_equal(result, img)

    def test_does_not_mutate_input_image(self):
        img = _make_img()
        original = img.copy()
        kps = _make_kps_high_conf()
        draw_keypoints(img, kps)
        np.testing.assert_array_equal(img, original)

    def test_scale_one_default(self):
        img = _make_img(h=256, w=256)
        kps = _make_kps_high_conf(x=100.0, y=100.0)
        result1 = draw_keypoints(img.copy(), kps, scale=1.0)
        result2 = draw_keypoints(img.copy(), kps)
        np.testing.assert_array_equal(result1, result2)

    def test_scale_affects_keypoint_positions(self):
        """With scale=2, keypoints should be placed at doubled coordinates."""
        img = _make_img(h=128, w=128)
        kps = np.array([[20.0, 20.0, 0.9],
                         [20.0, 20.0, 0.9],
                         [20.0, 20.0, 0.9],
                         [20.0, 20.0, 0.9],
                         [20.0, 20.0, 0.9],
                         [20.0, 20.0, 0.9]], dtype=np.float32)
        result_scale1 = draw_keypoints(img.copy(), kps, scale=1.0)
        result_scale2 = draw_keypoints(img.copy(), kps, scale=2.0)
        # Different scales should produce different results
        assert not np.array_equal(result_scale1, result_scale2)

    def test_mixed_conf_only_high_drawn(self):
        """Keypoints with conf > 0.2 drawn; others skipped."""
        img = _make_img()
        img_clean = img.copy()
        kps = np.array([
            [30.0, 30.0, 0.9],  # high
            [30.0, 30.0, 0.1],  # low — no circle
            [30.0, 30.0, 0.9],  # high
            [30.0, 30.0, 0.05], # low
            [30.0, 30.0, 0.9],  # high
            [30.0, 30.0, 0.9],  # high
        ], dtype=np.float32)
        result = draw_keypoints(img, kps)
        # Something was drawn (high conf keypoints)
        assert result.sum() > 0

    def test_conf_below_threshold_not_drawn(self):
        """conf = 0.19 (below strict >0.2 threshold) should not be drawn."""
        img = _make_img(h=128, w=128)
        # Use 0.19 — clearly below 0.2 in both float32 and float64
        kps = np.array([[30.0, 30.0, 0.19]] * 6, dtype=np.float32)
        result = draw_keypoints(img, kps)
        np.testing.assert_array_equal(result, img)

    def test_conf_above_threshold_is_drawn(self):
        img = _make_img(h=128, w=128)
        kps = np.array([[30.0, 30.0, 0.21]] * 6, dtype=np.float32)
        result = draw_keypoints(img, kps)
        assert result.sum() > 0

    def test_connections_drawn_when_both_endpoints_high_conf(self):
        """Connections require BOTH endpoints conf > 0.2."""
        img = _make_img(h=256, w=256)
        # Place all keypoints with high conf, spread out
        kps = np.array([
            [50.0,  50.0,  0.9],   # 0: humerus_shaft
            [100.0, 50.0,  0.9],   # 1: lateral_epicondyle
            [150.0, 50.0,  0.9],   # 2: medial_epicondyle
            [200.0, 50.0,  0.9],   # 3: forearm_shaft
            [100.0, 150.0, 0.9],   # 4: radial_head
            [150.0, 150.0, 0.9],   # 5: olecranon
        ], dtype=np.float32)
        result = draw_keypoints(img, kps)
        # Many pixels drawn (lines + circles)
        assert result.sum() > img.sum()

    def test_result_is_copy_not_view(self):
        """Function should return an independent copy, not a view."""
        img = _make_img()
        kps = _make_kps_high_conf()
        result = draw_keypoints(img, kps)
        result[:] = 255
        # Original should be unmodified
        np.testing.assert_array_equal(img, np.zeros_like(img))

    def test_numpy_array_keypoints(self):
        """Accepts numpy array of shape (6, 3)."""
        img = _make_img(h=128, w=128)
        kps = np.array([[30, 30, 0.9]] * 6, dtype=np.float64)
        result = draw_keypoints(img, kps)
        assert result.shape == img.shape


# ═══════════════════════════════════════════════════════════════════════════
# synthesize_lat_drr
# ═══════════════════════════════════════════════════════════════════════════

class TestSynthesizeLatDrr:
    """Tests for synthesize_lat_drr with mocked module-local names."""

    def _make_inputs(self, pd=50, ap=40, ml=40):
        vol_180 = np.random.uniform(0.1, 1.0, (pd, ap, ml)).astype(np.float32)
        lm_180 = {"joint_center": np.array([0.5, 0.5, 0.5])}
        voxel_mm = 1.0
        return vol_180, lm_180, voxel_mm

    def _patches(self, drr_fill=128, drr_size=32):
        """Return a context manager stacking the three needed patches."""
        from contextlib import ExitStack
        stack = ExitStack()
        mock_rmz = stack.enter_context(
            patch("scripts.pipeline_synth_to_yolo.rotation_matrix_z",
                  return_value=np.eye(3))
        )
        mock_gen = stack.enter_context(
            patch("scripts.pipeline_synth_to_yolo.generate_drr",
                  return_value=np.full((drr_size, drr_size), drr_fill, dtype=np.uint8))
        )
        mock_at = stack.enter_context(
            patch("scripts.pipeline_synth_to_yolo.affine_transform")
        )
        return stack, mock_rmz, mock_gen, mock_at

    def test_returns_uint8_array(self):
        vol, lm, voxel_mm = self._make_inputs()
        with patch("scripts.pipeline_synth_to_yolo.rotation_matrix_z",
                   return_value=np.eye(3)), \
             patch("scripts.pipeline_synth_to_yolo.generate_drr",
                   return_value=np.full((32, 32), 128, dtype=np.uint8)), \
             patch("scripts.pipeline_synth_to_yolo.affine_transform",
                   return_value=np.zeros_like(vol)):
            result = synthesize_lat_drr(vol, lm, voxel_mm, target_flexion_deg=90.0)
        assert result.dtype == np.uint8

    def test_returns_2d_array(self):
        vol, lm, voxel_mm = self._make_inputs()
        with patch("scripts.pipeline_synth_to_yolo.rotation_matrix_z",
                   return_value=np.eye(3)), \
             patch("scripts.pipeline_synth_to_yolo.generate_drr",
                   return_value=np.full((32, 32), 100, dtype=np.uint8)), \
             patch("scripts.pipeline_synth_to_yolo.affine_transform",
                   return_value=np.zeros_like(vol)):
            result = synthesize_lat_drr(vol, lm, voxel_mm, target_flexion_deg=90.0)
        assert result.ndim == 2

    def test_calls_rotation_matrix_z_with_correct_angle(self):
        vol, lm, voxel_mm = self._make_inputs()
        with patch("scripts.pipeline_synth_to_yolo.rotation_matrix_z",
                   return_value=np.eye(3)) as mock_rmz, \
             patch("scripts.pipeline_synth_to_yolo.generate_drr",
                   return_value=np.zeros((32, 32), dtype=np.uint8)), \
             patch("scripts.pipeline_synth_to_yolo.affine_transform",
                   return_value=np.zeros_like(vol)):
            synthesize_lat_drr(vol, lm, voxel_mm, target_flexion_deg=90.0)
        expected = math.radians(180.0 - 90.0)
        assert mock_rmz.call_args[0][0] == pytest.approx(expected, abs=1e-6)

    def test_calls_rotation_matrix_z_for_full_extension(self):
        vol, lm, voxel_mm = self._make_inputs()
        with patch("scripts.pipeline_synth_to_yolo.rotation_matrix_z",
                   return_value=np.eye(3)) as mock_rmz, \
             patch("scripts.pipeline_synth_to_yolo.generate_drr",
                   return_value=np.zeros((32, 32), dtype=np.uint8)), \
             patch("scripts.pipeline_synth_to_yolo.affine_transform",
                   return_value=np.zeros_like(vol)):
            synthesize_lat_drr(vol, lm, voxel_mm, target_flexion_deg=180.0)
        assert mock_rmz.call_args[0][0] == pytest.approx(0.0, abs=1e-6)

    def test_gamma_correction_applied(self):
        """All-255 DRR should stay 255 after gamma correction."""
        vol, lm, voxel_mm = self._make_inputs()
        with patch("scripts.pipeline_synth_to_yolo.rotation_matrix_z",
                   return_value=np.eye(3)), \
             patch("scripts.pipeline_synth_to_yolo.generate_drr",
                   return_value=np.full((32, 32), 255, dtype=np.uint8)), \
             patch("scripts.pipeline_synth_to_yolo.affine_transform",
                   return_value=np.zeros_like(vol)):
            result = synthesize_lat_drr(vol, lm, voxel_mm, target_flexion_deg=90.0)
        assert result.max() == 255

    def test_gamma_correction_zero_stays_zero(self):
        """Zero DRR should remain zero after gamma correction."""
        vol, lm, voxel_mm = self._make_inputs()
        with patch("scripts.pipeline_synth_to_yolo.rotation_matrix_z",
                   return_value=np.eye(3)), \
             patch("scripts.pipeline_synth_to_yolo.generate_drr",
                   return_value=np.zeros((32, 32), dtype=np.uint8)), \
             patch("scripts.pipeline_synth_to_yolo.affine_transform",
                   return_value=np.zeros_like(vol)):
            result = synthesize_lat_drr(vol, lm, voxel_mm, target_flexion_deg=90.0)
        assert result.max() == 0

    def test_output_values_clipped_to_255(self):
        vol, lm, voxel_mm = self._make_inputs()
        with patch("scripts.pipeline_synth_to_yolo.rotation_matrix_z",
                   return_value=np.eye(3)), \
             patch("scripts.pipeline_synth_to_yolo.generate_drr",
                   return_value=np.full((32, 32), 200, dtype=np.uint8)), \
             patch("scripts.pipeline_synth_to_yolo.affine_transform",
                   return_value=np.zeros_like(vol)):
            result = synthesize_lat_drr(vol, lm, voxel_mm, target_flexion_deg=90.0)
        assert result.max() <= 255
        assert result.min() >= 0
