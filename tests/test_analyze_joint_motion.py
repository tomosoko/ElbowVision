"""Tests for scripts/analyze_joint_motion.py pure-function utilities.

Covers the two pure functions that operate solely on numpy arrays:
  - calc_flexion_angle
  - bone_outline_2d
"""
import sys
import os

import numpy as np
import pytest

# Make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from analyze_joint_motion import calc_flexion_angle, bone_outline_2d


# ── Helpers ───────────────────────────────────────────────────────────────

def make_volume(shape=(8, 8, 8), fill=0.0):
    return np.full(shape, fill, dtype=np.float32)


def make_volume_with_sphere(shape=(8, 8, 8), center=None, radius=2, value=1.0):
    """Return a float volume with a sphere of given value."""
    vol = np.zeros(shape, dtype=np.float32)
    if center is None:
        center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if (i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2 <= radius**2:
                    vol[i, j, k] = value
    return vol


# ── calc_flexion_angle ────────────────────────────────────────────────────

class TestCalcFlexionAngle:
    """Tests for calc_flexion_angle(humerus_shaft, joint_center, forearm_shaft)."""

    def test_returns_float(self):
        result = calc_flexion_angle([0, 0, 0], [1, 0, 0], [2, 0, 0])
        assert isinstance(result, float)

    def test_straight_arm_returns_180(self):
        # Collinear: humerus behind joint, forearm ahead -> 180 degrees
        # Tolerance 1e-3 because the epsilon guard (1e-10) causes tiny numeric error
        result = calc_flexion_angle([0, 0, 0], [1, 0, 0], [2, 0, 0])
        assert abs(result - 180.0) < 1e-3

    def test_right_angle_flexion_90_degrees(self):
        # v1 = h-j = [-1, 0, 0], v2 = f-j = [0, 1, 0] -> 90 degrees
        result = calc_flexion_angle([0, 0, 0], [1, 0, 0], [1, 1, 0])
        assert abs(result - 90.0) < 1e-3

    def test_fully_folded_returns_0(self):
        # v1 = [-1, 0, 0], v2 = [-1, 0, 0] -> 0 degrees
        result = calc_flexion_angle([0, 0, 0], [1, 0, 0], [0, 0, 0])
        assert abs(result - 0.0) < 1e-3

    def test_45_degree_flexion(self):
        # v1 = h-j = [-1, 0, 0]
        # For angle=45°, v2 must be at 45° from v1: v2 = [-cos45, sin45, 0]
        # f = j + v2 = [1 - cos45, sin45, 0]
        angle_rad = np.radians(45)
        h = [0, 0, 0]
        j = [1, 0, 0]
        f = [1 - np.cos(angle_rad), np.sin(angle_rad), 0]
        result = calc_flexion_angle(h, j, f)
        assert abs(result - 45.0) < 1e-3

    def test_135_degree_flexion(self):
        # v1 = h-j = [-1, 0, 0]
        # For angle=135°, v2 at 135° from v1: v2 = [cos45, sin45, 0]
        # f = j + v2 = [1 + cos45, sin45, 0]
        angle_rad = np.radians(45)  # 180-135=45
        h = [0, 0, 0]
        j = [1, 0, 0]
        f = [1 + np.cos(angle_rad), np.sin(angle_rad), 0]
        result = calc_flexion_angle(h, j, f)
        assert abs(result - 135.0) < 1e-3

    def test_result_in_range_0_to_180(self):
        # Use correct geometry: v1 = h-j = [-1,0,0], v2 at target_angle from v1
        for angle_deg in [0, 30, 60, 90, 120, 150, 180]:
            angle_rad = np.radians(angle_deg)
            h = [0, 0, 0]
            j = [1, 0, 0]
            # v2 = (-cos(angle_deg), sin(angle_deg), 0) -> angle between v1 and v2 = angle_deg
            f = [1 - np.cos(angle_rad), np.sin(angle_rad), 0]
            result = calc_flexion_angle(h, j, f)
            assert 0.0 <= result <= 180.0

    def test_symmetry_humerus_forearm_swap(self):
        # Angle at joint should be same regardless of which limb is 'humerus' or 'forearm'
        h = [0, 0, 0]
        j = [1, 0, 0]
        f = [1, 1, 0]
        result1 = calc_flexion_angle(h, j, f)
        result2 = calc_flexion_angle(f, j, h)
        assert abs(result1 - result2) < 1e-6

    def test_3d_vectors(self):
        # Test with truly 3D coordinates
        h = [0.1, 0.2, 0.5]
        j = [0.5, 0.5, 0.5]
        f = [0.9, 0.8, 0.5]
        result = calc_flexion_angle(h, j, f)
        assert 0.0 <= result <= 180.0

    def test_accepts_numpy_arrays(self):
        h = np.array([0.0, 0.0, 0.0])
        j = np.array([0.5, 0.5, 0.5])
        f = np.array([1.0, 0.0, 0.0])
        result = calc_flexion_angle(h, j, f)
        assert isinstance(result, float)

    def test_accepts_lists(self):
        result = calc_flexion_angle([0, 0, 0], [1, 0, 0], [2, 0, 0])
        assert isinstance(result, float)

    def test_joint_center_at_origin(self):
        # Joint at origin; v1=[-1,0,0], v2=[1,0,0] -> antiparallel -> 180°
        result = calc_flexion_angle([-1, 0, 0], [0, 0, 0], [1, 0, 0])
        assert abs(result - 180.0) < 1e-3

    def test_zero_length_forearm_does_not_crash(self):
        # forearm_shaft == joint_center -> zero vector with epsilon guard
        result = calc_flexion_angle([0, 0, 0], [1, 0, 0], [1, 0, 0])
        # With epsilon guard, cos is clipped to [-1,1] so result is in [0,180]
        assert 0.0 <= result <= 180.0

    def test_zero_length_humerus_does_not_crash(self):
        # humerus_shaft == joint_center -> zero vector with epsilon guard
        result = calc_flexion_angle([1, 0, 0], [1, 0, 0], [2, 0, 0])
        assert 0.0 <= result <= 180.0

    def test_output_is_degrees_not_radians(self):
        # 90 degree angle: if result were in radians it would be pi/2 ≈ 1.57
        result = calc_flexion_angle([0, 0, 0], [1, 0, 0], [1, 1, 0])
        assert abs(result - 90.0) < 1e-3
        assert result > 10  # Not radians

    def test_different_vector_lengths_give_same_angle(self):
        # Scale should not affect angle
        result1 = calc_flexion_angle([0, 0, 0], [1, 0, 0], [1, 1, 0])
        result2 = calc_flexion_angle([0, 0, 0], [10, 0, 0], [10, 10, 0])
        assert abs(result1 - result2) < 1e-3

    def test_normalized_vs_large_vectors(self):
        result1 = calc_flexion_angle([0, 0, 0], [0.5, 0.5, 0.5], [1, 0, 0])
        result2 = calc_flexion_angle([0, 0, 0], [50, 50, 50], [100, 0, 0])
        assert abs(result1 - result2) < 1e-3

    def test_flexion_series_monotonically_consistent(self):
        # With correct geometry (v1=[-1,0,0], v2=(-cos(a), sin(a), 0)):
        # larger angle_deg -> larger returned angle -> values should increase
        angles = [30, 60, 90, 120, 150]
        joint = [1, 0, 0]
        humerus = [0, 0, 0]
        results = []
        for deg in angles:
            rad = np.radians(deg)
            forearm = [1 - np.cos(rad), np.sin(rad), 0]
            results.append(calc_flexion_angle(humerus, joint, forearm))
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1], (
                f"Expected increasing but got {results[i]:.2f} >= {results[i+1]:.2f}"
            )

    def test_float_coordinates(self):
        result = calc_flexion_angle(
            [0.123, 0.456, 0.789],
            [0.5, 0.5, 0.5],
            [0.876, 0.543, 0.210],
        )
        assert 0.0 <= result <= 180.0

    def test_normalized_landmark_coords_typical_range(self):
        # Typical ElbowVision normalized coords are in [0, 1]
        result = calc_flexion_angle(
            [0.3, 0.2, 0.5],  # humerus shaft (upper arm)
            [0.5, 0.5, 0.5],  # joint center
            [0.7, 0.6, 0.5],  # forearm shaft
        )
        assert 0.0 <= result <= 180.0


# ── bone_outline_2d ───────────────────────────────────────────────────────

class TestBoneOutline2D:
    """Tests for bone_outline_2d(volume, axis, bone_thresh=0.15)."""

    def test_returns_bool_array(self):
        vol = make_volume(fill=0.5)
        result = bone_outline_2d(vol, axis=1)
        assert result.dtype == bool

    def test_shape_axis0(self):
        vol = make_volume(shape=(6, 8, 10))
        result = bone_outline_2d(vol, axis=0)
        assert result.shape == (8, 10)

    def test_shape_axis1(self):
        vol = make_volume(shape=(6, 8, 10))
        result = bone_outline_2d(vol, axis=1)
        assert result.shape == (6, 10)

    def test_shape_axis2(self):
        vol = make_volume(shape=(6, 8, 10))
        result = bone_outline_2d(vol, axis=2)
        assert result.shape == (6, 8)

    def test_all_zeros_returns_all_false(self):
        vol = make_volume(fill=0.0)
        result = bone_outline_2d(vol, axis=1)
        assert not result.any()

    def test_all_above_thresh_returns_all_true(self):
        vol = make_volume(fill=1.0)
        result = bone_outline_2d(vol, axis=1, bone_thresh=0.15)
        assert result.all()

    def test_default_threshold_is_015(self):
        # Volume with exactly 0.15 should be False (strictly greater)
        vol = make_volume(fill=0.15)
        result = bone_outline_2d(vol, axis=1)
        assert not result.any()

    def test_just_above_threshold(self):
        vol = make_volume(fill=0.151)
        result = bone_outline_2d(vol, axis=1)
        assert result.all()

    def test_custom_threshold_higher(self):
        # All 0.5 with thresh=0.8 -> all False
        vol = make_volume(fill=0.5)
        result = bone_outline_2d(vol, axis=1, bone_thresh=0.8)
        assert not result.any()

    def test_custom_threshold_lower(self):
        # All 0.1 with thresh=0.05 -> all True
        vol = make_volume(fill=0.1)
        result = bone_outline_2d(vol, axis=1, bone_thresh=0.05)
        assert result.all()

    def test_partial_volume_some_true_some_false(self):
        vol = np.zeros((4, 4, 4), dtype=np.float32)
        # Place a high-value slice only at axis=0 index 0
        vol[0, :, :] = 0.5
        # axis=0 projection: max along first axis
        result = bone_outline_2d(vol, axis=0)
        assert result.all()  # max projection picks up the 0.5 slice
        # axis=1 projection: shape (4,4), should have column 0 affected
        result1 = bone_outline_2d(vol, axis=1)
        assert result1.shape == (4, 4)

    def test_max_projection_not_mean(self):
        # One voxel above threshold in a column -> whole column projected as True
        vol = np.zeros((4, 4, 4), dtype=np.float32)
        vol[0, 2, 2] = 1.0  # single voxel bright
        # axis=0 max projection: position (2,2) should be True
        result = bone_outline_2d(vol, axis=0)
        assert result[2, 2]
        # Position (0,0) should be False
        assert not result[0, 0]

    def test_bone_sphere_visible_from_all_axes(self):
        vol = make_volume_with_sphere(shape=(16, 16, 16), center=(8, 8, 8), radius=3, value=1.0)
        for axis in [0, 1, 2]:
            result = bone_outline_2d(vol, axis=axis)
            # The sphere center projection should be True
            assert result[8, 8]

    def test_result_is_2d(self):
        vol = make_volume(shape=(5, 6, 7))
        for axis in [0, 1, 2]:
            result = bone_outline_2d(vol, axis=axis)
            assert result.ndim == 2

    def test_zero_threshold_all_nonzero_true(self):
        vol = make_volume(fill=0.001)
        result = bone_outline_2d(vol, axis=0, bone_thresh=0.0)
        assert result.all()

    def test_axis0_matches_manual_max_projection(self):
        rng = np.random.default_rng(42)
        vol = rng.random((6, 8, 10)).astype(np.float32)
        expected = vol.max(axis=0) > 0.15
        result = bone_outline_2d(vol, axis=0)
        np.testing.assert_array_equal(result, expected)

    def test_axis1_matches_manual_max_projection(self):
        rng = np.random.default_rng(42)
        vol = rng.random((6, 8, 10)).astype(np.float32)
        expected = vol.max(axis=1) > 0.15
        result = bone_outline_2d(vol, axis=1)
        np.testing.assert_array_equal(result, expected)

    def test_axis2_matches_manual_max_projection(self):
        rng = np.random.default_rng(42)
        vol = rng.random((6, 8, 10)).astype(np.float32)
        expected = vol.max(axis=2) > 0.15
        result = bone_outline_2d(vol, axis=2)
        np.testing.assert_array_equal(result, expected)

    def test_threshold_boundary_not_included(self):
        # Exactly at threshold should be False (strict >)
        thresh = 0.3
        vol = np.full((4, 4, 4), thresh, dtype=np.float32)
        result = bone_outline_2d(vol, axis=1, bone_thresh=thresh)
        assert not result.any()

    def test_asymmetric_shape(self):
        vol = make_volume(shape=(3, 7, 11), fill=0.5)
        r0 = bone_outline_2d(vol, axis=0)
        r1 = bone_outline_2d(vol, axis=1)
        r2 = bone_outline_2d(vol, axis=2)
        assert r0.shape == (7, 11)
        assert r1.shape == (3, 11)
        assert r2.shape == (3, 7)

    def test_deterministic(self):
        vol = make_volume_with_sphere(value=0.8)
        result1 = bone_outline_2d(vol, axis=1)
        result2 = bone_outline_2d(vol, axis=1)
        np.testing.assert_array_equal(result1, result2)
