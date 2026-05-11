"""Unit tests for elbow-train/phantom_test.py pure functions.

Tests cover:
  - make_cylinder: cylindrical mask generation (shape, axis, radius, length)
  - make_ellipsoid: ellipsoidal mask generation (shape, semi-axes)
  - create_elbow_phantom: volume generation with ground truth dict
  - angle_between: vector angle computation (0-90 deg)
  - grade: threshold classification (OK/WARN/FAIL)
  - run_one / main: integration with mocked ct_reorient
"""

import sys
import os
import types
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

# ---------------------------------------------------------------------------
# Module-level mock: phantom_test imports ct_reorient at import time
# ---------------------------------------------------------------------------
_ct_reorient_mock = types.ModuleType("ct_reorient")
_ct_reorient_mock.detect_humeral_axis = MagicMock(
    return_value=(np.array([1.0, 0.0, 0.0]), np.array([100.0, 40.0, 40.0]))
)
_ct_reorient_mock.detect_transepicondylar_axis = MagicMock(
    return_value=np.array([0.0, 0.0, 1.0])
)

_saved = {}
for mod_name in ("ct_reorient",):
    _saved[mod_name] = sys.modules.get(mod_name)
    sys.modules[mod_name] = _ct_reorient_mock

# Ensure elbow-train is on path so phantom_test can be imported
_elbow_train_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "elbow-train"
)
if _elbow_train_dir not in sys.path:
    sys.path.insert(0, _elbow_train_dir)

from phantom_test import (
    make_cylinder,
    make_ellipsoid,
    create_elbow_phantom,
    angle_between,
    grade,
    run_one,
    main,
)

# Restore original modules
for mod_name, orig in _saved.items():
    if orig is None:
        sys.modules.pop(mod_name, None)
    else:
        sys.modules[mod_name] = orig


# ===========================================================================
# make_cylinder
# ===========================================================================
class TestMakeCylinder(unittest.TestCase):
    """Tests for make_cylinder mask generation."""

    def test_output_shape(self):
        shape = (30, 20, 20)
        mask = make_cylinder(shape, (22.5, 5.0, 5.0), [1, 0, 0], 3.0, 10.0, (1.0, 0.5, 0.5))
        self.assertEqual(mask.shape, shape)

    def test_output_dtype(self):
        mask = make_cylinder((20, 20, 20), (10, 5, 5), [1, 0, 0], 3.0, 8.0, (1.0, 0.5, 0.5))
        self.assertEqual(mask.dtype, np.bool_)

    def test_nonempty_mask(self):
        mask = make_cylinder((40, 40, 40), (20, 10, 10), [1, 0, 0], 5.0, 15.0, (1.0, 0.5, 0.5))
        self.assertGreater(mask.sum(), 0)

    def test_zero_radius_empty(self):
        mask = make_cylinder((20, 20, 20), (10, 5, 5), [1, 0, 0], 0.0, 10.0, (1.0, 1.0, 1.0))
        self.assertEqual(mask.sum(), 0)

    def test_zero_length_empty(self):
        mask = make_cylinder((20, 20, 20), (10, 5, 5), [1, 0, 0], 5.0, 0.0, (1.0, 1.0, 1.0))
        self.assertEqual(mask.sum(), 0)

    def test_z_axis_cylinder(self):
        """Cylinder along Z axis should have voxels spread along Z."""
        mask = make_cylinder((60, 20, 20), (30, 10, 10), [1, 0, 0], 3.0, 20.0, (1.0, 1.0, 1.0))
        z_coords = np.where(mask)[0]
        self.assertGreater(z_coords.max() - z_coords.min(), 10)

    def test_x_axis_cylinder(self):
        """Cylinder along X axis should have voxels spread along X."""
        mask = make_cylinder((20, 20, 60), (10, 10, 30), [0, 0, 1], 3.0, 20.0, (1.0, 1.0, 1.0))
        x_coords = np.where(mask)[2]
        self.assertGreater(x_coords.max() - x_coords.min(), 10)

    def test_larger_radius_more_voxels(self):
        center = (15, 10, 10)
        shape = (30, 20, 20)
        voxel = (1.0, 1.0, 1.0)
        m_small = make_cylinder(shape, center, [1, 0, 0], 2.0, 10.0, voxel)
        m_large = make_cylinder(shape, center, [1, 0, 0], 5.0, 10.0, voxel)
        self.assertGreater(m_large.sum(), m_small.sum())

    def test_longer_cylinder_more_voxels(self):
        center = (25, 10, 10)
        shape = (50, 20, 20)
        voxel = (1.0, 1.0, 1.0)
        m_short = make_cylinder(shape, center, [1, 0, 0], 3.0, 5.0, voxel)
        m_long = make_cylinder(shape, center, [1, 0, 0], 3.0, 15.0, voxel)
        self.assertGreater(m_long.sum(), m_short.sum())

    def test_unnormalized_axis_works(self):
        """Axis doesn't need to be unit vector."""
        mask = make_cylinder((30, 20, 20), (15, 10, 10), [10, 0, 0], 3.0, 10.0, (1.0, 1.0, 1.0))
        self.assertGreater(mask.sum(), 0)

    def test_diagonal_axis(self):
        mask = make_cylinder((40, 40, 40), (20, 20, 20), [1, 1, 0], 3.0, 12.0, (1.0, 1.0, 1.0))
        self.assertGreater(mask.sum(), 0)

    def test_anisotropic_voxel(self):
        """Anisotropic voxels should still produce a valid mask."""
        mask = make_cylinder((30, 60, 60), (22.5, 15.0, 15.0), [1, 0, 0], 5.0, 10.0, (1.5, 0.5, 0.5))
        self.assertGreater(mask.sum(), 0)


# ===========================================================================
# make_ellipsoid
# ===========================================================================
class TestMakeEllipsoid(unittest.TestCase):
    """Tests for make_ellipsoid mask generation."""

    def test_output_shape(self):
        shape = (30, 30, 30)
        mask = make_ellipsoid(shape, (15, 15, 15), (5, 5, 5), (1.0, 1.0, 1.0))
        self.assertEqual(mask.shape, shape)

    def test_output_dtype(self):
        mask = make_ellipsoid((20, 20, 20), (10, 10, 10), (4, 4, 4), (1.0, 1.0, 1.0))
        self.assertEqual(mask.dtype, np.bool_)

    def test_nonempty(self):
        mask = make_ellipsoid((30, 30, 30), (15, 15, 15), (5, 5, 5), (1.0, 1.0, 1.0))
        self.assertGreater(mask.sum(), 0)

    def test_sphere_symmetry(self):
        """Sphere (equal semi-axes) should be roughly symmetric."""
        shape = (40, 40, 40)
        mask = make_ellipsoid(shape, (20, 20, 20), (8, 8, 8), (1.0, 1.0, 1.0))
        z_range = np.ptp(np.where(mask)[0])
        y_range = np.ptp(np.where(mask)[1])
        x_range = np.ptp(np.where(mask)[2])
        self.assertAlmostEqual(z_range, y_range, delta=2)
        self.assertAlmostEqual(y_range, x_range, delta=2)

    def test_elongated_z(self):
        """Larger Z semi-axis should produce more spread in Z."""
        shape = (60, 30, 30)
        mask = make_ellipsoid(shape, (30, 15, 15), (20, 5, 5), (1.0, 1.0, 1.0))
        z_range = np.ptp(np.where(mask)[0])
        x_range = np.ptp(np.where(mask)[2])
        self.assertGreater(z_range, x_range)

    def test_larger_semi_axes_more_voxels(self):
        shape = (40, 40, 40)
        center = (20, 20, 20)
        voxel = (1.0, 1.0, 1.0)
        m_small = make_ellipsoid(shape, center, (3, 3, 3), voxel)
        m_large = make_ellipsoid(shape, center, (8, 8, 8), voxel)
        self.assertGreater(m_large.sum(), m_small.sum())

    def test_anisotropic_voxel(self):
        mask = make_ellipsoid((30, 60, 60), (22.5, 15.0, 15.0), (10, 8, 8), (1.5, 0.5, 0.5))
        self.assertGreater(mask.sum(), 0)

    def test_center_is_inside(self):
        """The center voxel should be inside the ellipsoid."""
        shape = (30, 30, 30)
        voxel = (1.0, 1.0, 1.0)
        mask = make_ellipsoid(shape, (15, 15, 15), (5, 5, 5), voxel)
        self.assertTrue(mask[15, 15, 15])


# ===========================================================================
# create_elbow_phantom
# ===========================================================================
class TestCreateElbowPhantom(unittest.TestCase):
    """Tests for create_elbow_phantom volume generation."""

    def test_returns_three_items(self):
        result = create_elbow_phantom()
        self.assertEqual(len(result), 3)

    def test_volume_shape(self):
        volume, gt, voxel_mm = create_elbow_phantom()
        self.assertEqual(volume.shape, (220, 160, 160))

    def test_volume_dtype(self):
        volume, gt, voxel_mm = create_elbow_phantom()
        self.assertEqual(volume.dtype, np.float32)

    def test_voxel_mm(self):
        _, _, voxel_mm = create_elbow_phantom()
        np.testing.assert_array_equal(voxel_mm, (1.5, 0.5, 0.5))

    def test_gt_keys(self):
        _, gt, _ = create_elbow_phantom()
        expected = {"humeral_axis", "transepicondylar_axis", "condyle_center_mm",
                    "epic_lat_mm", "epic_med_mm", "flex_deg"}
        self.assertEqual(set(gt.keys()), expected)

    def test_humeral_axis_direction(self):
        _, gt, _ = create_elbow_phantom()
        np.testing.assert_array_equal(gt["humeral_axis"], [1, 0, 0])

    def test_transepicondylar_axis_direction(self):
        _, gt, _ = create_elbow_phantom()
        np.testing.assert_array_equal(gt["transepicondylar_axis"], [0, 0, 1])

    def test_default_flex_deg(self):
        _, gt, _ = create_elbow_phantom()
        self.assertEqual(gt["flex_deg"], 30.0)

    def test_custom_flex_deg(self):
        _, gt, _ = create_elbow_phantom(flex_deg=60.0)
        self.assertEqual(gt["flex_deg"], 60.0)

    def test_zero_flex(self):
        _, gt, _ = create_elbow_phantom(flex_deg=0.0)
        self.assertEqual(gt["flex_deg"], 0.0)

    def test_background_is_minus1000(self):
        volume, _, _ = create_elbow_phantom()
        self.assertAlmostEqual(volume.min(), -1000.0)

    def test_bone_hu_values(self):
        volume, _, _ = create_elbow_phantom()
        bone_mask = volume > 0
        unique_vals = set(np.unique(volume[bone_mask]).tolist())
        # shaft=600, condyle=500, radius=550, ulna=520
        self.assertTrue(unique_vals.issubset({500.0, 520.0, 550.0, 600.0}))

    def test_has_bone_voxels(self):
        volume, _, _ = create_elbow_phantom()
        self.assertGreater((volume > 0).sum(), 0)

    def test_epicondyle_separation(self):
        """Lateral and medial epicondyles should be separated along X axis."""
        _, gt, _ = create_elbow_phantom()
        lat = gt["epic_lat_mm"]
        med = gt["epic_med_mm"]
        # X separation should be large (48mm: +24 and -24)
        self.assertAlmostEqual(abs(lat[2] - med[2]), 48.0, places=1)

    def test_epicondyles_symmetric_in_zy(self):
        """Epicondyles should have same Z and Y coordinates."""
        _, gt, _ = create_elbow_phantom()
        np.testing.assert_array_almost_equal(gt["epic_lat_mm"][:2], gt["epic_med_mm"][:2])

    def test_condyle_center_between_epicondyles(self):
        _, gt, _ = create_elbow_phantom()
        mid = (gt["epic_lat_mm"] + gt["epic_med_mm"]) / 2
        np.testing.assert_array_almost_equal(gt["condyle_center_mm"], mid)

    def test_different_flex_different_volume(self):
        """Different flexion should produce different volumes."""
        v0, _, _ = create_elbow_phantom(flex_deg=0)
        v90, _, _ = create_elbow_phantom(flex_deg=90)
        self.assertFalse(np.array_equal(v0, v90))

    def test_bone_occupancy_reasonable(self):
        """Bone should occupy a small fraction of the total volume."""
        volume, _, _ = create_elbow_phantom()
        ratio = (volume > 0).sum() / volume.size
        self.assertGreater(ratio, 0.01)
        self.assertLess(ratio, 0.5)


# ===========================================================================
# angle_between
# ===========================================================================
class TestAngleBetween(unittest.TestCase):
    """Tests for angle_between vector angle computation."""

    def test_parallel_same_direction(self):
        self.assertAlmostEqual(angle_between([1, 0, 0], [1, 0, 0]), 0.0, places=5)

    def test_parallel_opposite_direction(self):
        """Opposite direction should also give 0 (normalized to 0-90)."""
        self.assertAlmostEqual(angle_between([1, 0, 0], [-1, 0, 0]), 0.0, places=5)

    def test_perpendicular(self):
        self.assertAlmostEqual(angle_between([1, 0, 0], [0, 1, 0]), 90.0, places=5)

    def test_45_degrees(self):
        self.assertAlmostEqual(angle_between([1, 0, 0], [1, 1, 0]), 45.0, places=3)

    def test_3d_diagonal(self):
        """[1,1,1] vs [1,0,0] = arccos(1/sqrt(3)) ~ 54.74 degrees."""
        expected = np.degrees(np.arccos(1 / np.sqrt(3)))
        self.assertAlmostEqual(angle_between([1, 1, 1], [1, 0, 0]), expected, places=2)

    def test_symmetry(self):
        a, b = [1, 2, 3], [4, 5, 6]
        self.assertAlmostEqual(angle_between(a, b), angle_between(b, a), places=10)

    def test_returns_float(self):
        result = angle_between([1, 0, 0], [0, 1, 0])
        self.assertIsInstance(result, float)

    def test_range_0_to_90(self):
        """Output should always be in [0, 90]."""
        np.random.seed(42)
        for _ in range(20):
            a = np.random.randn(3)
            b = np.random.randn(3)
            ang = angle_between(a, b)
            self.assertGreaterEqual(ang, 0.0)
            self.assertLessEqual(ang, 90.0 + 1e-10)

    def test_scaled_vector_same_result(self):
        """Scaling a vector should not change the angle."""
        a, b = [1, 0, 0], [1, 1, 0]
        self.assertAlmostEqual(angle_between(a, b), angle_between([100, 0, 0], b), places=10)

    def test_negative_components(self):
        """Negative vectors should map to 0-90 range."""
        ang = angle_between([-1, 0, 0], [0, -1, 0])
        self.assertAlmostEqual(ang, 90.0, places=5)


# ===========================================================================
# grade
# ===========================================================================
class TestGrade(unittest.TestCase):
    """Tests for grade threshold classification."""

    def test_zero_is_ok(self):
        self.assertEqual(grade(0.0), "OK")

    def test_small_angle_ok(self):
        self.assertEqual(grade(2.9), "OK")

    def test_boundary_3_is_warn(self):
        self.assertEqual(grade(3.0), "WARN")

    def test_mid_warn(self):
        self.assertEqual(grade(5.0), "WARN")

    def test_boundary_10_is_fail(self):
        self.assertEqual(grade(10.0), "FAIL")

    def test_large_angle_fail(self):
        self.assertEqual(grade(45.0), "FAIL")

    def test_just_below_3(self):
        self.assertEqual(grade(2.999), "OK")

    def test_just_below_10(self):
        self.assertEqual(grade(9.999), "WARN")


# ===========================================================================
# run_one (integration with mocked ct_reorient)
# ===========================================================================
class TestRunOne(unittest.TestCase):
    """Tests for run_one with mocked ct_reorient."""

    def _make_mock_detect(self, humeral_axis=None, trans_axis=None):
        if humeral_axis is None:
            humeral_axis = np.array([1.0, 0.0, 0.0])
        if trans_axis is None:
            trans_axis = np.array([0.0, 0.0, 1.0])

        mock_humeral = MagicMock(return_value=(humeral_axis, np.array([100.0, 40.0, 40.0])))
        mock_trans = MagicMock(return_value=trans_axis)
        return mock_humeral, mock_trans

    @patch("phantom_test.detect_humeral_axis")
    @patch("phantom_test.detect_transepicondylar_axis")
    def test_returns_dict(self, mock_trans, mock_humeral):
        mock_humeral.return_value = (np.array([1.0, 0.0, 0.0]), np.array([100.0, 40.0, 40.0]))
        mock_trans.return_value = np.array([0.0, 0.0, 1.0])
        result = run_one(30)
        self.assertIsInstance(result, dict)

    @patch("phantom_test.detect_humeral_axis")
    @patch("phantom_test.detect_transepicondylar_axis")
    def test_result_keys(self, mock_trans, mock_humeral):
        mock_humeral.return_value = (np.array([1.0, 0.0, 0.0]), np.array([100.0, 40.0, 40.0]))
        mock_trans.return_value = np.array([0.0, 0.0, 1.0])
        result = run_one(30)
        expected_keys = {"flex_deg", "humeral_err", "trans_err", "humeral_axis", "sign_ok"}
        self.assertEqual(set(result.keys()), expected_keys)

    @patch("phantom_test.detect_humeral_axis")
    @patch("phantom_test.detect_transepicondylar_axis")
    def test_flex_deg_preserved(self, mock_trans, mock_humeral):
        mock_humeral.return_value = (np.array([1.0, 0.0, 0.0]), np.array([100.0, 40.0, 40.0]))
        mock_trans.return_value = np.array([0.0, 0.0, 1.0])
        result = run_one(45)
        self.assertEqual(result["flex_deg"], 45)

    @patch("phantom_test.detect_humeral_axis")
    @patch("phantom_test.detect_transepicondylar_axis")
    def test_perfect_detection_zero_error(self, mock_trans, mock_humeral):
        mock_humeral.return_value = (np.array([1.0, 0.0, 0.0]), np.array([100.0, 40.0, 40.0]))
        mock_trans.return_value = np.array([0.0, 0.0, 1.0])
        result = run_one(30)
        self.assertAlmostEqual(result["humeral_err"], 0.0, places=3)
        self.assertAlmostEqual(result["trans_err"], 0.0, places=3)

    @patch("phantom_test.detect_humeral_axis")
    @patch("phantom_test.detect_transepicondylar_axis")
    def test_imperfect_detection_nonzero_error(self, mock_trans, mock_humeral):
        # 10 degrees off
        mock_humeral.return_value = (
            np.array([np.cos(np.radians(10)), np.sin(np.radians(10)), 0.0]),
            np.array([100.0, 40.0, 40.0]),
        )
        mock_trans.return_value = np.array([0.0, 0.0, 1.0])
        result = run_one(30)
        self.assertAlmostEqual(result["humeral_err"], 10.0, places=1)

    @patch("phantom_test.detect_humeral_axis")
    @patch("phantom_test.detect_transepicondylar_axis")
    def test_sign_ok_negative_z(self, mock_trans, mock_humeral):
        """Negative axis[0] should give sign_ok=True."""
        mock_humeral.return_value = (np.array([-1.0, 0.0, 0.0]), np.array([100.0, 40.0, 40.0]))
        mock_trans.return_value = np.array([0.0, 0.0, 1.0])
        result = run_one(30)
        self.assertTrue(result["sign_ok"])

    @patch("phantom_test.detect_humeral_axis")
    @patch("phantom_test.detect_transepicondylar_axis")
    def test_sign_ok_positive_z(self, mock_trans, mock_humeral):
        """Positive axis[0] should give sign_ok=False."""
        mock_humeral.return_value = (np.array([1.0, 0.0, 0.0]), np.array([100.0, 40.0, 40.0]))
        mock_trans.return_value = np.array([0.0, 0.0, 1.0])
        result = run_one(30)
        self.assertFalse(result["sign_ok"])

    @patch("phantom_test.detect_humeral_axis")
    @patch("phantom_test.detect_transepicondylar_axis")
    def test_calls_detect_humeral_axis(self, mock_trans, mock_humeral):
        mock_humeral.return_value = (np.array([1.0, 0.0, 0.0]), np.array([100.0, 40.0, 40.0]))
        mock_trans.return_value = np.array([0.0, 0.0, 1.0])
        run_one(0)
        mock_humeral.assert_called_once()

    @patch("phantom_test.detect_humeral_axis")
    @patch("phantom_test.detect_transepicondylar_axis")
    def test_calls_detect_transepicondylar_axis(self, mock_trans, mock_humeral):
        mock_humeral.return_value = (np.array([1.0, 0.0, 0.0]), np.array([100.0, 40.0, 40.0]))
        mock_trans.return_value = np.array([0.0, 0.0, 1.0])
        run_one(0)
        mock_trans.assert_called_once()


# ===========================================================================
# main (integration)
# ===========================================================================
class TestMain(unittest.TestCase):
    """Tests for main function with mocked ct_reorient."""

    @patch("phantom_test.detect_humeral_axis")
    @patch("phantom_test.detect_transepicondylar_axis")
    def test_main_runs_without_error(self, mock_trans, mock_humeral):
        mock_humeral.return_value = (np.array([-1.0, 0.0, 0.0]), np.array([100.0, 40.0, 40.0]))
        mock_trans.return_value = np.array([0.0, 0.0, 1.0])
        # main() prints output but should not raise
        main()

    @patch("phantom_test.run_one")
    def test_main_calls_run_one_five_times(self, mock_run_one):
        mock_run_one.return_value = {
            "flex_deg": 0, "humeral_err": 1.0, "trans_err": 1.0,
            "humeral_axis": np.array([1.0, 0.0, 0.0]), "sign_ok": True,
        }
        main()
        self.assertEqual(mock_run_one.call_count, 5)

    @patch("phantom_test.run_one")
    def test_main_flexion_angles(self, mock_run_one):
        """main should test flexion angles 0, 30, 60, 75, 90."""
        mock_run_one.return_value = {
            "flex_deg": 0, "humeral_err": 1.0, "trans_err": 1.0,
            "humeral_axis": np.array([1.0, 0.0, 0.0]), "sign_ok": True,
        }
        main()
        called_angles = [call.args[0] for call in mock_run_one.call_args_list]
        self.assertEqual(called_angles, [0, 30, 60, 75, 90])


if __name__ == "__main__":
    unittest.main()
