"""Tests for approachB_landmark_model.py pure-function utilities.

Because the script runs heavy module-level code (CT loading, DRR generation,
matplotlib 3D figure creation) with no ``if __name__ == '__main__':`` guard,
we mock all external I/O before importing, then test only the pure
mathematical / algorithmic functions:

  - to_mm
  - build_humerus_frame
  - transform_to_frame
  - hf_to_norm
  - predict_hf
  - Constants: FOREARM_NAMES, HUMERUS_NAMES, KP_ORDER, ALL_NAMES
"""
from __future__ import annotations

import os
import sys
import unittest.mock

import numpy as np
import pytest

# ── Mocked imports BEFORE loading the script ─────────────────────────────────

_H = 64  # fake volume side length
_fake_vol = np.zeros((_H, _H, _H), dtype=np.uint8)

# Fake normalised landmarks that produce well-conditioned humerus geometry:
#   Y-axis = humerus_shaft -> joint_center  along (PD axis only)
#   Z-axis = medial -> lateral epicondyle   along (ML axis only)
_fake_lm = {
    "humerus_shaft":      (0.10, 0.50, 0.50),
    "lateral_epicondyle": (0.50, 0.50, 0.30),
    "medial_epicondyle":  (0.50, 0.50, 0.70),
    "joint_center":       (0.50, 0.50, 0.50),
    "forearm_shaft":      (0.90, 0.50, 0.50),
    "radial_head":        (0.80, 0.50, 0.30),
    "olecranon":          (0.80, 0.50, 0.70),
}
_fake_drr = np.zeros((128, 128), dtype=np.uint8)

_mock_synth = unittest.mock.MagicMock()
_mock_synth.load_ct_volume.return_value = (_fake_vol.copy(), np.ones(3), "L", 1.0)
_mock_synth.auto_detect_landmarks.return_value = dict(_fake_lm)
_mock_synth.generate_drr.return_value = _fake_drr.copy()

sys.modules["elbow_synth"] = _mock_synth

_scripts = os.path.join(os.path.dirname(__file__), "..", "scripts")
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

_mock_figure = unittest.mock.MagicMock()
_mock_ax = unittest.mock.MagicMock()
_mock_figure.add_subplot.return_value = _mock_ax

with (
    unittest.mock.patch("builtins.print"),
    unittest.mock.patch("matplotlib.pyplot.figure", return_value=_mock_figure),
    unittest.mock.patch("matplotlib.pyplot.savefig"),
    unittest.mock.patch("matplotlib.pyplot.close"),
    unittest.mock.patch("matplotlib.pyplot.tight_layout"),
    unittest.mock.patch("matplotlib.pyplot.suptitle"),
    unittest.mock.patch("os.makedirs"),
):
    import approachB_landmark_model as ab  # noqa: N813

# Remove mock so other test files can get the real elbow_synth when imported.
del sys.modules["elbow_synth"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_lm_mm() -> dict:
    """Return a canonical set of mm landmarks with well-conditioned geometry.

    Layout (all values in mm):
      humerus_shaft  -> [  0,  0,  0]   (origin of shaft)
      joint_center   -> [ 50,  0,  0]   (along X = PD axis)
      lateral_epi    -> [ 50,  0, 10]   (offset in ML+)
      medial_epi     -> [ 50,  0,-10]   (offset in ML-)
    This gives Y-axis = [1,0,0] and Z-axis = [0,0,1].
    """
    return {
        "joint_center":        np.array([50.0,  0.0,  0.0]),
        "humerus_shaft":       np.array([ 0.0,  0.0,  0.0]),
        "lateral_epicondyle":  np.array([50.0,  0.0, 10.0]),
        "medial_epicondyle":   np.array([50.0,  0.0,-10.0]),
        "forearm_shaft":       np.array([70.0, 20.0,  0.0]),
        "radial_head":         np.array([65.0, 15.0, 10.0]),
        "olecranon":           np.array([65.0, 15.0,-10.0]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_forearm_names_count(self):
        assert len(ab.FOREARM_NAMES) == 3

    def test_humerus_names_count(self):
        assert len(ab.HUMERUS_NAMES) == 4

    def test_kp_order_count(self):
        assert len(ab.KP_ORDER) == 6

    def test_all_names_count(self):
        # ALL_NAMES = HUMERUS_NAMES + FOREARM_NAMES = 4 + 3 = 7
        assert len(ab.ALL_NAMES) == 7

    def test_forearm_names_content(self):
        assert set(ab.FOREARM_NAMES) == {"forearm_shaft", "radial_head", "olecranon"}

    def test_humerus_names_contains_joint_center(self):
        assert "joint_center" in ab.HUMERUS_NAMES

    def test_humerus_names_contains_epicondyles(self):
        assert "lateral_epicondyle" in ab.HUMERUS_NAMES
        assert "medial_epicondyle" in ab.HUMERUS_NAMES

    def test_kp_order_excludes_joint_center(self):
        # KP_ORDER is the 6-keypoint pose list (no joint_center)
        assert "joint_center" not in ab.KP_ORDER

    def test_all_names_is_union_of_humerus_and_forearm(self):
        assert set(ab.ALL_NAMES) == set(ab.HUMERUS_NAMES) | set(ab.FOREARM_NAMES)

    def test_forearm_and_humerus_disjoint(self):
        assert set(ab.FOREARM_NAMES).isdisjoint(set(ab.HUMERUS_NAMES))


# ─────────────────────────────────────────────────────────────────────────────
# to_mm
# ─────────────────────────────────────────────────────────────────────────────

class TestToMm:
    def test_unit_voxel_unit_shape(self):
        """voxel_mm=1, shape=(1,1,1): output equals input coords."""
        lm = {"pt": (0.5, 0.3, 0.7)}
        result = ab.to_mm(lm, voxel_mm=1.0, vol_shape=(1, 1, 1))
        np.testing.assert_allclose(result["pt"], [0.5, 0.3, 0.7])

    def test_scale_by_voxel_mm(self):
        """voxel_mm=2, shape=(10,10,10): coord = norm * 10 * 2."""
        lm = {"pt": (0.5, 0.5, 0.5)}
        result = ab.to_mm(lm, voxel_mm=2.0, vol_shape=(10, 10, 10))
        np.testing.assert_allclose(result["pt"], [10.0, 10.0, 10.0])

    def test_anisotropic_shape(self):
        """Different shape values per axis are applied independently."""
        lm = {"pt": (1.0, 1.0, 1.0)}
        result = ab.to_mm(lm, voxel_mm=1.0, vol_shape=(10, 20, 30))
        np.testing.assert_allclose(result["pt"], [10.0, 20.0, 30.0])

    def test_origin_point_stays_zero(self):
        """Point at (0,0,0) normalised maps to (0,0,0) mm."""
        lm = {"pt": (0.0, 0.0, 0.0)}
        result = ab.to_mm(lm, voxel_mm=2.0, vol_shape=(64, 64, 64))
        np.testing.assert_allclose(result["pt"], [0.0, 0.0, 0.0])

    def test_multiple_landmarks_all_converted(self):
        lm = {
            "a": (0.0, 0.0, 0.0),
            "b": (1.0, 1.0, 1.0),
            "c": (0.5, 0.5, 0.5),
        }
        result = ab.to_mm(lm, voxel_mm=1.0, vol_shape=(100, 100, 100))
        np.testing.assert_allclose(result["a"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(result["b"], [100.0, 100.0, 100.0])
        np.testing.assert_allclose(result["c"], [50.0, 50.0, 50.0])

    def test_preserves_keys(self):
        lm = {"humerus_shaft": (0.1, 0.5, 0.5), "forearm_shaft": (0.9, 0.5, 0.5)}
        result = ab.to_mm(lm, voxel_mm=1.0, vol_shape=(64, 64, 64))
        assert set(result.keys()) == {"humerus_shaft", "forearm_shaft"}

    def test_output_values_are_ndarray(self):
        lm = {"pt": (0.5, 0.5, 0.5)}
        result = ab.to_mm(lm, voxel_mm=1.0, vol_shape=(64, 64, 64))
        assert isinstance(result["pt"], np.ndarray)

    def test_anisotropic_voxel_scaling(self):
        """Verify formula: mm = norm * shape * voxel_mm independently per axis."""
        lm = {"pt": (0.5, 0.5, 0.5)}
        result = ab.to_mm(lm, voxel_mm=0.5, vol_shape=(200, 100, 50))
        # pd: 0.5*200*0.5=50, ap: 0.5*100*0.5=25, ml: 0.5*50*0.5=12.5
        np.testing.assert_allclose(result["pt"], [50.0, 25.0, 12.5])

    def test_linearity(self):
        """Result is linear: midpoint of two inputs maps to midpoint of outputs."""
        lm_a = {"pt": (0.25, 0.25, 0.25)}
        lm_b = {"pt": (0.75, 0.75, 0.75)}
        lm_m = {"pt": (0.50, 0.50, 0.50)}
        r_a = ab.to_mm(lm_a, 1.0, (100, 100, 100))["pt"]
        r_b = ab.to_mm(lm_b, 1.0, (100, 100, 100))["pt"]
        r_m = ab.to_mm(lm_m, 1.0, (100, 100, 100))["pt"]
        np.testing.assert_allclose((r_a + r_b) / 2, r_m)

    def test_output_shape_is_3(self):
        lm = {"pt": (0.1, 0.2, 0.3)}
        result = ab.to_mm(lm, voxel_mm=1.0, vol_shape=(64, 64, 64))
        assert result["pt"].shape == (3,)


# ─────────────────────────────────────────────────────────────────────────────
# build_humerus_frame
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildHumerusFrame:
    def test_origin_is_joint_center(self):
        lm = _make_lm_mm()
        origin, _ = ab.build_humerus_frame(lm)
        np.testing.assert_allclose(origin, lm["joint_center"])

    def test_origin_shape(self):
        lm = _make_lm_mm()
        origin, _ = ab.build_humerus_frame(lm)
        assert origin.shape == (3,)

    def test_rotation_matrix_shape(self):
        lm = _make_lm_mm()
        _, R = ab.build_humerus_frame(lm)
        assert R.shape == (3, 3)

    def test_rotation_matrix_orthogonal(self):
        """R @ R.T should equal the identity matrix."""
        lm = _make_lm_mm()
        _, R = ab.build_humerus_frame(lm)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_rotation_matrix_proper(self):
        """det(R) should be +1 (proper rotation, not a reflection)."""
        lm = _make_lm_mm()
        _, R = ab.build_humerus_frame(lm)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)

    def test_axes_are_unit_vectors(self):
        lm = _make_lm_mm()
        _, R = ab.build_humerus_frame(lm)
        for i in range(3):
            assert np.linalg.norm(R[i]) == pytest.approx(1.0, abs=1e-10)

    def test_y_axis_along_humerus(self):
        """Row 1 of R (Y-axis) must be aligned with humerus_shaft -> joint_center."""
        lm = _make_lm_mm()
        _, R = ab.build_humerus_frame(lm)
        y_ax = R[1]
        expected = lm["joint_center"] - lm["humerus_shaft"]
        expected /= np.linalg.norm(expected)
        assert abs(np.dot(y_ax, expected)) == pytest.approx(1.0, abs=1e-10)

    def test_y_z_axes_perpendicular(self):
        lm = _make_lm_mm()
        _, R = ab.build_humerus_frame(lm)
        assert np.dot(R[1], R[2]) == pytest.approx(0.0, abs=1e-10)

    def test_x_axis_perpendicular_to_y_and_z(self):
        lm = _make_lm_mm()
        _, R = ab.build_humerus_frame(lm)
        assert np.dot(R[0], R[1]) == pytest.approx(0.0, abs=1e-10)
        assert np.dot(R[0], R[2]) == pytest.approx(0.0, abs=1e-10)

    def test_scaled_landmarks_same_R(self):
        """Uniformly scaling all landmarks should not change the rotation matrix."""
        lm = _make_lm_mm()
        lm_scaled = {k: v * 2.0 for k, v in lm.items()}
        _, R1 = ab.build_humerus_frame(lm)
        _, R2 = ab.build_humerus_frame(lm_scaled)
        np.testing.assert_allclose(R1, R2, atol=1e-10)

    def test_real_fake_landmarks(self):
        """Module-level fake landmarks produce a valid rotation matrix."""
        lm = ab.to_mm(_fake_lm, voxel_mm=1.0, vol_shape=(_H, _H, _H))
        origin, R = ab.build_humerus_frame(lm)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# transform_to_frame
# ─────────────────────────────────────────────────────────────────────────────

class TestTransformToFrame:
    def test_landmark_at_origin_maps_to_zero(self):
        origin = np.array([10.0, 20.0, 30.0])
        R = np.eye(3)
        lm = {"pt": origin.copy()}
        result = ab.transform_to_frame(lm, origin, R)
        np.testing.assert_allclose(result["pt"], [0.0, 0.0, 0.0], atol=1e-10)

    def test_identity_rotation_subtracts_origin(self):
        origin = np.array([1.0, 2.0, 3.0])
        R = np.eye(3)
        lm = {"pt": np.array([4.0, 6.0, 8.0])}
        result = ab.transform_to_frame(lm, origin, R)
        np.testing.assert_allclose(result["pt"], [3.0, 4.0, 5.0])

    def test_rotation_applied(self):
        """90-deg rotation around Z swaps X -> Y direction."""
        origin = np.zeros(3)
        # R @ [1,0,0] should give [0,-1,0] if R maps x->y, y->-x
        R = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        lm = {"pt": np.array([1.0, 0.0, 0.0])}
        result = ab.transform_to_frame(lm, origin, R)
        np.testing.assert_allclose(result["pt"], [0.0, -1.0, 0.0], atol=1e-10)

    def test_multiple_landmarks_all_transformed(self):
        origin = np.zeros(3)
        R = np.eye(3)
        lm = {
            "a": np.array([1.0, 0.0, 0.0]),
            "b": np.array([0.0, 2.0, 0.0]),
        }
        result = ab.transform_to_frame(lm, origin, R)
        assert set(result.keys()) == {"a", "b"}
        np.testing.assert_allclose(result["a"], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(result["b"], [0.0, 2.0, 0.0])

    def test_output_is_ndarray(self):
        lm = {"pt": np.array([1.0, 2.0, 3.0])}
        result = ab.transform_to_frame(lm, np.zeros(3), np.eye(3))
        assert isinstance(result["pt"], np.ndarray)

    def test_distance_preserved_with_pure_rotation(self):
        """Rotation (no translation) preserves distance from the origin."""
        origin = np.zeros(3)
        angle = np.pi / 4
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0.],
            [np.sin(angle),  np.cos(angle), 0.],
            [0.,             0.,            1.],
        ])
        pt = np.array([3.0, 4.0, 0.0])
        lm = {"pt": pt}
        result = ab.transform_to_frame(lm, origin, R)
        np.testing.assert_allclose(
            np.linalg.norm(result["pt"]), np.linalg.norm(pt), atol=1e-10
        )

    def test_inverse_via_transpose(self):
        """Applying R.T to the output (+ origin) recovers the original position."""
        origin = np.array([1.0, 2.0, 3.0])
        angle = np.pi / 3
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0.],
            [np.sin(angle),  np.cos(angle), 0.],
            [0.,             0.,            1.],
        ])
        lm = {"pt": np.array([5.0, 6.0, 7.0])}
        transformed = ab.transform_to_frame(lm, origin, R)
        recovered = R.T @ transformed["pt"] + origin
        np.testing.assert_allclose(recovered, lm["pt"], atol=1e-10)

    def test_empty_dict(self):
        result = ab.transform_to_frame({}, np.zeros(3), np.eye(3))
        assert result == {}


# ─────────────────────────────────────────────────────────────────────────────
# hf_to_norm
# ─────────────────────────────────────────────────────────────────────────────

class TestHfToNorm:
    def test_origin_maps_to_shape_half(self):
        """hf_pos=(0,0,0) -> pos_mm = origin -> norm = origin/(shape*voxel_mm)."""
        origin = np.array([50.0, 50.0, 50.0])
        R = np.eye(3)
        hf_zero = np.zeros(3)
        n_pd, n_ap, n_ml = ab.hf_to_norm(hf_zero, origin, R, 1.0, (100, 100, 100))
        assert n_pd == pytest.approx(0.5)
        assert n_ap == pytest.approx(0.5)
        assert n_ml == pytest.approx(0.5)

    def test_returns_tuple_of_three(self):
        result = ab.hf_to_norm(np.zeros(3), np.zeros(3), np.eye(3), 1.0, (100, 100, 100))
        assert len(result) == 3

    def test_identity_frame_divides_by_shape_times_voxel(self):
        """origin=0, R=I: result = hf_pos / (shape * voxel_mm)."""
        hf_pos = np.array([25.0, 50.0, 75.0])
        n_pd, n_ap, n_ml = ab.hf_to_norm(
            hf_pos, np.zeros(3), np.eye(3), 1.0, (100, 100, 100)
        )
        assert n_pd == pytest.approx(0.25)
        assert n_ap == pytest.approx(0.50)
        assert n_ml == pytest.approx(0.75)

    def test_voxel_mm_scales_result(self):
        """Doubling voxel_mm halves the normalised coords."""
        hf_pos = np.array([100.0, 100.0, 100.0])
        origin, R = np.zeros(3), np.eye(3)
        n1 = ab.hf_to_norm(hf_pos, origin, R, 1.0, (100, 100, 100))
        n2 = ab.hf_to_norm(hf_pos, origin, R, 2.0, (100, 100, 100))
        assert n1[0] == pytest.approx(1.0)
        assert n2[0] == pytest.approx(0.5)

    def test_anisotropic_shape(self):
        hf_pos = np.array([100.0, 100.0, 100.0])
        n_pd, n_ap, n_ml = ab.hf_to_norm(
            hf_pos, np.zeros(3), np.eye(3), 1.0, (200, 100, 50)
        )
        assert n_pd == pytest.approx(0.5)    # 100/(200*1)
        assert n_ap == pytest.approx(1.0)    # 100/(100*1)
        assert n_ml == pytest.approx(2.0)    # 100/(50*1)

    def test_round_trip_forearm_shaft(self):
        """to_mm -> to_frame -> hf_to_norm recovers the original norm coord."""
        lm_norm = {
            "joint_center":       (0.5, 0.5, 0.5),
            "humerus_shaft":      (0.1, 0.5, 0.5),
            "lateral_epicondyle": (0.5, 0.5, 0.3),
            "medial_epicondyle":  (0.5, 0.5, 0.7),
            "forearm_shaft":      (0.9, 0.5, 0.5),
        }
        voxel_mm = 1.0
        vol_shape = (100, 100, 100)

        lm_mm = ab.to_mm(lm_norm, voxel_mm, vol_shape)
        origin, R = ab.build_humerus_frame(lm_mm)
        lm_hf = ab.transform_to_frame(lm_mm, origin, R)

        hf_pos = lm_hf["forearm_shaft"]
        n_pd, n_ap, n_ml = ab.hf_to_norm(hf_pos, origin, R, voxel_mm, vol_shape)

        assert n_pd == pytest.approx(lm_norm["forearm_shaft"][0], abs=1e-10)
        assert n_ap == pytest.approx(lm_norm["forearm_shaft"][1], abs=1e-10)
        assert n_ml == pytest.approx(lm_norm["forearm_shaft"][2], abs=1e-10)

    def test_round_trip_humerus_shaft(self):
        """Round trip for humerus_shaft with non-unit voxel_mm."""
        lm_norm = {
            "joint_center":       (0.5, 0.5, 0.5),
            "humerus_shaft":      (0.1, 0.5, 0.5),
            "lateral_epicondyle": (0.5, 0.5, 0.3),
            "medial_epicondyle":  (0.5, 0.5, 0.7),
        }
        voxel_mm = 2.0
        vol_shape = (128, 128, 128)

        lm_mm = ab.to_mm(lm_norm, voxel_mm, vol_shape)
        origin, R = ab.build_humerus_frame(lm_mm)
        lm_hf = ab.transform_to_frame(lm_mm, origin, R)

        n_pd, n_ap, n_ml = ab.hf_to_norm(
            lm_hf["humerus_shaft"], origin, R, voxel_mm, vol_shape
        )
        assert n_pd == pytest.approx(lm_norm["humerus_shaft"][0], abs=1e-10)
        assert n_ap == pytest.approx(lm_norm["humerus_shaft"][1], abs=1e-10)
        assert n_ml == pytest.approx(lm_norm["humerus_shaft"][2], abs=1e-10)

    def test_rotation_changes_result(self):
        """Different rotation matrices produce different normalised coords."""
        hf_pos = np.array([10.0, 0.0, 0.0])
        origin = np.zeros(3)
        R1 = np.eye(3)
        angle = np.pi / 2
        R2 = np.array([
            [np.cos(angle), -np.sin(angle), 0.],
            [np.sin(angle),  np.cos(angle), 0.],
            [0.,             0.,            1.],
        ])
        n1 = ab.hf_to_norm(hf_pos, origin, R1, 1.0, (100, 100, 100))
        n2 = ab.hf_to_norm(hf_pos, origin, R2, 1.0, (100, 100, 100))
        assert n1 != n2


# ─────────────────────────────────────────────────────────────────────────────
# predict_hf
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictHf:
    def test_returns_ndarray_shape_3(self):
        for name in ab.ALL_NAMES:
            result = ab.predict_hf(name, 120.0)
            assert isinstance(result, np.ndarray)
            assert result.shape == (3,)

    def test_all_names_accepted(self):
        for name in ab.ALL_NAMES:
            result = ab.predict_hf(name, 135.0)
            assert result.shape == (3,)

    def test_all_values_finite(self):
        for name in ab.ALL_NAMES:
            result = ab.predict_hf(name, 120.0)
            assert np.all(np.isfinite(result)), f"{name} produced non-finite values"

    def test_various_angles_accepted(self):
        for angle in [60.0, 90.0, 120.0, 135.0, 150.0, 180.0]:
            result = ab.predict_hf("olecranon", angle)
            assert result.shape == (3,)
            assert np.all(np.isfinite(result))

    def test_forearm_names_use_arc_params(self):
        """Forearm landmarks should be in arc_params (not averaged like humerus)."""
        for name in ab.FOREARM_NAMES:
            assert name in ab.arc_params, f"{name} should be in arc_params"
