"""Tests for experiment_flexion_accuracy.py pure-function utilities.

Because the script runs heavy module-level code (CT loading, DRR generation,
matplotlib figure creation), we mock all external I/O before importing, then
test only the pure mathematical / algorithmic functions:

  - compute_dice
  - compute_ssim
  - landmark_error_mm
  - to_mm
  - procrustes_rotation
  - nearest_volume_for_angle
  - rotation_matrix_custom_axis
  - rotate_volume_data_driven
"""
import sys
import os
import unittest.mock
import numpy as np
import pytest

# ── Mocked imports BEFORE loading the script ─────────────────────────────────

_H = 64  # fake volume size
_fake_vol = np.zeros((_H, _H, _H), dtype=np.uint8)
_fake_drr = np.zeros((128, 128), dtype=np.uint8)
_fake_lm = {
    "humerus_shaft":      (0.10, 0.50, 0.50),
    "lateral_epicondyle": (0.50, 0.50, 0.30),
    "medial_epicondyle":  (0.50, 0.50, 0.70),
    "forearm_shaft":      (0.90, 0.50, 0.50),
    "radial_head":        (0.80, 0.50, 0.30),
    "olecranon":          (0.80, 0.50, 0.70),
}

_mock_synth = unittest.mock.MagicMock()
_mock_synth.load_ct_volume.return_value = (_fake_vol.copy(), np.ones(3), "L", 1.0)
_mock_synth.auto_detect_landmarks.return_value = dict(_fake_lm)
_mock_synth.generate_drr.return_value = _fake_drr.copy()
_mock_synth.rotate_volume_and_landmarks.return_value = (_fake_vol.copy(), dict(_fake_lm))
_mock_synth.rotation_matrix_x.return_value = np.eye(3)
_mock_synth.rotation_matrix_y.return_value = np.eye(3)
_mock_synth.rotation_matrix_z.return_value = np.eye(3)

sys.modules["elbow_synth"] = _mock_synth

_scripts = os.path.join(os.path.dirname(__file__), "..", "scripts")
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

with (
    unittest.mock.patch("builtins.print"),
    unittest.mock.patch("matplotlib.pyplot.savefig"),
    unittest.mock.patch("matplotlib.pyplot.close"),
    unittest.mock.patch("matplotlib.figure.Figure.savefig"),
):
    import experiment_flexion_accuracy as efa

# Remove mock so that other test files (flexion_2d_warp, make_yolo_label, etc.)
# get the real elbow_synth when they import it.
sys.modules.pop("elbow_synth", None)


# ── Helpers ───────────────────────────────────────────────────────────────────

KP_NAMES = efa.KP_NAMES  # ["humerus_shaft", "lateral_epicondyle", ...]


def make_binary_img(h=64, w=64, fill_top=True):
    """Return uint8 image: top half white if fill_top, else bottom half white."""
    img = np.zeros((h, w), dtype=np.uint8)
    if fill_top:
        img[: h // 2, :] = 255
    else:
        img[h // 2 :, :] = 255
    return img


def make_full_white(h=64, w=64):
    return np.full((h, w), 255, dtype=np.uint8)


def make_noise(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def make_lm(val=0.5):
    """All six keypoints at (val, val, val)."""
    return {n: (val, val, val) for n in KP_NAMES}


# ══════════════════════════════════════════════════════════════════════════════
# compute_dice
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeDice:
    def test_identical_images_dice_one(self):
        img = make_binary_img()
        assert efa.compute_dice(img, img) == pytest.approx(1.0, abs=1e-6)

    def test_disjoint_images_dice_zero(self):
        top = make_binary_img(fill_top=True)
        bottom = make_binary_img(fill_top=False)
        assert efa.compute_dice(top, bottom) == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap_between_zero_and_one(self):
        img1 = make_binary_img(fill_top=True)
        # shift by a quarter → 50 % overlap
        h = img1.shape[0]
        img2 = np.zeros_like(img1)
        img2[h // 4 : 3 * h // 4, :] = 255
        dice = efa.compute_dice(img1, img2)
        assert 0.0 < dice < 1.0

    def test_zero_both_images_returns_zero(self):
        z = np.zeros((64, 64), dtype=np.uint8)
        assert efa.compute_dice(z, z) == pytest.approx(0.0, abs=1e-4)

    def test_different_sizes_auto_resize(self):
        img32 = make_binary_img(32, 32, fill_top=True)
        img64 = make_binary_img(64, 64, fill_top=True)
        dice = efa.compute_dice(img32, img64)
        assert 0.0 <= dice <= 1.0

    def test_full_white_images_dice_one(self):
        img = make_full_white()
        assert efa.compute_dice(img, img) == pytest.approx(1.0, abs=1e-6)

    def test_return_type_float(self):
        img = make_binary_img()
        result = efa.compute_dice(img, img)
        assert isinstance(result, float)

    def test_symmetry(self):
        img1 = make_binary_img(fill_top=True)
        img2 = make_binary_img(fill_top=False)
        # Dice is symmetric
        assert efa.compute_dice(img1, img2) == pytest.approx(
            efa.compute_dice(img2, img1), abs=1e-6
        )


# ══════════════════════════════════════════════════════════════════════════════
# compute_ssim
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeSSIM:
    def test_identical_images_ssim_one(self):
        img = make_binary_img()
        assert efa.compute_ssim(img, img) == pytest.approx(1.0, abs=1e-6)

    def test_ssim_between_minus_one_and_one(self):
        img1 = make_noise(seed=0)
        img2 = make_noise(seed=1)
        val = efa.compute_ssim(img1, img2)
        assert -1.0 <= val <= 1.0

    def test_different_sizes_auto_resize(self):
        img32 = make_binary_img(32, 32)
        img64 = make_binary_img(64, 64)
        val = efa.compute_ssim(img32, img64)
        assert -1.0 <= val <= 1.0

    def test_inverted_image_lower_ssim(self):
        img = make_binary_img()
        inv = 255 - img
        ssim_same = efa.compute_ssim(img, img)
        ssim_diff = efa.compute_ssim(img, inv)
        assert ssim_same > ssim_diff

    def test_return_type_float(self):
        img = make_binary_img()
        assert isinstance(efa.compute_ssim(img, img), float)


# ══════════════════════════════════════════════════════════════════════════════
# to_mm
# ══════════════════════════════════════════════════════════════════════════════

class TestToMm:
    def test_origin_point_stays_origin(self):
        lm = {"humerus_shaft": (0.0, 0.0, 0.0)}
        result = efa.to_mm(lm, voxel_mm=2.0, vol_shape=(100, 100, 100))
        np.testing.assert_allclose(result["humerus_shaft"], [0.0, 0.0, 0.0])

    def test_unit_corner_correct_mm(self):
        lm = {"humerus_shaft": (1.0, 1.0, 1.0)}
        result = efa.to_mm(lm, voxel_mm=2.0, vol_shape=(10, 20, 30))
        np.testing.assert_allclose(result["humerus_shaft"], [20.0, 40.0, 60.0])

    def test_half_point(self):
        lm = {"humerus_shaft": (0.5, 0.5, 0.5)}
        result = efa.to_mm(lm, voxel_mm=1.0, vol_shape=(100, 200, 300))
        np.testing.assert_allclose(result["humerus_shaft"], [50.0, 100.0, 150.0])

    def test_multiple_landmarks_all_converted(self):
        lm = make_lm(0.5)
        result = efa.to_mm(lm, voxel_mm=1.0, vol_shape=(64, 64, 64))
        assert set(result.keys()) == set(lm.keys())

    def test_output_is_ndarray(self):
        lm = {"humerus_shaft": (0.2, 0.3, 0.4)}
        result = efa.to_mm(lm, voxel_mm=1.0, vol_shape=(64, 64, 64))
        assert isinstance(result["humerus_shaft"], np.ndarray)
        assert result["humerus_shaft"].shape == (3,)

    def test_voxel_mm_scales_linearly(self):
        lm = {"humerus_shaft": (0.5, 0.5, 0.5)}
        r1 = efa.to_mm(lm, voxel_mm=1.0, vol_shape=(100, 100, 100))
        r2 = efa.to_mm(lm, voxel_mm=2.0, vol_shape=(100, 100, 100))
        np.testing.assert_allclose(r2["humerus_shaft"], r1["humerus_shaft"] * 2)


# ══════════════════════════════════════════════════════════════════════════════
# landmark_error_mm
# ══════════════════════════════════════════════════════════════════════════════

class TestLandmarkErrorMm:
    def test_perfect_prediction_zero_error(self):
        lm = make_lm(0.5)
        errors = efa.landmark_error_mm(lm, lm, vol_shape=(64, 64, 64), voxel_mm=1.0)
        for name in KP_NAMES:
            assert errors[name] == pytest.approx(0.0, abs=1e-6)

    def test_known_offset_error(self):
        vol_shape = (100, 100, 100)
        voxel_mm = 1.0
        # Offset pred by 0.1 in PD dimension → 10 mm error
        lm_actual = {"humerus_shaft": (0.0, 0.0, 0.0)}
        lm_pred = {"humerus_shaft": (0.1, 0.0, 0.0)}
        errors = efa.landmark_error_mm(lm_pred, lm_actual, vol_shape, voxel_mm)
        assert errors["humerus_shaft"] == pytest.approx(10.0, abs=1e-4)

    def test_3d_offset_euclidean(self):
        vol_shape = (100, 100, 100)
        voxel_mm = 1.0
        lm_actual = {"olecranon": (0.0, 0.0, 0.0)}
        # Offset by (3, 4, 0) voxels → 5 mm (3-4-5 Pythagorean triple)
        lm_pred = {"olecranon": (0.03, 0.04, 0.0)}
        errors = efa.landmark_error_mm(lm_pred, lm_actual, vol_shape, voxel_mm)
        assert errors["olecranon"] == pytest.approx(5.0, abs=1e-3)

    def test_missing_landmark_skipped(self):
        lm_actual = {"humerus_shaft": (0.5, 0.5, 0.5)}
        lm_pred = {}  # empty
        errors = efa.landmark_error_mm(lm_pred, lm_actual, (64, 64, 64), 1.0)
        assert "humerus_shaft" not in errors

    def test_voxel_mm_scales_error(self):
        vol_shape = (100, 100, 100)
        lm_actual = {"humerus_shaft": (0.0, 0.0, 0.0)}
        lm_pred = {"humerus_shaft": (0.1, 0.0, 0.0)}
        e1 = efa.landmark_error_mm(lm_pred, lm_actual, vol_shape, voxel_mm=1.0)
        e2 = efa.landmark_error_mm(lm_pred, lm_actual, vol_shape, voxel_mm=2.0)
        assert e2["humerus_shaft"] == pytest.approx(e1["humerus_shaft"] * 2)

    def test_all_six_kp_names_present(self):
        lm = make_lm(0.5)
        errors = efa.landmark_error_mm(lm, lm, (64, 64, 64), 1.0)
        assert set(errors.keys()) == set(KP_NAMES)


# ══════════════════════════════════════════════════════════════════════════════
# procrustes_rotation
# ══════════════════════════════════════════════════════════════════════════════

class TestProcrustesRotation:
    def _make_pts(self):
        return np.array([[0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]], dtype=float)

    def test_identity_case_returns_identity(self):
        pts = self._make_pts()
        R, c_from, c_to = efa.procrustes_rotation(pts, pts)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_returns_rotation_matrix(self):
        pts = self._make_pts()
        R, _, _ = efa.procrustes_rotation(pts, pts)
        assert R.shape == (3, 3)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)

    def test_returns_correct_centers(self):
        pts = self._make_pts()
        _, c_from, c_to = efa.procrustes_rotation(pts, pts)
        expected_center = pts.mean(axis=0)
        np.testing.assert_allclose(c_from, expected_center, atol=1e-10)
        np.testing.assert_allclose(c_to, expected_center, atol=1e-10)

    def test_known_90deg_rotation_z(self):
        """R should recover a 90° rotation around Z."""
        pts_from = np.array([[1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0]], dtype=float)
        # Rotate 90° around Z: (x, y, z) → (-y, x, z)
        R_true = np.array([[0.0, -1.0, 0.0],
                            [1.0,  0.0, 0.0],
                            [0.0,  0.0, 1.0]])
        pts_to = (R_true @ (pts_from - pts_from.mean(0)).T).T + pts_from.mean(0)
        R, _, _ = efa.procrustes_rotation(pts_from, pts_to)
        np.testing.assert_allclose(R, R_true, atol=1e-8)

    def test_det_plus_one_not_reflection(self):
        rng = np.random.default_rng(42)
        pts_from = rng.standard_normal((5, 3))
        pts_to = rng.standard_normal((5, 3))
        R, _, _ = efa.procrustes_rotation(pts_from, pts_to)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-8)

    def test_centers_are_means(self):
        rng = np.random.default_rng(0)
        pts_from = rng.standard_normal((4, 3))
        pts_to = rng.standard_normal((4, 3))
        _, c_from, c_to = efa.procrustes_rotation(pts_from, pts_to)
        np.testing.assert_allclose(c_from, pts_from.mean(0), atol=1e-12)
        np.testing.assert_allclose(c_to, pts_to.mean(0), atol=1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# nearest_volume_for_angle
# ══════════════════════════════════════════════════════════════════════════════

class TestNearestVolumeForAngle:
    def test_exact_match_returns_self(self):
        assert efa.nearest_volume_for_angle(90.0) == 90.0
        assert efa.nearest_volume_for_angle(135.0) == 135.0
        assert efa.nearest_volume_for_angle(180.0) == 180.0

    def test_below_90_returns_90(self):
        assert efa.nearest_volume_for_angle(70.0) == 90.0
        assert efa.nearest_volume_for_angle(60.0) == 90.0

    def test_above_180_returns_180(self):
        assert efa.nearest_volume_for_angle(200.0) == 180.0

    def test_between_90_and_135_correct(self):
        # 110 is closer to 135? No: |110-90|=20, |110-135|=25 → 90
        assert efa.nearest_volume_for_angle(110.0) == 90.0
        # 120 is equidistant: |120-90|=30, |120-135|=15 → 135
        assert efa.nearest_volume_for_angle(120.0) == 135.0

    def test_between_135_and_180_correct(self):
        # 150: |150-135|=15, |150-180|=30 → 135
        assert efa.nearest_volume_for_angle(150.0) == 135.0
        # 160: |160-135|=25, |160-180|=20 → 180
        assert efa.nearest_volume_for_angle(160.0) == 180.0

    def test_custom_available_angles(self):
        result = efa.nearest_volume_for_angle(100.0, available_angles=[80.0, 120.0, 160.0])
        assert result == 80.0

    def test_single_available_angle(self):
        result = efa.nearest_volume_for_angle(42.0, available_angles=[135.0])
        assert result == 135.0

    def test_default_available_angles(self):
        # Default is [180.0, 135.0, 90.0]
        result = efa.nearest_volume_for_angle(130.0)
        assert result in [180.0, 135.0, 90.0]


# ══════════════════════════════════════════════════════════════════════════════
# rotation_matrix_custom_axis
# ══════════════════════════════════════════════════════════════════════════════

class TestRotationMatrixCustomAxis:
    def test_zero_degrees_is_identity(self):
        R = efa.rotation_matrix_custom_axis([0, 0, 1], 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_output_shape(self):
        R = efa.rotation_matrix_custom_axis([1, 0, 0], 45.0)
        assert R.shape == (3, 3)

    def test_is_rotation_matrix(self):
        R = efa.rotation_matrix_custom_axis([1, 1, 1], 37.5)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)

    def test_90deg_around_z(self):
        R = efa.rotation_matrix_custom_axis([0, 0, 1], 90.0)
        expected = np.array([[0, -1, 0],
                              [1,  0, 0],
                              [0,  0, 1]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_180deg_around_x(self):
        R = efa.rotation_matrix_custom_axis([1, 0, 0], 180.0)
        expected = np.array([[1,  0,  0],
                              [0, -1,  0],
                              [0,  0, -1]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_negative_and_positive_angle_inverse(self):
        R_pos = efa.rotation_matrix_custom_axis([0, 1, 0], 45.0)
        R_neg = efa.rotation_matrix_custom_axis([0, 1, 0], -45.0)
        np.testing.assert_allclose(R_pos @ R_neg, np.eye(3), atol=1e-10)

    def test_unnormalized_axis_still_rotation(self):
        """Non-unit axis should still produce a valid rotation matrix."""
        R = efa.rotation_matrix_custom_axis([2, 0, 0], 90.0)  # same as [1, 0, 0]
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-8)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# rotate_volume_data_driven
# ══════════════════════════════════════════════════════════════════════════════

class TestRotateVolumeDataDriven:
    """Tests for rotate_volume_data_driven."""

    _AXIS_Z = [0.0, 0.0, 1.0]

    def _vol(self, size=32):
        v = np.zeros((size, size, size), dtype=np.float32)
        v[size // 2 :, :, :] = 1.0  # forearm half is 1
        return v

    def _lm(self):
        return {
            "humerus_shaft":      (0.10, 0.50, 0.50),
            "lateral_epicondyle": (0.50, 0.45, 0.50),
            "medial_epicondyle":  (0.50, 0.55, 0.50),
            "forearm_shaft":      (0.90, 0.50, 0.50),
            "radial_head":        (0.80, 0.45, 0.50),
            "olecranon":          (0.80, 0.55, 0.50),
        }

    def test_returns_tuple_of_two(self):
        result = efa.rotate_volume_data_driven(
            self._vol(), self._lm(), 90.0, 180.0, self._AXIS_Z
        )
        assert len(result) == 2

    def test_output_volume_same_shape(self):
        vol = self._vol()
        out_vol, _ = efa.rotate_volume_data_driven(
            vol, self._lm(), 90.0, 180.0, self._AXIS_Z
        )
        assert out_vol.shape == vol.shape

    def test_output_landmarks_same_keys(self):
        lm = self._lm()
        _, out_lm = efa.rotate_volume_data_driven(
            self._vol(), lm, 90.0, 180.0, self._AXIS_Z
        )
        assert set(out_lm.keys()) == set(lm.keys())

    def test_humerus_shaft_unchanged(self):
        """humerus_shaft is part of the humerus set and should be preserved."""
        lm = self._lm()
        _, out_lm = efa.rotate_volume_data_driven(
            self._vol(), lm, 90.0, 180.0, self._AXIS_Z
        )
        assert out_lm["humerus_shaft"] == pytest.approx(lm["humerus_shaft"], abs=1e-6)

    def test_forearm_shaft_transformed(self):
        """Non-humerus landmarks should be moved when delta != 0."""
        lm = self._lm()
        _, out_lm = efa.rotate_volume_data_driven(
            self._vol(), lm, 90.0, 180.0, self._AXIS_Z
        )
        # forearm_shaft is not a humerus landmark → should be rotated
        orig = np.array(lm["forearm_shaft"])
        new = np.array(out_lm["forearm_shaft"])
        assert not np.allclose(orig, new, atol=1e-6)

    def test_zero_delta_no_rotation(self):
        """Same target and base flexion → volume unchanged."""
        vol = self._vol()
        lm = self._lm()
        out_vol, out_lm = efa.rotate_volume_data_driven(
            vol, lm, 135.0, 135.0, self._AXIS_Z
        )
        # Volume should be nearly the same (affine_transform with identity)
        np.testing.assert_allclose(out_vol, vol, atol=1.0)

    def test_output_landmarks_in_unit_range(self):
        """All landmark coordinates should remain in [0, 1]."""
        _, out_lm = efa.rotate_volume_data_driven(
            self._vol(), self._lm(), 90.0, 180.0, self._AXIS_Z
        )
        for name, coords in out_lm.items():
            for c in coords:
                assert 0.0 <= c <= 1.0, f"{name} coordinate {c} out of [0,1]"

    def test_voxel_mm_accepted(self):
        """voxel_mm parameter should be accepted without error."""
        out_vol, out_lm = efa.rotate_volume_data_driven(
            self._vol(), self._lm(), 90.0, 180.0, self._AXIS_Z, voxel_mm=0.5
        )
        assert out_vol.shape == self._vol().shape
