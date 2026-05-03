"""
Unit tests for scripts/measure_landmarks.py pure functions.

Covered:
  - Module constants (CT_DIR, TARGET_SIZE, HU_MIN, HU_MAX, SERIES_NUM, LATERALITY)
  - precise_landmark_measurement(): full landmark detection from synthetic volumes
    - Return structure (landmarks dict + extra_info dict)
    - 7 expected landmark names
    - Normalized coordinate ranges [0, 1]
    - Epicondyle detection at maximum ML-width level
    - Bone threshold (Otsu) validity
    - Olecranon posterior position
    - Radial head distal + lateral position
    - Humerus shaft proximal to epicondyle, forearm shaft distal
    - Symmetric volume produces midline ML landmarks
    - Edge cases: minimal bone, single-slice bone
  - shaft_centroid() via precise_landmark_measurement internal logic
  - generate_report(): text output structure, content validation
"""
from __future__ import annotations

import os
import sys
import types
import unittest.mock as mock
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Mock elbow_synth before importing the module ──────────────────────────────

_mock_elbow_synth = types.ModuleType("elbow_synth")
_mock_elbow_synth.load_ct_volume = MagicMock()
_mock_elbow_synth.auto_detect_landmarks = MagicMock()
_mock_elbow_synth.DEFAULT_LANDMARKS_NORMALIZED = {
    "humerus_shaft":      (0.25, 0.50, 0.50),
    "lateral_epicondyle": (0.50, 0.50, 0.75),
    "medial_epicondyle":  (0.50, 0.50, 0.25),
    "forearm_shaft":      (0.75, 0.50, 0.50),
    "radial_head":        (0.58, 0.40, 0.70),
    "olecranon":          (0.50, 0.80, 0.50),
    "joint_center":       (0.50, 0.50, 0.50),
}
sys.modules["elbow_synth"] = _mock_elbow_synth

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

with patch("os.makedirs"):
    import measure_landmarks as _ml

# Clean up sys.modules to avoid polluting other test files
del sys.modules["elbow_synth"]


# ── Helpers ──────────────────────────────────────────────────────────────────

EXPECTED_LANDMARK_NAMES = {
    "humerus_shaft", "lateral_epicondyle", "medial_epicondyle",
    "forearm_shaft", "radial_head", "olecranon", "joint_center",
}


def _make_elbow_volume(size: int = 64) -> np.ndarray:
    """
    Create a synthetic volume that mimics an elbow CT.

    The volume has shape (size, size, size) with values in [0, 1].
    Bone is represented as a high-intensity cylindrical structure
    with a wider region at the epicondyle level (mid PD).

    Axes: (PD, AP, ML) where PD=0 is proximal.
    """
    vol = np.zeros((size, size, size), dtype=np.float32)
    center_ap = size // 2
    center_ml = size // 2

    for pd_i in range(size):
        pd_frac = pd_i / size
        # Epicondyle region: wider in ML at mid-PD (0.4-0.6)
        if 0.35 <= pd_frac <= 0.65:
            # Wide bone at epicondyle level
            ml_half = int(size * 0.3)  # wide in ML
            ap_half = int(size * 0.15)
            for ap_i in range(max(0, center_ap - ap_half),
                              min(size, center_ap + ap_half)):
                for ml_i in range(max(0, center_ml - ml_half),
                                  min(size, center_ml + ml_half)):
                    vol[pd_i, ap_i, ml_i] = 0.8
            # Add posterior extension for olecranon
            if 0.45 <= pd_frac <= 0.55:
                for ap_i in range(center_ap + ap_half,
                                  min(size, center_ap + ap_half + int(size * 0.15))):
                    for ml_i in range(center_ml - 3, center_ml + 3):
                        vol[pd_i, ap_i, ml_i] = 0.8
        else:
            # Shaft: narrower cylindrical bone
            radius = int(size * 0.08)
            for ap_i in range(max(0, center_ap - radius),
                              min(size, center_ap + radius)):
                for ml_i in range(max(0, center_ml - radius),
                                  min(size, center_ml + radius)):
                    vol[pd_i, ap_i, ml_i] = 0.8

    return vol


def _make_simple_volume(size: int = 32) -> np.ndarray:
    """Minimal volume with bone only at the center slice."""
    vol = np.zeros((size, size, size), dtype=np.float32)
    mid = size // 2
    # Single wide bone slice at mid PD
    vol[mid, mid - 3:mid + 3, mid - 8:mid + 8] = 0.9
    # Thin shaft above and below
    vol[mid - 5:mid, mid - 2:mid + 2, mid - 2:mid + 2] = 0.9
    vol[mid + 1:mid + 6, mid - 2:mid + 2, mid - 2:mid + 2] = 0.9
    return vol


def _run_precise(vol: np.ndarray):
    """Run precise_landmark_measurement suppressing prints."""
    import io
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        return _ml.precise_landmark_measurement(vol)


# ── Module Constants ─────────────────────────────────────────────────────────

class TestModuleConstants:
    def test_target_size(self):
        assert _ml.TARGET_SIZE == 128

    def test_hu_min(self):
        assert _ml.HU_MIN == -200

    def test_hu_max(self):
        assert _ml.HU_MAX == 1000

    def test_series_num(self):
        assert _ml.SERIES_NUM == 1

    def test_laterality(self):
        assert _ml.LATERALITY == "R"

    def test_ct_dir_contains_raw_dicom(self):
        assert "raw_dicom" in _ml.CT_DIR

    def test_out_dir_contains_domain_gap(self):
        assert "domain_gap_analysis" in _ml.OUT_DIR


# ── precise_landmark_measurement: Return Structure ───────────────────────────

class TestPreciseLandmarkReturnStructure:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.vol = _make_elbow_volume(64)
        self.landmarks, self.extra = _run_precise(self.vol)

    def test_returns_tuple_of_two(self):
        assert isinstance(self.landmarks, dict)
        assert isinstance(self.extra, dict)

    def test_all_expected_landmark_names_present(self):
        assert set(self.landmarks.keys()) == EXPECTED_LANDMARK_NAMES

    def test_each_landmark_is_3_tuple(self):
        for name, coords in self.landmarks.items():
            assert len(coords) == 3, f"{name} should have 3 coordinates"

    def test_extra_info_has_required_keys(self):
        required = {"epicondyle_pd_idx", "ml_widths", "bone_area",
                     "bone_threshold", "most_posterior_ap"}
        assert required.issubset(set(self.extra.keys()))

    def test_ml_widths_shape_matches_pd_size(self):
        assert len(self.extra["ml_widths"]) == self.vol.shape[0]

    def test_bone_area_shape_matches_pd_size(self):
        assert len(self.extra["bone_area"]) == self.vol.shape[0]

    def test_epicondyle_pd_idx_is_int(self):
        assert isinstance(self.extra["epicondyle_pd_idx"], (int, np.integer))

    def test_bone_threshold_is_positive(self):
        assert self.extra["bone_threshold"] > 0


# ── precise_landmark_measurement: Coordinate Ranges ─────────────────────────

class TestCoordinateRanges:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.vol = _make_elbow_volume(64)
        self.landmarks, self.extra = _run_precise(self.vol)

    def test_all_coordinates_in_unit_range(self):
        for name, (pd, ap, ml) in self.landmarks.items():
            assert 0.0 <= pd <= 1.0, f"{name} PD={pd} out of [0,1]"
            assert 0.0 <= ap <= 1.0, f"{name} AP={ap} out of [0,1]"
            assert 0.0 <= ml <= 1.0, f"{name} ML={ml} out of [0,1]"

    def test_epicondyle_pd_in_central_range(self):
        """Epicondyle should be detected in the central 40% of PD axis."""
        pd_norm = self.extra["epicondyle_pd_idx"] / self.vol.shape[0]
        assert 0.30 <= pd_norm <= 0.70

    def test_most_posterior_ap_above_half(self):
        """Most posterior bone point should be in posterior half."""
        assert self.extra["most_posterior_ap"] >= 0.5


# ── precise_landmark_measurement: Anatomical Consistency ─────────────────────

class TestAnatomicalConsistency:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.vol = _make_elbow_volume(64)
        self.landmarks, self.extra = _run_precise(self.vol)

    def test_humerus_shaft_proximal_to_epicondyle(self):
        """Humerus shaft PD should be less than epicondyle PD (more proximal)."""
        hs_pd = self.landmarks["humerus_shaft"][0]
        epic_pd = self.landmarks["lateral_epicondyle"][0]
        assert hs_pd < epic_pd

    def test_forearm_shaft_distal_to_epicondyle(self):
        """Forearm shaft PD should be greater than epicondyle PD (more distal)."""
        fs_pd = self.landmarks["forearm_shaft"][0]
        epic_pd = self.landmarks["lateral_epicondyle"][0]
        assert fs_pd > epic_pd

    def test_lateral_epicondyle_ml_greater_than_medial(self):
        """Lateral epicondyle should be more lateral (higher ML) than medial."""
        lat_ml = self.landmarks["lateral_epicondyle"][2]
        med_ml = self.landmarks["medial_epicondyle"][2]
        assert lat_ml > med_ml

    def test_joint_center_ml_between_epicondyles(self):
        """Joint center ML should be between the two epicondyles."""
        jc_ml = self.landmarks["joint_center"][2]
        lat_ml = self.landmarks["lateral_epicondyle"][2]
        med_ml = self.landmarks["medial_epicondyle"][2]
        assert med_ml <= jc_ml <= lat_ml

    def test_joint_center_is_ml_midpoint_of_epicondyles(self):
        """Joint center ML = average of lateral and medial epicondyle ML."""
        jc_ml = self.landmarks["joint_center"][2]
        lat_ml = self.landmarks["lateral_epicondyle"][2]
        med_ml = self.landmarks["medial_epicondyle"][2]
        expected = (lat_ml + med_ml) / 2
        assert abs(jc_ml - expected) < 1e-6

    def test_olecranon_is_posterior(self):
        """Olecranon AP should be more posterior than epicondyle AP centroid."""
        ol_ap = self.landmarks["olecranon"][1]
        epic_ap = self.landmarks["lateral_epicondyle"][1]
        assert ol_ap > epic_ap

    def test_radial_head_distal_to_epicondyle(self):
        """Radial head PD should be distal to epicondyle."""
        rh_pd = self.landmarks["radial_head"][0]
        epic_pd = self.landmarks["lateral_epicondyle"][0]
        assert rh_pd > epic_pd

    def test_epicondyle_pd_same_for_lat_med_olecranon_jc(self):
        """Lateral/medial epicondyle, olecranon, and joint_center share PD."""
        epic_pd = self.landmarks["lateral_epicondyle"][0]
        assert self.landmarks["medial_epicondyle"][0] == epic_pd
        assert self.landmarks["olecranon"][0] == epic_pd
        assert self.landmarks["joint_center"][0] == epic_pd


# ── precise_landmark_measurement: Epicondyle Detection ──────────────────────

class TestEpicondyleDetection:
    def test_epicondyle_at_widest_ml_slice(self):
        """Epicondyle PD should be at the slice with maximum ML bone width."""
        vol = _make_elbow_volume(64)
        landmarks, extra = _run_precise(vol)
        epic_idx = extra["epicondyle_pd_idx"]
        ml_widths = extra["ml_widths"]
        # Within central 40%, the epicondyle should be at max ML width
        pd_s = int(vol.shape[0] * 0.30)
        pd_e = int(vol.shape[0] * 0.70)
        central_max_idx = pd_s + int(ml_widths[pd_s:pd_e].argmax())
        assert epic_idx == central_max_idx

    def test_ml_widths_nonzero_in_bone_region(self):
        """ML widths should be nonzero where bone exists."""
        vol = _make_elbow_volume(64)
        _, extra = _run_precise(vol)
        ml_widths = extra["ml_widths"]
        # At least some slices should have nonzero ML width
        assert (ml_widths > 0).sum() > 10

    def test_bone_area_nonzero_in_bone_region(self):
        vol = _make_elbow_volume(64)
        _, extra = _run_precise(vol)
        assert (extra["bone_area"] > 0).sum() > 10


# ── precise_landmark_measurement: Symmetric Volume ──────────────────────────

class TestSymmetricVolume:
    """A volume symmetric about ML midpoint should produce midline landmarks."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.size = 64
        self.vol = _make_elbow_volume(self.size)
        self.landmarks, self.extra = _run_precise(self.vol)

    def test_joint_center_near_ml_midpoint(self):
        jc_ml = self.landmarks["joint_center"][2]
        assert abs(jc_ml - 0.5) < 0.05

    def test_humerus_shaft_near_ml_midpoint(self):
        hs_ml = self.landmarks["humerus_shaft"][2]
        assert abs(hs_ml - 0.5) < 0.05

    def test_forearm_shaft_near_ml_midpoint(self):
        fs_ml = self.landmarks["forearm_shaft"][2]
        assert abs(fs_ml - 0.5) < 0.05

    def test_epicondyles_symmetric_about_midpoint(self):
        lat_ml = self.landmarks["lateral_epicondyle"][2]
        med_ml = self.landmarks["medial_epicondyle"][2]
        midpoint = (lat_ml + med_ml) / 2
        assert abs(midpoint - 0.5) < 0.05


# ── precise_landmark_measurement: Edge Cases ────────────────────────────────

class TestEdgeCases:
    def test_uniform_zero_volume_raises(self):
        """Volume with no bone raises ValueError (empty bone mask)."""
        vol = np.zeros((32, 32, 32), dtype=np.float32)
        with pytest.raises(ValueError):
            _run_precise(vol)

    def test_uniform_high_volume(self):
        """Volume that is all bone should still return valid landmarks."""
        vol = np.ones((32, 32, 32), dtype=np.float32) * 0.9
        landmarks, extra = _run_precise(vol)
        assert set(landmarks.keys()) == EXPECTED_LANDMARK_NAMES

    def test_small_volume(self):
        """Minimum viable volume size."""
        vol = _make_simple_volume(32)
        landmarks, extra = _run_precise(vol)
        assert set(landmarks.keys()) == EXPECTED_LANDMARK_NAMES
        for name, (pd, ap, ml) in landmarks.items():
            assert 0.0 <= pd <= 1.0
            assert 0.0 <= ap <= 1.0
            assert 0.0 <= ml <= 1.0

    def test_different_volume_sizes(self):
        """Test with different volume sizes to ensure normalization works."""
        for sz in [32, 48, 64]:
            vol = _make_elbow_volume(sz)
            landmarks, extra = _run_precise(vol)
            for name, (pd, ap, ml) in landmarks.items():
                assert 0.0 <= pd <= 1.0, f"size={sz} {name} PD={pd}"
                assert 0.0 <= ap <= 1.0, f"size={sz} {name} AP={ap}"
                assert 0.0 <= ml <= 1.0, f"size={sz} {name} ML={ml}"


# ── precise_landmark_measurement: Bone Threshold ────────────────────────────

class TestBoneThreshold:
    def test_threshold_between_background_and_bone(self):
        """Otsu threshold should separate background (0) from bone (~0.8)."""
        vol = _make_elbow_volume(64)
        _, extra = _run_precise(vol)
        thresh = extra["bone_threshold"]
        assert 0.0 < thresh < 0.8

    def test_threshold_splits_bimodal_distribution(self):
        """For a clean bimodal volume, threshold should be between modes."""
        vol = np.zeros((32, 32, 32), dtype=np.float32)
        vol[10:22, 10:22, 10:22] = 0.9  # bone block
        _, extra = _run_precise(vol)
        thresh = extra["bone_threshold"]
        # Otsu may pick a low threshold when background dominates;
        # key invariant: threshold > 0 and < bone intensity
        assert 0.0 < thresh < 0.9


# ── generate_report ──────────────────────────────────────────────────────────

class TestGenerateReport:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.auto_lm = {
            "humerus_shaft": (0.25, 0.50, 0.50),
            "lateral_epicondyle": (0.50, 0.50, 0.75),
            "medial_epicondyle": (0.50, 0.50, 0.25),
            "forearm_shaft": (0.75, 0.50, 0.50),
            "radial_head": (0.58, 0.40, 0.70),
            "olecranon": (0.50, 0.80, 0.50),
            "joint_center": (0.50, 0.50, 0.50),
        }
        self.precise_lm = {
            "humerus_shaft": (0.24, 0.48, 0.51),
            "lateral_epicondyle": (0.49, 0.51, 0.76),
            "medial_epicondyle": (0.49, 0.51, 0.24),
            "forearm_shaft": (0.74, 0.49, 0.51),
            "radial_head": (0.57, 0.39, 0.71),
            "olecranon": (0.49, 0.81, 0.51),
            "joint_center": (0.49, 0.51, 0.50),
        }
        self.default_lm = dict(_mock_elbow_synth.DEFAULT_LANDMARKS_NORMALIZED)
        self.extra_info = {
            "epicondyle_pd_idx": 32,
            "ml_widths": np.zeros(64),
            "bone_area": np.zeros(64),
            "bone_threshold": 0.45,
            "most_posterior_ap": 0.82,
        }
        self.extra_info["ml_widths"][32] = 20.0
        self.volume_shape = (64, 64, 64)
        self.voxel_mm = 1.5
        self.out_path = str(tmp_path / "report.txt")

    def test_report_returns_string(self):
        report = _ml.generate_report(
            self.auto_lm, self.precise_lm, self.default_lm,
            self.extra_info, self.volume_shape, self.voxel_mm, self.out_path)
        assert isinstance(report, str)

    def test_report_file_created(self):
        _ml.generate_report(
            self.auto_lm, self.precise_lm, self.default_lm,
            self.extra_info, self.volume_shape, self.voxel_mm, self.out_path)
        assert os.path.exists(self.out_path)

    def test_report_contains_header(self):
        report = _ml.generate_report(
            self.auto_lm, self.precise_lm, self.default_lm,
            self.extra_info, self.volume_shape, self.voxel_mm, self.out_path)
        assert "Phantom CT Landmark Measurement Report" in report

    def test_report_contains_volume_shape(self):
        report = _ml.generate_report(
            self.auto_lm, self.precise_lm, self.default_lm,
            self.extra_info, self.volume_shape, self.voxel_mm, self.out_path)
        assert str(self.volume_shape) in report

    def test_report_contains_voxel_size(self):
        report = _ml.generate_report(
            self.auto_lm, self.precise_lm, self.default_lm,
            self.extra_info, self.volume_shape, self.voxel_mm, self.out_path)
        assert "1.500" in report

    def test_report_contains_bone_threshold(self):
        report = _ml.generate_report(
            self.auto_lm, self.precise_lm, self.default_lm,
            self.extra_info, self.volume_shape, self.voxel_mm, self.out_path)
        assert "0.450" in report

    def test_report_contains_all_landmark_names(self):
        report = _ml.generate_report(
            self.auto_lm, self.precise_lm, self.default_lm,
            self.extra_info, self.volume_shape, self.voxel_mm, self.out_path)
        for name in EXPECTED_LANDMARK_NAMES:
            assert name in report

    def test_report_contains_auto_detected_section(self):
        report = _ml.generate_report(
            self.auto_lm, self.precise_lm, self.default_lm,
            self.extra_info, self.volume_shape, self.voxel_mm, self.out_path)
        assert "Auto-detected landmarks" in report

    def test_report_contains_precise_section(self):
        report = _ml.generate_report(
            self.auto_lm, self.precise_lm, self.default_lm,
            self.extra_info, self.volume_shape, self.voxel_mm, self.out_path)
        assert "Precise landmarks" in report

    def test_report_contains_default_section(self):
        report = _ml.generate_report(
            self.auto_lm, self.precise_lm, self.default_lm,
            self.extra_info, self.volume_shape, self.voxel_mm, self.out_path)
        assert "Default landmarks" in report

    def test_report_contains_difference_section(self):
        report = _ml.generate_report(
            self.auto_lm, self.precise_lm, self.default_lm,
            self.extra_info, self.volume_shape, self.voxel_mm, self.out_path)
        assert "Difference: Default vs Precise" in report

    def test_report_contains_paste_ready_section(self):
        report = _ml.generate_report(
            self.auto_lm, self.precise_lm, self.default_lm,
            self.extra_info, self.volume_shape, self.voxel_mm, self.out_path)
        assert "DEFAULT_LANDMARKS_NORMALIZED" in report

    def test_report_euclidean_distances_in_mm(self):
        """Report should contain mm-scale distances."""
        report = _ml.generate_report(
            self.auto_lm, self.precise_lm, self.default_lm,
            self.extra_info, self.volume_shape, self.voxel_mm, self.out_path)
        assert "mm" in report

    def test_report_file_content_matches_return(self):
        report = _ml.generate_report(
            self.auto_lm, self.precise_lm, self.default_lm,
            self.extra_info, self.volume_shape, self.voxel_mm, self.out_path)
        with open(self.out_path) as f:
            file_content = f.read()
        assert file_content == report

    def test_report_with_missing_default_landmark(self):
        """If a landmark is not in default_lm, report should note it."""
        partial_default = {k: v for k, v in self.default_lm.items()
                           if k != "joint_center"}
        report = _ml.generate_report(
            self.auto_lm, self.precise_lm, partial_default,
            self.extra_info, self.volume_shape, self.voxel_mm, self.out_path)
        assert "not in DEFAULT" in report


# ── precise_landmark_measurement: Determinism ────────────────────────────────

class TestDeterminism:
    def test_same_volume_produces_same_landmarks(self):
        """Two calls with the same volume should produce identical results."""
        vol = _make_elbow_volume(48)
        lm1, ex1 = _run_precise(vol)
        lm2, ex2 = _run_precise(vol)
        for name in EXPECTED_LANDMARK_NAMES:
            for i in range(3):
                assert lm1[name][i] == lm2[name][i], f"{name}[{i}] differs"
        assert ex1["epicondyle_pd_idx"] == ex2["epicondyle_pd_idx"]
        assert ex1["bone_threshold"] == ex2["bone_threshold"]


# ── precise_landmark_measurement: Sensitivity ───────────────────────────────

class TestSensitivity:
    def test_wider_epicondyle_shifts_ml_range(self):
        """A volume with wider epicondyle should have larger ML separation."""
        vol_narrow = _make_elbow_volume(64)
        vol_wide = _make_elbow_volume(64)
        # Make the wide volume even wider at epicondyle
        mid = 64 // 2
        for pd_i in range(int(64 * 0.35), int(64 * 0.65)):
            vol_wide[pd_i, mid - 8:mid + 8, mid - 28:mid + 28] = 0.8

        lm_narrow, _ = _run_precise(vol_narrow)
        lm_wide, _ = _run_precise(vol_wide)

        narrow_span = (lm_narrow["lateral_epicondyle"][2] -
                       lm_narrow["medial_epicondyle"][2])
        wide_span = (lm_wide["lateral_epicondyle"][2] -
                     lm_wide["medial_epicondyle"][2])
        assert wide_span > narrow_span
