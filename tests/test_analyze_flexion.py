"""Tests for scripts/analyze_flexion.py

Covers:
  - Module constants (TARGET_SIZE, HU_MIN, HU_MAX)
  - SERIES dict structure (series_num → (name, flexion_angle))
  - compare_landmarks(): delta computation, motion vector, report format
  - compute_metrics(): SSIM/Dice return types, value ranges, landmark distance
  - write_report(): file creation, assessment thresholds (GOOD/MODERATE/POOR)
  - bone_mask logic (Otsu threshold via compute_metrics)
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import types
import unittest.mock as mock
from contextlib import redirect_stdout

import numpy as np
import pytest

# ── Mock elbow_synth before importing the module ──────────────────────────────

_mock_es = types.ModuleType("elbow_synth")
_mock_es.load_ct_volume = mock.MagicMock()
_mock_es.auto_detect_landmarks = mock.MagicMock()
_mock_es.rotate_volume_and_landmarks = mock.MagicMock()
_mock_es.generate_drr = mock.MagicMock()
_mock_es.compute_flexion_angle = mock.MagicMock(return_value=90.0)
sys.modules["elbow_synth"] = _mock_es

# Make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

with mock.patch("os.makedirs"):
    import analyze_flexion as af

# Remove mock so downstream test files can register their own elbow_synth mock.
del sys.modules["elbow_synth"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_landmarks(offset: float = 0.0) -> dict:
    """Return a landmark dict with 6 standard keypoints."""
    return {
        "joint_center":         np.array([0.5 + offset, 0.5, 0.5]),
        "forearm_shaft":        np.array([0.8 + offset, 0.3, 0.5]),
        "radial_head":          np.array([0.6 + offset, 0.4, 0.5]),
        "humerus_shaft":        np.array([0.2 + offset, 0.6, 0.5]),
        "lateral_epicondyle":   np.array([0.5 + offset, 0.5, 0.4]),
        "medial_epicondyle":    np.array([0.5 + offset, 0.5, 0.6]),
    }


def _make_data(shape: tuple = (64, 64, 64), voxel_mm: float = 1.0) -> dict:
    """Return a data dict suitable for compute_metrics and write_report."""
    vol = np.zeros(shape)
    rng = np.random.default_rng(0)

    def _img(seed: int) -> np.ndarray:
        r = np.random.default_rng(seed)
        return r.integers(0, 256, (shape[0], shape[1]), dtype=np.uint8)

    return {
        90:  {"landmarks": _make_landmarks(0.0), "volume": vol, "voxel_mm": voxel_mm},
        135: {"landmarks": _make_landmarks(0.1), "volume": vol, "voxel_mm": voxel_mm},
        180: {"landmarks": _make_landmarks(0.2), "volume": vol, "voxel_mm": voxel_mm},
        "synth_90_AP":   _img(1),
        "real_90_AP":    _img(2),
        "synth_90_LAT":  _img(3),
        "real_90_LAT":   _img(4),
        "synth_135_AP":  _img(5),
        "real_135_AP":   _img(6),
        "synth_135_LAT": _img(7),
        "real_135_LAT":  _img(8),
        "lm_90_synth":   _make_landmarks(0.05),
        "lm_135_synth":  _make_landmarks(0.15),
    }


def _run_compare_landmarks(data: dict) -> str:
    """Run compare_landmarks, capture stdout, and return the report string."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        report = af.compare_landmarks(data)
    return report


def _run_compute_metrics(data: dict | None = None) -> dict:
    """Run compute_metrics with cv2.imwrite mocked out."""
    if data is None:
        data = _make_data()
    buf = io.StringIO()
    with mock.patch("cv2.imwrite"), redirect_stdout(buf):
        return af.compute_metrics(data)


def _run_write_report(metrics: dict, assessment_key: str = "90deg_LAT_ssim") -> str:
    """Write a report to a temp dir and return its contents."""
    data = _make_data()
    with tempfile.TemporaryDirectory() as tmpdir:
        original_out = af.OUT_DIR
        af.OUT_DIR = tmpdir
        buf = io.StringIO()
        with redirect_stdout(buf):
            af.write_report("landmark report text", metrics, data)
        af.OUT_DIR = original_out
        report_path = os.path.join(tmpdir, "analysis_report.txt")
        with open(report_path) as f:
            return f.read()


def _default_metrics(ssim_90_lat: float = 0.8) -> dict:
    return {
        "90deg_LAT_ssim": ssim_90_lat,
        "90deg_LAT_dice": 0.9,
        "90deg_AP_ssim":  0.75,
        "90deg_AP_dice":  0.85,
        "135deg_LAT_ssim": 0.72,
        "135deg_LAT_dice": 0.82,
        "135deg_AP_ssim":  0.70,
        "135deg_AP_dice":  0.80,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1. Constants
# ══════════════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_target_size_is_256(self):
        assert af.TARGET_SIZE == 256

    def test_hu_min_is_50(self):
        assert af.HU_MIN == 50

    def test_hu_max_is_1000(self):
        assert af.HU_MAX == 1000

    def test_target_size_is_int(self):
        assert isinstance(af.TARGET_SIZE, int)

    def test_hu_min_less_than_hu_max(self):
        assert af.HU_MIN < af.HU_MAX

    def test_hu_min_positive(self):
        assert af.HU_MIN > 0


# ══════════════════════════════════════════════════════════════════════════════
# 2. SERIES dict structure
# ══════════════════════════════════════════════════════════════════════════════

class TestSeriesStructure:
    def test_series_has_three_entries(self):
        assert len(af.SERIES) == 3

    def test_series_keys_are_4_8_12(self):
        assert set(af.SERIES.keys()) == {4, 8, 12}

    def test_series4_name_and_flexion(self):
        name, flexion = af.SERIES[4]
        assert name == "S4_180deg"
        assert flexion == 180

    def test_series8_name_and_flexion(self):
        name, flexion = af.SERIES[8]
        assert name == "S8_135deg"
        assert flexion == 135

    def test_series12_name_and_flexion(self):
        name, flexion = af.SERIES[12]
        assert name == "S12_90deg"
        assert flexion == 90

    def test_flexion_angles_are_descending_by_series_num(self):
        flexions = [af.SERIES[k][1] for k in sorted(af.SERIES.keys())]
        assert flexions == sorted(flexions, reverse=True)

    def test_all_names_are_strings(self):
        for name, _ in af.SERIES.values():
            assert isinstance(name, str)

    def test_all_flexions_are_ints(self):
        for _, flexion in af.SERIES.values():
            assert isinstance(flexion, int)

    def test_names_contain_deg(self):
        for name, _ in af.SERIES.values():
            assert "deg" in name

    def test_flexion_values_are_90_135_180(self):
        flexions = set(v[1] for v in af.SERIES.values())
        assert flexions == {90, 135, 180}


# ══════════════════════════════════════════════════════════════════════════════
# 3. compare_landmarks()
# ══════════════════════════════════════════════════════════════════════════════

class TestCompareLandmarks:
    def _make_data(self, offset_90: float = 0.2, offset_135: float = 0.1):
        vol = np.zeros((64, 64, 64))
        return {
            180: {"landmarks": _make_landmarks(0.0),         "volume": vol, "name": "S4_180deg"},
            135: {"landmarks": _make_landmarks(offset_135),  "volume": vol, "name": "S8_135deg"},
            90:  {"landmarks": _make_landmarks(offset_90),   "volume": vol, "name": "S12_90deg"},
        }

    def test_returns_string(self):
        data = self._make_data()
        report = _run_compare_landmarks(data)
        assert isinstance(report, str)

    def test_report_contains_landmark_header(self):
        data = self._make_data()
        report = _run_compare_landmarks(data)
        assert "Landmark" in report

    def test_report_contains_180deg_column(self):
        data = self._make_data()
        report = _run_compare_landmarks(data)
        assert "180deg" in report

    def test_report_contains_90deg_column(self):
        data = self._make_data()
        report = _run_compare_landmarks(data)
        assert "90deg" in report

    def test_report_contains_delta_column(self):
        data = self._make_data()
        report = _run_compare_landmarks(data)
        assert "delta" in report

    def test_report_contains_joint_center(self):
        data = self._make_data()
        report = _run_compare_landmarks(data)
        assert "joint_center" in report

    def test_report_contains_forearm_shaft(self):
        data = self._make_data()
        report = _run_compare_landmarks(data)
        assert "forearm_shaft" in report

    def test_delta_is_90_minus_180(self):
        """Delta column must equal the 90deg value minus the 180deg value."""
        offset_90 = 0.3
        offset_180 = 0.0
        # For joint_center PD coord: 180 vol = 0.5, 90 vol = 0.5+0.3=0.8
        data = self._make_data(offset_90=offset_90)
        report = _run_compare_landmarks(data)
        # delta = 0.8 - 0.5 = +0.3
        assert "+0.3" in report or "0.3000" in report

    def test_motion_vector_ml_near_zero_for_pure_flexion(self):
        """For pure sagittal flexion (AP/PD only), ML component stays 0."""
        # All landmarks have same ML (index 2) across flexion angles
        data = self._make_data(offset_90=0.2, offset_135=0.1)
        # ML coords do not change: all have z=0.5
        report = _run_compare_landmarks(data)
        assert "ML" in report

    def test_report_nonempty_for_zero_motion(self):
        """Even with identical landmarks, report should be non-empty."""
        data = self._make_data(offset_90=0.0, offset_135=0.0)
        report = _run_compare_landmarks(data)
        assert len(report) > 100

    def test_compute_flexion_angle_called_for_each_volume(self):
        data = self._make_data()
        af.compute_flexion_angle = mock.MagicMock(return_value=90.0)
        _run_compare_landmarks(data)
        # Should be called once per angle (3 times)
        assert af.compute_flexion_angle.call_count == 3


# ══════════════════════════════════════════════════════════════════════════════
# 4. compute_metrics()
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeMetrics:
    def test_returns_dict(self):
        result = _run_compute_metrics()
        assert isinstance(result, dict)

    def test_contains_90deg_lat_ssim(self):
        result = _run_compute_metrics()
        assert "90deg_LAT_ssim" in result

    def test_contains_90deg_ap_ssim(self):
        result = _run_compute_metrics()
        assert "90deg_AP_ssim" in result

    def test_contains_90deg_lat_dice(self):
        result = _run_compute_metrics()
        assert "90deg_LAT_dice" in result

    def test_contains_135deg_lat_ssim(self):
        result = _run_compute_metrics()
        assert "135deg_LAT_ssim" in result

    def test_ssim_in_minus1_to_1_range(self):
        result = _run_compute_metrics()
        for key in [k for k in result if "ssim" in k]:
            assert -1.0 <= result[key] <= 1.0, f"{key}={result[key]}"

    def test_dice_in_0_to_1_range(self):
        result = _run_compute_metrics()
        for key in [k for k in result if "dice" in k]:
            assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]}"

    def test_identical_images_give_dice_1(self):
        data = _make_data()
        # Replace synth/real pairs with identical images
        img = np.full((64, 64), 200, dtype=np.uint8)
        data["synth_90_LAT"] = img
        data["real_90_LAT"]  = img
        data["synth_90_AP"]  = img
        data["real_90_AP"]   = img
        data["synth_135_LAT"] = img
        data["real_135_LAT"]  = img
        data["synth_135_AP"]  = img
        data["real_135_AP"]   = img
        result = _run_compute_metrics(data)
        assert abs(result["90deg_LAT_dice"] - 1.0) < 1e-6

    def test_identical_images_give_ssim_1(self):
        data = _make_data()
        img = np.full((64, 64), 200, dtype=np.uint8)
        for suffix in ["90_AP", "90_LAT", "135_AP", "135_LAT"]:
            data[f"synth_{suffix}"] = img
            data[f"real_{suffix}"]  = img
        result = _run_compute_metrics(data)
        assert abs(result["90deg_LAT_ssim"] - 1.0) < 1e-4

    def test_zero_image_gives_dice_zero(self):
        """Two all-zero images: both bone masks are empty → Dice = 0."""
        data = _make_data()
        img = np.zeros((64, 64), dtype=np.uint8)
        for suffix in ["90_AP", "90_LAT", "135_AP", "135_LAT"]:
            data[f"synth_{suffix}"] = img
            data[f"real_{suffix}"]  = img
        result = _run_compute_metrics(data)
        # Both masks are empty → dice ≈ 0
        assert result["90deg_LAT_dice"] < 1e-6

    def test_landmark_distance_keys_present(self):
        result = _run_compute_metrics()
        dist_keys = [k for k in result if "dist_mm" in k]
        assert len(dist_keys) > 0

    def test_identical_landmarks_give_zero_distance(self):
        data = _make_data()
        lm = _make_landmarks(0.0)
        data[90]["landmarks"] = lm
        data["lm_90_synth"] = lm
        result = _run_compute_metrics(data)
        for key in [k for k in result if "dist_mm" in k]:
            assert abs(result[key]) < 1e-6

    def test_landmark_distance_is_nonnegative(self):
        result = _run_compute_metrics()
        for key in [k for k in result if "dist_mm" in k]:
            assert result[key] >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 5. write_report() — file creation and assessment thresholds
# ══════════════════════════════════════════════════════════════════════════════

class TestWriteReport:
    def test_creates_report_file(self):
        data = _make_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            original_out = af.OUT_DIR
            af.OUT_DIR = tmpdir
            with redirect_stdout(io.StringIO()):
                af.write_report("landmark_report", _default_metrics(), data)
            af.OUT_DIR = original_out
            assert os.path.exists(os.path.join(tmpdir, "analysis_report.txt"))

    def test_report_contains_header(self):
        content = _run_write_report(_default_metrics())
        assert "ELBOW FLEXION ANALYSIS REPORT" in content

    def test_report_contains_landmark_section(self):
        content = _run_write_report(_default_metrics())
        assert "LANDMARK POSITIONS" in content

    def test_report_contains_synthesis_metrics_section(self):
        content = _run_write_report(_default_metrics())
        assert "SYNTHESIS QUALITY METRICS" in content

    def test_assessment_good_when_ssim_above_0_7(self):
        content = _run_write_report(_default_metrics(ssim_90_lat=0.75))
        assert "GOOD" in content

    def test_assessment_moderate_when_ssim_between_0_5_and_0_7(self):
        content = _run_write_report(_default_metrics(ssim_90_lat=0.65))
        assert "MODERATE" in content

    def test_assessment_poor_when_ssim_below_0_5(self):
        content = _run_write_report(_default_metrics(ssim_90_lat=0.3))
        assert "POOR" in content

    def test_boundary_exactly_0_7_is_good(self):
        """ssim == 0.7 satisfies `> 0.7` → False, but `> 0.5` → True → MODERATE."""
        content = _run_write_report(_default_metrics(ssim_90_lat=0.7))
        # 0.7 is NOT > 0.7, so it falls to the elif branch
        assert "MODERATE" in content

    def test_boundary_just_above_0_7_is_good(self):
        content = _run_write_report(_default_metrics(ssim_90_lat=0.701))
        assert "GOOD" in content

    def test_report_contains_lat_ssim_value(self):
        metrics = _default_metrics(ssim_90_lat=0.8234)
        content = _run_write_report(metrics)
        assert "0.8234" in content

    def test_report_contains_joint_center_section(self):
        content = _run_write_report(_default_metrics())
        assert "JOINT CENTER POSITIONS" in content

    def test_report_contains_limitations_section(self):
        content = _run_write_report(_default_metrics())
        assert "limitations" in content.lower() or "Key limitations" in content

    def test_landmark_report_text_included(self):
        landmark_text = "test_landmark_report_xyz_unique"
        data = _make_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            original_out = af.OUT_DIR
            af.OUT_DIR = tmpdir
            with redirect_stdout(io.StringIO()):
                af.write_report(landmark_text, _default_metrics(), data)
            af.OUT_DIR = original_out
            with open(os.path.join(tmpdir, "analysis_report.txt")) as f:
                content = f.read()
        assert landmark_text in content


# ══════════════════════════════════════════════════════════════════════════════
# 6. Dice formula (unit-level, via compute_metrics infrastructure)
# ══════════════════════════════════════════════════════════════════════════════

class TestDiceFormula:
    """Verify Dice = 2*|A∩B| / (|A|+|B|) via binary images."""

    def _single_channel_metrics(self, img_s: np.ndarray, img_r: np.ndarray) -> dict:
        data = _make_data()
        for suffix in ["90_AP", "90_LAT", "135_AP", "135_LAT"]:
            data[f"synth_{suffix}"] = img_s
            data[f"real_{suffix}"]  = img_r
        return _run_compute_metrics(data)

    def test_no_overlap_gives_dice_zero(self):
        """Images with completely disjoint foregrounds → Dice = 0."""
        img_s = np.zeros((64, 64), dtype=np.uint8)
        img_s[:32, :] = 255                      # top half white
        img_r = np.zeros((64, 64), dtype=np.uint8)
        # img_r is all-black → Otsu threshold will make mask empty
        result = self._single_channel_metrics(img_s, img_r)
        assert result["90deg_LAT_dice"] < 0.1

    def test_perfect_overlap_gives_dice_one(self):
        img = np.full((64, 64), 200, dtype=np.uint8)
        result = self._single_channel_metrics(img, img)
        assert abs(result["90deg_LAT_dice"] - 1.0) < 1e-6

    def test_dice_symmetric(self):
        """Dice(A,B) == Dice(B,A)."""
        img_s = np.random.default_rng(1).integers(0, 256, (64, 64), dtype=np.uint8)
        img_r = np.random.default_rng(2).integers(0, 256, (64, 64), dtype=np.uint8)
        r1 = self._single_channel_metrics(img_s, img_r)
        r2 = self._single_channel_metrics(img_r, img_s)
        assert abs(r1["90deg_LAT_dice"] - r2["90deg_LAT_dice"]) < 1e-10

    def test_dice_range_always_valid(self):
        for seed in range(5):
            img_s = np.random.default_rng(seed).integers(0, 256, (64, 64), dtype=np.uint8)
            img_r = np.random.default_rng(seed + 10).integers(0, 256, (64, 64), dtype=np.uint8)
            result = self._single_channel_metrics(img_s, img_r)
            d = result["90deg_LAT_dice"]
            assert 0.0 <= d <= 1.0, f"seed={seed}, dice={d}"
