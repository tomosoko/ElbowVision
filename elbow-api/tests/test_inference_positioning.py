"""
Unit tests for inference.py positioning utility functions:
  - pct
  - angle_deg
  - full_angle
  - estimate_positioning_correction (all AP/LAT × rotation/flexion branches)
"""
import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from inference import pct, angle_deg, full_angle, estimate_positioning_correction


# ─── helpers ──────────────────────────────────────────────────────────────────

def _dummy_image(h: int = 200, w: int = 200) -> np.ndarray:
    """Gray dummy image — only shape is used by estimate_positioning_correction."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _landmarks(
    *,
    view_type: str,
    lat_x: float,
    lat_y: float,
    med_x: float,
    med_y: float,
    flexion: float | None = None,
) -> dict:
    """Build a minimal landmarks dict compatible with estimate_positioning_correction."""
    return {
        "lateral_epicondyle": {"x": lat_x, "y": lat_y},
        "medial_epicondyle":  {"x": med_x, "y": med_y},
        "qa": {"view_type": view_type},
        "angles": {"flexion": flexion},
    }


# Diagonal of 200×200 image ≈ 282.8 px
_DIAG = math.sqrt(200 ** 2 + 200 ** 2)


def _sep_for_ratio(ratio: float, h: int = 200, w: int = 200) -> float:
    """Return epic_sep (pixels) that yields the given epic_ratio."""
    return ratio * math.sqrt(h ** 2 + w ** 2)


# ─── pct ──────────────────────────────────────────────────────────────────────

class TestPct:
    def test_half(self):
        assert pct(50, 100) == 50.0

    def test_zero(self):
        assert pct(0, 100) == 0.0

    def test_full(self):
        assert pct(100, 100) == 100.0

    def test_fraction_rounded(self):
        # 1/3 ≈ 33.33
        assert pct(1, 3) == 33.33

    def test_float_input(self):
        assert pct(1.5, 6.0) == 25.0


# ─── angle_deg ────────────────────────────────────────────────────────────────

class TestAngleDeg:
    def test_horizontal_right(self):
        assert angle_deg({"x": 0, "y": 0}, {"x": 1, "y": 0}) == pytest.approx(0.0)

    def test_vertical_down(self):
        assert angle_deg({"x": 0, "y": 0}, {"x": 0, "y": 1}) == pytest.approx(90.0)

    def test_diagonal_45(self):
        assert angle_deg({"x": 0, "y": 0}, {"x": 1, "y": 1}) == pytest.approx(45.0)

    def test_horizontal_left(self):
        assert angle_deg({"x": 1, "y": 0}, {"x": 0, "y": 0}) == pytest.approx(180.0)

    def test_vertical_up(self):
        assert angle_deg({"x": 0, "y": 1}, {"x": 0, "y": 0}) == pytest.approx(-90.0)


# ─── full_angle ────────────────────────────────────────────────────────────────

class TestFullAngle:
    def test_zero_diff(self):
        assert full_angle(45.0, 45.0) == pytest.approx(0.0)

    def test_90_deg(self):
        assert full_angle(0.0, 90.0) == pytest.approx(90.0)

    def test_180_deg(self):
        assert full_angle(0.0, 180.0) == pytest.approx(180.0)

    def test_takes_shorter_arc_270(self):
        # |270| > 180 → 360 - 270 = 90
        assert full_angle(0.0, 270.0) == pytest.approx(90.0)

    def test_symmetry(self):
        assert full_angle(30.0, 150.0) == full_angle(150.0, 30.0)

    def test_wrap_across_360(self):
        # diff = |350 - 10| = 340 > 180 → 360 - 340 = 20
        assert full_angle(350.0, 10.0) == pytest.approx(20.0)


# ─── estimate_positioning_correction — AP view ────────────────────────────────

class TestEstimatePositioningCorrectionAP:
    """AP view: rotation evaluated by epic_ratio (lat/med separation / diagonal)."""

    def _ap(self, sep_px: float) -> dict:
        """Landmarks with given epic_separation along x axis, AP view."""
        img = _dummy_image()
        lm = _landmarks(
            view_type="AP",
            lat_x=100.0 + sep_px / 2,
            lat_y=100.0,
            med_x=100.0 - sep_px / 2,
            med_y=100.0,
        )
        return estimate_positioning_correction(img, lm)

    # ── rotation: good (epic_ratio >= 0.10) ─────────────────────────────────
    def test_ap_rotation_good_level(self):
        sep = _sep_for_ratio(0.12)
        result = self._ap(sep)
        assert result["rotation_level"] == "good"
        assert result["rotation_error"] == 0.0

    def test_ap_rotation_good_correction_not_needed(self):
        sep = _sep_for_ratio(0.15)
        result = self._ap(sep)
        assert result["correction_needed"] is False

    def test_ap_rotation_good_view_type(self):
        sep = _sep_for_ratio(0.20)
        result = self._ap(sep)
        assert result["view_type"] == "AP"

    # ── rotation: minor (0.05 <= epic_ratio < 0.10) ─────────────────────────
    def test_ap_rotation_minor_level(self):
        sep = _sep_for_ratio(0.075)
        result = self._ap(sep)
        assert result["rotation_level"] == "minor"

    def test_ap_rotation_minor_error_positive(self):
        sep = _sep_for_ratio(0.075)
        result = self._ap(sep)
        assert result["rotation_error"] > 0.0

    def test_ap_rotation_minor_correction_needed(self):
        sep = _sep_for_ratio(0.075)
        result = self._ap(sep)
        assert result["correction_needed"] is True

    # ── rotation: major (epic_ratio < 0.05) ─────────────────────────────────
    def test_ap_rotation_major_level(self):
        sep = _sep_for_ratio(0.02)
        result = self._ap(sep)
        assert result["rotation_level"] == "major"

    def test_ap_rotation_major_error_up_to_45(self):
        sep = _sep_for_ratio(0.00)
        result = self._ap(sep)
        assert 0.0 < result["rotation_error"] <= 45.0

    # ── AP has no flexion fields ─────────────────────────────────────────────
    def test_ap_no_flexion_level(self):
        sep = _sep_for_ratio(0.12)
        result = self._ap(sep)
        assert result["flexion_level"] is None

    def test_ap_no_flexion_advice(self):
        sep = _sep_for_ratio(0.12)
        result = self._ap(sep)
        assert result["flexion_advice"] is None

    # ── overall_level equals rotation_level for AP ───────────────────────────
    def test_ap_overall_matches_rotation_good(self):
        result = self._ap(_sep_for_ratio(0.15))
        assert result["overall_level"] == "good"

    def test_ap_overall_matches_rotation_major(self):
        result = self._ap(_sep_for_ratio(0.01))
        assert result["overall_level"] == "major"

    # ── epic_separation_px and epic_ratio returned ───────────────────────────
    def test_ap_returns_epic_sep_and_ratio(self):
        sep = _sep_for_ratio(0.12)
        result = self._ap(sep)
        assert result["epic_separation_px"] > 0
        assert result["epic_ratio"] == pytest.approx(0.12, abs=1e-3)


# ─── estimate_positioning_correction — LAT view, rotation ─────────────────────

class TestEstimatePositioningCorrectionLATRotation:
    """LAT view: rotation good when upper epicondyles overlap (epic_ratio < 0.04)."""

    def _lat(self, sep_px: float, flexion: float = 90.0) -> dict:
        img = _dummy_image()
        lm = _landmarks(
            view_type="LAT",
            lat_x=100.0 + sep_px / 2,
            lat_y=100.0,
            med_x=100.0 - sep_px / 2,
            med_y=100.0,
            flexion=flexion,
        )
        return estimate_positioning_correction(img, lm)

    def test_lat_rotation_good(self):
        sep = _sep_for_ratio(0.02)
        result = self._lat(sep)
        assert result["rotation_level"] == "good"
        assert result["rotation_error"] == 0.0

    def test_lat_rotation_minor(self):
        sep = _sep_for_ratio(0.07)
        result = self._lat(sep)
        assert result["rotation_level"] == "minor"
        assert result["rotation_error"] > 0.0

    def test_lat_rotation_major(self):
        sep = _sep_for_ratio(0.15)
        result = self._lat(sep)
        assert result["rotation_level"] == "major"
        assert result["rotation_error"] > 0.0

    def test_lat_rotation_major_error_capped(self):
        # Very large separation → error ≤ 45°
        sep = _sep_for_ratio(1.0)
        result = self._lat(sep)
        assert result["rotation_error"] <= 45.0


# ─── estimate_positioning_correction — LAT view, flexion ──────────────────────

class TestEstimatePositioningCorrectionLATFlexion:
    """LAT view: flexion evaluated separately from rotation."""

    def _lat(self, flexion: float) -> dict:
        """LAT with perfect rotation (epic_ratio ≈ 0) and given flexion."""
        img = _dummy_image()
        sep = _sep_for_ratio(0.01)   # near-zero → rotation good
        lm = _landmarks(
            view_type="LAT",
            lat_x=100.0 + sep / 2,
            lat_y=100.0,
            med_x=100.0 - sep / 2,
            med_y=100.0,
            flexion=flexion,
        )
        return estimate_positioning_correction(img, lm)

    # good: 80–100°
    def test_flexion_good_at_90(self):
        result = self._lat(90.0)
        assert result["flexion_level"] == "good"

    def test_flexion_good_at_boundary_80(self):
        result = self._lat(80.0)
        assert result["flexion_level"] == "good"

    def test_flexion_good_at_boundary_100(self):
        result = self._lat(100.0)
        assert result["flexion_level"] == "good"

    # minor short: 70 <= flexion < 80
    def test_flexion_minor_short(self):
        result = self._lat(75.0)
        assert result["flexion_level"] == "minor"

    # major short: flexion < 70
    def test_flexion_major_short(self):
        result = self._lat(60.0)
        assert result["flexion_level"] == "major"

    # minor over: 100 < flexion <= 110
    def test_flexion_minor_over(self):
        result = self._lat(105.0)
        assert result["flexion_level"] == "minor"

    # major over: flexion > 110
    def test_flexion_major_over(self):
        result = self._lat(120.0)
        assert result["flexion_level"] == "major"

    # flexion advice populated for LAT
    def test_flexion_advice_not_none_for_lat(self):
        result = self._lat(90.0)
        assert result["flexion_advice"] is not None

    # flexion_deg rounded and returned
    def test_flexion_deg_returned(self):
        result = self._lat(85.3)
        assert result["flexion_deg"] == pytest.approx(85.3, abs=0.1)

    # no flexion when flexion=None
    def test_flexion_none_when_missing(self):
        img = _dummy_image()
        sep = _sep_for_ratio(0.01)
        lm = _landmarks(
            view_type="LAT",
            lat_x=100.0 + sep / 2, lat_y=100.0,
            med_x=100.0 - sep / 2, med_y=100.0,
            flexion=None,
        )
        result = estimate_positioning_correction(img, lm)
        assert result["flexion_level"] is None
        assert result["flexion_deg"] is None


# ─── estimate_positioning_correction — overall_level logic ────────────────────

class TestEstimatePositioningCorrectionOverall:
    """overall_level = worst of rotation_level and flexion_level."""

    def _lat(self, sep_ratio: float, flexion: float) -> dict:
        img = _dummy_image()
        sep = _sep_for_ratio(sep_ratio)
        lm = _landmarks(
            view_type="LAT",
            lat_x=100.0 + sep / 2, lat_y=100.0,
            med_x=100.0 - sep / 2, med_y=100.0,
            flexion=flexion,
        )
        return estimate_positioning_correction(img, lm)

    def test_both_good(self):
        result = self._lat(0.01, 90.0)
        assert result["overall_level"] == "good"
        assert result["correction_needed"] is False

    def test_rotation_minor_flexion_good(self):
        result = self._lat(0.07, 90.0)
        assert result["overall_level"] == "minor"

    def test_rotation_good_flexion_major(self):
        result = self._lat(0.01, 60.0)
        assert result["overall_level"] == "major"

    def test_rotation_major_flexion_minor(self):
        result = self._lat(0.15, 75.0)
        assert result["overall_level"] == "major"

    def test_correction_needed_when_minor(self):
        result = self._lat(0.07, 90.0)
        assert result["correction_needed"] is True

    def test_correction_not_needed_when_both_good(self):
        result = self._lat(0.01, 90.0)
        assert result["correction_needed"] is False
