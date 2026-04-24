"""
Unit tests for inference.validate_angle_with_edges.

Tests cover:
  - output schema validation
  - black / grayscale image fallback (no edge detection)
  - high / medium / low confidence branches
  - edge_lines count
  - note string content
  - various image shapes and dtypes
"""
import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from inference import validate_angle_with_edges


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_bone_image(
    h: int = 512,
    w: int = 384,
    upper_angle_deg: float = 70.0,
    lower_angle_deg: float = 110.0,
) -> np.ndarray:
    """
    Synthetic X-ray-like image: one thick line in the upper half (humerus) and
    one thick line in the lower half (forearm), both at the given angles.
    HoughLinesP reliably detects these lines.
    """
    canvas = np.zeros((h, w), dtype=np.uint8)
    cx = w // 2
    length = int(min(h, w) * 0.35)

    for cy, angle_deg in [(h // 4, upper_angle_deg), (h * 3 // 4, lower_angle_deg)]:
        dx = int(length * math.cos(math.radians(angle_deg)))
        dy = int(length * math.sin(math.radians(angle_deg)))
        cv2.line(canvas, (cx - dx, cy - dy), (cx + dx, cy + dy), 200, 8)

    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


# Fixture: bone image and its measured edge_angle
@pytest.fixture(scope="module")
def bone_image():
    return _make_bone_image()


@pytest.fixture(scope="module")
def detected_edge_angle(bone_image):
    """Run once to get the stable edge_angle for the synthetic image."""
    r = validate_angle_with_edges(bone_image, 90.0)
    assert r["edge_angle"] is not None, "fixture: edge must be detected"
    return r["edge_angle"]


# ─── output schema ─────────────────────────────────────────────────────────────

class TestOutputSchema:
    REQUIRED_KEYS = {"edge_angle", "agreement_deg", "confidence", "edge_lines", "note"}

    def test_schema_on_detection_success(self, bone_image, detected_edge_angle):
        r = validate_angle_with_edges(bone_image, detected_edge_angle)
        assert self.REQUIRED_KEYS == set(r.keys())

    def test_schema_on_detection_failure(self):
        black = np.zeros((256, 256, 3), dtype=np.uint8)
        r = validate_angle_with_edges(black, 90.0)
        assert self.REQUIRED_KEYS == set(r.keys())

    def test_confidence_is_string(self, bone_image, detected_edge_angle):
        r = validate_angle_with_edges(bone_image, detected_edge_angle)
        assert isinstance(r["confidence"], str)

    def test_edge_lines_is_int(self, bone_image, detected_edge_angle):
        r = validate_angle_with_edges(bone_image, detected_edge_angle)
        assert isinstance(r["edge_lines"], int)

    def test_note_is_string(self, bone_image, detected_edge_angle):
        r = validate_angle_with_edges(bone_image, detected_edge_angle)
        assert isinstance(r["note"], str)


# ─── fallback: no edge detected ───────────────────────────────────────────────

class TestFallbackNoEdge:
    def test_black_bgr_image_returns_none_edge_angle(self):
        black = np.zeros((256, 256, 3), dtype=np.uint8)
        r = validate_angle_with_edges(black, 90.0)
        assert r["edge_angle"] is None

    def test_black_bgr_image_returns_none_agreement(self):
        black = np.zeros((256, 256, 3), dtype=np.uint8)
        r = validate_angle_with_edges(black, 90.0)
        assert r["agreement_deg"] is None

    def test_black_bgr_image_confidence_is_low(self):
        black = np.zeros((256, 256, 3), dtype=np.uint8)
        r = validate_angle_with_edges(black, 90.0)
        assert r["confidence"] == "low"

    def test_black_bgr_image_edge_lines_is_zero(self):
        black = np.zeros((256, 256, 3), dtype=np.uint8)
        r = validate_angle_with_edges(black, 90.0)
        assert r["edge_lines"] == 0

    def test_grayscale_black_image_fallback(self):
        gray = np.zeros((256, 256), dtype=np.uint8)
        r = validate_angle_with_edges(gray, 90.0)
        assert r["edge_angle"] is None
        assert r["confidence"] == "low"


# ─── confidence branches ──────────────────────────────────────────────────────

class TestConfidenceBranches:
    def test_high_confidence_when_agreement_lte_3(self, bone_image, detected_edge_angle):
        r = validate_angle_with_edges(bone_image, detected_edge_angle - 2.0)
        assert r["confidence"] == "high"
        assert r["agreement_deg"] <= 3.0

    def test_medium_confidence_when_agreement_between_3_and_8(
        self, bone_image, detected_edge_angle
    ):
        r = validate_angle_with_edges(bone_image, detected_edge_angle - 5.0)
        assert r["confidence"] == "medium"
        assert 3.0 < r["agreement_deg"] <= 8.0

    def test_low_confidence_when_agreement_gt_8(self, bone_image, detected_edge_angle):
        r = validate_angle_with_edges(bone_image, detected_edge_angle - 15.0)
        assert r["confidence"] == "low"
        assert r["agreement_deg"] > 8.0

    def test_exact_match_is_high(self, bone_image, detected_edge_angle):
        r = validate_angle_with_edges(bone_image, detected_edge_angle)
        assert r["confidence"] == "high"
        assert r["agreement_deg"] == 0.0

    def test_confidence_values_limited_to_three_levels(self, bone_image, detected_edge_angle):
        for offset in [0.0, 2.0, 5.0, 15.0]:
            r = validate_angle_with_edges(bone_image, detected_edge_angle - offset)
            assert r["confidence"] in {"high", "medium", "low"}


# ─── edge_angle and agreement ──────────────────────────────────────────────────

class TestEdgeAngleAndAgreement:
    def test_edge_angle_is_float(self, bone_image, detected_edge_angle):
        r = validate_angle_with_edges(bone_image, detected_edge_angle)
        assert isinstance(r["edge_angle"], float)

    def test_edge_angle_nonnegative(self, bone_image, detected_edge_angle):
        r = validate_angle_with_edges(bone_image, detected_edge_angle)
        assert r["edge_angle"] >= 0.0

    def test_agreement_deg_is_nonnegative(self, bone_image, detected_edge_angle):
        for offset in [0.0, 3.0, 5.0, 20.0]:
            r = validate_angle_with_edges(bone_image, detected_edge_angle - offset)
            assert r["agreement_deg"] >= 0.0

    def test_agreement_matches_abs_diff(self, bone_image, detected_edge_angle):
        offset = 5.0
        r = validate_angle_with_edges(bone_image, detected_edge_angle - offset)
        assert abs(r["agreement_deg"] - offset) < 0.5  # rounding tolerance

    def test_edge_lines_positive_when_detected(self, bone_image, detected_edge_angle):
        r = validate_angle_with_edges(bone_image, detected_edge_angle)
        assert r["edge_lines"] > 0


# ─── note string content ──────────────────────────────────────────────────────

class TestNoteContent:
    def test_high_note_contains_agreement(self, bone_image, detected_edge_angle):
        r = validate_angle_with_edges(bone_image, detected_edge_angle)
        assert "差" in r["note"]

    def test_fallback_note_mentions_edge(self):
        black = np.zeros((256, 256, 3), dtype=np.uint8)
        r = validate_angle_with_edges(black, 90.0)
        assert "検出" in r["note"]

    def test_medium_note_contains_agreement(self, bone_image, detected_edge_angle):
        r = validate_angle_with_edges(bone_image, detected_edge_angle - 5.0)
        assert "差" in r["note"]

    def test_low_note_for_detected_low(self, bone_image, detected_edge_angle):
        r = validate_angle_with_edges(bone_image, detected_edge_angle - 15.0)
        assert "差" in r["note"]


# ─── input shape robustness ────────────────────────────────────────────────────

class TestInputShapeRobustness:
    def test_tall_image(self):
        img = _make_bone_image(h=768, w=256)
        r = validate_angle_with_edges(img, 90.0)
        assert set(r.keys()) >= {"edge_angle", "confidence"}

    def test_wide_image(self):
        img = _make_bone_image(h=256, w=512)
        r = validate_angle_with_edges(img, 90.0)
        assert set(r.keys()) >= {"edge_angle", "confidence"}

    def test_small_image_does_not_crash(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        r = validate_angle_with_edges(img, 90.0)
        assert r["confidence"] in {"high", "medium", "low"}

    def test_grayscale_input_accepted(self):
        """validate_angle_with_edges handles single-channel uint8 input."""
        img = _make_bone_image(h=512, w=384)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        r = validate_angle_with_edges(gray, 90.0)
        assert set(r.keys()) >= {"edge_angle", "confidence"}
