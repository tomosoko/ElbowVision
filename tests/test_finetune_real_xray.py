"""Tests for scripts/finetune_real_xray.py

Covers:
  - Module constants (ALL_IMAGES, KP_NAMES, FLIP_IDX, AUG_COUNT, VAL_RATIO)
  - _compute_bbox_from_kps(): bounding box from keypoints with margin & clamping
  - _format_yolo_label(): YOLO pose label string formatting
  - _augment_image_and_kps(): random augmentation preserving keypoint consistency
"""
from __future__ import annotations

import math
import os
import random
import sys
import types

import numpy as np
import pytest
from PIL import Image

# ── Mock heavy deps before importing ─────────────────────────────────────────

_saved_elbow_synth = sys.modules.get("elbow_synth")
sys.modules.setdefault("elbow_synth", types.ModuleType("elbow_synth"))

# ultralytics may not be installed
sys.modules.setdefault("ultralytics", types.ModuleType("ultralytics"))

# Make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import finetune_real_xray as ftr

# Restore elbow_synth
if _saved_elbow_synth is not None:
    sys.modules["elbow_synth"] = _saved_elbow_synth
else:
    sys.modules.pop("elbow_synth", None)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstants:
    """Verify module-level constants."""

    def test_all_images_is_list(self):
        assert isinstance(ftr.ALL_IMAGES, list)

    def test_all_images_nonempty(self):
        assert len(ftr.ALL_IMAGES) > 0

    def test_all_images_are_png(self):
        for name in ftr.ALL_IMAGES:
            assert name.endswith(".png"), f"{name} should be .png"

    def test_kp_names_has_six_entries(self):
        assert len(ftr.KP_NAMES) == 6

    def test_kp_names_contains_expected(self):
        expected = {
            "humerus_shaft", "lateral_epicondyle", "medial_epicondyle",
            "forearm_shaft", "radial_head", "olecranon",
        }
        assert set(ftr.KP_NAMES) == expected

    def test_flip_idx_length_matches_kp_names(self):
        assert len(ftr.FLIP_IDX) == len(ftr.KP_NAMES)

    def test_flip_idx_swaps_epicondyles(self):
        # lateral (1) <-> medial (2) swap
        assert ftr.FLIP_IDX[1] == 2
        assert ftr.FLIP_IDX[2] == 1

    def test_flip_idx_preserves_others(self):
        assert ftr.FLIP_IDX[0] == 0
        assert ftr.FLIP_IDX[3] == 3
        assert ftr.FLIP_IDX[4] == 4
        assert ftr.FLIP_IDX[5] == 5

    def test_flip_idx_is_involution(self):
        """Applying FLIP_IDX twice returns to original order."""
        double = [ftr.FLIP_IDX[ftr.FLIP_IDX[i]] for i in range(6)]
        assert double == list(range(6))

    def test_aug_count_positive(self):
        assert isinstance(ftr.AUG_COUNT, int)
        assert ftr.AUG_COUNT > 0

    def test_val_ratio_between_0_and_1(self):
        assert 0 < ftr.VAL_RATIO < 1


# ═══════════════════════════════════════════════════════════════════════════════
# _compute_bbox_from_kps
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeBboxFromKps:
    """Tests for _compute_bbox_from_kps."""

    def test_single_visible_keypoint(self):
        """One visible keypoint should produce a bbox centred on it."""
        kps = [(0.5, 0.5, 2), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
        cx, cy, w, h = ftr._compute_bbox_from_kps(kps, margin=0.08)
        assert abs(cx - 0.5) < 1e-6
        assert abs(cy - 0.5) < 1e-6
        # width/height = 2*margin when points are coincident
        assert abs(w - 0.16) < 1e-6
        assert abs(h - 0.16) < 1e-6

    def test_two_visible_keypoints(self):
        """Bounding box spans between two visible keypoints + margin."""
        kps = [(0.2, 0.3, 2), (0.8, 0.7, 2), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
        cx, cy, w, h = ftr._compute_bbox_from_kps(kps, margin=0.0)
        assert abs(cx - 0.5) < 1e-6
        assert abs(cy - 0.5) < 1e-6
        assert abs(w - 0.6) < 1e-6
        assert abs(h - 0.4) < 1e-6

    def test_no_visible_keypoints_fallback(self):
        """All invisible keypoints should return full-image fallback."""
        kps = [(0, 0, 0)] * 6
        cx, cy, w, h = ftr._compute_bbox_from_kps(kps)
        assert cx == 0.5
        assert cy == 0.5
        assert w == 1.0
        assert h == 1.0

    def test_margin_expands_bbox(self):
        """Larger margin produces wider/taller bbox."""
        kps = [(0.5, 0.5, 2), (0.6, 0.6, 2)] + [(0, 0, 0)] * 4
        _, _, w1, h1 = ftr._compute_bbox_from_kps(kps, margin=0.05)
        _, _, w2, h2 = ftr._compute_bbox_from_kps(kps, margin=0.15)
        assert w2 > w1
        assert h2 > h1

    def test_bbox_width_clamped_to_one(self):
        """Bbox width should not exceed 1.0."""
        kps = [(0.0, 0.5, 2), (1.0, 0.5, 2)] + [(0, 0, 0)] * 4
        _, _, w, _ = ftr._compute_bbox_from_kps(kps, margin=0.5)
        assert w <= 1.0

    def test_bbox_height_clamped_to_one(self):
        """Bbox height should not exceed 1.0."""
        kps = [(0.5, 0.0, 2), (0.5, 1.0, 2)] + [(0, 0, 0)] * 4
        _, _, _, h = ftr._compute_bbox_from_kps(kps, margin=0.5)
        assert h <= 1.0

    def test_center_clamped_inside_image(self):
        """Bbox center should ensure the box stays inside [0, 1]."""
        # Keypoints near top-left corner
        kps = [(0.05, 0.05, 2), (0.1, 0.1, 2)] + [(0, 0, 0)] * 4
        cx, cy, w, h = ftr._compute_bbox_from_kps(kps, margin=0.08)
        # Left edge should be >= 0
        assert cx - w / 2 >= -1e-9
        # Top edge should be >= 0
        assert cy - h / 2 >= -1e-9

    def test_center_clamped_bottom_right(self):
        """Bbox near bottom-right stays inside image."""
        kps = [(0.9, 0.9, 2), (0.95, 0.95, 2)] + [(0, 0, 0)] * 4
        cx, cy, w, h = ftr._compute_bbox_from_kps(kps, margin=0.08)
        assert cx + w / 2 <= 1.0 + 1e-9
        assert cy + h / 2 <= 1.0 + 1e-9

    def test_all_six_visible(self):
        """All six visible keypoints should produce a valid bbox."""
        kps = [
            (0.3, 0.2, 2), (0.4, 0.3, 2), (0.5, 0.3, 2),
            (0.6, 0.7, 2), (0.45, 0.5, 2), (0.35, 0.4, 2),
        ]
        cx, cy, w, h = ftr._compute_bbox_from_kps(kps, margin=0.08)
        assert 0 < cx < 1
        assert 0 < cy < 1
        assert 0 < w <= 1
        assert 0 < h <= 1

    def test_zero_margin(self):
        """With zero margin, bbox exactly spans keypoints."""
        kps = [(0.3, 0.4, 2), (0.7, 0.8, 2)] + [(0, 0, 0)] * 4
        cx, cy, w, h = ftr._compute_bbox_from_kps(kps, margin=0.0)
        assert abs(w - 0.4) < 1e-6
        assert abs(h - 0.4) < 1e-6

    def test_return_type(self):
        """Return should be a 4-tuple of floats."""
        kps = [(0.5, 0.5, 2)] + [(0, 0, 0)] * 5
        result = ftr._compute_bbox_from_kps(kps)
        assert len(result) == 4
        for v in result:
            assert isinstance(v, float)


# ═══════════════════════════════════════════════════════════════════════════════
# _format_yolo_label
# ═══════════════════════════════════════════════════════════════════════════════


class TestFormatYoloLabel:
    """Tests for _format_yolo_label."""

    def test_starts_with_class_zero(self):
        bbox = (0.5, 0.5, 1.0, 1.0)
        kps = [(0.1, 0.2, 2)] * 6
        line = ftr._format_yolo_label(bbox, kps)
        assert line.startswith("0 ")

    def test_contains_bbox(self):
        bbox = (0.5, 0.5, 0.8, 0.6)
        kps = [(0.1, 0.2, 2)] * 6
        line = ftr._format_yolo_label(bbox, kps)
        parts = line.split()
        assert abs(float(parts[1]) - 0.5) < 1e-5
        assert abs(float(parts[2]) - 0.5) < 1e-5
        assert abs(float(parts[3]) - 0.8) < 1e-5
        assert abs(float(parts[4]) - 0.6) < 1e-5

    def test_keypoint_count(self):
        """Should contain 6 keypoints x 3 values = 18 values after bbox."""
        bbox = (0.5, 0.5, 1.0, 1.0)
        kps = [(0.1 * i, 0.2 * i, 2) for i in range(6)]
        line = ftr._format_yolo_label(bbox, kps)
        parts = line.split()
        # 1 class + 4 bbox + 6*(2 float + 1 vis) = 5 + 18 = 23
        assert len(parts) == 23

    def test_invisible_keypoint_zeros(self):
        bbox = (0.5, 0.5, 1.0, 1.0)
        kps = [(0.0, 0.0, 0)] * 6
        line = ftr._format_yolo_label(bbox, kps)
        parts = line.split()
        # All kp x,y should be 0, vis should be 0
        for i in range(6):
            base = 5 + i * 3
            assert parts[base + 2] == "0"

    def test_visibility_preserved(self):
        bbox = (0.5, 0.5, 1.0, 1.0)
        kps = [(0.5, 0.5, 2), (0.3, 0.3, 1), (0, 0, 0),
               (0.7, 0.7, 2), (0.1, 0.1, 1), (0, 0, 0)]
        line = ftr._format_yolo_label(bbox, kps)
        parts = line.split()
        vis_values = [parts[5 + i * 3 + 2] for i in range(6)]
        assert vis_values == ["2", "1", "0", "2", "1", "0"]

    def test_returns_string(self):
        bbox = (0.5, 0.5, 1.0, 1.0)
        kps = [(0.5, 0.5, 2)] * 6
        assert isinstance(ftr._format_yolo_label(bbox, kps), str)

    def test_precision_six_decimals(self):
        """Coordinates should have 6 decimal places."""
        bbox = (0.123456789, 0.5, 1.0, 1.0)
        kps = [(0.5, 0.5, 2)] * 6
        line = ftr._format_yolo_label(bbox, kps)
        parts = line.split()
        # cx should be formatted to 6 decimal places
        assert parts[1] == "0.123457"

    def test_single_line(self):
        """Label should be a single line (no newlines)."""
        bbox = (0.5, 0.5, 1.0, 1.0)
        kps = [(0.5, 0.5, 2)] * 6
        line = ftr._format_yolo_label(bbox, kps)
        assert "\n" not in line


# ═══════════════════════════════════════════════════════════════════════════════
# _augment_image_and_kps
# ═══════════════════════════════════════════════════════════════════════════════


def _make_test_image(w=256, h=256):
    """Create a simple test image."""
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _make_kp_list():
    """Create a standard 6-keypoint list (all visible, centered)."""
    return [
        (0.3, 0.2, 2), (0.4, 0.3, 2), (0.5, 0.3, 2),
        (0.6, 0.7, 2), (0.45, 0.5, 2), (0.35, 0.4, 2),
    ]


class TestAugmentImageAndKps:
    """Tests for _augment_image_and_kps."""

    def test_returns_image_and_kps(self):
        img = _make_test_image()
        kps = _make_kp_list()
        rng = random.Random(42)
        out_img, out_kps = ftr._augment_image_and_kps(img, kps, rng)
        assert isinstance(out_img, Image.Image)
        assert isinstance(out_kps, list)

    def test_output_kps_same_length(self):
        img = _make_test_image()
        kps = _make_kp_list()
        rng = random.Random(42)
        _, out_kps = ftr._augment_image_and_kps(img, kps, rng)
        assert len(out_kps) == len(kps)

    def test_keypoints_in_0_1_range(self):
        """All visible keypoints should be clamped to [0, 1]."""
        img = _make_test_image()
        kps = _make_kp_list()
        for seed in range(20):
            rng = random.Random(seed)
            _, out_kps = ftr._augment_image_and_kps(img, kps, rng)
            for xn, yn, vis in out_kps:
                assert 0.0 <= xn <= 1.0, f"x={xn} out of range (seed={seed})"
                assert 0.0 <= yn <= 1.0, f"y={yn} out of range (seed={seed})"

    def test_invisible_keypoints_remain_zero(self):
        """Invisible keypoints (vis=0) should stay at (0, 0, 0)."""
        img = _make_test_image()
        kps = [(0.5, 0.5, 2), (0, 0, 0), (0, 0, 0),
               (0.5, 0.5, 2), (0, 0, 0), (0, 0, 0)]
        for seed in range(10):
            rng = random.Random(seed)
            _, out_kps = ftr._augment_image_and_kps(img, kps, rng)
            for i in [1, 2, 4, 5]:
                assert out_kps[i][2] == 0, f"Invisible kp {i} gained visibility (seed={seed})"

    def test_output_image_same_size(self):
        """Augmented image should have same dimensions as input."""
        img = _make_test_image(w=320, h=240)
        kps = _make_kp_list()
        rng = random.Random(42)
        out_img, _ = ftr._augment_image_and_kps(img, kps, rng)
        assert out_img.size == img.size

    def test_does_not_mutate_input(self):
        """Original image and keypoint list should not be modified."""
        img = _make_test_image()
        kps = _make_kp_list()
        kps_copy = [tuple(k) for k in kps]
        rng = random.Random(42)
        ftr._augment_image_and_kps(img, kps, rng)
        assert kps == kps_copy

    def test_different_seeds_produce_different_results(self):
        """Different RNG seeds should (usually) produce different keypoints."""
        img = _make_test_image()
        kps = _make_kp_list()
        results = []
        for seed in range(5):
            rng = random.Random(seed)
            _, out_kps = ftr._augment_image_and_kps(img, kps, rng)
            results.append(out_kps)
        # At least 2 of 5 should differ
        n_unique = len(set(str(r) for r in results))
        assert n_unique >= 2

    def test_deterministic_with_same_seed(self):
        """Same RNG seed should produce identical results."""
        img = _make_test_image()
        kps = _make_kp_list()
        np.random.seed(0)
        rng1 = random.Random(42)
        out1_img, out1_kps = ftr._augment_image_and_kps(img, kps, rng1)
        np.random.seed(0)
        rng2 = random.Random(42)
        out2_img, out2_kps = ftr._augment_image_and_kps(img, kps, rng2)
        assert out1_kps == out2_kps

    def test_visibility_values_valid(self):
        """Visibility should only be 0, 1, or 2."""
        img = _make_test_image()
        kps = _make_kp_list()
        for seed in range(15):
            rng = random.Random(seed)
            _, out_kps = ftr._augment_image_and_kps(img, kps, rng)
            for _, _, vis in out_kps:
                assert vis in (0, 1, 2), f"Invalid vis={vis} (seed={seed})"

    def test_kp_tuple_structure(self):
        """Each keypoint should be a tuple of (float, float, int)."""
        img = _make_test_image()
        kps = _make_kp_list()
        rng = random.Random(42)
        _, out_kps = ftr._augment_image_and_kps(img, kps, rng)
        for xn, yn, vis in out_kps:
            assert isinstance(xn, (int, float))
            assert isinstance(yn, (int, float))
            assert isinstance(vis, (int, float))

    def test_flip_changes_keypoint_positions(self):
        """With flip, keypoint x-coordinates should change (1-x + reorder)."""
        img = _make_test_image()
        kps = [
            (0.3, 0.2, 2), (0.2, 0.3, 2), (0.6, 0.3, 2),
            (0.5, 0.7, 2), (0.4, 0.5, 2), (0.35, 0.4, 2),
        ]
        # Across many seeds, at least one result should differ from original
        found_different = False
        for seed in range(50):
            rng = random.Random(seed)
            _, out_kps = ftr._augment_image_and_kps(img, kps, rng)
            # Check if any kp moved significantly from original position
            for orig, aug in zip(kps, out_kps):
                if orig[2] > 0 and aug[2] > 0:
                    if abs(orig[0] - aug[0]) > 0.05 or abs(orig[1] - aug[1]) > 0.05:
                        found_different = True
                        break
            if found_different:
                break
        assert found_different, "No augmentation detected across 50 seeds"

    def test_grayscale_image(self):
        """Should handle grayscale (L mode) images."""
        arr = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        kps = _make_kp_list()
        rng = random.Random(42)
        out_img, out_kps = ftr._augment_image_and_kps(img, kps, rng)
        assert len(out_kps) == 6


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: _compute_bbox_from_kps + _format_yolo_label
# ═══════════════════════════════════════════════════════════════════════════════


class TestBboxAndLabelIntegration:
    """Test that bbox -> label pipeline produces valid output."""

    def test_pipeline_produces_valid_label(self):
        kps = _make_kp_list()
        bbox = ftr._compute_bbox_from_kps(kps)
        label = ftr._format_yolo_label(bbox, kps)
        parts = label.split()
        assert len(parts) == 23
        assert parts[0] == "0"
        # All values should be parseable as float
        for p in parts:
            float(p)

    def test_augmented_kps_produce_valid_label(self):
        """After augmentation, bbox+label should still be valid."""
        img = _make_test_image()
        kps = _make_kp_list()
        rng = random.Random(42)
        _, aug_kps = ftr._augment_image_and_kps(img, kps, rng)
        bbox = ftr._compute_bbox_from_kps(aug_kps)
        label = ftr._format_yolo_label(bbox, aug_kps)
        parts = label.split()
        assert len(parts) == 23
        # bbox values should be in [0, 1]
        for i in range(1, 5):
            v = float(parts[i])
            assert 0.0 <= v <= 1.0 + 1e-6

    def test_invisible_only_produces_fallback_bbox(self):
        kps = [(0, 0, 0)] * 6
        bbox = ftr._compute_bbox_from_kps(kps)
        label = ftr._format_yolo_label(bbox, kps)
        parts = label.split()
        # cx=0.5, cy=0.5, w=1.0, h=1.0
        assert abs(float(parts[1]) - 0.5) < 1e-5
        assert abs(float(parts[2]) - 0.5) < 1e-5


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_bbox_keypoints_at_extremes(self):
        """Keypoints at image boundaries."""
        kps = [(0.0, 0.0, 2), (1.0, 1.0, 2)] + [(0, 0, 0)] * 4
        cx, cy, w, h = ftr._compute_bbox_from_kps(kps, margin=0.0)
        assert abs(cx - 0.5) < 1e-6
        assert abs(cy - 0.5) < 1e-6
        assert abs(w - 1.0) < 1e-6
        assert abs(h - 1.0) < 1e-6

    def test_bbox_coincident_keypoints(self):
        """All visible keypoints at the same position."""
        kps = [(0.3, 0.7, 2)] * 6
        cx, cy, w, h = ftr._compute_bbox_from_kps(kps, margin=0.08)
        assert abs(cx - 0.3) < 1e-6
        assert abs(cy - 0.7) < 1e-6
        # w = h = 2*margin = 0.16
        assert abs(w - 0.16) < 1e-6

    def test_label_with_mixed_visibility(self):
        """Label correctly handles mixed visible/invisible keypoints."""
        bbox = (0.5, 0.5, 0.5, 0.5)
        kps = [(0.1, 0.2, 2), (0, 0, 0), (0.3, 0.4, 1),
               (0, 0, 0), (0.5, 0.6, 2), (0, 0, 0)]
        line = ftr._format_yolo_label(bbox, kps)
        parts = line.split()
        # Check visibility flags
        assert parts[7] == "2"   # kp0
        assert parts[10] == "0"  # kp1
        assert parts[13] == "1"  # kp2
        assert parts[16] == "0"  # kp3
        assert parts[19] == "2"  # kp4
        assert parts[22] == "0"  # kp5

    def test_tiny_image_augmentation(self):
        """Augmentation on a very small image should not crash."""
        img = _make_test_image(w=16, h=16)
        kps = [(0.5, 0.5, 2)] + [(0, 0, 0)] * 5
        rng = random.Random(42)
        out_img, out_kps = ftr._augment_image_and_kps(img, kps, rng)
        assert len(out_kps) == 6

    def test_large_margin_bbox(self):
        """Very large margin should be clamped to image bounds."""
        kps = [(0.5, 0.5, 2)] + [(0, 0, 0)] * 5
        cx, cy, w, h = ftr._compute_bbox_from_kps(kps, margin=5.0)
        assert w <= 1.0
        assert h <= 1.0
