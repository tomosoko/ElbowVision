"""
Unit tests for inference.py image-processing utility functions:
  - apply_gradcam_overlay
  - detect_bone_landmarks_classical
  - _decode_image
  - _record_stats
"""
import copy
import io
import sys
import os

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import inference
from inference import (
    apply_gradcam_overlay,
    detect_bone_landmarks_classical,
    _decode_image,
    _record_stats,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bgr_image(h: int = 256, w: int = 256) -> np.ndarray:
    """Solid grey BGR image."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _xray_like(h: int = 512, w: int = 384) -> np.ndarray:
    """Synthetic X-ray–like image: bright vertical bone stripe on dark background."""
    img = np.zeros((h, w), dtype=np.uint8)
    cx = w // 2
    # upper bone (humerus)
    img[50:260, cx - 30:cx + 30] = 200
    # lower bone (forearm)
    img[270:470, cx - 25:cx + 25] = 180
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _png_bytes(h: int = 64, w: int = 64) -> bytes:
    pil = Image.fromarray(np.full((h, w, 3), 100, dtype=np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(h: int = 64, w: int = 64) -> bytes:
    pil = Image.fromarray(np.full((h, w, 3), 100, dtype=np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _landmark_dict(*, carrying_angle=None, flexion=None, engine="Classical CV", score=75):
    """Minimal landmarks dict accepted by _record_stats."""
    return {
        "angles": {
            "carrying_angle": carrying_angle,
            "flexion": flexion,
        },
        "qa": {
            "inference_engine": engine,
            "score": score,
        },
    }


# ─── apply_gradcam_overlay ────────────────────────────────────────────────────

class TestApplyGradcamOverlay:
    def test_output_shape_matches_input(self):
        img = _bgr_image(200, 300)
        cam = np.random.rand(50, 75).astype(np.float32)
        out = apply_gradcam_overlay(img, cam)
        assert out.shape == img.shape

    def test_output_is_uint8(self):
        img = _bgr_image()
        cam = np.random.rand(32, 32).astype(np.float32)
        out = apply_gradcam_overlay(img, cam)
        assert out.dtype == np.uint8

    def test_alpha_zero_returns_original(self):
        img = _bgr_image()
        cam = np.ones((16, 16), dtype=np.float32)
        out = apply_gradcam_overlay(img, cam, alpha=0.0)
        np.testing.assert_array_equal(out, img)

    def test_alpha_one_is_pure_heatmap(self):
        img = _bgr_image()
        cam = np.zeros((16, 16), dtype=np.float32)
        out = apply_gradcam_overlay(img, cam, alpha=1.0)
        # All-zero cam → COLORMAP_JET maps 0 → dark blue (B~128, G≈0, R≈0)
        assert out.shape == img.shape

    def test_default_alpha_blends(self):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        cam = np.ones((32, 32), dtype=np.float32)
        out = apply_gradcam_overlay(img, cam)
        # Blended output should not be all-zero (cam adds colour)
        assert out.sum() > 0

    def test_non_square_image(self):
        img = _bgr_image(100, 400)
        cam = np.random.rand(25, 100).astype(np.float32)
        out = apply_gradcam_overlay(img, cam)
        assert out.shape == (100, 400, 3)

    def test_cam_all_zeros(self):
        img = _bgr_image()
        cam = np.zeros((16, 16), dtype=np.float32)
        out = apply_gradcam_overlay(img, cam)
        assert out.shape == img.shape

    def test_cam_all_ones(self):
        img = _bgr_image()
        cam = np.ones((16, 16), dtype=np.float32)
        out = apply_gradcam_overlay(img, cam)
        assert out.shape == img.shape


# ─── detect_bone_landmarks_classical ──────────────────────────────────────────

class TestDetectBoneLandmarksClassicalKeys:
    """Return-value structure tests — work on any image."""

    REQUIRED_LANDMARK_KEYS = {
        "humerus_shaft", "condyle_center",
        "lateral_epicondyle", "medial_epicondyle",
        "forearm_shaft", "forearm_ext",
        "qa", "angles",
    }
    REQUIRED_QA_KEYS = {
        "view_type", "score", "status", "message",
        "color", "symmetry_ratio", "positioning_advice", "inference_engine",
    }
    REQUIRED_ANGLE_KEYS = {
        "carrying_angle", "flexion",
        "pronation_sup", "ps_label",
        "varus_valgus", "vv_label",
    }

    @pytest.fixture(scope="class")
    def result(self):
        return detect_bone_landmarks_classical(_xray_like())

    def test_top_level_keys(self, result):
        assert self.REQUIRED_LANDMARK_KEYS.issubset(result.keys())

    def test_qa_keys(self, result):
        assert self.REQUIRED_QA_KEYS.issubset(result["qa"].keys())

    def test_angle_keys(self, result):
        assert self.REQUIRED_ANGLE_KEYS.issubset(result["angles"].keys())

    def test_landmark_point_has_xy(self, result):
        for key in ("humerus_shaft", "condyle_center", "lateral_epicondyle",
                    "medial_epicondyle", "forearm_shaft", "forearm_ext"):
            pt = result[key]
            assert "x" in pt and "y" in pt, f"{key} missing x/y"

    def test_landmark_point_has_pct(self, result):
        for key in ("humerus_shaft", "condyle_center", "lateral_epicondyle",
                    "medial_epicondyle", "forearm_shaft", "forearm_ext"):
            pt = result[key]
            assert "x_pct" in pt and "y_pct" in pt, f"{key} missing x_pct/y_pct"

    def test_view_type_is_valid(self, result):
        assert result["qa"]["view_type"] in ("AP", "LAT")

    def test_qa_status_is_valid(self, result):
        assert result["qa"]["status"] in ("GOOD", "FAIR", "POOR")

    def test_qa_score_in_range(self, result):
        assert 0 <= result["qa"]["score"] <= 100

    def test_symmetry_ratio_in_range(self, result):
        assert 0.0 <= result["qa"]["symmetry_ratio"] <= 1.0

    def test_inference_engine_label(self, result):
        assert result["qa"]["inference_engine"] == "Classical CV"

    def test_pronation_sup_capped(self, result):
        ps = result["angles"]["pronation_sup"]
        assert -30.0 <= ps <= 30.0

    def test_either_carrying_or_flexion_is_none(self, result):
        a = result["angles"]
        # For a given view exactly one of carrying_angle / flexion is None
        assert (a["carrying_angle"] is None) != (a["flexion"] is None)


class TestDetectBoneLandmarksClassicalInputVariants:
    def test_grayscale_input(self):
        gray = np.zeros((256, 192), dtype=np.uint8)
        gray[40:130, 80:112] = 200
        gray[140:230, 82:110] = 180
        result = detect_bone_landmarks_classical(gray)
        assert "qa" in result

    def test_bgr_input(self):
        result = detect_bone_landmarks_classical(_xray_like())
        assert "qa" in result

    def test_black_image_does_not_crash(self):
        """All-black image: CLAHE preprocessing creates non-trivial output.
        Function must not raise and must return the required structure."""
        black = np.zeros((256, 192, 3), dtype=np.uint8)
        result = detect_bone_landmarks_classical(black)
        assert "qa" in result and "angles" in result

    def test_different_aspect_ratios(self):
        tall = np.zeros((768, 256, 3), dtype=np.uint8)
        tall[60:380, 100:156] = 200
        tall[390:720, 102:154] = 180
        result = detect_bone_landmarks_classical(tall)
        assert "qa" in result

    def test_small_image(self):
        small = np.zeros((64, 48, 3), dtype=np.uint8)
        result = detect_bone_landmarks_classical(small)
        assert "angles" in result


# ─── _decode_image ────────────────────────────────────────────────────────────

class TestDecodeImage:
    def test_png_returns_bgr_array(self):
        data = _png_bytes(64, 64)
        arr = _decode_image(data, "test.png")
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (64, 64, 3)

    def test_jpeg_returns_bgr_array(self):
        data = _jpeg_bytes(32, 48)
        arr = _decode_image(data, "photo.jpg")
        assert arr.ndim == 3 and arr.shape[2] == 3

    def test_uppercase_extension(self):
        data = _png_bytes(16, 16)
        arr = _decode_image(data, "IMAGE.PNG")
        assert arr.shape == (16, 16, 3)

    def test_invalid_bytes_raises(self):
        with pytest.raises((ValueError, Exception)):
            _decode_image(b"not an image at all", "bad.png")

    def test_output_dtype_is_uint8(self):
        data = _png_bytes()
        arr = _decode_image(data, "img.png")
        assert arr.dtype == np.uint8

    def test_pixel_values_preserved(self):
        # Create a red image, encode as PNG, decode and check channel
        pil = Image.fromarray(np.full((8, 8, 3), [255, 0, 0], dtype=np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        arr = _decode_image(buf.getvalue(), "red.png")
        # PIL RGB → cv2 BGR: red channel becomes index 2
        assert arr[4, 4, 2] == 255  # R in BGR
        assert arr[4, 4, 0] == 0    # B in BGR


# ─── _record_stats ────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_inference_stats():
    """Save and restore module-level _inference_stats around each test."""
    original = copy.deepcopy(inference._inference_stats)
    yield
    inference._inference_stats.clear()
    inference._inference_stats.update(original)


class TestRecordStats:
    def test_increments_total(self):
        before = inference._inference_stats["total_inferences"]
        _record_stats(_landmark_dict(carrying_angle=10.0))
        assert inference._inference_stats["total_inferences"] == before + 1

    def test_increments_twice(self):
        before = inference._inference_stats["total_inferences"]
        _record_stats(_landmark_dict())
        _record_stats(_landmark_dict())
        assert inference._inference_stats["total_inferences"] == before + 2

    def test_carrying_angle_appended(self):
        _record_stats(_landmark_dict(carrying_angle=12.5))
        assert 12.5 in inference._inference_stats["carrying_angles"]

    def test_flexion_appended(self):
        _record_stats(_landmark_dict(flexion=88.0))
        assert 88.0 in inference._inference_stats["flexion_angles"]

    def test_carrying_none_not_appended(self):
        before = len(inference._inference_stats["carrying_angles"])
        _record_stats(_landmark_dict(carrying_angle=None))
        assert len(inference._inference_stats["carrying_angles"]) == before

    def test_flexion_none_not_appended(self):
        before = len(inference._inference_stats["flexion_angles"])
        _record_stats(_landmark_dict(flexion=None))
        assert len(inference._inference_stats["flexion_angles"]) == before

    def test_qa_score_appended(self):
        _record_stats(_landmark_dict(score=95))
        assert 95 in inference._inference_stats["qa_scores"]

    def test_classical_cv_engine_counted(self):
        before = inference._inference_stats["engine_counts"]["classical_cv"]
        _record_stats(_landmark_dict(engine="Classical CV"))
        assert inference._inference_stats["engine_counts"]["classical_cv"] == before + 1

    def test_yolo_engine_counted(self):
        before = inference._inference_stats["engine_counts"]["yolo_pose"]
        _record_stats(_landmark_dict(engine="YOLOv8-Pose"))
        assert inference._inference_stats["engine_counts"]["yolo_pose"] == before + 1

    def test_yolo_does_not_increment_classical(self):
        before = inference._inference_stats["engine_counts"]["classical_cv"]
        _record_stats(_landmark_dict(engine="YOLOv8-Pose"))
        assert inference._inference_stats["engine_counts"]["classical_cv"] == before

    def test_multiple_angles_accumulated(self):
        _record_stats(_landmark_dict(carrying_angle=5.0))
        _record_stats(_landmark_dict(carrying_angle=10.0))
        angles = inference._inference_stats["carrying_angles"]
        assert 5.0 in angles and 10.0 in angles
