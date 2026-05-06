"""Unit tests for scripts/inference_test.py pure functions.

Tests constants, collect_images, build_csv_row, draw_overlay, and print_summary
without requiring model weights or heavy external dependencies.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module-level mocking: inference_test.py imports from elbow-api/main.py
# which requires heavy dependencies (YOLO, torch, etc). We mock those imports.
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ensure scripts/ is on path for import
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Pre-populate sys.modules with mocks for heavy deps in main.py
_saved_modules: dict[str, types.ModuleType | None] = {}


def _mock_main_module():
    """Create a mock 'main' module so inference_test can import from it."""
    mock_main = types.ModuleType("main")
    mock_main._decode_image = MagicMock(return_value=np.zeros((256, 256, 3), dtype=np.uint8))
    mock_main.detect_with_yolo_pose = MagicMock(return_value=None)
    mock_main.detect_bone_landmarks_classical = MagicMock(return_value={})
    mock_main.estimate_positioning_correction = MagicMock(return_value={})
    mock_main.validate_angle_with_edges = MagicMock(return_value=None)
    mock_main.yolo_model = None
    mock_main.convnext_model = None
    mock_main.convnext_transforms = None
    mock_main.device = "cpu"
    return mock_main


# Install mock before importing inference_test
_elbow_api_dir = str(_PROJECT_ROOT / "elbow-api")
if _elbow_api_dir not in sys.path:
    sys.path.insert(0, _elbow_api_dir)

_saved_modules["main"] = sys.modules.get("main")
sys.modules["main"] = _mock_main_module()

try:
    import inference_test as mod
finally:
    # Restore original module state after import
    if _saved_modules["main"] is None:
        sys.modules.pop("main", None)
    else:
        sys.modules["main"] = _saved_modules["main"]


# ===========================================================================
# Constants
# ===========================================================================

class TestConstants:
    """Test module-level constants."""

    def test_supported_ext_is_tuple(self):
        assert isinstance(mod.SUPPORTED_EXT, tuple)

    def test_supported_ext_contains_png(self):
        assert ".png" in mod.SUPPORTED_EXT

    def test_supported_ext_contains_jpg(self):
        assert ".jpg" in mod.SUPPORTED_EXT

    def test_supported_ext_contains_jpeg(self):
        assert ".jpeg" in mod.SUPPORTED_EXT

    def test_supported_ext_contains_dcm(self):
        assert ".dcm" in mod.SUPPORTED_EXT

    def test_supported_ext_contains_dicom(self):
        assert ".dicom" in mod.SUPPORTED_EXT

    def test_supported_ext_all_lowercase_with_dot(self):
        for ext in mod.SUPPORTED_EXT:
            assert ext.startswith(".")
            assert ext == ext.lower()

    def test_kpt_colors_is_dict(self):
        assert isinstance(mod.KPT_COLORS, dict)

    def test_kpt_colors_has_expected_keys(self):
        expected = {
            "humerus_shaft", "condyle_center", "lateral_epicondyle",
            "medial_epicondyle", "forearm_shaft", "radial_head", "olecranon",
        }
        assert set(mod.KPT_COLORS.keys()) == expected

    def test_kpt_colors_values_are_bgr_tuples(self):
        for name, color in mod.KPT_COLORS.items():
            assert isinstance(color, tuple), f"{name} color is not a tuple"
            assert len(color) == 3, f"{name} color doesn't have 3 channels"
            for c in color:
                assert 0 <= c <= 255, f"{name} has out-of-range channel value {c}"

    def test_csv_columns_is_list(self):
        assert isinstance(mod.CSV_COLUMNS, list)

    def test_csv_columns_starts_with_image_id(self):
        assert mod.CSV_COLUMNS[0] == "image_id"

    def test_csv_columns_contains_view_type(self):
        assert "view_type" in mod.CSV_COLUMNS

    def test_csv_columns_contains_ai_fields(self):
        ai_fields = ["ai_carrying_angle", "ai_flexion_deg", "ai_rotation_error"]
        for f in ai_fields:
            assert f in mod.CSV_COLUMNS

    def test_csv_columns_contains_manual_fields(self):
        assert "manual_carrying_angle" in mod.CSV_COLUMNS
        assert "manual_flexion_deg" in mod.CSV_COLUMNS

    def test_csv_columns_contains_edge_fields(self):
        assert "edge_angle" in mod.CSV_COLUMNS
        assert "edge_confidence" in mod.CSV_COLUMNS
        assert "edge_agreement_deg" in mod.CSV_COLUMNS

    def test_csv_columns_no_duplicates(self):
        assert len(mod.CSV_COLUMNS) == len(set(mod.CSV_COLUMNS))

    def test_axis_color_is_bgr_tuple(self):
        assert isinstance(mod.AXIS_COLOR, tuple)
        assert len(mod.AXIS_COLOR) == 3

    def test_angle_arc_color_is_bgr_tuple(self):
        assert isinstance(mod.ANGLE_ARC_COLOR, tuple)
        assert len(mod.ANGLE_ARC_COLOR) == 3


# ===========================================================================
# collect_images
# ===========================================================================

class TestCollectImages:
    """Test collect_images function."""

    def test_empty_directory(self, tmp_path):
        result = mod.collect_images(str(tmp_path))
        assert result == []

    def test_finds_png_files(self, tmp_path):
        (tmp_path / "test.png").touch()
        result = mod.collect_images(str(tmp_path))
        assert len(result) == 1
        assert result[0].endswith("test.png")

    def test_finds_jpg_files(self, tmp_path):
        (tmp_path / "test.jpg").touch()
        result = mod.collect_images(str(tmp_path))
        assert len(result) == 1

    def test_finds_jpeg_files(self, tmp_path):
        (tmp_path / "test.jpeg").touch()
        result = mod.collect_images(str(tmp_path))
        assert len(result) == 1

    def test_finds_dcm_files(self, tmp_path):
        (tmp_path / "test.dcm").touch()
        result = mod.collect_images(str(tmp_path))
        assert len(result) == 1

    def test_finds_dicom_files(self, tmp_path):
        (tmp_path / "test.dicom").touch()
        result = mod.collect_images(str(tmp_path))
        assert len(result) == 1

    def test_ignores_hidden_files(self, tmp_path):
        (tmp_path / ".hidden.png").touch()
        result = mod.collect_images(str(tmp_path))
        assert len(result) == 0

    def test_ignores_unsupported_extensions(self, tmp_path):
        (tmp_path / "readme.txt").touch()
        (tmp_path / "data.csv").touch()
        (tmp_path / "code.py").touch()
        result = mod.collect_images(str(tmp_path))
        assert len(result) == 0

    def test_returns_sorted_filenames(self, tmp_path):
        (tmp_path / "c.png").touch()
        (tmp_path / "a.png").touch()
        (tmp_path / "b.png").touch()
        result = mod.collect_images(str(tmp_path))
        names = [os.path.basename(p) for p in result]
        assert names == ["a.png", "b.png", "c.png"]

    def test_returns_full_paths(self, tmp_path):
        (tmp_path / "test.png").touch()
        result = mod.collect_images(str(tmp_path))
        assert os.path.isabs(result[0])

    def test_mixed_supported_and_unsupported(self, tmp_path):
        (tmp_path / "xray.png").touch()
        (tmp_path / "notes.txt").touch()
        (tmp_path / "scan.dcm").touch()
        result = mod.collect_images(str(tmp_path))
        assert len(result) == 2

    def test_case_insensitive_extension(self, tmp_path):
        (tmp_path / "test.PNG").touch()
        (tmp_path / "test2.Jpg").touch()
        result = mod.collect_images(str(tmp_path))
        assert len(result) == 2


# ===========================================================================
# build_csv_row
# ===========================================================================

def _make_result(
    filename="test.png",
    view_type="LAT",
    engine="yolov8_pose",
    carrying=10.5,
    flexion=85.0,
    rotation_error=5.2,
    qa_score=8,
    qa_status="good",
    overall_level="acceptable",
    edge_angle=11.0,
    edge_confidence=0.85,
    edge_agreement_deg=0.5,
):
    """Create a mock inference result dict."""
    return {
        "filename": filename,
        "landmarks": {
            "angles": {
                "carrying_angle": carrying,
                "flexion": flexion,
                "pronation_sup": None,
                "varus_valgus": None,
                "ps_label": "",
                "vv_label": "",
            },
            "qa": {
                "view_type": view_type,
                "inference_engine": engine,
                "score": qa_score,
                "status": qa_status,
            },
        },
        "correction": {
            "rotation_error": rotation_error,
            "rotation_level": "minor",
            "flexion_level": "normal",
            "overall_level": overall_level,
        },
        "edge_validation": {
            "edge_angle": edge_angle,
            "confidence": edge_confidence,
            "agreement_deg": edge_agreement_deg,
        } if edge_angle is not None else None,
    }


class TestBuildCsvRow:
    """Test build_csv_row function."""

    def test_returns_dict(self):
        row = mod.build_csv_row(_make_result())
        assert isinstance(row, dict)

    def test_image_id_strips_extension(self):
        row = mod.build_csv_row(_make_result(filename="xray_001.png"))
        assert row["image_id"] == "xray_001"

    def test_view_type_extracted(self):
        row = mod.build_csv_row(_make_result(view_type="AP"))
        assert row["view_type"] == "AP"

    def test_inference_engine_extracted(self):
        row = mod.build_csv_row(_make_result(engine="yolov8_pose"))
        assert row["inference_engine"] == "yolov8_pose"

    def test_carrying_angle_value(self):
        row = mod.build_csv_row(_make_result(carrying=12.3))
        assert row["ai_carrying_angle"] == 12.3

    def test_flexion_value(self):
        row = mod.build_csv_row(_make_result(flexion=90.0))
        assert row["ai_flexion_deg"] == 90.0

    def test_rotation_error_value(self):
        row = mod.build_csv_row(_make_result(rotation_error=3.5))
        assert row["ai_rotation_error"] == 3.5

    def test_qa_score_extracted(self):
        row = mod.build_csv_row(_make_result(qa_score=7))
        assert row["qa_score"] == 7

    def test_qa_status_extracted(self):
        row = mod.build_csv_row(_make_result(qa_status="warning"))
        assert row["qa_status"] == "warning"

    def test_overall_level_extracted(self):
        row = mod.build_csv_row(_make_result(overall_level="critical"))
        assert row["overall_level"] == "critical"

    def test_edge_validation_values(self):
        row = mod.build_csv_row(_make_result(
            edge_angle=15.0, edge_confidence=0.9, edge_agreement_deg=1.2
        ))
        assert row["edge_angle"] == 15.0
        assert row["edge_confidence"] == 0.9
        assert row["edge_agreement_deg"] == 1.2

    def test_edge_validation_none(self):
        row = mod.build_csv_row(_make_result(edge_angle=None))
        assert row["edge_angle"] is None
        assert row["edge_confidence"] is None
        assert row["edge_agreement_deg"] is None

    def test_manual_fields_empty_string(self):
        row = mod.build_csv_row(_make_result())
        assert row["manual_carrying_angle"] == ""
        assert row["manual_flexion_deg"] == ""

    def test_none_carrying_angle(self):
        row = mod.build_csv_row(_make_result(carrying=None))
        assert row["ai_carrying_angle"] is None

    def test_none_flexion(self):
        row = mod.build_csv_row(_make_result(flexion=None))
        assert row["ai_flexion_deg"] is None

    def test_none_rotation_error(self):
        row = mod.build_csv_row(_make_result(rotation_error=None))
        result = _make_result(rotation_error=None)
        result["correction"].pop("rotation_error", None)
        row = mod.build_csv_row(result)
        assert row["ai_rotation_error"] is None

    def test_all_csv_columns_present(self):
        row = mod.build_csv_row(_make_result())
        for col in mod.CSV_COLUMNS:
            assert col in row, f"Missing column: {col}"

    def test_classical_cv_default_engine(self):
        result = _make_result()
        result["landmarks"]["qa"].pop("inference_engine")
        row = mod.build_csv_row(result)
        assert row["inference_engine"] == "classical_cv"


# ===========================================================================
# draw_overlay
# ===========================================================================

class TestDrawOverlay:
    """Test draw_overlay function."""

    def _make_landmarks(self):
        return {
            "humerus_shaft": {"x": 128, "y": 50},
            "condyle_center": {"x": 128, "y": 128},
            "forearm_shaft": {"x": 128, "y": 200},
            "lateral_epicondyle": {"x": 100, "y": 128},
            "medial_epicondyle": {"x": 156, "y": 128},
            "radial_head": {"x": 110, "y": 160},
            "olecranon": {"x": 146, "y": 160},
            "angles": {
                "carrying_angle": 10.0,
                "flexion": 85.0,
            },
            "qa": {
                "view_type": "LAT",
                "inference_engine": "yolov8_pose",
                "score": 8,
                "status": "good",
            },
        }

    def _make_correction(self):
        return {
            "rotation_error": 5.0,
            "overall_level": "acceptable",
        }

    def test_returns_ndarray(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        result = mod.draw_overlay(img, self._make_landmarks(), self._make_correction())
        assert isinstance(result, np.ndarray)

    def test_preserves_shape(self):
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        result = mod.draw_overlay(img, self._make_landmarks(), self._make_correction())
        assert result.shape == (512, 512, 3)

    def test_preserves_dtype(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        result = mod.draw_overlay(img, self._make_landmarks(), self._make_correction())
        assert result.dtype == np.uint8

    def test_does_not_modify_original(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        original = img.copy()
        mod.draw_overlay(img, self._make_landmarks(), self._make_correction())
        np.testing.assert_array_equal(img, original)

    def test_output_differs_from_input(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        result = mod.draw_overlay(img, self._make_landmarks(), self._make_correction())
        assert not np.array_equal(img, result), "Overlay should modify the image"

    def test_handles_missing_landmarks(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        landmarks = {
            "angles": {},
            "qa": {"view_type": "?", "score": 0, "status": "?"},
        }
        correction = {"overall_level": "?"}
        result = mod.draw_overlay(img, landmarks, correction)
        assert result.shape == img.shape

    def test_handles_none_angles(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        landmarks = self._make_landmarks()
        landmarks["angles"]["carrying_angle"] = None
        landmarks["angles"]["flexion"] = None
        correction = {"rotation_error": None, "overall_level": "?"}
        result = mod.draw_overlay(img, landmarks, correction)
        assert result.shape == img.shape

    def test_works_with_large_image(self):
        img = np.zeros((1024, 1024, 3), dtype=np.uint8)
        result = mod.draw_overlay(img, self._make_landmarks(), self._make_correction())
        assert result.shape == (1024, 1024, 3)

    def test_works_with_small_image(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        landmarks = self._make_landmarks()
        # Scale landmarks to small image
        for k in ["humerus_shaft", "condyle_center", "forearm_shaft",
                   "lateral_epicondyle", "medial_epicondyle", "radial_head", "olecranon"]:
            if k in landmarks and isinstance(landmarks[k], dict):
                landmarks[k]["x"] = landmarks[k]["x"] // 4
                landmarks[k]["y"] = landmarks[k]["y"] // 4
        result = mod.draw_overlay(img, landmarks, self._make_correction())
        assert result.shape == (64, 64, 3)


# ===========================================================================
# print_summary
# ===========================================================================

class TestPrintSummary:
    """Test print_summary function."""

    def test_no_crash_with_empty_results(self, capsys):
        mod.print_summary([])
        captured = capsys.readouterr()
        assert "画像数: 0" in captured.out

    def test_displays_image_count(self, capsys):
        results = [_make_result() for _ in range(3)]
        mod.print_summary(results)
        captured = capsys.readouterr()
        assert "画像数: 3" in captured.out

    def test_displays_engine_info(self, capsys):
        results = [_make_result(engine="yolov8_pose")]
        mod.print_summary(results)
        captured = capsys.readouterr()
        assert "yolov8_pose" in captured.out

    def test_displays_carrying_angle_stats(self, capsys):
        results = [_make_result(carrying=10.0), _make_result(carrying=12.0)]
        mod.print_summary(results)
        captured = capsys.readouterr()
        assert "Carrying Angle" in captured.out

    def test_displays_flexion_stats(self, capsys):
        results = [_make_result(flexion=80.0), _make_result(flexion=90.0)]
        mod.print_summary(results)
        captured = capsys.readouterr()
        assert "Flexion" in captured.out

    def test_handles_none_angles(self, capsys):
        results = [_make_result(carrying=None, flexion=None, rotation_error=None)]
        # Patch rotation_error to None
        results[0]["correction"]["rotation_error"] = None
        mod.print_summary(results)
        captured = capsys.readouterr()
        assert "データなし" in captured.out

    def test_mixed_engines(self, capsys):
        results = [
            _make_result(engine="yolov8_pose"),
            _make_result(engine="classical_cv"),
        ]
        mod.print_summary(results)
        captured = capsys.readouterr()
        assert "yolov8_pose" in captured.out
        assert "classical_cv" in captured.out
