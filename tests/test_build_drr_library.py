"""Tests for scripts/build_drr_library.py

Covers:
  - Module constant TARGET_SIZE
  - preprocess_drr(): grayscale/BGR input, resizing to 256x256, CLAHE, dtype
  - build_library(): file output, .npz structure (angles/drrs/meta), correctness
    of stored values, directory creation, elbow_synth call counts
"""
from __future__ import annotations

import json
import os
import sys
import types
import unittest.mock as mock
from pathlib import Path

import cv2
import numpy as np
import pytest

# ── Mock elbow_synth before importing the module ──────────────────────────────

_mock_elbow_synth = types.ModuleType("elbow_synth")
_mock_elbow_synth.load_ct_volume = mock.MagicMock()
_mock_elbow_synth.auto_detect_landmarks = mock.MagicMock()
_mock_elbow_synth.rotate_volume_and_landmarks = mock.MagicMock()
_mock_elbow_synth.generate_drr = mock.MagicMock()
_saved_elbow_synth = sys.modules.get("elbow_synth")
sys.modules["elbow_synth"] = _mock_elbow_synth

# Make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import build_drr_library as bdl

# Restore original state so downstream test files can import the real elbow_synth
if _saved_elbow_synth is not None:
    sys.modules["elbow_synth"] = _saved_elbow_synth
else:
    del sys.modules["elbow_synth"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _setup_mocks(
    voxel_mm: float = 1.0,
    lat: str = "R",
    drr_shape: tuple = (256, 256),
) -> None:
    """Reset and configure all elbow_synth mocks."""
    _mock_elbow_synth.load_ct_volume.reset_mock()
    _mock_elbow_synth.auto_detect_landmarks.reset_mock()
    _mock_elbow_synth.rotate_volume_and_landmarks.reset_mock()
    _mock_elbow_synth.generate_drr.reset_mock()

    vol = np.zeros((64, 64, 64), dtype=np.float32)
    landmarks = {"joint_center": np.array([0.5, 0.5, 0.5])}

    _mock_elbow_synth.load_ct_volume.return_value = (vol, None, lat, voxel_mm)
    _mock_elbow_synth.auto_detect_landmarks.return_value = landmarks
    _mock_elbow_synth.rotate_volume_and_landmarks.return_value = (vol, landmarks)
    drr_img = np.random.randint(0, 256, drr_shape, dtype=np.uint8)
    _mock_elbow_synth.generate_drr.return_value = drr_img


def _build_small(tmp_path, **kwargs) -> Path:
    """Run build_library with 3-angle range (88–90°) by default."""
    defaults = dict(
        ct_dir="fake_ct_dir",
        laterality="R",
        series_num=4,
        hu_min=50.0,
        hu_max=800.0,
        base_flexion=180.0,
        angle_min=88.0,
        angle_max=90.0,
        angle_step=1.0,
    )
    defaults.update(kwargs)
    out_path = tmp_path / "lib.npz"
    bdl.build_library(out_path=str(out_path), **defaults)
    return out_path


# ── Module constants ───────────────────────────────────────────────────────────

class TestConstants:
    def test_target_size_value(self):
        assert bdl.TARGET_SIZE == 256

    def test_target_size_is_int(self):
        assert isinstance(bdl.TARGET_SIZE, int)

    def test_preprocess_drr_callable(self):
        assert callable(bdl.preprocess_drr)

    def test_build_library_callable(self):
        assert callable(bdl.build_library)


# ── preprocess_drr ─────────────────────────────────────────────────────────────

class TestPreprocessDrr:
    def test_grayscale_input_output_shape(self):
        img = np.zeros((256, 256), dtype=np.uint8)
        out = bdl.preprocess_drr(img)
        assert out.shape == (256, 256)

    def test_output_dtype_is_uint8(self):
        img = np.zeros((256, 256), dtype=np.uint8)
        out = bdl.preprocess_drr(img)
        assert out.dtype == np.uint8

    def test_bgr_input_converted_to_grayscale(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        out = bdl.preprocess_drr(img)
        assert out.ndim == 2

    def test_bgr_input_output_shape(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        out = bdl.preprocess_drr(img)
        assert out.shape == (256, 256)

    def test_bgr_input_output_dtype(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        out = bdl.preprocess_drr(img)
        assert out.dtype == np.uint8

    def test_smaller_input_upscaled(self):
        img = np.zeros((128, 128), dtype=np.uint8)
        out = bdl.preprocess_drr(img)
        assert out.shape == (256, 256)

    def test_larger_input_downscaled(self):
        img = np.zeros((512, 512), dtype=np.uint8)
        out = bdl.preprocess_drr(img)
        assert out.shape == (256, 256)

    def test_non_square_input_resized(self):
        img = np.zeros((100, 200), dtype=np.uint8)
        out = bdl.preprocess_drr(img)
        assert out.shape == (256, 256)

    def test_all_zero_input_returns_valid(self):
        img = np.zeros((256, 256), dtype=np.uint8)
        out = bdl.preprocess_drr(img)
        assert out is not None

    def test_all_max_input_returns_valid(self):
        img = np.full((256, 256), 255, dtype=np.uint8)
        out = bdl.preprocess_drr(img)
        assert out is not None

    def test_output_values_in_valid_range(self):
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (256, 256), dtype=np.uint8)
        out = bdl.preprocess_drr(img)
        assert int(out.min()) >= 0
        assert int(out.max()) <= 255

    def test_clahe_enhances_contrast(self):
        """CLAHE should increase dynamic range of a low-contrast image."""
        img = np.full((256, 256), 128, dtype=np.uint8)
        # Add subtle gradient so CLAHE has something to work with
        for i in range(256):
            img[i, :] = np.clip(128 + i // 16, 0, 255)
        img = img.astype(np.uint8)
        out = bdl.preprocess_drr(img)
        # CLAHE should spread the histogram → higher value range
        assert int(out.max()) - int(out.min()) >= int(img.max()) - int(img.min())

    def test_input_not_mutated(self):
        img = np.arange(256 * 256, dtype=np.uint8).reshape(256, 256)
        img_copy = img.copy()
        bdl.preprocess_drr(img)
        np.testing.assert_array_equal(img, img_copy)

    def test_returns_numpy_array(self):
        img = np.zeros((256, 256), dtype=np.uint8)
        out = bdl.preprocess_drr(img)
        assert isinstance(out, np.ndarray)

    @pytest.mark.parametrize("size", [64, 128, 256, 384, 512])
    def test_various_sizes_all_produce_256x256(self, size):
        rng = np.random.default_rng(size)
        img = rng.integers(0, 256, (size, size), dtype=np.uint8)
        out = bdl.preprocess_drr(img)
        assert out.shape == (256, 256)


# ── build_library ──────────────────────────────────────────────────────────────

class TestBuildLibrary:
    def test_creates_output_file(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path)
        assert out_path.exists()

    def test_npz_contains_angles_key(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path)
        data = np.load(str(out_path), allow_pickle=True)
        assert "angles" in data

    def test_npz_contains_drrs_key(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path)
        data = np.load(str(out_path), allow_pickle=True)
        assert "drrs" in data

    def test_npz_contains_meta_key(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path)
        data = np.load(str(out_path), allow_pickle=True)
        assert "meta" in data

    def test_angles_dtype_float32(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path)
        data = np.load(str(out_path), allow_pickle=True)
        assert data["angles"].dtype == np.float32

    def test_drrs_dtype_uint8(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path)
        data = np.load(str(out_path), allow_pickle=True)
        assert data["drrs"].dtype == np.uint8

    def test_drrs_shape_n_256_256(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path)
        data = np.load(str(out_path), allow_pickle=True)
        assert data["drrs"].ndim == 3
        assert data["drrs"].shape[1] == 256
        assert data["drrs"].shape[2] == 256

    def test_drrs_count_matches_angles(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path)
        data = np.load(str(out_path), allow_pickle=True)
        assert data["drrs"].shape[0] == len(data["angles"])

    def test_angles_count_three_for_88_to_90(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path)
        data = np.load(str(out_path), allow_pickle=True)
        # 88, 89, 90 → 3 angles
        assert len(data["angles"]) == 3

    def test_angles_count_six_for_60_to_65(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path, angle_min=60.0, angle_max=65.0, angle_step=1.0)
        data = np.load(str(out_path), allow_pickle=True)
        # 60, 61, 62, 63, 64, 65 → 6 angles
        assert len(data["angles"]) == 6

    def test_angles_values_correct(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path, angle_min=60.0, angle_max=62.0, angle_step=1.0)
        data = np.load(str(out_path), allow_pickle=True)
        np.testing.assert_allclose(data["angles"], [60.0, 61.0, 62.0], atol=0.01)

    def test_angle_step_5_degrees_count(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path, angle_min=60.0, angle_max=80.0, angle_step=5.0)
        data = np.load(str(out_path), allow_pickle=True)
        # 60, 65, 70, 75, 80 → 5 angles
        assert len(data["angles"]) == 5

    def test_meta_is_valid_json(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path)
        data = np.load(str(out_path), allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        assert isinstance(meta, dict)

    def test_meta_laterality_value(self, tmp_path):
        _setup_mocks(lat="R")
        out_path = _build_small(tmp_path, laterality="R")
        data = np.load(str(out_path), allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        assert meta["laterality"] == "R"

    def test_meta_series_num_value(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path, series_num=8)
        data = np.load(str(out_path), allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        assert meta["series_num"] == 8

    def test_meta_angle_min_value(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path, angle_min=60.0, angle_max=90.0)
        data = np.load(str(out_path), allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        assert meta["angle_min"] == 60.0

    def test_meta_angle_max_value(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path, angle_min=60.0, angle_max=90.0)
        data = np.load(str(out_path), allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        assert meta["angle_max"] == 90.0

    def test_meta_voxel_mm_value(self, tmp_path):
        _setup_mocks(voxel_mm=0.5)
        out_path = _build_small(tmp_path)
        data = np.load(str(out_path), allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        assert meta["voxel_mm"] == pytest.approx(0.5)

    def test_meta_hu_min_value(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path, hu_min=100.0)
        data = np.load(str(out_path), allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        assert meta["hu_min"] == 100.0

    def test_meta_hu_max_value(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path, hu_max=1200.0)
        data = np.load(str(out_path), allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        assert meta["hu_max"] == 1200.0

    def test_meta_target_size_is_256(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path)
        data = np.load(str(out_path), allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        assert meta["target_size"] == 256

    def test_meta_n_drrs_matches_actual(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path)
        data = np.load(str(out_path), allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        assert meta["n_drrs"] == len(data["angles"])

    def test_meta_base_flexion_stored(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path, base_flexion=135.0)
        data = np.load(str(out_path), allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        assert meta["base_flexion"] == 135.0

    def test_meta_angle_step_stored(self, tmp_path):
        _setup_mocks()
        out_path = _build_small(tmp_path, angle_step=2.0,
                                 angle_min=80.0, angle_max=90.0)
        data = np.load(str(out_path), allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        assert meta["angle_step"] == 2.0

    def test_creates_parent_directory(self, tmp_path):
        _setup_mocks()
        out_path = tmp_path / "subdir" / "deeper" / "lib.npz"
        bdl.build_library(
            ct_dir="fake", laterality="R", series_num=4,
            hu_min=50.0, hu_max=800.0, base_flexion=180.0,
            angle_min=88.0, angle_max=90.0, angle_step=1.0,
            out_path=str(out_path),
        )
        assert out_path.exists()

    def test_rotate_called_for_each_angle(self, tmp_path):
        _setup_mocks()
        _build_small(tmp_path, angle_min=88.0, angle_max=90.0, angle_step=1.0)
        # 88, 89, 90 → 3 angles → rotate called 3 times
        assert _mock_elbow_synth.rotate_volume_and_landmarks.call_count == 3

    def test_generate_drr_called_for_each_angle(self, tmp_path):
        _setup_mocks()
        _build_small(tmp_path, angle_min=88.0, angle_max=90.0, angle_step=1.0)
        assert _mock_elbow_synth.generate_drr.call_count == 3

    def test_load_ct_volume_called_once(self, tmp_path):
        _setup_mocks()
        _build_small(tmp_path)
        assert _mock_elbow_synth.load_ct_volume.call_count == 1

    def test_auto_detect_landmarks_called_once(self, tmp_path):
        _setup_mocks()
        _build_small(tmp_path)
        assert _mock_elbow_synth.auto_detect_landmarks.call_count == 1

    def test_drr_axis_is_lat(self, tmp_path):
        """generate_drr should always be called with axis='LAT'."""
        _setup_mocks()
        _build_small(tmp_path)
        for call in _mock_elbow_synth.generate_drr.call_args_list:
            assert call.kwargs.get("axis") == "LAT" or call.args[1] == "LAT"
