"""Tests for scripts/batch_multi_patient.py

Covers:
  - Module constants (N_WORKERS, TARGET_SIZE, ANGLE_MIN/MAX, VAL_RATIO, SID_MM)
  - PatientConfig dataclass defaults and field types
  - _apply_domain_aug(): shape, dtype, value range, determinism under seed
  - build_pooled_dataset(): CSV creation, train/val split ratio, column schema,
    empty-dataset handling, deterministic count totals
"""
from __future__ import annotations

import csv
import os
import random
import sys
import types
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

# ── Mock elbow_synth before importing the module ──────────────────────────────

_mock_elbow_synth = types.ModuleType("elbow_synth")
_mock_elbow_synth.load_ct_volume = mock.MagicMock()
_mock_elbow_synth.auto_detect_landmarks = mock.MagicMock()
_mock_elbow_synth.rotate_volume_and_landmarks = mock.MagicMock()
_mock_elbow_synth.generate_drr = mock.MagicMock()
sys.modules["elbow_synth"] = _mock_elbow_synth

# Make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import batch_multi_patient as bmp

# Remove mock so downstream test files (e.g. test_build_drr_library.py) can
# register their own elbow_synth mock without interference.
del sys.modules["elbow_synth"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_patient_dir(base: Path, patient_id: str, rows: list[dict]) -> Path:
    """Create a patient directory with a labels.csv file."""
    patient_dir = base / "patients" / patient_id
    img_dir = patient_dir / "images"
    img_dir.mkdir(parents=True)

    label_csv = patient_dir / "labels.csv"
    if rows:
        with open(label_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "angle_deg", "patient_id"])
            w.writeheader()
            w.writerows(rows)
    return patient_dir


def _make_gray_image(h: int = 64, w: int = 64, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_n_workers_is_positive_int(self):
        assert isinstance(bmp.N_WORKERS, int)
        assert bmp.N_WORKERS > 0

    def test_n_workers_value(self):
        assert bmp.N_WORKERS == 10

    def test_target_size_is_256(self):
        assert bmp.TARGET_SIZE == 256

    def test_sid_mm_is_1000(self):
        assert bmp.SID_MM == 1000.0

    def test_val_ratio_range(self):
        assert 0.0 < bmp.VAL_RATIO < 1.0

    def test_val_ratio_value(self):
        assert bmp.VAL_RATIO == pytest.approx(0.15)

    def test_angle_min_value(self):
        assert bmp.ANGLE_MIN == pytest.approx(90.0)

    def test_angle_max_value(self):
        assert bmp.ANGLE_MAX == pytest.approx(180.0)

    def test_angle_min_less_than_angle_max(self):
        assert bmp.ANGLE_MIN < bmp.ANGLE_MAX

    def test_sample_csv_content_has_header(self):
        assert "patient_id" in bmp.SAMPLE_CSV_CONTENT
        assert "ct_dir" in bmp.SAMPLE_CSV_CONTENT
        assert "laterality" in bmp.SAMPLE_CSV_CONTENT


# ══════════════════════════════════════════════════════════════════════════════
# PatientConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestPatientConfig:
    def test_required_fields(self):
        cfg = bmp.PatientConfig(patient_id="P001", ct_dir="/data/P001")
        assert cfg.patient_id == "P001"
        assert cfg.ct_dir == "/data/P001"

    def test_laterality_defaults_none(self):
        cfg = bmp.PatientConfig(patient_id="P001", ct_dir="/data/P001")
        assert cfg.laterality is None

    def test_series_num_defaults_none(self):
        cfg = bmp.PatientConfig(patient_id="P001", ct_dir="/data/P001")
        assert cfg.series_num is None

    def test_hu_min_default(self):
        cfg = bmp.PatientConfig(patient_id="P001", ct_dir="/data/P001")
        assert cfg.hu_min == pytest.approx(-400.0)

    def test_hu_max_default(self):
        cfg = bmp.PatientConfig(patient_id="P001", ct_dir="/data/P001")
        assert cfg.hu_max == pytest.approx(1500.0)

    def test_override_all_fields(self):
        cfg = bmp.PatientConfig(
            patient_id="P002",
            ct_dir="/data/P002",
            laterality="L",
            series_num=4,
            hu_min=50.0,
            hu_max=800.0,
        )
        assert cfg.laterality == "L"
        assert cfg.series_num == 4
        assert cfg.hu_min == pytest.approx(50.0)
        assert cfg.hu_max == pytest.approx(800.0)

    def test_patient_id_stored_as_string(self):
        cfg = bmp.PatientConfig(patient_id="P001", ct_dir="/data")
        assert isinstance(cfg.patient_id, str)

    def test_series_num_int_or_none(self):
        cfg = bmp.PatientConfig(patient_id="P001", ct_dir="/data", series_num=8)
        assert cfg.series_num == 8
        assert isinstance(cfg.series_num, int)


# ══════════════════════════════════════════════════════════════════════════════
# _apply_domain_aug
# ══════════════════════════════════════════════════════════════════════════════

class TestApplyDomainAug:
    def test_output_shape_matches_input_square(self):
        img = _make_gray_image(64, 64)
        out = bmp._apply_domain_aug(img)
        assert out.shape == (64, 64)

    def test_output_shape_matches_input_rect(self):
        img = _make_gray_image(128, 256)
        out = bmp._apply_domain_aug(img)
        assert out.shape == (128, 256)

    def test_output_dtype_is_uint8(self):
        img = _make_gray_image()
        out = bmp._apply_domain_aug(img)
        assert out.dtype == np.uint8

    def test_output_values_in_valid_range(self):
        img = _make_gray_image()
        out = bmp._apply_domain_aug(img)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_all_zeros_input_stays_uint8(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        out = bmp._apply_domain_aug(img)
        assert out.dtype == np.uint8
        assert out.shape == (32, 32)

    def test_all_255_input_clipped_to_uint8(self):
        img = np.full((32, 32), 255, dtype=np.uint8)
        out = bmp._apply_domain_aug(img)
        assert out.max() <= 255
        assert out.dtype == np.uint8

    def test_does_not_modify_input_array(self):
        img = _make_gray_image()
        original = img.copy()
        bmp._apply_domain_aug(img)
        np.testing.assert_array_equal(img, original)

    def test_output_is_different_from_input(self):
        """Augmentation should produce a different image (with high probability)."""
        random.seed(0)
        np.random.seed(0)
        img = _make_gray_image(64, 64, seed=7)
        out = bmp._apply_domain_aug(img)
        # At minimum, noise is always added — arrays should differ
        assert not np.array_equal(img, out)

    def test_deterministic_under_same_seed(self):
        img = _make_gray_image(32, 32)
        random.seed(123)
        np.random.seed(123)
        out1 = bmp._apply_domain_aug(img.copy())
        random.seed(123)
        np.random.seed(123)
        out2 = bmp._apply_domain_aug(img.copy())
        np.testing.assert_array_equal(out1, out2)

    def test_two_different_seeds_usually_differ(self):
        img = _make_gray_image(64, 64)
        random.seed(1)
        np.random.seed(1)
        out1 = bmp._apply_domain_aug(img.copy())
        random.seed(999)
        np.random.seed(999)
        out2 = bmp._apply_domain_aug(img.copy())
        # Very unlikely to be identical with different seeds
        assert not np.array_equal(out1, out2)

    def test_float32_input_accepted(self):
        img = _make_gray_image().astype(np.float32)
        out = bmp._apply_domain_aug(img)
        assert out.dtype == np.uint8

    def test_single_pixel_image(self):
        img = np.array([[128]], dtype=np.uint8)
        out = bmp._apply_domain_aug(img)
        assert out.shape == (1, 1)
        assert out.dtype == np.uint8

    def test_large_image_performance(self):
        """256×256 image should complete without error."""
        img = _make_gray_image(256, 256)
        out = bmp._apply_domain_aug(img)
        assert out.shape == (256, 256)


# ══════════════════════════════════════════════════════════════════════════════
# build_pooled_dataset
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildPooledDataset:
    def _make_rows(self, patient_id: str, n: int) -> list[dict]:
        return [
            {
                "filename": f"angle{i:03d}_aug00_rot.png",
                "angle_deg": 90.0 + i,
                "patient_id": patient_id,
            }
            for i in range(n)
        ]

    def test_creates_pooled_directory(self, tmp_path):
        _make_patient_dir(tmp_path, "P001", self._make_rows("P001", 20))
        bmp.build_pooled_dataset(tmp_path, val_ratio=0.2)
        assert (tmp_path / "pooled").exists()

    def test_creates_train_csv(self, tmp_path):
        _make_patient_dir(tmp_path, "P001", self._make_rows("P001", 20))
        bmp.build_pooled_dataset(tmp_path, val_ratio=0.2)
        assert (tmp_path / "pooled" / "train.csv").exists()

    def test_creates_val_csv(self, tmp_path):
        _make_patient_dir(tmp_path, "P001", self._make_rows("P001", 20))
        bmp.build_pooled_dataset(tmp_path, val_ratio=0.2)
        assert (tmp_path / "pooled" / "val.csv").exists()

    def test_total_rows_equals_input(self, tmp_path):
        n = 40
        _make_patient_dir(tmp_path, "P001", self._make_rows("P001", n))
        bmp.build_pooled_dataset(tmp_path, val_ratio=0.25)

        with open(tmp_path / "pooled" / "train.csv") as f:
            train_n = sum(1 for _ in csv.DictReader(f))
        with open(tmp_path / "pooled" / "val.csv") as f:
            val_n = sum(1 for _ in csv.DictReader(f))

        assert train_n + val_n == n

    def test_val_ratio_respected(self, tmp_path):
        n = 100
        val_ratio = 0.2
        _make_patient_dir(tmp_path, "P001", self._make_rows("P001", n))
        bmp.build_pooled_dataset(tmp_path, val_ratio=val_ratio)

        with open(tmp_path / "pooled" / "val.csv") as f:
            val_n = sum(1 for _ in csv.DictReader(f))

        expected_val = int(n * val_ratio)
        assert val_n == expected_val

    def test_train_csv_has_required_columns(self, tmp_path):
        _make_patient_dir(tmp_path, "P001", self._make_rows("P001", 10))
        bmp.build_pooled_dataset(tmp_path, val_ratio=0.2)

        with open(tmp_path / "pooled" / "train.csv") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

        assert "img_path" in fieldnames
        assert "angle_deg" in fieldnames
        assert "patient_id" in fieldnames
        assert "filename" in fieldnames

    def test_val_csv_has_required_columns(self, tmp_path):
        _make_patient_dir(tmp_path, "P001", self._make_rows("P001", 10))
        bmp.build_pooled_dataset(tmp_path, val_ratio=0.2)

        with open(tmp_path / "pooled" / "val.csv") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

        assert "img_path" in fieldnames
        assert "angle_deg" in fieldnames

    def test_img_path_contains_patient_dir(self, tmp_path):
        _make_patient_dir(tmp_path, "P001", self._make_rows("P001", 10))
        bmp.build_pooled_dataset(tmp_path, val_ratio=0.0)

        with open(tmp_path / "pooled" / "train.csv") as f:
            rows = list(csv.DictReader(f))

        assert all("P001" in r["img_path"] for r in rows)

    def test_multiple_patients_pooled(self, tmp_path):
        for pid in ["P001", "P002", "P003"]:
            _make_patient_dir(tmp_path, pid, self._make_rows(pid, 20))
        bmp.build_pooled_dataset(tmp_path, val_ratio=0.15)

        with open(tmp_path / "pooled" / "train.csv") as f:
            train_n = sum(1 for _ in csv.DictReader(f))
        with open(tmp_path / "pooled" / "val.csv") as f:
            val_n = sum(1 for _ in csv.DictReader(f))

        assert train_n + val_n == 60

    def test_patient_ids_from_multiple_patients_present(self, tmp_path):
        for pid in ["P001", "P002"]:
            _make_patient_dir(tmp_path, pid, self._make_rows(pid, 20))
        bmp.build_pooled_dataset(tmp_path, val_ratio=0.0)

        with open(tmp_path / "pooled" / "train.csv") as f:
            pids = {r["patient_id"] for r in csv.DictReader(f)}

        assert "P001" in pids
        assert "P002" in pids

    def test_empty_dataset_no_crash(self, tmp_path, capsys):
        (tmp_path / "patients").mkdir()
        bmp.build_pooled_dataset(tmp_path)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_empty_dataset_no_output_files(self, tmp_path):
        (tmp_path / "patients").mkdir()
        bmp.build_pooled_dataset(tmp_path)
        assert not (tmp_path / "pooled" / "train.csv").exists()

    def test_patient_dir_without_labels_csv_skipped(self, tmp_path):
        # Create patient dir with no labels.csv
        (tmp_path / "patients" / "P_EMPTY" / "images").mkdir(parents=True)
        # Also create one valid patient
        _make_patient_dir(tmp_path, "P001", self._make_rows("P001", 10))
        bmp.build_pooled_dataset(tmp_path, val_ratio=0.0)

        with open(tmp_path / "pooled" / "train.csv") as f:
            rows = list(csv.DictReader(f))

        # Only P001 rows should appear
        assert len(rows) == 10
        assert all(r["patient_id"] == "P001" for r in rows)

    def test_val_ratio_zero_all_train(self, tmp_path):
        n = 30
        _make_patient_dir(tmp_path, "P001", self._make_rows("P001", n))
        bmp.build_pooled_dataset(tmp_path, val_ratio=0.0)

        with open(tmp_path / "pooled" / "train.csv") as f:
            train_n = sum(1 for _ in csv.DictReader(f))
        with open(tmp_path / "pooled" / "val.csv") as f:
            val_n = sum(1 for _ in csv.DictReader(f))

        assert val_n == 0
        assert train_n == n

    def test_angle_deg_preserved_in_output(self, tmp_path):
        rows = [{"filename": "f.png", "angle_deg": 123.456, "patient_id": "P001"}]
        _make_patient_dir(tmp_path, "P001", rows)
        bmp.build_pooled_dataset(tmp_path, val_ratio=0.0)

        with open(tmp_path / "pooled" / "train.csv") as f:
            out_rows = list(csv.DictReader(f))

        assert float(out_rows[0]["angle_deg"]) == pytest.approx(123.456)
