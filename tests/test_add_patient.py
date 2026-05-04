"""Tests for scripts/add_patient.py

Covers:
  - Module constants (CSV_FIELDS)
  - _python(): returns venv path when exists, else 'python3'
  - append_to_csv(): creates CSV with header, appends rows, handles empty list
  - deduplicate_csv(): removes duplicate (patient_id, xray_path) pairs, preserves order
  - build_library(): calls subprocess with correct args, skips when library exists
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import types
import unittest.mock as mock
from pathlib import Path

import pytest

# ── Mock elbow_synth before importing ─────────────────────────────────────────

_mock_elbow_synth = types.ModuleType("elbow_synth")
_saved_elbow_synth = sys.modules.get("elbow_synth")
sys.modules["elbow_synth"] = _mock_elbow_synth

# Make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import add_patient as ap

# Restore original state so downstream test files can import the real elbow_synth
if _saved_elbow_synth is not None:
    sys.modules["elbow_synth"] = _saved_elbow_synth
else:
    sys.modules.pop("elbow_synth", None)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstants:
    """Verify module-level constants."""

    def test_csv_fields_is_list(self):
        assert isinstance(ap.CSV_FIELDS, list)

    def test_csv_fields_length(self):
        assert len(ap.CSV_FIELDS) == 10

    def test_csv_fields_contains_patient_id(self):
        assert "patient_id" in ap.CSV_FIELDS

    def test_csv_fields_contains_xray_path(self):
        assert "xray_path" in ap.CSV_FIELDS

    def test_csv_fields_contains_gt_angle(self):
        assert "gt_angle_deg" in ap.CSV_FIELDS

    def test_csv_fields_contains_library_path(self):
        assert "library_path" in ap.CSV_FIELDS

    def test_csv_fields_contains_ct_dir(self):
        assert "ct_dir" in ap.CSV_FIELDS

    def test_csv_fields_contains_laterality(self):
        assert "laterality" in ap.CSV_FIELDS

    def test_csv_fields_contains_series_num(self):
        assert "series_num" in ap.CSV_FIELDS

    def test_csv_fields_contains_hu_min(self):
        assert "hu_min" in ap.CSV_FIELDS

    def test_csv_fields_contains_hu_max(self):
        assert "hu_max" in ap.CSV_FIELDS

    def test_csv_fields_contains_note(self):
        assert "note" in ap.CSV_FIELDS

    def test_csv_fields_no_duplicates(self):
        assert len(ap.CSV_FIELDS) == len(set(ap.CSV_FIELDS))


# ═══════════════════════════════════════════════════════════════════════════════
# _python()
# ═══════════════════════════════════════════════════════════════════════════════


class TestPythonHelper:
    """Test _python() venv detection."""

    def test_returns_string(self):
        result = ap._python()
        assert isinstance(result, str)

    def test_returns_venv_when_exists(self):
        """If _VENV_PYTHON exists, _python() returns its string path."""
        original = ap._VENV_PYTHON
        try:
            # Point to a file that definitely exists
            ap._VENV_PYTHON = Path(sys.executable)
            result = ap._python()
            assert result == str(Path(sys.executable))
        finally:
            ap._VENV_PYTHON = original

    def test_returns_python3_when_no_venv(self):
        """If _VENV_PYTHON doesn't exist, _python() returns 'python3'."""
        original = ap._VENV_PYTHON
        try:
            ap._VENV_PYTHON = Path("/nonexistent/venv/bin/python3")
            result = ap._python()
            assert result == "python3"
        finally:
            ap._VENV_PYTHON = original


# ═══════════════════════════════════════════════════════════════════════════════
# append_to_csv()
# ═══════════════════════════════════════════════════════════════════════════════


class TestAppendToCsv:
    """Test CSV append functionality."""

    def _make_row(self, **overrides):
        """Create a minimal valid row dict."""
        base = {
            "patient_id": "test001",
            "ct_dir": "data/raw_dicom/ct",
            "xray_path": "data/real_xray/images/test.png",
            "gt_angle_deg": 90.0,
            "laterality": "R",
            "series_num": 4,
            "hu_min": 50.0,
            "hu_max": 800.0,
            "library_path": "data/drr_library/test.npz",
            "note": "",
        }
        base.update(overrides)
        return base

    def test_creates_new_csv_with_header(self, tmp_path):
        csv_path = tmp_path / "patients.csv"
        original = ap._CSV_PATH
        try:
            ap._CSV_PATH = csv_path
            ap.append_to_csv([self._make_row()])
            assert csv_path.exists()
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                assert set(reader.fieldnames) == set(ap.CSV_FIELDS)
        finally:
            ap._CSV_PATH = original

    def test_appends_single_row(self, tmp_path):
        csv_path = tmp_path / "patients.csv"
        original = ap._CSV_PATH
        try:
            ap._CSV_PATH = csv_path
            ap.append_to_csv([self._make_row(patient_id="p001")])
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 1
            assert rows[0]["patient_id"] == "p001"
        finally:
            ap._CSV_PATH = original

    def test_appends_multiple_rows(self, tmp_path):
        csv_path = tmp_path / "patients.csv"
        original = ap._CSV_PATH
        try:
            ap._CSV_PATH = csv_path
            rows_in = [
                self._make_row(patient_id="p001", xray_path="a.png"),
                self._make_row(patient_id="p001", xray_path="b.png"),
            ]
            ap.append_to_csv(rows_in)
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 2
        finally:
            ap._CSV_PATH = original

    def test_appends_to_existing_file(self, tmp_path):
        csv_path = tmp_path / "patients.csv"
        original = ap._CSV_PATH
        try:
            ap._CSV_PATH = csv_path
            ap.append_to_csv([self._make_row(patient_id="p001")])
            ap.append_to_csv([self._make_row(patient_id="p002")])
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 2
            assert rows[0]["patient_id"] == "p001"
            assert rows[1]["patient_id"] == "p002"
        finally:
            ap._CSV_PATH = original

    def test_empty_rows_does_not_create_file(self, tmp_path):
        """Appending empty list still creates the file (with header if new)."""
        csv_path = tmp_path / "patients.csv"
        original = ap._CSV_PATH
        try:
            ap._CSV_PATH = csv_path
            ap.append_to_csv([])
            # File is created with header only
            assert csv_path.exists()
            with open(csv_path) as f:
                lines = f.read().strip().split("\n")
            assert len(lines) == 1  # header only
        finally:
            ap._CSV_PATH = original

    def test_preserves_all_fields(self, tmp_path):
        csv_path = tmp_path / "patients.csv"
        original = ap._CSV_PATH
        try:
            ap._CSV_PATH = csv_path
            row = self._make_row(
                patient_id="px",
                gt_angle_deg=123.5,
                laterality="L",
                note="test note",
            )
            ap.append_to_csv([row])
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            assert rows[0]["patient_id"] == "px"
            assert rows[0]["gt_angle_deg"] == "123.5"
            assert rows[0]["laterality"] == "L"
            assert rows[0]["note"] == "test note"
        finally:
            ap._CSV_PATH = original

    def test_extra_fields_ignored(self, tmp_path):
        """Extra keys in the row dict are ignored (extrasaction='ignore')."""
        csv_path = tmp_path / "patients.csv"
        original = ap._CSV_PATH
        try:
            ap._CSV_PATH = csv_path
            row = self._make_row(extra_field="should_be_ignored")
            ap.append_to_csv([row])
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                assert "extra_field" not in reader.fieldnames
        finally:
            ap._CSV_PATH = original


# ═══════════════════════════════════════════════════════════════════════════════
# deduplicate_csv()
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeduplicateCsv:
    """Test CSV deduplication by (patient_id, xray_path)."""

    def _write_csv(self, path: Path, rows: list[dict]):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ap.CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def _make_row(self, pid="p001", xray="a.png", gt=90.0, **kw):
        base = {
            "patient_id": pid,
            "ct_dir": "data/ct",
            "xray_path": xray,
            "gt_angle_deg": gt,
            "laterality": "R",
            "series_num": 4,
            "hu_min": 50,
            "hu_max": 800,
            "library_path": "data/lib.npz",
            "note": "",
        }
        base.update(kw)
        return base

    def test_no_duplicates_unchanged(self, tmp_path):
        csv_path = tmp_path / "p.csv"
        original = ap._CSV_PATH
        try:
            ap._CSV_PATH = csv_path
            rows = [
                self._make_row("p1", "a.png"),
                self._make_row("p2", "b.png"),
            ]
            self._write_csv(csv_path, rows)
            ap.deduplicate_csv()
            with open(csv_path) as f:
                result = list(csv.DictReader(f))
            assert len(result) == 2
        finally:
            ap._CSV_PATH = original

    def test_removes_exact_duplicates(self, tmp_path):
        csv_path = tmp_path / "p.csv"
        original = ap._CSV_PATH
        try:
            ap._CSV_PATH = csv_path
            row = self._make_row("p1", "a.png")
            self._write_csv(csv_path, [row, row])
            ap.deduplicate_csv()
            with open(csv_path) as f:
                result = list(csv.DictReader(f))
            assert len(result) == 1
        finally:
            ap._CSV_PATH = original

    def test_preserves_first_occurrence(self, tmp_path):
        csv_path = tmp_path / "p.csv"
        original = ap._CSV_PATH
        try:
            ap._CSV_PATH = csv_path
            rows = [
                self._make_row("p1", "a.png", note="first"),
                self._make_row("p1", "a.png", note="second"),
            ]
            self._write_csv(csv_path, rows)
            ap.deduplicate_csv()
            with open(csv_path) as f:
                result = list(csv.DictReader(f))
            assert len(result) == 1
            assert result[0]["note"] == "first"
        finally:
            ap._CSV_PATH = original

    def test_different_xray_same_patient_kept(self, tmp_path):
        """Same patient_id but different xray_path are kept."""
        csv_path = tmp_path / "p.csv"
        original = ap._CSV_PATH
        try:
            ap._CSV_PATH = csv_path
            rows = [
                self._make_row("p1", "a.png"),
                self._make_row("p1", "b.png"),
            ]
            self._write_csv(csv_path, rows)
            ap.deduplicate_csv()
            with open(csv_path) as f:
                result = list(csv.DictReader(f))
            assert len(result) == 2
        finally:
            ap._CSV_PATH = original

    def test_same_xray_different_patient_kept(self, tmp_path):
        """Same xray_path but different patient_id are kept."""
        csv_path = tmp_path / "p.csv"
        original = ap._CSV_PATH
        try:
            ap._CSV_PATH = csv_path
            rows = [
                self._make_row("p1", "a.png"),
                self._make_row("p2", "a.png"),
            ]
            self._write_csv(csv_path, rows)
            ap.deduplicate_csv()
            with open(csv_path) as f:
                result = list(csv.DictReader(f))
            assert len(result) == 2
        finally:
            ap._CSV_PATH = original

    def test_multiple_duplicates(self, tmp_path):
        csv_path = tmp_path / "p.csv"
        original = ap._CSV_PATH
        try:
            ap._CSV_PATH = csv_path
            rows = [
                self._make_row("p1", "a.png"),
                self._make_row("p1", "a.png"),
                self._make_row("p2", "b.png"),
                self._make_row("p2", "b.png"),
                self._make_row("p2", "b.png"),
            ]
            self._write_csv(csv_path, rows)
            ap.deduplicate_csv()
            with open(csv_path) as f:
                result = list(csv.DictReader(f))
            assert len(result) == 2
        finally:
            ap._CSV_PATH = original

    def test_preserves_order(self, tmp_path):
        csv_path = tmp_path / "p.csv"
        original = ap._CSV_PATH
        try:
            ap._CSV_PATH = csv_path
            rows = [
                self._make_row("p3", "c.png"),
                self._make_row("p1", "a.png"),
                self._make_row("p2", "b.png"),
                self._make_row("p1", "a.png"),
            ]
            self._write_csv(csv_path, rows)
            ap.deduplicate_csv()
            with open(csv_path) as f:
                result = list(csv.DictReader(f))
            assert [r["patient_id"] for r in result] == ["p3", "p1", "p2"]
        finally:
            ap._CSV_PATH = original

    def test_nonexistent_file_no_error(self, tmp_path):
        original = ap._CSV_PATH
        try:
            ap._CSV_PATH = tmp_path / "nonexistent.csv"
            ap.deduplicate_csv()  # should not raise
        finally:
            ap._CSV_PATH = original

    def test_empty_csv_only_header(self, tmp_path):
        csv_path = tmp_path / "p.csv"
        original = ap._CSV_PATH
        try:
            ap._CSV_PATH = csv_path
            self._write_csv(csv_path, [])
            ap.deduplicate_csv()
            with open(csv_path) as f:
                result = list(csv.DictReader(f))
            assert len(result) == 0
        finally:
            ap._CSV_PATH = original

    def test_header_preserved_after_dedup(self, tmp_path):
        csv_path = tmp_path / "p.csv"
        original = ap._CSV_PATH
        try:
            ap._CSV_PATH = csv_path
            self._write_csv(csv_path, [self._make_row()])
            ap.deduplicate_csv()
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                assert set(reader.fieldnames) == set(ap.CSV_FIELDS)
        finally:
            ap._CSV_PATH = original


# ═══════════════════════════════════════════════════════════════════════════════
# build_library()
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildLibrary:
    """Test DRR library building logic."""

    def test_skips_existing_library(self, tmp_path):
        lib_dir = tmp_path / "drr_library"
        lib_dir.mkdir()
        expected_name = "test001_series4_R_60to180.npz"
        (lib_dir / expected_name).touch()

        original = ap._LIBRARY_DIR
        try:
            ap._LIBRARY_DIR = lib_dir
            result = ap.build_library(
                patient_id="test001",
                ct_dir="/some/ct",
                laterality="R",
                series_num=4,
                hu_min=50.0,
                hu_max=800.0,
                angle_min=60.0,
                angle_max=180.0,
            )
            assert result == lib_dir / expected_name
        finally:
            ap._LIBRARY_DIR = original

    def test_library_name_format(self, tmp_path):
        """Verify the generated library filename pattern."""
        lib_dir = tmp_path / "drr_library"
        lib_dir.mkdir()
        # Pre-create the file so it skips subprocess
        expected = "p009_series8_L_70to170.npz"
        (lib_dir / expected).touch()

        original = ap._LIBRARY_DIR
        try:
            ap._LIBRARY_DIR = lib_dir
            result = ap.build_library(
                patient_id="p009",
                ct_dir="/ct",
                laterality="L",
                series_num=8,
                hu_min=50,
                hu_max=800,
                angle_min=70,
                angle_max=170,
            )
            assert result.name == expected
        finally:
            ap._LIBRARY_DIR = original

    @mock.patch("add_patient.subprocess.run")
    def test_calls_subprocess_when_missing(self, mock_run, tmp_path):
        lib_dir = tmp_path / "drr_library"
        lib_dir.mkdir()
        mock_run.return_value = mock.MagicMock(returncode=0)

        original_lib = ap._LIBRARY_DIR
        try:
            ap._LIBRARY_DIR = lib_dir
            result = ap.build_library(
                patient_id="newp",
                ct_dir="/some/ct",
                laterality="R",
                series_num=4,
                hu_min=50.0,
                hu_max=800.0,
                angle_min=60.0,
                angle_max=180.0,
            )
            assert mock_run.called
            call_args = mock_run.call_args[0][0]
            assert "--ct_dir" in call_args
            assert "--out_path" in call_args
            assert "--laterality" in call_args
            assert "--series_num" in call_args
        finally:
            ap._LIBRARY_DIR = original_lib

    @mock.patch("add_patient.subprocess.run")
    def test_subprocess_failure_exits(self, mock_run, tmp_path):
        lib_dir = tmp_path / "drr_library"
        lib_dir.mkdir()
        mock_run.return_value = mock.MagicMock(returncode=1)

        original_lib = ap._LIBRARY_DIR
        try:
            ap._LIBRARY_DIR = lib_dir
            with pytest.raises(SystemExit):
                ap.build_library(
                    patient_id="failp",
                    ct_dir="/some/ct",
                    laterality="R",
                    series_num=4,
                    hu_min=50.0,
                    hu_max=800.0,
                    angle_min=60.0,
                    angle_max=180.0,
                )
        finally:
            ap._LIBRARY_DIR = original_lib

    @mock.patch("add_patient.subprocess.run")
    def test_subprocess_args_include_angle_step(self, mock_run, tmp_path):
        """build_library passes --angle_step 1 to build_drr_library.py."""
        lib_dir = tmp_path / "drr_library"
        lib_dir.mkdir()
        mock_run.return_value = mock.MagicMock(returncode=0)

        original_lib = ap._LIBRARY_DIR
        try:
            ap._LIBRARY_DIR = lib_dir
            ap.build_library(
                patient_id="stepp",
                ct_dir="/ct",
                laterality="R",
                series_num=4,
                hu_min=50,
                hu_max=800,
                angle_min=60,
                angle_max=180,
            )
            call_args = mock_run.call_args[0][0]
            assert "--angle_step" in call_args
            idx = call_args.index("--angle_step")
            assert call_args[idx + 1] == "1"
        finally:
            ap._LIBRARY_DIR = original_lib

    def test_returns_path_object(self, tmp_path):
        lib_dir = tmp_path / "drr_library"
        lib_dir.mkdir()
        (lib_dir / "ret_series4_R_60to180.npz").touch()

        original = ap._LIBRARY_DIR
        try:
            ap._LIBRARY_DIR = lib_dir
            result = ap.build_library(
                patient_id="ret",
                ct_dir="/ct",
                laterality="R",
                series_num=4,
                hu_min=50,
                hu_max=800,
                angle_min=60,
                angle_max=180,
            )
            assert isinstance(result, Path)
        finally:
            ap._LIBRARY_DIR = original


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: append_to_csv + deduplicate_csv
# ═══════════════════════════════════════════════════════════════════════════════


class TestAppendAndDedup:
    """Test the combined workflow of appending then deduplicating."""

    def _make_row(self, pid="p001", xray="a.png", **kw):
        base = {
            "patient_id": pid,
            "ct_dir": "data/ct",
            "xray_path": xray,
            "gt_angle_deg": 90.0,
            "laterality": "R",
            "series_num": 4,
            "hu_min": 50,
            "hu_max": 800,
            "library_path": "data/lib.npz",
            "note": "",
        }
        base.update(kw)
        return base

    def test_append_twice_then_dedup(self, tmp_path):
        csv_path = tmp_path / "p.csv"
        original = ap._CSV_PATH
        try:
            ap._CSV_PATH = csv_path
            ap.append_to_csv([self._make_row("p1", "a.png")])
            ap.append_to_csv([self._make_row("p1", "a.png")])
            ap.deduplicate_csv()
            with open(csv_path) as f:
                result = list(csv.DictReader(f))
            assert len(result) == 1
        finally:
            ap._CSV_PATH = original

    def test_append_different_then_dedup_keeps_all(self, tmp_path):
        csv_path = tmp_path / "p.csv"
        original = ap._CSV_PATH
        try:
            ap._CSV_PATH = csv_path
            ap.append_to_csv([self._make_row("p1", "a.png")])
            ap.append_to_csv([self._make_row("p2", "b.png")])
            ap.deduplicate_csv()
            with open(csv_path) as f:
                result = list(csv.DictReader(f))
            assert len(result) == 2
        finally:
            ap._CSV_PATH = original

    def test_dedup_idempotent(self, tmp_path):
        csv_path = tmp_path / "p.csv"
        original = ap._CSV_PATH
        try:
            ap._CSV_PATH = csv_path
            ap.append_to_csv([
                self._make_row("p1", "a.png"),
                self._make_row("p1", "a.png"),
            ])
            ap.deduplicate_csv()
            ap.deduplicate_csv()  # second call should be no-op
            with open(csv_path) as f:
                result = list(csv.DictReader(f))
            assert len(result) == 1
        finally:
            ap._CSV_PATH = original
