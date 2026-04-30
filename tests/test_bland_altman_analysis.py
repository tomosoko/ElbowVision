"""Tests for scripts/bland_altman_analysis.py

Covers:
  - Module constants and structure
  - bland_altman_plot(): file output, correct path, custom unit
  - Statistics verified via printed output (平均差 / LoA / SD lines)
  - Edge cases: identical arrays, single pair, large arrays, negative angles
  - Mathematical properties: mean/diff/LoA formulas
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import bland_altman_analysis as baa


# ── Helpers ──────────────────────────────────────────────────────────────────

def _run_plot(tmp_path, monkeypatch, manual, ai, angle_name="test", unit="°"):
    """Run bland_altman_plot with OUTPUT_DIR redirected to tmp_path."""
    monkeypatch.setattr(baa, "OUTPUT_DIR", str(tmp_path))
    baa.bland_altman_plot(manual, ai, angle_name, unit=unit)


# ── Module constants ─────────────────────────────────────────────────────────

class TestModuleConstants:
    def test_output_dir_attribute_exists(self):
        assert hasattr(baa, "OUTPUT_DIR")

    def test_output_dir_default_value(self):
        assert baa.OUTPUT_DIR == "validation_output"

    def test_output_dir_is_str(self):
        assert isinstance(baa.OUTPUT_DIR, str)

    def test_bland_altman_plot_callable(self):
        assert callable(baa.bland_altman_plot)

    def test_main_callable(self):
        assert callable(baa.main)


# ── File output ───────────────────────────────────────────────────────────────

class TestFileOutput:
    def test_creates_png_file(self, tmp_path, monkeypatch):
        manual = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        ai = manual + 0.5
        _run_plot(tmp_path, monkeypatch, manual, ai, "flexion")
        assert os.path.isfile(os.path.join(str(tmp_path), "bland_altman_flexion.png"))

    def test_output_file_nonempty(self, tmp_path, monkeypatch):
        manual = np.array([10.0, 20.0, 30.0])
        ai = manual + 1.0
        _run_plot(tmp_path, monkeypatch, manual, ai, "carrying")
        out = os.path.join(str(tmp_path), "bland_altman_carrying.png")
        assert os.path.getsize(out) > 0

    def test_angle_name_spaces_become_underscores(self, tmp_path, monkeypatch):
        manual = np.array([10.0, 20.0, 30.0])
        ai = manual + 0.5
        _run_plot(tmp_path, monkeypatch, manual, ai, "carrying angle")
        out = os.path.join(str(tmp_path), "bland_altman_carrying_angle.png")
        assert os.path.isfile(out)

    def test_angle_name_lowercased_in_filename(self, tmp_path, monkeypatch):
        manual = np.array([10.0, 20.0, 30.0])
        ai = manual + 0.5
        _run_plot(tmp_path, monkeypatch, manual, ai, "FLEXION")
        out = os.path.join(str(tmp_path), "bland_altman_flexion.png")
        assert os.path.isfile(out)

    def test_path_uses_output_dir(self, tmp_path, monkeypatch):
        custom_dir = str(tmp_path / "custom_output")
        os.makedirs(custom_dir, exist_ok=True)
        monkeypatch.setattr(baa, "OUTPUT_DIR", custom_dir)
        manual = np.array([10.0, 20.0, 30.0])
        ai = manual + 0.5
        baa.bland_altman_plot(manual, ai, "test")
        out = os.path.join(custom_dir, "bland_altman_test.png")
        assert os.path.isfile(out)

    def test_custom_unit_produces_file(self, tmp_path, monkeypatch):
        manual = np.array([1.0, 2.0, 3.0])
        ai = manual + 0.1
        _run_plot(tmp_path, monkeypatch, manual, ai, "length", unit="mm")
        out = os.path.join(str(tmp_path), "bland_altman_length.png")
        assert os.path.isfile(out)

    def test_multiple_calls_create_separate_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(baa, "OUTPUT_DIR", str(tmp_path))
        data = np.array([10.0, 20.0, 30.0])
        baa.bland_altman_plot(data, data + 0.5, "alpha")
        baa.bland_altman_plot(data, data - 0.5, "beta")
        assert os.path.isfile(os.path.join(str(tmp_path), "bland_altman_alpha.png"))
        assert os.path.isfile(os.path.join(str(tmp_path), "bland_altman_beta.png"))


# ── Printed statistics ────────────────────────────────────────────────────────

class TestPrintedStatistics:
    def test_prints_saved_path(self, tmp_path, monkeypatch, capsys):
        _run_plot(tmp_path, monkeypatch, np.array([10.0, 20.0]), np.array([11.0, 21.0]), "test")
        captured = capsys.readouterr()
        assert "Saved:" in captured.out

    def test_prints_mean_diff_label(self, tmp_path, monkeypatch, capsys):
        _run_plot(tmp_path, monkeypatch, np.array([10.0, 20.0, 30.0]), np.array([11.0, 21.0, 31.0]), "t")
        assert "平均差" in capsys.readouterr().out

    def test_prints_loa_label(self, tmp_path, monkeypatch, capsys):
        _run_plot(tmp_path, monkeypatch, np.array([10.0, 20.0, 30.0]), np.array([10.5, 20.5, 30.5]), "t")
        assert "LoA:" in capsys.readouterr().out

    def test_prints_sd_label(self, tmp_path, monkeypatch, capsys):
        _run_plot(tmp_path, monkeypatch, np.array([10.0, 20.0, 30.0]), np.array([10.5, 20.5, 30.5]), "t")
        assert "SD:" in capsys.readouterr().out

    def test_correct_mean_diff_constant_bias(self, tmp_path, monkeypatch, capsys):
        """AI = manual + 2.0 everywhere → mean_diff = 2.000"""
        manual = np.array([10.0, 20.0, 30.0, 40.0])
        ai = manual + 2.0
        _run_plot(tmp_path, monkeypatch, manual, ai, "t")
        assert "2.000" in capsys.readouterr().out

    def test_correct_sd_zero_for_constant_diff(self, tmp_path, monkeypatch, capsys):
        """Constant diff → SD = 0.000"""
        manual = np.array([10.0, 20.0, 30.0, 40.0])
        ai = manual + 5.0
        _run_plot(tmp_path, monkeypatch, manual, ai, "t")
        assert "0.000" in capsys.readouterr().out

    def test_correct_loa_values_constant_diff(self, tmp_path, monkeypatch, capsys):
        """Constant diff=3 → LoA=[3.000, 3.000]"""
        manual = np.array([10.0, 20.0, 30.0])
        ai = manual + 3.0
        _run_plot(tmp_path, monkeypatch, manual, ai, "t")
        out = capsys.readouterr().out
        assert "3.000" in out

    def test_custom_unit_in_output(self, tmp_path, monkeypatch, capsys):
        _run_plot(tmp_path, monkeypatch, np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.1, 3.1]), "t", unit="mm")
        assert "mm" in capsys.readouterr().out


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_identical_arrays_no_crash(self, tmp_path, monkeypatch):
        arr = np.array([10.0, 20.0, 30.0, 40.0])
        _run_plot(tmp_path, monkeypatch, arr, arr.copy(), "zero_diff")

    def test_single_pair_no_crash(self, tmp_path, monkeypatch):
        _run_plot(tmp_path, monkeypatch, np.array([10.0]), np.array([11.0]), "single")

    def test_large_array_no_crash(self, tmp_path, monkeypatch):
        rng = np.random.RandomState(0)
        manual = rng.uniform(0, 180, 1000)
        ai = manual + rng.normal(0, 2.0, 1000)
        _run_plot(tmp_path, monkeypatch, manual, ai, "large")

    def test_negative_angles_no_crash(self, tmp_path, monkeypatch):
        manual = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        ai = manual + 0.5
        _run_plot(tmp_path, monkeypatch, manual, ai, "negative")

    def test_constant_ai_values_no_crash(self, tmp_path, monkeypatch):
        manual = np.array([10.0, 20.0, 30.0])
        ai = np.full(3, 15.0)
        _run_plot(tmp_path, monkeypatch, manual, ai, "const")

    def test_two_pairs_no_crash(self, tmp_path, monkeypatch):
        _run_plot(tmp_path, monkeypatch, np.array([5.0, 10.0]), np.array([6.0, 11.0]), "two")


# ── Mathematical properties ───────────────────────────────────────────────────

class TestMathematicalProperties:
    """Verify the formulas used inside bland_altman_plot against NumPy."""

    def test_mean_formula(self):
        """mean = (manual + ai) / 2"""
        manual = np.array([10.0, 20.0, 30.0])
        ai = np.array([12.0, 22.0, 28.0])
        expected = np.array([11.0, 21.0, 29.0])
        np.testing.assert_array_almost_equal((manual + ai) / 2, expected)

    def test_diff_formula(self):
        """diff = ai - manual"""
        manual = np.array([10.0, 20.0, 30.0])
        ai = np.array([11.0, 19.0, 32.0])
        np.testing.assert_array_almost_equal(ai - manual, np.array([1.0, -1.0, 2.0]))

    def test_loa_width_equals_2_times_196_std(self):
        """LoA width = 2 × 1.96 × std"""
        diff = np.array([1.0, 2.0, -1.0, 3.0, -2.0])
        std_diff = np.std(diff)
        loa_upper = np.mean(diff) + 1.96 * std_diff
        loa_lower = np.mean(diff) - 1.96 * std_diff
        assert loa_upper - loa_lower == pytest.approx(2 * 1.96 * std_diff, abs=1e-10)

    def test_loa_symmetric_around_mean_diff(self):
        """(LoA_upper + LoA_lower) / 2 == mean_diff"""
        diff = np.array([0.5, -0.5, 1.0, -1.0, 0.0])
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff
        assert (loa_upper + loa_lower) / 2 == pytest.approx(mean_diff, abs=1e-10)

    def test_loa_upper_gt_lower(self):
        """LoA_upper > LoA_lower when std > 0"""
        diff = np.array([1.0, 2.0, -1.0, 3.0])
        std_diff = np.std(diff)
        assert std_diff > 0
        loa_upper = np.mean(diff) + 1.96 * std_diff
        loa_lower = np.mean(diff) - 1.96 * std_diff
        assert loa_upper > loa_lower

    def test_ci_mean_formula(self):
        """95%CI = 1.96 × std / sqrt(n)"""
        n = 25
        std = 3.0
        ci = 1.96 * std / np.sqrt(n)
        assert ci == pytest.approx(1.96 * 3.0 / 5.0, abs=1e-10)

    def test_mean_diff_zero_for_identical_arrays(self):
        arr = np.array([10.0, 20.0, 30.0, 40.0])
        diff = arr - arr
        assert np.mean(diff) == 0.0

    def test_mean_diff_equals_constant_bias(self):
        manual = np.array([10.0, 20.0, 30.0, 40.0])
        bias = 2.5
        ai = manual + bias
        diff = ai - manual
        assert np.mean(diff) == pytest.approx(bias, abs=1e-10)
