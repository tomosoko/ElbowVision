"""
Unit tests for analyze_ct_hu.py pure statistical functions.

analyze_hu() is tested with synthetic numpy volumes. During import we only
patch os.makedirs (module-level side effect); in each test we use
patch.object on _act.plt so that matplotlib I/O is suppressed without
polluting sys.modules for other test files.

Covered:
  - analyze_hu() return value structure (9 percentiles)
  - Non-air filtering threshold (> -500 HU)
  - Percentile accuracy vs numpy reference
  - HU material classification printed output
  - Matplotlib plot call patterns
  - Edge cases (empty regions, single voxel, uniform volumes)
"""
import sys
import io
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import pathlib

_SCRIPTS_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# Only patch os.makedirs to suppress directory creation at import time.
# matplotlib is imported normally so other test files are not affected.
with patch("os.makedirs"):
    import analyze_ct_hu as _act


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_mock_fig():
    """Return (fig_mock, axes_mock) where axes supports 2-D indexing."""
    fig = MagicMock()
    axes = MagicMock()   # MagicMock's __getitem__ returns another MagicMock
    return fig, axes


def _run(vol, label=""):
    """Call analyze_hu with mocked plt; return pvals."""
    fig, axes = _make_mock_fig()
    with patch.object(_act.plt, "subplots", return_value=(fig, axes)), \
         patch.object(_act.plt, "close"):
        return _act.analyze_hu(vol, label)


def _run_capture(vol, label=""):
    """Call analyze_hu, capture stdout, return (stdout_text, pvals)."""
    fig, axes = _make_mock_fig()
    with patch.object(_act.plt, "subplots", return_value=(fig, axes)), \
         patch.object(_act.plt, "close"):
        buf = io.StringIO()
        with redirect_stdout(buf):
            pvals = _act.analyze_hu(vol, label)
        return buf.getvalue(), pvals


# ─────────────────────────────────────────────────────────────────────────────
# 1. Return value structure
# ─────────────────────────────────────────────────────────────────────────────
class TestReturnValueStructure:
    def test_returns_numpy_array(self):
        vol = np.zeros((4, 4, 4))
        result = _run(vol)
        assert isinstance(result, np.ndarray)

    def test_returns_nine_elements(self):
        vol = np.zeros((4, 4, 4))
        result = _run(vol)
        assert len(result) == 9

    def test_returns_nine_elements_large_volume(self):
        rng = np.random.default_rng(0)
        vol = rng.uniform(-400, 1500, (20, 20, 20))
        result = _run(vol)
        assert len(result) == 9

    def test_percentiles_are_ascending(self):
        rng = np.random.default_rng(1)
        vol = rng.uniform(-300, 1200, (10, 10, 10))
        result = _run(vol)
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1]

    def test_returns_float_values(self):
        vol = np.linspace(0, 1000, 64).reshape(4, 4, 4)
        result = _run(vol)
        assert all(isinstance(float(v), float) for v in result)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Percentile accuracy (non-air values only)
# ─────────────────────────────────────────────────────────────────────────────
class TestPercentileAccuracy:
    _PERCENTILE_LIST = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    def _expected(self, vol):
        flat = vol.flatten()
        non_air = flat[flat > -500]
        return np.percentile(non_air, self._PERCENTILE_LIST)

    def test_p50_median_matches_numpy(self):
        vol = np.arange(0, 125, dtype=np.float32).reshape(5, 5, 5)
        result = _run(vol)
        expected = self._expected(vol)
        assert abs(result[4] - expected[4]) < 1e-3   # index 4 = P50

    def test_all_nine_percentiles_match_numpy(self):
        rng = np.random.default_rng(42)
        vol = rng.uniform(-300, 1500, (15, 15, 15))
        result = _run(vol)
        expected = self._expected(vol)
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_uniform_volume_all_same(self):
        vol = np.full((6, 6, 6), 300.0)
        result = _run(vol)
        np.testing.assert_allclose(result, 300.0)

    def test_p1_is_near_minimum_of_non_air(self):
        vol = np.linspace(-400, 1000, 200).reshape(10, 10, 2)
        result = _run(vol)
        expected = self._expected(vol)
        assert abs(result[0] - expected[0]) < 1e-3

    def test_p99_is_near_maximum(self):
        vol = np.linspace(-400, 1000, 200).reshape(10, 10, 2)
        result = _run(vol)
        expected = self._expected(vol)
        assert abs(result[8] - expected[8]) < 1e-3

    def test_float32_input_gives_correct_result(self):
        vol = np.linspace(-200, 800, 125, dtype=np.float32).reshape(5, 5, 5)
        result = _run(vol)
        expected = self._expected(vol)
        np.testing.assert_allclose(result, expected, rtol=1e-3)

    def test_float64_input_gives_correct_result(self):
        vol = np.linspace(-200, 800, 125, dtype=np.float64).reshape(5, 5, 5)
        result = _run(vol)
        expected = self._expected(vol)
        np.testing.assert_allclose(result, expected, rtol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Non-air filtering (> -500 HU threshold)
# ─────────────────────────────────────────────────────────────────────────────
class TestNonAirFiltering:
    def test_air_voxels_excluded_from_percentile(self):
        # non_air = [0.0, 500.0], P50 should be 250.0
        vol = np.array([-1000.0, -800.0, 0.0, 500.0]).reshape(2, 2, 1)
        result = _run(vol)
        expected_p50 = float(np.percentile([0.0, 500.0], 50))
        assert abs(result[4] - expected_p50) < 1.0

    def test_boundary_minus500_is_excluded(self):
        # flat > -500, so -500.0 itself is NOT in non_air
        vol = np.array([-500.0, -499.9, 100.0]).reshape(3, 1, 1)
        result = _run(vol)
        expected = np.percentile([-499.9, 100.0], [1, 5, 10, 25, 50, 75, 90, 95, 99])
        np.testing.assert_allclose(result, expected, rtol=1e-3)

    def test_minus499_included_in_non_air(self):
        # -499 > -500 → should be included
        vol = np.array([-499.0, 200.0]).reshape(2, 1, 1)
        result = _run(vol)
        expected = np.percentile([-499.0, 200.0], [1, 5, 10, 25, 50, 75, 90, 95, 99])
        np.testing.assert_allclose(result, expected, rtol=1e-3)

    def test_all_air_raises(self):
        # All voxels < -500 → non_air is empty → np.percentile raises
        vol = np.full((3, 3, 3), -1000.0)
        with pytest.raises((ValueError, IndexError)):
            _run(vol)

    def test_mixed_air_and_bone(self):
        # 50% air, 50% bone
        vol = np.array([-1000.0, -900.0, 300.0, 500.0]).reshape(2, 2, 1)
        result = _run(vol)
        # non_air = [300.0, 500.0]
        expected_p50 = float(np.percentile([300.0, 500.0], 50))
        assert abs(result[4] - expected_p50) < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Material classification output (bone / soft / air)
# ─────────────────────────────────────────────────────────────────────────────
class TestMaterialClassification:
    def test_all_bone_100_percent(self):
        vol = np.full((4, 4, 4), 500.0)
        text, _ = _run_capture(vol)
        assert "100.0%" in text

    def test_all_soft_tissue_100_percent(self):
        vol = np.full((4, 4, 4), 0.0)   # 0 HU → soft tissue
        text, _ = _run_capture(vol)
        assert "100.0%" in text

    def test_bone_threshold_at_100hu(self):
        # Exactly 2 out of 4 voxels are >= 100 HU
        vol = np.array([-50.0, 50.0, 100.0, 500.0]).reshape(2, 2, 1)
        text, _ = _run_capture(vol)
        assert "50.0%" in text

    def test_air_percentage_printed(self):
        # 2 air voxels out of 4
        vol = np.array([-1000.0, -600.0, 0.0, 300.0]).reshape(2, 2, 1)
        text, _ = _run_capture(vol)
        assert "50.0%" in text

    def test_bone_boundary_99hu_is_soft(self):
        # 99.0 < 100 → soft tissue, not bone
        vol = np.array([99.0, 99.0, 0.0, 0.0]).reshape(2, 2, 1)
        text, _ = _run_capture(vol)
        # 0 voxels in bone bucket → 0.0%
        assert "0.0%" in text

    def test_bone_boundary_100hu_is_bone(self):
        vol = np.array([100.0, 100.0]).reshape(2, 1, 1)
        text, _ = _run_capture(vol)
        # 100% bone
        assert "100.0%" in text


# ─────────────────────────────────────────────────────────────────────────────
# 5. Matplotlib plot call verification
# ─────────────────────────────────────────────────────────────────────────────
class TestPlotCalls:
    def _vol(self):
        return np.linspace(0, 500, 64).reshape(4, 4, 4)

    def test_subplots_called_once(self):
        fig, axes = _make_mock_fig()
        with patch.object(_act.plt, "subplots", return_value=(fig, axes)) as mock_sub, \
             patch.object(_act.plt, "close"):
            _act.analyze_hu(self._vol(), "")
            mock_sub.assert_called_once()

    def test_subplots_called_with_2_2(self):
        fig, axes = _make_mock_fig()
        with patch.object(_act.plt, "subplots", return_value=(fig, axes)) as mock_sub, \
             patch.object(_act.plt, "close"):
            _act.analyze_hu(self._vol(), "")
            args, _ = mock_sub.call_args
            assert args[0] == 2 and args[1] == 2

    def test_figure_savefig_called(self):
        fig, axes = _make_mock_fig()
        with patch.object(_act.plt, "subplots", return_value=(fig, axes)), \
             patch.object(_act.plt, "close"):
            _act.analyze_hu(self._vol(), "")
            fig.savefig.assert_called_once()

    def test_savefig_path_contains_label(self):
        fig, axes = _make_mock_fig()
        with patch.object(_act.plt, "subplots", return_value=(fig, axes)), \
             patch.object(_act.plt, "close"):
            _act.analyze_hu(self._vol(), "_mytest")
            path_arg = fig.savefig.call_args[0][0]
            assert "_mytest" in path_arg

    def test_savefig_path_contains_base_name(self):
        fig, axes = _make_mock_fig()
        with patch.object(_act.plt, "subplots", return_value=(fig, axes)), \
             patch.object(_act.plt, "close"):
            _act.analyze_hu(self._vol(), "")
            path_arg = fig.savefig.call_args[0][0]
            assert "ct_hu_analysis" in path_arg

    def test_plt_close_called(self):
        fig, axes = _make_mock_fig()
        with patch.object(_act.plt, "subplots", return_value=(fig, axes)), \
             patch.object(_act.plt, "close") as mock_close:
            _act.analyze_hu(self._vol(), "")
            mock_close.assert_called()

    def test_suptitle_called(self):
        fig, axes = _make_mock_fig()
        with patch.object(_act.plt, "subplots", return_value=(fig, axes)), \
             patch.object(_act.plt, "close"):
            _act.analyze_hu(self._vol(), "")
            fig.suptitle.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Edge cases
# ─────────────────────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_single_positive_voxel(self):
        vol = np.array([[[200.0]]])
        result = _run(vol)
        # All 9 percentiles of a single-element array = that element
        np.testing.assert_allclose(result, 200.0)

    def test_single_tissue_voxel(self):
        vol = np.array([[[-50.0]]])
        result = _run(vol)
        np.testing.assert_allclose(result, -50.0)

    def test_two_voxel_volume(self):
        vol = np.array([0.0, 1000.0]).reshape(2, 1, 1)
        result = _run(vol)
        assert len(result) == 9

    def test_large_volume_no_error(self):
        rng = np.random.default_rng(7)
        vol = rng.uniform(-1000, 2000, (50, 50, 50))
        result = _run(vol)
        assert len(result) == 9

    def test_empty_bone_region_no_crash(self):
        # All soft tissue: HU in (-500, 100)
        vol = np.full((5, 5, 5), 50.0)
        result = _run(vol)
        assert len(result) == 9

    def test_empty_shell_region_no_crash(self):
        # Volume with no voxels in (-200, 200)
        vol = np.array([-1000.0, 500.0, 1000.0]).reshape(3, 1, 1)
        result = _run(vol)
        assert len(result) == 9

    def test_negative_hu_range_only(self):
        vol = np.linspace(-490, -100, 27).reshape(3, 3, 3)
        result = _run(vol)
        assert len(result) == 9
        assert result[0] < result[8]   # ascending

    def test_label_empty_string(self):
        vol = np.linspace(0, 500, 27).reshape(3, 3, 3)
        result = _run(vol, label="")
        assert len(result) == 9

    def test_label_with_series_number(self):
        vol = np.linspace(0, 500, 27).reshape(3, 3, 3)
        result = _run(vol, label="_series4")
        assert len(result) == 9

    def test_high_hu_bone_phantom(self):
        # Typical bone phantom values (400–800 HU)
        rng = np.random.default_rng(10)
        vol = rng.uniform(400, 800, (10, 10, 10))
        result = _run(vol)
        assert result[0] > 390    # P1 near min
        assert result[8] < 810    # P99 near max
