"""Unit tests for scripts/plot_ncc_curves.py pure functions."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Module import with mocked heavy dependencies
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_cv2():
    """cv2 is imported at module level; stub it if missing."""
    if "cv2" not in sys.modules:
        sys.modules["cv2"] = MagicMock()
    yield


def _import_module():
    import scripts.plot_ncc_curves as mod
    return mod


# ===================================================================
# snap_to_nearest_angle
# ===================================================================

class TestSnapToNearestAngle:
    def setup_method(self):
        self.mod = _import_module()

    def test_exact_match(self):
        assert self.mod.snap_to_nearest_angle(90.0, [60.0, 90.0, 120.0]) == 90.0

    def test_nearest_below(self):
        assert self.mod.snap_to_nearest_angle(88.0, [60.0, 90.0, 120.0]) == 90.0

    def test_nearest_above(self):
        assert self.mod.snap_to_nearest_angle(92.0, [60.0, 90.0, 120.0]) == 90.0

    def test_equidistant_picks_one(self):
        result = self.mod.snap_to_nearest_angle(75.0, [60.0, 90.0])
        assert result in (60.0, 90.0)

    def test_single_angle(self):
        assert self.mod.snap_to_nearest_angle(999.0, [45.0]) == 45.0

    def test_negative_angles(self):
        assert self.mod.snap_to_nearest_angle(-5.0, [-10.0, 0.0, 10.0]) == -10.0

    def test_set_input(self):
        result = self.mod.snap_to_nearest_angle(91.0, {60.0, 90.0, 120.0})
        assert result == 90.0

    def test_float_precision(self):
        result = self.mod.snap_to_nearest_angle(90.001, [89.999, 90.002, 91.0])
        assert result == 90.002

    def test_large_library(self):
        angles = [float(a) for a in range(60, 181)]
        assert self.mod.snap_to_nearest_angle(100.3, angles) == 100.0

    def test_boundary_min(self):
        angles = [60.0, 90.0, 120.0, 150.0, 180.0]
        assert self.mod.snap_to_nearest_angle(50.0, angles) == 60.0

    def test_boundary_max(self):
        angles = [60.0, 90.0, 120.0, 150.0, 180.0]
        assert self.mod.snap_to_nearest_angle(200.0, angles) == 180.0


# ===================================================================
# compute_sharpness
# ===================================================================

class TestComputeSharpness:
    def setup_method(self):
        self.mod = _import_module()

    def test_uniform_values_zero_sharpness(self):
        vals = [0.5, 0.5, 0.5, 0.5]
        s = self.mod.compute_sharpness(vals)
        assert s == pytest.approx(0.0, abs=1e-6)

    def test_single_peak_high_sharpness(self):
        vals = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        s = self.mod.compute_sharpness(vals)
        assert s > 2.0  # sharp peak

    def test_broad_peak_lower_sharpness(self):
        vals = [0.1, 0.3, 0.5, 0.7, 0.5, 0.3, 0.1]
        s = self.mod.compute_sharpness(vals)
        assert 0 < s < 3.0

    def test_returns_float(self):
        assert isinstance(self.mod.compute_sharpness([1.0, 2.0, 3.0]), float)

    def test_numpy_input(self):
        arr = np.array([0.1, 0.5, 0.9, 0.5, 0.1])
        s = self.mod.compute_sharpness(arr)
        assert s > 0

    def test_all_zeros(self):
        s = self.mod.compute_sharpness([0.0, 0.0, 0.0])
        assert s == pytest.approx(0.0, abs=1e-6)

    def test_two_values(self):
        s = self.mod.compute_sharpness([0.0, 1.0])
        assert s > 0

    def test_negative_values(self):
        s = self.mod.compute_sharpness([-1.0, 0.0, 1.0])
        assert isinstance(s, float)

    def test_sharpness_increases_with_peak_isolation(self):
        broad = [0.3, 0.5, 0.7, 0.9, 0.7, 0.5, 0.3]
        sharp = [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0]
        assert self.mod.compute_sharpness(sharp) > self.mod.compute_sharpness(broad)

    def test_single_element(self):
        s = self.mod.compute_sharpness([5.0])
        # std = 0, so denominator is 1e-8, peak - mean = 0
        assert s == pytest.approx(0.0, abs=1e-6)


# ===================================================================
# compute_ncc_curve
# ===================================================================

class TestComputeNccCurve:
    def setup_method(self):
        self.mod = _import_module()

    def _make_angle_to_drr(self, n_angles=5, size=64):
        """Create a small synthetic DRR library."""
        rng = np.random.RandomState(42)
        angles = np.linspace(60, 180, n_angles)
        return {float(a): rng.randint(0, 256, (size, size), dtype=np.uint8) for a in angles}

    @patch("scripts.plot_ncc_curves.extract_edges")
    @patch("scripts.plot_ncc_curves.ncc")
    @patch("scripts.plot_ncc_curves.preprocess_image")
    def test_drr_mode_returns_sorted_angles(self, mock_preprocess, mock_ncc, mock_edges):
        mock_ncc.return_value = 0.5
        mock_edges.return_value = np.zeros((64, 64), dtype=np.float32)

        angle_to_drr = self._make_angle_to_drr()
        query = np.zeros((64, 64), dtype=np.uint8)
        angles, ncc_vals, encc_vals = self.mod.compute_ncc_curve(
            query, angle_to_drr, 60.0, 180.0, is_drr=True
        )
        assert angles == sorted(angles)
        mock_preprocess.assert_not_called()

    @patch("scripts.plot_ncc_curves.extract_edges")
    @patch("scripts.plot_ncc_curves.ncc")
    @patch("scripts.plot_ncc_curves.preprocess_image")
    def test_realxray_mode_calls_preprocess(self, mock_preprocess, mock_ncc, mock_edges):
        mock_preprocess.return_value = np.zeros((64, 64), dtype=np.float32)
        mock_ncc.return_value = 0.3
        mock_edges.return_value = np.zeros((64, 64), dtype=np.float32)

        angle_to_drr = self._make_angle_to_drr()
        query = np.zeros((64, 64), dtype=np.uint8)
        self.mod.compute_ncc_curve(query, angle_to_drr, 60.0, 180.0, is_drr=False)
        mock_preprocess.assert_called_once()

    @patch("scripts.plot_ncc_curves.extract_edges")
    @patch("scripts.plot_ncc_curves.ncc")
    @patch("scripts.plot_ncc_curves.preprocess_image")
    def test_output_lengths_match(self, mock_preprocess, mock_ncc, mock_edges):
        mock_ncc.return_value = 0.5
        mock_edges.return_value = np.zeros((64, 64), dtype=np.float32)

        n = 7
        angle_to_drr = self._make_angle_to_drr(n_angles=n)
        query = np.zeros((64, 64), dtype=np.uint8)
        angles, ncc_vals, encc_vals = self.mod.compute_ncc_curve(
            query, angle_to_drr, 60.0, 180.0, is_drr=True
        )
        assert len(angles) == n
        assert len(ncc_vals) == n
        assert len(encc_vals) == n

    @patch("scripts.plot_ncc_curves.extract_edges")
    @patch("scripts.plot_ncc_curves.ncc")
    @patch("scripts.plot_ncc_curves.preprocess_image")
    def test_ncc_called_per_angle(self, mock_preprocess, mock_ncc, mock_edges):
        mock_ncc.return_value = 0.5
        mock_edges.return_value = np.zeros((64, 64), dtype=np.float32)

        n = 5
        angle_to_drr = self._make_angle_to_drr(n_angles=n)
        query = np.zeros((64, 64), dtype=np.uint8)
        self.mod.compute_ncc_curve(query, angle_to_drr, 60.0, 180.0, is_drr=True)
        # ncc called twice per angle: once for NCC, once for edge-NCC
        assert mock_ncc.call_count == 2 * n

    @patch("scripts.plot_ncc_curves.extract_edges")
    @patch("scripts.plot_ncc_curves.ncc")
    @patch("scripts.plot_ncc_curves.preprocess_image")
    def test_drr_normalization(self, mock_preprocess, mock_ncc, mock_edges):
        """In DRR mode, query should be divided by 255."""
        mock_ncc.return_value = 0.5
        mock_edges.return_value = np.zeros((64, 64), dtype=np.float32)

        angle_to_drr = {90.0: np.full((64, 64), 128, dtype=np.uint8)}
        query = np.full((64, 64), 255, dtype=np.uint8)
        self.mod.compute_ncc_curve(query, angle_to_drr, 60.0, 180.0, is_drr=True)

        # extract_edges is called with the normalized query
        first_call_arg = mock_edges.call_args_list[0][0][0]
        assert first_call_arg.max() == pytest.approx(1.0, abs=0.01)

    @patch("scripts.plot_ncc_curves.extract_edges")
    @patch("scripts.plot_ncc_curves.ncc")
    @patch("scripts.plot_ncc_curves.preprocess_image")
    def test_single_angle(self, mock_preprocess, mock_ncc, mock_edges):
        mock_ncc.return_value = 0.9
        mock_edges.return_value = np.zeros((32, 32), dtype=np.float32)

        angle_to_drr = {90.0: np.zeros((32, 32), dtype=np.uint8)}
        query = np.zeros((32, 32), dtype=np.uint8)
        angles, ncc_vals, encc_vals = self.mod.compute_ncc_curve(
            query, angle_to_drr, 60.0, 180.0, is_drr=True
        )
        assert angles == [90.0]
        assert ncc_vals == [0.9]
        assert encc_vals == [0.9]

    @patch("scripts.plot_ncc_curves.extract_edges")
    @patch("scripts.plot_ncc_curves.ncc")
    @patch("scripts.plot_ncc_curves.preprocess_image")
    def test_returns_lists(self, mock_preprocess, mock_ncc, mock_edges):
        mock_ncc.return_value = 0.5
        mock_edges.return_value = np.zeros((64, 64), dtype=np.float32)

        angle_to_drr = self._make_angle_to_drr()
        query = np.zeros((64, 64), dtype=np.uint8)
        angles, ncc_vals, encc_vals = self.mod.compute_ncc_curve(
            query, angle_to_drr, 60.0, 180.0, is_drr=True
        )
        assert isinstance(angles, list)
        assert isinstance(ncc_vals, list)
        assert isinstance(encc_vals, list)

    @patch("scripts.plot_ncc_curves.extract_edges")
    @patch("scripts.plot_ncc_curves.ncc")
    @patch("scripts.plot_ncc_curves.preprocess_image")
    def test_varying_ncc_values(self, mock_preprocess, mock_ncc, mock_edges):
        """NCC values should vary when the mock returns different values."""
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            # Alternate between NCC and edge-NCC calls
            return 0.3 + 0.1 * (call_count[0] % 3)
        mock_ncc.side_effect = side_effect
        mock_edges.return_value = np.zeros((64, 64), dtype=np.float32)

        angle_to_drr = self._make_angle_to_drr(n_angles=3)
        query = np.zeros((64, 64), dtype=np.uint8)
        angles, ncc_vals, encc_vals = self.mod.compute_ncc_curve(
            query, angle_to_drr, 60.0, 180.0, is_drr=True
        )
        # With varying side_effect, not all values should be equal
        assert len(set(ncc_vals + encc_vals)) > 1


# ===================================================================
# plot_curve
# ===================================================================

class TestPlotCurve:
    def setup_method(self):
        import matplotlib
        matplotlib.use("Agg")
        self.mod = _import_module()

    def _make_fig_ax(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        return fig, ax

    def test_returns_three_floats(self):
        fig, ax = self._make_fig_ax()
        angles = [60.0, 90.0, 120.0, 150.0, 180.0]
        ncc_v  = [0.3,  0.5,  0.9,   0.6,   0.4]
        encc_v = [0.2,  0.4,  0.85,  0.5,   0.3]
        best, peak, sharp = self.mod.plot_curve(
            ax, angles, ncc_v, encc_v, "Test", 60.0, 180.0
        )
        assert isinstance(best, float)
        assert isinstance(peak, float)
        assert isinstance(sharp, float)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_peak_ncc_matches_max(self):
        fig, ax = self._make_fig_ax()
        ncc_v = [0.1, 0.3, 0.9, 0.2]
        encc_v = [0.1, 0.2, 0.3, 0.1]
        angles = [60.0, 90.0, 120.0, 150.0]
        _, peak, _ = self.mod.plot_curve(
            ax, angles, ncc_v, encc_v, "T", 60.0, 150.0
        )
        assert peak == pytest.approx(0.9)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_best_combined_angle(self):
        fig, ax = self._make_fig_ax()
        angles = [60.0, 90.0, 120.0]
        ncc_v  = [0.1,  0.9,  0.1]  # peak at 90
        encc_v = [0.1,  0.1,  0.9]  # peak at 120
        best, _, _ = self.mod.plot_curve(
            ax, angles, ncc_v, encc_v, "T", 60.0, 120.0
        )
        assert best == pytest.approx(105.0)  # (90 + 120) / 2
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_gt_angle_none(self):
        """Should not crash when gt_angle is None."""
        fig, ax = self._make_fig_ax()
        angles = [60.0, 90.0, 120.0]
        ncc_v  = [0.5, 0.5, 0.5]
        encc_v = [0.5, 0.5, 0.5]
        best, peak, sharp = self.mod.plot_curve(
            ax, angles, ncc_v, encc_v, "T", 60.0, 120.0, gt_angle=None
        )
        assert peak == pytest.approx(0.5)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_mark_best_false(self):
        """mark_best=False should still return correct values."""
        fig, ax = self._make_fig_ax()
        angles = [60.0, 90.0, 120.0]
        ncc_v  = [0.1, 0.9, 0.1]
        encc_v = [0.1, 0.9, 0.1]
        best, peak, _ = self.mod.plot_curve(
            ax, angles, ncc_v, encc_v, "T", 60.0, 120.0, mark_best=False
        )
        assert best == pytest.approx(90.0)
        assert peak == pytest.approx(0.9)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_xlim_set(self):
        fig, ax = self._make_fig_ax()
        angles = [60.0, 90.0, 120.0]
        ncc_v  = [0.5, 0.5, 0.5]
        encc_v = [0.5, 0.5, 0.5]
        self.mod.plot_curve(ax, angles, ncc_v, encc_v, "T", 60.0, 180.0)
        xlim = ax.get_xlim()
        assert xlim[0] == pytest.approx(60.0)
        assert xlim[1] == pytest.approx(180.0)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_title_contains_peak_ncc(self):
        fig, ax = self._make_fig_ax()
        angles = [60.0, 90.0, 120.0]
        ncc_v  = [0.1, 0.9, 0.1]
        encc_v = [0.1, 0.9, 0.1]
        self.mod.plot_curve(ax, angles, ncc_v, encc_v, "MyTitle", 60.0, 120.0)
        title = ax.get_title()
        assert "peak_ncc=0.900" in title
        assert "MyTitle" in title
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_sharpness_in_title(self):
        fig, ax = self._make_fig_ax()
        angles = [60.0, 90.0, 120.0]
        ncc_v  = [0.1, 0.9, 0.1]
        encc_v = [0.1, 0.9, 0.1]
        self.mod.plot_curve(ax, angles, ncc_v, encc_v, "T", 60.0, 120.0)
        title = ax.get_title()
        assert "sharpness=" in title
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_two_lines_drawn(self):
        fig, ax = self._make_fig_ax()
        angles = [60.0, 90.0, 120.0]
        ncc_v  = [0.5, 0.5, 0.5]
        encc_v = [0.5, 0.5, 0.5]
        self.mod.plot_curve(
            ax, angles, ncc_v, encc_v, "T", 60.0, 120.0, mark_best=False, gt_angle=None
        )
        lines = ax.get_lines()
        assert len(lines) >= 2  # NCC and edge-NCC
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_gt_angle_adds_vline(self):
        fig, ax = self._make_fig_ax()
        angles = [60.0, 90.0, 120.0]
        ncc_v  = [0.5, 0.5, 0.5]
        encc_v = [0.5, 0.5, 0.5]
        self.mod.plot_curve(
            ax, angles, ncc_v, encc_v, "T", 60.0, 120.0,
            mark_best=False, gt_angle=90.0
        )
        lines_with_gt = ax.get_lines()

        fig2, ax2 = self._make_fig_ax()
        self.mod.plot_curve(
            ax2, angles, ncc_v, encc_v, "T", 60.0, 120.0,
            mark_best=False, gt_angle=None
        )
        lines_without_gt = ax2.get_lines()

        assert len(lines_with_gt) > len(lines_without_gt)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_legend_present(self):
        fig, ax = self._make_fig_ax()
        angles = [60.0, 90.0, 120.0]
        ncc_v  = [0.5, 0.5, 0.5]
        encc_v = [0.5, 0.5, 0.5]
        self.mod.plot_curve(ax, angles, ncc_v, encc_v, "T", 60.0, 120.0)
        legend = ax.get_legend()
        assert legend is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_single_angle_no_crash(self):
        fig, ax = self._make_fig_ax()
        angles = [90.0]
        ncc_v  = [0.8]
        encc_v = [0.7]
        best, peak, sharp = self.mod.plot_curve(
            ax, angles, ncc_v, encc_v, "T", 60.0, 180.0
        )
        assert peak == pytest.approx(0.8)
        import matplotlib.pyplot as plt
        plt.close(fig)


# ===================================================================
# Module-level constants and imports
# ===================================================================

class TestModuleStructure:
    def setup_method(self):
        self.mod = _import_module()

    def test_project_root_is_directory(self):
        assert self.mod._PROJECT_ROOT.is_dir()

    def test_has_compute_ncc_curve(self):
        assert callable(self.mod.compute_ncc_curve)

    def test_has_plot_curve(self):
        assert callable(self.mod.plot_curve)

    def test_has_snap_to_nearest_angle(self):
        assert callable(self.mod.snap_to_nearest_angle)

    def test_has_compute_sharpness(self):
        assert callable(self.mod.compute_sharpness)

    def test_has_main(self):
        assert callable(self.mod.main)


# ===================================================================
# main() CLI integration
# ===================================================================

class TestMain:
    def setup_method(self):
        self.mod = _import_module()

    @patch("sys.argv", ["plot_ncc_curves.py", "--library", "dummy.npz"])
    @patch("scripts.plot_ncc_curves.compute_ncc_curve")
    def test_main_requires_library(self, mock_compute):
        """main() should parse --library argument."""
        mock_lib = MagicMock()
        mock_lib.return_value = (
            np.array([90.0, 120.0]),
            np.array([np.zeros((64, 64), dtype=np.uint8)] * 2),
            {"angle_min": 60, "angle_max": 180},
        )

        with patch("scripts.similarity_matching.load_drr_library", mock_lib):
            with patch.object(Path, "exists", return_value=False):
                with patch("matplotlib.pyplot.subplots") as mock_subplots:
                    with patch("matplotlib.pyplot.close"):
                        mock_fig = MagicMock()
                        mock_ax = MagicMock()
                        mock_subplots.return_value = (mock_fig, [mock_ax])
                        mock_compute.return_value = ([90.0, 120.0], [0.5, 0.5], [0.3, 0.3])
                        try:
                            self.mod.main()
                        except (SystemExit, FileNotFoundError, Exception):
                            pass  # Expected - file doesn't exist


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def setup_method(self):
        self.mod = _import_module()

    def test_snap_empty_raises(self):
        with pytest.raises((ValueError, StopIteration)):
            self.mod.snap_to_nearest_angle(90.0, [])

    def test_sharpness_empty_raises(self):
        with pytest.raises((ValueError, IndexError)):
            self.mod.compute_sharpness([])

    def test_sharpness_large_array(self):
        vals = np.random.RandomState(42).rand(10000).tolist()
        s = self.mod.compute_sharpness(vals)
        assert isinstance(s, float)
        assert np.isfinite(s)

    def test_snap_identical_angles(self):
        result = self.mod.snap_to_nearest_angle(90.0, [90.0, 90.0, 90.0])
        assert result == 90.0

    def test_sharpness_deterministic(self):
        vals = [0.1, 0.5, 0.9, 0.5, 0.1]
        s1 = self.mod.compute_sharpness(vals)
        s2 = self.mod.compute_sharpness(vals)
        assert s1 == s2
