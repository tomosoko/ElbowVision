"""
Unit tests for scripts/generate_figures.py

Tests cover:
- _draw_3d_box helper
- _draw_placeholder helper
- generate_fig1_pipeline (self-contained pipeline diagram)
- generate_fig2_drr_algorithm (4-panel DRR algorithm figure)
- generate_fig3_drr_variations (8-panel DRR variations, CSV + image mocks)
- generate_fig4_bland_altman (Bland-Altman plot with mock data)
- generate_fig4_bland_altman_v6 (v6 Bland-Altman 2-panel)
- main() argparse dispatcher
"""
from __future__ import annotations

import os
import sys
import csv
import types
import tempfile
from dataclasses import dataclass
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# ── Module import setup ──────────────────────────────────────────────
# generate_figures.py lives in scripts/ and uses relative imports from
# bland_altman (also in scripts/). We add scripts/ to sys.path.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import generate_figures as gf

# ── Helpers / Fixtures ───────────────────────────────────────────────

@pytest.fixture
def tmp_png(tmp_path):
    """Return a temporary PNG file path."""
    return str(tmp_path / "test_output.png")


@pytest.fixture
def tmp_dir(tmp_path):
    """Return a temporary directory path string."""
    return str(tmp_path)


@dataclass
class _FakeBlandAltman:
    """Mimics the result of compute_bland_altman."""
    n: int = 100
    mean_diff: float = 0.1
    loa_upper: float = 1.5
    loa_lower: float = -1.3
    mae: float = 0.5
    rmse: float = 0.7
    icc: float = 0.998
    r_squared: float = 0.996


# =====================================================================
# _draw_3d_box
# =====================================================================

class TestDraw3dBox:
    """Tests for the _draw_3d_box helper function."""

    def test_adds_patches_to_axes(self):
        """_draw_3d_box should add a rectangle patch and two fill polygons."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        children_before = len(ax.get_children())
        gf._draw_3d_box(ax, 1, 1, 4, 4, "#AABBCC")
        children_after = len(ax.get_children())
        # Should add at least 3 elements (1 Rectangle + 2 Polygon from fill)
        assert children_after - children_before >= 3
        plt.close(fig)

    def test_accepts_different_colors(self):
        """Should not raise for valid color strings."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for color in ["#FF0000", "blue", "#90CAF9", "green"]:
            gf._draw_3d_box(ax, 0, 0, 3, 3, color)
        plt.close(fig)

    def test_zero_size_box(self):
        """Zero-size box should not raise."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        gf._draw_3d_box(ax, 5, 5, 0, 0, "#000000")
        plt.close(fig)

    def test_large_dimensions(self):
        """Large box dimensions should not raise."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        gf._draw_3d_box(ax, 0, 0, 100, 100, "#FFFFFF")
        plt.close(fig)


# =====================================================================
# _draw_placeholder
# =====================================================================

class TestDrawPlaceholder:
    """Tests for the _draw_placeholder helper function."""

    def test_sets_facecolor(self):
        """Should set axis facecolor to light gray."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        gf._draw_placeholder(ax)
        # facecolor is stored as RGBA tuple
        fc = ax.get_facecolor()
        # #F5F5F5 = (245/255, 245/255, 245/255, 1.0)
        assert fc[0] == pytest.approx(245 / 255, abs=0.01)
        assert fc[1] == pytest.approx(245 / 255, abs=0.01)
        assert fc[2] == pytest.approx(245 / 255, abs=0.01)
        plt.close(fig)

    def test_adds_text(self):
        """Should add text to the axes."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        gf._draw_placeholder(ax)
        texts = ax.texts
        assert len(texts) >= 1
        assert "No image" in texts[0].get_text()
        plt.close(fig)

    def test_text_centered(self):
        """Text should be centered (ha=center, va=center)."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        gf._draw_placeholder(ax)
        txt = ax.texts[0]
        assert txt.get_ha() == "center"
        assert txt.get_va() == "center"
        plt.close(fig)


# =====================================================================
# generate_fig1_pipeline
# =====================================================================

class TestGenerateFig1Pipeline:
    """Tests for the system pipeline figure generator."""

    def test_creates_file(self, tmp_png):
        """Should save a PNG file to the specified path."""
        gf.generate_fig1_pipeline(tmp_png)
        assert os.path.isfile(tmp_png)

    def test_file_not_empty(self, tmp_png):
        """Saved file should have non-zero size."""
        gf.generate_fig1_pipeline(tmp_png)
        assert os.path.getsize(tmp_png) > 0

    def test_output_is_valid_png(self, tmp_png):
        """File should start with PNG magic bytes."""
        gf.generate_fig1_pipeline(tmp_png)
        with open(tmp_png, "rb") as f:
            header = f.read(8)
        assert header[:4] == b"\x89PNG"

    def test_different_output_paths(self, tmp_path):
        """Should work with different file names."""
        for name in ["a.png", "fig_test.png", "out.png"]:
            p = str(tmp_path / name)
            gf.generate_fig1_pipeline(p)
            assert os.path.isfile(p)

    def test_overwrites_existing_file(self, tmp_png):
        """Should overwrite an existing file without error."""
        with open(tmp_png, "w") as f:
            f.write("dummy")
        gf.generate_fig1_pipeline(tmp_png)
        with open(tmp_png, "rb") as f:
            assert f.read(4) == b"\x89PNG"


# =====================================================================
# generate_fig2_drr_algorithm
# =====================================================================

class TestGenerateFig2DrrAlgorithm:
    """Tests for the DRR algorithm 4-panel figure."""

    def test_creates_file(self, tmp_png):
        """Should save a PNG file."""
        gf.generate_fig2_drr_algorithm(tmp_png)
        assert os.path.isfile(tmp_png)

    def test_file_not_empty(self, tmp_png):
        """Saved file should have non-zero size."""
        gf.generate_fig2_drr_algorithm(tmp_png)
        assert os.path.getsize(tmp_png) > 1000  # At least 1KB for a real figure

    def test_is_valid_png(self, tmp_png):
        """Output should be valid PNG."""
        gf.generate_fig2_drr_algorithm(tmp_png)
        with open(tmp_png, "rb") as f:
            assert f.read(4) == b"\x89PNG"

    def test_deterministic_output(self, tmp_path):
        """Two runs should produce identical files (uses fixed seed)."""
        p1 = str(tmp_path / "run1.png")
        p2 = str(tmp_path / "run2.png")
        gf.generate_fig2_drr_algorithm(p1)
        gf.generate_fig2_drr_algorithm(p2)
        s1 = os.path.getsize(p1)
        s2 = os.path.getsize(p2)
        assert s1 == s2


# =====================================================================
# generate_fig3_drr_variations
# =====================================================================

class TestGenerateFig3DrrVariations:
    """Tests for the DRR variations 8-panel figure."""

    def test_creates_file_no_csv(self, tmp_png):
        """Should create a figure even when no CSV/images exist."""
        gf.generate_fig3_drr_variations(tmp_png)
        assert os.path.isfile(tmp_png)

    def test_is_valid_png_no_data(self, tmp_png):
        """Output should be valid PNG even without real data."""
        gf.generate_fig3_drr_variations(tmp_png)
        with open(tmp_png, "rb") as f:
            assert f.read(4) == b"\x89PNG"

    def test_with_mock_csv_no_images(self, tmp_path, tmp_png):
        """Should handle CSV with rows but no matching image files."""
        # Create mock dataset structure
        dataset_dir = os.path.join(str(tmp_path), "dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        csv_path = os.path.join(dataset_dir, "dataset_summary.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "view_type", "flexion_deg"])
            writer.writeheader()
            writer.writerow({"filename": "img001.png", "view_type": "AP", "flexion_deg": "0"})
            writer.writerow({"filename": "img002.png", "view_type": "LAT", "flexion_deg": "90"})

        # Patch PROJECT_ROOT to use our temp dir
        with patch.object(gf, "PROJECT_ROOT", str(tmp_path)):
            gf.generate_fig3_drr_variations(tmp_png)
        assert os.path.isfile(tmp_png)

    def test_with_mock_csv_and_images(self, tmp_path, tmp_png):
        """Should load real images when CSV and files exist."""
        import cv2

        dataset_dir = os.path.join(str(tmp_path), "data", "yolo_dataset")
        images_dir = os.path.join(dataset_dir, "images", "train")
        os.makedirs(images_dir, exist_ok=True)

        csv_path = os.path.join(dataset_dir, "dataset_summary.csv")
        # Create a small gray image
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        cv2.imwrite(os.path.join(images_dir, "test_ap_0.png"), img)
        cv2.imwrite(os.path.join(images_dir, "test_lat_90.png"), img)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "view_type", "flexion_deg"])
            writer.writeheader()
            writer.writerow({"filename": "test_ap_0.png", "view_type": "AP", "flexion_deg": "0"})
            writer.writerow({"filename": "test_lat_90.png", "view_type": "LAT", "flexion_deg": "90"})

        with patch.object(gf, "PROJECT_ROOT", str(tmp_path)):
            gf.generate_fig3_drr_variations(tmp_png)
        assert os.path.isfile(tmp_png)
        assert os.path.getsize(tmp_png) > 0


# =====================================================================
# generate_fig4_bland_altman
# =====================================================================

class TestGenerateFig4BlandAltman:
    """Tests for the Bland-Altman plot figure."""

    def test_missing_csv_returns_early(self, tmp_png, capsys):
        """Should print warning and return when predictions CSV is missing."""
        with patch.object(gf, "PROJECT_ROOT", "/nonexistent"):
            gf.generate_fig4_bland_altman(tmp_png)
        assert not os.path.isfile(tmp_png)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_creates_figure_with_mock_data(self, tmp_path, tmp_png):
        """Should create figure when CSV exists and bland_altman is available."""
        import pandas as pd

        # Create mock predictions CSV
        results_dir = os.path.join(str(tmp_path), "results", "bland_altman")
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, "predictions.csv")

        np.random.seed(42)
        n = 50
        gt = np.random.uniform(10, 90, n)
        pred = gt + np.random.normal(0, 0.5, n)
        df = pd.DataFrame({"gt_flexion_deg": gt, "pred_flexion_deg": pred})
        df.to_csv(csv_path, index=False)

        # Mock bland_altman module
        fake_ba = _FakeBlandAltman()
        mock_module = MagicMock()
        mock_module.compute_bland_altman.return_value = fake_ba

        with patch.object(gf, "PROJECT_ROOT", str(tmp_path)):
            with patch.dict(sys.modules, {"bland_altman": mock_module}):
                gf.generate_fig4_bland_altman(tmp_png)

        assert os.path.isfile(tmp_png)
        with open(tmp_png, "rb") as f:
            assert f.read(4) == b"\x89PNG"

    def test_figure_size_reasonable(self, tmp_path, tmp_png):
        """Generated figure should be a reasonable file size."""
        import pandas as pd

        results_dir = os.path.join(str(tmp_path), "results", "bland_altman")
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, "predictions.csv")

        np.random.seed(42)
        n = 30
        gt = np.random.uniform(10, 90, n)
        pred = gt + np.random.normal(0, 0.5, n)
        pd.DataFrame({"gt_flexion_deg": gt, "pred_flexion_deg": pred}).to_csv(csv_path, index=False)

        fake_ba = _FakeBlandAltman(n=n)
        mock_module = MagicMock()
        mock_module.compute_bland_altman.return_value = fake_ba

        with patch.object(gf, "PROJECT_ROOT", str(tmp_path)):
            with patch.dict(sys.modules, {"bland_altman": mock_module}):
                gf.generate_fig4_bland_altman(tmp_png)

        size = os.path.getsize(tmp_png)
        assert size > 5000  # At least 5KB for a scatter plot


# =====================================================================
# generate_fig4_bland_altman_v6
# =====================================================================

class TestGenerateFig4BlandAltmanV6:
    """Tests for the v6 Bland-Altman 2-panel figure."""

    def test_missing_csv_returns_early(self, tmp_png, capsys):
        """Should print warning and return when v6 CSV is missing."""
        with patch.object(gf, "PROJECT_ROOT", "/nonexistent"):
            gf.generate_fig4_bland_altman_v6(tmp_png)
        assert not os.path.isfile(tmp_png)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_creates_two_panel_figure(self, tmp_path, tmp_png):
        """Should create a 2-panel figure (GT vs Pred + Bland-Altman)."""
        import pandas as pd

        results_dir = os.path.join(str(tmp_path), "results", "bland_altman")
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, "predictions_v6.csv")

        np.random.seed(42)
        n = 80
        gt = np.random.uniform(5, 85, n)
        pred = gt + np.random.normal(0.1, 0.4, n)
        pd.DataFrame({"gt_flexion_deg": gt, "pred_flexion_deg": pred}).to_csv(csv_path, index=False)

        fake_ba = _FakeBlandAltman(n=n)
        mock_module = MagicMock()
        mock_module.compute_bland_altman.return_value = fake_ba

        with patch.object(gf, "PROJECT_ROOT", str(tmp_path)):
            with patch.dict(sys.modules, {"bland_altman": mock_module}):
                gf.generate_fig4_bland_altman_v6(tmp_png)

        assert os.path.isfile(tmp_png)
        with open(tmp_png, "rb") as f:
            assert f.read(4) == b"\x89PNG"
        # v6 2-panel figure should be larger than single-panel
        assert os.path.getsize(tmp_png) > 5000


# =====================================================================
# main() dispatcher
# =====================================================================

class TestMain:
    """Tests for the main() CLI dispatcher."""

    def test_all_figures_dispatched(self, tmp_path):
        """With no --fig arg, all 5 generators should be called."""
        out_dir = str(tmp_path / "figures")
        with patch("sys.argv", ["generate_figures.py", "--out_dir", out_dir]):
            with patch.object(gf, "generate_fig1_pipeline") as m1, \
                 patch.object(gf, "generate_fig2_drr_algorithm") as m2, \
                 patch.object(gf, "generate_fig3_drr_variations") as m3, \
                 patch.object(gf, "generate_fig4_bland_altman") as m4, \
                 patch.object(gf, "generate_fig4_bland_altman_v6") as m5:
                gf.main()
        assert m1.call_count == 1
        assert m2.call_count == 1
        assert m3.call_count == 1
        assert m4.call_count == 1
        assert m5.call_count == 1

    def test_single_figure_selection(self, tmp_path):
        """--fig 2 should only call generate_fig2_drr_algorithm."""
        out_dir = str(tmp_path / "figures")
        with patch("sys.argv", ["generate_figures.py", "--out_dir", out_dir, "--fig", "2"]):
            with patch.object(gf, "generate_fig1_pipeline") as m1, \
                 patch.object(gf, "generate_fig2_drr_algorithm") as m2, \
                 patch.object(gf, "generate_fig3_drr_variations") as m3, \
                 patch.object(gf, "generate_fig4_bland_altman") as m4, \
                 patch.object(gf, "generate_fig4_bland_altman_v6") as m5:
                gf.main()
        assert m1.call_count == 0
        assert m2.call_count == 1
        assert m3.call_count == 0
        assert m4.call_count == 0
        assert m5.call_count == 0

    def test_creates_output_directory(self, tmp_path):
        """Should create the output directory if it doesn't exist."""
        out_dir = str(tmp_path / "new_dir" / "figures")
        assert not os.path.isdir(out_dir)
        with patch("sys.argv", ["generate_figures.py", "--out_dir", out_dir]):
            with patch.object(gf, "generate_fig1_pipeline"), \
                 patch.object(gf, "generate_fig2_drr_algorithm"), \
                 patch.object(gf, "generate_fig3_drr_variations"), \
                 patch.object(gf, "generate_fig4_bland_altman"), \
                 patch.object(gf, "generate_fig4_bland_altman_v6"):
                gf.main()
        assert os.path.isdir(out_dir)

    def test_output_paths_correct(self, tmp_path):
        """Each generator should receive the correct file path."""
        out_dir = str(tmp_path / "figures")
        with patch("sys.argv", ["generate_figures.py", "--out_dir", out_dir]):
            with patch.object(gf, "generate_fig1_pipeline") as m1, \
                 patch.object(gf, "generate_fig2_drr_algorithm") as m2, \
                 patch.object(gf, "generate_fig3_drr_variations") as m3, \
                 patch.object(gf, "generate_fig4_bland_altman") as m4, \
                 patch.object(gf, "generate_fig4_bland_altman_v6") as m5:
                gf.main()
        assert m1.call_args[0][0] == os.path.join(out_dir, "fig1_pipeline.png")
        assert m2.call_args[0][0] == os.path.join(out_dir, "fig2_drr_algorithm.png")
        assert m3.call_args[0][0] == os.path.join(out_dir, "fig3_drr_variations.png")
        assert m4.call_args[0][0] == os.path.join(out_dir, "fig4_bland_altman.png")
        assert m5.call_args[0][0] == os.path.join(out_dir, "fig4_bland_altman_v6.png")

    def test_generator_exception_caught(self, tmp_path, capsys):
        """If a generator raises, main() should catch and print ERROR."""
        out_dir = str(tmp_path / "figures")
        with patch("sys.argv", ["generate_figures.py", "--out_dir", out_dir, "--fig", "1"]):
            with patch.object(gf, "generate_fig1_pipeline", side_effect=RuntimeError("boom")):
                gf.main()
        captured = capsys.readouterr()
        assert "ERROR" in captured.out

    def test_fig5_calls_v6(self, tmp_path):
        """--fig 5 should call generate_fig4_bland_altman_v6."""
        out_dir = str(tmp_path / "figures")
        with patch("sys.argv", ["generate_figures.py", "--out_dir", out_dir, "--fig", "5"]):
            with patch.object(gf, "generate_fig1_pipeline") as m1, \
                 patch.object(gf, "generate_fig2_drr_algorithm") as m2, \
                 patch.object(gf, "generate_fig3_drr_variations") as m3, \
                 patch.object(gf, "generate_fig4_bland_altman") as m4, \
                 patch.object(gf, "generate_fig4_bland_altman_v6") as m5:
                gf.main()
        assert m5.call_count == 1
        assert m1.call_count == 0


# =====================================================================
# PROJECT_ROOT constant
# =====================================================================

class TestConstants:
    """Tests for module-level constants."""

    def test_project_root_is_absolute(self):
        """PROJECT_ROOT should be an absolute path."""
        assert os.path.isabs(gf.PROJECT_ROOT)

    def test_project_root_points_to_project(self):
        """PROJECT_ROOT should end with ElbowVision."""
        assert gf.PROJECT_ROOT.endswith("ElbowVision")


# =====================================================================
# rcParams configuration
# =====================================================================

class TestMatplotlibConfig:
    """Tests for matplotlib rcParams configuration."""

    def test_font_family_is_serif(self):
        """Font family should be set to serif for publication quality."""
        import matplotlib.pyplot as plt
        # Re-import to re-apply rcParams (other tests may modify them)
        import importlib
        importlib.reload(gf)
        assert "serif" in plt.rcParams["font.family"]

    def test_dpi_is_300(self):
        """DPI should be 300 for publication quality."""
        import matplotlib.pyplot as plt
        assert plt.rcParams["figure.dpi"] == 300

    def test_savefig_bbox_tight(self):
        """savefig.bbox should be tight."""
        import matplotlib.pyplot as plt
        assert plt.rcParams["savefig.bbox"] == "tight"


# =====================================================================
# Integration: fig1 and fig2 produce distinct figures
# =====================================================================

class TestFigureDistinctness:
    """Verify different figures produce different outputs."""

    def test_fig1_and_fig2_differ(self, tmp_path):
        """Fig1 and Fig2 should produce different file sizes (different content)."""
        p1 = str(tmp_path / "fig1.png")
        p2 = str(tmp_path / "fig2.png")
        gf.generate_fig1_pipeline(p1)
        gf.generate_fig2_drr_algorithm(p2)
        # Different figures should have different sizes
        s1 = os.path.getsize(p1)
        s2 = os.path.getsize(p2)
        assert s1 != s2


# =====================================================================
# Edge cases
# =====================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_fig3_with_empty_csv(self, tmp_path, tmp_png):
        """Should handle CSV with headers but no data rows."""
        dataset_dir = os.path.join(str(tmp_path), "data", "yolo_dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        csv_path = os.path.join(dataset_dir, "dataset_summary.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "view_type", "flexion_deg"])
            writer.writeheader()
            # No data rows

        with patch.object(gf, "PROJECT_ROOT", str(tmp_path)):
            gf.generate_fig3_drr_variations(tmp_png)
        assert os.path.isfile(tmp_png)

    def test_fig1_closes_figure(self, tmp_png):
        """Should close the figure after saving to avoid memory leaks."""
        import matplotlib.pyplot as plt
        before = len(plt.get_fignums())
        gf.generate_fig1_pipeline(tmp_png)
        after = len(plt.get_fignums())
        assert after == before  # No leaked figures

    def test_fig2_closes_figure(self, tmp_png):
        """Should close the figure after saving."""
        import matplotlib.pyplot as plt
        before = len(plt.get_fignums())
        gf.generate_fig2_drr_algorithm(tmp_png)
        after = len(plt.get_fignums())
        assert after == before

    def test_fig3_closes_figure(self, tmp_png):
        """Should close the figure after saving."""
        import matplotlib.pyplot as plt
        before = len(plt.get_fignums())
        gf.generate_fig3_drr_variations(tmp_png)
        after = len(plt.get_fignums())
        assert after == before

    def test_placeholder_text_italic(self):
        """Placeholder text should be italic."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        gf._draw_placeholder(ax)
        txt = ax.texts[0]
        assert txt.get_fontstyle() == "italic"
        plt.close(fig)

    def test_draw_3d_box_negative_coords(self):
        """Should handle negative coordinates."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        gf._draw_3d_box(ax, -5, -5, 3, 3, "#AABBCC")
        plt.close(fig)
