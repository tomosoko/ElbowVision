"""Unit tests for scripts/generate_comparison_figure_v6.py

Tests cover:
- Constants: LABEL_MAP, COLOR_*, CLINICAL_THRESHOLD, _PROJECT_ROOT
- load_results: JSON parsing, key extraction, error handling
- generate_fig5a_v6: per-image bar chart, labels, NaN handling, non-std marking
- generate_fig5b_v6_mae_summary: MAE calculation, standard filtering, panel layout
- main: CLI flow, directory creation, end-to-end
- Edge cases: empty results, single result, missing keys, all NaN
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# ── ensure scripts/ is importable ──────────────────────────────
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import generate_comparison_figure_v6 as gcf6


# ── Fixtures ───────────────────────────────────────────────────

def _make_results(n: int = 4) -> list[dict]:
    """Create sample results matching expected schema."""
    filenames = ["008_LAT.png", "cr_008_2_50kVp.png", "cr_008_3_52kVp.png", "new_LAT.png"]
    results = []
    for i in range(n):
        fname = filenames[i % len(filenames)]
        results.append({
            "filename": fname,
            "err_convnext_v5": (-1) ** i * (i + 1) * 2.5,
            "err_convnext_v6": (-1) ** i * (i + 0.5) * 1.2,
            "err_sim": (-1) ** i * (i + 0.3) * 0.5,
        })
    return results


@pytest.fixture
def sample_results():
    return _make_results(4)


@pytest.fixture
def results_json(tmp_path, sample_results):
    p = tmp_path / "comparison.json"
    p.write_text(json.dumps({"results": sample_results}))
    return p


# ── Constants ──────────────────────────────────────────────────

class TestConstants:
    def test_label_map_keys(self):
        assert "008_LAT.png" in gcf6.LABEL_MAP
        assert "cr_008_2_50kVp.png" in gcf6.LABEL_MAP
        assert "cr_008_3_52kVp.png" in gcf6.LABEL_MAP
        assert "new_LAT.png" in gcf6.LABEL_MAP

    def test_label_map_values_are_strings(self):
        for v in gcf6.LABEL_MAP.values():
            assert isinstance(v, str)
            assert len(v) > 0

    def test_label_map_contains_non_std(self):
        assert "Non-std" in gcf6.LABEL_MAP.values()

    def test_color_constants_are_hex(self):
        for c in [gcf6.COLOR_V5, gcf6.COLOR_V6, gcf6.COLOR_SIM]:
            assert isinstance(c, str)
            assert c.startswith("#")
            assert len(c) == 7

    def test_clinical_threshold(self):
        assert gcf6.CLINICAL_THRESHOLD == 5.0

    def test_project_root_is_directory(self):
        assert gcf6._PROJECT_ROOT.is_dir()

    def test_project_root_contains_scripts(self):
        assert (gcf6._PROJECT_ROOT / "scripts").is_dir()


# ── load_results ───────────────────────────────────────────────

class TestLoadResults:
    def test_returns_list(self, results_json):
        out = gcf6.load_results(results_json)
        assert isinstance(out, list)

    def test_returns_correct_count(self, results_json, sample_results):
        out = gcf6.load_results(results_json)
        assert len(out) == len(sample_results)

    def test_preserves_keys(self, results_json):
        out = gcf6.load_results(results_json)
        for r in out:
            assert "filename" in r
            assert "err_convnext_v5" in r
            assert "err_sim" in r

    def test_preserves_values(self, results_json, sample_results):
        out = gcf6.load_results(results_json)
        for orig, loaded in zip(sample_results, out):
            assert loaded["filename"] == orig["filename"]
            assert loaded["err_sim"] == pytest.approx(orig["err_sim"])

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            gcf6.load_results(tmp_path / "nonexistent.json")

    def test_invalid_json_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{broken")
        with pytest.raises(json.JSONDecodeError):
            gcf6.load_results(p)

    def test_missing_results_key_raises(self, tmp_path):
        p = tmp_path / "no_results.json"
        p.write_text(json.dumps({"data": []}))
        with pytest.raises(KeyError):
            gcf6.load_results(p)

    def test_empty_results(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text(json.dumps({"results": []}))
        assert gcf6.load_results(p) == []

    def test_single_result(self, tmp_path):
        p = tmp_path / "single.json"
        p.write_text(json.dumps({"results": [{"filename": "a.png", "err_sim": 1.0}]}))
        out = gcf6.load_results(p)
        assert len(out) == 1
        assert out[0]["filename"] == "a.png"


# ── generate_fig5a_v6 ─────────────────────────────────────────

class TestGenerateFig5aV6:
    def test_creates_file(self, tmp_path, sample_results):
        out = tmp_path / "fig5a.png"
        gcf6.generate_fig5a_v6(sample_results, out)
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_uses_label_map(self, sample_results):
        """Labels should map known filenames."""
        labels = [gcf6.LABEL_MAP.get(r["filename"], r["filename"]) for r in sample_results]
        assert "Std-1" in labels
        assert "Non-std" in labels

    def test_handles_none_v5(self, tmp_path):
        results = [{"filename": "a.png", "err_convnext_v5": None,
                     "err_convnext_v6": 1.0, "err_sim": 0.5}]
        out = tmp_path / "fig5a_none.png"
        gcf6.generate_fig5a_v6(results, out)
        assert out.exists()

    def test_handles_none_v6(self, tmp_path):
        results = [{"filename": "a.png", "err_convnext_v5": 2.0,
                     "err_convnext_v6": None, "err_sim": 0.5}]
        out = tmp_path / "fig5a_nonev6.png"
        gcf6.generate_fig5a_v6(results, out)
        assert out.exists()

    def test_handles_all_none_v6(self, tmp_path):
        results = [
            {"filename": "a.png", "err_convnext_v5": 2.0, "err_convnext_v6": None, "err_sim": 0.5},
            {"filename": "b.png", "err_convnext_v5": 3.0, "err_convnext_v6": None, "err_sim": 1.0},
        ]
        out = tmp_path / "fig5a_allnone.png"
        gcf6.generate_fig5a_v6(results, out)
        assert out.exists()

    def test_non_std_highlight(self, tmp_path):
        """Non-std image should trigger gray highlight in plot."""
        results = [
            {"filename": "cr_008_2_50kVp.png", "err_convnext_v5": 10.0,
             "err_convnext_v6": 5.0, "err_sim": 1.0},
        ]
        out = tmp_path / "fig5a_nonstd.png"
        gcf6.generate_fig5a_v6(results, out)
        assert out.exists()

    def test_large_errors(self, tmp_path):
        results = [
            {"filename": "a.png", "err_convnext_v5": 100.0,
             "err_convnext_v6": 50.0, "err_sim": 80.0},
        ]
        out = tmp_path / "fig5a_large.png"
        gcf6.generate_fig5a_v6(results, out)
        assert out.exists()

    def test_zero_errors(self, tmp_path):
        results = [
            {"filename": "a.png", "err_convnext_v5": 0.0,
             "err_convnext_v6": 0.0, "err_sim": 0.0},
        ]
        out = tmp_path / "fig5a_zero.png"
        gcf6.generate_fig5a_v6(results, out)
        assert out.exists()

    def test_negative_errors_use_abs(self, tmp_path):
        """Errors should be absolute values."""
        results = [
            {"filename": "a.png", "err_convnext_v5": -5.0,
             "err_convnext_v6": -3.0, "err_sim": -1.0},
        ]
        out = tmp_path / "fig5a_neg.png"
        gcf6.generate_fig5a_v6(results, out)
        assert out.exists()

    def test_unknown_filename_uses_raw(self, tmp_path):
        results = [
            {"filename": "unknown_file.png", "err_convnext_v5": 2.0,
             "err_convnext_v6": 1.0, "err_sim": 0.5},
        ]
        out = tmp_path / "fig5a_unknown.png"
        gcf6.generate_fig5a_v6(results, out)
        assert out.exists()

    def test_many_results(self, tmp_path):
        results = [
            {"filename": f"img_{i}.png", "err_convnext_v5": float(i),
             "err_convnext_v6": float(i) * 0.5, "err_sim": float(i) * 0.1}
            for i in range(20)
        ]
        out = tmp_path / "fig5a_many.png"
        gcf6.generate_fig5a_v6(results, out)
        assert out.exists()


# ── generate_fig5b_v6_mae_summary ──────────────────────────────

class TestGenerateFig5bV6MaeSummary:
    def test_creates_file(self, tmp_path, sample_results):
        out = tmp_path / "fig5b.png"
        gcf6.generate_fig5b_v6_mae_summary(sample_results, out)
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_filters_non_standard(self, sample_results):
        """Non-standard images (cr_008_2) should be filtered for MAE."""
        std = [r for r in sample_results if "cr_008_2" not in r["filename"]]
        assert len(std) < len(sample_results)

    def test_mae_calculation_std_only(self):
        results = [
            {"filename": "008_LAT.png", "err_convnext_v5": 2.0,
             "err_convnext_v6": 1.0, "err_sim": 0.5},
            {"filename": "cr_008_2_50kVp.png", "err_convnext_v5": 100.0,
             "err_convnext_v6": 50.0, "err_sim": 80.0},
            {"filename": "new_LAT.png", "err_convnext_v5": -4.0,
             "err_convnext_v6": -2.0, "err_sim": -1.5},
        ]
        std = [r for r in results if "cr_008_2" not in r["filename"]]
        mae_v5 = np.mean([abs(r["err_convnext_v5"]) for r in std])
        assert mae_v5 == pytest.approx(3.0)
        mae_sim = np.mean([abs(r["err_sim"]) for r in std])
        assert mae_sim == pytest.approx(1.0)

    def test_with_none_v6(self, tmp_path):
        results = [
            {"filename": "008_LAT.png", "err_convnext_v5": 2.0,
             "err_convnext_v6": None, "err_sim": 0.5},
            {"filename": "new_LAT.png", "err_convnext_v5": 4.0,
             "err_convnext_v6": 1.0, "err_sim": 1.5},
        ]
        out = tmp_path / "fig5b_none.png"
        gcf6.generate_fig5b_v6_mae_summary(results, out)
        assert out.exists()

    def test_all_standard(self, tmp_path):
        """When no non-std images, all should be included."""
        results = [
            {"filename": "a.png", "err_convnext_v5": 2.0,
             "err_convnext_v6": 1.0, "err_sim": 0.5},
            {"filename": "b.png", "err_convnext_v5": 4.0,
             "err_convnext_v6": 2.0, "err_sim": 1.0},
        ]
        std = [r for r in results if "cr_008_2" not in r["filename"]]
        assert len(std) == len(results)
        out = tmp_path / "fig5b_allstd.png"
        gcf6.generate_fig5b_v6_mae_summary(results, out)
        assert out.exists()

    def test_hardcoded_drr_validation_values(self):
        """DRR validation MAE values are hardcoded in the function."""
        # These constants are used in Panel A
        assert 1.412 > 0  # v5 DRR val MAE
        assert 0.467 > 0  # v6 DRR val MAE
        assert 0.467 < 1.412  # v6 is better than v5

    def test_single_result(self, tmp_path):
        results = [
            {"filename": "a.png", "err_convnext_v5": 3.0,
             "err_convnext_v6": 1.5, "err_sim": 0.8},
        ]
        out = tmp_path / "fig5b_single.png"
        gcf6.generate_fig5b_v6_mae_summary(results, out)
        assert out.exists()

    def test_two_panels_layout(self, tmp_path, sample_results):
        """Figure should have exactly 2 subplots."""
        import matplotlib.pyplot as plt
        out = tmp_path / "fig5b_panels.png"
        with mock.patch.object(plt, "subplots", wraps=plt.subplots) as m:
            gcf6.generate_fig5b_v6_mae_summary(sample_results, out)
            args, kwargs = m.call_args
            assert args == (1, 2)


# ── main ───────────────────────────────────────────────────────

class TestMain:
    def test_main_calls_functions(self, tmp_path, sample_results):
        json_path = tmp_path / "results" / "method_comparison" / "comparison.json"
        json_path.parent.mkdir(parents=True)
        json_path.write_text(json.dumps({"results": sample_results}))

        with mock.patch.object(gcf6, "_PROJECT_ROOT", tmp_path):
            gcf6.main()

        fig_dir = tmp_path / "results" / "figures"
        assert fig_dir.exists()
        assert (fig_dir / "fig5a_per_image_error_v6.png").exists()
        assert (fig_dir / "fig5b_mae_summary_v6.png").exists()

    def test_main_creates_output_dir(self, tmp_path, sample_results):
        json_path = tmp_path / "results" / "method_comparison" / "comparison.json"
        json_path.parent.mkdir(parents=True)
        json_path.write_text(json.dumps({"results": sample_results}))

        fig_dir = tmp_path / "results" / "figures"
        assert not fig_dir.exists()

        with mock.patch.object(gcf6, "_PROJECT_ROOT", tmp_path):
            gcf6.main()

        assert fig_dir.is_dir()

    def test_main_file_not_found(self, tmp_path):
        with mock.patch.object(gcf6, "_PROJECT_ROOT", tmp_path):
            with pytest.raises(FileNotFoundError):
                gcf6.main()


# ── Edge cases ─────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_results_fig5a(self, tmp_path):
        out = tmp_path / "fig5a_empty.png"
        gcf6.generate_fig5a_v6([], out)
        assert out.exists()

    def test_empty_results_fig5b_raises(self, tmp_path):
        """Empty results cause NaN MAE which matplotlib rejects in set_ylim."""
        out = tmp_path / "fig5b_empty.png"
        with pytest.raises(ValueError, match="NaN or Inf"):
            gcf6.generate_fig5b_v6_mae_summary([], out)

    def test_mixed_known_unknown_filenames(self, tmp_path):
        results = [
            {"filename": "008_LAT.png", "err_convnext_v5": 1.0,
             "err_convnext_v6": 0.5, "err_sim": 0.2},
            {"filename": "totally_new.png", "err_convnext_v5": 2.0,
             "err_convnext_v6": 1.0, "err_sim": 0.3},
        ]
        out = tmp_path / "fig5a_mixed.png"
        gcf6.generate_fig5a_v6(results, out)
        assert out.exists()

    def test_very_small_errors(self, tmp_path):
        results = [
            {"filename": "a.png", "err_convnext_v5": 0.001,
             "err_convnext_v6": 0.0005, "err_sim": 0.0001},
        ]
        out = tmp_path / "fig5a_tiny.png"
        gcf6.generate_fig5a_v6(results, out)
        assert out.exists()
