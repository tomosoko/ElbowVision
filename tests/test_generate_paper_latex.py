"""
Unit tests for generate_paper_latex.py

Covers:
  _latex_escape()               — pure string escaping
  _table_wrap()                 — pure LaTeX table wrapper
  gen_table_drr_dataset()       — static table (no file reads)
  gen_table1_bland_altman()     — parses results/bland_altman/summary.txt
  gen_table1b_loo()             — parses results/self_test_loo/self_test_summary.txt
  gen_table2_method_comparison()— parses results/method_comparison/comparison.json (or skips)
  gen_table3_metric_comparison()— parses results/metric_comparison/metric_comparison.csv
  gen_table_robustness()        — parses results/robustness/robustness_results.csv (or skips)

Strategy:
  - Pure functions tested directly.
  - File-writing functions use tmp_path fixtures for output.
  - Fallback / skip behaviour tested by overriding mod._PROJECT_ROOT to a
    tmp directory that contains no result files.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "generate_paper_latex.py"
)


def _load_module(project_root_override: Path | None = None):
    """Import generate_paper_latex fresh; optionally patch its _PROJECT_ROOT."""
    spec = importlib.util.spec_from_file_location("generate_paper_latex", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if project_root_override is not None:
        mod._PROJECT_ROOT = project_root_override
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# _latex_escape
# ─────────────────────────────────────────────────────────────────────────────

class TestLatexEscape:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.fn = _load_module()._latex_escape

    def test_underscore_escaped(self):
        assert self.fn("foo_bar") == r"foo\_bar"

    def test_percent_escaped(self):
        assert self.fn("50%") == r"50\%"

    def test_ampersand_escaped(self):
        assert self.fn("A&B") == r"A\&B"

    def test_all_three_specials(self):
        result = self.fn("a_b%c&d")
        assert r"\_" in result
        assert r"\%" in result
        assert r"\&" in result

    def test_no_specials_unchanged(self):
        assert self.fn("hello world") == "hello world"

    def test_empty_string(self):
        assert self.fn("") == ""

    def test_integer_coerced_to_str(self):
        assert self.fn(42) == "42"

    def test_multiple_underscores(self):
        assert self.fn("a_b_c") == r"a\_b\_c"

    def test_multiple_percent(self):
        assert self.fn("10%20%") == r"10\%20\%"


# ─────────────────────────────────────────────────────────────────────────────
# _table_wrap
# ─────────────────────────────────────────────────────────────────────────────

class TestTableWrap:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.fn = _load_module()._table_wrap

    def test_contains_begin_table(self):
        assert r"\begin{table}" in self.fn("c", "cap", "lbl")

    def test_contains_end_table(self):
        assert r"\end{table}" in self.fn("c", "cap", "lbl")

    def test_default_placement_htb(self):
        assert "[htb]" in self.fn("c", "cap", "lbl")

    def test_custom_placement(self):
        assert "[!ht]" in self.fn("c", "cap", "lbl", placement="!ht")

    def test_centering_present(self):
        assert r"\centering" in self.fn("c", "cap", "lbl")

    def test_caption_present(self):
        assert "My Caption" in self.fn("c", "My Caption", "lbl")

    def test_label_present(self):
        assert "tab:mytest" in self.fn("c", "cap", "tab:mytest")

    def test_content_present(self):
        assert "INNER CONTENT" in self.fn("INNER CONTENT", "cap", "lbl")

    def test_structure_order(self):
        out = self.fn("content", "caption", "label")
        begin_pos   = out.index(r"\begin{table}")
        center_pos  = out.index(r"\centering")
        caption_pos = out.index(r"\caption")
        end_pos     = out.index(r"\end{table}")
        assert begin_pos < center_pos < caption_pos < end_pos


# ─────────────────────────────────────────────────────────────────────────────
# gen_table_drr_dataset  (static — no file reads)
# ─────────────────────────────────────────────────────────────────────────────

class TestGenTableDrrDataset:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.fn      = _load_module().gen_table_drr_dataset
        self.out_dir = tmp_path
        self.out_file = tmp_path / "table0_drr_dataset.tex"

    def _content(self) -> str:
        self.fn(self.out_dir)
        return self.out_file.read_text()

    def test_creates_output_file(self):
        self.fn(self.out_dir)
        assert self.out_file.exists()

    def test_contains_angle_range(self):
        assert "90--180" in self._content()

    def test_contains_augmentation_count(self):
        assert "15" in self._content()

    def test_contains_total_images(self):
        assert "1,365" in self._content()

    def test_contains_image_size(self):
        assert "256" in self._content()

    def test_contains_beer_lambert(self):
        assert "Beer" in self._content()

    def test_contains_label(self):
        assert "tab:drr_dataset" in self._content()

    def test_contains_table_environment(self):
        c = self._content()
        assert r"\begin{table}" in c
        assert r"\end{table}" in c

    def test_contains_tabular_environment(self):
        assert r"\begin{tabular}" in self._content()

    def test_idempotent(self):
        self.fn(self.out_dir)
        c1 = self.out_file.read_text()
        self.fn(self.out_dir)
        c2 = self.out_file.read_text()
        assert c1 == c2


# ─────────────────────────────────────────────────────────────────────────────
# gen_table1_bland_altman
# ─────────────────────────────────────────────────────────────────────────────

class TestGenTable1BlandAltman:
    """summary.txt exists → parses actual values."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.mod      = _load_module()
        self.out_dir  = tmp_path
        self.out_file = tmp_path / "table1_drr_bland_altman.tex"

    def _content(self) -> str:
        self.mod.gen_table1_bland_altman(self.out_dir)
        return self.out_file.read_text()

    def test_creates_output_file(self):
        self.mod.gen_table1_bland_altman(self.out_dir)
        assert self.out_file.exists()

    def test_parsed_n_value(self):
        # summary.txt: (n=273)
        assert "273" in self._content()

    def test_parsed_mae_value(self):
        # summary.txt: MAE=1.412  →  formatted "1.41"
        assert "1.41" in self._content()

    def test_parsed_icc_value(self):
        # summary.txt: ICC(3,1)=0.9989  →  "0.999"
        assert "0.999" in self._content()

    def test_clinical_pass(self):
        assert "PASS" in self._content()

    def test_mae_in_bold(self):
        assert r"\textbf{" in self._content()

    def test_contains_label(self):
        assert "tab:bland_altman_drr" in self._content()

    def test_fallback_when_no_summary(self, tmp_path):
        """With a fresh tmp root (no results files), hardcoded defaults are used."""
        mod = _load_module(project_root_override=tmp_path)
        out = tmp_path / "out"
        out.mkdir()
        mod.gen_table1_bland_altman(out)
        content = (out / "table1_drr_bland_altman.tex").read_text()
        # Hardcoded defaults: n=273, mae_val=1.41
        assert "273" in content
        assert "1.41" in content


# ─────────────────────────────────────────────────────────────────────────────
# gen_table1b_loo
# ─────────────────────────────────────────────────────────────────────────────

class TestGenTable1bLoo:
    """LOO summary files exist → parses actual values."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.mod      = _load_module()
        self.out_dir  = tmp_path
        self.out_file = tmp_path / "table1b_loo_validation.tex"

    def _content(self) -> str:
        self.mod.gen_table1b_loo(self.out_dir)
        return self.out_file.read_text()

    def test_creates_output_file(self):
        self.mod.gen_table1b_loo(self.out_dir)
        assert self.out_file.exists()

    def test_parsed_loo_n(self):
        # self_test_summary.txt: n = 121
        assert "121" in self._content()

    def test_parsed_loo_mae(self):
        # self_test_summary.txt: MAE = 0.545
        assert "0.545" in self._content()

    def test_standard_drr_row(self):
        # Hardcoded standard (DRR≡query) row: MAE=0.015
        assert "0.015" in self._content()

    def test_contains_label(self):
        assert "tab:loo_validation" in self._content()

    def test_fallback_when_no_summary(self, tmp_path):
        """No LOO files → hardcoded defaults."""
        mod = _load_module(project_root_override=tmp_path)
        out = tmp_path / "out"
        out.mkdir()
        mod.gen_table1b_loo(out)
        content = (out / "table1b_loo_validation.tex").read_text()
        # Default: n=121, MAE=0.545
        assert "121" in content
        assert "0.545" in content


# ─────────────────────────────────────────────────────────────────────────────
# gen_table2_method_comparison
# ─────────────────────────────────────────────────────────────────────────────

class TestGenTable2MethodComparisonWithJson:
    """comparison.json exists on disk → generates table."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.mod      = _load_module()
        self.out_dir  = tmp_path
        self.out_file = tmp_path / "table2_method_comparison.tex"

    def _content(self) -> str:
        self.mod.gen_table2_method_comparison(self.out_dir)
        return self.out_file.read_text()

    def test_creates_output_file(self):
        self.mod.gen_table2_method_comparison(self.out_dir)
        assert self.out_file.exists()

    def test_contains_similarity_matching_header(self):
        assert "Similarity Matching" in self._content()

    def test_contains_convnext_header(self):
        assert "ConvNeXt" in self._content()

    def test_contains_label(self):
        assert "tab:method_comparison" in self._content()


class TestGenTable2SkipsWhenJsonMissing:
    def test_no_file_created_when_json_missing(self, tmp_path):
        mod = _load_module(project_root_override=tmp_path)
        out = tmp_path / "out"
        out.mkdir()
        mod.gen_table2_method_comparison(out)
        assert not (out / "table2_method_comparison.tex").exists()


# ─────────────────────────────────────────────────────────────────────────────
# gen_table3_metric_comparison
# ─────────────────────────────────────────────────────────────────────────────

class TestGenTable3MetricComparison:
    """metric_comparison.csv exists → parses actual metric data."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.mod      = _load_module()
        self.out_dir  = tmp_path
        self.out_file = tmp_path / "table3_metric_comparison.tex"

    def _content(self) -> str:
        self.mod.gen_table3_metric_comparison(self.out_dir)
        return self.out_file.read_text()

    def test_creates_output_file(self):
        self.mod.gen_table3_metric_comparison(self.out_dir)
        assert self.out_file.exists()

    def test_contains_ncc_metric(self):
        assert "NCC" in self._content()

    def test_contains_combined_metric(self):
        assert "Combined" in self._content()

    def test_contains_nmi_metric(self):
        assert "NMI" in self._content()

    def test_contains_label(self):
        assert "tab:metric_comparison" in self._content()

    def test_fallback_when_csv_missing(self, tmp_path):
        """No CSV → uses fallback_bias/mae hardcoded values, still creates file."""
        mod = _load_module(project_root_override=tmp_path)
        out = tmp_path / "out"
        out.mkdir()
        mod.gen_table3_metric_comparison(out)
        content = (out / "table3_metric_comparison.tex").read_text()
        # Fallback NCC bias = "+5.0"
        assert "5.0" in content


# ─────────────────────────────────────────────────────────────────────────────
# gen_table_robustness
# ─────────────────────────────────────────────────────────────────────────────

class TestGenTableRobustness:
    """robustness_results.csv exists → generates table."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.mod      = _load_module()
        self.out_dir  = tmp_path
        self.out_file = tmp_path / "tableS1_robustness.tex"

    def _content(self) -> str:
        self.mod.gen_table_robustness(self.out_dir)
        return self.out_file.read_text()

    def test_creates_output_file(self):
        self.mod.gen_table_robustness(self.out_dir)
        assert self.out_file.exists()

    def test_contains_perturbation_name(self):
        # CSV has "gaussian_noise" → latex title-cases to "Gaussian Noise"
        c = self._content().lower()
        assert "gaussian" in c

    def test_contains_label(self):
        assert "tab:robustness" in self._content()

    def test_contains_table_environment(self):
        c = self._content()
        assert r"\begin{table}" in c

    def test_no_file_when_csv_missing(self, tmp_path):
        mod = _load_module(project_root_override=tmp_path)
        out = tmp_path / "out"
        out.mkdir()
        mod.gen_table_robustness(out)
        assert not (out / "tableS1_robustness.tex").exists()
