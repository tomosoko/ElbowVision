"""
Unit tests for scripts/calibrate_confidence.py

Tests cover:
  - load_results: CSV parsing, missing-field handling, column aliases
  - compute_threshold_metrics: rejection/pass logic, edge cases
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.calibrate_confidence import compute_threshold_metrics, load_results


# ─── helpers ─────────────────────────────────────────────────────────────────

def _write_csv(tmp_path: Path, rows: list[dict], filename: str = "results.csv") -> str:
    """Write a list of dicts to a temp CSV and return the path string."""
    if not rows:
        path = tmp_path / filename
        path.write_text("peak_ncc,sharpness,error\n", encoding="utf-8")
        return str(path)
    fieldnames = list(rows[0].keys())
    path = tmp_path / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def _make_rows(n: int, peak_ncc: float = 0.8, sharpness: float = 2.0,
               error: float = 1.0) -> list[dict]:
    return [{"peak_ncc": peak_ncc, "sharpness": sharpness, "error": error}
            for _ in range(n)]


# ─── load_results ─────────────────────────────────────────────────────────────

class TestLoadResults:
    """Tests for load_results(csv_path)."""

    def test_returns_list(self, tmp_path):
        path = _write_csv(tmp_path, _make_rows(3))
        result = load_results(path)
        assert isinstance(result, list)

    def test_empty_csv_returns_empty_list(self, tmp_path):
        path = _write_csv(tmp_path, [])
        assert load_results(path) == []

    def test_basic_row_parsed(self, tmp_path):
        path = _write_csv(tmp_path, [{"peak_ncc": "0.75", "sharpness": "1.5", "error": "2.3"}])
        result = load_results(path)
        assert len(result) == 1
        assert result[0]["peak_ncc"] == pytest.approx(0.75)
        assert result[0]["sharpness"] == pytest.approx(1.5)
        assert result[0]["error"] == pytest.approx(2.3)

    def test_multiple_rows_parsed(self, tmp_path):
        rows = [{"peak_ncc": str(i * 0.1), "sharpness": "1.0", "error": str(i)} for i in range(5)]
        path = _write_csv(tmp_path, rows)
        result = load_results(path)
        assert len(result) == 5

    def test_source_field_is_file_stem(self, tmp_path):
        path = _write_csv(tmp_path, _make_rows(1), filename="myresults.csv")
        result = load_results(path)
        assert result[0]["source"] == "myresults"

    def test_error_deg_column_alias(self, tmp_path):
        """'error_deg' column should be read as 'error'."""
        rows = [{"peak_ncc": "0.5", "sharpness": "1.0", "error_deg": "3.5"}]
        path = _write_csv(tmp_path, rows)
        result = load_results(path)
        assert len(result) == 1
        assert result[0]["error"] == pytest.approx(3.5)

    def test_error_column_takes_priority_over_error_deg(self, tmp_path):
        """When both 'error' and 'error_deg' exist, 'error' wins."""
        rows = [{"peak_ncc": "0.5", "sharpness": "1.0", "error": "2.0", "error_deg": "9.0"}]
        path = _write_csv(tmp_path, rows)
        result = load_results(path)
        assert result[0]["error"] == pytest.approx(2.0)

    def test_row_missing_peak_ncc_skipped(self, tmp_path):
        rows = [{"sharpness": "1.0", "error": "2.0"}]
        path = _write_csv(tmp_path, rows)
        result = load_results(path)
        assert result == []

    def test_row_missing_sharpness_skipped(self, tmp_path):
        rows = [{"peak_ncc": "0.5", "error": "2.0"}]
        path = _write_csv(tmp_path, rows)
        result = load_results(path)
        assert result == []

    def test_row_missing_error_column_skipped(self, tmp_path):
        rows = [{"peak_ncc": "0.5", "sharpness": "1.0"}]
        path = _write_csv(tmp_path, rows)
        result = load_results(path)
        assert result == []

    def test_row_with_empty_peak_ncc_skipped(self, tmp_path):
        rows = [{"peak_ncc": "", "sharpness": "1.0", "error": "2.0"}]
        path = _write_csv(tmp_path, rows)
        result = load_results(path)
        assert result == []

    def test_row_with_empty_sharpness_skipped(self, tmp_path):
        rows = [{"peak_ncc": "0.5", "sharpness": "", "error": "2.0"}]
        path = _write_csv(tmp_path, rows)
        result = load_results(path)
        assert result == []

    def test_row_with_empty_error_skipped(self, tmp_path):
        rows = [{"peak_ncc": "0.5", "sharpness": "1.0", "error": ""}]
        path = _write_csv(tmp_path, rows)
        result = load_results(path)
        assert result == []

    def test_row_with_non_numeric_peak_ncc_skipped(self, tmp_path):
        rows = [{"peak_ncc": "abc", "sharpness": "1.0", "error": "2.0"}]
        path = _write_csv(tmp_path, rows)
        result = load_results(path)
        assert result == []

    def test_row_with_non_numeric_error_skipped(self, tmp_path):
        rows = [{"peak_ncc": "0.5", "sharpness": "1.0", "error": "N/A"}]
        path = _write_csv(tmp_path, rows)
        result = load_results(path)
        assert result == []

    def test_mixed_valid_invalid_rows(self, tmp_path):
        rows = [
            {"peak_ncc": "0.5", "sharpness": "1.0", "error": "2.0"},
            {"peak_ncc": "",    "sharpness": "1.0", "error": "2.0"},  # invalid
            {"peak_ncc": "0.7", "sharpness": "1.2", "error": "1.5"},
        ]
        path = _write_csv(tmp_path, rows)
        result = load_results(path)
        assert len(result) == 2

    def test_float_values_stored_as_float(self, tmp_path):
        rows = [{"peak_ncc": "0.123456", "sharpness": "3.14", "error": "0.999"}]
        path = _write_csv(tmp_path, rows)
        result = load_results(path)
        assert isinstance(result[0]["peak_ncc"], float)
        assert isinstance(result[0]["sharpness"], float)
        assert isinstance(result[0]["error"], float)

    def test_integer_values_parsed(self, tmp_path):
        rows = [{"peak_ncc": "1", "sharpness": "2", "error": "3"}]
        path = _write_csv(tmp_path, rows)
        result = load_results(path)
        assert result[0]["peak_ncc"] == pytest.approx(1.0)
        assert result[0]["error"] == pytest.approx(3.0)

    def test_negative_error_parsed(self, tmp_path):
        """Negative values are valid floats and should be parsed."""
        rows = [{"peak_ncc": "0.5", "sharpness": "-1.0", "error": "-0.5"}]
        path = _write_csv(tmp_path, rows)
        result = load_results(path)
        assert len(result) == 1
        assert result[0]["error"] == pytest.approx(-0.5)

    def test_extra_columns_ignored(self, tmp_path):
        rows = [{"peak_ncc": "0.5", "sharpness": "1.0", "error": "2.0", "extra_col": "ignored"}]
        path = _write_csv(tmp_path, rows)
        result = load_results(path)
        assert len(result) == 1

    def test_zero_error_parsed(self, tmp_path):
        rows = [{"peak_ncc": "0.9", "sharpness": "3.0", "error": "0.0"}]
        path = _write_csv(tmp_path, rows)
        result = load_results(path)
        assert result[0]["error"] == pytest.approx(0.0)


# ─── compute_threshold_metrics ────────────────────────────────────────────────

class TestComputeThresholdMetrics:
    """Tests for compute_threshold_metrics(rows, score_col, thresholds, error_threshold)."""

    def _make_rows(self, data: list[tuple[float, float]],
                   score_col: str = "peak_ncc") -> list[dict]:
        """Build rows from (score, error) tuples."""
        return [{score_col: s, "error": e} for s, e in data]

    # ── return structure ──────────────────────────────────────────────────────

    def test_returns_list(self):
        rows = self._make_rows([(0.5, 1.0)])
        result = compute_threshold_metrics(rows, "peak_ncc", [0.3])
        assert isinstance(result, list)

    def test_one_threshold_one_result(self):
        rows = self._make_rows([(0.5, 1.0)])
        result = compute_threshold_metrics(rows, "peak_ncc", [0.3])
        assert len(result) == 1

    def test_multiple_thresholds_length(self):
        rows = self._make_rows([(0.5, 1.0)] * 5)
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = compute_threshold_metrics(rows, "peak_ncc", thresholds)
        assert len(result) == len(thresholds)

    def test_result_keys_present(self):
        rows = self._make_rows([(0.5, 1.0)])
        result = compute_threshold_metrics(rows, "peak_ncc", [0.3])[0]
        expected_keys = {"threshold", "rejection_rate", "true_reject_rate",
                         "false_pass_rate", "n_rejected", "n_accepted", "mae_accepted"}
        assert expected_keys.issubset(result.keys())

    def test_threshold_value_stored(self):
        rows = self._make_rows([(0.5, 1.0)])
        result = compute_threshold_metrics(rows, "peak_ncc", [0.42])[0]
        assert result["threshold"] == pytest.approx(0.42, abs=1e-3)

    # ── rejection logic ───────────────────────────────────────────────────────

    def test_all_below_threshold_fully_rejected(self):
        rows = self._make_rows([(0.2, 1.0), (0.3, 2.0), (0.1, 0.5)])
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5])[0]
        assert result["rejection_rate"] == pytest.approx(1.0)
        assert result["n_rejected"] == 3
        assert result["n_accepted"] == 0

    def test_all_above_threshold_none_rejected(self):
        rows = self._make_rows([(0.8, 1.0), (0.9, 2.0), (0.7, 0.5)])
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5])[0]
        assert result["rejection_rate"] == pytest.approx(0.0)
        assert result["n_rejected"] == 0
        assert result["n_accepted"] == 3

    def test_partial_rejection(self):
        rows = self._make_rows([(0.3, 1.0), (0.7, 2.0), (0.5, 0.5), (0.8, 1.5)])
        result = compute_threshold_metrics(rows, "peak_ncc", [0.6])[0]
        # score >= 0.6 → accepted: 0.7, 0.8; rejected: 0.3, 0.5
        assert result["n_rejected"] == 2
        assert result["n_accepted"] == 2
        assert result["rejection_rate"] == pytest.approx(0.5)

    def test_score_exactly_at_threshold_is_accepted(self):
        rows = self._make_rows([(0.5, 1.0)])
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5])[0]
        # score >= threshold → accepted
        assert result["n_accepted"] == 1
        assert result["n_rejected"] == 0

    def test_score_just_below_threshold_is_rejected(self):
        rows = self._make_rows([(0.499, 1.0)])
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5])[0]
        assert result["n_rejected"] == 1
        assert result["n_accepted"] == 0

    # ── true_reject_rate ──────────────────────────────────────────────────────

    def test_true_reject_rate_all_high_error_rejected(self):
        rows = self._make_rows([(0.2, 5.0), (0.3, 6.0)])  # all high error, all rejected
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5], error_threshold=3.0)[0]
        assert result["true_reject_rate"] == pytest.approx(1.0)

    def test_true_reject_rate_no_high_error_cases(self):
        rows = self._make_rows([(0.2, 1.0), (0.3, 2.0)])  # all low error
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5], error_threshold=3.0)[0]
        # n_high = 0 → true_reject_rate = 0
        assert result["true_reject_rate"] == pytest.approx(0.0)

    def test_true_reject_rate_partial(self):
        rows = self._make_rows([
            (0.2, 5.0),  # high error, rejected ✓
            (0.8, 5.0),  # high error, accepted ✗
            (0.2, 1.0),  # low error, rejected (irrelevant for TRR)
        ])
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5], error_threshold=3.0)[0]
        # n_high = 2, true_reject = 1 → TRR = 0.5
        assert result["true_reject_rate"] == pytest.approx(0.5)

    # ── false_pass_rate ───────────────────────────────────────────────────────

    def test_false_pass_rate_zero_when_no_high_error_accepted(self):
        rows = self._make_rows([(0.8, 1.0), (0.9, 2.0)])  # all accepted, all low error
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5], error_threshold=3.0)[0]
        assert result["false_pass_rate"] == pytest.approx(0.0)

    def test_false_pass_rate_one_when_all_accepted_have_high_error(self):
        rows = self._make_rows([(0.8, 5.0), (0.9, 6.0)])  # all accepted, all high error
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5], error_threshold=3.0)[0]
        assert result["false_pass_rate"] == pytest.approx(1.0)

    def test_false_pass_rate_zero_when_no_accepted(self):
        rows = self._make_rows([(0.2, 5.0), (0.3, 6.0)])  # all rejected
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5], error_threshold=3.0)[0]
        assert result["false_pass_rate"] == pytest.approx(0.0)

    def test_false_pass_rate_partial(self):
        rows = self._make_rows([
            (0.8, 5.0),  # accepted, high error (false pass)
            (0.9, 1.0),  # accepted, low error (ok)
            (0.8, 5.0),  # accepted, high error (false pass)
            (0.9, 1.0),  # accepted, low error (ok)
        ])
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5], error_threshold=3.0)[0]
        # 4 accepted, 2 with high error → FPR = 0.5
        assert result["false_pass_rate"] == pytest.approx(0.5)

    # ── mae_accepted ──────────────────────────────────────────────────────────

    def test_mae_accepted_correct_value(self):
        rows = self._make_rows([(0.8, 2.0), (0.9, 4.0)])  # both accepted
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5])[0]
        assert result["mae_accepted"] == pytest.approx(3.0)

    def test_mae_accepted_returns_999_when_no_accepted(self):
        rows = self._make_rows([(0.2, 1.0), (0.3, 2.0)])  # all rejected
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5])[0]
        assert result["mae_accepted"] == 999

    def test_mae_accepted_single_row(self):
        rows = self._make_rows([(0.8, 7.5)])
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5])[0]
        assert result["mae_accepted"] == pytest.approx(7.5)

    def test_mae_accepted_zero_error(self):
        rows = self._make_rows([(0.8, 0.0), (0.9, 0.0)])
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5])[0]
        assert result["mae_accepted"] == pytest.approx(0.0)

    # ── empty rows ────────────────────────────────────────────────────────────

    def test_empty_rows_returns_list_same_length_as_thresholds(self):
        result = compute_threshold_metrics([], "peak_ncc", [0.3, 0.5, 0.7])
        assert len(result) == 3

    def test_empty_rows_rejection_rate_zero(self):
        result = compute_threshold_metrics([], "peak_ncc", [0.5])[0]
        assert result["rejection_rate"] == 0

    def test_empty_rows_true_reject_rate_zero(self):
        result = compute_threshold_metrics([], "peak_ncc", [0.5])[0]
        assert result["true_reject_rate"] == 0

    def test_empty_rows_false_pass_rate_zero(self):
        result = compute_threshold_metrics([], "peak_ncc", [0.5])[0]
        assert result["false_pass_rate"] == 0

    def test_empty_rows_mae_accepted_999(self):
        result = compute_threshold_metrics([], "peak_ncc", [0.5])[0]
        assert result["mae_accepted"] == 999

    # ── sharpness score column ────────────────────────────────────────────────

    def test_sharpness_column_used_correctly(self):
        rows = [{"sharpness": 0.3, "error": 5.0}, {"sharpness": 2.0, "error": 1.0}]
        result = compute_threshold_metrics(rows, "sharpness", [1.0], error_threshold=3.0)[0]
        # score < 1.0 → rejected: sharpness=0.3 (high error)
        # score >= 1.0 → accepted: sharpness=2.0 (low error)
        assert result["n_rejected"] == 1
        assert result["n_accepted"] == 1
        assert result["false_pass_rate"] == pytest.approx(0.0)

    # ── multiple thresholds ordering ──────────────────────────────────────────

    def test_higher_threshold_higher_rejection_rate(self):
        rows = self._make_rows([(0.3, 1.0), (0.5, 2.0), (0.7, 3.0), (0.9, 0.5)])
        results = compute_threshold_metrics(rows, "peak_ncc", [0.2, 0.5, 0.8])
        rates = [r["rejection_rate"] for r in results]
        assert rates[0] <= rates[1] <= rates[2]

    def test_threshold_values_match_input_order(self):
        rows = self._make_rows([(0.5, 1.0)] * 10)
        thresholds = [0.1, 0.5, 0.9]
        result = compute_threshold_metrics(rows, "peak_ncc", thresholds)
        for r, thr in zip(result, thresholds):
            assert r["threshold"] == pytest.approx(thr, abs=1e-3)

    # ── custom error_threshold ────────────────────────────────────────────────

    def test_custom_error_threshold_5(self):
        rows = self._make_rows([(0.8, 4.9), (0.8, 5.1)])  # both accepted
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5], error_threshold=5.0)[0]
        # error=4.9 < 5.0 → low; error=5.1 > 5.0 → high
        assert result["false_pass_rate"] == pytest.approx(0.5)

    def test_error_exactly_at_threshold_not_high_error(self):
        """error > threshold, not >=, so exactly equal is NOT high error."""
        rows = self._make_rows([(0.8, 3.0)])  # error == 3.0, threshold == 3.0
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5], error_threshold=3.0)[0]
        # error is NOT > 3.0, so n_high = 0 → FPR = 0
        assert result["false_pass_rate"] == pytest.approx(0.0)

    def test_error_just_above_threshold_is_high_error(self):
        rows = self._make_rows([(0.8, 3.001)])  # accepted, high error
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5], error_threshold=3.0)[0]
        assert result["false_pass_rate"] == pytest.approx(1.0)

    # ── counts ────────────────────────────────────────────────────────────────

    def test_n_rejected_plus_n_accepted_equals_total(self):
        rows = self._make_rows([(0.3, 1.0), (0.6, 2.0), (0.8, 3.5)])
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5])[0]
        assert result["n_rejected"] + result["n_accepted"] == len(rows)

    def test_all_rows_counted_at_zero_threshold(self):
        rows = self._make_rows([(0.0, 1.0), (0.5, 2.0), (1.0, 3.0)])
        result = compute_threshold_metrics(rows, "peak_ncc", [0.0])[0]
        # score >= 0.0 → all accepted
        assert result["n_accepted"] == 3
        assert result["n_rejected"] == 0

    # ── rounding ─────────────────────────────────────────────────────────────

    def test_rejection_rate_is_rounded(self):
        """Values should be rounded to 3 decimal places."""
        rows = self._make_rows([(0.2, 5.0)] * 1 + [(0.8, 1.0)] * 2)
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5])[0]
        # 1/3 ≈ 0.333
        assert result["rejection_rate"] == pytest.approx(0.333, abs=1e-3)

    def test_mae_accepted_is_rounded(self):
        rows = self._make_rows([(0.8, 1.0), (0.9, 2.0), (0.7, 3.0)])
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5])[0]
        # MAE = (1+2+3)/3 = 2.0 exactly
        assert result["mae_accepted"] == pytest.approx(2.0, abs=1e-3)

    # ── large dataset ─────────────────────────────────────────────────────────

    def test_large_dataset_consistent(self):
        rng = np.random.default_rng(42)
        scores = rng.uniform(0.0, 1.0, 500).tolist()
        errors = rng.uniform(0.0, 10.0, 500).tolist()
        rows = [{"peak_ncc": s, "error": e} for s, e in zip(scores, errors)]
        result = compute_threshold_metrics(rows, "peak_ncc", [0.5], error_threshold=3.0)[0]
        # sanity checks
        assert 0.0 <= result["rejection_rate"] <= 1.0
        assert 0.0 <= result["false_pass_rate"] <= 1.0
        assert 0.0 <= result["true_reject_rate"] <= 1.0
        assert result["n_rejected"] + result["n_accepted"] == 500
