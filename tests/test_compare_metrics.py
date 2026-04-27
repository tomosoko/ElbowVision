"""
Unit tests for scripts/compare_metrics.py

Tests cover:
  - ALL_METRICS constant: contents and membership
  - evaluate_xray_all_metrics: return-key structure, invariants, edge cases
    * err_X == abs(bias_X) for every metric X
    * bias_X == round(pred_X - gt_angle, 2) for every metric X
    * elapsed_s >= 0, peak_ncc in [-1, 1], sharpness is finite float
    * restricted `metrics` parameter
    * multiple gt_angle values
    * coarse_step / fine_range variations
    * single-angle and small-library corner cases
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from scripts.compare_metrics import ALL_METRICS, evaluate_xray_all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_xray(seed: int = 0, size: int = 256) -> np.ndarray:
    """
    合成X線画像: 大部分が明るい（>15）のでcrop_to_boneは素通り(70%ルール)。
    CLAHEとncc計算に耐えるだけの輝度変化を持たせる。
    size=256 は preprocess_image のデフォルト出力サイズに合わせる。
    """
    rng = np.random.default_rng(seed)
    base = np.full((size, size), 80, dtype=np.uint8)
    # ランダムな明暗パターン (幅広い階調)
    noise = rng.integers(0, 120, (size, size), dtype=np.int32)
    img = np.clip(base.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    return img


def _make_library(
    angles: list[float],
    seed: int = 42,
    size: int = 256,
) -> dict[float, np.ndarray]:
    """各角度に独立した合成DRR(uint8)を生成する。"""
    rng = np.random.default_rng(seed)
    return {
        float(a): rng.integers(50, 220, (size, size), dtype=np.uint8)
        for a in angles
    }


def _run(
    xray: np.ndarray | None = None,
    angles: list[float] | None = None,
    gt_angle: float = 90.0,
    angle_min: float | None = None,
    angle_max: float | None = None,
    coarse_step: float = 10.0,
    fine_range: float = 5.0,
    metrics: list[str] | None = None,
) -> dict:
    """共通ラッパー: デフォルト設定でevaluate_xray_all_metricsを呼ぶ。"""
    if xray is None:
        xray = _make_xray()
    if angles is None:
        angles = [60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0]
    library = _make_library(angles)
    a_min = angle_min if angle_min is not None else min(angles)
    a_max = angle_max if angle_max is not None else max(angles)
    return evaluate_xray_all_metrics(
        xray,
        library,
        a_min,
        a_max,
        gt_angle,
        coarse_step=coarse_step,
        fine_range=fine_range,
        metrics=metrics,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ALL_METRICS constant
# ─────────────────────────────────────────────────────────────────────────────

class TestAllMetrics:
    def test_is_list(self):
        assert isinstance(ALL_METRICS, list)

    def test_length(self):
        assert len(ALL_METRICS) == 5

    def test_contains_ncc(self):
        assert "ncc" in ALL_METRICS

    def test_contains_edge_ncc(self):
        assert "edge_ncc" in ALL_METRICS

    def test_contains_combined(self):
        assert "combined" in ALL_METRICS

    def test_contains_nmi(self):
        assert "nmi" in ALL_METRICS

    def test_contains_combined_nmi(self):
        assert "combined_nmi" in ALL_METRICS

    def test_no_duplicates(self):
        assert len(ALL_METRICS) == len(set(ALL_METRICS))

    def test_all_are_strings(self):
        assert all(isinstance(m, str) for m in ALL_METRICS)


# ─────────────────────────────────────────────────────────────────────────────
# Return structure: required keys always present
# ─────────────────────────────────────────────────────────────────────────────

class TestReturnStructure:
    @pytest.fixture(scope="class")
    def result(self):
        return _run()

    def test_returns_dict(self, result):
        assert isinstance(result, dict)

    def test_has_elapsed_s(self, result):
        assert "elapsed_s" in result

    def test_has_peak_ncc(self, result):
        assert "peak_ncc" in result

    def test_has_sharpness(self, result):
        assert "sharpness" in result

    @pytest.mark.parametrize("m", ALL_METRICS)
    def test_has_pred_key(self, result, m):
        assert f"pred_{m}" in result

    @pytest.mark.parametrize("m", ALL_METRICS)
    def test_has_err_key(self, result, m):
        assert f"err_{m}" in result

    @pytest.mark.parametrize("m", ALL_METRICS)
    def test_has_bias_key(self, result, m):
        assert f"bias_{m}" in result


# ─────────────────────────────────────────────────────────────────────────────
# Invariants: err = abs(bias), bias = pred - gt
# ─────────────────────────────────────────────────────────────────────────────

class TestInvariants:
    @pytest.fixture(scope="class")
    def result(self):
        return _run(gt_angle=90.0)

    @pytest.mark.parametrize("m", ALL_METRICS)
    def test_err_equals_abs_bias(self, result, m):
        err  = result[f"err_{m}"]
        bias = result[f"bias_{m}"]
        assert abs(err - abs(bias)) < 0.02, f"{m}: err={err}, |bias|={abs(bias)}"

    @pytest.mark.parametrize("m", ALL_METRICS)
    def test_bias_equals_pred_minus_gt(self, result, m):
        pred = result[f"pred_{m}"]
        bias = result[f"bias_{m}"]
        expected = round(pred - 90.0, 2)
        assert abs(bias - expected) < 0.02, f"{m}: bias={bias}, pred-gt={expected}"

    @pytest.mark.parametrize("m", ALL_METRICS)
    def test_err_nonnegative(self, result, m):
        assert result[f"err_{m}"] >= 0.0

    @pytest.mark.parametrize("m", ALL_METRICS)
    def test_pred_is_float(self, result, m):
        assert isinstance(result[f"pred_{m}"], float)


# ─────────────────────────────────────────────────────────────────────────────
# elapsed_s, peak_ncc, sharpness sanity
# ─────────────────────────────────────────────────────────────────────────────

class TestSanityValues:
    @pytest.fixture(scope="class")
    def result(self):
        return _run()

    def test_elapsed_s_nonnegative(self, result):
        assert result["elapsed_s"] >= 0.0

    def test_elapsed_s_is_float(self, result):
        assert isinstance(result["elapsed_s"], float)

    def test_peak_ncc_in_valid_range(self, result):
        # NCC は -1 〜 +1
        assert -1.0 <= result["peak_ncc"] <= 1.0

    def test_peak_ncc_is_float(self, result):
        assert isinstance(result["peak_ncc"], float)

    def test_sharpness_is_finite(self, result):
        import math
        assert math.isfinite(result["sharpness"])

    def test_sharpness_is_float(self, result):
        assert isinstance(result["sharpness"], float)


# ─────────────────────────────────────────────────────────────────────────────
# Restricted metrics parameter
# ─────────────────────────────────────────────────────────────────────────────

class TestRestrictedMetrics:
    def test_single_metric_ncc_has_ncc_keys(self):
        result = _run(metrics=["ncc"])
        assert "pred_ncc" in result
        assert "err_ncc" in result
        assert "bias_ncc" in result

    def test_single_metric_ncc_no_edge_ncc_keys(self):
        result = _run(metrics=["ncc"])
        assert "pred_edge_ncc" not in result
        assert "err_edge_ncc" not in result

    def test_single_metric_ncc_no_combined_keys(self):
        result = _run(metrics=["ncc"])
        assert "pred_combined" not in result

    def test_single_metric_ncc_no_nmi_keys(self):
        result = _run(metrics=["ncc"])
        assert "pred_nmi" not in result
        assert "pred_combined_nmi" not in result

    def test_two_metrics_ncc_nmi_have_both_keys(self):
        result = _run(metrics=["ncc", "nmi"])
        assert "pred_ncc" in result
        assert "pred_nmi" in result

    def test_two_metrics_ncc_nmi_no_other_keys(self):
        result = _run(metrics=["ncc", "nmi"])
        assert "pred_edge_ncc" not in result
        assert "pred_combined" not in result
        assert "pred_combined_nmi" not in result

    def test_edge_ncc_alone_returns_correct_keys(self):
        result = _run(metrics=["edge_ncc"])
        assert "pred_edge_ncc" in result
        assert "err_edge_ncc" in result
        assert "bias_edge_ncc" in result

    def test_combined_metrics_only(self):
        result = _run(metrics=["combined", "combined_nmi"])
        assert "pred_combined" in result
        assert "pred_combined_nmi" in result
        assert "pred_ncc" not in result
        assert "pred_nmi" not in result

    def test_none_metrics_uses_all(self):
        """metrics=None はデフォルトの ALL_METRICS と同一。"""
        result = _run(metrics=None)
        for m in ALL_METRICS:
            assert f"pred_{m}" in result


# ─────────────────────────────────────────────────────────────────────────────
# Bias direction: sign follows pred - gt
# ─────────────────────────────────────────────────────────────────────────────

class TestBiasDirection:
    @pytest.mark.parametrize("gt_angle", [70.0, 90.0, 110.0])
    def test_bias_sign_consistent_with_pred(self, gt_angle):
        """bias = pred - gt のため、bias>0 ⟺ pred>gt かつ bias<0 ⟺ pred<gt."""
        result = _run(gt_angle=gt_angle, metrics=["ncc"])
        pred = result["pred_ncc"]
        bias = result["bias_ncc"]
        if pred > gt_angle:
            assert bias > 0 or abs(bias) < 0.05  # ほぼ一致なら許容
        elif pred < gt_angle:
            assert bias < 0 or abs(bias) < 0.05
        else:
            assert abs(bias) < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Determinism: 同一入力で同一出力
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_identical_inputs_give_identical_output(self):
        xray = _make_xray(seed=7)
        library = _make_library([60.0, 90.0, 120.0], seed=7)
        kwargs = dict(
            angle_to_drr=library, angle_min=60.0, angle_max=120.0,
            gt_angle=90.0, coarse_step=10.0, fine_range=5.0, metrics=["ncc"]
        )
        r1 = evaluate_xray_all_metrics(xray, **kwargs)
        r2 = evaluate_xray_all_metrics(xray, **kwargs)
        assert r1["pred_ncc"] == r2["pred_ncc"]
        assert r1["err_ncc"]  == r2["err_ncc"]

    def test_different_gt_gives_different_bias(self):
        """gt_angle が違えば bias は違う（pred は同じでも bias は変わる）。"""
        xray = _make_xray(seed=3)
        library = _make_library([60.0, 90.0, 120.0], seed=3)
        base_kwargs = dict(
            angle_to_drr=library, angle_min=60.0, angle_max=120.0,
            coarse_step=10.0, fine_range=5.0, metrics=["ncc"]
        )
        r1 = evaluate_xray_all_metrics(xray, gt_angle=80.0, **base_kwargs)
        r2 = evaluate_xray_all_metrics(xray, gt_angle=100.0, **base_kwargs)
        # 同じ pred でも gt が違えば bias は20°ずれる
        assert abs(r1["bias_ncc"] - r2["bias_ncc"]) > 1.0


# ─────────────────────────────────────────────────────────────────────────────
# coarse_step / fine_range variations
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchParams:
    def test_coarse_step_5_completes(self):
        result = _run(coarse_step=5.0, fine_range=5.0, metrics=["ncc"])
        assert "pred_ncc" in result

    def test_coarse_step_20_completes(self):
        result = _run(coarse_step=20.0, fine_range=3.0, metrics=["ncc"])
        assert "pred_ncc" in result

    def test_fine_range_0_completes(self):
        """fine_range=0 は粗探索のみ（fine_angles が空になる）。"""
        result = _run(coarse_step=10.0, fine_range=0.0, metrics=["ncc"])
        assert "pred_ncc" in result

    def test_fine_range_large_completes(self):
        result = _run(
            angles=[60.0, 90.0, 120.0],
            coarse_step=30.0, fine_range=20.0, metrics=["ncc"]
        )
        assert "pred_ncc" in result

    def test_invariants_hold_for_small_coarse_step(self):
        result = _run(coarse_step=5.0, fine_range=5.0, metrics=["ncc"])
        err  = result["err_ncc"]
        bias = result["bias_ncc"]
        assert abs(err - abs(bias)) < 0.02


# ─────────────────────────────────────────────────────────────────────────────
# Small library corner cases
# ─────────────────────────────────────────────────────────────────────────────

class TestSmallLibrary:
    def test_three_angle_library_completes(self):
        result = _run(
            angles=[80.0, 90.0, 100.0],
            angle_min=80.0, angle_max=100.0,
            gt_angle=90.0,
            coarse_step=10.0, fine_range=5.0,
            metrics=["ncc"],
        )
        assert "pred_ncc" in result

    def test_two_angle_library_completes(self):
        """2角度ライブラリでもクラッシュしない。"""
        result = _run(
            angles=[85.0, 95.0],
            angle_min=85.0, angle_max=95.0,
            gt_angle=90.0,
            coarse_step=5.0, fine_range=3.0,
            metrics=["ncc"],
        )
        assert "pred_ncc" in result

    def test_invariants_hold_for_two_angle_library(self):
        result = _run(
            angles=[85.0, 95.0],
            angle_min=85.0, angle_max=95.0,
            gt_angle=90.0,
            coarse_step=5.0, fine_range=3.0,
            metrics=["ncc"],
        )
        err  = result["err_ncc"]
        bias = result["bias_ncc"]
        assert abs(err - abs(bias)) < 0.02

    def test_wide_angle_range_completes(self):
        angles = [float(a) for a in range(60, 181, 10)]
        result = _run(
            angles=angles,
            angle_min=60.0, angle_max=180.0,
            gt_angle=120.0,
            coarse_step=10.0, fine_range=10.0,
            metrics=["ncc", "combined"],
        )
        assert "pred_ncc" in result
        assert "pred_combined" in result


# ─────────────────────────────────────────────────────────────────────────────
# Pred angle is within plausible bounds
# ─────────────────────────────────────────────────────────────────────────────

class TestPredBounds:
    @pytest.mark.parametrize("m", ALL_METRICS)
    def test_pred_within_extended_range(self, m):
        """予測角は angle_min - fine_range 〜 angle_max + fine_range に収まる。"""
        angles = [60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0]
        result = _run(
            angles=angles,
            angle_min=60.0, angle_max=120.0,
            gt_angle=90.0,
            coarse_step=10.0, fine_range=5.0,
            metrics=[m],
        )
        pred = result[f"pred_{m}"]
        # 抛物線補間がわずかにはみ出す可能性を考慮し +/-15 の余裕
        assert 45.0 <= pred <= 135.0, f"{m}: pred={pred}"


# ─────────────────────────────────────────────────────────────────────────────
# combined / combined_nmi derivation smoke tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCombinedMetrics:
    @pytest.fixture(scope="class")
    def result(self):
        return _run(metrics=["ncc", "edge_ncc", "nmi", "combined", "combined_nmi"])

    def test_combined_pred_is_float(self, result):
        assert isinstance(result["pred_combined"], float)

    def test_combined_nmi_pred_is_float(self, result):
        assert isinstance(result["pred_combined_nmi"], float)

    def test_combined_invariant(self, result):
        """combined も err = abs(bias) を満たす。"""
        err  = result["err_combined"]
        bias = result["bias_combined"]
        assert abs(err - abs(bias)) < 0.02

    def test_combined_nmi_invariant(self, result):
        err  = result["err_combined_nmi"]
        bias = result["bias_combined_nmi"]
        assert abs(err - abs(bias)) < 0.02
