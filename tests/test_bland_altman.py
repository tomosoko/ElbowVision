"""
Bland-Altman 検証スクリプトのユニットテスト

ダミーデータで全関数の動作を確認する。
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# テスト対象モジュールをインポート
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from bland_altman import (
    BlandAltmanResult,
    compute_bland_altman,
    compute_icc,
    format_summary,
    load_csv,
    plot_bland_altman,
    run_analysis,
)


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_data():
    """完全一致に近いダミーデータ（ノイズ小）"""
    rng = np.random.RandomState(42)
    gt = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 12.0, 18.0, 22.0, 28.0, 35.0])
    noise = rng.normal(0, 0.5, size=len(gt))
    pred = gt + noise
    return gt, pred


@pytest.fixture
def dummy_csv(tmp_path):
    """ダミーCSVファイルを作成して返す。"""
    rng = np.random.RandomState(42)
    n = 20
    gt_carrying = rng.uniform(5, 20, n)
    gt_flexion = rng.uniform(80, 170, n)
    pred_carrying = gt_carrying + rng.normal(0, 1.0, n)
    pred_flexion = gt_flexion + rng.normal(0, 2.0, n)

    df = pd.DataFrame({
        "filename": [f"img_{i:03d}.png" for i in range(n)],
        "gt_carrying_angle": gt_carrying,
        "pred_carrying_angle": pred_carrying,
        "gt_flexion_deg": gt_flexion,
        "pred_flexion_deg": pred_flexion,
    })
    csv_path = str(tmp_path / "test_data.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# compute_bland_altman のテスト
# ---------------------------------------------------------------------------

class TestComputeBlandAltman:
    def test_basic_stats(self, dummy_data):
        gt, pred = dummy_data
        result = compute_bland_altman(gt, pred)

        assert isinstance(result, BlandAltmanResult)
        assert result.n == len(gt)

        # バイアスはノイズが小さいのでゼロ付近
        assert abs(result.mean_diff) < 1.0

        # LoA が mean_diff +/- 1.96*SD
        expected_upper = result.mean_diff + 1.96 * result.std_diff
        expected_lower = result.mean_diff - 1.96 * result.std_diff
        assert abs(result.loa_upper - expected_upper) < 1e-10
        assert abs(result.loa_lower - expected_lower) < 1e-10

    def test_mae_positive(self, dummy_data):
        gt, pred = dummy_data
        result = compute_bland_altman(gt, pred)
        assert result.mae >= 0

    def test_rmse_ge_mae(self, dummy_data):
        """RMSE >= MAE は常に成り立つ。"""
        gt, pred = dummy_data
        result = compute_bland_altman(gt, pred)
        assert result.rmse >= result.mae - 1e-10

    def test_perfect_agreement(self):
        """GT と Pred が完全一致の場合。"""
        gt = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        pred = gt.copy()
        result = compute_bland_altman(gt, pred)

        assert result.mean_diff == 0.0
        assert result.mae == 0.0
        assert result.rmse == 0.0
        assert result.r_squared == pytest.approx(1.0, abs=1e-10)
        assert result.icc == pytest.approx(1.0, abs=1e-10)

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="長さが一致しません"):
            compute_bland_altman(np.array([1, 2]), np.array([1, 2, 3]))

    def test_too_few_points(self):
        with pytest.raises(ValueError, match="2点未満"):
            compute_bland_altman(np.array([1.0]), np.array([1.0]))

    def test_r_squared_range(self, dummy_data):
        gt, pred = dummy_data
        result = compute_bland_altman(gt, pred)
        # r^2 は一般に [-inf, 1] だが、近似が良ければ 0-1 近辺
        assert result.r_squared <= 1.0 + 1e-10

    def test_pearson_r(self, dummy_data):
        gt, pred = dummy_data
        result = compute_bland_altman(gt, pred)
        # ノイズが小さいので相関が高い
        assert result.pearson_r > 0.9


# ---------------------------------------------------------------------------
# compute_icc のテスト
# ---------------------------------------------------------------------------

class TestComputeICC:
    def test_perfect_agreement(self):
        gt = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        assert compute_icc(gt, gt.copy()) == pytest.approx(1.0, abs=1e-10)

    def test_high_agreement(self, dummy_data):
        gt, pred = dummy_data
        icc = compute_icc(gt, pred)
        assert icc > 0.9

    def test_icc_range(self):
        """ICCは-1から1の範囲にクリップされる。"""
        gt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        icc = compute_icc(gt, pred)
        assert -1.0 <= icc <= 1.0

    def test_single_point(self):
        assert compute_icc(np.array([1.0]), np.array([2.0])) == 0.0


# ---------------------------------------------------------------------------
# plot_bland_altman のテスト
# ---------------------------------------------------------------------------

class TestPlotBlandAltman:
    def test_creates_png(self, dummy_data, tmp_path):
        gt, pred = dummy_data
        result = compute_bland_altman(gt, pred)
        out_path = str(tmp_path / "test_plot.png")

        plot_bland_altman(gt, pred, result, "test_angle", out_path)
        assert os.path.isfile(out_path)
        assert os.path.getsize(out_path) > 0


# ---------------------------------------------------------------------------
# load_csv のテスト
# ---------------------------------------------------------------------------

class TestLoadCSV:
    def test_loads_and_strips_columns(self, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        with open(csv_path, "w") as f:
            f.write(" filename , gt_carrying_angle , pred_carrying_angle\n")
            f.write("img_001.png, 10.0, 10.5\n")

        df = load_csv(csv_path)
        assert "filename" in df.columns
        assert "gt_carrying_angle" in df.columns
        assert "pred_carrying_angle" in df.columns


# ---------------------------------------------------------------------------
# format_summary のテスト
# ---------------------------------------------------------------------------

class TestFormatSummary:
    def test_contains_all_metrics(self, dummy_data):
        gt, pred = dummy_data
        result = compute_bland_altman(gt, pred)
        summary = format_summary({"carrying_angle": result})

        assert "Mean Bias" in summary
        assert "MAE" in summary
        assert "RMSE" in summary
        assert "ICC" in summary
        assert "r^2" in summary
        assert "95% LoA" in summary
        assert "carrying_angle" in summary

    def test_clinical_threshold_judgment(self, dummy_data):
        gt, pred = dummy_data
        result = compute_bland_altman(gt, pred)
        summary = format_summary({"carrying_angle": result})

        # 臨床許容範囲の判定が含まれる
        assert "Clinical Threshold" in summary
        assert "PASS" in summary or "FAIL" in summary


# ---------------------------------------------------------------------------
# run_analysis (統合テスト)
# ---------------------------------------------------------------------------

class TestRunAnalysis:
    def test_full_pipeline(self, dummy_csv, tmp_path):
        out_dir = str(tmp_path / "output")
        results = run_analysis(dummy_csv, out_dir)

        # 2角度分の結果が返る
        assert "carrying_angle" in results
        assert "flexion_deg" in results

        # PNGが生成されている
        assert os.path.isfile(os.path.join(out_dir, "bland_altman_carrying.png"))
        assert os.path.isfile(os.path.join(out_dir, "bland_altman_flexion.png"))

        # サマリーテキストが生成されている
        assert os.path.isfile(os.path.join(out_dir, "summary.txt"))

    def test_missing_columns(self, tmp_path):
        """列が不足しているCSVでもスキップしてエラーにならない。"""
        csv_path = str(tmp_path / "partial.csv")
        pd.DataFrame({
            "filename": ["a.png", "b.png", "c.png"],
            "gt_carrying_angle": [10.0, 15.0, 20.0],
            "pred_carrying_angle": [10.5, 14.8, 20.3],
        }).to_csv(csv_path, index=False)

        out_dir = str(tmp_path / "output2")
        results = run_analysis(csv_path, out_dir)

        assert "carrying_angle" in results
        assert "flexion_deg" not in results
