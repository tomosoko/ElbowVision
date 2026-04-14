"""
eval_realxray_batch.py — bland_altman_analysis() のユニットテスト

Phase 2 論文に直接使われる ICC・MAE・Bias・LoA の計算が正しいことを保証する。
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from eval_realxray_batch import bland_altman_analysis


# ── ヘルパー ──────────────────────────────────────────────────────────────────

def _write_pred_csv(tmp_path: Path, gt_angles, pred_angles, *, note: str = "") -> str:
    """予測CSVをtmp_pathに書き込み、パスを返す"""
    csv_path = str(tmp_path / "predictions.csv")
    fieldnames = [
        "patient_id", "xray_path", "gt_flexion_deg", "pred_flexion_deg",
        "error_deg", "peak_ncc", "sharpness", "elapsed_s", "note",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, (gt, pred) in enumerate(zip(gt_angles, pred_angles)):
            w.writerow({
                "patient_id": f"P{i:03d}",
                "xray_path": f"img_{i:03d}.png",
                "gt_flexion_deg": gt,
                "pred_flexion_deg": pred,
                "error_deg": abs(pred - gt),
                "peak_ncc": 0.85,
                "sharpness": 1.0,
                "elapsed_s": 0.1,
                "note": note,
            })
    return csv_path


def _read_summary(out_dir: Path) -> str:
    return (out_dir / "summary.txt").read_text(encoding="utf-8")


# ── 出力ファイル生成テスト ───────────────────────────────────────────────────

class TestOutputFiles:
    def test_generates_summary_txt(self, tmp_path):
        gt = list(range(90, 181, 10))
        pred = [g + 1.0 for g in gt]
        csv_path = _write_pred_csv(tmp_path, gt, pred)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        bland_altman_analysis(csv_path, str(out_dir))
        assert (out_dir / "summary.txt").exists()

    def test_generates_ba_plot_png(self, tmp_path):
        gt = list(range(90, 181, 10))
        pred = [g + 1.0 for g in gt]
        csv_path = _write_pred_csv(tmp_path, gt, pred)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        bland_altman_analysis(csv_path, str(out_dir))
        assert (out_dir / "bland_altman_realxray.png").exists()
        assert (out_dir / "bland_altman_realxray.png").stat().st_size > 0


# ── 統計値テスト ─────────────────────────────────────────────────────────────

class TestStatistics:
    def test_perfect_agreement_bias_zero(self, tmp_path):
        """GT=Pred のとき Bias=0.000"""
        gt = [90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
        pred = gt.copy()
        csv_path = _write_pred_csv(tmp_path, gt, pred)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        bland_altman_analysis(csv_path, str(out_dir))
        summary = _read_summary(out_dir)
        # Bias 行を取り出して値を確認
        bias_line = next(l for l in summary.splitlines() if "Mean Bias" in l)
        bias_val = float(bias_line.split("=")[1].strip().rstrip("°"))
        assert abs(bias_val) < 1e-10

    def test_perfect_agreement_mae_zero(self, tmp_path):
        gt = list(range(90, 181, 10))
        pred = gt.copy()
        csv_path = _write_pred_csv(tmp_path, gt, pred)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        bland_altman_analysis(csv_path, str(out_dir))
        summary = _read_summary(out_dir)
        mae_line = next(l for l in summary.splitlines() if l.strip().startswith("MAE"))
        mae_val = float(mae_line.split("=")[1].strip().rstrip("°"))
        assert mae_val < 1e-10

    def test_known_bias(self, tmp_path):
        """一定オフセット +3° のデータは Bias ≈ +3.000"""
        gt = list(range(90, 181, 10))
        pred = [g + 3.0 for g in gt]
        csv_path = _write_pred_csv(tmp_path, gt, pred)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        bland_altman_analysis(csv_path, str(out_dir))
        summary = _read_summary(out_dir)
        bias_line = next(l for l in summary.splitlines() if "Mean Bias" in l)
        bias_val = float(bias_line.split("=")[1].strip().rstrip("°"))
        assert abs(bias_val - 3.0) < 1e-6

    def test_known_mae(self, tmp_path):
        """各誤差が一定 (2.0°) のとき MAE = 2.000"""
        gt = [90.0, 100.0, 110.0, 120.0, 130.0]
        pred = [g + 2.0 for g in gt]
        csv_path = _write_pred_csv(tmp_path, gt, pred)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        bland_altman_analysis(csv_path, str(out_dir))
        summary = _read_summary(out_dir)
        mae_line = next(l for l in summary.splitlines() if l.strip().startswith("MAE"))
        mae_val = float(mae_line.split("=")[1].strip().rstrip("°"))
        assert abs(mae_val - 2.0) < 1e-6

    def test_sample_count_in_summary(self, tmp_path):
        """summary.txt にサンプル数が正しく記録される"""
        n = 15
        gt = list(np.linspace(90, 180, n))
        pred = list(np.array(gt) + 1.0)
        csv_path = _write_pred_csv(tmp_path, gt, pred)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        bland_altman_analysis(csv_path, str(out_dir))
        summary = _read_summary(out_dir)
        n_line = next(l for l in summary.splitlines() if l.strip().startswith("n "))
        n_val = int(n_line.split("=")[1].strip())
        assert n_val == n


# ── ICC テスト ────────────────────────────────────────────────────────────────

class TestICC:
    def test_icc_near_one_for_perfect_data(self, tmp_path):
        """完全一致データで ICC ≈ 1.0"""
        gt = list(np.linspace(90, 180, 12))
        pred = gt.copy()
        csv_path = _write_pred_csv(tmp_path, gt, pred)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        bland_altman_analysis(csv_path, str(out_dir))
        summary = _read_summary(out_dir)
        icc_line = next(l for l in summary.splitlines() if "ICC" in l and "=" in l)
        # "ICC(3,1)       = 1.0000" 形式
        icc_str = icc_line.split("=")[1].strip().split()[0]
        icc_val = float(icc_str)
        assert icc_val > 0.999

    def test_icc_low_for_random_data(self, tmp_path):
        """GTとPredが無相関のデータは ICC が低い"""
        rng = np.random.RandomState(42)
        gt = list(np.linspace(90, 180, 20))
        pred = list(rng.uniform(90, 180, 20))  # GT と無関係なランダム値
        csv_path = _write_pred_csv(tmp_path, gt, pred)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        bland_altman_analysis(csv_path, str(out_dir))
        summary = _read_summary(out_dir)
        icc_line = next(l for l in summary.splitlines() if "ICC" in l and "=" in l)
        icc_str = icc_line.split("=")[1].strip().split()[0]
        icc_val = float(icc_str)
        assert icc_val < 0.90

    def test_icc_high_for_small_noise(self, tmp_path):
        """小ノイズ (±0.5°) のデータは ICC ≥ 0.90"""
        rng = np.random.RandomState(7)
        gt = np.linspace(90, 180, 20)
        pred = gt + rng.normal(0, 0.5, 20)
        csv_path = _write_pred_csv(tmp_path, gt.tolist(), pred.tolist())
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        bland_altman_analysis(csv_path, str(out_dir))
        summary = _read_summary(out_dir)
        icc_line = next(l for l in summary.splitlines() if "ICC" in l and "=" in l)
        icc_str = icc_line.split("=")[1].strip().split()[0]
        icc_val = float(icc_str)
        assert icc_val >= 0.90


# ── PASS/FAIL 判定テスト ──────────────────────────────────────────────────────

class TestPassFail:
    def test_all_pass_for_excellent_data(self, tmp_path):
        """優秀なデータ (小ノイズ・バイアスなし) はすべて PASS"""
        rng = np.random.RandomState(0)
        gt = np.linspace(90, 180, 20)
        pred = gt + rng.normal(0, 0.3, 20)
        csv_path = _write_pred_csv(tmp_path, gt.tolist(), pred.tolist())
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        bland_altman_analysis(csv_path, str(out_dir))
        summary = _read_summary(out_dir)
        assert summary.count("PASS") == 3
        assert "FAIL" not in summary

    def test_bias_fail_for_large_offset(self, tmp_path):
        """系統バイアス +10° は Bias ≤±3° を FAIL"""
        gt = list(range(90, 181, 10))
        pred = [g + 10.0 for g in gt]
        csv_path = _write_pred_csv(tmp_path, gt, pred)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        bland_altman_analysis(csv_path, str(out_dir))
        summary = _read_summary(out_dir)
        assert "FAIL" in summary

    def test_loa_fail_for_large_variance(self, tmp_path):
        """大きなばらつき (±15°) は LoA ≤±8° を FAIL"""
        rng = np.random.RandomState(1)
        gt = np.linspace(90, 180, 30)
        pred = gt + rng.normal(0, 8.0, 30)   # SD≈8° → LoA≈±16°
        csv_path = _write_pred_csv(tmp_path, gt.tolist(), pred.tolist())
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        bland_altman_analysis(csv_path, str(out_dir))
        summary = _read_summary(out_dir)
        assert "FAIL" in summary


# ── エッジケース ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_insufficient_data_no_crash(self, tmp_path):
        """n=1 でも例外なくスキップされる"""
        csv_path = str(tmp_path / "predictions.csv")
        fieldnames = [
            "patient_id", "xray_path", "gt_flexion_deg", "pred_flexion_deg",
            "error_deg", "peak_ncc", "sharpness", "elapsed_s", "note",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerow({
                "patient_id": "P001", "xray_path": "img.png",
                "gt_flexion_deg": 90.0, "pred_flexion_deg": 91.0,
                "error_deg": 1.0, "peak_ncc": 0.9, "sharpness": 1.0,
                "elapsed_s": 0.1, "note": "",
            })
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        # 例外が出ないことを確認
        bland_altman_analysis(csv_path, str(out_dir))

    def test_missing_gt_rows_excluded(self, tmp_path):
        """gt_flexion_deg が空の行は集計から除外される"""
        csv_path = str(tmp_path / "predictions.csv")
        fieldnames = [
            "patient_id", "xray_path", "gt_flexion_deg", "pred_flexion_deg",
            "error_deg", "peak_ncc", "sharpness", "elapsed_s", "note",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            # 正常 n=6 + GT空 n=2
            for i, angle in enumerate([90, 100, 110, 120, 130, 140]):
                w.writerow({
                    "patient_id": f"P{i}", "xray_path": f"img_{i}.png",
                    "gt_flexion_deg": angle, "pred_flexion_deg": angle + 1,
                    "error_deg": 1.0, "peak_ncc": 0.9, "sharpness": 1.0,
                    "elapsed_s": 0.1, "note": "",
                })
            for j in range(2):
                w.writerow({
                    "patient_id": f"ERR{j}", "xray_path": "err.png",
                    "gt_flexion_deg": "", "pred_flexion_deg": 90.0,
                    "error_deg": "", "peak_ncc": 0.3, "sharpness": 0.5,
                    "elapsed_s": 0.1, "note": "error",
                })
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        bland_altman_analysis(csv_path, str(out_dir))
        summary = _read_summary(out_dir)
        n_line = next(l for l in summary.splitlines() if l.strip().startswith("n "))
        n_val = int(n_line.split("=")[1].strip())
        assert n_val == 6

    def test_zero_gt_angle_included_in_ba(self, tmp_path):
        """gt_flexion_deg = 0.0 の行は集計に含まれなければならない
        （旧コードの falsy チェックで 0.0 が除外されるバグのリグレッションテスト）"""
        # 有効行: GT=0.0, GT=10.0, ... GT=50.0 (計6行)
        gt = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        pred = [g + 1.0 for g in gt]
        csv_path = _write_pred_csv(tmp_path, gt, pred)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        bland_altman_analysis(csv_path, str(out_dir))
        summary = _read_summary(out_dir)
        n_line = next(l for l in summary.splitlines() if l.strip().startswith("n "))
        n_val = int(n_line.split("=")[1].strip())
        # 0.0 の行が除外されたら n=5 になる（バグ再現）→ 正しくは n=6
        assert n_val == 6, f"GT=0.0 が除外されている (n={n_val}, expected 6)"

    def test_summary_contains_header(self, tmp_path):
        """summary.txt に Bland-Altman のヘッダーが含まれる"""
        gt = list(range(90, 181, 10))
        pred = [g + 0.5 for g in gt]
        csv_path = _write_pred_csv(tmp_path, gt, pred)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        bland_altman_analysis(csv_path, str(out_dir))
        summary = _read_summary(out_dir)
        assert "Bland-Altman Analysis" in summary
        assert "95% LoA" in summary
        assert "ICC" in summary
