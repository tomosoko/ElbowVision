"""
パイプライン健全性チェックスクリプト

全スクリプト・データ・結果ファイルの存在を確認し、
研究パイプラインの現状を報告する。

使い方:
  python scripts/check_pipeline_health.py
  python scripts/check_pipeline_health.py --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _check(label: str, path: Path, required: bool = True) -> bool:
    exists = path.exists()
    prefix = "[✓]" if exists else ("[✗]" if required else "[?]")
    note   = "" if exists else (" ← 必須" if required else " ← オプション（Phase 2で生成）")
    print(f"  {prefix} {label}: {path.relative_to(_PROJECT_ROOT)}{note}")
    return exists


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    R = _PROJECT_ROOT
    ok = True

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   ElbowVision パイプライン健全性チェック              ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Phase 1 必須データ ────────────────────────────────────────────────────
    print("【Phase 1: 基礎データ】")
    ok &= _check("CT RAW DICOM (Series 4)",
                 R / "data/raw_dicom/ct_volume",
                 required=False)
    ok &= _check("DRR ライブラリ (.npz)",
                 R / "data/drr_library/patient008_series4_R_60to180.npz",
                 required=True)
    ok &= _check("実X線 画像ディレクトリ",
                 R / "data/real_xray/images",
                 required=True)
    ok &= _check("実X線 GT CSV",
                 R / "data/real_xray/ground_truth.csv",
                 required=True)
    print()

    # ── 学習済みモデル ──────────────────────────────────────────────────────
    print("【学習済みモデル】")
    ok &= _check("ConvNeXt best.pth",
                 R / "runs/angle_estimator/best.pth",
                 required=False)
    print()

    # ── Phase 1 解析結果 ──────────────────────────────────────────────────────
    print("【Phase 1 解析結果】")
    _check("Bland-Altman summary.txt",
           R / "results/bland_altman/summary.txt")
    _check("Bland-Altman predictions.csv",
           R / "results/bland_altman/predictions.csv")
    _check("LOO self-test results.csv",
           R / "results/self_test_loo/self_test_results.csv")
    _check("LOO self-test summary.txt",
           R / "results/self_test_loo/self_test_summary.txt")
    _check("Metric comparison CSV",
           R / "results/metric_comparison/metric_comparison.csv")
    _check("Robustness results CSV",
           R / "results/robustness/robustness_results.csv")
    _check("Confidence calibration CSV",
           R / "results/confidence_calibration/realxray_eval.csv",
           required=False)
    print()

    # ── 論文用図 ──────────────────────────────────────────────────────────────
    print("【論文用図】")
    for fig_name in [
        "fig1_pipeline.png",
        "fig2_drr_algorithm.png",
        "fig3_drr_variations.png",
        "fig4_bland_altman.png",
        "fig5a_per_image_error.png",
        "fig5b_mae_summary.png",
        "fig5c_prediction_scatter.png",
        "fig6_overview_summary.png",
        "fig7_ncc_curves.png",
        "fig8_metric_bias.png",
        "fig9_loo_accuracy.png",
    ]:
        _check(fig_name, R / "results/figures" / fig_name, required=False)
    print()

    # ── LaTeX テーブル ────────────────────────────────────────────────────────
    print("【LaTeX テーブル】")
    for tex_name in [
        "table0_drr_dataset.tex",
        "table1_drr_bland_altman.tex",
        "table1b_loo_validation.tex",
        "table2_method_comparison.tex",
        "table3_metric_comparison.tex",
        "tableS1_robustness.tex",
    ]:
        _check(tex_name, R / "results/paper_latex" / tex_name, required=False)
    print()

    # ── Phase 2 ファイル ────────────────────────────────────────────────────────
    print("【Phase 2 準備（オプション）】")
    _check("Phase 2 患者リスト CSV",
           R / "data/real_xray/patients_phase2.csv",
           required=False)
    _check("Phase 2 eval results",
           R / "results/phase2_eval/predictions.csv",
           required=False)
    print()

    # ── 主要スクリプト ────────────────────────────────────────────────────────
    if args.verbose:
        print("【主要スクリプト】")
        for script in [
            "scripts/similarity_matching.py",
            "scripts/build_drr_library.py",
            "scripts/self_test_library.py",
            "scripts/compare_metrics.py",
            "scripts/test_robustness.py",
            "scripts/calibrate_confidence.py",
            "scripts/plot_ncc_curves.py",
            "scripts/generate_overview_figure.py",
            "scripts/generate_metric_bias_figure.py",
            "scripts/generate_loo_figure.py",
            "scripts/generate_paper_latex.py",
            "scripts/add_patient.py",
            "scripts/eval_realxray_batch.py",
            "scripts/run_analysis_pipeline.sh",
        ]:
            _check(script, R / script)
        print()

    # ── LOO サマリー表示 ─────────────────────────────────────────────────────
    loo_summary = R / "results/self_test_loo/self_test_summary.txt"
    if loo_summary.exists():
        print("【LOO検証 最新結果】")
        for line in loo_summary.read_text().splitlines():
            if any(k in line for k in ["MAE", "RMSE", "Mode", "n =", "Bias", "boundary"]):
                print(f"  {line.strip()}")
        print()

    # ── BA サマリー表示 ──────────────────────────────────────────────────────
    ba_summary = R / "results/bland_altman/summary.txt"
    if ba_summary.exists():
        print("【Bland-Altman 最新結果 (ConvNeXt val)】")
        for line in ba_summary.read_text().splitlines():
            if any(k in line for k in ["n=", "MAE", "Bias", "LoA", "ICC", "RMSE"]):
                print(f"  {line.strip()}")
        print()

    # ── 最終判定 ────────────────────────────────────────────────────────────
    if ok:
        print("✓ Phase 1 パイプライン: 正常")
    else:
        print("✗ 一部の必須ファイルが不足しています")
        print("  → run_analysis_pipeline.sh を実行して生成してください")

    print()


if __name__ == "__main__":
    main()
