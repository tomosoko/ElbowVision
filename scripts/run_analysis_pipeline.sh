#!/usr/bin/env bash
# run_analysis_pipeline.sh — ElbowVision Phase 1 解析パイプライン
#
# 全解析スクリプトを順番に実行し、論文用の図・表・CSVを生成する。
#
# 前提条件:
#   - data/drr_library/patient008_series4_R_60to180.npz が存在すること
#   - data/real_xray/ に画像・ground_truth.csv があること
#   - elbow-api/venv が構築済みであること
#
# 使い方:
#   cd /Users/kohei/develop/research/ElbowVision
#   bash scripts/run_analysis_pipeline.sh
#
#   # 特定ステップのみ実行:
#   bash scripts/run_analysis_pipeline.sh --steps "loo,metrics,latex"
#
# 出力:
#   results/self_test_loo/       — LOO検証結果
#   results/metric_comparison/   — メトリクス比較
#   results/robustness/          — 頑健性評価
#   results/confidence_calibration/ — 信頼度キャリブレーション
#   results/figures/             — 論文用図
#   results/paper_latex/         — LaTeXテーブル

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

PYTHON="$BASE_DIR/elbow-api/venv/bin/python3"
LIBRARY="data/drr_library/patient008_series4_R_60to180.npz"

# ── 引数パース ────────────────────────────────────────────────────────────────
STEPS_ARG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --steps) STEPS_ARG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

should_run() {
    local step="$1"
    if [[ -z "$STEPS_ARG" ]]; then
        return 0  # run all
    fi
    echo "$STEPS_ARG" | tr ',' '\n' | grep -qx "$step"
}

# ── ユーティリティ ────────────────────────────────────────────────────────────
hr() { echo ""; echo "$(printf '=%.0s' {1..60})"; echo "  $1"; echo "$(printf '=%.0s' {1..60})"; }
ok() { echo "  [OK] $1"; }
skip() { echo "  [SKIP] $1 (前提ファイルなし)"; }
fail() { echo "  [FAIL] $1"; echo "$2"; }

echo ""
echo "ElbowVision Phase 1 解析パイプライン"
echo "  Base : $BASE_DIR"
echo "  Python: $PYTHON"
echo "  Library: $LIBRARY"

# ── STEP 1: LOO 検証 ─────────────────────────────────────────────────────────
if should_run "loo"; then
    hr "STEP 1: DRR Leave-One-Out 検証（全121角度）"
    if [ ! -f "$BASE_DIR/$LIBRARY" ]; then
        skip "DRRライブラリが存在しない: $LIBRARY"
    else
        $PYTHON scripts/self_test_library.py \
            --library "$LIBRARY" \
            --test_angles all \
            --loo \
            --out_dir results/self_test_loo/
        ok "LOO検証完了 → results/self_test_loo/"

        # 境界除外版も保存
        $PYTHON scripts/self_test_library.py \
            --library "$LIBRARY" \
            --test_angles all \
            --loo \
            --exclude_boundary \
            --out_dir results/self_test_loo_no_boundary/
        ok "LOO（境界除外）完了 → results/self_test_loo_no_boundary/"
    fi
fi

# ── STEP 2: メトリクス比較（実X線） ──────────────────────────────────────────
if should_run "metrics"; then
    hr "STEP 2: メトリクス比較（実X線 NCC/edge-NCC/combined/NMI）"
    GT_CSV="data/real_xray/ground_truth.csv"
    XRAY_DIR="data/real_xray/images"
    if [ ! -f "$BASE_DIR/$GT_CSV" ] || [ ! -d "$BASE_DIR/$XRAY_DIR" ]; then
        skip "実X線データが存在しない: $XRAY_DIR / $GT_CSV"
    else
        $PYTHON scripts/compare_metrics.py \
            --library "$LIBRARY" \
            --xray_dir "$XRAY_DIR" \
            --gt_csv "$GT_CSV" \
            --out_dir results/metric_comparison/
        ok "メトリクス比較完了 → results/metric_comparison/"
    fi
fi

# ── STEP 3: 頑健性テスト ──────────────────────────────────────────────────────
if should_run "robustness"; then
    hr "STEP 3: 頑健性テスト（画像劣化 vs MAE）"
    if [ ! -f "$BASE_DIR/$LIBRARY" ]; then
        skip "DRRライブラリが存在しない"
    else
        $PYTHON scripts/test_robustness.py \
            --library "$LIBRARY" \
            --test_angles "90,100,110,120,130,140,150,160,170,180" \
            --out_dir results/robustness/ \
            --metric combined
        ok "頑健性テスト完了 → results/robustness/"
    fi
fi

# ── STEP 4: 信頼度キャリブレーション ─────────────────────────────────────────
if should_run "confidence"; then
    hr "STEP 4: 信頼度スコア解析（peak_ncc / sharpness）"
    LOO_CSV="results/self_test_loo/self_test_results.csv"
    RXRAY_CSV="results/confidence_calibration/realxray_eval.csv"
    if [ ! -f "$BASE_DIR/$LOO_CSV" ]; then
        skip "LOO CSVが存在しない（先にSTEP 1を実行）"
    else
        RXRAY_ARG=""
        if [ -f "$BASE_DIR/$RXRAY_CSV" ]; then
            RXRAY_ARG="--realxray_csv $RXRAY_CSV"
        fi
        $PYTHON scripts/calibrate_confidence.py \
            --loo_csv "$LOO_CSV" \
            $RXRAY_ARG \
            --out_dir results/confidence_calibration/
        ok "信頼度解析完了 → results/confidence_calibration/"
    fi
fi

# ── STEP 5: NCC曲線プロット ───────────────────────────────────────────────────
if should_run "ncc_curves"; then
    hr "STEP 5: NCC類似度曲線（DRR vs 実X線 ドメインギャップ可視化）"
    if [ ! -f "$BASE_DIR/$LIBRARY" ]; then
        skip "DRRライブラリが存在しない"
    else
        $PYTHON scripts/plot_ncc_curves.py \
            --library "$LIBRARY" \
            --out_dir results/figures/ \
            --gt_angle 90.0
        ok "NCC曲線完了 → results/figures/fig7_ncc_curves.png"
    fi
fi

# ── STEP 6: 総合サマリー図 ────────────────────────────────────────────────────
if should_run "overview"; then
    hr "STEP 6: 総合サマリー図（4パネル）"
    $PYTHON scripts/generate_overview_figure.py \
        --out_dir results/figures/
    ok "総合サマリー図完了 → results/figures/fig6_overview_summary.png"
fi

# ── STEP 6b: メトリクスバイアス図 ────────────────────────────────────────────
if should_run "metric_bias"; then
    hr "STEP 6b: メトリクスバイアス図（NCC/edge-NCC bias cancellation）"
    MC_CSV="results/metric_comparison/metric_comparison.csv"
    if [ ! -f "$BASE_DIR/$MC_CSV" ]; then
        skip "メトリクス比較CSVが存在しない（先にSTEP 2を実行）"
    else
        $PYTHON scripts/generate_metric_bias_figure.py \
            --csv "$MC_CSV" \
            --out_dir results/figures/
        ok "メトリクスバイアス図完了 → results/figures/fig8_metric_bias.png"
    fi
fi

# ── STEP 6c: LOO精度マップ図 ──────────────────────────────────────────────────
if should_run "loo_fig"; then
    hr "STEP 6c: LOO精度マップ図（fig9: GT vs Pred / Error / peak_ncc）"
    LOO_CSV="results/self_test_loo/self_test_results.csv"
    if [ ! -f "$BASE_DIR/$LOO_CSV" ]; then
        skip "LOO CSVが存在しない（先にSTEP 1を実行）"
    else
        $PYTHON scripts/generate_loo_figure.py \
            --loo_csv "$LOO_CSV" \
            --out_dir results/figures/
        ok "LOO精度マップ図完了 → results/figures/fig9_loo_accuracy.png"
    fi
fi

# ── STEP 7: LaTeX テーブル生成 ────────────────────────────────────────────────
if should_run "latex"; then
    hr "STEP 7: LaTeX テーブル自動生成"
    $PYTHON scripts/generate_paper_latex.py \
        --out_dir results/paper_latex/
    ok "LaTeXテーブル完了 → results/paper_latex/"
fi

# ── 最終サマリー ──────────────────────────────────────────────────────────────
hr "パイプライン完了"
echo ""
echo "生成された主要結果:"
echo ""

LOO_SUMMARY="$BASE_DIR/results/self_test_loo/self_test_summary.txt"
if [ -f "$LOO_SUMMARY" ]; then
    echo "  [LOO検証]"
    grep -E "MAE|RMSE|Mode" "$LOO_SUMMARY" | head -6 | sed 's/^/    /'
fi
echo ""

MC_CSV="$BASE_DIR/results/metric_comparison/metric_comparison.csv"
if [ -f "$MC_CSV" ]; then
    echo "  [メトリクス比較] → results/metric_comparison/metric_comparison.csv"
fi

echo ""
echo "  [論文用図] → results/figures/"
ls "$BASE_DIR/results/figures/"*.png 2>/dev/null | xargs -I{} basename {} | sed 's/^/    /'

echo ""
echo "  [LaTeXテーブル] → results/paper_latex/"
ls "$BASE_DIR/results/paper_latex/"*.tex 2>/dev/null | xargs -I{} basename {} | sed 's/^/    /'
echo ""
