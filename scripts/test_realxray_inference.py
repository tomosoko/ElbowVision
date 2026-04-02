"""
STEP 7: 実ファントムX線 → YOLOv8-Pose 推論テスト
=====================================================

DRRで訓練したYOLOモデルを実X線に適用し、
前処理（CLAHE + ヒストグラムマッチング）の有無でキーポイント検出精度を比較する。

使い方:
  cd /Users/kohei/Dev/vision/ElbowVision
  source elbow-api/venv/bin/activate
  python scripts/test_realxray_inference.py

出力:
  results/realxray_inference/
  ├── inference_AP.png   ← AP像: 前処理なし vs 前処理ありの比較
  ├── inference_LAT.png  ← LAT像: 同上
  └── inference_report.txt ← 定量結果
"""

import os
import sys
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── パス設定 ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "elbow-api", "models", "yolo_pose_best.pt")

REAL_XRAY = {
    "AP":  os.path.join(PROJECT_ROOT, "data/real_xray/images/008_AP.png"),
    "LAT": os.path.join(PROJECT_ROOT, "data/real_xray/images/008_LAT.png"),
}

DRR_DATASET = os.path.join(PROJECT_ROOT, "data/yolo_dataset_v2/images")

OUT_DIR = os.path.join(PROJECT_ROOT, "results/realxray_inference")
os.makedirs(OUT_DIR, exist_ok=True)

# キーポイント定義
KP_NAMES = [
    "humerus_shaft",
    "lateral_epicondyle",
    "medial_epicondyle",
    "forearm_shaft",
    "radial_head",
    "olecranon",
]

KP_COLORS_BGR = [
    (0,   0,   255),  # 赤: humerus_shaft
    (0,   255, 0  ),  # 緑: lateral_epicondyle
    (255, 0,   0  ),  # 青: medial_epicondyle
    (0,   255, 255),  # 黄: forearm_shaft
    (255, 0,   255),  # マゼンタ: radial_head
    (255, 255, 0  ),  # シアン: olecranon
]

KP_COLORS_RGB = [(r, g, b) for (b, g, r) in KP_COLORS_BGR]


# ── ヒストグラムマッチング ────────────────────────────────────────────────────

def build_drr_cdf(dataset_dir: str, n_samples: int = 30) -> Optional[np.ndarray]:
    """
    DRR訓練データからサンプリングして平均CDFを構築する。
    実X線をDRR分布に近づけるヒストグラムマッチングに使用。
    """
    hist_sum = np.zeros(256, dtype=np.float64)
    count = 0
    for split in ["train", "val"]:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            continue
        files = sorted(f for f in os.listdir(split_dir) if f.endswith(".png"))
        step = max(1, len(files) // n_samples)
        for f in files[::step]:
            img = cv2.imread(os.path.join(split_dir, f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
            hist_sum += h
            count += 1
    if count == 0:
        return None
    cdf = hist_sum.cumsum()
    cdf /= cdf[-1] + 1e-8
    return cdf


def histogram_match(img: np.ndarray, ref_cdf: np.ndarray) -> np.ndarray:
    """画像ヒストグラムを ref_cdf に合わせるLUTを適用"""
    src_hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    src_cdf = src_hist.cumsum()
    src_cdf /= src_cdf[-1] + 1e-8
    lut = np.zeros(256, dtype=np.uint8)
    for s in range(256):
        lut[s] = int(np.argmin(np.abs(ref_cdf - src_cdf[s])))
    return lut[img]


# ── 前処理パイプライン ────────────────────────────────────────────────────────

def preprocess_xray(img_gray: np.ndarray,
                    drr_cdf: Optional[np.ndarray],
                    target_size: int = 256) -> np.ndarray:
    """
    実X線をDRR訓練データに近づける前処理:
    1. 256px にリサイズ（訓練時と同サイズ）
    2. CLAHE（コントラスト強調）
    3. DRR輝度分布へのヒストグラムマッチング
    """
    img = cv2.resize(img_gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    if drr_cdf is not None:
        img = histogram_match(img, drr_cdf)
    return img


# ── キーポイント描画 ──────────────────────────────────────────────────────────

def draw_keypoints(img_bgr: np.ndarray,
                   keypoints: Optional[np.ndarray],
                   confs: Optional[np.ndarray]) -> np.ndarray:
    """キーポイントをカラーコードで描画"""
    vis = img_bgr.copy()
    if keypoints is None:
        return vis
    for i, (kp, conf) in enumerate(zip(keypoints, confs)):
        x, y = int(kp[0]), int(kp[1])
        color = KP_COLORS_BGR[i]
        cv2.circle(vis, (x, y), 7, color, -1)
        cv2.circle(vis, (x, y), 9, (255, 255, 255), 1)
        label = f"{KP_NAMES[i][:6]}:{conf:.2f}"
        cv2.putText(vis, label, (x + 11, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, color, 1, cv2.LINE_AA)
    return vis


# ── 推論 ──────────────────────────────────────────────────────────────────────

def infer(model, img_bgr: np.ndarray, conf_thresh: float = 0.1):
    """
    YOLOv8-Pose で推論。検出なしの場合は (None, None, 0.0) を返す。
    NumPy 2.x と torch の非互換を回避するため、一時ファイル経由で推論する。
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, img_bgr)
    try:
        results = model(tmp_path, verbose=False, conf=conf_thresh)
    finally:
        os.remove(tmp_path)
    if not results or results[0].keypoints is None:
        return None, None, 0.0

    kps_all  = results[0].keypoints.xy.cpu().numpy()
    confs_all = results[0].keypoints.conf
    boxes    = results[0].boxes

    if len(kps_all) == 0:
        return None, None, 0.0

    box_conf = float(boxes.conf[0]) if len(boxes) > 0 else 0.0
    kps  = kps_all[0]
    confs = confs_all[0].cpu().numpy() if confs_all is not None else np.ones(6)
    return kps, confs, box_conf


# ── 1ビュー分の比較処理 ───────────────────────────────────────────────────────

def run_inference_on_view(model, view: str, real_path: str,
                           drr_cdf: Optional[np.ndarray]) -> dict:
    """
    AP または LAT 像に対して:
    - 前処理なし（256px リサイズのみ）
    - 前処理あり（CLAHE + ヒストグラムマッチング）
    の2条件で推論し、並列比較図を保存する。
    """
    print(f"\n{'─'*55}")
    print(f"  {view} 像推論")
    print(f"{'─'*55}")

    img_gray = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"  ERROR: 画像読み込み失敗: {real_path}")
        return {}

    # 前処理なし（256px のみ）
    raw = cv2.resize(img_gray, (256, 256), interpolation=cv2.INTER_AREA)
    raw_bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

    # 前処理あり
    prep = preprocess_xray(img_gray, drr_cdf)
    prep_bgr = cv2.cvtColor(prep, cv2.COLOR_GRAY2BGR)

    kps_raw,  confs_raw,  bconf_raw  = infer(model, raw_bgr)
    kps_prep, confs_prep, bconf_prep = infer(model, prep_bgr)

    # ── ターミナル出力 ──
    for label, kps, confs, bconf in [
        ("前処理なし", kps_raw,  confs_raw,  bconf_raw),
        ("前処理あり", kps_prep, confs_prep, bconf_prep),
    ]:
        print(f"  [{label}] bbox_conf={bconf:.3f}")
        if kps is not None:
            for i, (kp, c) in enumerate(zip(kps, confs)):
                print(f"    KP{i} {KP_NAMES[i]:<22}: ({kp[0]:5.1f},{kp[1]:5.1f}) conf={c:.3f}")
        else:
            print("    → 検出なし")

    # ── 可視化 ──
    vis_raw  = draw_keypoints(raw_bgr,  kps_raw,  confs_raw)
    vis_prep = draw_keypoints(prep_bgr, kps_prep, confs_prep)

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    fig.suptitle(f"Real X-ray Inference — {view} View", fontsize=14, fontweight="bold")

    for ax, vis, kps, bconf, title in [
        (axes[0], vis_raw,  kps_raw,  bconf_raw,  "No preprocessing"),
        (axes[1], vis_prep, kps_prep, bconf_prep, "CLAHE + Histogram Match"),
    ]:
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        detected_str = "DETECTED" if kps is not None else "NOT DETECTED"
        ax.set_title(f"{title}\nbbox_conf={bconf:.3f}  [{detected_str}]", fontsize=11)
        ax.axis("off")

    # 凡例
    legend = [
        Patch(facecolor=np.array(c) / 255.0, label=n)
        for c, n in zip(KP_COLORS_RGB, KP_NAMES)
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    out_path = os.path.join(OUT_DIR, f"inference_{view}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 保存: {out_path}")

    return {
        "view":           view,
        "raw_detected":   kps_raw  is not None,
        "prep_detected":  kps_prep is not None,
        "raw_bbox_conf":  bconf_raw,
        "prep_bbox_conf": bconf_prep,
        "kps_raw":        kps_raw,
        "kps_prep":       kps_prep,
        "confs_raw":      confs_raw,
        "confs_prep":     confs_prep,
    }


# ── エントリポイント ──────────────────────────────────────────────────────────

def main():
    # YOLO ロード
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics がインストールされていません")
        print("  pip install ultralytics")
        sys.exit(1)

    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"ERROR: YOLOモデルが見つかりません: {YOLO_MODEL_PATH}")
        sys.exit(1)

    print(f"モデルロード: {os.path.relpath(YOLO_MODEL_PATH, PROJECT_ROOT)}")
    model = YOLO(YOLO_MODEL_PATH)

    # DRR CDF 構築（ヒストグラムマッチング用）
    print(f"\nDRR訓練データからCDF構築中...")
    drr_cdf = build_drr_cdf(DRR_DATASET, n_samples=30)
    if drr_cdf is not None:
        print("  DRR CDF 構築完了（ヒストグラムマッチング有効）")
    else:
        print("  WARNING: DRRデータが見つかりません。ヒストグラムマッチングを無効化。")

    # 推論実行
    results = []
    for view in ["AP", "LAT"]:
        if not os.path.exists(REAL_XRAY[view]):
            print(f"\nSKIP: 実X線画像が見つかりません: {REAL_XRAY[view]}")
            continue
        r = run_inference_on_view(model, view, REAL_XRAY[view], drr_cdf)
        if r:
            results.append(r)

    if not results:
        print("\nERROR: 推論可能な実X線画像がありません")
        sys.exit(1)

    # ── サマリー ──
    print(f"\n{'='*55}")
    print("  推論結果サマリー")
    print(f"{'='*55}")

    lines = ["Domain Gap Inference Test Report", "=" * 55, ""]
    for r in results:
        raw_ok  = r["raw_detected"]
        prep_ok = r["prep_detected"]
        print(f"  {r['view']}像:")
        print(f"    前処理なし: {'検出OK' if raw_ok  else '検出なし'} "
              f"(bbox_conf={r['raw_bbox_conf']:.3f})")
        print(f"    前処理あり: {'検出OK' if prep_ok else '検出なし'} "
              f"(bbox_conf={r['prep_bbox_conf']:.3f})")

        lines += [
            f"[{r['view']} View]",
            f"  No preprocess : {'DETECTED' if raw_ok  else 'NOT DETECTED'} "
            f"(bbox_conf={r['raw_bbox_conf']:.3f})",
            f"  Preprocessed  : {'DETECTED' if prep_ok else 'NOT DETECTED'} "
            f"(bbox_conf={r['prep_bbox_conf']:.3f})",
        ]
        if r["confs_prep"] is not None:
            for i, (kp, c) in enumerate(zip(r["kps_prep"], r["confs_prep"])):
                lines.append(f"    KP{i} {KP_NAMES[i]:<22}: ({kp[0]:5.1f},{kp[1]:5.1f}) conf={c:.3f}")
        lines.append("")

    # 総合評価
    all_detected_prep = all(r["prep_detected"] for r in results)
    any_detected_raw  = any(r["raw_detected"]  for r in results)

    print(f"\n  STEP 7 評価:")
    if all_detected_prep:
        verdict = "PASS"
        msg = "前処理ありで全ビュー検出成功 → STEP 7 合格"
        action = "次のアクション: 本格データ収集へ（または実X線でFine-tuning）"
    elif any_detected_raw:
        verdict = "PARTIAL"
        msg = "部分的に検出可能（前処理で改善余地あり）"
        action = "次のアクション: 実X線でFine-tuning（finetune_real_xray.py）を推奨"
    else:
        verdict = "FAIL"
        msg = "未検出 → DRR生成パラメータの見直しが必要"
        action = "次のアクション: compare_drr_real.py の改善提案を参照"

    print(f"  [{verdict}] {msg}")
    print(f"  {action}")

    lines += [
        "Verdict: " + verdict,
        msg,
        action,
    ]

    # レポート保存
    report_path = os.path.join(OUT_DIR, "inference_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n結果保存先: {os.path.relpath(OUT_DIR, PROJECT_ROOT)}/")
    print(f"  inference_AP.png, inference_LAT.png, inference_report.txt")


if __name__ == "__main__":
    main()
