"""
ElbowVision 実X線推論テストスクリプト

指定ディレクトリのX線画像（PNG/DICOM）を一括推論し、
キーポイント・角度オーバーレイ画像とCSVを出力する。

使い方:
  cd ElbowVision/elbow-api
  python ../scripts/inference_test.py \
    --input_dir ../data/real_xray/images/ \
    --output_dir ../results/inference_test/ \
    --format both
"""
import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# elbow-api/main.py の推論ロジックを直接importするためパスを追加
_ELBOW_API_DIR = os.path.join(os.path.dirname(__file__), "..", "elbow-api")
sys.path.insert(0, os.path.abspath(_ELBOW_API_DIR))

from main import (
    _decode_image,
    detect_with_yolo_pose,
    detect_bone_landmarks_classical,
    estimate_positioning_correction,
    validate_angle_with_edges,
    yolo_model,
    convnext_model,
    convnext_transforms,
    device as convnext_device,
)

# ─── 定数 ──────────────────────────────────────────────────────────────────────
SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".dcm", ".dicom")

# キーポイント描画色 (BGR)
KPT_COLORS = {
    "humerus_shaft": (0, 255, 0),       # 緑
    "condyle_center": (0, 255, 255),     # 黄
    "lateral_epicondyle": (255, 0, 0),   # 青
    "medial_epicondyle": (255, 0, 128),  # 紫
    "forearm_shaft": (0, 128, 255),      # 橙
    "radial_head": (255, 128, 0),        # 水色
    "olecranon": (0, 0, 255),            # 赤
}

AXIS_COLOR = (0, 200, 200)  # 軸線
ANGLE_ARC_COLOR = (0, 255, 255)  # 角度弧


def draw_overlay(image: np.ndarray, landmarks: dict, correction: dict) -> np.ndarray:
    """キーポイント・軸線・角度をオーバーレイ描画"""
    vis = image.copy()
    h, w = vis.shape[:2]
    radius = max(6, int(min(h, w) * 0.008))
    thickness = max(2, int(min(h, w) * 0.003))
    font_scale = max(0.5, min(h, w) / 1200)

    # キーポイント描画
    for name, color in KPT_COLORS.items():
        pt = landmarks.get(name)
        if pt is None:
            continue
        cx, cy = int(pt["x"]), int(pt["y"])
        cv2.circle(vis, (cx, cy), radius, color, -1)
        cv2.circle(vis, (cx, cy), radius + 2, (255, 255, 255), 1)
        label_y = cy - radius - 5
        cv2.putText(vis, name.replace("_", " "), (cx + radius + 3, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, color, 1, cv2.LINE_AA)

    # 軸線（上腕骨 → 顆中心 → 前腕）
    def get_pt(name):
        p = landmarks.get(name)
        if p is None:
            return None
        return (int(p["x"]), int(p["y"]))

    humerus = get_pt("humerus_shaft")
    condyle = get_pt("condyle_center")
    forearm = get_pt("forearm_shaft")

    if humerus and condyle:
        cv2.line(vis, humerus, condyle, AXIS_COLOR, thickness, cv2.LINE_AA)
    if condyle and forearm:
        cv2.line(vis, condyle, forearm, AXIS_COLOR, thickness, cv2.LINE_AA)

    # 角度弧の描画
    if condyle:
        arc_radius = max(30, int(min(h, w) * 0.05))
        cv2.ellipse(vis, condyle, (arc_radius, arc_radius), 0, 0, 360,
                    ANGLE_ARC_COLOR, 1, cv2.LINE_AA)

    # 角度テキスト
    angles = landmarks.get("angles", {})
    qa = landmarks.get("qa", {})
    texts = []

    view_type = qa.get("view_type", "?")
    engine = qa.get("inference_engine", "classical_cv")
    qa_score = qa.get("score", 0)
    qa_status = qa.get("status", "?")

    texts.append(f"View: {view_type}  Engine: {engine}")
    texts.append(f"QA: {qa_score} ({qa_status})")

    carrying = angles.get("carrying_angle")
    flexion = angles.get("flexion")
    rotation_err = correction.get("rotation_error")

    if carrying is not None:
        texts.append(f"Carrying Angle: {carrying:.1f} deg")
    if flexion is not None:
        texts.append(f"Flexion: {flexion:.1f} deg")
    if rotation_err is not None:
        texts.append(f"Rotation Error: {rotation_err:.1f} deg")

    overall = correction.get("overall_level", "?")
    texts.append(f"Overall: {overall}")

    # テキスト背景付き描画（左上）
    y0 = 30
    for i, txt in enumerate(texts):
        y = y0 + int(i * 28 * font_scale * 1.5)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(vis, (10, y - th - 5), (10 + tw + 10, y + 5), (0, 0, 0), -1)
        cv2.putText(vis, txt, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


def run_inference(image_path: str) -> dict:
    """単一画像に対して推論を実行し結果を返す"""
    filename = os.path.basename(image_path)
    with open(image_path, "rb") as f:
        content = f.read()

    image_array = _decode_image(content, filename)

    # YOLOv8-Pose（プライマリ） → Classical CV（フォールバック）
    landmarks = detect_with_yolo_pose(image_array)
    if landmarks is None:
        landmarks = detect_bone_landmarks_classical(image_array)

    # エッジバリデーション
    angles = landmarks["angles"]
    primary_angle = angles["carrying_angle"] if angles["carrying_angle"] is not None else angles["flexion"]
    edge_validation = None
    if primary_angle is not None:
        edge_validation = validate_angle_with_edges(image_array, primary_angle)

    # ポジショニング補正推定
    correction = estimate_positioning_correction(image_array, landmarks)

    # ConvNeXt セカンドオピニオン（LAT屈曲角の上書き）
    if convnext_model is not None:
        try:
            import torch
            from PIL import Image as PILImage
            img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(img_rgb)
            img_t = convnext_transforms(pil_img).to(convnext_device)
            with torch.no_grad():
                pred = convnext_model(img_t.unsqueeze(0))[0].cpu().numpy()
            view = landmarks["qa"]["view_type"]
            if view == "LAT":
                landmarks["angles"]["flexion"] = round(float(pred[1]), 1)
            elif view == "AP":
                correction["rotation_error"] = round(float(pred[0]), 1)
        except Exception as e:
            print(f"ConvNeXt inference failed: {e}")

    return {
        "filename": filename,
        "image_array": image_array,
        "landmarks": landmarks,
        "correction": correction,
        "edge_validation": edge_validation,
    }


def collect_images(input_dir: str) -> list:
    """対象ディレクトリから画像ファイルを収集"""
    files = []
    for f in sorted(os.listdir(input_dir)):
        if f.lower().endswith(SUPPORTED_EXT) and not f.startswith("."):
            files.append(os.path.join(input_dir, f))
    return files


def build_csv_row(result: dict) -> dict:
    """推論結果からCSV行データを構築（Bland-Altman検証用フォーマット）"""
    lm = result["landmarks"]
    angles = lm["angles"]
    qa = lm["qa"]
    corr = result["correction"]
    edge = result["edge_validation"]

    return {
        "image_id": os.path.splitext(result["filename"])[0],
        "view_type": qa.get("view_type", ""),
        "inference_engine": qa.get("inference_engine", "classical_cv"),
        "ai_carrying_angle": angles.get("carrying_angle"),
        "ai_flexion_deg": angles.get("flexion"),
        "ai_rotation_error": corr.get("rotation_error"),
        "ai_pronation_sup": angles.get("pronation_sup"),
        "ai_varus_valgus": angles.get("varus_valgus"),
        "ps_label": angles.get("ps_label", ""),
        "vv_label": angles.get("vv_label", ""),
        "qa_score": qa.get("score"),
        "qa_status": qa.get("status", ""),
        "rotation_level": corr.get("rotation_level", ""),
        "flexion_level": corr.get("flexion_level", ""),
        "overall_level": corr.get("overall_level", ""),
        "edge_angle": edge["edge_angle"] if edge else None,
        "edge_confidence": edge["confidence"] if edge else None,
        "edge_agreement_deg": edge["agreement_deg"] if edge else None,
        # Bland-Altman検証用: 手動計測値を後から記入する列
        "manual_carrying_angle": "",
        "manual_flexion_deg": "",
    }


CSV_COLUMNS = [
    "image_id", "view_type", "inference_engine",
    "ai_carrying_angle", "ai_flexion_deg", "ai_rotation_error",
    "ai_pronation_sup", "ai_varus_valgus",
    "ps_label", "vv_label",
    "qa_score", "qa_status",
    "rotation_level", "flexion_level", "overall_level",
    "edge_angle", "edge_confidence", "edge_agreement_deg",
    "manual_carrying_angle", "manual_flexion_deg",
]


def print_summary(results: list):
    """サマリー統計をターミナルに表示"""
    carrying_vals = []
    flexion_vals = []
    rotation_vals = []
    engines = {}

    for r in results:
        angles = r["landmarks"]["angles"]
        corr = r["correction"]
        qa = r["landmarks"]["qa"]

        engine = qa.get("inference_engine", "classical_cv")
        engines[engine] = engines.get(engine, 0) + 1

        if angles.get("carrying_angle") is not None:
            carrying_vals.append(angles["carrying_angle"])
        if angles.get("flexion") is not None:
            flexion_vals.append(angles["flexion"])
        if corr.get("rotation_error") is not None:
            rotation_vals.append(corr["rotation_error"])

    print("\n" + "=" * 60)
    print("  ElbowVision 推論テスト サマリー")
    print("=" * 60)
    print(f"  画像数: {len(results)}")
    print(f"  推論エンジン: {', '.join(f'{k}({v})' for k, v in engines.items())}")
    print("-" * 60)

    def stats_line(name, vals):
        if not vals:
            print(f"  {name}: データなし")
            return
        arr = np.array(vals)
        print(f"  {name}:")
        print(f"    平均: {arr.mean():.1f} deg  |  SD: {arr.std():.1f} deg")
        print(f"    範囲: {arr.min():.1f} - {arr.max():.1f} deg  |  N={len(vals)}")

    stats_line("Carrying Angle (AP)", carrying_vals)
    stats_line("Flexion (LAT)", flexion_vals)
    stats_line("Rotation Error", rotation_vals)

    print("-" * 60)
    print("  次のステップ:")
    print("    1. CSVの manual_carrying_angle / manual_flexion_deg に手動計測値を記入")
    print("    2. bland_altman_analysis.py でBland-Altman検証を実行")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="ElbowVision 実X線推論テスト（API不要・直接推論）"
    )
    parser.add_argument(
        "--input_dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "real_xray", "images"),
        help="入力画像ディレクトリ (default: data/real_xray/images/)"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "results", "inference_test"),
        help="出力ディレクトリ (default: results/inference_test/)"
    )
    parser.add_argument(
        "--format", type=str, choices=["png", "csv", "both"], default="both",
        help="出力フォーマット: png=オーバーレイ画像, csv=結果CSV, both=両方"
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(input_dir):
        print(f"エラー: 入力ディレクトリが見つかりません: {input_dir}")
        sys.exit(1)

    images = collect_images(input_dir)
    if not images:
        print(f"エラー: 対象画像が見つかりません: {input_dir}")
        print(f"  対応形式: {', '.join(SUPPORTED_EXT)}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    engine_name = "YOLOv8-Pose" if yolo_model is not None else "Classical CV (fallback)"
    print(f"推論エンジン: {engine_name}")
    print(f"入力: {input_dir} ({len(images)} 枚)")
    print(f"出力: {output_dir}")
    print(f"フォーマット: {args.format}")
    print()

    results = []
    csv_rows = []

    for i, img_path in enumerate(images, 1):
        fname = os.path.basename(img_path)
        t0 = time.time()

        try:
            result = run_inference(img_path)
            elapsed = time.time() - t0
            results.append(result)

            angles = result["landmarks"]["angles"]
            qa = result["landmarks"]["qa"]
            corr = result["correction"]

            carrying = angles.get("carrying_angle")
            flexion = angles.get("flexion")
            rot_err = corr.get("rotation_error")

            carrying_str = f"{carrying:.1f}" if carrying is not None else "-"
            flexion_str = f"{flexion:.1f}" if flexion is not None else "-"
            rot_str = f"{rot_err:.1f}" if rot_err is not None else "-"

            print(f"  [{i}/{len(images)}] {fname:<25} "
                  f"carrying={carrying_str:>6}  flexion={flexion_str:>6}  "
                  f"rot_err={rot_str:>5}  QA={qa['score']}({qa['status']})  "
                  f"{elapsed:.2f}s")

            # オーバーレイ画像保存
            if args.format in ("png", "both"):
                overlay = draw_overlay(result["image_array"], result["landmarks"], corr)
                out_name = os.path.splitext(fname)[0] + "_result.png"
                cv2.imwrite(os.path.join(output_dir, out_name), overlay)

            # CSV行
            csv_rows.append(build_csv_row(result))

        except Exception as e:
            print(f"  [{i}/{len(images)}] {fname:<25} ERROR: {e}")

    # CSV出力
    if args.format in ("csv", "both") and csv_rows:
        csv_path = os.path.join(output_dir, "inference_results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nCSV出力: {csv_path}")

    # サマリー
    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
