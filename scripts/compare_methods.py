"""
ElbowVision 手法比較スクリプト

ConvNeXt直接推定 vs 類似度マッチング を実X線で比較。
JSRT発表・論文のTable 2用データ生成。

使い方:
  python scripts/compare_methods.py \
    --ct_dir "data/raw_dicom/ct_volume/..." \
    --xray_dir data/real_xray/images/ \
    --gt_csv   data/real_xray/ground_truth.csv \
    --out_dir  results/method_comparison/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "elbow-train"))

from elbow_synth import (
    load_ct_volume,
    auto_detect_landmarks,
    rotate_volume_and_landmarks,
    generate_drr,
)

# ── ConvNeXt 推論 ─────────────────────────────────────────────────────────────

def load_convnext(model_path: str):
    """ConvNeXt-Smallモデルをロード"""
    import torch
    import torchvision.models as tvm

    class AngleEstimator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = tvm.convnext_small(
                weights=tvm.ConvNeXt_Small_Weights.IMAGENET1K_V1
            )
            self.backbone.classifier[2] = torch.nn.Linear(768, 1)

        def forward(self, x):
            return self.backbone(x).squeeze(-1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AngleEstimator().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    # チェックポイントは直接state_dict（OrderedDict）
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def predict_convnext(img_bgr: np.ndarray, model, device) -> float:
    """ConvNeXtで屈曲角を推定

    前処理: グレースケール → 256x256リサイズ → 90°時計回り回転 → RGB変換 → ImageNet正規化
    注意: DRR訓練データとのドメインギャップにより実X線では誤差~12-16°が残る
    """
    import torch
    import torchvision.transforms.functional as TF

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    # 256x256にリサイズしてから90°時計回り（DRR座標系に近づける）
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    tensor = TF.to_tensor(gray_rgb)
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    tensor = TF.normalize(tensor, mean, std)
    tensor = TF.resize(tensor, [224, 224])

    with torch.no_grad():
        out = model(tensor.unsqueeze(0).to(device))
    angle = float(out.item()) * (180 - 90) + 90
    return float(np.clip(angle, 90, 180))


# ── 類似度マッチング ──────────────────────────────────────────────────────────

def preprocess_drr(img: np.ndarray, size: int = 256) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return img.astype(np.float32) / 255.0


def crop_xray(img: np.ndarray, dark_thresh: int = 15, margin: float = 0.15) -> np.ndarray:
    mask = img > dark_thresh
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h, w = img.shape
    if ((rmax-rmin) * (cmax-cmin)) / (h * w) > 0.7:
        return img
    pad_r = int((rmax-rmin) * margin)
    pad_c = int((cmax-cmin) * margin)
    return img[max(0, rmin-pad_r):min(h-1, rmax+pad_r)+1,
               max(0, cmin-pad_c):min(w-1, cmax+pad_c)+1]


def preprocess_xray(img: np.ndarray, size: int = 256) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = crop_xray(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return img.astype(np.float32) / 255.0


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    af = a.ravel() - a.mean()
    bf = b.ravel() - b.mean()
    return float(np.dot(af, bf) / (np.sqrt((af**2).sum() * (bf**2).sum()) + 1e-8))


def edges(img: np.ndarray) -> np.ndarray:
    u8 = (img * 255).astype(np.uint8)
    return cv2.Canny(u8, 30, 90).astype(np.float32) / 255.0


def predict_similarity(
    xray_img: np.ndarray,
    volume: np.ndarray,
    landmarks: dict,
    lat: str,
    voxel_mm: float,
    base_flexion: float = 180.0,
    coarse_step: float = 5.0,
    fine_step: float = 1.0,
    fine_range: float = 10.0,
    angle_min: float = 60.0,
    angle_max: float = 180.0,
) -> tuple[float, dict]:
    """類似度マッチングで屈曲角を推定"""
    xray_norm = preprocess_xray(xray_img)
    xray_edge = edges(xray_norm)
    all_scores: dict[float, dict[str, float]] = {}

    def gen(angle):
        rv, _ = rotate_volume_and_landmarks(
            volume, landmarks,
            forearm_rotation_deg=0.0,
            flexion_deg=angle,
            base_flexion=base_flexion,
            valgus_deg=0.0,
        )
        drr = generate_drr(rv, axis="LAT", sid_mm=1000.0, voxel_mm=voxel_mm)
        dn = preprocess_drr(drr)
        return {"ncc": ncc(dn, xray_norm), "edge_ncc": ncc(edges(dn), xray_edge)}

    coarse_angles = np.arange(angle_min, angle_max + coarse_step, coarse_step).tolist()
    for a in coarse_angles:
        all_scores[a] = gen(a)

    coarse_best = max(coarse_angles, key=lambda a: all_scores[a]["ncc"])

    fine_min = max(angle_min, coarse_best - fine_range)
    fine_max = min(angle_max, coarse_best + fine_range)
    fine_angles = [a for a in np.arange(fine_min, fine_max + fine_step, fine_step)
                   if a not in all_scores]
    for a in fine_angles:
        all_scores[a] = gen(a)

    best_ncc  = max(all_scores, key=lambda a: all_scores[a]["ncc"])
    best_encc = max(all_scores, key=lambda a: all_scores[a]["edge_ncc"])
    combined  = (best_ncc + best_encc) / 2.0
    return combined, all_scores


# ── メイン ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ConvNeXt vs 類似度マッチング 比較評価")
    parser.add_argument("--ct_dir", required=True)
    parser.add_argument("--xray_dir", required=True)
    parser.add_argument("--gt_csv", required=True,
                        help="CSV: filename, gt_angle, laterality (省略可)")
    parser.add_argument("--model_path", default="runs/angle_estimator/best.pth")
    parser.add_argument("--series_num", type=int, default=4)
    parser.add_argument("--laterality", default="R")
    parser.add_argument("--hu_min", type=float, default=50.0)
    parser.add_argument("--hu_max", type=float, default=800.0)
    parser.add_argument("--out_dir", default="results/method_comparison")
    args = parser.parse_args()

    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # GT CSV読み込み
    gt_map: dict[str, float] = {}
    with open(_PROJECT_ROOT / args.gt_csv) as f:
        for row in csv.DictReader(f):
            gt_map[row["filename"]] = float(row["gt_angle"])

    # CT読み込み
    print("CT読み込み中...")
    volume, _, lat, voxel_mm = load_ct_volume(
        str(_PROJECT_ROOT / args.ct_dir),
        laterality=args.laterality,
        series_num=args.series_num,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        target_size=256,
    )
    landmarks = auto_detect_landmarks(volume, laterality=lat)

    # ConvNeXtモデルロード
    print("ConvNeXtモデルロード中...")
    model, device = load_convnext(str(_PROJECT_ROOT / args.model_path))

    # 各X線で評価
    results = []
    xray_dir = _PROJECT_ROOT / args.xray_dir

    for fname, gt_angle in sorted(gt_map.items()):
        xray_path = xray_dir / fname
        if not xray_path.exists():
            print(f"  SKIP: {fname} (not found)")
            continue

        xray_img = cv2.imread(str(xray_path))
        if xray_img is None:
            print(f"  SKIP: {fname} (read error)")
            continue

        print(f"\n[{fname}] GT={gt_angle}°")

        # ConvNeXt
        t0 = time.time()
        pred_convnext = predict_convnext(xray_img, model, device)
        t_convnext = time.time() - t0
        err_convnext = abs(pred_convnext - gt_angle)
        print(f"  ConvNeXt: {pred_convnext:.1f}° (err={err_convnext:.1f}°, {t_convnext:.1f}s)")

        # 類似度マッチング
        t0 = time.time()
        xray_gray = cv2.cvtColor(xray_img, cv2.COLOR_BGR2GRAY)
        pred_sim, _ = predict_similarity(xray_gray, volume, landmarks, lat, voxel_mm)
        t_sim = time.time() - t0
        err_sim = abs(pred_sim - gt_angle)
        print(f"  Similarity: {pred_sim:.1f}° (err={err_sim:.1f}°, {t_sim:.0f}s)")

        results.append({
            "filename":      fname,
            "gt_angle":      gt_angle,
            "pred_convnext": pred_convnext,
            "err_convnext":  err_convnext,
            "pred_sim":      pred_sim,
            "err_sim":       err_sim,
            "t_convnext":    round(t_convnext, 2),
            "t_sim":         round(t_sim, 1),
        })

    if not results:
        print("評価データなし")
        return

    # サマリー
    print(f"\n{'='*60}")
    print(f"{'X線':20} | {'GT':>6} | {'ConvNeXt':>9} | {'Err':>6} | {'Sim':>6} | {'Err':>6}")
    print("-" * 60)
    for r in results:
        print(f"{r['filename']:20} | {r['gt_angle']:6.1f} | {r['pred_convnext']:9.1f} | "
              f"{r['err_convnext']:6.1f} | {r['pred_sim']:6.1f} | {r['err_sim']:6.1f}")

    mae_cnx = np.mean([r["err_convnext"] for r in results])
    mae_sim = np.mean([r["err_sim"] for r in results])
    print("-" * 60)
    print(f"{'MAE':20} | {'':>6} | {'':>9} | {mae_cnx:6.2f} | {'':>6} | {mae_sim:6.2f}")

    # JSON保存
    summary = {
        "n": len(results),
        "mae_convnext": round(float(mae_cnx), 3),
        "mae_similarity": round(float(mae_sim), 3),
        "results": results,
    }
    (out_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"\n結果保存: {out_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
