"""
DRRライブラリ事前構築スクリプト

患者CTからDRRを全角度分事前生成し、.npz キャッシュに保存する。
類似度マッチングの 41s/患者 → ~1s/患者 に短縮するための前処理ステップ。

使い方:
  python scripts/build_drr_library.py \
    --ct_dir "data/raw_dicom/ct_volume/..." \
    --out_path  data/drr_library/patient001.npz \
    --angle_min 60 --angle_max 180 --angle_step 1 \
    --laterality L --series_num 4

出力 .npz の構造:
  angles: float32 array (N,)        — 角度リスト (度)
  drrs:   uint8   array (N, H, W)   — DRR画像 (256x256)
  meta:   JSON文字列                 — CT情報・生成パラメータ
"""

from __future__ import annotations

import argparse
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

TARGET_SIZE = 256


def preprocess_drr(img: np.ndarray) -> np.ndarray:
    """DRR → uint8 256x256 グレースケール (CLAHE適用済み)"""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return img  # uint8


def build_library(
    ct_dir: str,
    laterality: str,
    series_num: int,
    hu_min: float,
    hu_max: float,
    base_flexion: float,
    angle_min: float,
    angle_max: float,
    angle_step: float,
    out_path: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"CT読み込み: {ct_dir}")
    volume, _, lat, voxel_mm = load_ct_volume(
        ct_dir,
        laterality=laterality,
        series_num=series_num,
        hu_min=hu_min,
        hu_max=hu_max,
        target_size=TARGET_SIZE,
    )
    landmarks = auto_detect_landmarks(volume, laterality=lat)

    angles = np.arange(angle_min, angle_max + angle_step / 2, angle_step, dtype=np.float32)
    n = len(angles)
    drrs = np.empty((n, TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

    print(f"DRR生成: {n}角度 ({angle_min}°〜{angle_max}°, step={angle_step}°)")
    t0 = time.time()
    for i, angle in enumerate(angles):
        rv, _ = rotate_volume_and_landmarks(
            volume, landmarks,
            forearm_rotation_deg=0.0,
            flexion_deg=float(angle),
            base_flexion=base_flexion,
            valgus_deg=0.0,
        )
        drr = generate_drr(rv, axis="LAT", sid_mm=1000.0, voxel_mm=voxel_mm)
        drrs[i] = preprocess_drr(drr)

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (n - i - 1)
        print(f"\r  {i+1}/{n}: {angle:.1f}° — 経過{elapsed:.0f}s ETA{eta:.0f}s", end="", flush=True)

    print(f"\n生成完了: {time.time()-t0:.1f}s ({(time.time()-t0)/n:.2f}s/角度)")

    meta = {
        "ct_dir": str(ct_dir),
        "laterality": lat,
        "series_num": series_num,
        "hu_min": hu_min,
        "hu_max": hu_max,
        "base_flexion": base_flexion,
        "angle_min": float(angle_min),
        "angle_max": float(angle_max),
        "angle_step": float(angle_step),
        "voxel_mm": float(voxel_mm),
        "target_size": TARGET_SIZE,
        "n_drrs": n,
    }

    np.savez_compressed(
        str(out_path),
        angles=angles,
        drrs=drrs,
        meta=np.array(json.dumps(meta, ensure_ascii=False)),
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"保存: {out_path} ({size_mb:.1f} MB, {n}枚)")


def main() -> None:
    parser = argparse.ArgumentParser(description="DRRライブラリ事前構築")
    parser.add_argument("--ct_dir", required=True)
    parser.add_argument("--out_path", required=True, help="出力 .npz ファイルパス")
    parser.add_argument("--laterality", default="L")
    parser.add_argument("--series_num", type=int, default=4)
    parser.add_argument("--hu_min", type=float, default=50.0)
    parser.add_argument("--hu_max", type=float, default=800.0)
    parser.add_argument("--base_flexion", type=float, default=180.0)
    parser.add_argument("--angle_min", type=float, default=60.0)
    parser.add_argument("--angle_max", type=float, default=180.0)
    parser.add_argument("--angle_step", type=float, default=1.0,
                        help="角度ステップ（デフォルト1°、粗探索のみなら5°）")
    args = parser.parse_args()

    build_library(
        ct_dir=str(_PROJECT_ROOT / args.ct_dir),
        laterality=args.laterality,
        series_num=args.series_num,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        base_flexion=args.base_flexion,
        angle_min=args.angle_min,
        angle_max=args.angle_max,
        angle_step=args.angle_step,
        out_path=str(_PROJECT_ROOT / args.out_path),
    )


if __name__ == "__main__":
    main()
