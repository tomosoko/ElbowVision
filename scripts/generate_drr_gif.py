"""
DRR GIFアニメーション生成 — CTから回旋角度を変えてDRRを連番生成し、GIF/MP4にする

新人向けプレゼン用デモ素材。
CTボリュームを読み込み、前腕の回旋角度を少しずつ変えながらDRRを生成。
肘が回る様子がアニメーションになる。

使い方:
  cd /Users/kohei/develop/research/ElbowVision
  source elbow-api/venv/bin/activate
  python scripts/generate_drr_gif.py
"""

import os
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "elbow-train"))

from elbow_synth import (
    load_ct_volume,
    auto_detect_landmarks,
    rotate_volume_and_landmarks,
    generate_drr,
)

# ── 設定 ──
CT_DIR = os.path.join(PROJECT_ROOT, "data/raw_dicom/ct")
OUT_DIR = os.path.join(PROJECT_ROOT, "slides")
FRAMES_DIR = os.path.join(OUT_DIR, "drr_frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

TARGET_SIZE = 256
HU_MIN = 50
HU_MAX = 800
LATERALITY = "L"

# 回旋角度の範囲（-30° ~ +30° を往復）
ROTATION_ANGLES = list(range(-30, 31, 3)) + list(range(30, -31, -3))
# 1フレームの表示時間（ms）
FRAME_DURATION_MS = 100


def add_label(img, text, position=(10, 25), font_scale=0.7):
    """画像にテキストラベルを追加"""
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
    cv2.putText(
        img_color, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (0, 255, 255), 2, cv2.LINE_AA,
    )
    return img_color


def main():
    t0 = time.time()
    print("=" * 50)
    print("DRR GIF Animation Generator")
    print("=" * 50)

    # CTボリューム読み込み
    print("\n[1/3] Loading CT volume...")
    volume, voxel_spacing, lat, voxel_mm = load_ct_volume(
        CT_DIR,
        target_size=TARGET_SIZE,
        laterality=LATERALITY,
        hu_min=HU_MIN,
        hu_max=HU_MAX,
    )
    landmarks = auto_detect_landmarks(volume)
    print(f"  Volume shape: {volume.shape}, voxel_mm: {voxel_mm:.3f}")

    # DRR連番生成
    print(f"\n[2/3] Generating {len(ROTATION_ANGLES)} DRR frames...")
    frames = []
    for i, rot_deg in enumerate(ROTATION_ANGLES):
        rot_vol, rot_lm = rotate_volume_and_landmarks(
            volume, landmarks,
            forearm_rotation_deg=float(rot_deg),
            flexion_deg=0.0,
            base_flexion=180,
        )
        drr = generate_drr(rot_vol, axis="AP", voxel_mm=voxel_mm)
        drr = cv2.resize(drr, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

        # ラベル付き
        labeled = add_label(drr, f"Rotation: {rot_deg:+d} deg")
        frames.append(labeled)

        # 個別フレーム保存
        fname = os.path.join(FRAMES_DIR, f"frame_{i:03d}.png")
        cv2.imwrite(fname, labeled)

        progress = (i + 1) / len(ROTATION_ANGLES) * 100
        print(f"  [{progress:5.1f}%] rotation={rot_deg:+3d}deg -> frame_{i:03d}.png")

    # GIF生成（Pillowを使用）
    print("\n[3/3] Creating GIF animation...")
    try:
        from PIL import Image

        pil_frames = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb))

        gif_path = os.path.join(OUT_DIR, "drr_animation.gif")
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=FRAME_DURATION_MS,
            loop=0,
        )
        print(f"  GIF saved: {gif_path}")
        print(f"  Size: {os.path.getsize(gif_path) / 1024:.0f} KB")
    except ImportError:
        print("  Pillow not available, skipping GIF. Frames saved in drr_frames/")

    # MP4も生成（ffmpegが使えれば）
    mp4_path = os.path.join(OUT_DIR, "drr_animation.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(mp4_path, fourcc, 10, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"  MP4 saved: {mp4_path}")
    print(f"  Size: {os.path.getsize(mp4_path) / 1024:.0f} KB")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — {len(frames)} frames generated")


if __name__ == "__main__":
    main()
