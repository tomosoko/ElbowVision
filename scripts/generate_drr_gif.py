"""
DRR GIFアニメーション生成 — 3軸（回旋・屈曲・視点）で変化するDRRを生成

新人向けプレゼン用デモ素材。
CTボリュームを読み込み、3つのアニメーションを生成:
  1. 回旋（前腕の回内/回外）- AP view
  2. 屈曲（肘の曲げ伸ばし）- LAT view
  3. 視点回転（AP→LAT遷移）- 固定ポーズ

使い方:
  cd /Users/kohei/develop/research/ElbowVision
  /Users/kohei/develop/research/ElbowVision/elbow-api/venv/bin/python scripts/generate_drr_gif.py
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

CT_DIR = os.path.join(PROJECT_ROOT, "data/raw_dicom/ct")
OUT_DIR = os.path.join(PROJECT_ROOT, "slides")
FRAMES_DIR = os.path.join(OUT_DIR, "drr_frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

TARGET_SIZE = 256
HU_MIN = 50
HU_MAX = 800
LATERALITY = "L"
FRAME_DURATION_MS = 120


def add_labels(img, lines):
    """画像に複数行のテキストラベルを追加"""
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
    for i, text in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(
            img_color, text, (8, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 255), 1, cv2.LINE_AA,
        )
    return img_color


def make_gif(pil_frames, path):
    """PIL Imagesのリストからループするgifを生成"""
    pil_frames[0].save(
        path, save_all=True, append_images=pil_frames[1:],
        duration=FRAME_DURATION_MS, loop=0,
    )
    kb = os.path.getsize(path) / 1024
    print(f"    -> {path} ({kb:.0f} KB, {len(pil_frames)} frames)")


def bounce(values):
    """値のリストを往復にする（[1,2,3] -> [1,2,3,2,1]）"""
    return list(values) + list(reversed(values))[1:-1]


def main():
    from PIL import Image

    t0 = time.time()
    print("=" * 60)
    print("DRR 3-Axis GIF Animation Generator")
    print("=" * 60)

    # CTボリューム読み込み
    print("\n[1/4] Loading CT volume...")
    volume, voxel_spacing, lat, voxel_mm = load_ct_volume(
        CT_DIR, target_size=TARGET_SIZE, laterality=LATERALITY,
        hu_min=HU_MIN, hu_max=HU_MAX,
    )
    landmarks = auto_detect_landmarks(volume)
    print(f"  Volume: {volume.shape}, voxel_mm: {voxel_mm:.3f}")

    # ── GIF 1: 回旋（Rotation） ──
    print("\n[2/4] Generating rotation animation (AP view)...")
    rotation_angles = bounce(range(-25, 26, 5))
    rot_frames = []
    for rot_deg in rotation_angles:
        rot_vol, _ = rotate_volume_and_landmarks(
            volume, landmarks,
            forearm_rotation_deg=float(rot_deg), flexion_deg=0.0, base_flexion=180,
        )
        drr = generate_drr(rot_vol, axis="AP", voxel_mm=voxel_mm)
        drr = cv2.resize(drr, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        labeled = add_labels(drr, ["AP View", f"Rotation: {rot_deg:+d} deg"])
        rgb = cv2.cvtColor(labeled, cv2.COLOR_BGR2RGB)
        rot_frames.append(Image.fromarray(rgb))
        print(f"    rot={rot_deg:+3d}")
    make_gif(rot_frames, os.path.join(OUT_DIR, "drr_rotation.gif"))

    # ── GIF 2: 屈曲（Flexion） ──
    print("\n[3/4] Generating flexion animation (LAT view)...")
    flexion_offsets = bounce(range(-20, 21, 4))
    flex_frames = []
    for flex_off in flexion_offsets:
        rot_vol, _ = rotate_volume_and_landmarks(
            volume, landmarks,
            forearm_rotation_deg=0.0, flexion_deg=float(flex_off), base_flexion=180,
        )
        drr = generate_drr(rot_vol, axis="LAT", voxel_mm=voxel_mm)
        drr = cv2.resize(drr, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        labeled = add_labels(drr, ["LAT View", f"Flexion: {flex_off:+d} deg"])
        rgb = cv2.cvtColor(labeled, cv2.COLOR_BGR2RGB)
        flex_frames.append(Image.fromarray(rgb))
        print(f"    flex={flex_off:+3d}")
    make_gif(flex_frames, os.path.join(OUT_DIR, "drr_flexion.gif"))

    # ── GIF 3: 3軸合成（回旋+屈曲を同時に変化） ──
    print("\n[4/4] Generating combined 3-axis animation...")
    n_frames = 36
    combined_frames = []
    for i in range(n_frames):
        t = i / n_frames * 2 * np.pi
        rot_deg = 25 * np.sin(t)
        flex_deg = 15 * np.sin(t * 0.7)
        # AP/LAT切り替えはsinで補間（実際にはAP固定で回旋で視点変化を表現）
        rot_vol, _ = rotate_volume_and_landmarks(
            volume, landmarks,
            forearm_rotation_deg=float(rot_deg),
            flexion_deg=float(flex_deg),
            base_flexion=180,
        )
        drr = generate_drr(rot_vol, axis="AP", voxel_mm=voxel_mm)
        drr = cv2.resize(drr, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        labeled = add_labels(drr, [
            "3-Axis DRR Demo",
            f"Rot: {rot_deg:+5.1f}  Flex: {flex_deg:+5.1f}",
        ])
        rgb = cv2.cvtColor(labeled, cv2.COLOR_BGR2RGB)
        combined_frames.append(Image.fromarray(rgb))
        print(f"    frame {i+1}/{n_frames}: rot={rot_deg:+5.1f} flex={flex_deg:+5.1f}")
    make_gif(combined_frames, os.path.join(OUT_DIR, "drr_combined.gif"))

    # ── 3つ横に並べた合成GIF ──
    print("\n  Creating side-by-side composite GIF...")
    max_len = max(len(rot_frames), len(flex_frames), len(combined_frames))
    composite_frames = []
    gap = 4
    w = TARGET_SIZE * 3 + gap * 2
    h = TARGET_SIZE + 30  # 下にキャプション用スペース
    for i in range(max_len):
        canvas = Image.new("RGB", (w, h), (20, 20, 40))
        r = rot_frames[i % len(rot_frames)]
        f = flex_frames[i % len(flex_frames)]
        c = combined_frames[i % len(combined_frames)]
        canvas.paste(r, (0, 0))
        canvas.paste(f, (TARGET_SIZE + gap, 0))
        canvas.paste(c, (TARGET_SIZE * 2 + gap * 2, 0))
        composite_frames.append(canvas)
    make_gif(composite_frames, os.path.join(OUT_DIR, "drr_animation.gif"))

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  drr_rotation.gif  — 回旋のみ")
    print(f"  drr_flexion.gif   — 屈曲のみ")
    print(f"  drr_combined.gif  — 3軸合成")
    print(f"  drr_animation.gif — 3つ横並び（スライド用）")


if __name__ == "__main__":
    main()
