#!/usr/bin/env python3
"""
ElbowVision: Real X-ray Fine-tuning Pipeline
=============================================
DRR学習済モデルを実X線画像でfine-tuneするパイプライン。

手順:
  1. auto-annotate: 現行モデルで全実画像の6kpアノテーション自動生成
  2. augment:       JSONアノテーション → YOLO形式変換 + 20x augmentation
  3. finetune:      backbone凍結 + 低学習率でfine-tune
  4. evaluate:      before/after比較画像を生成

使い方:
  # Step 1: 自動アノテーション生成（→ 手動修正後に再実行）
  python scripts/finetune_real_xray.py --step auto-annotate

  # Step 2-4: augmentation → fine-tune → evaluate（一括）
  python scripts/finetune_real_xray.py --step all

  # 個別実行
  python scripts/finetune_real_xray.py --step augment
  python scripts/finetune_real_xray.py --step finetune
  python scripts/finetune_real_xray.py --step evaluate
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
REAL_IMG_DIR = BASE_DIR / "data" / "real_xray" / "images"
ANNOT_DIR = BASE_DIR / "data" / "real_xray"
FINETUNE_DIR = BASE_DIR / "data" / "finetune_real"
MODEL_PATH = BASE_DIR / "elbow-api" / "models" / "yolo_pose_best.pt"
RESULT_DIR = BASE_DIR / "results" / "finetune_real"

# All real X-ray images available for fine-tuning
ALL_IMAGES = [
    "008_AP.png",
    "008_LAT.png",
    "new_AP.png",
    "new_LAT.png",
    "cr_008_2_50kVp.png",
    "cr_008_3_52kVp.png",
]

# 6 keypoint names in order matching the model
KP_NAMES = [
    "humerus_shaft",
    "lateral_epicondyle",
    "medial_epicondyle",
    "forearm_shaft",
    "radial_head",
    "olecranon",
]

# For horizontal flip: lat/med epicondyles swap (idx 1<->2), rest stay
FLIP_IDX = [0, 2, 1, 3, 4, 5]

# Augmentation count per image
AUG_COUNT = 20

# Train/val split ratio
VAL_RATIO = 0.15


# ===========================================================================
# Step 1: Auto-annotate using current model
# ===========================================================================
def auto_annotate():
    """Run YOLO inference on all real images and save 6kp JSON annotations."""
    from ultralytics import YOLO

    print("=" * 60)
    print("Step 1: Auto-annotate real X-ray images")
    print("=" * 60)

    model = YOLO(str(MODEL_PATH))

    for img_name in ALL_IMAGES:
        img_path = REAL_IMG_DIR / img_name
        if not img_path.exists():
            print(f"  [SKIP] {img_name} not found")
            continue

        stem = img_path.stem
        out_json = ANNOT_DIR / f"{stem}_kpts.json"

        # Run inference
        img = Image.open(img_path)
        w, h = img.size

        results = model.predict(
            source=str(img_path),
            imgsz=512,
            conf=0.25,
            device="mps",
            verbose=False,
        )

        r = results[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            print(f"  [WARN] No detection for {img_name}")
            # Write empty placeholder
            data = {
                "image": f"data/real_xray/images/{img_name}",
                "width": w,
                "height": h,
                "needs_manual_annotation": True,
                "auto_generated": True,
                "keypoints": {},
            }
            with open(out_json, "w") as f:
                json.dump(data, f, indent=2)
            continue

        # Take best detection (highest confidence)
        kpts = r.keypoints.data[0].cpu().numpy()  # (6, 3) = x, y, conf
        boxes = r.boxes.data[0].cpu().numpy()       # x1, y1, x2, y2, conf, cls

        keypoints_dict = {}
        low_conf_kps = []
        for i, name in enumerate(KP_NAMES):
            kx, ky, kconf = kpts[i]
            entry = {
                "x": float(kx),
                "y": float(ky),
                "x_norm": float(kx / w),
                "y_norm": float(ky / h),
                "confidence": float(kconf),
            }
            if kconf < 0.3:
                entry["note"] = "low confidence - needs verification"
                low_conf_kps.append(name)
            keypoints_dict[name] = entry

        data = {
            "image": f"data/real_xray/images/{img_name}",
            "width": w,
            "height": h,
            "auto_generated": True,
            "needs_manual_verification": True,
            "detection_confidence": float(boxes[4]),
            "keypoints": keypoints_dict,
        }

        with open(out_json, "w") as f:
            json.dump(data, f, indent=2)

        n_ok = sum(1 for name in KP_NAMES if name not in low_conf_kps)
        print(f"  {img_name}: {n_ok}/6 kp confident, det_conf={boxes[4]:.3f}")
        if low_conf_kps:
            print(f"    Low confidence: {', '.join(low_conf_kps)}")

    print()
    print("Annotations saved to data/real_xray/*_kpts.json")
    print(">>> Please verify/correct annotations manually before fine-tuning:")
    print("    python elbow-train/annotate_keypoints.py <image> <output.json>")
    print()


# ===========================================================================
# Step 2: Augment — JSON → YOLO format + heavy augmentation
# ===========================================================================
def _load_annotation(img_name: str):
    """Load JSON annotation, return (keypoints_list, w, h) or None."""
    stem = Path(img_name).stem
    json_path = ANNOT_DIR / f"{stem}_kpts.json"
    if not json_path.exists():
        return None

    with open(json_path) as f:
        data = json.load(f)

    w = data["width"]
    h = data["height"]

    kps = data.get("keypoints", {})
    if len(kps) < 4:
        return None

    # Build ordered keypoints list [(x_norm, y_norm, visibility), ...]
    kp_list = []
    for name in KP_NAMES:
        if name in kps:
            entry = kps[name]
            xn = entry.get("x_norm", entry["x"] / w)
            yn = entry.get("y_norm", entry["y"] / h)
            vis = 2  # labeled and visible
            kp_list.append((xn, yn, vis))
        else:
            kp_list.append((0.0, 0.0, 0))  # not labeled

    return kp_list, w, h


def _compute_bbox_from_kps(kp_list, margin=0.08):
    """Compute bounding box from visible keypoints with margin."""
    xs = [kp[0] for kp in kp_list if kp[2] > 0]
    ys = [kp[1] for kp in kp_list if kp[2] > 0]
    if not xs:
        return 0.5, 0.5, 1.0, 1.0

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add margin
    w = (x_max - x_min) + margin * 2
    h = (y_max - y_min) + margin * 2
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    # Clamp
    w = min(w, 1.0)
    h = min(h, 1.0)
    cx = max(w / 2, min(1 - w / 2, cx))
    cy = max(h / 2, min(1 - h / 2, cy))

    return cx, cy, w, h


def _format_yolo_label(bbox, kp_list):
    """Format single YOLO pose label line."""
    cx, cy, bw, bh = bbox
    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for xn, yn, vis in kp_list:
        parts.append(f"{xn:.6f} {yn:.6f} {vis}")
    return " ".join(parts)


def _augment_image_and_kps(img_pil, kp_list, rng):
    """Apply one random augmentation, return (augmented_img, new_kp_list)."""
    img = img_pil.copy()
    kps = list(kp_list)
    w, h = img.size

    # 1. Horizontal flip (50% chance)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        new_kps = []
        for xn, yn, vis in kps:
            if vis > 0:
                new_kps.append((1.0 - xn, yn, vis))
            else:
                new_kps.append((0.0, 0.0, 0))
        # Apply flip_idx reordering
        kps = [new_kps[FLIP_IDX[i]] for i in range(len(new_kps))]

    # 2. Random rotation ±5°
    angle = rng.uniform(-5, 5)
    if abs(angle) > 0.5:
        # Rotate image
        img_np = np.array(img)
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_np = cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        img = Image.fromarray(img_np)

        # Rotate keypoints
        rad = math.radians(-angle)  # cv2 uses counter-clockwise
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        cx_px, cy_px = w / 2, h / 2
        new_kps = []
        for xn, yn, vis in kps:
            if vis > 0:
                px, py = xn * w, yn * h
                dx, dy = px - cx_px, py - cy_px
                nx = cos_a * dx - sin_a * dy + cx_px
                ny = sin_a * dx + cos_a * dy + cy_px
                new_kps.append((nx / w, ny / h, vis))
            else:
                new_kps.append((0.0, 0.0, 0))
        kps = new_kps

    # 3. Random brightness/contrast
    if rng.random() < 0.8:
        brightness = rng.uniform(0.7, 1.3)
        contrast = rng.uniform(0.7, 1.3)
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)

    # 4. Gaussian noise
    if rng.random() < 0.5:
        img_np = np.array(img).astype(np.float32)
        sigma = rng.uniform(3, 15)
        noise = np.random.normal(0, sigma, img_np.shape)
        img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)

    # 5. Random crop 90-100%
    if rng.random() < 0.7:
        crop_frac = rng.uniform(0.90, 1.0)
        crop_w = int(w * crop_frac)
        crop_h = int(h * crop_frac)
        max_x = w - crop_w
        max_y = h - crop_h
        x0 = rng.randint(0, max(max_x, 1))
        y0 = rng.randint(0, max(max_y, 1))

        img = img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
        img = img.resize((w, h), Image.LANCZOS)

        # Adjust keypoints
        new_kps = []
        for xn, yn, vis in kps:
            if vis > 0:
                # Map to crop coordinates
                px = xn * w - x0
                py = yn * h - y0
                new_xn = px / crop_w
                new_yn = py / crop_h
                # Check if still in bounds
                if 0 <= new_xn <= 1 and 0 <= new_yn <= 1:
                    new_kps.append((new_xn, new_yn, vis))
                else:
                    new_kps.append((
                        max(0, min(1, new_xn)),
                        max(0, min(1, new_yn)),
                        1,  # downgrade to "not visible but labeled"
                    ))
            else:
                new_kps.append((0.0, 0.0, 0))
        kps = new_kps

    # Clamp all keypoints to [0, 1]
    kps = [
        (max(0, min(1, xn)), max(0, min(1, yn)), vis)
        for xn, yn, vis in kps
    ]

    return img, kps


def augment():
    """Convert annotations to YOLO format and generate augmented dataset."""
    print("=" * 60)
    print("Step 2: Augment — build fine-tune dataset")
    print("=" * 60)

    # Clean output
    if FINETUNE_DIR.exists():
        shutil.rmtree(FINETUNE_DIR)

    for split in ("train", "val"):
        (FINETUNE_DIR / "images" / split).mkdir(parents=True)
        (FINETUNE_DIR / "labels" / split).mkdir(parents=True)

    rng = random.Random(42)
    np.random.seed(42)

    # Collect all valid image+annotation pairs
    valid_pairs = []
    for img_name in ALL_IMAGES:
        img_path = REAL_IMG_DIR / img_name
        if not img_path.exists():
            print(f"  [SKIP] {img_name}: image not found")
            continue

        result = _load_annotation(img_name)
        if result is None:
            print(f"  [SKIP] {img_name}: no annotation or < 4 keypoints")
            continue

        kp_list, w, h = result
        n_vis = sum(1 for kp in kp_list if kp[2] > 0)
        print(f"  {img_name}: {n_vis}/6 keypoints annotated")
        valid_pairs.append((img_name, kp_list))

    if not valid_pairs:
        print("\nERROR: No valid image+annotation pairs found!")
        print("Run --step auto-annotate first, then verify annotations.")
        sys.exit(1)

    print(f"\n  {len(valid_pairs)} images with annotations")
    print(f"  Generating {AUG_COUNT}x augmentations each...")

    # Generate augmented samples
    all_samples = []  # (img_pil, kp_list, base_name, aug_idx)

    for img_name, kp_list in valid_pairs:
        img_path = REAL_IMG_DIR / img_name
        img_pil = Image.open(img_path).convert("RGB")
        stem = Path(img_name).stem

        # Original (no augmentation)
        all_samples.append((img_pil.copy(), list(kp_list), stem, 0))

        # Augmented copies
        for aug_i in range(1, AUG_COUNT + 1):
            aug_img, aug_kps = _augment_image_and_kps(img_pil, kp_list, rng)
            all_samples.append((aug_img, aug_kps, stem, aug_i))

    # Shuffle and split train/val
    rng.shuffle(all_samples)
    n_val = max(1, int(len(all_samples) * VAL_RATIO))
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    print(f"  Total: {len(all_samples)} (train={len(train_samples)}, val={n_val})")

    # Write samples
    for split, samples in [("train", train_samples), ("val", val_samples)]:
        for img_pil, kp_list, stem, aug_i in samples:
            fname = f"{stem}_aug{aug_i:03d}"
            img_out = FINETUNE_DIR / "images" / split / f"{fname}.png"
            lbl_out = FINETUNE_DIR / "labels" / split / f"{fname}.txt"

            # Save image
            img_pil.save(str(img_out))

            # Compute bbox and write label
            bbox = _compute_bbox_from_kps(kp_list)
            label = _format_yolo_label(bbox, kp_list)
            lbl_out.write_text(label + "\n")

    # Write dataset.yaml
    yaml_path = FINETUNE_DIR / "dataset.yaml"
    yaml_content = f"""path: {FINETUNE_DIR}
train: images/train
val: images/val
nc: 1
names: [elbow_joint]
kpt_shape: [6, 3]
flip_idx: [0, 2, 1, 3, 4, 5]
"""
    yaml_path.write_text(yaml_content)

    print(f"\n  Dataset written to {FINETUNE_DIR}")
    print(f"  dataset.yaml: {yaml_path}")
    print()


# ===========================================================================
# Step 3: Fine-tune
# ===========================================================================
def finetune():
    """Fine-tune YOLO pose model on real X-ray augmented dataset."""
    from ultralytics import YOLO

    print("=" * 60)
    print("Step 3: Fine-tune YOLO pose on real X-ray data")
    print("=" * 60)

    yaml_path = FINETUNE_DIR / "dataset.yaml"
    if not yaml_path.exists():
        print("ERROR: dataset.yaml not found. Run --step augment first.")
        sys.exit(1)

    model = YOLO(str(MODEL_PATH))

    # Freeze backbone (first 10 layers)
    print("  Freezing first 10 backbone layers...")
    for i, (name, param) in enumerate(model.model.named_parameters()):
        if i < 10:
            param.requires_grad = False
            print(f"    Frozen: {name}")

    print()
    print("  Training config:")
    print("    base model:  yolo_pose_best.pt (DRR-trained)")
    print("    lr0:         0.00001")
    print("    epochs:      20")
    print("    imgsz:       512")
    print("    batch:       8")
    print("    device:      mps")
    print("    freeze:      10 layers")
    print()

    # Fine-tune
    results = model.train(
        data=str(yaml_path),
        epochs=20,
        imgsz=512,
        batch=8,
        lr0=0.00001,
        lrf=0.01,
        warmup_epochs=2,
        freeze=10,
        device="mps",
        cache="ram",
        project=str(RESULT_DIR),
        name="train",
        exist_ok=True,
        # Augmentation (YOLO built-in, in addition to our offline aug)
        hsv_h=0.01,
        hsv_s=0.1,
        hsv_v=0.2,
        degrees=3.0,
        translate=0.05,
        scale=0.1,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.0,       # Disable mosaic for small dataset
        mixup=0.0,        # Disable mixup for small dataset
        copy_paste=0.0,
        close_mosaic=0,
        # Regularization
        weight_decay=0.001,
        dropout=0.1,
    )

    # Find the best model
    best_path = RESULT_DIR / "train" / "weights" / "best.pt"
    if best_path.exists():
        # Copy to models directory
        out_model = BASE_DIR / "elbow-api" / "models" / "yolo_pose_finetuned.pt"
        shutil.copy2(best_path, out_model)
        print(f"\n  Fine-tuned model saved to: {out_model}")
    else:
        print("\n  WARNING: best.pt not found in training output")

    print()
    return results


# ===========================================================================
# Step 4: Evaluate before/after
# ===========================================================================
def evaluate():
    """Compare DRR-only model vs fine-tuned model on real X-rays."""
    from ultralytics import YOLO

    print("=" * 60)
    print("Step 4: Evaluate before/after fine-tuning")
    print("=" * 60)

    finetuned_path = BASE_DIR / "elbow-api" / "models" / "yolo_pose_finetuned.pt"
    if not finetuned_path.exists():
        # Try training output
        finetuned_path = RESULT_DIR / "train" / "weights" / "best.pt"
    if not finetuned_path.exists():
        print("ERROR: Fine-tuned model not found. Run --step finetune first.")
        sys.exit(1)

    eval_dir = RESULT_DIR / "comparison"
    eval_dir.mkdir(parents=True, exist_ok=True)

    model_before = YOLO(str(MODEL_PATH))
    model_after = YOLO(str(finetuned_path))

    KP_COLORS = [
        (0, 255, 0),      # humerus_shaft - green
        (255, 0, 0),      # lat_epi - blue
        (255, 0, 128),    # med_epi - purple
        (0, 128, 255),    # forearm_shaft - orange
        (255, 128, 0),    # radial_head - cyan
        (0, 0, 255),      # olecranon - red
    ]

    summary = []

    for img_name in ALL_IMAGES:
        img_path = REAL_IMG_DIR / img_name
        if not img_path.exists():
            continue

        img_orig = cv2.imread(str(img_path))
        if img_orig is None:
            continue

        # Run both models
        res_before = model_before.predict(
            str(img_path), imgsz=512, conf=0.1, device="mps", verbose=False
        )
        res_after = model_after.predict(
            str(img_path), imgsz=512, conf=0.1, device="mps", verbose=False
        )

        # Draw side-by-side comparison
        h, w = img_orig.shape[:2]
        canvas = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)

        def draw_kps(img, result, label):
            vis = img.copy()
            det_conf = 0.0
            kp_confs = []
            if result[0].keypoints is not None and len(result[0].keypoints) > 0:
                kpts = result[0].keypoints.data[0].cpu().numpy()
                det_conf = float(result[0].boxes.data[0][4])
                for i, (name, color) in enumerate(zip(KP_NAMES, KP_COLORS)):
                    x, y, conf = kpts[i]
                    kp_confs.append(conf)
                    if conf > 0.1:
                        cv2.circle(vis, (int(x), int(y)), 8, color, -1)
                        cv2.circle(vis, (int(x), int(y)), 8, (255, 255, 255), 2)
                        cv2.putText(
                            vis, f"{name[:3]} {conf:.2f}",
                            (int(x) + 10, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        )
            # Title bar
            cv2.rectangle(vis, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(
                vis, f"{label} (det={det_conf:.3f})",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )
            return vis, det_conf, kp_confs

        vis_before, conf_b, kpc_b = draw_kps(img_orig, res_before, "BEFORE (DRR-only)")
        vis_after, conf_a, kpc_a = draw_kps(img_orig, res_after, "AFTER (fine-tuned)")

        canvas[:, :w] = vis_before
        canvas[:, w + 20:] = vis_after

        out_path = eval_dir / f"compare_{Path(img_name).stem}.png"
        cv2.imwrite(str(out_path), canvas)

        # Summary
        avg_kp_b = np.mean(kpc_b) if kpc_b else 0
        avg_kp_a = np.mean(kpc_a) if kpc_a else 0
        summary.append({
            "image": img_name,
            "det_conf_before": conf_b,
            "det_conf_after": conf_a,
            "avg_kp_conf_before": avg_kp_b,
            "avg_kp_conf_after": avg_kp_a,
        })
        print(f"  {img_name}:")
        print(f"    det_conf:    {conf_b:.3f} -> {conf_a:.3f}")
        print(f"    avg_kp_conf: {avg_kp_b:.3f} -> {avg_kp_a:.3f}")

    # Also run GT comparison if annotations exist
    print()
    print("  Ground truth comparison (keypoint distance):")
    for img_name in ALL_IMAGES:
        result = _load_annotation(img_name)
        if result is None:
            continue

        kp_list, w_img, h_img = result
        img_path = REAL_IMG_DIR / img_name

        for model_label, model_obj in [("before", model_before), ("after", model_after)]:
            res = model_obj.predict(
                str(img_path), imgsz=512, conf=0.1, device="mps", verbose=False
            )
            if res[0].keypoints is None or len(res[0].keypoints) == 0:
                continue

            kpts = res[0].keypoints.data[0].cpu().numpy()
            dists = []
            for i, (gt_xn, gt_yn, gt_vis) in enumerate(kp_list):
                if gt_vis > 0:
                    pred_x, pred_y = kpts[i][0] / w_img, kpts[i][1] / h_img
                    dist = math.sqrt((pred_x - gt_xn) ** 2 + (pred_y - gt_yn) ** 2)
                    dists.append(dist)

            if dists:
                mean_dist = np.mean(dists) * 100  # percent of image
                print(f"    {img_name} [{model_label}]: mean_kp_error={mean_dist:.2f}%")

    # Save summary JSON
    summary_path = eval_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Comparison images saved to: {eval_dir}")
    print(f"  Summary: {summary_path}")
    print()


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO pose on real X-rays")
    parser.add_argument(
        "--step",
        choices=["auto-annotate", "augment", "finetune", "evaluate", "all"],
        default="all",
        help="Which step to run (default: all = augment+finetune+evaluate)",
    )
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if args.step == "auto-annotate":
        auto_annotate()
    elif args.step == "augment":
        augment()
    elif args.step == "finetune":
        finetune()
    elif args.step == "evaluate":
        evaluate()
    elif args.step == "all":
        augment()
        finetune()
        evaluate()


if __name__ == "__main__":
    main()
