"""
DICOM → PNG 一括変換スクリプト

使い方:
  python dicom_to_png.py --input data/raw_dicom/ --output data/images/all/

オプション:
  --clahe       CLAHE コントラスト強調を適用（推奨）
  --size 512    出力サイズ（正方形にリサイズ、デフォルト: 元サイズ）
  --split       train/val を 80:20 で自動分割して配置
"""
import argparse
import os
import random
import shutil
import sys

import cv2
import numpy as np

try:
    import pydicom
    import pydicom.config
    pydicom.config.enforce_valid_values = False
except ImportError:
    print("pydicom が見つかりません。pip install pydicom を実行してください。")
    sys.exit(1)


def apply_windowing(arr, center, width):
    lower = center - width / 2.0
    upper = center + width / 2.0
    arr = np.clip(arr, lower, upper)
    return ((arr - lower) / width * 255).astype(np.uint8)


def dicom_to_array(path: str, apply_clahe: bool = True, output_size: int = 0) -> np.ndarray:
    ds = pydicom.dcmread(path)
    pixel = ds.pixel_array.astype(np.float32)

    # MONOCHROME1 反転
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixel = pixel.max() - pixel

    # ウィンドウイング
    wc = ds.get("WindowCenter")
    ww = ds.get("WindowWidth")
    if isinstance(wc, pydicom.multival.MultiValue): wc = wc[0]
    if isinstance(ww, pydicom.multival.MultiValue): ww = ww[0]

    if wc is not None and ww is not None:
        img = apply_windowing(pixel, float(wc), float(ww))
    else:
        mn, mx = pixel.min(), pixel.max()
        img = ((pixel - mn) / max(mx - mn, 1) * 255).astype(np.uint8)

    # CLAHE コントラスト強調
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

    # BGR化（YOLO訓練はBGRでもグレースケールでも可）
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # リサイズ
    if output_size > 0:
        img = cv2.resize(img, (output_size, output_size), interpolation=cv2.INTER_AREA)

    return img


def convert_dir(input_dir: str, output_dir: str, apply_clahe: bool, output_size: int) -> list:
    os.makedirs(output_dir, exist_ok=True)
    converted = []
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.dcm', '.dicom'))]

    if not files:
        print(f"DICOMファイルが見つかりません: {input_dir}")
        return converted

    print(f"{len(files)} ファイルを変換中...")
    for fname in sorted(files):
        in_path = os.path.join(input_dir, fname)
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, f"{stem}.png")
        try:
            img = dicom_to_array(in_path, apply_clahe=apply_clahe, output_size=output_size)
            cv2.imwrite(out_path, img)
            converted.append(out_path)
            print(f"  OK: {fname} -> {os.path.basename(out_path)}")
        except Exception as e:
            print(f"  SKIP: {fname} ({e})")

    return converted


def split_train_val(all_images: list, base_dir: str, val_ratio: float = 0.2):
    """変換済みPNGをtrain/valに分割してコピー"""
    random.shuffle(all_images)
    n_val = max(1, int(len(all_images) * val_ratio))
    val_set = all_images[:n_val]
    train_set = all_images[n_val:]

    train_dir = os.path.join(base_dir, "train")
    val_dir   = os.path.join(base_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for p in train_set:
        shutil.copy(p, os.path.join(train_dir, os.path.basename(p)))
    for p in val_set:
        shutil.copy(p, os.path.join(val_dir, os.path.basename(p)))

    print(f"\ntrain: {len(train_set)} 枚 → {train_dir}")
    print(f"val:   {len(val_set)} 枚 → {val_dir}")


def main():
    parser = argparse.ArgumentParser(description="DICOM → PNG 一括変換")
    parser.add_argument("--input",  required=True, help="DICOMファイルのディレクトリ")
    parser.add_argument("--output", required=True, help="PNG出力先ディレクトリ")
    parser.add_argument("--clahe",  action="store_true", default=True,
                        help="CLAHEコントラスト強調を適用（デフォルト: ON）")
    parser.add_argument("--no-clahe", dest="clahe", action="store_false")
    parser.add_argument("--size",   type=int, default=0,
                        help="出力サイズ（0=元サイズ, 512推奨）")
    parser.add_argument("--split",  action="store_true",
                        help="train/val を 80:20 で自動分割")
    args = parser.parse_args()

    print(f"入力: {args.input}")
    print(f"出力: {args.output}")
    print(f"CLAHE: {'ON' if args.clahe else 'OFF'}")
    print(f"サイズ: {'元サイズ' if args.size == 0 else f'{args.size}x{args.size}'}")
    print()

    converted = convert_dir(args.input, args.output, args.clahe, args.size)

    print(f"\n変換完了: {len(converted)} 枚")

    if args.split and converted:
        images_base = os.path.join(os.path.dirname(args.output), "images")
        split_train_val(converted, images_base)
        print("\n次のステップ:")
        print("  1. data/images/train/ と data/images/val/ の画像を LabelStudio でアノテーション")
        print("  2. YOLOラベルを data/labels/train/ と data/labels/val/ に配置")
        print("  3. python elbow-train/train_yolo_pose.py を実行")


if __name__ == "__main__":
    main()
