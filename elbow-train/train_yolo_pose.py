"""
ElbowVision YOLOv8-Pose 訓練スクリプト

使い方:
  1. elbow_synth.py で DRR を生成
     python elbow_synth.py --ct_dir data/raw_dicom/ct/ --out_dir data/yolo_dataset/
  2. python train_yolo_pose.py

キーポイント定義（6点 -- dataset.yaml kpt_shape: [6, 3] と一致）:
  0: humerus_shaft      -- 上腕骨幹部（近位）
  1: lateral_epicondyle  -- 外側上顆
  2: medial_epicondyle   -- 内側上顆
  3: forearm_shaft       -- 前腕骨幹部（遠位：尺骨/橈骨）
  4: radial_head         -- 橈骨頭
  5: olecranon           -- 肘頭

dataset.yaml 構造:
  path: .../data/yolo_dataset
  train: images/train
  val: images/val
  nc: 1
  names: [elbow_joint]
  kpt_shape: [6, 3]
  flip_idx: [0, 2, 1, 3, 4, 5]
"""

import os

from ultralytics import YOLO


def main():
    print("ElbowVision: YOLOv8-Pose 訓練パイプライン開始")

    model = YOLO("yolov8n-pose.pt")

    yaml_candidates = [
        os.path.join(
            os.path.dirname(__file__), "..", "data", "yolo_dataset", "dataset.yaml"
        ),
        os.path.join(os.path.dirname(__file__), "dataset.yaml"),
        os.path.join(os.path.dirname(__file__), "..", "data", "dataset.yaml"),
    ]
    yaml_path = next(
        (p for p in yaml_candidates if os.path.exists(p)), yaml_candidates[0]
    )
    yaml_path = os.path.abspath(yaml_path)

    if not os.path.exists(yaml_path):
        print(f"Dataset config not found at: {yaml_path}")
        print("dataset.yaml を作成してから再実行してください。")
        print("  python elbow_synth.py --ct_dir ... --out_dir data/yolo_dataset/")
        return

    # dataset.yaml の内容を表示
    print(f"Dataset config: {yaml_path}")
    with open(yaml_path) as f:
        print(f"  {f.read().strip()}")

    print("Training... (GPU/MPS が利用可能な場合は自動使用)")

    # imgsz=512: elbow_synth.py のDRRサイズ（512x512）に合わせる
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=512,
        batch=16,
        device="mps",  # Apple Silicon。CPU使用時は 'cpu' に変更
        name="elbowvision_pose_model",
        pose=1.5,  # キーポイント損失の重み（精度重視）
        patience=20,  # Early stopping
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
    )

    print("Training Complete!")
    best_path = os.path.join(
        "runs", "pose", "elbowvision_pose_model", "weights", "best.pt"
    )
    print(f"Best model saved at: {best_path}")
    print("このファイルを elbow-api/ にコピーして使用してください。")
    print(f"  cp {best_path} ../elbow-api/")


if __name__ == "__main__":
    main()
