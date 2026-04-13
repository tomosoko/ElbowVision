"""ElbowVision v6 — 投影軸統一データセット訓練スクリプト
- LAT: 90°vol AP投影固定（v4方式）
- AP: 180°vol AP投影
- yolov8s-pose (v4_sgd_v2と同設定、mAP50=0.995実績)
"""
from ultralytics import YOLO
import torch
import time

print("=" * 60)
print("  ElbowVision v6 — YOLOv8s-pose 訓練")
print(f"  MPS GPU: {torch.backends.mps.is_available()}")
print(f"  PyTorch: {torch.__version__}")
print("=" * 60)

model = YOLO("yolov8s-pose.pt")  # v4で安定動作実績あり (mAP50=0.995)

start = time.time()
results = model.train(
    data="data/yolo_dataset_v6/dataset.yaml",
    epochs=200,
    imgsz=256,
    batch=64,
    device="mps",
    workers=8,
    patience=30,
    project="/Users/kohei/develop/research/ElbowVision/runs",
    name="elbow_v6",
    # v4より大きいbbox(w≈0.67)のためDFL勾配が大きい → loss weight調整
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    warmup_epochs=3,
    close_mosaic=10,
    amp=False,        # MPS環境でbfloat16回避のため必須
    # Loss weights: dfl/box を下げてbbox回帰勾配を弱める（v4 default: dfl=1.5, box=7.5）
    dfl=0.5,
    box=3.0,
    pose=12.0,
    # データ拡張（v4_sgd_v2と同じ）
    fliplr=0.5,
    mosaic=0.0,
    degrees=5.0,
    translate=0.05,
    scale=0.2,
    save=True,
    save_period=20,
    verbose=True,
)

elapsed = time.time() - start
print(f"\n{'=' * 60}")
print(f"  訓練完了: {elapsed:.0f}秒 ({elapsed/60:.1f}分)")
print(f"  Best mAP50(P): {results.results_dict.get('metrics/mAP50(P)', 'N/A')}")
print(f"  モデル保存先: runs/elbow_v6/weights/best.pt")
print("=" * 60)
