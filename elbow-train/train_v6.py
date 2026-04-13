"""ElbowVision v6 — 投影軸統一データセット訓練スクリプト
- LAT: 90°vol AP投影固定（v4方式）
- AP: 180°vol AP投影
- yolo11n-pose (Ultralytics YOLO11)
"""
from ultralytics import YOLO
import torch
import time

print("=" * 60)
print("  ElbowVision v6 — YOLO11n-pose 訓練")
print(f"  MPS GPU: {torch.backends.mps.is_available()}")
print(f"  PyTorch: {torch.__version__}")
print("=" * 60)

model = YOLO("yolo11n-pose.pt")

start = time.time()
results = model.train(
    data="data/yolo_dataset_v6/dataset.yaml",
    epochs=200,
    imgsz=256,
    batch=64,
    device="mps",
    workers=8,
    patience=30,
    project="runs",
    name="elbow_v6",
    # データ拡張
    fliplr=0.5,
    mosaic=1.0,
    degrees=10.0,
    translate=0.1,
    scale=0.3,
    # optimizer (Muon uses bfloat16, not supported on MPS → AdamW)
    optimizer="AdamW",
    # 学習率
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=5,
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
