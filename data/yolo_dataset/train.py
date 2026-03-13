from ultralytics import YOLO
model = YOLO('yolov8n-pose.pt')
model.train(
    data='/content/data/yolo_dataset/dataset.yaml',
    epochs=100,
    imgsz=128,
    batch=32,
    device=0,
    project='/content/drive/MyDrive/ElbowVision/runs',
    name='elbow_drr_v1',
    patience=30,
)
