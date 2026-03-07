import os
import math
import uvicorn
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pydicom
import pydicom.config
import io
import cv2
import numpy as np
import base64

# DICOMメタデータの欠損に対して寛容に処理
pydicom.config.enforce_valid_values = False

app = FastAPI(title="ElbowVision API", version="1.0.0")

# ─── YOLOv8 Pose Model（プライマリ） ──────────────────────────────────────────

try:
    from ultralytics import YOLO
    YOLO_INSTALLED = True
except ImportError:
    YOLO_INSTALLED = False

_YOLO_CANDIDATE_PATHS = [
    "best.pt",
    "elbow_train/runs/pose/elbowvision_pose_model/weights/best.pt",
    "runs/pose/elbowvision_pose_model/weights/best.pt",
    "runs/pose/train/weights/best.pt",
]
YOLO_MODEL_PATH = next((p for p in _YOLO_CANDIDATE_PATHS if os.path.exists(p)), _YOLO_CANDIDATE_PATHS[0])
yolo_model = None

if YOLO_INSTALLED and os.path.exists(YOLO_MODEL_PATH):
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("Loaded YOLOv8 Pose Model.")
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
else:
    print("YOLOv8 Pose model not found. Falling back to Classical CV.")

# ─── ConvNeXt 角度回帰モデル + Grad-CAM（セカンドオピニオン） ────────────────

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False

# 出力: [carrying_angle, flexion, pronation_sup, varus_valgus]
CONVNEXT_OUTPUT_DIM = 4
CONVNEXT_MODEL_PATH = "elbow_convnext_best.pth"

convnext_model = None
device = None

if TORCH_INSTALLED:
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    class ElbowConvNeXt(nn.Module):
        """
        ConvNeXt-Small バックボーンによる肘関節角度回帰モデル。
        出力: [carrying_angle, flexion, pronation_sup, varus_valgus]
        AP像では carrying_angle のみ有効、LAT像では flexion のみ有効。
        """
        def __init__(self):
            super().__init__()
            self.backbone = models.convnext_small(weights=None)
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 128),
                nn.GELU(),
                nn.Linear(128, CONVNEXT_OUTPUT_DIM),
            )

        def forward(self, x):
            return self.backbone(x)

    if os.path.exists(CONVNEXT_MODEL_PATH):
        try:
            convnext_model = ElbowConvNeXt()
            convnext_model.load_state_dict(torch.load(CONVNEXT_MODEL_PATH, map_location=device))
            convnext_model.to(device)
            convnext_model.eval()
            print(f"Loaded ConvNeXt Angle Regressor on {device}.")
        except Exception as e:
            print(f"Failed to load ConvNeXt model: {e}")
    else:
        print(f"ConvNeXt model not found ({CONVNEXT_MODEL_PATH}). Second-opinion disabled.")

    convnext_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ─── Grad-CAM（ConvNeXtのlastconv層対応） ──────────────────────────────────────

class GradCAM:
    """
    ConvNeXt-Smallの最終特徴マップに対してGrad-CAMを計算。
    AIが「どこを見て角度を判断したか」を可視化するXAIツール。
    """
    def __init__(self, model: "nn.Module"):
        self.model = model
        self.activations: Optional["torch.Tensor"] = None
        self.gradients:   Optional["torch.Tensor"] = None
        # ConvNeXt-Smallの最終ステージ末尾ブロック
        target_layer = model.backbone.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor: "torch.Tensor", target_idx: Optional[int] = None) -> np.ndarray:
        """
        Grad-CAMヒートマップを生成（0〜1正規化）。
        target_idx: None=全出力の和, 0=carrying, 1=flexion, 2=pronation, 3=varus_valgus
        """
        self.model.eval()
        t = img_tensor.unsqueeze(0).to(device)
        t.requires_grad_(True)
        output = self.model(t)
        self.model.zero_grad()
        score = output.sum() if target_idx is None else output[0, target_idx]
        score.backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def apply_gradcam_overlay(image_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1 - alpha, heatmap, alpha, 0)


gradcam_engine: Optional[GradCAM] = None
if TORCH_INSTALLED and convnext_model is not None:
    try:
        gradcam_engine = GradCAM(convnext_model)
        print("GradCAM engine initialized.")
    except Exception as e:
        print(f"GradCAM init failed: {e}")

# ─── CORS ──────────────────────────────────────────────────────────────────────
_CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health Check ───────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "version": "1.0.0",
        "engines": {
            "yolo_pose":    yolo_model is not None,
            "convnext_xai": convnext_model is not None,
            "gradcam_xai":  gradcam_engine is not None,
            "classical_cv": True,
        },
        "message": "ElbowVision AI API is running."
    }


# ─── 共通ユーティリティ ─────────────────────────────────────────────────────────
def apply_windowing(image_array, center, width):
    if center is None or width is None:
        return image_array
    lower = center - (width / 2.0)
    upper = center + (width / 2.0)
    windowed = np.clip(image_array, lower, upper)
    windowed = ((windowed - lower) / width) * 255.0
    return windowed.astype(np.uint8)


def pct(v, total):
    return round(v / total * 100, 2)


def angle_deg(p1, p2):
    return math.degrees(math.atan2(p2["y"] - p1["y"], p2["x"] - p1["x"]))


def full_angle(a1, a2):
    diff = abs(a1 - a2) % 360
    return 360 - diff if diff > 180 else diff


# ─── YOLOv8-Pose 推論（プライマリ） ───────────────────────────────────────────
def detect_with_yolo_pose(image_array: np.ndarray) -> Optional[dict]:
    """
    キーポイント順:
      0: humerus_shaft    — 上腕骨幹部（近位）
      1: lateral_epicondyle — 外側上顆
      2: medial_epicondyle  — 内側上顆
      3: forearm_shaft    — 前腕骨幹部（遠位）
    """
    if yolo_model is None:
        return None

    h, w = image_array.shape[:2]

    try:
        results = yolo_model(image_array, verbose=False)
        if not results or len(results) == 0:
            return None
        result = results[0]
        if result.keypoints is None or result.keypoints.xy is None:
            return None

        kpts  = result.keypoints.xy[0].cpu().numpy()
        confs = result.keypoints.conf[0].cpu().numpy() if result.keypoints.conf is not None else np.ones(4)

        if len(kpts) < 4:
            return None

        humerus_pt  = {"x": float(kpts[0][0]), "y": float(kpts[0][1])}
        lat_epic_pt = {"x": float(kpts[1][0]), "y": float(kpts[1][1])}
        med_epic_pt = {"x": float(kpts[2][0]), "y": float(kpts[2][1])}
        forearm_pt  = {"x": float(kpts[3][0]), "y": float(kpts[3][1])}

        condyle_mid = {
            "x": (lat_epic_pt["x"] + med_epic_pt["x"]) / 2,
            "y": (lat_epic_pt["y"] + med_epic_pt["y"]) / 2,
        }

        epic_sep  = math.sqrt((lat_epic_pt["x"] - med_epic_pt["x"])**2 + (lat_epic_pt["y"] - med_epic_pt["y"])**2)
        view_type = "AP" if epic_sep > w * 0.08 else "LAT"

        humerus_axis_angle = angle_deg(humerus_pt, condyle_mid)
        forearm_axis_angle = angle_deg(condyle_mid, forearm_pt)
        joint_angle = round(full_angle(humerus_axis_angle, forearm_axis_angle), 1)

        if view_type == "AP":
            carrying_angle, flexion = joint_angle, None
        else:
            flexion, carrying_angle = joint_angle, None

        shaft_midx = (humerus_pt["x"] + forearm_pt["x"]) / 2
        lat_offset = lat_epic_pt["x"] - shaft_midx
        med_offset = med_epic_pt["x"] - shaft_midx
        if abs(lat_offset) + abs(med_offset) > 1e-3:
            asymmetry = (abs(lat_offset) - abs(med_offset)) / (abs(lat_offset) + abs(med_offset))
        else:
            asymmetry = 0.0
        pronation_sup = round(max(-30.0, min(30.0, asymmetry * 30.0)), 1)
        ps_label = "回内 (Pronation)" if pronation_sup > 2 else ("回外 (Supination)" if pronation_sup < -2 else "中立 (Neutral)")

        condyle_tilt = angle_deg(med_epic_pt, lat_epic_pt)
        varus_valgus = round(condyle_tilt, 1)
        vv_label = "外反 (Valgus)" if varus_valgus > 2 else ("内反 (Varus)" if varus_valgus < -2 else "中立 (Neutral)")

        avg_conf = float(np.mean(confs))
        if avg_conf > 0.7:
            qa_score, qa_status, qa_color = 95, "GOOD", "green"
            qa_msg = f"YOLOv8-Pose: 高信頼度検出 (conf={avg_conf:.2f})"
        elif avg_conf > 0.4:
            qa_score, qa_status, qa_color = 70, "FAIR", "yellow"
            qa_msg = f"YOLOv8-Pose: 中程度の信頼度 (conf={avg_conf:.2f})"
        else:
            qa_score, qa_status, qa_color = 40, "POOR", "red"
            qa_msg = f"YOLOv8-Pose: 低信頼度 (conf={avg_conf:.2f}). 再撮影を強く推奨。"

        if view_type == "AP" and abs(pronation_sup) > 5:
            direction = "回内" if pronation_sup > 0 else "回外"
            correction = "回外" if pronation_sup > 0 else "回内"
            positioning_advice = f"► {direction}が検出されました。側面撮影時は前腕を「{correction}」させてください。"
        else:
            positioning_advice = "► ポジショニングは良好です。現在の軸を維持してください。"

        forearm_ext = {
            "x": condyle_mid["x"] + (forearm_pt["x"] - condyle_mid["x"]) * 1.5,
            "y": condyle_mid["y"] + (forearm_pt["y"] - condyle_mid["y"]) * 1.5,
        }

        return {
            "humerus_shaft":      {"x": int(humerus_pt["x"]),    "y": int(humerus_pt["y"]),    "x_pct": pct(humerus_pt["x"], w),    "y_pct": pct(humerus_pt["y"], h)},
            "condyle_center":     {"x": int(condyle_mid["x"]),   "y": int(condyle_mid["y"]),   "x_pct": pct(condyle_mid["x"], w),   "y_pct": pct(condyle_mid["y"], h)},
            "lateral_epicondyle": {"x": int(lat_epic_pt["x"]),   "y": int(lat_epic_pt["y"]),   "x_pct": pct(lat_epic_pt["x"], w),   "y_pct": pct(lat_epic_pt["y"], h)},
            "medial_epicondyle":  {"x": int(med_epic_pt["x"]),   "y": int(med_epic_pt["y"]),   "x_pct": pct(med_epic_pt["x"], w),   "y_pct": pct(med_epic_pt["y"], h)},
            "forearm_shaft":      {"x": int(forearm_pt["x"]),    "y": int(forearm_pt["y"]),    "x_pct": pct(forearm_pt["x"], w),    "y_pct": pct(forearm_pt["y"], h)},
            "forearm_ext":        {"x": int(forearm_ext["x"]),   "y": int(forearm_ext["y"]),   "x_pct": pct(forearm_ext["x"], w),   "y_pct": pct(forearm_ext["y"], h)},
            "qa": {
                "view_type": view_type, "score": qa_score, "status": qa_status,
                "message": qa_msg, "color": qa_color, "symmetry_ratio": 1.0,
                "positioning_advice": positioning_advice,
                "inference_engine": "YOLOv8-Pose",
                "keypoint_confidences": [round(float(c), 3) for c in confs],
            },
            "angles": {
                "carrying_angle": carrying_angle, "flexion": flexion,
                "pronation_sup": pronation_sup, "ps_label": ps_label,
                "varus_valgus": varus_valgus, "vv_label": vv_label,
            },
        }
    except Exception as e:
        print(f"YOLOv8 Pose inference failed: {e}")
        return None


# ─── Classical CV フォールバック ───────────────────────────────────────────────
def detect_bone_landmarks_classical(image_array: np.ndarray) -> dict:
    h, w = image_array.shape[:2]

    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array.copy()

    gray = gray.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    _, bone_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_CLOSE, kernel)
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bone_mask)
    min_area = h * w * 0.01
    bone_regions = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            bone_regions.append({
                "label": i, "area": area,
                "cx": float(centroids[i][0]), "cy": float(centroids[i][1]),
            })
    bone_regions.sort(key=lambda r: r["cy"])

    humerus_pt  = {"x": w * 0.5, "y": h * 0.15}
    condyle_mid = {"x": w * 0.5, "y": h * 0.5}
    forearm_pt  = {"x": w * 0.5, "y": h * 0.85}
    lat_epic_pt = {"x": w * 0.5 + w * 0.08, "y": h * 0.5}
    med_epic_pt = {"x": w * 0.5 - w * 0.08, "y": h * 0.5}

    upper_regions = [r for r in bone_regions if r["cy"] < h * 0.55]
    lower_regions = [r for r in bone_regions if r["cy"] >= h * 0.45]

    if upper_regions:
        humerus_bone = max(upper_regions, key=lambda r: r["area"])
        hm = (labels == humerus_bone["label"]).astype(np.uint8)
        ys, xs = np.where(hm)
        if len(ys):
            bottom_row = int(ys.max())
            cols_at_bottom = xs[ys == bottom_row]
            condyle_mid = {"x": float(cols_at_bottom.mean()), "y": float(bottom_row)}
            lat_epic_pt = {"x": float(cols_at_bottom.mean()) + w * 0.05, "y": float(bottom_row)}
            med_epic_pt = {"x": float(cols_at_bottom.mean()) - w * 0.05, "y": float(bottom_row)}
            top_row = int(ys.min())
            cols_at_top = xs[ys == top_row]
            humerus_pt = {"x": float(cols_at_top.mean()), "y": float(top_row)}

    if lower_regions:
        forearm_bone = max(lower_regions, key=lambda r: r["area"])
        fm = (labels == forearm_bone["label"]).astype(np.uint8)
        yf, xf = np.where(fm)
        if len(yf):
            cols_bottom = xf[yf == int(yf.max())]
            forearm_pt = {"x": float(cols_bottom.mean()), "y": float(yf.max())}

    epic_sep  = math.sqrt((lat_epic_pt["x"] - med_epic_pt["x"])**2 + (lat_epic_pt["y"] - med_epic_pt["y"])**2)
    view_type = "AP" if epic_sep > w * 0.08 else "LAT"

    shaft_midx = (humerus_pt["x"] + forearm_pt["x"]) / 2
    lat_offset = lat_epic_pt["x"] - shaft_midx
    med_offset = med_epic_pt["x"] - shaft_midx
    if abs(lat_offset) + abs(med_offset) > 1e-3:
        asymmetry = (abs(lat_offset) - abs(med_offset)) / (abs(lat_offset) + abs(med_offset))
    else:
        asymmetry = 0.0
    pronation_sup = round(max(-30.0, min(30.0, asymmetry * 30.0)), 1)
    ps_label = "回内 (Pronation)" if pronation_sup > 2 else ("回外 (Supination)" if pronation_sup < -2 else "中立 (Neutral)")

    humerus_axis_angle = angle_deg(humerus_pt, condyle_mid)
    forearm_axis_angle = angle_deg(condyle_mid, forearm_pt)
    joint_angle = round(full_angle(humerus_axis_angle, forearm_axis_angle), 1)

    if view_type == "AP":
        carrying_angle, flexion = joint_angle, None
    else:
        flexion, carrying_angle = joint_angle, None

    varus_valgus = round(angle_deg(med_epic_pt, lat_epic_pt), 1)
    vv_label = "外反 (Valgus)" if varus_valgus > 2 else ("内反 (Varus)" if varus_valgus < -2 else "中立 (Neutral)")

    if view_type == "AP":
        left_dist  = abs(shaft_midx - min(lat_epic_pt["x"], med_epic_pt["x"]))
        right_dist = abs(max(lat_epic_pt["x"], med_epic_pt["x"]) - shaft_midx)
        symmetry_ratio = min(left_dist, right_dist) / max(left_dist, right_dist) if max(left_dist, right_dist) > 0 else 1.0
    else:
        symmetry_ratio = 1.0 - abs(pronation_sup) / 30.0

    if symmetry_ratio > 0.9:
        qa_score, qa_status, qa_color = 95, "GOOD", "green"
        qa_msg = "ポジショニングは良好です。"
        positioning_advice = "► ポジショニングは良好です。現在の軸を維持してください。"
    elif symmetry_ratio > 0.75:
        qa_score, qa_status, qa_color = 75, "FAIR", "yellow"
        qa_msg = f"軽度の位置ズレが検出されました。非対称性({symmetry_ratio:.2f})"
        positioning_advice = "► 軽微なズレがあります。前腕の回旋を微調整してください。"
    else:
        qa_score, qa_status, qa_color = 45, "POOR", "red"
        qa_msg = f"重度のポジショニングエラー。非対称性({symmetry_ratio:.2f})。再撮影を推奨。"
        positioning_advice = "► 再撮影を推奨します。前腕の回旋・肘の位置を確認してください。"

    forearm_ext = {
        "x": condyle_mid["x"] + (forearm_pt["x"] - condyle_mid["x"]) * 1.5,
        "y": condyle_mid["y"] + (forearm_pt["y"] - condyle_mid["y"]) * 1.5,
    }

    return {
        "humerus_shaft":      {"x": int(humerus_pt["x"]),   "y": int(humerus_pt["y"]),   "x_pct": pct(humerus_pt["x"], w),   "y_pct": pct(humerus_pt["y"], h)},
        "condyle_center":     {"x": int(condyle_mid["x"]),  "y": int(condyle_mid["y"]),  "x_pct": pct(condyle_mid["x"], w),  "y_pct": pct(condyle_mid["y"], h)},
        "lateral_epicondyle": {"x": int(lat_epic_pt["x"]),  "y": int(lat_epic_pt["y"]),  "x_pct": pct(lat_epic_pt["x"], w),  "y_pct": pct(lat_epic_pt["y"], h)},
        "medial_epicondyle":  {"x": int(med_epic_pt["x"]),  "y": int(med_epic_pt["y"]),  "x_pct": pct(med_epic_pt["x"], w),  "y_pct": pct(med_epic_pt["y"], h)},
        "forearm_shaft":      {"x": int(forearm_pt["x"]),   "y": int(forearm_pt["y"]),   "x_pct": pct(forearm_pt["x"], w),   "y_pct": pct(forearm_pt["y"], h)},
        "forearm_ext":        {"x": int(forearm_ext["x"]),  "y": int(forearm_ext["y"]),  "x_pct": pct(forearm_ext["x"], w),  "y_pct": pct(forearm_ext["y"], h)},
        "qa": {
            "view_type": view_type, "score": qa_score, "status": qa_status,
            "message": qa_msg, "color": qa_color,
            "symmetry_ratio": round(symmetry_ratio, 2),
            "positioning_advice": positioning_advice,
        },
        "angles": {
            "carrying_angle": carrying_angle, "flexion": flexion,
            "pronation_sup": pronation_sup, "ps_label": ps_label,
            "varus_valgus": varus_valgus, "vv_label": vv_label,
        },
    }


# ─── DICOM/画像デコード ─────────────────────────────────────────────────────────
def _decode_image(content: bytes, filename: str) -> np.ndarray:
    fname = filename.lower()
    if fname.endswith(('.dcm', '.dicom')):
        ds = pydicom.dcmread(io.BytesIO(content))
        pixel = ds.pixel_array
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixel = np.max(pixel) - pixel
        wc = ds.get("WindowCenter")
        ww = ds.get("WindowWidth")
        if isinstance(wc, pydicom.multival.MultiValue): wc = wc[0]
        if isinstance(ww, pydicom.multival.MultiValue): ww = ww[0]
        if wc is not None and ww is not None:
            pixel = apply_windowing(pixel, float(wc), float(ww))
        else:
            mn, mx = pixel.min(), pixel.max()
            pixel = ((pixel - mn) / max(mx - mn, 1) * 255).astype(np.uint8)
        if len(pixel.shape) == 2:
            return cv2.cvtColor(pixel, cv2.COLOR_GRAY2BGR)
        return pixel
    else:
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image.")
        return img


# ─── エンドポイント ────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    allowed_ext = ('.dcm', '.dicom', '.png', '.jpg', '.jpeg')
    if not (file.filename or "").lower().endswith(allowed_ext):
        raise HTTPException(status_code=400, detail=f"Supported formats: {', '.join(allowed_ext)}")
    try:
        content = await file.read()
        fname = (file.filename or "").lower()
        if fname.endswith(('.dcm', '.dicom')):
            ds = pydicom.dcmread(io.BytesIO(content))
            metadata = {
                "PatientName":  str(ds.get("PatientName", "Unknown")),
                "PatientID":    str(ds.get("PatientID", "Unknown")),
                "StudyDate":    str(ds.get("StudyDate", "Unknown")),
                "Modality":     str(ds.get("Modality", "Unknown")),
                "Manufacturer": str(ds.get("Manufacturer", "Unknown")),
                "Rows": ds.get("Rows"), "Columns": ds.get("Columns"),
            }
            response_data = {"metadata": metadata}
            if hasattr(ds, "pixel_array"):
                image_array = _decode_image(content, fname)
                _, buffer = cv2.imencode('.png', image_array)
                response_data["image_data"] = f"data:image/png;base64,{base64.b64encode(buffer).decode()}"
            else:
                response_data["image_data"] = None
        else:
            nparr = np.frombuffer(content, np.uint8)
            image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image_array is None:
                raise HTTPException(status_code=400, detail="Failed to decode image file.")
            h, w = image_array.shape[:2]
            metadata = {"PatientName": "N/A", "PatientID": "N/A", "StudyDate": "N/A",
                        "Modality": "CR (estimated)", "Manufacturer": "N/A", "Rows": h, "Columns": w}
            _, buffer = cv2.imencode('.png', image_array)
            response_data = {
                "metadata": metadata,
                "image_data": f"data:image/png;base64,{base64.b64encode(buffer).decode()}"
            }
        return JSONResponse(content=response_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.post("/api/analyze")
async def analyze_elbow(file: UploadFile = File(...)):
    """
    肘X線画像を解析。YOLOv8-Pose（プライマリ）→ Classical CV（フォールバック）。
    ConvNeXtが利用可能な場合はセカンドオピニオン角度も同時に返す。
    """
    try:
        content = await file.read()
        fname = (file.filename or "").lower()
        image_array = _decode_image(content, fname)

        landmarks = detect_with_yolo_pose(image_array)
        if landmarks is None:
            landmarks = detect_bone_landmarks_classical(image_array)

        # ConvNeXt セカンドオピニオン
        second_opinion = None
        if TORCH_INSTALLED and convnext_model is not None:
            try:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(image_rgb)
                img_tensor = convnext_transforms(pil_img).to(device)
                with torch.no_grad():
                    pred = convnext_model(img_tensor.unsqueeze(0))[0].cpu().numpy()
                view = landmarks["qa"]["view_type"]
                second_opinion = {
                    "carrying_angle": round(float(pred[0]), 1) if view == "AP"  else None,
                    "flexion":        round(float(pred[1]), 1) if view == "LAT" else None,
                    "pronation_sup":  round(float(pred[2]), 1),
                    "varus_valgus":   round(float(pred[3]), 1),
                    "model": "ConvNeXt-Small",
                }
            except Exception as e:
                print(f"ConvNeXt inference failed: {e}")

        return JSONResponse(content={
            "success": True,
            "landmarks": landmarks,
            "second_opinion": second_opinion,
            "image_size": {"width": image_array.shape[1], "height": image_array.shape[0]},
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/gradcam")
async def gradcam_endpoint(file: UploadFile = File(...), target: str = "all"):
    """
    Grad-CAM XAI エンドポイント（ConvNeXt-Small）。
    target: "all" | "carrying" | "flexion" | "pronation" | "varus_valgus"
    """
    if gradcam_engine is None or convnext_model is None:
        return JSONResponse(content={
            "success": False,
            "error": "ConvNeXt XAIエンジン未ロード。elbow_convnext_best.pth が必要です。",
            "engine_used": "unavailable"
        }, status_code=503)

    try:
        content = await file.read()
        fname = (file.filename or "").lower()
        image_bgr = _decode_image(content, fname)
        h, w = image_bgr.shape[:2]

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        img_tensor = convnext_transforms(pil_img).to(device)

        target_map = {"all": None, "carrying": 0, "flexion": 1, "pronation": 2, "varus_valgus": 3}
        target_idx = target_map.get(target.lower())

        cam = gradcam_engine.generate(img_tensor, target_idx=target_idx)

        convnext_model.eval()
        with torch.no_grad():
            pred = convnext_model(img_tensor.unsqueeze(0))[0].cpu().numpy()

        predicted_angles = {
            "外反角":  round(float(pred[0]), 1),
            "屈曲角":  round(float(pred[1]), 1),
            "回内外":  round(float(pred[2]), 1),
            "内反外反": round(float(pred[3]), 1),
        }

        overlay = apply_gradcam_overlay(image_bgr, cam, alpha=0.45)
        cam_resized = cv2.resize(cam, (w, h))
        raw_heatmap_bgr = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

        _, buf_overlay = cv2.imencode(".png", overlay)
        _, buf_heatmap = cv2.imencode(".png", raw_heatmap_bgr)

        return JSONResponse(content={
            "success": True,
            "engine_used": "gradcam_convnext_small",
            "target": target,
            "predicted_angles": predicted_angles,
            "heatmap_overlay": f"data:image/png;base64,{base64.b64encode(buf_overlay).decode()}",
            "raw_heatmap":     f"data:image/png;base64,{base64.b64encode(buf_heatmap).decode()}",
            "image_size": {"width": w, "height": h},
            "note": "赤＝高注目領域（AIが角度判断に使った箇所）、青＝低注目領域"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
