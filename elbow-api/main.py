import os
import math
import time
import zipfile
import tempfile
import uvicorn
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
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

app = FastAPI(title="ElbowVision API", version="2.0.0")

# ─── インメモリ推論統計 ──────────────────────────────────────────────────────
_inference_stats = {
    "total_inferences": 0,
    "carrying_angles": [],
    "flexion_angles": [],
    "qa_scores": [],
    "engine_counts": {"yolo_pose": 0, "classical_cv": 0},
    "started_at": time.time(),
}

# ─── YOLOv8 Pose Model（プライマリ） ──────────────────────────────────────────

try:
    from ultralytics import YOLO
    YOLO_INSTALLED = True
except ImportError:
    YOLO_INSTALLED = False

_YOLO_CANDIDATE_PATHS = [
    "models/yolo_pose_best.pt",
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

# ─── ConvNeXt ポジショニングズレ回帰モデル + Grad-CAM（セカンドオピニオン） ──

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from PIL import Image
    # 訓練スクリプトと同一モデル定義を共有（Single Source of Truth）
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "training"))
    from convnext_model import ElbowConvNeXt
    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False

# 出力: [rotation_error_deg, flexion_deg]
CONVNEXT_MODEL_PATH = "elbow_convnext_best.pth"

convnext_model = None
device = None

if TORCH_INSTALLED:
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    if os.path.exists(CONVNEXT_MODEL_PATH):
        try:
            convnext_model = ElbowConvNeXt(pretrained=False)
            convnext_model.load_state_dict(torch.load(CONVNEXT_MODEL_PATH, map_location=device))
            convnext_model.to(device)
            convnext_model.eval()
            print(f"Loaded ConvNeXt Positioning Regressor on {device}.")
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
        target_idx: None=全出力の和, 0=rotation_error_deg(AP), 1=flexion_deg(LAT)
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
        "version": "2.0.0",
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


# ─── ポジショニング補正推定 ────────────────────────────────────────────────────
def estimate_positioning_correction(image_array: np.ndarray, landmarks: dict) -> dict:
    """
    外顆間距離（体格指標）と屈曲角から、理想ポジショニングへの補正量を推定する。
    放射線技師が実際に行う操作（発泡スチロールで高さ調整・前腕回旋等）に合わせたアドバイスを生成。

    AP像: 外顆間距離が大きい = 良好な回外位
    LAT像: 外顆間距離≈0 かつ 屈曲90° = 理想側面位

    戻り値:
      view_type              : AP / LAT
      epic_separation_px     : 内外側上顆間の距離（体格指標）
      epic_ratio             : 外顆間距離 / 画像対角線
      rotation_error         : 理想位からの回旋ズレ量（°）
      rotation_level         : "good" / "minor" / "major"
      rotation_advice        : 回旋補正の実践的アドバイス
      flexion_deg            : 検出された屈曲角（LAT像のみ）
      flexion_level          : "good" / "minor" / "major"（LAT像のみ）
      flexion_advice         : 屈曲角補正の実践的アドバイス（LAT像のみ）
      overall_level          : 総合判定（rotation/flexionの悪い方）
      correction_needed      : 補正が必要かどうか
    """
    h, w = image_array.shape[:2]
    diagonal = math.sqrt(h ** 2 + w ** 2)

    lat_epic  = landmarks["lateral_epicondyle"]
    med_epic  = landmarks["medial_epicondyle"]
    view_type = landmarks["qa"]["view_type"]
    angles    = landmarks["angles"]

    epic_sep   = math.sqrt((lat_epic["x"] - med_epic["x"]) ** 2 + (lat_epic["y"] - med_epic["y"]) ** 2)
    epic_ratio = epic_sep / max(diagonal, 1.0)

    # ── 回旋ズレ評価 ──────────────────────────────────────────────────────────
    if view_type == "AP":
        # AP像: 外顆間距離 大 = 回外位良好
        if epic_ratio >= 0.10:
            rotation_error, rotation_level = 0.0, "good"
            rotation_advice = "回外位が適切です。手のひらが上を向いた状態を維持してください。"
        elif epic_ratio >= 0.05:
            deficit        = (0.10 - epic_ratio) / 0.05
            rotation_error = round(deficit * 25.0, 1)
            rotation_level = "minor"
            rotation_advice = (f"前腕をもう少し回外してください（手のひらをさらに上に向ける）。"
                               f"推定ズレ: 約{rotation_error:.0f}°。")
        else:
            rotation_error = round(min((0.10 - epic_ratio) / 0.10 * 45.0, 45.0), 1)
            rotation_level = "major"
            rotation_advice = (f"前腕が大きく回内しています。手のひらが完全に上を向くよう"
                               f"腕全体を回外してください（推定ズレ: 約{rotation_error:.0f}°）。再撮影推奨。")
    else:  # LAT
        # LAT像: 外顆間距離≈0 = 上顆が重なる = 良好
        if epic_ratio < 0.04:
            rotation_error, rotation_level = 0.0, "good"
            rotation_advice = "内外側上顆が重なっています。回旋位置は適切です。"
        elif epic_ratio < 0.10:
            rotation_error = round(epic_ratio / 0.10 * 20.0, 1)
            rotation_level = "minor"
            rotation_advice = (f"上顆が少しズレています（推定{rotation_error:.0f}°）。"
                               f"前腕を内側に少し回旋させるか、肘の位置を微調整してください。")
        else:
            rotation_error = round(min(epic_ratio / 0.10 * 20.0, 45.0), 1)
            rotation_level = "major"
            rotation_advice = (f"上顆が大きくズレています（推定{rotation_error:.0f}°）。"
                               f"肘の下に発泡スチロール等を置いて高さを調整し、"
                               f"前腕を内側に回旋させて上顆が重なるようにしてください。再撮影推奨。")

    # ── 屈曲角評価（LAT像のみ） ───────────────────────────────────────────────
    flexion_deg    = angles.get("flexion")
    flexion_level  = None
    flexion_advice = None

    if view_type == "LAT" and flexion_deg is not None:
        if 80.0 <= flexion_deg <= 100.0:
            flexion_level  = "good"
            flexion_advice = f"屈曲角良好（{flexion_deg:.1f}°）。90°に近い適切な位置です。"
        elif flexion_deg < 80.0:
            dev = round(90.0 - flexion_deg, 1)
            flexion_level  = "major" if flexion_deg < 70.0 else "minor"
            flexion_advice = (f"肘の屈曲が浅すぎます（{flexion_deg:.1f}°）。"
                              f"肘の下に台（発泡スチロール等）を置いて高さを出し、"
                              f"肘をもう約{dev:.0f}°曲げてください。")
        else:
            dev = round(flexion_deg - 90.0, 1)
            flexion_level  = "major" if flexion_deg > 110.0 else "minor"
            flexion_advice = (f"肘の屈曲が深すぎます（{flexion_deg:.1f}°）。"
                              f"前腕を少し下げて約{dev:.0f}°伸ばし、90°に近づけてください。")

    # ── 総合判定 ──────────────────────────────────────────────────────────────
    level_order = {"good": 0, "minor": 1, "major": 2, None: 0}
    overall_level = max(rotation_level, flexion_level or "good", key=lambda x: level_order[x])
    correction_needed = overall_level != "good"

    return {
        "view_type":        view_type,
        "epic_separation_px": round(epic_sep, 1),
        "epic_ratio":       round(epic_ratio, 4),
        "rotation_error":   rotation_error,
        "rotation_level":   rotation_level,
        "rotation_advice":  rotation_advice,
        "flexion_deg":      round(flexion_deg, 1) if flexion_deg is not None else None,
        "flexion_level":    flexion_level,
        "flexion_advice":   flexion_advice,
        "overall_level":    overall_level,
        "correction_needed": correction_needed,
    }


# ─── エッジバリデーション ──────────────────────────────────────────────────────
def validate_angle_with_edges(image_array: np.ndarray, primary_angle: float) -> dict:
    """
    Canny + Hough変換で骨幹の軸線を検出し、プライマリ角度（YOLOv8 or Classical CV）を検証する。

    手順:
      1. CLAHE でコントラスト強調 → Canny エッジ検出
      2. 上半分（上腕骨域）と下半分（前腕骨域）に分割
      3. 各領域で HoughLinesP → 長さ重み付き平均角度を算出
      4. 上腕骨軸と前腕骨軸から関節角度を計算
      5. プライマリ角度との差分でconfidenceを判定

    戻り値:
      edge_angle    : エッジから算出した関節角度（検出失敗時は None）
      agreement_deg : プライマリ角度との差（小さいほど良い）
      confidence    : "high" / "medium" / "low"
      edge_lines    : 検出した直線数
      note          : 判定コメント
    """
    h, w = image_array.shape[:2]

    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) == 3 else image_array.copy()
    gray = gray.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(blurred, otsu_thresh * 0.4, otsu_thresh)

    def dominant_angle(region: np.ndarray) -> Optional[float]:
        rh, rw = region.shape[:2]
        min_len = max(rh, rw) * 0.25
        lines = cv2.HoughLinesP(region, 1, np.pi / 180, threshold=30,
                                minLineLength=int(min_len), maxLineGap=20)
        if lines is None:
            return None
        angles, weights = [], []
        for x1, y1, x2, y2 in lines[:, 0]:
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(ang) < 15 or abs(ang) > 165:  # 水平に近い線（骨幹と垂直）を除外
                continue
            angles.append(ang)
            weights.append(length)
        if not angles:
            return None
        return float(np.average(angles, weights=np.array(weights)))

    split = int(h * 0.5)
    upper_edge = edges[:split, :]
    lower_edge = edges[split:, :]

    upper_ang = dominant_angle(upper_edge)
    lower_ang = dominant_angle(lower_edge)

    def count_lines(region):
        min_len = int(region.shape[0] * 0.25)
        result = cv2.HoughLinesP(region, 1, np.pi/180, 30, minLineLength=min_len, maxLineGap=20)
        return len(result) if result is not None else 0

    n_lines = count_lines(upper_edge) + count_lines(lower_edge)

    if upper_ang is None or lower_ang is None:
        return {
            "edge_angle": None, "agreement_deg": None,
            "confidence": "low", "edge_lines": n_lines,
            "note": "骨幹ラインの検出に失敗（エッジ不明瞭）",
        }

    edge_angle = round(full_angle(upper_ang, lower_ang), 1)
    agreement  = round(abs(edge_angle - primary_angle), 1)

    if agreement <= 3.0:
        confidence, note = "high",   f"エッジ一致 ✓ 差{agreement:.1f}° — 角度信頼性が高い"
    elif agreement <= 8.0:
        confidence, note = "medium", f"エッジ軽度不一致 差{agreement:.1f}° — ポジショニング微調整を推奨"
    else:
        confidence, note = "low",    f"エッジ不一致 差{agreement:.1f}° — 再撮影またはキーポイント確認を推奨"

    return {
        "edge_angle":    edge_angle,
        "agreement_deg": agreement,
        "confidence":    confidence,
        "edge_lines":    n_lines,
        "note":          note,
    }


# ─── YOLOv8-Pose 推論（プライマリ） ───────────────────────────────────────────
def detect_with_yolo_pose(image_array: np.ndarray) -> Optional[dict]:
    """
    キーポイント順（6点）:
      0: humerus_shaft      — 上腕骨幹部（近位）
      1: lateral_epicondyle — 外側上顆
      2: medial_epicondyle  — 内側上顆
      3: forearm_shaft      — 前腕骨幹部（遠位）
      4: radial_head        — 橈骨頭（前腕回旋の直接指標）
      5: olecranon          — 肘頭（LAT屈曲角・AP/LAT判定に使用）
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
        confs = result.keypoints.conf[0].cpu().numpy() if result.keypoints.conf is not None else np.ones(6)

        if len(kpts) < 6:
            return None

        humerus_pt   = {"x": float(kpts[0][0]), "y": float(kpts[0][1])}
        lat_epic_pt  = {"x": float(kpts[1][0]), "y": float(kpts[1][1])}
        med_epic_pt  = {"x": float(kpts[2][0]), "y": float(kpts[2][1])}
        forearm_pt   = {"x": float(kpts[3][0]), "y": float(kpts[3][1])}
        radial_pt    = {"x": float(kpts[4][0]), "y": float(kpts[4][1])}
        olecranon_pt = {"x": float(kpts[5][0]), "y": float(kpts[5][1])}

        condyle_mid = {
            "x": (lat_epic_pt["x"] + med_epic_pt["x"]) / 2,
            "y": (lat_epic_pt["y"] + med_epic_pt["y"]) / 2,
        }

        # AP/LAT判定: 上顆間距離 + 肘頭位置（後方突出）の2指標で判定
        epic_sep = math.sqrt((lat_epic_pt["x"] - med_epic_pt["x"])**2 + (lat_epic_pt["y"] - med_epic_pt["y"])**2)
        olecranon_posterior = olecranon_pt["y"] > condyle_mid["y"] + h * 0.03
        view_type = "AP" if (epic_sep > w * 0.06 and not olecranon_posterior) else "LAT"

        humerus_axis_angle = angle_deg(humerus_pt, condyle_mid)
        forearm_axis_angle = angle_deg(condyle_mid, forearm_pt)
        joint_angle = round(full_angle(humerus_axis_angle, forearm_axis_angle), 1)

        if view_type == "AP":
            carrying_angle, flexion = joint_angle, None
        else:
            # LAT像: olecranon-condyle-radial_head の三点角度で屈曲角を計算
            ol_angle = angle_deg(olecranon_pt, condyle_mid)
            rh_angle = angle_deg(condyle_mid, radial_pt)
            flexion  = round(full_angle(ol_angle, rh_angle), 1)
            carrying_angle = None

        # 前腕回旋: radial_head の condyle_mid からのML方向ズレで定量化
        radial_offset = radial_pt["x"] - condyle_mid["x"]
        pronation_sup = round(max(-30.0, min(30.0, -radial_offset / max(w * 0.01, 1e-3))), 1)
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
            "radial_head":        {"x": int(radial_pt["x"]),     "y": int(radial_pt["y"]),     "x_pct": pct(radial_pt["x"], w),     "y_pct": pct(radial_pt["y"], h)},
            "olecranon":          {"x": int(olecranon_pt["x"]),  "y": int(olecranon_pt["y"]),  "x_pct": pct(olecranon_pt["x"], w),  "y_pct": pct(olecranon_pt["y"], h)},
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

        # エッジバリデーション（AP/LATで有効な角度のみ検証）
        angles = landmarks["angles"]
        primary_angle = angles["carrying_angle"] if angles["carrying_angle"] is not None else angles["flexion"]
        edge_validation = None
        if primary_angle is not None:
            edge_validation = validate_angle_with_edges(image_array, primary_angle)

        # ポジショニング補正推定（外顆間距離 × 体格）
        positioning_correction = estimate_positioning_correction(image_array, landmarks)

        # ConvNeXt セカンドオピニオン（ポジショニングズレ量推定）
        second_opinion = None
        if TORCH_INSTALLED and convnext_model is not None:
            try:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(image_rgb)
                img_tensor = convnext_transforms(pil_img).to(device)
                with torch.no_grad():
                    pred = convnext_model(img_tensor.unsqueeze(0))[0].cpu().numpy()
                view = landmarks["qa"]["view_type"]
                # pred[0]=rotation_error_deg, pred[1]=flexion_deg
                second_opinion = {
                    "rotation_error_deg": round(float(pred[0]), 1) if view == "AP"  else None,
                    "flexion_deg":        round(float(pred[1]), 1) if view == "LAT" else None,
                    "model": "ConvNeXt-Small",
                }
            except Exception as e:
                print(f"ConvNeXt inference failed: {e}")

        # 推論統計の記録
        _record_stats(landmarks)

        return JSONResponse(content={
            "success": True,
            "landmarks": landmarks,
            "edge_validation": edge_validation,
            "positioning_correction": positioning_correction,
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
    target: "all" | "rotation"（回旋ズレ AP）| "flexion"（屈曲角 LAT）
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

        target_map = {"all": None, "rotation": 0, "flexion": 1}
        target_idx = target_map.get(target.lower())

        cam = gradcam_engine.generate(img_tensor, target_idx=target_idx)

        convnext_model.eval()
        with torch.no_grad():
            pred = convnext_model(img_tensor.unsqueeze(0))[0].cpu().numpy()

        predicted_angles = {
            "回旋ズレ(AP)": round(float(pred[0]), 1),
            "屈曲角(LAT)":  round(float(pred[1]), 1),
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


# ─── 統計記録ヘルパー ─────────────────────────────────────────────────────────
def _record_stats(landmarks: dict):
    _inference_stats["total_inferences"] += 1
    angles = landmarks.get("angles", {})
    qa = landmarks.get("qa", {})
    engine = qa.get("inference_engine", "classical_cv")
    if "YOLOv8" in str(engine):
        _inference_stats["engine_counts"]["yolo_pose"] += 1
    else:
        _inference_stats["engine_counts"]["classical_cv"] += 1
    if angles.get("carrying_angle") is not None:
        _inference_stats["carrying_angles"].append(angles["carrying_angle"])
    if angles.get("flexion") is not None:
        _inference_stats["flexion_angles"].append(angles["flexion"])
    if qa.get("score") is not None:
        _inference_stats["qa_scores"].append(qa["score"])


# ─── 単一画像の解析処理（内部共通） ──────────────────────────────────────────
def _analyze_single_image(image_array: np.ndarray) -> dict:
    """analyze_elbowとbatch_analyzeで共有する解析ロジック"""
    landmarks = detect_with_yolo_pose(image_array)
    if landmarks is None:
        landmarks = detect_bone_landmarks_classical(image_array)

    angles = landmarks["angles"]
    primary_angle = angles["carrying_angle"] if angles["carrying_angle"] is not None else angles["flexion"]
    edge_validation = None
    if primary_angle is not None:
        edge_validation = validate_angle_with_edges(image_array, primary_angle)

    positioning_correction = estimate_positioning_correction(image_array, landmarks)

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
                "rotation_error_deg": round(float(pred[0]), 1) if view == "AP" else None,
                "flexion_deg": round(float(pred[1]), 1) if view == "LAT" else None,
                "model": "ConvNeXt-Small",
            }
        except Exception as e:
            print(f"ConvNeXt inference failed: {e}")

    _record_stats(landmarks)

    return {
        "success": True,
        "landmarks": landmarks,
        "edge_validation": edge_validation,
        "positioning_correction": positioning_correction,
        "second_opinion": second_opinion,
        "image_size": {"width": image_array.shape[1], "height": image_array.shape[0]},
    }


# ─── POST /api/batch-analyze ─────────────────────────────────────────────────
@app.post("/api/batch-analyze")
async def batch_analyze(file: UploadFile = File(...)):
    """
    複数画像の一括推論。ZIPファイルまたは単一画像をアップロード。
    ZIPの場合は中のPNG/JPG/DICOM画像を全て解析し、結果をJSON配列で返す。
    """
    try:
        content = await file.read()
        fname = (file.filename or "").lower()

        results: List[dict] = []
        allowed_img_ext = ('.png', '.jpg', '.jpeg', '.dcm', '.dicom')

        if fname.endswith('.zip'):
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "upload.zip")
                with open(zip_path, "wb") as f:
                    f.write(content)
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        names = sorted([
                            n for n in zf.namelist()
                            if n.lower().endswith(allowed_img_ext)
                            and not n.startswith('__MACOSX')
                        ])
                        if not names:
                            raise HTTPException(
                                status_code=400,
                                detail="ZIP内に対応画像ファイル(PNG/JPG/DICOM)が見つかりません。"
                            )
                        for img_name in names:
                            try:
                                img_bytes = zf.read(img_name)
                                image_array = _decode_image(img_bytes, img_name)
                                result = _analyze_single_image(image_array)
                                result["filename"] = os.path.basename(img_name)
                                results.append(result)
                            except Exception as e:
                                results.append({
                                    "filename": os.path.basename(img_name),
                                    "success": False,
                                    "error": str(e),
                                })
                except zipfile.BadZipFile:
                    raise HTTPException(status_code=400, detail="不正なZIPファイルです。")
        else:
            # 単一画像
            image_array = _decode_image(content, fname)
            result = _analyze_single_image(image_array)
            result["filename"] = file.filename or "unknown"
            results.append(result)

        # サマリー
        successful = [r for r in results if r.get("success")]
        carrying_angles = [
            r["landmarks"]["angles"]["carrying_angle"]
            for r in successful
            if r.get("landmarks", {}).get("angles", {}).get("carrying_angle") is not None
        ]
        flexion_angles = [
            r["landmarks"]["angles"]["flexion"]
            for r in successful
            if r.get("landmarks", {}).get("angles", {}).get("flexion") is not None
        ]
        qa_scores = [
            r["landmarks"]["qa"]["score"]
            for r in successful
            if r.get("landmarks", {}).get("qa", {}).get("score") is not None
        ]

        summary = {
            "total_images": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "avg_carrying_angle": round(float(np.mean(carrying_angles)), 1) if carrying_angles else None,
            "avg_flexion_angle": round(float(np.mean(flexion_angles)), 1) if flexion_angles else None,
            "avg_qa_score": round(float(np.mean(qa_scores)), 1) if qa_scores else None,
        }

        return JSONResponse(content={"results": results, "summary": summary})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── GET /api/model-info ─────────────────────────────────────────────────────
@app.get("/api/model-info")
async def model_info():
    """使用中のモデル情報を返す"""
    info = {
        "primary_engine": "YOLOv8-Pose" if yolo_model is not None else "Classical CV (fallback)",
        "yolo": {
            "loaded": yolo_model is not None,
            "model_path": YOLO_MODEL_PATH if yolo_model is not None else None,
            "version": "YOLOv8 (ultralytics)",
            "task": "pose",
            "keypoints": 6,
            "keypoint_names": [
                "humerus_shaft", "lateral_epicondyle", "medial_epicondyle",
                "forearm_shaft", "radial_head", "olecranon"
            ],
        },
        "convnext": {
            "loaded": convnext_model is not None,
            "model_path": CONVNEXT_MODEL_PATH if convnext_model is not None else None,
            "version": "ConvNeXt-Small",
            "outputs": ["rotation_error_deg", "flexion_deg"],
            "device": str(device) if device is not None else None,
        },
        "gradcam": {
            "available": gradcam_engine is not None,
        },
        "classical_cv": {
            "available": True,
            "note": "YOLOv8未ロード時のフォールバック。CLAHE + Otsu + ConnectedComponents。",
        },
        "fallback_active": yolo_model is None,
    }
    return JSONResponse(content=info)


# ─── POST /api/compare ───────────────────────────────────────────────────────
@app.post("/api/compare")
async def compare_images(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
):
    """
    2枚の画像を比較。角度差分・QA改善度を計算。
    ポジショニング改善前後の比較に使用。
    """
    try:
        content1 = await file1.read()
        content2 = await file2.read()

        img1 = _decode_image(content1, file1.filename or "image1.png")
        img2 = _decode_image(content2, file2.filename or "image2.png")

        result1 = _analyze_single_image(img1)
        result2 = _analyze_single_image(img2)

        angles1 = result1["landmarks"]["angles"]
        angles2 = result2["landmarks"]["angles"]
        qa1 = result1["landmarks"]["qa"]["score"]
        qa2 = result2["landmarks"]["qa"]["score"]

        # 角度差分
        carrying_diff = None
        if angles1.get("carrying_angle") is not None and angles2.get("carrying_angle") is not None:
            carrying_diff = round(angles2["carrying_angle"] - angles1["carrying_angle"], 1)

        flexion_diff = None
        if angles1.get("flexion") is not None and angles2.get("flexion") is not None:
            flexion_diff = round(angles2["flexion"] - angles1["flexion"], 1)

        qa_diff = round(qa2 - qa1, 1)

        # ポジショニング改善判定
        pc1 = result1.get("positioning_correction", {})
        pc2 = result2.get("positioning_correction", {})
        rot_err1 = pc1.get("rotation_error", 0)
        rot_err2 = pc2.get("rotation_error", 0)
        rotation_improvement = round(rot_err1 - rot_err2, 1)

        if qa_diff > 10:
            improvement_label = "大幅改善"
        elif qa_diff > 0:
            improvement_label = "改善"
        elif qa_diff == 0:
            improvement_label = "変化なし"
        elif qa_diff > -10:
            improvement_label = "やや悪化"
        else:
            improvement_label = "悪化"

        return JSONResponse(content={
            "image1": {
                "filename": file1.filename,
                "carrying_angle": angles1.get("carrying_angle"),
                "flexion": angles1.get("flexion"),
                "qa_score": qa1,
                "positioning_level": pc1.get("overall_level"),
                "rotation_error": rot_err1,
            },
            "image2": {
                "filename": file2.filename,
                "carrying_angle": angles2.get("carrying_angle"),
                "flexion": angles2.get("flexion"),
                "qa_score": qa2,
                "positioning_level": pc2.get("overall_level"),
                "rotation_error": rot_err2,
            },
            "comparison": {
                "carrying_angle_diff": carrying_diff,
                "flexion_diff": flexion_diff,
                "qa_score_diff": qa_diff,
                "rotation_improvement_deg": rotation_improvement,
                "improvement_label": improvement_label,
            },
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── GET /api/stats ──────────────────────────────────────────────────────────
@app.get("/api/stats")
async def inference_stats():
    """これまでの推論統計を返す"""
    ca = _inference_stats["carrying_angles"]
    fa = _inference_stats["flexion_angles"]
    qa = _inference_stats["qa_scores"]
    uptime = time.time() - _inference_stats["started_at"]

    # QAスコア分布
    qa_distribution = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
    for s in qa:
        if s >= 90:
            qa_distribution["excellent"] += 1
        elif s >= 75:
            qa_distribution["good"] += 1
        elif s >= 50:
            qa_distribution["fair"] += 1
        else:
            qa_distribution["poor"] += 1

    return JSONResponse(content={
        "total_inferences": _inference_stats["total_inferences"],
        "uptime_seconds": round(uptime, 1),
        "engine_counts": _inference_stats["engine_counts"],
        "carrying_angle": {
            "count": len(ca),
            "mean": round(float(np.mean(ca)), 1) if ca else None,
            "std": round(float(np.std(ca)), 1) if ca else None,
            "min": round(float(min(ca)), 1) if ca else None,
            "max": round(float(max(ca)), 1) if ca else None,
        },
        "flexion_angle": {
            "count": len(fa),
            "mean": round(float(np.mean(fa)), 1) if fa else None,
            "std": round(float(np.std(fa)), 1) if fa else None,
            "min": round(float(min(fa)), 1) if fa else None,
            "max": round(float(max(fa)), 1) if fa else None,
        },
        "qa_score": {
            "count": len(qa),
            "mean": round(float(np.mean(qa)), 1) if qa else None,
            "distribution": qa_distribution,
        },
    })


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
