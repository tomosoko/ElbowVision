"""
ElbowVision 推論エンジン
- モデルロード (YOLO, ConvNeXt)
- 推論関数 (YOLO pose, ConvNeXt regression, Classical CV fallback)
- GradCAM
- 角度計算ヘルパー
- 画像デコード
"""
import os
import math
import time
from typing import Optional

import cv2
import numpy as np
import pydicom
import pydicom.config
import io

from med_image_pipeline import apply_windowing, apply_clahe_to_gray, apply_gaussian_blur

# DICOMメタデータの欠損に対して寛容に処理
pydicom.config.enforce_valid_values = False

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

_API_DIR = os.path.dirname(os.path.abspath(__file__))
_YOLO_CANDIDATE_PATHS = [
    os.path.join(_API_DIR, "models", "yolo_pose_best.pt"),
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
CONVNEXT_MODEL_PATH = os.path.join(_API_DIR, "elbow_convnext_best.pth")

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


# ─── 共通ユーティリティ ─────────────────────────────────────────────────────────
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

    enhanced = apply_clahe_to_gray(gray, clip_limit=2.0, tile_grid_size=(8, 8))
    blurred = apply_gaussian_blur(enhanced, kernel_size=(3, 3))
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
    キーポイント順:
      必須4点（現行モデル）:
        0: humerus_shaft      — 上腕骨幹部（近位）
        1: lateral_epicondyle — 外側上顆
        2: medial_epicondyle  — 内側上顆
        3: forearm_shaft      — 前腕骨幹部（遠位）
      オプション2点（6KPモデル）:
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
        n_kpts = len(kpts)
        confs = result.keypoints.conf[0].cpu().numpy() if result.keypoints.conf is not None else np.ones(n_kpts)

        if n_kpts < 4:
            return None

        humerus_pt   = {"x": float(kpts[0][0]), "y": float(kpts[0][1])}
        lat_epic_pt  = {"x": float(kpts[1][0]), "y": float(kpts[1][1])}
        med_epic_pt  = {"x": float(kpts[2][0]), "y": float(kpts[2][1])}
        forearm_pt   = {"x": float(kpts[3][0]), "y": float(kpts[3][1])}

        # オプションKP（6KPモデルの場合のみ）
        radial_pt    = {"x": float(kpts[4][0]), "y": float(kpts[4][1])} if n_kpts >= 5 else None
        olecranon_pt = {"x": float(kpts[5][0]), "y": float(kpts[5][1])} if n_kpts >= 6 else None

        condyle_mid = {
            "x": (lat_epic_pt["x"] + med_epic_pt["x"]) / 2,
            "y": (lat_epic_pt["y"] + med_epic_pt["y"]) / 2,
        }

        # AP/LAT判定:
        # 主指標: 前腕軸の垂直からの傾き
        #   伸展AP像(150-180°): 前腕はほぼ垂直 (|dx/dy| < tan35° ≈ 0.70)
        #   屈曲LAT像(60-120°): 前腕は水平寄り (|dx/dy| > 0.70)
        # 上顆間距離は v6 LAT(AP投影)では AP像と区別できないため補助指標のみ
        fa_dx = forearm_pt["x"] - condyle_mid["x"]
        fa_dy = forearm_pt["y"] - condyle_mid["y"]
        forearm_oblique = abs(fa_dx) > abs(fa_dy) * 0.70  # 前腕が垂直から35°以上傾いている
        epic_sep = math.sqrt((lat_epic_pt["x"] - med_epic_pt["x"])**2 + (lat_epic_pt["y"] - med_epic_pt["y"])**2)
        if forearm_oblique:
            view_type = "LAT"  # 前腕が水平寄り → 屈曲LAT像
        elif olecranon_pt is not None:
            olecranon_posterior = olecranon_pt["y"] > condyle_mid["y"] + h * 0.03
            view_type = "AP" if (epic_sep > w * 0.06 and not olecranon_posterior) else "LAT"
        else:
            view_type = "AP" if epic_sep > w * 0.06 else "LAT"

        humerus_axis_angle = angle_deg(humerus_pt, condyle_mid)
        forearm_axis_angle = angle_deg(condyle_mid, forearm_pt)
        joint_angle = round(full_angle(humerus_axis_angle, forearm_axis_angle), 1)

        if view_type == "AP":
            carrying_angle, flexion = joint_angle, None
        else:
            if olecranon_pt is not None and radial_pt is not None:
                # 6KPモデル: olecranon-condyle-radial_head の三点角度で屈曲角
                ol_angle = angle_deg(olecranon_pt, condyle_mid)
                rh_angle = angle_deg(condyle_mid, radial_pt)
                flexion  = round(full_angle(ol_angle, rh_angle), 1)
            else:
                # 4KPモデル: humerus-condyle-forearm から屈曲角を推定
                flexion = joint_angle
            carrying_angle = None

        # 前腕回旋（radial_headがある場合のみ）
        if radial_pt is not None:
            radial_offset = radial_pt["x"] - condyle_mid["x"]
            pronation_sup = round(max(-30.0, min(30.0, -radial_offset / max(w * 0.01, 1e-3))), 1)
        else:
            pronation_sup = 0.0
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
            positioning_advice = f"► {direction}が検出されました。前腕を「{correction}」させてください。"
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
            **({"radial_head": {"x": int(radial_pt["x"]), "y": int(radial_pt["y"]), "x_pct": pct(radial_pt["x"], w), "y_pct": pct(radial_pt["y"], h)}} if radial_pt else {}),
            **({"olecranon": {"x": int(olecranon_pt["x"]), "y": int(olecranon_pt["y"]), "x_pct": pct(olecranon_pt["x"], w), "y_pct": pct(olecranon_pt["y"], h)}} if olecranon_pt else {}),
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
    enhanced = apply_clahe_to_gray(gray, clip_limit=3.0, tile_grid_size=(8, 8))
    blurred = apply_gaussian_blur(enhanced, kernel_size=(5, 5))

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
            "inference_engine": "Classical CV",
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

    # ConvNeXt セカンドオピニオン — estimate_positioning_correctionより先に実行
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

    # ConvNeXt屈曲角でlandmarks.angles.flexionを上書き（LAT像のみ）
    # estimate_positioning_correctionより前に適用しないとflexion_adviceが不正確になる
    if (second_opinion is not None
            and second_opinion.get("flexion_deg") is not None):
        landmarks["angles"]["flexion"] = second_opinion["flexion_deg"]

    # ポジショニング補正推定 — ConvNeXt上書き後に実行
    positioning_correction = estimate_positioning_correction(image_array, landmarks)

    _record_stats(landmarks)

    return {
        "success": True,
        "landmarks": landmarks,
        "edge_validation": edge_validation,
        "positioning_correction": positioning_correction,
        "second_opinion": second_opinion,
        "image_size": {"width": image_array.shape[1], "height": image_array.shape[0]},
    }
