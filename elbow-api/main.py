import os
import time
import zipfile
import tempfile
import uvicorn
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pydicom
import io
import cv2
import numpy as np
import base64

# ─── 推論エンジンからインポート ──────────────────────────────────────────────
from inference import (
    # モデル・エンジン状態
    yolo_model, convnext_model, gradcam_engine, device,
    YOLO_MODEL_PATH, CONVNEXT_MODEL_PATH,
    TORCH_INSTALLED,
    _inference_stats,
    # 推論関数
    detect_with_yolo_pose,
    detect_bone_landmarks_classical,
    validate_angle_with_edges,
    estimate_positioning_correction,
    _decode_image,
    _record_stats,
    _analyze_single_image,
    # GradCAM
    GradCAM,
    apply_gradcam_overlay,
    # ヘルパー
    pct, angle_deg, full_angle,
    # med_image_pipeline re-export (テスト互換)
    apply_windowing,
)

# ConvNeXt関連（条件付きインポート）
if TORCH_INSTALLED:
    from inference import convnext_transforms, torch, Image

app = FastAPI(title="ElbowVision API", version="2.0.0")

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
        "version": "2.1.0-v6",
        "engines": {
            "yolo_pose":    yolo_model is not None,
            "convnext_xai": convnext_model is not None,
            "gradcam_xai":  gradcam_engine is not None,
            "classical_cv": True,
        },
        "message": "ElbowVision AI API is running."
    }


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

        # ConvNeXt セカンドオピニオン（ポジショニングズレ量推定）
        # NOTE: estimate_positioning_correctionより先に実行してlandmarksを更新する必要がある
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

        # ConvNeXt屈曲角でlandmarks.angles.flexionを上書き（LAT像のみ）
        # v6 LAT像はAP投影なのでYOLO幾何計算では正確な屈曲角が得られない
        # estimate_positioning_correctionより前に適用しないとflexion_adviceが不正確になる
        if (second_opinion is not None
                and second_opinion.get("flexion_deg") is not None):
            landmarks["angles"]["flexion"] = second_opinion["flexion_deg"]

        # ポジショニング補正推定（外顆間距離 × 体格）— ConvNeXt上書き後に実行
        positioning_correction = estimate_positioning_correction(image_array, landmarks)

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
            "version": "YOLOv8s-Pose v6 (mAP50=0.995)",
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
            "version": "ConvNeXt-Small v6 (MAE=0.467deg, ICC=0.9988)",
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
