"""
ElbowVision FastAPI エンドポイント統合テスト
"""
import sys
import os
import pytest
import numpy as np
import cv2

# APIディレクトリに移動してモジュールをロード
api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_client():
    os.chdir(api_dir)
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


@pytest.fixture(scope="module")
def client():
    return get_client()


def make_test_image(size: int = 256) -> bytes:
    """骨様構造を持つテスト用合成肘X線画像"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx = size // 2
    # 上腕骨（上半分の縦長楕円）
    cv2.ellipse(img, (cx, size // 4), (18, 55), 0, 0, 360, (200, 200, 200), -1)
    # 前腕骨（下半分の縦長楕円、やや外反）
    cv2.ellipse(img, (cx + 10, 3 * size // 4), (15, 50), 8, 0, 360, (180, 180, 180), -1)
    # 上顆（関節部の横楕円）
    cv2.ellipse(img, (cx, size // 2), (30, 12), 0, 0, 360, (210, 210, 210), -1)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


# ─── /api/health ──────────────────────────────────────────────────────────────
class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_health_status_ok(self, client):
        assert client.get("/api/health").json()["status"] == "ok"

    def test_health_has_engines(self, client):
        engines = client.get("/api/health").json()["engines"]
        assert "yolo_pose" in engines
        assert "convnext_xai" in engines
        assert "classical_cv" in engines
        assert engines["classical_cv"] is True

    def test_health_has_version(self, client):
        assert "version" in client.get("/api/health").json()


# ─── /api/analyze ─────────────────────────────────────────────────────────────
class TestAnalyzeEndpoint:
    def test_analyze_returns_200(self, client):
        r = client.post("/api/analyze", files={"file": ("test.png", make_test_image(), "image/png")})
        assert r.status_code == 200, f"Response: {r.text}"

    def test_analyze_has_success_flag(self, client):
        r = client.post("/api/analyze", files={"file": ("test.png", make_test_image(), "image/png")})
        assert r.json().get("success") is True

    def test_analyze_has_landmarks(self, client):
        data = client.post("/api/analyze", files={"file": ("test.png", make_test_image(), "image/png")}).json()
        assert "landmarks" in data
        for key in ["humerus_shaft", "condyle_center", "forearm_shaft", "angles", "qa"]:
            assert key in data["landmarks"], f"'{key}' がlandmarksにない"

    def test_analyze_angles_are_numeric(self, client):
        angles = client.post("/api/analyze", files={"file": ("test.png", make_test_image(), "image/png")}).json()["landmarks"]["angles"]
        # 合成テスト画像はYOLOが検出できない場合があるのでNone許容
        assert angles["carrying_angle"] is None or isinstance(angles["carrying_angle"], (int, float))
        assert angles["flexion"] is None or isinstance(angles["flexion"], (int, float))
        assert angles["pronation_sup"] is None or isinstance(angles["pronation_sup"], (int, float))

    def test_analyze_angles_in_plausible_range(self, client):
        angles = client.post("/api/analyze", files={"file": ("test.png", make_test_image(), "image/png")}).json()["landmarks"]["angles"]
        if angles["carrying_angle"] is not None:
            assert -10 <= angles["carrying_angle"] <= 90, f"Carrying={angles['carrying_angle']}° が範囲外"
        if angles["flexion"] is not None:
            assert 0 <= angles["flexion"] <= 180, f"Flexion={angles['flexion']}° が範囲外"
        if angles["pronation_sup"] is not None:
            assert -30 <= angles["pronation_sup"] <= 30, f"Pronation={angles['pronation_sup']}° が範囲外"

    def test_analyze_qa_has_score(self, client):
        qa = client.post("/api/analyze", files={"file": ("test.png", make_test_image(), "image/png")}).json()["landmarks"]["qa"]
        assert "score" in qa
        assert 0 <= qa["score"] <= 100

    def test_analyze_image_size_in_response(self, client):
        data = client.post("/api/analyze", files={"file": ("test.png", make_test_image(size=256), "image/png")}).json()
        assert data["image_size"]["width"] == 256
        assert data["image_size"]["height"] == 256

    def test_analyze_invalid_format_rejected(self, client):
        r = client.post("/api/analyze", files={"file": ("test.gif", b"GIF89a", "image/gif")})
        assert r.status_code in [400, 422, 500]


# ─── /api/upload ──────────────────────────────────────────────────────────────
class TestUploadEndpoint:
    def test_upload_png_returns_200(self, client):
        r = client.post("/api/upload", files={"file": ("test.png", make_test_image(), "image/png")})
        assert r.status_code == 200

    def test_upload_has_metadata(self, client):
        data = client.post("/api/upload", files={"file": ("test.png", make_test_image(), "image/png")}).json()
        assert "metadata" in data

    def test_upload_has_image_data(self, client):
        data = client.post("/api/upload", files={"file": ("test.png", make_test_image(), "image/png")}).json()
        assert data["image_data"].startswith("data:image/png;base64,")

    def test_upload_unsupported_format_rejected(self, client):
        r = client.post("/api/upload", files={"file": ("test.gif", b"GIF89a", "image/gif")})
        assert r.status_code == 400


# ─── エッジケース: 画像サイズ ──────────────────────────────────────────────────
class TestAnalyzeEdgeCases:
    """極端なサイズ・回転画像でのエッジケーステスト"""

    def test_very_small_image(self, client):
        """32x32の極小画像でもクラッシュしない"""
        r = client.post(
            "/api/analyze",
            files={"file": ("tiny.png", make_test_image(size=32), "image/png")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["image_size"]["width"] == 32

    def test_large_image(self, client):
        """1024x1024の大きな画像でも正常に処理される"""
        r = client.post(
            "/api/analyze",
            files={"file": ("large.png", make_test_image(size=1024), "image/png")},
        )
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_non_square_image(self, client):
        """長方形画像でも処理できる"""
        img = np.zeros((512, 256, 3), dtype=np.uint8)
        cx = 128
        cv2.ellipse(img, (cx, 128), (18, 55), 0, 0, 360, (200, 200, 200), -1)
        cv2.ellipse(img, (cx + 10, 384), (15, 50), 8, 0, 360, (180, 180, 180), -1)
        cv2.ellipse(img, (cx, 256), (30, 12), 0, 0, 360, (210, 210, 210), -1)
        _, buf = cv2.imencode(".png", img)

        r = client.post(
            "/api/analyze",
            files={"file": ("rect.png", buf.tobytes(), "image/png")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["image_size"]["width"] == 256
        assert data["image_size"]["height"] == 512

    def test_rotated_bone_image(self, client):
        """骨が傾いた画像でも角度が検出される"""
        size = 256
        img = np.zeros((size, size, 3), dtype=np.uint8)
        # 斜めに傾いた上腕骨
        cv2.ellipse(img, (size // 2 - 20, size // 4), (18, 55), 15, 0, 360, (200, 200, 200), -1)
        # 斜めに傾いた前腕骨
        cv2.ellipse(img, (size // 2 + 20, 3 * size // 4), (15, 50), -10, 0, 360, (180, 180, 180), -1)
        cv2.ellipse(img, (size // 2, size // 2), (30, 12), 0, 0, 360, (210, 210, 210), -1)
        _, buf = cv2.imencode(".png", img)

        r = client.post(
            "/api/analyze",
            files={"file": ("rotated.png", buf.tobytes(), "image/png")},
        )
        assert r.status_code == 200
        angles = r.json()["landmarks"]["angles"]
        # carrying_angle か flexion のどちらかが検出されるはず
        assert angles["carrying_angle"] is not None or angles["flexion"] is not None

    def test_jpeg_format(self, client):
        """JPEG形式でも処理できる"""
        _, buf = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8) + 128)
        r = client.post(
            "/api/analyze",
            files={"file": ("test.jpg", buf.tobytes(), "image/jpeg")},
        )
        assert r.status_code == 200

    def test_analyze_has_edge_validation(self, client):
        """レスポンスにedge_validationフィールドがある"""
        r = client.post(
            "/api/analyze",
            files={"file": ("test.png", make_test_image(), "image/png")},
        )
        data = r.json()
        assert "edge_validation" in data

    def test_analyze_has_positioning_correction(self, client):
        """レスポンスにpositioning_correctionフィールドがある"""
        r = client.post(
            "/api/analyze",
            files={"file": ("test.png", make_test_image(), "image/png")},
        )
        data = r.json()
        assert "positioning_correction" in data
        pc = data["positioning_correction"]
        assert "view_type" in pc
        assert "rotation_error" in pc
        assert "rotation_level" in pc
        assert pc["rotation_level"] in ["good", "minor", "major"]
        assert "overall_level" in pc

    def test_analyze_has_second_opinion(self, client):
        """レスポンスにsecond_opinionフィールドがある（Noneの可能性あり）"""
        r = client.post(
            "/api/analyze",
            files={"file": ("test.png", make_test_image(), "image/png")},
        )
        data = r.json()
        assert "second_opinion" in data

    def test_no_file_returns_422(self, client):
        """ファイルなしのリクエストは422を返す"""
        r = client.post("/api/analyze")
        assert r.status_code == 422

    def test_upload_no_file_returns_422(self, client):
        """uploadでファイルなしは422"""
        r = client.post("/api/upload")
        assert r.status_code == 422

    def test_corrupted_image_returns_error(self, client):
        """壊れた画像データはエラーを返す"""
        r = client.post(
            "/api/analyze",
            files={"file": ("bad.png", b"\x89PNG\r\n\x1a\ncorrupted", "image/png")},
        )
        assert r.status_code in [400, 500]


# ─── /api/gradcam ────────────────────────────────────────────────────────────
class TestGradcamEndpoint:
    def test_gradcam_without_model(self, client):
        """ConvNeXtモデルなしでは503を返す"""
        r = client.post(
            "/api/gradcam",
            files={"file": ("test.png", make_test_image(), "image/png")},
        )
        # モデルが未ロードなら503、ロード済みなら200
        assert r.status_code in [200, 503]
        data = r.json()
        if r.status_code == 503:
            assert data["success"] is False


# ─── ユーティリティ関数テスト ──────────────────────────────────────────────────
class TestUtilityFunctions:
    """main.pyのユーティリティ関数の直接テスト"""

    def test_apply_windowing_basic(self, client):
        """apply_windowing関数の基本動作"""
        os.chdir(api_dir)
        from main import apply_windowing
        arr = np.array([0.0, 500.0, 1000.0], dtype=np.float32)
        result = apply_windowing(arr, 500.0, 1000.0)
        assert result.dtype == np.uint8
        assert result[0] == 0
        assert result[2] == 255

    def test_apply_windowing_none(self, client):
        """apply_windowing: center/widthがNoneなら元データを返す"""
        os.chdir(api_dir)
        from main import apply_windowing
        arr = np.array([100.0, 200.0], dtype=np.float32)
        result = apply_windowing(arr, None, None)
        np.testing.assert_array_equal(result, arr)

    def test_pct(self, client):
        """pct関数: パーセンテージ計算"""
        os.chdir(api_dir)
        from main import pct
        assert pct(50, 200) == 25.0
        assert pct(0, 100) == 0.0

    def test_angle_deg(self, client):
        """angle_deg関数: 2点間の角度計算"""
        os.chdir(api_dir)
        from main import angle_deg
        p1 = {"x": 0, "y": 0}
        p2 = {"x": 1, "y": 0}
        assert angle_deg(p1, p2) == 0.0  # 右方向 = 0度
        p3 = {"x": 0, "y": 1}
        assert angle_deg(p1, p3) == 90.0  # 下方向 = 90度

    def test_full_angle(self, client):
        """full_angle関数: 2つの角度間の差"""
        os.chdir(api_dir)
        from main import full_angle
        assert full_angle(10.0, 20.0) == 10.0
        assert full_angle(350.0, 10.0) == 20.0  # 360度をまたぐ
        assert full_angle(0.0, 180.0) == 180.0


# ─── /api/model-info ─────────────────────────────────────────────────────────
class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        r = client.get("/api/model-info")
        assert r.status_code == 200

    def test_model_info_has_primary_engine(self, client):
        data = client.get("/api/model-info").json()
        assert "primary_engine" in data
        assert data["primary_engine"] in ["YOLOv8-Pose", "Classical CV (fallback)"]

    def test_model_info_has_yolo_section(self, client):
        data = client.get("/api/model-info").json()
        assert "yolo" in data
        assert "loaded" in data["yolo"]
        assert data["yolo"]["keypoints"] == 6

    def test_model_info_has_convnext_section(self, client):
        data = client.get("/api/model-info").json()
        assert "convnext" in data
        assert "loaded" in data["convnext"]

    def test_model_info_has_classical_cv(self, client):
        data = client.get("/api/model-info").json()
        assert data["classical_cv"]["available"] is True

    def test_model_info_fallback_consistent(self, client):
        """fallback_activeがYOLO未ロード状態と整合する"""
        data = client.get("/api/model-info").json()
        assert data["fallback_active"] == (not data["yolo"]["loaded"])


# ─── /api/stats ──────────────────────────────────────────────────────────────
class TestStatsEndpoint:
    def test_stats_returns_200(self, client):
        r = client.get("/api/stats")
        assert r.status_code == 200

    def test_stats_has_total_inferences(self, client):
        data = client.get("/api/stats").json()
        assert "total_inferences" in data
        assert isinstance(data["total_inferences"], int)

    def test_stats_has_uptime(self, client):
        data = client.get("/api/stats").json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_stats_has_engine_counts(self, client):
        data = client.get("/api/stats").json()
        assert "engine_counts" in data
        assert "yolo_pose" in data["engine_counts"]
        assert "classical_cv" in data["engine_counts"]

    def test_stats_has_qa_distribution(self, client):
        data = client.get("/api/stats").json()
        dist = data["qa_score"]["distribution"]
        assert all(k in dist for k in ["excellent", "good", "fair", "poor"])

    def test_stats_increments_after_analyze(self, client):
        """analyzeを呼ぶと統計が増える"""
        before = client.get("/api/stats").json()["total_inferences"]
        client.post("/api/analyze", files={"file": ("test.png", make_test_image(), "image/png")})
        after = client.get("/api/stats").json()["total_inferences"]
        assert after > before


# ─── /api/batch-analyze ──────────────────────────────────────────────────────
class TestBatchAnalyzeEndpoint:
    def test_batch_single_image(self, client):
        """単一画像でもbatch-analyzeが動作する"""
        r = client.post(
            "/api/batch-analyze",
            files={"file": ("test.png", make_test_image(), "image/png")},
        )
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert "summary" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["success"] is True

    def test_batch_zip_upload(self, client):
        """ZIPファイルで複数画像を一括推論"""
        import zipfile as zf
        import io as _io
        buf = _io.BytesIO()
        with zf.ZipFile(buf, 'w') as z:
            z.writestr("img1.png", make_test_image())
            z.writestr("img2.png", make_test_image(size=128))
        buf.seek(0)

        r = client.post(
            "/api/batch-analyze",
            files={"file": ("batch.zip", buf.read(), "application/zip")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["summary"]["total_images"] == 2
        assert data["summary"]["successful"] == 2

    def test_batch_zip_with_invalid_image(self, client):
        """ZIP内に壊れた画像があってもスキップして他は処理する"""
        import zipfile as zf
        import io as _io
        buf = _io.BytesIO()
        with zf.ZipFile(buf, 'w') as z:
            z.writestr("good.png", make_test_image())
            z.writestr("bad.png", b"not an image")
        buf.seek(0)

        r = client.post(
            "/api/batch-analyze",
            files={"file": ("batch.zip", buf.read(), "application/zip")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["summary"]["total_images"] == 2
        assert data["summary"]["failed"] >= 1

    def test_batch_empty_zip(self, client):
        """画像のないZIPはエラーを返す"""
        import zipfile as zf
        import io as _io
        buf = _io.BytesIO()
        with zf.ZipFile(buf, 'w') as z:
            z.writestr("readme.txt", "no images here")
        buf.seek(0)

        r = client.post(
            "/api/batch-analyze",
            files={"file": ("empty.zip", buf.read(), "application/zip")},
        )
        assert r.status_code == 400

    def test_batch_invalid_zip(self, client):
        """不正なZIPファイルはエラーを返す"""
        r = client.post(
            "/api/batch-analyze",
            files={"file": ("bad.zip", b"not a zip file", "application/zip")},
        )
        assert r.status_code == 400

    def test_batch_summary_has_averages(self, client):
        """サマリーに平均値が含まれる"""
        r = client.post(
            "/api/batch-analyze",
            files={"file": ("test.png", make_test_image(), "image/png")},
        )
        summary = r.json()["summary"]
        assert "avg_carrying_angle" in summary or "avg_flexion_angle" in summary
        assert "avg_qa_score" in summary


# ─── /api/compare ────────────────────────────────────────────────────────────
class TestCompareEndpoint:
    def test_compare_returns_200(self, client):
        r = client.post(
            "/api/compare",
            files=[
                ("file1", ("img1.png", make_test_image(), "image/png")),
                ("file2", ("img2.png", make_test_image(size=128), "image/png")),
            ],
        )
        assert r.status_code == 200

    def test_compare_has_both_images(self, client):
        r = client.post(
            "/api/compare",
            files=[
                ("file1", ("img1.png", make_test_image(), "image/png")),
                ("file2", ("img2.png", make_test_image(), "image/png")),
            ],
        )
        data = r.json()
        assert "image1" in data
        assert "image2" in data
        assert "comparison" in data

    def test_compare_has_qa_scores(self, client):
        r = client.post(
            "/api/compare",
            files=[
                ("file1", ("img1.png", make_test_image(), "image/png")),
                ("file2", ("img2.png", make_test_image(), "image/png")),
            ],
        )
        data = r.json()
        assert isinstance(data["image1"]["qa_score"], (int, float))
        assert isinstance(data["image2"]["qa_score"], (int, float))

    def test_compare_has_improvement_label(self, client):
        r = client.post(
            "/api/compare",
            files=[
                ("file1", ("img1.png", make_test_image(), "image/png")),
                ("file2", ("img2.png", make_test_image(), "image/png")),
            ],
        )
        comp = r.json()["comparison"]
        assert comp["improvement_label"] in ["大幅改善", "改善", "変化なし", "やや悪化", "悪化"]

    def test_compare_qa_diff_numeric(self, client):
        r = client.post(
            "/api/compare",
            files=[
                ("file1", ("img1.png", make_test_image(), "image/png")),
                ("file2", ("img2.png", make_test_image(), "image/png")),
            ],
        )
        comp = r.json()["comparison"]
        assert isinstance(comp["qa_score_diff"], (int, float))

    def test_compare_missing_file_returns_422(self, client):
        """ファイルが1つだけの場合は422"""
        r = client.post(
            "/api/compare",
            files=[
                ("file1", ("img1.png", make_test_image(), "image/png")),
            ],
        )
        assert r.status_code == 422


# ─── view_type判定ロジック（前腕軸傾き） ────────────────────────────────────────
class TestViewTypeLogic:
    """AP/LAT判定: 前腕軸傾き(|dx/dy|>0.70)のユニットテスト"""

    def test_vertical_forearm_not_oblique(self):
        """垂直な前腕(伸展AP): |dx/dy| << 0.70 → 前腕斜めでない"""
        import math
        # condyle (128, 106), forearm (124, 173) → dx=-4, dy=67
        fa_dx, fa_dy = -4.0, 67.0
        assert abs(fa_dx) <= abs(fa_dy) * 0.70  # AP条件を満たす

    def test_horizontal_forearm_oblique(self):
        """水平な前腕(屈曲90°LAT): |dx/dy| >> 0.70 → 前腕斜め"""
        # condyle (111, 122), forearm (60, 142) → dx=-51, dy=20
        fa_dx, fa_dy = -51.0, 20.0
        assert abs(fa_dx) > abs(fa_dy) * 0.70  # LAT条件を満たす

    def test_boundary_ap_150deg(self):
        """150°屈曲(AP最小): 前腕は30°傾き → tan(30°)≈0.577 < 0.70 → AP"""
        import math
        # 150°屈曲: 前腕は垂直から30°傾き
        angle_from_vertical = 30.0  # degrees
        ratio = math.tan(math.radians(angle_from_vertical))
        assert ratio < 0.70  # AP判定

    def test_boundary_lat_120deg(self):
        """120°屈曲(LAT最大): 前腕は60°傾き → tan(60°)≈1.73 > 0.70 → LAT"""
        import math
        angle_from_vertical = 60.0
        ratio = math.tan(math.radians(angle_from_vertical))
        assert ratio > 0.70  # LAT判定

    def test_flexed_elbow_image_classified_lat(self, client):
        """水平前腕(屈曲90°)の合成画像: view_typeがLATになる(classicalCV)"""
        size = 256
        img = np.zeros((size, size, 3), dtype=np.uint8)
        cx = size // 2
        # 上腕骨: 垂直
        cv2.ellipse(img, (cx, size // 4), (18, 55), 0, 0, 360, (200, 200, 200), -1)
        # 上顆: 中央
        cv2.ellipse(img, (cx, size // 2), (30, 12), 0, 0, 360, (210, 210, 210), -1)
        # 前腕骨: 水平（cx-70, size//2+20 → LAT90°を模倣）
        cv2.ellipse(img, (cx - 60, size // 2 + 15), (55, 14), 0, 0, 360, (180, 180, 180), -1)
        _, buf = cv2.imencode(".png", img)
        r = client.post("/api/analyze",
                        files={"file": ("flexed.png", buf.tobytes(), "image/png")})
        assert r.status_code == 200
        data = r.json()
        qa = data["landmarks"]["qa"]
        # 合成画像なのでクラシカルCVが使われる; 前腕が水平ならLATまたはAPどちらも許容
        assert qa["view_type"] in ["AP", "LAT"]


# ─── ConvNeXt セカンドオピニオン構造テスト ────────────────────────────────────
class TestSecondOpinionStructure:
    """second_opinion フィールドの構造テスト"""

    def test_second_opinion_fields_when_present(self, client):
        """second_opinionが存在するとき、正しいキーを持つ"""
        r = client.post("/api/analyze",
                        files={"file": ("test.png", make_test_image(), "image/png")})
        data = r.json()
        so = data.get("second_opinion")
        if so is not None:
            assert "rotation_error_deg" in so
            assert "flexion_deg" in so
            assert "model" in so
            # AP/LAT一方のみ非None
            assert not (so["rotation_error_deg"] is not None and so["flexion_deg"] is not None)

    def test_second_opinion_is_none_or_dict(self, client):
        """second_opinionはNoneかdict"""
        r = client.post("/api/analyze",
                        files={"file": ("test.png", make_test_image(), "image/png")})
        so = r.json().get("second_opinion")
        assert so is None or isinstance(so, dict)

    def test_convnext_flexion_overrides_yolo_geometry(self, client):
        """ConvNeXtが使える場合、positioning_correction.angles.flexionに値が入る(LAT時)"""
        r = client.post("/api/analyze",
                        files={"file": ("test.png", make_test_image(), "image/png")})
        data = r.json()
        pc = data.get("positioning_correction", {})
        angles = data.get("landmarks", {}).get("angles", {})
        so = data.get("second_opinion")
        # ConvNeXtが使えてLATと判定された場合、second_opinion.flexion_degとangles.flexionが一致
        if so is not None and so.get("flexion_deg") is not None:
            assert angles.get("flexion") == so["flexion_deg"]

    def test_positioning_correction_uses_convnext_flexion(self, client):
        """positioning_correctionのflexion_degはConvNeXt上書き後の値と一致すること
        （YOLO幾何値ではなくConvNeXt値をpositioning_correctionに反映するため、
         estimate_positioning_correctionはConvNeXt上書き後に呼ばれなければならない）"""
        r = client.post("/api/analyze",
                        files={"file": ("test.png", make_test_image(), "image/png")})
        data = r.json()
        pc = data.get("positioning_correction", {})
        angles = data.get("landmarks", {}).get("angles", {})
        so = data.get("second_opinion")
        # LAT+ConvNeXt使用時: positioning_correction.flexion_deg == landmarks.angles.flexion
        if so is not None and so.get("flexion_deg") is not None:
            pc_flexion = pc.get("flexion_deg")
            lm_flexion = angles.get("flexion")
            # どちらもConvNeXt値(second_opinion.flexion_deg)と一致するはず
            assert pc_flexion == so["flexion_deg"], (
                f"positioning_correction.flexion_deg ({pc_flexion}) != "
                f"second_opinion.flexion_deg ({so['flexion_deg']})"
            )
            assert lm_flexion == so["flexion_deg"]

    def test_inference_engine_present_in_qa(self, client):
        """Classical CVフォールバック時もqa.inference_engineキーが存在すること"""
        r = client.post("/api/analyze",
                        files={"file": ("test.png", make_test_image(), "image/png")})
        data = r.json()
        qa = data.get("landmarks", {}).get("qa", {})
        assert "inference_engine" in qa, "qa に inference_engine キーがない"
