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
        assert isinstance(angles["carrying_angle"], (int, float))
        assert angles["flexion"] is None or isinstance(angles["flexion"], (int, float))
        assert angles["pronation_sup"] is None or isinstance(angles["pronation_sup"], (int, float))

    def test_analyze_angles_in_plausible_range(self, client):
        angles = client.post("/api/analyze", files={"file": ("test.png", make_test_image(), "image/png")}).json()["landmarks"]["angles"]
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
