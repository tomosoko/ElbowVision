"""
inference_test.py のユニットテスト
"""
import csv
import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

# スクリプトをimportできるようにパス追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "elbow-api"))

import inference_test as it


# ─── テスト用ダミー画像生成 ────────────────────────────────────────────────────

def _make_dummy_xray(path: str, w: int = 512, h: int = 768):
    """骨っぽい構造を持つダミーX線画像を生成"""
    img = np.zeros((h, w), dtype=np.uint8)
    # 背景にノイズ
    img[:] = 30
    # 上腕骨（上部の白い帯）
    cv2.line(img, (w // 2, 50), (w // 2, h // 2 - 30), 200, 40)
    # 前腕骨（下部の白い帯）
    cv2.line(img, (w // 2, h // 2 + 30), (w // 2, h - 50), 200, 35)
    # 顆部（中央の太い部分）
    cv2.ellipse(img, (w // 2, h // 2), (60, 30), 0, 0, 360, 180, -1)
    # BGRに変換
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(path, bgr)
    return path


@pytest.fixture
def dummy_image_dir(tmp_path):
    """テスト用画像ディレクトリを作成"""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    _make_dummy_xray(str(img_dir / "test_AP.png"))
    _make_dummy_xray(str(img_dir / "test_LAT.png"))
    return str(img_dir)


@pytest.fixture
def output_dir(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    return str(out)


# ─── collect_images テスト ─────────────────────────────────────────────────────

class TestCollectImages:
    def test_finds_png_files(self, dummy_image_dir):
        images = it.collect_images(dummy_image_dir)
        assert len(images) == 2
        assert all(f.endswith(".png") for f in images)

    def test_sorted_order(self, dummy_image_dir):
        images = it.collect_images(dummy_image_dir)
        basenames = [os.path.basename(f) for f in images]
        assert basenames == sorted(basenames)

    def test_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert it.collect_images(str(empty)) == []

    def test_ignores_hidden_files(self, tmp_path):
        d = tmp_path / "hidden"
        d.mkdir()
        _make_dummy_xray(str(d / ".hidden.png"))
        _make_dummy_xray(str(d / "visible.png"))
        images = it.collect_images(str(d))
        assert len(images) == 1
        assert "visible.png" in images[0]

    def test_supports_multiple_extensions(self, tmp_path):
        d = tmp_path / "multi"
        d.mkdir()
        _make_dummy_xray(str(d / "a.png"))
        _make_dummy_xray(str(d / "b.jpg"))
        _make_dummy_xray(str(d / "c.jpeg"))
        # .txtは無視される
        (d / "d.txt").write_text("not an image")
        images = it.collect_images(str(d))
        assert len(images) == 3


# ─── run_inference テスト ──────────────────────────────────────────────────────

class TestRunInference:
    def test_returns_required_keys(self, dummy_image_dir):
        images = it.collect_images(dummy_image_dir)
        result = it.run_inference(images[0])
        assert "filename" in result
        assert "image_array" in result
        assert "landmarks" in result
        assert "correction" in result

    def test_landmarks_have_angles(self, dummy_image_dir):
        images = it.collect_images(dummy_image_dir)
        result = it.run_inference(images[0])
        angles = result["landmarks"]["angles"]
        assert "carrying_angle" in angles
        assert "flexion" in angles
        assert "pronation_sup" in angles
        assert "varus_valgus" in angles

    def test_correction_has_rotation_error(self, dummy_image_dir):
        images = it.collect_images(dummy_image_dir)
        result = it.run_inference(images[0])
        corr = result["correction"]
        assert "rotation_error" in corr
        assert "overall_level" in corr

    def test_qa_fields(self, dummy_image_dir):
        images = it.collect_images(dummy_image_dir)
        result = it.run_inference(images[0])
        qa = result["landmarks"]["qa"]
        assert "view_type" in qa
        assert "score" in qa
        assert qa["view_type"] in ("AP", "LAT")
        assert isinstance(qa["score"], (int, float))


# ─── build_csv_row テスト ──────────────────────────────────────────────────────

class TestBuildCsvRow:
    def test_all_columns_present(self, dummy_image_dir):
        images = it.collect_images(dummy_image_dir)
        result = it.run_inference(images[0])
        row = it.build_csv_row(result)
        for col in it.CSV_COLUMNS:
            assert col in row, f"Missing column: {col}"

    def test_bland_altman_columns_empty(self, dummy_image_dir):
        """手動計測列は空文字で初期化されること"""
        images = it.collect_images(dummy_image_dir)
        result = it.run_inference(images[0])
        row = it.build_csv_row(result)
        assert row["manual_carrying_angle"] == ""
        assert row["manual_flexion_deg"] == ""

    def test_image_id_no_extension(self, dummy_image_dir):
        images = it.collect_images(dummy_image_dir)
        result = it.run_inference(images[0])
        row = it.build_csv_row(result)
        assert "." not in row["image_id"]


# ─── draw_overlay テスト ───────────────────────────────────────────────────────

class TestDrawOverlay:
    def test_output_same_size(self, dummy_image_dir):
        images = it.collect_images(dummy_image_dir)
        result = it.run_inference(images[0])
        overlay = it.draw_overlay(
            result["image_array"], result["landmarks"], result["correction"]
        )
        assert overlay.shape == result["image_array"].shape

    def test_output_is_color(self, dummy_image_dir):
        images = it.collect_images(dummy_image_dir)
        result = it.run_inference(images[0])
        overlay = it.draw_overlay(
            result["image_array"], result["landmarks"], result["correction"]
        )
        assert len(overlay.shape) == 3
        assert overlay.shape[2] == 3

    def test_overlay_modifies_image(self, dummy_image_dir):
        """オーバーレイ後は元画像と異なること"""
        images = it.collect_images(dummy_image_dir)
        result = it.run_inference(images[0])
        original = result["image_array"].copy()
        overlay = it.draw_overlay(original, result["landmarks"], result["correction"])
        assert not np.array_equal(overlay, result["image_array"])


# ─── CSV出力の統合テスト ───────────────────────────────────────────────────────

class TestCsvOutput:
    def test_csv_written(self, dummy_image_dir, output_dir):
        """CSV出力が正しく書き込まれること"""
        images = it.collect_images(dummy_image_dir)
        rows = []
        for img_path in images:
            result = it.run_inference(img_path)
            rows.append(it.build_csv_row(result))

        csv_path = os.path.join(output_dir, "test_results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=it.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        # 読み戻し検証
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            read_rows = list(reader)
        assert len(read_rows) == 2
        assert set(reader.fieldnames) == set(it.CSV_COLUMNS)


# ─── エッジケース ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_small_image(self, tmp_path):
        """極小画像でもエラーにならないこと"""
        d = tmp_path / "small"
        d.mkdir()
        path = str(d / "tiny.png")
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        cv2.imwrite(path, img)
        result = it.run_inference(path)
        assert result["landmarks"] is not None

    def test_grayscale_like_xray(self, tmp_path):
        """実際のX線に近いグレースケール画像"""
        d = tmp_path / "gray"
        d.mkdir()
        path = str(d / "gray.png")
        gray = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(path, bgr)
        result = it.run_inference(path)
        assert "angles" in result["landmarks"]
