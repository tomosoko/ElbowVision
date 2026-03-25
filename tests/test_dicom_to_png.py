"""
dicom_to_png.py のユニットテスト

DICOMファイルのPNG変換パイプラインをテストする。
ダミーDICOMデータを pydicom で生成してテストを行う。
"""
import os
import sys
import tempfile
import shutil

import cv2
import numpy as np
import pydicom
import pydicom.uid
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "elbow-train"))
from dicom_to_png import apply_windowing, dicom_to_array, convert_dir, split_train_val


# ─── ヘルパー ──────────────────────────────────────────────────────────────────


def _make_dummy_dicom(path: str,
                      rows: int = 64,
                      cols: int = 64,
                      photometric: str = "MONOCHROME2",
                      window_center: float = None,
                      window_width: float = None) -> str:
    """テスト用ダミーDICOMファイルを生成"""
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()

    ds = pydicom.dataset.FileDataset(
        path, {}, file_meta=file_meta, preamble=b"\x00" * 128
    )
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric

    # 骨を模した円形の高輝度領域を持つダミーピクセルデータ
    img = np.zeros((rows, cols), dtype=np.uint16)
    cy, cx = rows // 2, cols // 2
    for y in range(rows):
        for x in range(cols):
            if (y - cy) ** 2 + (x - cx) ** 2 < (min(rows, cols) // 4) ** 2:
                img[y, x] = 2000
            else:
                img[y, x] = 200

    ds.PixelData = img.tobytes()

    if window_center is not None:
        ds.WindowCenter = window_center
    if window_width is not None:
        ds.WindowWidth = window_width

    pydicom.dcmwrite(path, ds)
    return path


@pytest.fixture
def tmp_dir():
    """一時ディレクトリを作成し、テスト後に削除"""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ─── apply_windowing テスト ────────────────────────────────────────────────────


class TestApplyWindowing:
    def test_basic_windowing(self):
        """ウィンドウ内の値が0-255に正規化される"""
        arr = np.array([0.0, 500.0, 1000.0], dtype=np.float32)
        result = apply_windowing(arr, center=500.0, width=1000.0)
        assert result.dtype == np.uint8
        assert result[0] == 0
        assert result[2] == 255
        assert 100 < result[1] < 155  # 中間値

    def test_clipping(self):
        """ウィンドウ外の値はクリップされる"""
        arr = np.array([-1000.0, 5000.0], dtype=np.float32)
        result = apply_windowing(arr, center=500.0, width=1000.0)
        assert result[0] == 0
        assert result[1] == 255

    def test_narrow_window(self):
        """狭いウィンドウでもクラッシュしない"""
        arr = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        result = apply_windowing(arr, center=200.0, width=10.0)
        assert result.dtype == np.uint8

    def test_output_range(self):
        """出力値が常に0-255の範囲"""
        arr = np.random.uniform(-2000, 4000, size=100).astype(np.float32)
        result = apply_windowing(arr, center=1000.0, width=2000.0)
        assert result.min() >= 0
        assert result.max() <= 255


# ─── dicom_to_array テスト ─────────────────────────────────────────────────────


class TestDicomToArray:
    def test_basic_conversion(self, tmp_dir):
        """DICOMからBGR画像に正しく変換される"""
        path = _make_dummy_dicom(
            os.path.join(tmp_dir, "test.dcm"),
            window_center=1100.0,
            window_width=1800.0,
        )
        img = dicom_to_array(path, apply_clahe=False, output_size=0)
        assert img.ndim == 3
        assert img.shape[2] == 3  # BGR
        assert img.dtype == np.uint8

    def test_with_clahe(self, tmp_dir):
        """CLAHE適用時も正常に動作する"""
        path = _make_dummy_dicom(
            os.path.join(tmp_dir, "test.dcm"),
            window_center=1100.0,
            window_width=1800.0,
        )
        img = dicom_to_array(path, apply_clahe=True, output_size=0)
        assert img.shape[:2] == (64, 64)

    def test_resize(self, tmp_dir):
        """指定サイズにリサイズされる"""
        path = _make_dummy_dicom(os.path.join(tmp_dir, "test.dcm"))
        img = dicom_to_array(path, apply_clahe=False, output_size=128)
        assert img.shape == (128, 128, 3)

    def test_monochrome1_inversion(self, tmp_dir):
        """MONOCHROME1で値が反転される"""
        path = _make_dummy_dicom(
            os.path.join(tmp_dir, "mono1.dcm"),
            photometric="MONOCHROME1",
        )
        img_mono1 = dicom_to_array(path, apply_clahe=False, output_size=0)

        path2 = _make_dummy_dicom(
            os.path.join(tmp_dir, "mono2.dcm"),
            photometric="MONOCHROME2",
        )
        img_mono2 = dicom_to_array(path2, apply_clahe=False, output_size=0)

        # MONOCHROME1は反転されるので結果が異なるはず
        assert not np.array_equal(img_mono1, img_mono2)

    def test_no_window_fallback(self, tmp_dir):
        """Window情報なしでもフォールバック正規化で動作する"""
        path = _make_dummy_dicom(os.path.join(tmp_dir, "nowin.dcm"))
        img = dicom_to_array(path, apply_clahe=False, output_size=0)
        assert img.dtype == np.uint8
        assert img.max() > 0  # 全黒ではない

    def test_has_contrast(self, tmp_dir):
        """骨と背景にコントラストがある"""
        path = _make_dummy_dicom(
            os.path.join(tmp_dir, "contrast.dcm"),
            window_center=1100.0,
            window_width=2000.0,
        )
        img = dicom_to_array(path, apply_clahe=False, output_size=0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        center_val = int(gray[32, 32])  # 骨領域
        corner_val = int(gray[0, 0])    # 背景
        assert center_val != corner_val


# ─── convert_dir テスト ────────────────────────────────────────────────────────


class TestConvertDir:
    def test_converts_multiple_files(self, tmp_dir):
        """複数DICOMファイルがPNGに変換される"""
        in_dir = os.path.join(tmp_dir, "input")
        out_dir = os.path.join(tmp_dir, "output")
        os.makedirs(in_dir)

        for i in range(3):
            _make_dummy_dicom(os.path.join(in_dir, f"img_{i:03d}.dcm"))

        converted = convert_dir(in_dir, out_dir, apply_clahe=False, output_size=0)
        assert len(converted) == 3
        for p in converted:
            assert os.path.exists(p)
            assert p.endswith(".png")

    def test_empty_directory(self, tmp_dir):
        """DICOMファイルがないディレクトリでは空リストを返す"""
        in_dir = os.path.join(tmp_dir, "empty")
        out_dir = os.path.join(tmp_dir, "output")
        os.makedirs(in_dir)
        converted = convert_dir(in_dir, out_dir, apply_clahe=False, output_size=0)
        assert converted == []

    def test_non_dicom_files_ignored(self, tmp_dir):
        """DICOM以外のファイルは無視される"""
        in_dir = os.path.join(tmp_dir, "mixed")
        out_dir = os.path.join(tmp_dir, "output")
        os.makedirs(in_dir)

        # DICOMファイル1つ + テキストファイル1つ
        _make_dummy_dicom(os.path.join(in_dir, "valid.dcm"))
        with open(os.path.join(in_dir, "notes.txt"), "w") as f:
            f.write("not a DICOM")

        converted = convert_dir(in_dir, out_dir, apply_clahe=False, output_size=0)
        assert len(converted) == 1

    def test_output_dir_created(self, tmp_dir):
        """出力ディレクトリが自動生成される"""
        in_dir = os.path.join(tmp_dir, "input")
        out_dir = os.path.join(tmp_dir, "new", "nested", "output")
        os.makedirs(in_dir)
        _make_dummy_dicom(os.path.join(in_dir, "test.dcm"))
        convert_dir(in_dir, out_dir, apply_clahe=False, output_size=0)
        assert os.path.isdir(out_dir)


# ─── split_train_val テスト ────────────────────────────────────────────────────


class TestSplitTrainVal:
    def test_split_ratio(self, tmp_dir):
        """train/valの分割比率が正しい"""
        # ダミー画像ファイルを10個作成
        images = []
        for i in range(10):
            p = os.path.join(tmp_dir, f"img_{i}.png")
            cv2.imwrite(p, np.zeros((32, 32, 3), dtype=np.uint8))
            images.append(p)

        base_dir = os.path.join(tmp_dir, "split")
        split_train_val(images, base_dir, val_ratio=0.2)

        train_files = os.listdir(os.path.join(base_dir, "train"))
        val_files = os.listdir(os.path.join(base_dir, "val"))
        assert len(train_files) + len(val_files) == 10
        assert len(val_files) == 2  # 10 * 0.2 = 2

    def test_minimum_one_val(self, tmp_dir):
        """画像が少なくてもvalに最低1枚入る"""
        images = []
        for i in range(2):
            p = os.path.join(tmp_dir, f"img_{i}.png")
            cv2.imwrite(p, np.zeros((32, 32, 3), dtype=np.uint8))
            images.append(p)

        base_dir = os.path.join(tmp_dir, "split")
        split_train_val(images, base_dir, val_ratio=0.2)

        val_files = os.listdir(os.path.join(base_dir, "val"))
        assert len(val_files) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
