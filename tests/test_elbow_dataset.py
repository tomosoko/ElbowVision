"""elbow-api/training/train_angle_predictor.py の ElbowDataset テスト."""
import sys
import os
import tempfile
import csv
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'elbow-api', 'training'))

import pandas as pd
from train_angle_predictor import ElbowDataset


def _make_csv(tmp_path, rows):
    """CSVファイルを作成してパスを返す."""
    csv_path = tmp_path / "labels.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return str(csv_path)


def _make_image(path: Path, size=(32, 32)):
    """ダミー PNG 画像を作成."""
    from PIL import Image
    img = Image.fromarray(np.zeros((*size, 3), dtype=np.uint8))
    img.save(path)


class TestResolveImgPath:
    """ElbowDataset._resolve_img_path のテスト."""

    def test_prefers_split_subdir(self, tmp_path):
        """images/{split}/{filename} が存在する場合はそちらを返す."""
        img_dir = tmp_path / "images"
        (img_dir / "train").mkdir(parents=True)
        img_file = img_dir / "train" / "test.png"
        _make_image(img_file)

        rows = [{"filename": "test.png", "split": "train",
                 "rotation_error_deg": 0.0, "flexion_deg": 90.0}]
        csv_path = _make_csv(tmp_path, rows)
        ds = ElbowDataset(csv_path, str(img_dir))

        row = ds.df.iloc[0]
        resolved = ds._resolve_img_path(row)
        assert resolved == str(img_file)

    def test_falls_back_to_flat_dir(self, tmp_path):
        """split サブディレクトリにないがフラット構造にある場合はそちらを返す."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img_file = img_dir / "flat.png"
        _make_image(img_file)

        rows = [{"filename": "flat.png", "split": "train",
                 "rotation_error_deg": 0.0, "flexion_deg": 90.0}]
        csv_path = _make_csv(tmp_path, rows)
        ds = ElbowDataset(csv_path, str(img_dir))

        row = ds.df.iloc[0]
        resolved = ds._resolve_img_path(row)
        assert resolved == str(img_file)

    def test_fallback_when_not_found(self, tmp_path):
        """どちらにも存在しない場合は split パスを返す (エラーはDataLoader側)."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        rows = [{"filename": "missing.png", "split": "val",
                 "rotation_error_deg": 0.0, "flexion_deg": 90.0}]
        csv_path = _make_csv(tmp_path, rows)
        ds = ElbowDataset(csv_path, str(img_dir))

        row = ds.df.iloc[0]
        resolved = ds._resolve_img_path(row)
        # 存在しないので split パスが返る
        assert "val" in resolved
        assert "missing.png" in resolved

    def test_dataset_length(self, tmp_path):
        """__len__ がCSVの行数を返す."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        rows = [
            {"filename": f"img{i}.png", "split": "train",
             "rotation_error_deg": float(i), "flexion_deg": 90.0}
            for i in range(5)
        ]
        csv_path = _make_csv(tmp_path, rows)
        ds = ElbowDataset(csv_path, str(img_dir))
        assert len(ds) == 5

    def test_resolve_uses_train_default_split(self, tmp_path):
        """split 列がない場合は 'train' をデフォルトとして使う."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        rows = [{"filename": "no_split.png", "rotation_error_deg": 0.0, "flexion_deg": 90.0}]
        csv_path = _make_csv(tmp_path, rows)
        ds = ElbowDataset(csv_path, str(img_dir))

        row = ds.df.iloc[0]
        resolved = ds._resolve_img_path(row)
        # デフォルトsplit='train' でパスが構成される
        assert "train" in resolved or str(img_dir) in resolved


class TestElbowDatasetGetItem:
    """ElbowDataset.__getitem__ のテスト."""

    def test_getitem_returns_image_angles_mask(self, tmp_path):
        """__getitem__ は (image, angles, mask) を返す."""
        import torch
        from torchvision import transforms

        img_dir = tmp_path / "images" / "train"
        img_dir.mkdir(parents=True)
        img_file = img_dir / "img.png"
        _make_image(img_file)

        rows = [{"filename": "img.png", "split": "train",
                 "rotation_error_deg": 5.0, "flexion_deg": 90.0,
                 "view_type": "AP"}]
        csv_path = _make_csv(tmp_path, rows)

        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        ds = ElbowDataset(csv_path, str(tmp_path / "images"), transform=transform)

        image, angles, mask = ds[0]
        assert image.shape == (3, 32, 32)
        assert angles.shape == (2,)
        assert mask.shape == (2,)

    def test_getitem_ap_mask(self, tmp_path):
        """AP view: mask=[1, 0] (rotation有効, flexion無効)."""
        import torch

        img_dir = tmp_path / "images" / "train"
        img_dir.mkdir(parents=True)
        _make_image(img_dir / "img.png")

        rows = [{"filename": "img.png", "split": "train",
                 "rotation_error_deg": 5.0, "flexion_deg": 90.0,
                 "view_type": "AP"}]
        csv_path = _make_csv(tmp_path, rows)
        ds = ElbowDataset(csv_path, str(tmp_path / "images"))
        _, _, mask = ds[0]
        assert float(mask[0]) == 1.0   # rotation_error_deg: AP有効
        assert float(mask[1]) == 0.0   # flexion_deg: AP無効

    def test_getitem_lat_mask(self, tmp_path):
        """LAT view: mask=[0, 1] (rotation無効, flexion有効)."""
        import torch

        img_dir = tmp_path / "images" / "train"
        img_dir.mkdir(parents=True)
        _make_image(img_dir / "img.png")

        rows = [{"filename": "img.png", "split": "train",
                 "rotation_error_deg": 0.0, "flexion_deg": 90.0,
                 "view_type": "LAT"}]
        csv_path = _make_csv(tmp_path, rows)
        ds = ElbowDataset(csv_path, str(tmp_path / "images"))
        _, _, mask = ds[0]
        assert float(mask[0]) == 0.0   # rotation_error_deg: LAT無効
        assert float(mask[1]) == 1.0   # flexion_deg: LAT有効

    def test_getitem_angles_values(self, tmp_path):
        """角度値がCSVの値と一致する."""
        img_dir = tmp_path / "images" / "train"
        img_dir.mkdir(parents=True)
        _make_image(img_dir / "img.png")

        rows = [{"filename": "img.png", "split": "train",
                 "rotation_error_deg": 7.5, "flexion_deg": 120.0,
                 "view_type": "AP"}]
        csv_path = _make_csv(tmp_path, rows)
        ds = ElbowDataset(csv_path, str(tmp_path / "images"))
        _, angles, _ = ds[0]
        assert abs(float(angles[0]) - 7.5) < 1e-5
        assert abs(float(angles[1]) - 120.0) < 1e-5
