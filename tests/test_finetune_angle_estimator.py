"""Tests for scripts/finetune_angle_estimator.py pure functions and model architecture.

Covers:
  - build_model() architecture: output shape, layer structure, parameter count
  - freeze_backbone_partial(): parameter freezing/unfreezing logic
  - RealXrayDataset: constants, __len__, _augment, normalization formula
  - Normalization / denormalization round-trip consistency

Note: weights=None is used everywhere so no download is required.
      No real images or CSV files are needed — everything is mocked.
"""
import csv
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from finetune_angle_estimator import (
    build_model,
    freeze_backbone_partial,
    RealXrayDataset,
)


# ── build_model() ─────────────────────────────────────────────────────────────

class TestBuildModel:
    """Verify AngleEstimator model architecture from build_model()."""

    def test_returns_nn_module(self):
        model = build_model()
        assert isinstance(model, nn.Module)

    def test_has_backbone(self):
        model = build_model()
        assert hasattr(model, "backbone")

    def test_backbone_is_convnext(self):
        model = build_model()
        assert "convnext" in type(model.backbone).__name__.lower()

    def test_classifier_output_is_1(self):
        model = build_model()
        final_layer = model.backbone.classifier[2]
        assert isinstance(final_layer, nn.Linear)
        assert final_layer.out_features == 1

    def test_classifier_input_is_768(self):
        model = build_model()
        final_layer = model.backbone.classifier[2]
        assert final_layer.in_features == 768

    def test_forward_single_batch(self):
        model = build_model()
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1,)

    def test_forward_multi_batch(self):
        model = build_model()
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4,)

    def test_output_is_float(self):
        model = build_model()
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.dtype == torch.float32

    def test_model_has_parameters(self):
        model = build_model()
        total = sum(p.numel() for p in model.parameters())
        assert total > 0

    def test_total_params_convnext_small_range(self):
        """ConvNeXt-Small has ~50M params."""
        model = build_model()
        total = sum(p.numel() for p in model.parameters())
        assert 40_000_000 < total < 60_000_000

    def test_gradient_flows(self):
        model = build_model()
        model.train()
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)
        assert has_grad

    def test_eval_mode(self):
        model = build_model()
        model.eval()
        assert not model.training

    def test_train_mode(self):
        model = build_model()
        model.train()
        assert model.training


# ── freeze_backbone_partial() ────────────────────────────────────────────────

class TestFreezeBackbonePartial:
    """Verify partial freezing of ConvNeXt backbone."""

    def test_returns_model(self):
        model = build_model()
        result = freeze_backbone_partial(model, unfreeze_last_n_stages=2)
        assert result is model

    def test_classifier_always_trainable(self):
        model = build_model()
        freeze_backbone_partial(model, unfreeze_last_n_stages=0)
        for p in model.backbone.classifier.parameters():
            assert p.requires_grad

    def test_freeze_all_but_classifier(self):
        """With unfreeze_last_n_stages=0, only classifier should be trainable."""
        model = build_model()
        freeze_backbone_partial(model, unfreeze_last_n_stages=0)
        for p in model.backbone.features.parameters():
            assert not p.requires_grad

    def test_unfreeze_last_2_stages(self):
        model = build_model()
        freeze_backbone_partial(model, unfreeze_last_n_stages=2)
        features = list(model.backbone.features.children())
        n_blocks = len(features)
        # Last 2 blocks should have trainable params
        for i in range(n_blocks - 2, n_blocks):
            has_trainable = any(p.requires_grad for p in features[i].parameters())
            assert has_trainable, f"Block {i} should be trainable"
        # Earlier blocks should be frozen
        for i in range(0, n_blocks - 2):
            all_frozen = all(not p.requires_grad for p in features[i].parameters())
            assert all_frozen, f"Block {i} should be frozen"

    def test_unfreeze_last_1_stage(self):
        model = build_model()
        freeze_backbone_partial(model, unfreeze_last_n_stages=1)
        features = list(model.backbone.features.children())
        n_blocks = len(features)
        # Only last block trainable
        has_trainable = any(p.requires_grad for p in features[n_blocks - 1].parameters())
        assert has_trainable
        # Second-to-last should be frozen
        all_frozen = all(not p.requires_grad for p in features[n_blocks - 2].parameters())
        assert all_frozen

    def test_unfreeze_all_stages(self):
        model = build_model()
        features = list(model.backbone.features.children())
        n_blocks = len(features)
        freeze_backbone_partial(model, unfreeze_last_n_stages=n_blocks)
        # All features should be trainable
        for p in model.backbone.features.parameters():
            assert p.requires_grad

    def test_trainable_count_increases_with_stages(self):
        counts = []
        for n in [0, 1, 2, 4]:
            model = build_model()
            freeze_backbone_partial(model, unfreeze_last_n_stages=n)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            counts.append(trainable)
        # More unfrozen stages → more trainable params
        for i in range(len(counts) - 1):
            assert counts[i] < counts[i + 1]

    def test_default_2_stages_trainable_percentage(self):
        """Default unfreeze_last_n_stages=2 should unfreeze roughly 20-60% of params."""
        model = build_model()
        freeze_backbone_partial(model, unfreeze_last_n_stages=2)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        pct = trainable / total * 100
        assert 10 < pct < 70

    def test_convnext_small_has_8_feature_blocks(self):
        model = build_model()
        features = list(model.backbone.features.children())
        assert len(features) == 8

    def test_frozen_params_have_no_grad_after_backward(self):
        model = build_model()
        freeze_backbone_partial(model, unfreeze_last_n_stages=1)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        out.sum().backward()
        features = list(model.backbone.features.children())
        # First block should have no grad
        for p in features[0].parameters():
            assert p.grad is None


# ── RealXrayDataset constants ────────────────────────────────────────────────

class TestDatasetConstants:
    def test_angle_min_is_90(self):
        assert RealXrayDataset.ANGLE_MIN == 90.0

    def test_angle_max_is_180(self):
        assert RealXrayDataset.ANGLE_MAX == 180.0

    def test_angle_range(self):
        assert RealXrayDataset.ANGLE_MAX - RealXrayDataset.ANGLE_MIN == 90.0


# ── RealXrayDataset init and __len__ ─────────────────────────────────────────

class TestDatasetInit:
    """Test dataset creation with mock CSV and images."""

    @pytest.fixture
    def dataset_dir(self, tmp_path):
        """Create temp dir with fake images and GT CSV."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        # Create fake grayscale images
        for name in ["img1.png", "img2.png", "img3.png"]:
            img = np.random.randint(20, 240, (256, 256), dtype=np.uint8)
            cv2.imwrite(str(img_dir / name), img)
        # Create GT CSV
        csv_path = tmp_path / "gt.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "gt_angle", "note"])
            writer.writeheader()
            writer.writerow({"filename": "img1.png", "gt_angle": "90.0", "note": "LAT"})
            writer.writerow({"filename": "img2.png", "gt_angle": "135.0", "note": "oblique"})
            writer.writerow({"filename": "img3.png", "gt_angle": "180.0", "note": "AP"})
        return tmp_path

    def test_loads_all_samples(self, dataset_dir):
        ds = RealXrayDataset(str(dataset_dir / "gt.csv"), str(dataset_dir / "images"),
                             augment=False)
        assert len(ds.samples) == 3

    def test_len_no_augment(self, dataset_dir):
        ds = RealXrayDataset(str(dataset_dir / "gt.csv"), str(dataset_dir / "images"),
                             augment=False)
        assert len(ds) == 3

    def test_len_with_augment_default(self, dataset_dir):
        ds = RealXrayDataset(str(dataset_dir / "gt.csv"), str(dataset_dir / "images"),
                             augment=True, aug_factor=20)
        assert len(ds) == 60  # 3 * 20

    def test_len_with_custom_aug_factor(self, dataset_dir):
        ds = RealXrayDataset(str(dataset_dir / "gt.csv"), str(dataset_dir / "images"),
                             augment=True, aug_factor=5)
        assert len(ds) == 15  # 3 * 5

    def test_skips_missing_images(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        # Only create img1
        img = np.random.randint(20, 240, (256, 256), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "img1.png"), img)
        csv_path = tmp_path / "gt.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "gt_angle", "note"])
            writer.writeheader()
            writer.writerow({"filename": "img1.png", "gt_angle": "90.0", "note": ""})
            writer.writerow({"filename": "missing.png", "gt_angle": "120.0", "note": ""})
        ds = RealXrayDataset(str(csv_path), str(img_dir), augment=False)
        assert len(ds.samples) == 1

    def test_raises_on_empty_dataset(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        csv_path = tmp_path / "gt.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "gt_angle", "note"])
            writer.writeheader()
            writer.writerow({"filename": "nonexistent.png", "gt_angle": "90.0", "note": ""})
        with pytest.raises(FileNotFoundError):
            RealXrayDataset(str(csv_path), str(img_dir), augment=False)

    def test_samples_store_angles(self, dataset_dir):
        ds = RealXrayDataset(str(dataset_dir / "gt.csv"), str(dataset_dir / "images"),
                             augment=False)
        angles = [a for _, a in ds.samples]
        assert 90.0 in angles
        assert 135.0 in angles
        assert 180.0 in angles


# ── RealXrayDataset __getitem__ ──────────────────────────────────────────────

class TestDatasetGetItem:
    """Test __getitem__ output shape, dtype, normalization."""

    @pytest.fixture
    def dataset(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.random.randint(20, 240, (256, 256), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.png"), img)
        csv_path = tmp_path / "gt.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "gt_angle", "note"])
            writer.writeheader()
            writer.writerow({"filename": "test.png", "gt_angle": "120.0", "note": ""})
        return RealXrayDataset(str(csv_path), str(img_dir), augment=False)

    def test_returns_three_items(self, dataset):
        tensor, label, fname = dataset[0]
        assert tensor is not None
        assert label is not None
        assert fname is not None

    def test_tensor_shape(self, dataset):
        tensor, _, _ = dataset[0]
        assert tensor.shape == (3, 224, 224)

    def test_tensor_dtype(self, dataset):
        tensor, _, _ = dataset[0]
        assert tensor.dtype == torch.float32

    def test_label_dtype(self, dataset):
        _, label, _ = dataset[0]
        assert label.dtype == torch.float32

    def test_label_normalized_range(self, dataset):
        """Label for 120° should be (120-90)/(180-90) = 0.333..."""
        _, label, _ = dataset[0]
        expected = (120.0 - 90.0) / (180.0 - 90.0)
        assert abs(float(label) - expected) < 1e-5

    def test_label_for_90_degrees(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.random.randint(20, 240, (256, 256), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "t.png"), img)
        csv_path = tmp_path / "gt.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "gt_angle", "note"])
            writer.writeheader()
            writer.writerow({"filename": "t.png", "gt_angle": "90.0", "note": ""})
        ds = RealXrayDataset(str(csv_path), str(img_dir), augment=False)
        _, label, _ = ds[0]
        assert abs(float(label) - 0.0) < 1e-5

    def test_label_for_180_degrees(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.random.randint(20, 240, (256, 256), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "t.png"), img)
        csv_path = tmp_path / "gt.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "gt_angle", "note"])
            writer.writeheader()
            writer.writerow({"filename": "t.png", "gt_angle": "180.0", "note": ""})
        ds = RealXrayDataset(str(csv_path), str(img_dir), augment=False)
        _, label, _ = ds[0]
        assert abs(float(label) - 1.0) < 1e-5

    def test_fname_matches(self, dataset):
        _, _, fname = dataset[0]
        assert fname == "test.png"


# ── RealXrayDataset._augment ─────────────────────────────────────────────────

class TestAugment:
    """Test the _augment method properties."""

    @pytest.fixture
    def dataset_for_aug(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "t.png"), img)
        csv_path = tmp_path / "gt.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "gt_angle", "note"])
            writer.writeheader()
            writer.writerow({"filename": "t.png", "gt_angle": "120.0", "note": ""})
        return RealXrayDataset(str(csv_path), str(img_dir), augment=True, aug_factor=5)

    def test_augment_output_shape(self, dataset_for_aug):
        gray = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        result = dataset_for_aug._augment(gray)
        assert result.shape == (256, 256)

    def test_augment_output_dtype(self, dataset_for_aug):
        gray = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        result = dataset_for_aug._augment(gray)
        assert result.dtype == np.uint8

    def test_augment_clipping(self, dataset_for_aug):
        gray = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        result = dataset_for_aug._augment(gray)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_augment_produces_different_results(self, dataset_for_aug):
        """Multiple augmentations should not all be identical."""
        gray = np.ones((256, 256), dtype=np.uint8) * 128
        results = [dataset_for_aug._augment(gray.copy()) for _ in range(10)]
        # At least some should differ
        diffs = sum(not np.array_equal(results[0], r) for r in results[1:])
        assert diffs > 0

    def test_augmented_item_differs_from_base(self, dataset_for_aug):
        """Index >= len(samples) triggers augmentation."""
        np.random.seed(42)
        tensor_base, _, _ = dataset_for_aug[0]  # base (no aug)
        # Items with idx >= len(samples) are augmented
        tensor_aug, _, _ = dataset_for_aug[1]  # augmented
        # They should differ (not exact same pixel values)
        assert not torch.allclose(tensor_base, tensor_aug, atol=1e-3)


# ── Normalization round-trip ─────────────────────────────────────────────────

class TestNormalizationRoundTrip:
    """Verify norm/denorm consistency for the angle range."""

    @staticmethod
    def normalize(angle_deg: float) -> float:
        return (angle_deg - RealXrayDataset.ANGLE_MIN) / \
               (RealXrayDataset.ANGLE_MAX - RealXrayDataset.ANGLE_MIN)

    @staticmethod
    def denormalize(norm_val: float) -> float:
        return norm_val * (RealXrayDataset.ANGLE_MAX - RealXrayDataset.ANGLE_MIN) + \
               RealXrayDataset.ANGLE_MIN

    def test_roundtrip_90(self):
        assert abs(self.denormalize(self.normalize(90.0)) - 90.0) < 1e-10

    def test_roundtrip_180(self):
        assert abs(self.denormalize(self.normalize(180.0)) - 180.0) < 1e-10

    def test_roundtrip_135(self):
        assert abs(self.denormalize(self.normalize(135.0)) - 135.0) < 1e-10

    def test_normalize_90_is_0(self):
        assert abs(self.normalize(90.0)) < 1e-10

    def test_normalize_180_is_1(self):
        assert abs(self.normalize(180.0) - 1.0) < 1e-10

    def test_normalize_midpoint(self):
        assert abs(self.normalize(135.0) - 0.5) < 1e-10

    def test_denormalize_0_is_90(self):
        assert abs(self.denormalize(0.0) - 90.0) < 1e-10

    def test_denormalize_1_is_180(self):
        assert abs(self.denormalize(1.0) - 180.0) < 1e-10

    @pytest.mark.parametrize("angle", [90.0, 100.0, 110.0, 120.0, 135.0, 150.0, 170.0, 180.0])
    def test_roundtrip_parametric(self, angle):
        assert abs(self.denormalize(self.normalize(angle)) - angle) < 1e-10


# ── _load_and_preprocess ─────────────────────────────────────────────────────

class TestLoadAndPreprocess:
    """Test image preprocessing pipeline."""

    @pytest.fixture
    def dataset_with_image(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        # Create a known image with bone-like region in center
        img = np.zeros((512, 512), dtype=np.uint8)
        img[100:400, 100:400] = 180  # bright center
        cv2.imwrite(str(img_dir / "bone.png"), img)
        csv_path = tmp_path / "gt.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "gt_angle", "note"])
            writer.writeheader()
            writer.writerow({"filename": "bone.png", "gt_angle": "120.0", "note": ""})
        return RealXrayDataset(str(csv_path), str(img_dir), augment=False)

    def test_output_shape_256x256(self, dataset_with_image):
        img_path = dataset_with_image.img_dir / "bone.png"
        result = dataset_with_image._load_and_preprocess(img_path)
        assert result.shape == (256, 256)

    def test_output_dtype_uint8(self, dataset_with_image):
        img_path = dataset_with_image.img_dir / "bone.png"
        result = dataset_with_image._load_and_preprocess(img_path)
        assert result.dtype == np.uint8

    def test_output_not_all_zeros(self, dataset_with_image):
        img_path = dataset_with_image.img_dir / "bone.png"
        result = dataset_with_image._load_and_preprocess(img_path)
        assert result.sum() > 0

    def test_raises_on_missing_image(self, dataset_with_image):
        with pytest.raises(FileNotFoundError):
            dataset_with_image._load_and_preprocess(Path("/nonexistent/img.png"))
