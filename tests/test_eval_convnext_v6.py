"""Tests for scripts/eval_convnext_v6.py

Covers:
  - Module constants (MODEL_PATH, VAL_CSV, IMG_DIR, OUT_DIR paths)
  - VAL_TRANSFORMS: pipeline structure and output shape/dtype
  - ValDataset: LAT filtering, __len__, __getitem__, fallback path
  - compute_bland_altman_stats: dict keys and value propagation
  - save_bland_altman_plot: file creation and matplotlib calls
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Mock heavy deps before importing ─────────────────────────────────────────

# Mock torch, torchvision, PIL at minimum to avoid GPU/model loading
_mocks_installed: dict[str, types.ModuleType | None] = {}


def _ensure_mock(name: str, mock_mod: types.ModuleType | None = None):
    _mocks_installed[name] = sys.modules.get(name)
    if mock_mod is None:
        mock_mod = types.ModuleType(name)
    sys.modules[name] = mock_mod


# convnext_model is imported lazily inside load_model, mock it
_mock_convnext = types.ModuleType("convnext_model")
_mock_convnext.ElbowConvNeXt = MagicMock  # type: ignore[attr-defined]
_ensure_mock("convnext_model", _mock_convnext)

# bland_altman is imported lazily inside compute_bland_altman_stats
_mock_ba = types.ModuleType("bland_altman")


class _MockBAResult:
    def __init__(self, n=100, mean_diff=0.1, std_diff=0.5, loa_upper=1.1,
                 loa_lower=-0.9, mae=0.3, rmse=0.4, pearson_r=0.99,
                 r_squared=0.98, icc=0.995):
        self.n = n
        self.mean_diff = mean_diff
        self.std_diff = std_diff
        self.loa_upper = loa_upper
        self.loa_lower = loa_lower
        self.mae = mae
        self.rmse = rmse
        self.pearson_r = pearson_r
        self.r_squared = r_squared
        self.icc = icc


_mock_ba.compute_bland_altman = lambda gt, pred: _MockBAResult(n=len(gt))  # type: ignore[attr-defined]
_ensure_mock("bland_altman", _mock_ba)

# Make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import eval_convnext_v6 as ev6

# Restore mocked modules
for name, orig in _mocks_installed.items():
    if orig is not None:
        sys.modules[name] = orig
    else:
        sys.modules.pop(name, None)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_val_df(n_lat: int = 5, n_ap: int = 3) -> pd.DataFrame:
    """Create a minimal DataFrame mimicking convnext_labels.csv."""
    rows = []
    for i in range(n_lat):
        rows.append({
            "filename": f"lat_{i:03d}.png",
            "view_type": "LAT",
            "flexion_deg": 90.0 + i * 5.0,
            "split": "val",
        })
    for i in range(n_ap):
        rows.append({
            "filename": f"ap_{i:03d}.png",
            "view_type": "AP",
            "flexion_deg": 0.0,
            "split": "val",
        })
    return pd.DataFrame(rows)


def _make_dummy_image(tmp_path: Path, name: str, subdir: str = "val") -> Path:
    """Create a tiny 8x8 RGB PNG for testing."""
    from PIL import Image
    d = tmp_path / subdir
    d.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8))
    p = d / name
    img.save(str(p))
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstants:
    """Module-level constants and paths."""

    def test_project_root_is_parent_of_scripts(self):
        assert ev6.PROJECT_ROOT == PROJECT_ROOT

    def test_model_path_under_elbow_api(self):
        assert "elbow-api" in str(ev6.MODEL_PATH)
        assert ev6.MODEL_PATH.name == "elbow_convnext_best.pth"

    def test_val_csv_path(self):
        assert "yolo_dataset_v6" in str(ev6.VAL_CSV)
        assert ev6.VAL_CSV.name == "convnext_labels.csv"

    def test_img_dir_path(self):
        assert "yolo_dataset_v6" in str(ev6.IMG_DIR)
        assert ev6.IMG_DIR.name == "images"

    def test_out_dir_path(self):
        assert "bland_altman" in str(ev6.OUT_DIR)

    def test_paths_are_absolute(self):
        for p in [ev6.MODEL_PATH, ev6.VAL_CSV, ev6.IMG_DIR, ev6.OUT_DIR]:
            assert p.is_absolute(), f"{p} is not absolute"


# ═══════════════════════════════════════════════════════════════════════════════
# VAL_TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════════


class TestValTransforms:
    """Validate the torchvision transform pipeline."""

    def test_transforms_is_compose(self):
        from torchvision.transforms import Compose
        assert isinstance(ev6.VAL_TRANSFORMS, Compose)

    def test_transforms_has_four_steps(self):
        assert len(ev6.VAL_TRANSFORMS.transforms) == 4

    def test_transforms_starts_with_resize(self):
        from torchvision.transforms import Resize
        assert isinstance(ev6.VAL_TRANSFORMS.transforms[0], Resize)

    def test_resize_target_is_232(self):
        t = ev6.VAL_TRANSFORMS.transforms[0]
        # Resize stores size as int or sequence
        size = t.size if isinstance(t.size, int) else t.size[0]
        assert size == 232

    def test_center_crop_224(self):
        from torchvision.transforms import CenterCrop
        t = ev6.VAL_TRANSFORMS.transforms[1]
        assert isinstance(t, CenterCrop)
        size = t.size if isinstance(t.size, int) else t.size[0]
        assert size == 224

    def test_has_to_tensor(self):
        from torchvision.transforms import ToTensor
        assert isinstance(ev6.VAL_TRANSFORMS.transforms[2], ToTensor)

    def test_has_normalize(self):
        from torchvision.transforms import Normalize
        assert isinstance(ev6.VAL_TRANSFORMS.transforms[3], Normalize)

    def test_normalize_imagenet_mean(self):
        t = ev6.VAL_TRANSFORMS.transforms[3]
        expected = [0.485, 0.456, 0.406]
        np.testing.assert_allclose(list(t.mean), expected, atol=1e-6)

    def test_normalize_imagenet_std(self):
        t = ev6.VAL_TRANSFORMS.transforms[3]
        expected = [0.229, 0.224, 0.225]
        np.testing.assert_allclose(list(t.std), expected, atol=1e-6)

    def test_output_shape_rgb(self):
        """Transform a dummy PIL image -> tensor [3, 224, 224]."""
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        out = ev6.VAL_TRANSFORMS(img)
        assert out.shape == (3, 224, 224)

    def test_output_dtype_float(self):
        import torch
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        out = ev6.VAL_TRANSFORMS(img)
        assert out.dtype == torch.float32

    def test_output_is_normalized(self):
        """After ImageNet normalize, black image (0) yields negative values."""
        from PIL import Image
        img = Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8))
        out = ev6.VAL_TRANSFORMS(img)
        # ToTensor maps 0->0.0, then Normalize subtracts mean -> negative
        assert out.min().item() < 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# ValDataset
# ═══════════════════════════════════════════════════════════════════════════════


class TestValDataset:
    """ValDataset: LAT filtering, length, __getitem__."""

    def test_filters_lat_only(self):
        df = _make_val_df(n_lat=5, n_ap=3)
        ds = ev6.ValDataset(df, Path("/dummy"))
        assert len(ds) == 5

    def test_filters_case_insensitive(self):
        df = _make_val_df(n_lat=3, n_ap=2)
        # Mix case
        df.loc[0, "view_type"] = "lat"
        df.loc[1, "view_type"] = "Lat"
        ds = ev6.ValDataset(df, Path("/dummy"))
        assert len(ds) == 3

    def test_empty_when_no_lat(self):
        df = _make_val_df(n_lat=0, n_ap=5)
        ds = ev6.ValDataset(df, Path("/dummy"))
        assert len(ds) == 0

    def test_all_lat(self):
        df = _make_val_df(n_lat=10, n_ap=0)
        ds = ev6.ValDataset(df, Path("/dummy"))
        assert len(ds) == 10

    def test_getitem_returns_three_elements(self, tmp_path):
        df = _make_val_df(n_lat=2, n_ap=0)
        _make_dummy_image(tmp_path, "lat_000.png", subdir="val")
        ds = ev6.ValDataset(df, tmp_path)
        img, gt, fname = ds[0]
        assert img is not None
        assert isinstance(gt, float)
        assert isinstance(fname, str)

    def test_getitem_tensor_shape(self, tmp_path):
        df = _make_val_df(n_lat=1, n_ap=0)
        _make_dummy_image(tmp_path, "lat_000.png", subdir="val")
        ds = ev6.ValDataset(df, tmp_path)
        img, _, _ = ds[0]
        assert img.shape == (3, 224, 224)

    def test_getitem_gt_value(self, tmp_path):
        df = _make_val_df(n_lat=2, n_ap=0)
        _make_dummy_image(tmp_path, "lat_000.png", subdir="val")
        ds = ev6.ValDataset(df, tmp_path)
        _, gt, _ = ds[0]
        assert gt == pytest.approx(90.0)

    def test_getitem_filename(self, tmp_path):
        df = _make_val_df(n_lat=2, n_ap=0)
        _make_dummy_image(tmp_path, "lat_000.png", subdir="val")
        ds = ev6.ValDataset(df, tmp_path)
        _, _, fname = ds[0]
        assert fname == "lat_000.png"

    def test_getitem_fallback_path(self, tmp_path):
        """If images/<split>/<filename> doesn't exist, try images/<filename>."""
        df = _make_val_df(n_lat=1, n_ap=0)
        # Put image directly in img_dir (no val/ subdir)
        _make_dummy_image(tmp_path, "lat_000.png", subdir=".")
        # Rename so the "." subdir file is directly under tmp_path
        ds = ev6.ValDataset(df, tmp_path)
        img, gt, fname = ds[0]
        assert img.shape == (3, 224, 224)

    def test_split_column_default_val(self, tmp_path):
        """When 'split' column is missing, default to 'val'."""
        df = _make_val_df(n_lat=1, n_ap=0)
        df = df.drop(columns=["split"])
        _make_dummy_image(tmp_path, "lat_000.png", subdir="val")
        ds = ev6.ValDataset(df, tmp_path)
        img, _, _ = ds[0]
        assert img.shape == (3, 224, 224)

    def test_index_reset(self):
        """After filtering, index should be 0-based."""
        df = _make_val_df(n_lat=3, n_ap=5)
        ds = ev6.ValDataset(df, Path("/dummy"))
        assert ds.df.index.tolist() == [0, 1, 2]

    def test_preserves_flexion_values(self):
        df = _make_val_df(n_lat=4, n_ap=0)
        ds = ev6.ValDataset(df, Path("/dummy"))
        expected = [90.0, 95.0, 100.0, 105.0]
        actual = ds.df["flexion_deg"].tolist()
        assert actual == expected


# ═══════════════════════════════════════════════════════════════════════════════
# compute_bland_altman_stats
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeBlandAltmanStats:
    """compute_bland_altman_stats: dict structure and value propagation."""

    def _call(self, n: int = 50):
        gt = np.linspace(60, 180, n)
        pred = gt + np.random.normal(0, 0.5, n)
        # Re-mock bland_altman for this test
        with patch.dict(sys.modules, {"bland_altman": _mock_ba}):
            return ev6.compute_bland_altman_stats(gt, pred)

    def test_returns_dict(self):
        result = self._call()
        assert isinstance(result, dict)

    def test_has_all_expected_keys(self):
        result = self._call()
        expected_keys = {
            "n", "mean_bias", "sd_diff", "loa_upper", "loa_lower",
            "mae", "rmse", "pearson_r", "r_squared", "icc_31",
        }
        assert set(result.keys()) == expected_keys

    def test_n_matches_input_length(self):
        result = self._call(n=75)
        assert result["n"] == 75

    def test_mean_bias_is_float(self):
        result = self._call()
        assert isinstance(result["mean_bias"], float)

    def test_sd_diff_is_float(self):
        result = self._call()
        assert isinstance(result["sd_diff"], float)

    def test_loa_upper_is_float(self):
        result = self._call()
        assert isinstance(result["loa_upper"], float)

    def test_loa_lower_is_float(self):
        result = self._call()
        assert isinstance(result["loa_lower"], float)

    def test_mae_is_float(self):
        result = self._call()
        assert isinstance(result["mae"], float)

    def test_rmse_is_float(self):
        result = self._call()
        assert isinstance(result["rmse"], float)

    def test_pearson_r_is_float(self):
        result = self._call()
        assert isinstance(result["pearson_r"], float)

    def test_r_squared_is_float(self):
        result = self._call()
        assert isinstance(result["r_squared"], float)

    def test_icc_31_is_float(self):
        result = self._call()
        assert isinstance(result["icc_31"], float)

    def test_loa_upper_greater_than_lower(self):
        result = self._call()
        assert result["loa_upper"] > result["loa_lower"]

    def test_default_mock_values_propagated(self):
        """Verify mock result values are correctly mapped to dict keys."""
        gt = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.1, 2.1, 3.1])
        with patch.dict(sys.modules, {"bland_altman": _mock_ba}):
            result = ev6.compute_bland_altman_stats(gt, pred)
        assert result["mean_bias"] == pytest.approx(0.1)
        assert result["sd_diff"] == pytest.approx(0.5)
        assert result["icc_31"] == pytest.approx(0.995)


# ═══════════════════════════════════════════════════════════════════════════════
# save_bland_altman_plot
# ═══════════════════════════════════════════════════════════════════════════════


class TestSaveBlandAltmanPlot:
    """save_bland_altman_plot: file output and matplotlib usage."""

    @pytest.fixture()
    def _sample_data(self):
        gt = np.linspace(60, 180, 50)
        pred = gt + np.random.normal(0, 1.0, 50)
        stats = {
            "n": 50,
            "mean_bias": 0.1,
            "sd_diff": 0.9,
            "loa_upper": 1.86,
            "loa_lower": -1.66,
            "mae": 0.7,
            "rmse": 0.9,
            "pearson_r": 0.999,
            "r_squared": 0.998,
            "icc_31": 0.997,
        }
        return gt, pred, stats

    def test_creates_output_file(self, tmp_path, _sample_data):
        gt, pred, stats = _sample_data
        out = tmp_path / "test_plot.png"
        ev6.save_bland_altman_plot(gt, pred, stats, out)
        assert out.exists()

    def test_output_file_nonzero(self, tmp_path, _sample_data):
        gt, pred, stats = _sample_data
        out = tmp_path / "test_plot.png"
        ev6.save_bland_altman_plot(gt, pred, stats, out)
        assert out.stat().st_size > 0

    def test_output_is_valid_png(self, tmp_path, _sample_data):
        gt, pred, stats = _sample_data
        out = tmp_path / "test_plot.png"
        ev6.save_bland_altman_plot(gt, pred, stats, out)
        # PNG magic bytes
        with open(out, "rb") as f:
            magic = f.read(4)
        assert magic[1:4] == b"PNG"

    def test_works_with_single_point(self, tmp_path):
        gt = np.array([90.0])
        pred = np.array([91.0])
        stats = {
            "n": 1, "mean_bias": 1.0, "sd_diff": 0.0,
            "loa_upper": 1.0, "loa_lower": 1.0,
            "mae": 1.0, "rmse": 1.0,
            "pearson_r": 1.0, "r_squared": 1.0, "icc_31": 1.0,
        }
        out = tmp_path / "single.png"
        ev6.save_bland_altman_plot(gt, pred, stats, out)
        assert out.exists()

    def test_works_with_large_n(self, tmp_path):
        n = 500
        gt = np.linspace(60, 180, n)
        pred = gt + np.random.normal(0, 0.5, n)
        stats = {
            "n": n, "mean_bias": 0.01, "sd_diff": 0.5,
            "loa_upper": 0.99, "loa_lower": -0.97,
            "mae": 0.4, "rmse": 0.5,
            "pearson_r": 0.9999, "r_squared": 0.9998, "icc_31": 0.9997,
        }
        out = tmp_path / "large.png"
        ev6.save_bland_altman_plot(gt, pred, stats, out)
        assert out.exists()

    def test_negative_bias_renders(self, tmp_path, _sample_data):
        gt, pred, stats = _sample_data
        stats["mean_bias"] = -2.5
        out = tmp_path / "neg_bias.png"
        ev6.save_bland_altman_plot(gt, pred, stats, out)
        assert out.exists()

    def test_output_to_subdirectory(self, tmp_path, _sample_data):
        gt, pred, stats = _sample_data
        subdir = tmp_path / "sub" / "dir"
        subdir.mkdir(parents=True)
        out = subdir / "nested.png"
        ev6.save_bland_altman_plot(gt, pred, stats, out)
        assert out.exists()

    def test_identical_gt_pred(self, tmp_path):
        """Perfect agreement should still render without error."""
        gt = np.linspace(60, 180, 30)
        pred = gt.copy()
        stats = {
            "n": 30, "mean_bias": 0.0, "sd_diff": 0.0,
            "loa_upper": 0.0, "loa_lower": 0.0,
            "mae": 0.0, "rmse": 0.0,
            "pearson_r": 1.0, "r_squared": 1.0, "icc_31": 1.0,
        }
        out = tmp_path / "perfect.png"
        ev6.save_bland_altman_plot(gt, pred, stats, out)
        assert out.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# load_model
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadModel:
    """load_model: checkpoint loading branches."""

    def test_loads_model_state_dict_key(self, tmp_path):
        """When checkpoint has 'model_state_dict' key, use it."""
        import torch
        # Create a minimal dummy checkpoint
        dummy_state = {"dummy.weight": torch.zeros(1)}
        ckpt_path = tmp_path / "model.pth"
        torch.save({"model_state_dict": dummy_state}, str(ckpt_path))

        mock_model = MagicMock()
        mock_class = MagicMock(return_value=mock_model)

        with patch.dict(sys.modules, {"convnext_model": types.ModuleType("convnext_model")}):
            sys.modules["convnext_model"].ElbowConvNeXt = mock_class  # type: ignore[attr-defined]
            result = ev6.load_model(ckpt_path, torch.device("cpu"))

        mock_model.load_state_dict.assert_called_once_with(dummy_state)
        mock_model.eval.assert_called_once()

    def test_loads_raw_state_dict(self, tmp_path):
        """When checkpoint has no 'model_state_dict' key, load directly."""
        import torch
        dummy_state = {"dummy.weight": torch.zeros(1)}
        ckpt_path = tmp_path / "model.pth"
        torch.save(dummy_state, str(ckpt_path))

        mock_model = MagicMock()
        mock_class = MagicMock(return_value=mock_model)

        with patch.dict(sys.modules, {"convnext_model": types.ModuleType("convnext_model")}):
            sys.modules["convnext_model"].ElbowConvNeXt = mock_class  # type: ignore[attr-defined]
            result = ev6.load_model(ckpt_path, torch.device("cpu"))

        mock_model.load_state_dict.assert_called_once_with(dummy_state)

    def test_model_set_to_eval(self, tmp_path):
        import torch
        ckpt_path = tmp_path / "model.pth"
        torch.save({"model_state_dict": {}}, str(ckpt_path))

        mock_model = MagicMock()
        mock_class = MagicMock(return_value=mock_model)

        with patch.dict(sys.modules, {"convnext_model": types.ModuleType("convnext_model")}):
            sys.modules["convnext_model"].ElbowConvNeXt = mock_class  # type: ignore[attr-defined]
            ev6.load_model(ckpt_path, torch.device("cpu"))

        mock_model.eval.assert_called_once()

    def test_model_moved_to_device(self, tmp_path):
        import torch
        ckpt_path = tmp_path / "model.pth"
        torch.save({"model_state_dict": {}}, str(ckpt_path))

        mock_model = MagicMock()
        mock_class = MagicMock(return_value=mock_model)

        with patch.dict(sys.modules, {"convnext_model": types.ModuleType("convnext_model")}):
            sys.modules["convnext_model"].ElbowConvNeXt = mock_class  # type: ignore[attr-defined]
            ev6.load_model(ckpt_path, torch.device("cpu"))

        mock_model.to.assert_called_once_with(torch.device("cpu"))

    def test_pretrained_false(self, tmp_path):
        import torch
        ckpt_path = tmp_path / "model.pth"
        torch.save({"model_state_dict": {}}, str(ckpt_path))

        mock_model = MagicMock()
        mock_class = MagicMock(return_value=mock_model)

        with patch.dict(sys.modules, {"convnext_model": types.ModuleType("convnext_model")}):
            sys.modules["convnext_model"].ElbowConvNeXt = mock_class  # type: ignore[attr-defined]
            ev6.load_model(ckpt_path, torch.device("cpu"))

        mock_class.assert_called_once_with(pretrained=False)
