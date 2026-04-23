#!/usr/bin/env python3
"""
DRR Synthesis Shared Utilities
==============================
Common infrastructure extracted from ct_to_xray_{final,improved,direct}.py.

Provides:
  - Shared constants (paths, series, HU window, target size)
  - Volume loading (load_all_volumes)
  - Metrics (compute_dice, compute_ssim, compute_all_metrics, resize_to_match)
  - Bone segmentation and humerus/forearm separation (segment_and_split_bones)
  - Histogram matching (histogram_match)
  - Real X-ray loading (load_real_xray)
  - elbow_synth re-exports for convenience
"""

import os
import sys

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ── Paths and constants ──────────────────────────────────────────────────

ROOT = "~/develop/research/ElbowVision"
sys.path.insert(0, os.path.join(ROOT, "elbow-train"))

CT_DIR = os.path.join(
    ROOT, "data/raw_dicom/ct_volume",
    "ﾃｽﾄ 008_0009900008_20260310_108Y_F_000",
)

REAL_XRAY_LAT = os.path.join(ROOT, "data/real_xray/images/cr_008_3_52kVp.png")
REAL_XRAY_LAT2 = os.path.join(ROOT, "data/real_xray/images/008_LAT.png")

# Series number -> flexion angle (deg)
SERIES = [(4, 180.0), (8, 135.0), (12, 90.0)]

TARGET_SIZE = 128
HU_MIN, HU_MAX = 50, 1000


# ── Re-exports from elbow_synth (lazy) ───────────────────────────────────
# Importing elbow_synth at module level would fail on machines without the
# full runtime environment.  Each script already does its own import, so
# we only re-export for convenience in new code.

def _import_elbow_synth():
    """Lazy import helper -- call from functions that need elbow_synth."""
    import elbow_synth  # noqa: F811
    return elbow_synth


# ── Volume loading ────────────────────────────────────────────────────────

def load_all_volumes(ct_dir=CT_DIR, series=None, hu_min=HU_MIN,
                     hu_max=HU_MAX, target_size=TARGET_SIZE,
                     laterality='L', verbose=True):
    """
    Load CT volumes for all series and detect landmarks.

    Returns
    -------
    volumes : dict[float, dict]
        Keyed by flexion angle (e.g. 180.0, 135.0, 90.0).
        Each value has keys: vol, lm, voxel_mm, laterality.
    """
    es = _import_elbow_synth()
    if series is None:
        series = SERIES

    volumes = {}
    for sn, angle in series:
        if verbose:
            print(f"\n--- Series {sn}, flexion={angle} deg ---")
        vol, _spacing, lat, vox_mm = es.load_ct_volume(
            ct_dir, laterality=laterality, series_num=sn,
            hu_min=hu_min, hu_max=hu_max, target_size=target_size,
        )
        lm = es.auto_detect_landmarks(vol, laterality=lat)
        volumes[angle] = dict(vol=vol, lm=lm, voxel_mm=vox_mm, laterality=lat)
        if verbose:
            print(f"  Volume shape: {vol.shape}, voxel_mm: {vox_mm:.3f}")

    return volumes


# ── Metrics ───────────────────────────────────────────────────────────────

def resize_to_match(img, target):
    """Resize *img* to match *target* shape if they differ."""
    if img.shape != target.shape:
        return cv2.resize(img, (target.shape[1], target.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    return img


def compute_dice(pred, gt, method='otsu'):
    """
    Bone Dice coefficient between two grayscale DRR images.

    Parameters
    ----------
    method : str
        'otsu' (default) -- binarise with Otsu threshold.
        'fixed' -- binarise at intensity > 30.
    """
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    if method == 'otsu':
        _, gb = cv2.threshold(gt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, pb = cv2.threshold(pred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        gb = (gt > 30).astype(np.uint8) * 255
        pb = (pred > 30).astype(np.uint8) * 255
    inter = np.logical_and(gb > 0, pb > 0).sum()
    return 2.0 * inter / (np.sum(gb > 0) + np.sum(pb > 0) + 1e-8)


def compute_ssim(pred, gt):
    """SSIM between two grayscale DRR images (auto-resize)."""
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    return ssim(gt, pred, data_range=255)


def compute_all_metrics(pred, gt):
    """Return (ssim, dice) tuple."""
    return float(compute_ssim(pred, gt)), float(compute_dice(pred, gt))


# ── Bone segmentation & humerus/forearm split ─────────────────────────────

def segment_and_split_bones(vol, lm, bone_only=True, verbose=True):
    """
    Segment bone from a CT volume and split into humerus/forearm sub-volumes.

    Parameters
    ----------
    vol : ndarray (PD, AP, ML)
        Normalised CT volume (0-1 after HU window).
    lm : dict
        Landmark dict from auto_detect_landmarks (normalised coords).
    bone_only : bool
        If True, zero out soft tissue before splitting.  If False, keep
        full intensity (for combined-volume DRR in ct_to_xray_final.py).

    Returns
    -------
    result : dict with keys:
        vol_humerus, vol_forearm -- sub-volumes (PD, AP, ML)
        vol_bone -- bone-only volume (before split)
        bone_thresh -- threshold used
        jc_vox -- joint centre in voxel coords (ndarray shape (3,))
        joint_pd_idx -- PD index of joint centre
        humerus_weight, forearm_weight -- 1-D weight arrays (PD,)
    """
    pd_size, ap_size, ml_size = vol.shape

    # Joint centre in voxel coords
    jc = lm['joint_center']
    jc_vox = np.array([jc[0] * pd_size, jc[1] * ap_size, jc[2] * ml_size])
    joint_pd_idx = int(jc_vox[0])

    # Bone threshold (percentile-based)
    bone_vals = vol[vol > 0.01]
    if len(bone_vals) > 0:
        bone_thresh = float(np.percentile(bone_vals, 50))
    else:
        bone_thresh = 0.3
    bone_mask = vol > bone_thresh

    if verbose:
        print(f"  Bone threshold: {bone_thresh:.3f}")
        print(f"  Joint center PD={joint_pd_idx}/{pd_size}")

    # Smooth sigmoid transition zone
    blend_half = max(3, int(pd_size * 0.04))
    humerus_weight = np.zeros(pd_size, dtype=np.float32)
    for i in range(pd_size):
        if i < joint_pd_idx - blend_half:
            humerus_weight[i] = 1.0
        elif i > joint_pd_idx + blend_half:
            humerus_weight[i] = 0.0
        else:
            humerus_weight[i] = 0.5 * (1.0 - (i - joint_pd_idx) / blend_half)
    forearm_weight = 1.0 - humerus_weight

    vol_bone = vol * bone_mask.astype(np.float32)

    if bone_only:
        vol_humerus = vol_bone * humerus_weight[:, None, None]
        vol_forearm = vol_bone * forearm_weight[:, None, None]
    else:
        vol_humerus = vol * humerus_weight[:, None, None]
        vol_forearm = vol * forearm_weight[:, None, None]

    if verbose:
        print(f"  Humerus bone voxels: {int((vol_humerus > 0.01).sum())}")
        print(f"  Forearm bone voxels: {int((vol_forearm > 0.01).sum())}")

    return dict(
        vol_humerus=vol_humerus,
        vol_forearm=vol_forearm,
        vol_bone=vol_bone,
        bone_thresh=bone_thresh,
        bone_mask=bone_mask,
        jc_vox=jc_vox,
        joint_pd_idx=joint_pd_idx,
        humerus_weight=humerus_weight,
        forearm_weight=forearm_weight,
    )


# ── Histogram matching ────────────────────────────────────────────────────

def histogram_match(source, template):
    """Match the histogram of *source* to *template* (uint8 images)."""
    s_cdf = np.zeros(256, np.float64)
    t_cdf = np.zeros(256, np.float64)
    for i in range(256):
        s_cdf[i] = np.sum(source <= i)
        t_cdf[i] = np.sum(template <= i)
    s_cdf /= s_cdf[-1]
    t_cdf /= t_cdf[-1]
    mapping = np.array(
        [np.argmin(np.abs(s_cdf[i] - t_cdf)) for i in range(256)],
        dtype=np.uint8,
    )
    return mapping[source]


# ── Real X-ray loading ────────────────────────────────────────────────────

def load_real_xray(paths=None, verbose=True):
    """
    Try to load a real CR X-ray from a list of candidate paths.

    Returns
    -------
    img : ndarray or None
        Grayscale image, or None if no file found.
    """
    if paths is None:
        paths = [REAL_XRAY_LAT, REAL_XRAY_LAT2]
    for rpath in paths:
        if os.path.exists(rpath):
            img = cv2.imread(rpath, cv2.IMREAD_GRAYSCALE)
            if verbose:
                print(f"  Real CR: {rpath} ({img.shape})")
            return img
    if verbose:
        print("  No real CR X-ray found")
    return None
