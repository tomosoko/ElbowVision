"""
Unit tests for scripts/ct_to_xray_improved.py pure functions.

Since ct_to_xray_improved.py runs heavy top-level code (loads CT volumes, generates
DRRs), we extract the pure functions via AST/exec rather than importing the module.

Covered (7 functions, 50 tests):
  - to_mm(): normalized landmark dict → mm-scale coordinate arrays
  - to_voxel(): normalized landmark dict → voxel coordinate arrays
  - build_humerus_frame(): orthonormal humerus-fixed coordinate system
  - transform_to_frame(): transforms landmarks into frame
  - procrustes_align(): rigid registration via SVD
  - compute_rotation_axis_from_arc(): rotation axis from 3 arc points
  - build_rotation_around_axis(): rotation matrix around arbitrary axis
  - Module constants (FOREARM_NAMES, HUMERUS_NAMES)
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# ── Extract pure functions from the script without executing top-level code ──

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ct_to_xray_improved.py"
_source = _SCRIPT.read_text()

# Parse AST and extract only function defs and constant assignments we need
_tree = ast.parse(_source)

_FUNC_NAMES = {
    "to_mm", "to_voxel", "build_humerus_frame", "transform_to_frame",
    "procrustes_align", "compute_rotation_axis_from_arc",
    "build_rotation_around_axis",
}
_CONST_NAMES = {"FOREARM_NAMES", "HUMERUS_NAMES"}

_extracted_nodes = []
for node in _tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in _FUNC_NAMES:
        _extracted_nodes.append(node)
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in _CONST_NAMES:
                _extracted_nodes.append(node)

# Build a mini-module with just the functions + their deps
_mini = ast.Module(body=_extracted_nodes, type_ignores=[])
ast.fix_missing_locations(_mini)
_code = compile(_mini, str(_SCRIPT), "exec")

_ns = {"np": np, "__builtins__": __builtins__}
# Some functions need scipy
from scipy.optimize import least_squares  # noqa: E402
from scipy.spatial.transform import Rotation as _Rot  # noqa: E402

_ns["Rotation"] = _Rot
_ns["least_squares"] = least_squares
exec(_code, _ns)  # noqa: S102

to_mm = _ns["to_mm"]
to_voxel = _ns["to_voxel"]
build_humerus_frame = _ns["build_humerus_frame"]
transform_to_frame = _ns["transform_to_frame"]
procrustes_align = _ns["procrustes_align"]
compute_rotation_axis_from_arc = _ns["compute_rotation_axis_from_arc"]
build_rotation_around_axis = _ns["build_rotation_around_axis"]
FOREARM_NAMES = _ns["FOREARM_NAMES"]
HUMERUS_NAMES = _ns["HUMERUS_NAMES"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sample_landmarks_norm():
    """Sample normalized landmark dict."""
    return {
        "humerus_shaft": (0.25, 0.50, 0.50),
        "lateral_epicondyle": (0.50, 0.50, 0.75),
        "medial_epicondyle": (0.50, 0.50, 0.25),
        "forearm_shaft": (0.75, 0.50, 0.50),
        "radial_head": (0.60, 0.40, 0.70),
        "olecranon": (0.55, 0.70, 0.50),
        "joint_center": (0.50, 0.50, 0.50),
    }


def _sample_landmarks_mm():
    """Sample mm-scale landmarks for frame building (reasonable anatomy)."""
    return {
        "humerus_shaft": np.array([0.0, 80.0, 0.0]),
        "lateral_epicondyle": np.array([0.0, 0.0, 15.0]),
        "medial_epicondyle": np.array([0.0, 0.0, -15.0]),
        "forearm_shaft": np.array([0.0, -60.0, 0.0]),
        "radial_head": np.array([5.0, -10.0, 12.0]),
        "olecranon": np.array([-10.0, -5.0, 0.0]),
        "joint_center": np.array([0.0, 0.0, 0.0]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for to_mm()
# ═══════════════════════════════════════════════════════════════════════════════

class TestToMm:
    def test_basic_scaling(self):
        lm = {"pt": (0.5, 0.5, 0.5)}
        result = to_mm(lm, voxel_mm=2.0, vol_shape=(100, 80, 60))
        expected = np.array([0.5 * 100 * 2.0, 0.5 * 80 * 2.0, 0.5 * 60 * 2.0])
        np.testing.assert_allclose(result["pt"], expected)

    def test_zero_coordinates(self):
        lm = {"origin": (0.0, 0.0, 0.0)}
        result = to_mm(lm, voxel_mm=1.5, vol_shape=(128, 128, 128))
        np.testing.assert_allclose(result["origin"], [0.0, 0.0, 0.0])

    def test_unit_coordinates(self):
        lm = {"corner": (1.0, 1.0, 1.0)}
        result = to_mm(lm, voxel_mm=1.0, vol_shape=(64, 64, 64))
        np.testing.assert_allclose(result["corner"], [64.0, 64.0, 64.0])

    def test_multiple_landmarks(self):
        lm = _sample_landmarks_norm()
        result = to_mm(lm, voxel_mm=1.0, vol_shape=(128, 128, 128))
        assert len(result) == len(lm)
        for name in lm:
            assert name in result
            assert isinstance(result[name], np.ndarray)
            assert result[name].shape == (3,)

    def test_non_uniform_shape(self):
        lm = {"pt": (0.5, 0.5, 0.5)}
        result = to_mm(lm, voxel_mm=1.0, vol_shape=(100, 200, 300))
        np.testing.assert_allclose(result["pt"], [50.0, 100.0, 150.0])

    def test_voxel_mm_scaling(self):
        lm = {"pt": (0.5, 0.5, 0.5)}
        r1 = to_mm(lm, voxel_mm=1.0, vol_shape=(100, 100, 100))
        r2 = to_mm(lm, voxel_mm=2.0, vol_shape=(100, 100, 100))
        np.testing.assert_allclose(r2["pt"], 2.0 * r1["pt"])


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for to_voxel()
# ═══════════════════════════════════════════════════════════════════════════════

class TestToVoxel:
    def test_basic_scaling(self):
        lm = {"pt": (0.5, 0.5, 0.5)}
        result = to_voxel(lm, voxel_mm=2.0, vol_shape=(100, 80, 60))
        expected = np.array([0.5 * 100, 0.5 * 80, 0.5 * 60])
        np.testing.assert_allclose(result["pt"], expected)

    def test_zero_coordinates(self):
        lm = {"origin": (0.0, 0.0, 0.0)}
        result = to_voxel(lm, voxel_mm=1.0, vol_shape=(128, 128, 128))
        np.testing.assert_allclose(result["origin"], [0.0, 0.0, 0.0])

    def test_unit_coordinates(self):
        lm = {"corner": (1.0, 1.0, 1.0)}
        result = to_voxel(lm, voxel_mm=1.0, vol_shape=(64, 64, 64))
        np.testing.assert_allclose(result["corner"], [64.0, 64.0, 64.0])

    def test_voxel_mm_independent(self):
        """to_voxel should NOT depend on voxel_mm."""
        lm = {"pt": (0.5, 0.5, 0.5)}
        r1 = to_voxel(lm, voxel_mm=1.0, vol_shape=(100, 100, 100))
        r2 = to_voxel(lm, voxel_mm=3.0, vol_shape=(100, 100, 100))
        np.testing.assert_allclose(r1["pt"], r2["pt"])

    def test_to_mm_to_voxel_ratio(self):
        """to_mm = to_voxel * voxel_mm for isotropic voxels."""
        lm = {"pt": (0.3, 0.7, 0.5)}
        vox_mm = 2.5
        shape = (80, 80, 80)
        mm = to_mm(lm, vox_mm, shape)
        vox = to_voxel(lm, vox_mm, shape)
        np.testing.assert_allclose(mm["pt"], vox["pt"] * vox_mm)

    def test_multiple_landmarks(self):
        lm = _sample_landmarks_norm()
        result = to_voxel(lm, voxel_mm=1.0, vol_shape=(128, 128, 128))
        assert len(result) == len(lm)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for build_humerus_frame()
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildHumerusFrame:
    def test_returns_origin_and_rotation(self):
        lm = _sample_landmarks_mm()
        origin, R = build_humerus_frame(lm)
        assert origin.shape == (3,)
        assert R.shape == (3, 3)

    def test_origin_is_joint_center(self):
        lm = _sample_landmarks_mm()
        origin, _ = build_humerus_frame(lm)
        np.testing.assert_allclose(origin, lm["joint_center"])

    def test_rotation_orthonormal(self):
        lm = _sample_landmarks_mm()
        _, R = build_humerus_frame(lm)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_rotation_det_positive(self):
        """Rotation matrix should have det = +1 (proper rotation)."""
        lm = _sample_landmarks_mm()
        _, R = build_humerus_frame(lm)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_y_axis_along_humerus(self):
        """Y-axis (R[1]) should point from humerus_shaft toward joint_center."""
        lm = _sample_landmarks_mm()
        _, R = build_humerus_frame(lm)
        y_ax = R[1]
        expected_dir = lm["joint_center"] - lm["humerus_shaft"]
        expected_dir = expected_dir / np.linalg.norm(expected_dir)
        np.testing.assert_allclose(y_ax, expected_dir, atol=1e-10)

    def test_z_axis_along_epicondyle(self):
        """Z-axis should have component along lateral-medial epicondyle direction."""
        lm = _sample_landmarks_mm()
        _, R = build_humerus_frame(lm)
        z_ax = R[2]
        ml_dir = lm["lateral_epicondyle"] - lm["medial_epicondyle"]
        ml_dir = ml_dir / np.linalg.norm(ml_dir)
        assert abs(np.dot(z_ax, ml_dir)) > 0.9

    def test_right_hand_rule(self):
        """X = Y cross Z."""
        lm = _sample_landmarks_mm()
        _, R = build_humerus_frame(lm)
        x_ax = R[0]
        y_ax = R[1]
        z_ax = R[2]
        np.testing.assert_allclose(x_ax, np.cross(y_ax, z_ax), atol=1e-10)

    def test_axes_are_unit_vectors(self):
        lm = _sample_landmarks_mm()
        _, R = build_humerus_frame(lm)
        for i in range(3):
            assert np.isclose(np.linalg.norm(R[i]), 1.0, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for transform_to_frame()
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransformToFrame:
    def test_origin_maps_to_zero(self):
        lm = _sample_landmarks_mm()
        origin, R = build_humerus_frame(lm)
        result = transform_to_frame(lm, origin, R)
        np.testing.assert_allclose(result["joint_center"], [0, 0, 0], atol=1e-10)

    def test_preserves_distances(self):
        """Rigid transform preserves pairwise distances."""
        lm = _sample_landmarks_mm()
        origin, R = build_humerus_frame(lm)
        result = transform_to_frame(lm, origin, R)
        names = list(lm.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                d_orig = np.linalg.norm(lm[names[i]] - lm[names[j]])
                d_trans = np.linalg.norm(result[names[i]] - result[names[j]])
                assert np.isclose(d_orig, d_trans, atol=1e-10)

    def test_all_landmarks_transformed(self):
        lm = _sample_landmarks_mm()
        origin, R = build_humerus_frame(lm)
        result = transform_to_frame(lm, origin, R)
        assert set(result.keys()) == set(lm.keys())

    def test_identity_frame(self):
        """Identity rotation + zero origin = no change."""
        lm = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([4.0, 5.0, 6.0])}
        result = transform_to_frame(lm, np.zeros(3), np.eye(3))
        np.testing.assert_allclose(result["a"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result["b"], [4.0, 5.0, 6.0])

    def test_translation_only(self):
        lm = {"pt": np.array([10.0, 20.0, 30.0])}
        result = transform_to_frame(lm, np.array([5.0, 10.0, 15.0]), np.eye(3))
        np.testing.assert_allclose(result["pt"], [5.0, 10.0, 15.0])


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for procrustes_align()
# ═══════════════════════════════════════════════════════════════════════════════

class TestProcrustesAlign:
    def test_identity_alignment(self):
        """Same point set → R ≈ I, t ≈ 0."""
        pts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]], dtype=float)
        R, t = procrustes_align(pts, pts)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(t, [0, 0, 0], atol=1e-10)

    def test_pure_translation(self):
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        shift = np.array([10, 20, 30])
        tgt = src + shift
        R, t = procrustes_align(src, tgt)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(t, shift, atol=1e-10)

    def test_known_rotation_90deg_z(self):
        """90-deg rotation around Z-axis."""
        src = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=float)
        R_true = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        tgt = (R_true @ src.T).T
        R, t = procrustes_align(src, tgt)
        np.testing.assert_allclose(R, R_true, atol=1e-10)
        np.testing.assert_allclose(t, [0, 0, 0], atol=1e-10)

    def test_combined_rotation_translation(self):
        src = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
        angle = np.pi / 4
        R_true = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        t_true = np.array([5, -3, 7])
        tgt = (R_true @ src.T).T + t_true
        R, t = procrustes_align(src, tgt)
        reconstructed = (R @ src.T).T + t
        np.testing.assert_allclose(reconstructed, tgt, atol=1e-10)

    def test_rotation_is_proper(self):
        """Recovered R should have det = +1."""
        src = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 3, 4]], dtype=float)
        R_true = Rotation.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
        tgt = (R_true @ src.T).T + np.array([1, 2, 3])
        R, t = procrustes_align(src, tgt)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_reconstruction_error(self):
        """Residuals should be near zero for noise-free data."""
        np.random.seed(42)
        src = np.random.randn(10, 3)
        R_true = Rotation.from_euler('xyz', [10, 20, 30], degrees=True).as_matrix()
        t_true = np.array([1, 2, 3])
        tgt = (R_true @ src.T).T + t_true
        R, t = procrustes_align(src, tgt)
        reconstructed = (R @ src.T).T + t
        residuals = np.linalg.norm(reconstructed - tgt, axis=1)
        assert np.all(residuals < 1e-10)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for compute_rotation_axis_from_arc()
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeRotationAxisFromArc:
    def test_xy_plane_arc(self):
        """Points on XY plane → axis along Z."""
        pts = np.array([
            [10, 0, 0],
            [0, 10, 0],
            [-10, 0, 0],
        ], dtype=float)
        axis, center = compute_rotation_axis_from_arc(pts)
        assert abs(abs(axis[2]) - 1.0) < 1e-6
        assert abs(axis[0]) < 1e-6
        assert abs(axis[1]) < 1e-6

    def test_xz_plane_arc(self):
        """Points on XZ plane → axis along Y."""
        pts = np.array([
            [10, 0, 0],
            [0, 0, 10],
            [-10, 0, 0],
        ], dtype=float)
        axis, center = compute_rotation_axis_from_arc(pts)
        assert abs(abs(axis[1]) - 1.0) < 1e-6

    def test_axis_is_unit_vector(self):
        pts = np.array([
            [10, 0, 0],
            [0, 10, 0],
            [-10, 0, 0],
        ], dtype=float)
        axis, _ = compute_rotation_axis_from_arc(pts)
        assert np.isclose(np.linalg.norm(axis), 1.0, atol=1e-10)

    def test_center_equidistant(self):
        """Circle center should be equidistant from all 3 points."""
        angles = [0, np.pi / 3, 2 * np.pi / 3]
        pts = np.array([[5 * np.cos(a), 5 * np.sin(a), 0] for a in angles])
        _, center = compute_rotation_axis_from_arc(pts)
        distances = [np.linalg.norm(pts[i] - center) for i in range(3)]
        np.testing.assert_allclose(distances[0], distances[1], atol=1e-6)
        np.testing.assert_allclose(distances[1], distances[2], atol=1e-6)

    def test_known_radius(self):
        """Circle of known radius 5 → center distances should be 5."""
        angles = [0, np.pi / 3, 2 * np.pi / 3]
        pts = np.array([[5 * np.cos(a), 5 * np.sin(a), 0] for a in angles])
        _, center = compute_rotation_axis_from_arc(pts)
        for i in range(3):
            assert np.isclose(np.linalg.norm(pts[i] - center), 5.0, atol=1e-6)

    def test_collinear_fallback(self):
        """Collinear points → fallback without crash."""
        pts = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
        ], dtype=float)
        axis, center = compute_rotation_axis_from_arc(pts)
        assert axis.shape == (3,)

    def test_tilted_plane_arc(self):
        """Arc in a tilted plane → axis perpendicular to that plane."""
        angles = [0, np.pi / 3, 2 * np.pi / 3]
        c45 = np.cos(np.pi / 4)
        s45 = np.sin(np.pi / 4)
        pts = np.array([
            [10 * np.cos(a), 10 * np.sin(a) * c45, 10 * np.sin(a) * s45]
            for a in angles
        ])
        axis, _ = compute_rotation_axis_from_arc(pts)
        expected = np.array([0, -s45, c45])
        dot = abs(np.dot(axis, expected))
        assert dot > 0.99

    def test_large_radius_arc(self):
        """Large radius circle still works."""
        R = 1000.0
        pts = np.array([
            [R, 0, 0],
            [R * np.cos(0.01), R * np.sin(0.01), 0],
            [R * np.cos(0.02), R * np.sin(0.02), 0],
        ])
        axis, center = compute_rotation_axis_from_arc(pts)
        assert abs(abs(axis[2]) - 1.0) < 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for build_rotation_around_axis()
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildRotationAroundAxis:
    def test_zero_rotation_identity(self):
        axis = np.array([0, 0, 1.0])
        center = np.array([10, 20, 30.0])
        R_inv, offset = build_rotation_around_axis(axis, 0.0, center)
        np.testing.assert_allclose(R_inv, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(offset, [0, 0, 0], atol=1e-10)

    def test_360_rotation_identity(self):
        axis = np.array([0, 0, 1.0])
        center = np.array([5, 5, 5.0])
        R_inv, offset = build_rotation_around_axis(axis, 360.0, center)
        np.testing.assert_allclose(R_inv, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(offset, [0, 0, 0], atol=1e-10)

    def test_center_is_fixed(self):
        """The center point should map to itself under the rotation."""
        axis = np.array([1, 1, 1.0]) / np.sqrt(3)
        center = np.array([10, 20, 30.0])
        R_inv, offset = build_rotation_around_axis(axis, 45.0, center)
        mapped = R_inv @ center + offset
        np.testing.assert_allclose(mapped, center, atol=1e-10)

    def test_rotation_preserves_distance_from_axis(self):
        """Points should remain the same distance from the rotation axis."""
        axis = np.array([0, 0, 1.0])
        center = np.array([0, 0, 0.0])
        R_inv, offset = build_rotation_around_axis(axis, 90.0, center)
        pt = np.array([5.0, 0.0, 3.0])
        mapped = R_inv @ pt + offset
        dist_orig = np.sqrt(pt[0] ** 2 + pt[1] ** 2)
        dist_mapped = np.sqrt(mapped[0] ** 2 + mapped[1] ** 2)
        assert np.isclose(dist_orig, dist_mapped, atol=1e-10)

    def test_90deg_z_axis(self):
        """90-deg rotation around Z at origin: verify R_fwd maps (1,0,0) → (0,1,0)."""
        axis = np.array([0, 0, 1.0])
        center = np.array([0, 0, 0.0])
        R_inv, offset = build_rotation_around_axis(axis, 90.0, center)
        R_fwd = R_inv.T
        pt = np.array([1.0, 0.0, 0.0])
        rotated = R_fwd @ pt
        np.testing.assert_allclose(rotated, [0, 1, 0], atol=1e-10)

    def test_180_deg_double_is_360(self):
        """Applying 180-deg rotation twice gives identity."""
        axis = np.array([0, 1, 0.0])
        center = np.array([5, 5, 5.0])
        R_inv1, off1 = build_rotation_around_axis(axis, 180.0, center)
        R_inv_total = R_inv1 @ R_inv1
        off_total = R_inv1 @ off1 + off1
        np.testing.assert_allclose(R_inv_total, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(off_total, [0, 0, 0], atol=1e-10)

    def test_negative_angle(self):
        """Negative angle should be opposite of positive."""
        axis = np.array([0, 0, 1.0])
        center = np.array([0, 0, 0.0])
        R_pos, _ = build_rotation_around_axis(axis, 30.0, center)
        R_neg, _ = build_rotation_around_axis(axis, -30.0, center)
        product = R_pos.T @ R_neg.T
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    def test_arbitrary_axis_orthonormal(self):
        """Rotation around arbitrary axis should still be orthonormal."""
        axis = np.array([1, 2, 3.0])
        axis = axis / np.linalg.norm(axis)
        center = np.array([10, 20, 30.0])
        R_inv, offset = build_rotation_around_axis(axis, 73.0, center)
        np.testing.assert_allclose(R_inv @ R_inv.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R_inv), 1.0, atol=1e-10)

    def test_axis_aligned_point_unchanged(self):
        """A point on the rotation axis should remain fixed."""
        axis = np.array([0, 0, 1.0])
        center = np.array([3, 4, 0.0])
        R_inv, offset = build_rotation_around_axis(axis, 60.0, center)
        pt_on_axis = center + 10.0 * axis
        mapped = R_inv @ pt_on_axis + offset
        np.testing.assert_allclose(mapped, pt_on_axis, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

class TestModuleConstants:
    def test_forearm_names(self):
        assert FOREARM_NAMES == ["forearm_shaft", "radial_head", "olecranon"]

    def test_humerus_names(self):
        expected = ["humerus_shaft", "lateral_epicondyle", "medial_epicondyle",
                     "joint_center"]
        assert HUMERUS_NAMES == expected
