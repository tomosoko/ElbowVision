"""
Unit tests for elbow-train/create_phantom.py pure functions.

Covers:
- Constants (NX, NY, NZ, PX, PY, PZ, HU, CX, CY)
- _init_grid(): global grid initialization
- cyl(): Z-axis elliptical cylinder mask (3D)
- ell(): ellipsoid mask (3D)
- ell2(): ellipse mask (2D slice)
- shell(): cortical bone shell + fill
- build_phantom(): full phantom volume generation
- write_dicom_series(): DICOM output (mocked I/O)
"""

import sys
import os
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

# Mock pydicom before import (heavy dependency)
_pydicom_mock = MagicMock()
_pydicom_mock.uid.generate_uid = MagicMock(return_value="1.2.3.4.5")
_pydicom_mock.dataset.Dataset = MagicMock
_pydicom_mock.dataset.FileDataset = MagicMock(return_value=MagicMock())
_pydicom_mock.dataset.FileMetaDataset = MagicMock(return_value=MagicMock())
sys.modules.setdefault("pydicom", _pydicom_mock)
sys.modules.setdefault("pydicom.dataset", _pydicom_mock.dataset)
sys.modules.setdefault("pydicom.uid", _pydicom_mock.uid)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "elbow-train"))

from create_phantom import (
    NX, NY, NZ, PX, PY, PZ, HU, CX, CY,
    _init_grid, cyl, ell, ell2, shell, build_phantom,
)


# ============================================================================
# Constants
# ============================================================================

class TestConstants:
    """Test volume and HU constant definitions."""

    def test_volume_dimensions(self):
        assert NX == 256
        assert NY == 256
        assert NZ == 180

    def test_voxel_spacing(self):
        assert PX == 0.5
        assert PY == 0.5
        assert PZ == 1.0

    def test_center_coordinates(self):
        assert CX == NX // 2  # 128
        assert CY == NY // 2  # 128

    def test_hu_air(self):
        assert HU['air'] == -1000

    def test_hu_fat(self):
        assert HU['fat'] == -80

    def test_hu_muscle(self):
        assert HU['muscle'] == 50

    def test_hu_cancellous_bone(self):
        assert HU['cancel'] == 350

    def test_hu_cortical_bone(self):
        assert HU['cortex'] == 800

    def test_hu_marrow(self):
        assert HU['marrow'] == -80

    def test_hu_ordering(self):
        """HU values should follow air < fat <= marrow < muscle < cancel < cortex."""
        assert HU['air'] < HU['fat']
        assert HU['fat'] <= HU['marrow']
        assert HU['marrow'] < HU['muscle']
        assert HU['muscle'] < HU['cancel']
        assert HU['cancel'] < HU['cortex']

    def test_physical_volume_size_mm(self):
        """Physical volume size in mm."""
        assert NX * PX == 128.0  # 256 * 0.5
        assert NY * PY == 128.0
        assert NZ * PZ == 180.0


# ============================================================================
# _init_grid
# ============================================================================

class TestInitGrid:
    """Test global grid initialization."""

    def test_init_grid_creates_3d_grids(self):
        import create_phantom as mod
        _init_grid()
        assert mod._KK is not None
        assert mod._II is not None
        assert mod._JJ is not None

    def test_init_grid_creates_2d_grids(self):
        import create_phantom as mod
        _init_grid()
        assert mod._ii2 is not None
        assert mod._jj2 is not None

    def test_3d_grid_shapes(self):
        import create_phantom as mod
        _init_grid()
        assert mod._KK.shape == (NZ, NY, NX)
        assert mod._II.shape == (NZ, NY, NX)
        assert mod._JJ.shape == (NZ, NY, NX)

    def test_2d_grid_shapes(self):
        import create_phantom as mod
        _init_grid()
        assert mod._ii2.shape == (NY, NX)
        assert mod._jj2.shape == (NY, NX)

    def test_3d_grid_dtype(self):
        import create_phantom as mod
        _init_grid()
        assert mod._KK.dtype == np.float32


# ============================================================================
# cyl (elliptical cylinder mask)
# ============================================================================

class TestCyl:
    """Test Z-axis elliptical cylinder mask generation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _init_grid()

    def test_returns_bool_array(self):
        mask = cyl(CX, CY, 10, 10, 0, NZ - 1)
        assert mask.dtype == bool

    def test_shape_matches_volume(self):
        mask = cyl(CX, CY, 10, 10, 0, NZ - 1)
        assert mask.shape == (NZ, NY, NX)

    def test_center_voxel_inside(self):
        mask = cyl(CX, CY, 10, 10, 0, NZ - 1)
        assert mask[NZ // 2, CY, CX] is np.True_

    def test_far_corner_outside(self):
        mask = cyl(CX, CY, 10, 10, 0, NZ - 1)
        assert mask[0, 0, 0] is np.False_

    def test_z_range_limits(self):
        """Cylinder should only exist within k0..k1 range."""
        mask = cyl(CX, CY, 50, 50, 50, 100)
        # Inside Z range at center
        assert mask[75, CY, CX] is np.True_
        # Outside Z range at center
        assert mask[10, CY, CX] is np.False_
        assert mask[150, CY, CX] is np.False_

    def test_elliptical_cross_section(self):
        """Different radii in j and i should produce elliptical shape."""
        rj, ri = 40, 20
        mask = cyl(CX, CY, rj, ri, 0, NZ - 1)
        mid_k = NZ // 2
        # Point at edge of j-radius, center i -> inside
        assert mask[mid_k, CY, CX + rj - 1] is np.True_
        # Point beyond j-radius -> outside
        assert mask[mid_k, CY, CX + rj + 5] is np.False_
        # Point at edge of i-radius, center j -> inside
        assert mask[mid_k, CY + ri - 1, CX] is np.True_
        # Point beyond i-radius -> outside
        assert mask[mid_k, CY + ri + 5, CX] is np.False_

    def test_zero_height_cylinder(self):
        """Cylinder with k0==k1 should have exactly 1 slice."""
        mask = cyl(CX, CY, 30, 30, 90, 90)
        assert mask[90, CY, CX] is np.True_
        assert mask[89, CY, CX] is np.False_
        assert mask[91, CY, CX] is np.False_

    def test_symmetric_around_center(self):
        """Circular cylinder should be symmetric in j and i."""
        r = 20
        mask = cyl(CX, CY, r, r, 0, NZ - 1)
        mid_k = NZ // 2
        # Symmetric offsets from center
        assert mask[mid_k, CY, CX + 10] == mask[mid_k, CY, CX - 10]
        assert mask[mid_k, CY + 10, CX] == mask[mid_k, CY - 10, CX]


# ============================================================================
# ell (ellipsoid mask)
# ============================================================================

class TestEll:
    """Test 3D ellipsoid mask generation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _init_grid()

    def test_returns_bool_array(self):
        mask = ell(CX, CY, NZ // 2, 10, 10, 10)
        assert mask.dtype == bool

    def test_shape_matches_volume(self):
        mask = ell(CX, CY, NZ // 2, 10, 10, 10)
        assert mask.shape == (NZ, NY, NX)

    def test_center_voxel_inside(self):
        mask = ell(CX, CY, NZ // 2, 10, 10, 10)
        assert mask[NZ // 2, CY, CX] is np.True_

    def test_far_corner_outside(self):
        mask = ell(CX, CY, NZ // 2, 10, 10, 10)
        assert mask[0, 0, 0] is np.False_

    def test_ellipsoidal_asymmetry(self):
        """Different radii should produce different extents along each axis."""
        rj, ri, rk = 30, 15, 10
        ck = NZ // 2
        mask = ell(CX, CY, ck, rj, ri, rk)
        # j-axis: larger -> farther reach
        assert mask[ck, CY, CX + rj - 2] is np.True_
        assert mask[ck, CY, CX + rj + 5] is np.False_
        # i-axis: medium
        assert mask[ck, CY + ri - 2, CX] is np.True_
        assert mask[ck, CY + ri + 5, CX] is np.False_
        # k-axis: smallest
        assert mask[ck + rk - 2, CY, CX] is np.True_
        assert mask[ck + rk + 5, CY, CX] is np.False_

    def test_sphere_symmetry(self):
        """Equal radii should produce a sphere (symmetric)."""
        r = 20
        ck = NZ // 2
        mask = ell(CX, CY, ck, r, r, r)
        # Check symmetry along all axes
        for offset in [5, 10, 15]:
            assert mask[ck, CY, CX + offset] == mask[ck, CY, CX - offset]
            assert mask[ck, CY + offset, CX] == mask[ck, CY - offset, CX]
            assert mask[ck + offset, CY, CX] == mask[ck - offset, CY, CX]


# ============================================================================
# ell2 (2D ellipse mask)
# ============================================================================

class TestEll2:
    """Test 2D ellipse mask generation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _init_grid()

    def test_returns_bool_array(self):
        mask = ell2(CX, CY, 10, 10)
        assert mask.dtype == bool

    def test_shape_matches_2d(self):
        mask = ell2(CX, CY, 10, 10)
        assert mask.shape == (NY, NX)

    def test_center_inside(self):
        mask = ell2(CX, CY, 10, 10)
        assert mask[CY, CX] is np.True_

    def test_corner_outside(self):
        mask = ell2(CX, CY, 10, 10)
        assert mask[0, 0] is np.False_

    def test_ellipse_extent(self):
        rj, ri = 30, 15
        mask = ell2(CX, CY, rj, ri)
        # j direction (cols)
        assert mask[CY, CX + rj - 2] is np.True_
        assert mask[CY, CX + rj + 5] is np.False_
        # i direction (rows)
        assert mask[CY + ri - 2, CX] is np.True_
        assert mask[CY + ri + 5, CX] is np.False_

    def test_off_center_ellipse(self):
        """Ellipse centered away from CX/CY."""
        cj, ci = 50, 50
        mask = ell2(cj, ci, 10, 10)
        assert mask[ci, cj] is np.True_
        assert mask[CY, CX] is np.False_  # far from this small ellipse

    def test_circle_has_expected_area(self):
        """Circle area should approximate pi*r^2."""
        r = 30
        mask = ell2(CX, CY, r, r)
        pixel_count = mask.sum()
        expected = np.pi * r * r
        # Allow 5% tolerance for discretization
        assert abs(pixel_count - expected) / expected < 0.05


# ============================================================================
# shell
# ============================================================================

class TestShell:
    """Test cortical bone shell + fill function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _init_grid()

    def test_shell_fills_outer_ring_with_cortex(self):
        vol = np.zeros((NZ, NY, NX), dtype=np.float32)
        outer = cyl(CX, CY, 20, 20, 80, 100)
        inner = cyl(CX, CY, 12, 12, 80, 100)
        shell(vol, outer, inner, HU['cortex'], HU['marrow'])

        mid_k = 90
        # Cortex ring: outer but not inner
        cortex_mask = outer & ~inner
        assert vol[cortex_mask].min() == HU['cortex']
        assert vol[cortex_mask].max() == HU['cortex']

    def test_shell_fills_inner_with_fill_hu(self):
        vol = np.zeros((NZ, NY, NX), dtype=np.float32)
        outer = cyl(CX, CY, 20, 20, 80, 100)
        inner = cyl(CX, CY, 12, 12, 80, 100)
        shell(vol, outer, inner, HU['cortex'], HU['marrow'])

        assert vol[inner].min() == HU['marrow']
        assert vol[inner].max() == HU['marrow']

    def test_shell_does_not_modify_outside(self):
        vol = np.full((NZ, NY, NX), HU['air'], dtype=np.float32)
        outer = cyl(CX, CY, 20, 20, 80, 100)
        inner = cyl(CX, CY, 12, 12, 80, 100)
        shell(vol, outer, inner, HU['cortex'], HU['marrow'])

        # Check a voxel far outside
        assert vol[0, 0, 0] == HU['air']

    def test_shell_with_cancel_fill(self):
        """Shell can use cancellous bone as fill."""
        vol = np.zeros((NZ, NY, NX), dtype=np.float32)
        outer = ell(CX, CY, 64, 22, 17, 19)
        inner = ell(CX, CY, 64, 13, 10, 12)
        shell(vol, outer, inner, HU['cortex'], HU['cancel'])

        assert vol[inner].min() == HU['cancel']


# ============================================================================
# build_phantom
# ============================================================================

class TestBuildPhantom:
    """Test full phantom volume generation."""

    @pytest.fixture(scope="class")
    def phantom_R(self):
        """Build right arm phantom once for all tests in this class."""
        return build_phantom('R')

    @pytest.fixture(scope="class")
    def phantom_L(self):
        """Build left arm phantom once for all tests in this class."""
        return build_phantom('L')

    def test_returns_ndarray(self, phantom_R):
        assert isinstance(phantom_R, np.ndarray)

    def test_shape(self, phantom_R):
        assert phantom_R.shape == (NZ, NY, NX)

    def test_dtype_float(self, phantom_R):
        assert phantom_R.dtype in (np.float32, np.float64)

    def test_hu_range_reasonable(self, phantom_R):
        """HU values should be within expected range after smoothing."""
        assert phantom_R.min() >= -1100  # near air
        assert phantom_R.max() <= 900    # near cortex

    def test_contains_air(self, phantom_R):
        """Corners should be near air HU."""
        corner_hu = phantom_R[0, 0, 0]
        assert corner_hu < -900  # near -1000

    def test_contains_bone(self, phantom_R):
        """Volume should contain voxels near cortical bone HU."""
        assert phantom_R.max() > 500  # should have cortex-like values

    def test_contains_soft_tissue(self, phantom_R):
        """Volume should contain muscle-range HU values."""
        mid_k = NZ // 2
        mid_slice = phantom_R[mid_k]
        # There should be voxels in muscle range (20-80 HU)
        muscle_mask = (mid_slice > 20) & (mid_slice < 80)
        assert muscle_mask.sum() > 0

    def test_laterality_R_vs_L_differ(self, phantom_R, phantom_L):
        """R and L phantoms should differ (mirrored anatomy)."""
        assert not np.allclose(phantom_R, phantom_L)

    def test_laterality_R_vs_L_same_range(self, phantom_R, phantom_L):
        """R and L should have similar HU range."""
        assert abs(phantom_R.min() - phantom_L.min()) < 10
        assert abs(phantom_R.max() - phantom_L.max()) < 10

    def test_laterality_L_mirror_symmetry(self, phantom_R, phantom_L):
        """L phantom should be approximately a left-right mirror of R."""
        # Flip R along j-axis (left-right mirror)
        r_flipped = phantom_R[:, :, ::-1]
        # After mirroring, structures should overlap significantly
        # (not exact due to discrete voxel effects, but correlated)
        corr = np.corrcoef(r_flipped.ravel(), phantom_L.ravel())[0, 1]
        assert corr > 0.9  # strongly correlated

    def test_humerus_shaft_present(self, phantom_R):
        """Upper slices should contain bone (humerus shaft)."""
        top_slice = phantom_R[170]  # near proximal end
        assert top_slice.max() > 200  # should have bone

    def test_soft_tissue_envelope(self, phantom_R):
        """Mid-slice should have soft tissue surrounding bone."""
        mid_k = NZ // 2
        sl = phantom_R[mid_k]
        # Tissue (non-air) count
        tissue = (sl > -500).sum()
        total = NY * NX
        # Tissue should be a reasonable fraction (arm cross-section)
        assert 0.05 < tissue / total < 0.8

    def test_gaussian_smoothing_applied(self, phantom_R):
        """After smoothing, there should be no perfectly sharp transitions."""
        # Check a bone-air boundary region: values shouldn't jump from -1000 to 800
        mid_k = 100
        sl = phantom_R[mid_k]
        # Get gradient magnitude
        grad = np.abs(np.diff(sl, axis=1))
        # Max gradient should be less than full air-to-cortex jump
        assert grad.max() < (HU['cortex'] - HU['air'])

    def test_distal_region_has_bone_spread(self, phantom_R):
        """Distal slices (k~40-60) should have bone spread across width (multiple structures)."""
        sl = phantom_R[55]
        bone_mask = sl > 200
        # Bone should be present in multiple columns (spread across the slice)
        cols_with_bone = np.any(bone_mask, axis=0)
        bone_col_indices = np.where(cols_with_bone)[0]
        if len(bone_col_indices) > 1:
            spread = bone_col_indices[-1] - bone_col_indices[0]
            # Bone structures should span a reasonable width (radius + ulna separated)
            assert spread > 20  # at least 10mm spread


# ============================================================================
# write_dicom_series (mocked I/O)
# ============================================================================

class TestWriteDicomSeries:
    """Test DICOM writing with mocked pydicom."""

    def test_creates_output_directory(self, tmp_path):
        """write_dicom_series should create the output directory."""
        from create_phantom import write_dicom_series

        out_dir = str(tmp_path / "dicom_out")
        # Use a tiny volume to keep test fast
        tiny_vol = np.zeros((4, 4, 4), dtype=np.float32)

        with patch("create_phantom.pydicom") as mock_pydicom:
            mock_pydicom.uid.generate_uid.return_value = "1.2.3"
            mock_pydicom.dataset.FileMetaDataset.return_value = MagicMock()
            mock_pydicom.dataset.FileDataset.return_value = MagicMock()
            write_dicom_series(tiny_vol, out_dir, 'R')

        assert os.path.isdir(out_dir)

    def test_writes_correct_number_of_slices(self, tmp_path):
        """Should write one DICOM file per slice."""
        from create_phantom import write_dicom_series

        out_dir = str(tmp_path / "dicom_out2")
        nz = 5
        tiny_vol = np.zeros((nz, 4, 4), dtype=np.float32)

        with patch("create_phantom.pydicom") as mock_pydicom:
            mock_pydicom.uid.generate_uid.return_value = "1.2.3"
            mock_pydicom.dataset.FileMetaDataset.return_value = MagicMock()
            mock_pydicom.dataset.FileDataset.return_value = MagicMock()
            write_dicom_series(tiny_vol, out_dir, 'R')

        # dcmwrite should be called nz times
        assert mock_pydicom.dcmwrite.call_count == nz


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _init_grid()

    def test_cyl_with_radius_1(self):
        """Minimal cylinder should contain at least the center voxel."""
        mask = cyl(CX, CY, 1, 1, 90, 90)
        assert mask.sum() >= 1

    def test_ell_with_radius_1(self):
        """Minimal ellipsoid should contain at least the center."""
        mask = ell(CX, CY, 90, 1, 1, 1)
        assert mask.sum() >= 1

    def test_ell2_with_radius_1(self):
        """Minimal 2D ellipse should contain at least the center."""
        mask = ell2(CX, CY, 1, 1)
        assert mask.sum() >= 1

    def test_cyl_no_overlap_z_ranges(self):
        """Two cylinders with non-overlapping Z ranges should not intersect."""
        mask1 = cyl(CX, CY, 20, 20, 0, 50)
        mask2 = cyl(CX, CY, 20, 20, 60, 100)
        assert (mask1 & mask2).sum() == 0

    def test_shell_inner_larger_than_outer(self):
        """If inner >= outer everywhere, cortex ring is empty."""
        vol = np.zeros((NZ, NY, NX), dtype=np.float32)
        outer = cyl(CX, CY, 10, 10, 80, 100)
        inner = cyl(CX, CY, 20, 20, 80, 100)  # inner larger
        shell(vol, outer, inner, HU['cortex'], HU['marrow'])
        # Cortex region (outer & ~inner) should be empty since inner covers outer
        cortex_ring = outer & ~inner
        assert cortex_ring.sum() == 0
        # Inner values should still be set
        assert vol[inner & outer].min() == HU['marrow']
