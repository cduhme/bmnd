import numpy as np
import pytest

from bmnd.utils import extract_patches_strided


def test_extract_patches_strided_shape_formula_3d():
    volume = np.arange(120, dtype=np.float32).reshape(4, 5, 6)
    patch_size = (2, 3, 4)
    patches = extract_patches_strided(volume, patch_size)
    assert patches.shape == (3, 3, 3, 2, 3, 4)
    for origin in np.ndindex(3, 3, 3):
        slices = tuple(slice(start, start + size) for start, size in zip(origin, patch_size))
        np.testing.assert_array_equal(patches[origin], volume[slices])


def test_extract_patches_strided_is_view():
    volume = np.arange(9, dtype=np.float32).reshape(3, 3)
    patches = extract_patches_strided(volume, (2, 2))
    patches[0, 0, 0, 0] = -99.0
    assert volume[0, 0] == -99.0


def test_extract_patches_strided_ndim_mismatch_raises_assertion():
    with pytest.raises(AssertionError, match="patch_size must match volume.ndim"):
        extract_patches_strided(np.zeros((3, 3), dtype=np.float32), (2, 2, 2))
