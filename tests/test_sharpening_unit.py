import numpy as np

from bmnd.sharpening import sharpen_group_dc, spatial_sharpen


def test_sharpen_group_dc_noop_for_alpha_one():
    group = np.array([[4.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    original = group.copy()

    sharpen_group_dc(group, alpha_3d=1.0)

    assert np.array_equal(group, original)


def test_sharpen_group_dc_changes_dc_slice():
    group = np.array([[4.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    sharpen_group_dc(group, alpha_3d=2.0)

    assert np.allclose(group[0], np.array([2.0, 1.0], dtype=np.float32))
    assert np.array_equal(group[1], np.array([2.0, 3.0], dtype=np.float32))


def test_spatial_sharpen_noop_for_alpha_one():
    patch = np.array([-4.0, 1.0], dtype=np.float32)
    original = patch.copy()

    spatial_sharpen(patch, alpha=1.0)

    assert np.array_equal(patch, original)


def test_spatial_sharpen_changes_values():
    patch = np.array([-4.0, 1.0], dtype=np.float32)

    spatial_sharpen(patch, alpha=2.0)

    assert np.allclose(patch, np.array([-2.0, 1.0], dtype=np.float32))
