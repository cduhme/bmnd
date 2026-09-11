import numpy as np

from bmnd.patchaggregation import scatter_add_patches


def test_scatter_add_patches_single_patch():
    accum_num = np.zeros((3, 3), dtype=np.float32)
    accum_den = np.zeros((3, 3), dtype=np.float32)
    group_filt = np.ones((1, 2, 2), dtype=np.float32)
    weights = np.array([1.0], dtype=np.float32)
    sel_global = np.array([0], dtype=np.int64)
    counts = (2, 2)
    block_size = (2, 2)
    w_patch = np.ones(block_size, dtype=np.float32)

    scatter_add_patches(
        accum_num=accum_num,
        accum_den=accum_den,
        group_filt=group_filt,
        weights=weights,
        sel_global=sel_global,
        counts=counts,
        block_size=block_size,
        w_patch=w_patch,
    )

    expected = np.zeros((3, 3), dtype=np.float32)
    expected[0:2, 0:2] = 1.0
    assert np.array_equal(accum_num, expected)
    assert np.array_equal(accum_den, expected)
