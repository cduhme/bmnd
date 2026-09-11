import numpy as np

from bmnd.utils import extract_patches_strided, kaiser_window_nd


def test_extract_patches_strided_contents():
    volume = np.arange(9, dtype=np.float32).reshape(3, 3)

    patches = extract_patches_strided(volume, (2, 2))

    assert patches.shape == (2, 2, 2, 2)
    assert np.array_equal(patches[0, 0], volume[0:2, 0:2])
    assert np.array_equal(patches[0, 1], volume[0:2, 1:3])
    assert np.array_equal(patches[1, 0], volume[1:3, 0:2])
    assert np.array_equal(patches[1, 1], volume[1:3, 1:3])


def test_kaiser_window_nd_matches_separable_definition():
    window = kaiser_window_nd((3, 4), beta=2.0)

    assert window.shape == (3, 4)
    assert window.dtype == np.float32
    expected = np.kaiser(3, 2.0)[:, None] * np.kaiser(4, 2.0)[None, :]
    np.testing.assert_allclose(window, expected, rtol=1e-6)
