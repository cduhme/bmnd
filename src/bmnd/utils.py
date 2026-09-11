import numpy as np
from numpy.typing import NDArray


def kaiser_window_nd(
    patch_size: tuple[int, ...], beta: float = 2.0
) -> NDArray[np.float32]:
    axes = [np.kaiser(sz, beta) for sz in patch_size]
    w: NDArray[np.float64] = axes[0]
    for a in axes[1:]:
        w = np.multiply.outer(w, a)
    return w.astype(np.float32)


def extract_patches_strided(
    volume: NDArray[np.float32], patch_size: tuple[int, ...]
) -> NDArray[np.float32]:
    vol_arr = np.asarray(volume)
    patch_size = tuple(patch_size)
    ndim = vol_arr.ndim
    assert len(patch_size) == ndim, "patch_size must match volume.ndim"

    out_shape = (
        tuple(vol_arr.shape[d] - patch_size[d] + 1 for d in range(ndim)) + patch_size
    )
    out_strides = vol_arr.strides + vol_arr.strides

    view = np.lib.stride_tricks.as_strided(
        vol_arr, shape=out_shape, strides=out_strides
    )
    return view
