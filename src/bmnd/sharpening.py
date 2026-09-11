import numpy as np
from numpy.typing import NDArray


def sharpen_group_dc(group_coeffs: NDArray[np.float32], alpha_3d: float) -> None:
    if alpha_3d is None or alpha_3d == 1.0:
        return

    # DC slice = coefficients at group frequency 0
    C0 = group_coeffs[0]

    # Non-linear sharpening: C0 <- sign(C0) * |C0|^(1/alpha)
    mag = np.asarray(np.abs(C0), dtype=C0.dtype)
    np.multiply(np.sign(C0), mag ** (1.0 / alpha_3d), out=C0)


def spatial_sharpen(patch: NDArray[np.float32], alpha: float) -> None:
    if alpha is None or alpha == 1.0:
        return

    # x -> sign(x) * |x|^(1/alpha)
    mag = np.abs(patch)
    np.multiply(np.sign(patch), mag ** (1.0 / alpha), out=patch)
