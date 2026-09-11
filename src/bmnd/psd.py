from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import correlate


@dataclass
class ProcessedPSD:
    sigma_psd: NDArray[np.float32]
    correlation_kernel: NDArray[np.float32]
    reduced_sigma_psd: NDArray[np.float32]
    blur_kernel: NDArray[np.float32]
    blurred_sigma_psd: NDArray[np.float32]


def _validate_sigma(sigma: object) -> float:
    if isinstance(sigma, (bool, np.bool_)):
        raise ValueError(f"sigma must be finite and non-negative, got {sigma!r}.")
    try:
        sigma_value = float(sigma)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"sigma must be finite and non-negative, got {sigma!r}."
        ) from exc
    if not np.isfinite(sigma_value) or sigma_value < 0.0:
        raise ValueError(f"sigma must be finite and non-negative, got {sigma!r}.")
    return sigma_value


def _validate_sigma_psd(
    shape: tuple[int, ...],
    sigma_psd: NDArray[np.floating] | float,
) -> NDArray[np.float32]:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            sigma_psd_arr = np.asarray(sigma_psd, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "sigma_psd must contain only finite non-negative values."
        ) from exc
    if not np.all(np.isfinite(sigma_psd_arr)) or np.any(sigma_psd_arr < 0.0):
        raise ValueError("sigma_psd must contain only finite non-negative values.")
    if (
        sigma_psd_arr.ndim != 0
        and sigma_psd_arr.size != 1
        and sigma_psd_arr.shape != shape
    ):
        raise ValueError(
            "sigma_psd must either be scalar or have the same shape as volume"
        )
    return sigma_psd_arr


def resolve_nf(
    nf: tuple[int, ...] | int | None,
    shape: tuple[int, ...],
) -> tuple[tuple[int, ...] | None, bool]:
    if nf is None:
        return None, False
    if isinstance(nf, bool):
        raise ValueError(f"nf must be a per-axis integer tuple or zero, got {nf!r}.")
    if isinstance(nf, (int, np.integer)):
        if int(nf) == 0:
            return tuple(min(int(size), 16) for size in shape), True
        raise ValueError(
            "nf must be zero or contain one positive integer per volume axis, "
            f"got {nf!r}."
        )

    try:
        values = tuple(nf)
    except TypeError as exc:
        raise ValueError(
            "nf must be zero or contain one positive integer per volume axis, "
            f"got {nf!r}."
        ) from exc
    if not values:
        return None, False
    if len(values) != len(shape):
        raise ValueError(
            f"nf must contain {len(shape)} values for shape {shape}, got {values}."
        )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise ValueError(f"nf values must be integers, got {values}.")

    resolved = tuple(int(value) for value in values)
    if any(value < 0 for value in resolved):
        raise ValueError(f"nf values must be non-negative, got {resolved}.")
    if any(value == 0 for value in resolved):
        return tuple(min(int(size), 16) for size in shape), True
    return resolved, False


def scalar_sigma_to_psd(
    shape: tuple[int, ...],
    sigma: float,
) -> NDArray[np.float32]:
    sigma_value = _validate_sigma(sigma)
    volume_size = float(np.prod(shape))
    value = volume_size * sigma_value * sigma_value
    if not np.isfinite(value) or value > np.finfo(np.float32).max:
        raise ValueError(f"sigma is too large for a float32 PSD, got {sigma!r}.")
    return np.full(shape, value, dtype=np.float32)


def normalize_sigma_psd(
    shape: tuple[int, ...],
    sigma: float | None = None,
    sigma_psd: NDArray[np.floating] | float | None = None,
) -> NDArray[np.float32] | None:
    if sigma is not None and sigma_psd is not None:
        raise ValueError("sigma and sigma_psd are mutually exclusive.")
    if sigma is None and sigma_psd is None:
        return None

    if sigma_psd is None:
        assert sigma is not None
        return scalar_sigma_to_psd(shape, sigma)

    sigma_psd_arr = _validate_sigma_psd(shape, sigma_psd)
    if sigma_psd_arr.ndim == 0 or sigma_psd_arr.size == 1:
        scalar_sigma = float(np.asarray(sigma_psd_arr).reshape(()))
        return scalar_sigma_to_psd(shape, scalar_sigma)

    return sigma_psd_arr.astype(np.float32, copy=False)


def autocovariance_from_psd(sigma_psd: NDArray[np.floating]) -> NDArray[np.float32]:
    psd = np.asarray(sigma_psd, dtype=np.float32)
    autocovariance = np.real(np.fft.ifftn(psd) / float(psd.size)).astype(
        np.float32, copy=False
    )
    return autocovariance


def get_kernel_from_psd(
    sigma_psd: NDArray[np.floating],
    *,
    single_dim_psd: bool = False,
) -> NDArray[np.float32]:
    psd = np.asarray(sigma_psd, dtype=np.float32)
    if single_dim_psd:
        scalar_value = float(np.linalg.norm(np.sqrt(psd / float(psd.size))))
        return np.array([scalar_value], dtype=np.float32)

    signal_std = np.sqrt(psd / float(psd.size))
    return np.fft.fftshift(
        np.real(np.fft.ifftn(signal_std)).astype(np.float32, copy=False)
    ).astype(np.float32, copy=False)


def gaussian_kernel_nd(
    size: tuple[int, ...], sigma_scale: tuple[float, ...]
) -> NDArray[np.float32]:
    axes = []
    for axis_size, sigma in zip(size, sigma_scale, strict=False):
        center = (axis_size - 1) / 2.0
        coords = np.arange(axis_size, dtype=np.float32) - center
        safe_sigma = max(float(sigma), 1e-6)
        axis = np.exp(-0.5 * (coords / safe_sigma) ** 2)
        axis /= np.sum(axis, dtype=np.float32)
        axes.append(axis.astype(np.float32))

    kernel = axes[0]
    for axis in axes[1:]:
        kernel = np.multiply.outer(kernel, axis)
    kernel = kernel.astype(np.float32, copy=False)
    kernel /= np.sum(kernel, dtype=np.float32)
    return kernel


def process_psd_for_nf(
    sigma_psd: NDArray[np.floating],
    nf: tuple[int, ...] | int | None,
) -> NDArray[np.float32]:
    psd = np.array(sigma_psd, dtype=np.float32, copy=True)
    resolved_nf, _ = resolve_nf(nf, tuple(int(size) for size in psd.shape))
    _, _, blurred = _reduce_and_blur_psd(psd, resolved_nf)
    return blurred


def build_nf_blur_kernel(
    sigma_psd_shape: tuple[int, ...],
    nf: tuple[int, ...],
) -> tuple[NDArray[np.float32], tuple[float, ...]]:
    kernel_sizes = []
    kernel_sigmas = []
    for axis in range(len(sigma_psd_shape)):
        half_width = np.floor(0.5 * sigma_psd_shape[axis] / nf[axis])
        kernel_sizes.append(int(1 + 2 * half_width))
        kernel_sigmas.append(float(1 + 2 * half_width / 20.0))

    blur_kernel = gaussian_kernel_nd(tuple(kernel_sizes), tuple(kernel_sigmas))
    return blur_kernel, tuple(kernel_sigmas)


def _reduce_and_blur_psd(
    sigma_psd: NDArray[np.float32],
    nf: tuple[int, ...] | None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    psd = np.array(sigma_psd, dtype=np.float32, copy=True)
    if nf is None:
        unit_kernel = np.ones((1,) * psd.ndim, dtype=np.float32)
        return psd, unit_kernel, psd.copy()

    reduced = psd.copy()
    single_kernels = []
    for axis in range(psd.ndim):
        kernel_shape = [1] * psd.ndim
        kernel_shape[axis] = 3
        single_kernels.append(
            np.ones(tuple(kernel_shape), dtype=np.float32) / np.float32(3.0)
        )

    for axis in range(psd.ndim):
        original_ratio = psd.shape[axis] / float(nf[axis])
        ratio = reduced.shape[axis] / float(nf[axis])
        while ratio > 16.0:
            mid_corr = correlate(reduced, single_kernels[axis], mode="wrap")
            slicer = [slice(None)] * psd.ndim
            slicer[axis] = slice(1, None, 3)
            reduced = mid_corr[tuple(slicer)]
            ratio = reduced.shape[axis] / float(nf[axis])
        reduced *= np.float32(ratio / original_ratio)

    blur_kernel, _ = build_nf_blur_kernel(
        tuple(int(size) for size in reduced.shape),
        nf,
    )
    blurred = correlate(reduced, blur_kernel, mode="wrap").astype(
        np.float32,
        copy=False,
    )
    return reduced.astype(np.float32, copy=False), blur_kernel, blurred


def preprocess_psd(
    sigma_psd: NDArray[np.floating],
    nf: tuple[int, ...] | int | None,
    *,
    single_dim_psd: bool = False,
) -> ProcessedPSD:
    psd = np.array(sigma_psd, dtype=np.float32, copy=True)
    resolved_nf, _ = resolve_nf(nf, tuple(int(size) for size in psd.shape))
    correlation_kernel = get_kernel_from_psd(psd, single_dim_psd=single_dim_psd)

    reduction_nf = None if single_dim_psd else resolved_nf
    reduced, blur_kernel, blurred = _reduce_and_blur_psd(psd, reduction_nf)
    return ProcessedPSD(
        sigma_psd=psd,
        correlation_kernel=correlation_kernel,
        reduced_sigma_psd=reduced.astype(np.float32, copy=False),
        blur_kernel=blur_kernel,
        blurred_sigma_psd=blurred,
    )
