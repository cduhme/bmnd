from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def estimate_gaussian_global_sigma(
    volume: NDArray[np.floating],
    eps: float = 1e-10,
) -> float:
    vol = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(vol)
    non_zero = finite & (vol != 0.0)
    if np.any(non_zero):
        ref = vol[non_zero]
    elif np.any(finite):
        ref = vol[finite]
    else:
        ref = np.array([0.0], dtype=np.float32)

    med = np.median(ref)
    mad = float(np.median(np.abs(ref - med)))
    mad_scale = 0.6745
    global_sigma = mad / mad_scale

    if global_sigma <= 0.0:
        q25, q75 = np.percentile(ref, [25.0, 75.0])
        iqr = float(q75 - q25)
        iqr_scale = 1.349
        if iqr > 0.0:
            global_sigma = iqr / iqr_scale

    if global_sigma <= 0.0:
        p5, p95 = np.percentile(ref, [5.0, 95.0])
        trimmed = ref[(ref >= p5) & (ref <= p95)]
        if trimmed.size > 0:
            sigma_trim = float(np.std(trimmed, dtype=np.float32))
            if sigma_trim > 0.0:
                global_sigma = sigma_trim

    if global_sigma <= 0.0:
        global_sigma = float(np.sqrt(eps))
    if not np.isfinite(global_sigma) or global_sigma < 0.0:
        return 0.0
    return global_sigma


def estimate_poisson_global_sigma(
    volume: NDArray[np.floating],
    eps: float = 1e-10,
) -> float:
    vol = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(vol)
    nonnegative = (
        np.maximum(vol[finite], 0.0)
        if np.any(finite)
        else np.array([0.0], dtype=np.float32)
    )
    positive = nonnegative[nonnegative > 0.0]

    if positive.size > 0:
        positive_fraction = positive.size / max(nonnegative.size, 1)
        if positive_fraction < 0.1 and positive.size >= 3:
            sorted_vals = np.sort(positive, axis=None)
            trim = max(int(0.1 * sorted_vals.size), 1)
            if sorted_vals.size - 2 * trim > 0:
                mean_val = float(np.mean(sorted_vals[trim:-trim], dtype=np.float32))
            else:
                mean_val = float(np.mean(sorted_vals, dtype=np.float32))
        else:
            mean_val = float(np.mean(positive, dtype=np.float32))
    else:
        mean_val = float(np.mean(nonnegative, dtype=np.float32))

    global_sigma = float(np.sqrt(max(mean_val, eps)))

    if global_sigma <= 0.0 and positive.size > 0:
        q25, q75 = np.percentile(positive, [25.0, 75.0])
        iqr = float(q75 - q25)
        if iqr > 0.0:
            global_sigma = float(np.sqrt(max(iqr / 1.349, eps)))

    if global_sigma <= 0.0 and positive.size > 0:
        med = float(np.median(positive))
        mad = float(np.median(np.abs(positive - med)))
        if mad > 0.0:
            global_sigma = float(np.sqrt(max(mad / 0.6745, eps)))

    if global_sigma <= 0.0:
        global_sigma = float(np.sqrt(eps))
    if not np.isfinite(global_sigma) or global_sigma < 0.0:
        return 0.0
    return global_sigma
