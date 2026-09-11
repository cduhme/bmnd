import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _mutual_information_hist(
    x: np.ndarray, y: np.ndarray, bins: int = 64, data_range=None
) -> float:
    """Estimate mutual information in nats from a joint histogram."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if data_range is None:
        lo = min(float(x.min()), float(y.min()))
        hi = max(float(x.max()), float(y.max()))
    else:
        lo, hi = map(float, data_range)

    if hi <= lo:
        return 0.0

    h2d, _, _ = np.histogram2d(x, y, bins=bins, range=[[lo, hi], [lo, hi]])
    pxy = h2d.astype(np.float64)
    pxy_sum = pxy.sum()
    if pxy_sum == 0:
        return 0.0
    pxy /= pxy_sum

    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    eps = 1e-12
    denom = px @ py + eps
    mi = np.sum(pxy * np.log((pxy + eps) / denom))
    return float(mi)


def _pearson_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    x0 = x - x.mean()
    y0 = y - y.mean()

    denom = np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0))
    if denom == 0:
        return 1.0 if np.allclose(x, y) else 0.0

    return float(np.sum(x0 * y0) / denom)


def compute_metrics_nd(
    target: np.ndarray, prediction: np.ndarray, channel_axis=None, mi_bins: int = 64
):
    """Compute PSNR, SSIM, mutual information (nats), and Pearson correlation.

    Copied from the parent project's experiments/utils/evaluation.py.
    """
    if target.shape != prediction.shape:
        raise ValueError("target and prediction must have the same shape.")

    target_f = target.astype(np.float32, copy=False)
    pred_f = prediction.astype(np.float32, copy=False)

    if not np.isfinite(target_f).all() or not np.isfinite(pred_f).all():
        raise ValueError("Inputs contain NaN or Inf values.")

    data_min = float(target_f.min())
    data_max = float(target_f.max())
    data_range = data_max - data_min
    common_lo = float(min(target_f.min(), pred_f.min()))
    common_hi = float(max(target_f.max(), pred_f.max()))
    mi = _mutual_information_hist(
        target_f, pred_f, bins=mi_bins, data_range=(common_lo, common_hi)
    )
    corr = _pearson_corrcoef(target_f, pred_f)

    if data_range == 0:
        common_range = common_hi - common_lo
        if common_range <= 0:
            common_range = 1.0
        psnr = (
            np.inf
            if np.allclose(target_f, pred_f)
            else peak_signal_noise_ratio(target_f, pred_f, data_range=common_range)
        )
        ssim = 1.0 if np.allclose(target_f, pred_f) else 0.0
        return {"psnr": psnr, "ssim": ssim, "mi": mi, "corr": corr}

    psnr = peak_signal_noise_ratio(target_f, pred_f, data_range=data_range)

    if channel_axis is None:
        spatial_shape = target_f.shape
    else:
        spatial_shape = tuple(
            s for i, s in enumerate(target_f.shape) if i != channel_axis
        )

    min_spatial = min(spatial_shape)
    win_size = min(7, min_spatial)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        raise ValueError(f"Spatial dimensions too small for SSIM: {spatial_shape}")

    ssim = structural_similarity(
        target_f,
        pred_f,
        data_range=data_range,
        channel_axis=channel_axis,
        win_size=win_size,
    )

    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "mi": float(mi),
        "corr": float(corr),
    }


def make_clean_2d(shape: tuple[int, int] = (32, 32)) -> np.ndarray:
    """Create a deterministic 2D test image.

    Args:
        shape: Image shape (height, width).

    Returns:
        Clean float32 image with values in [0, 1].
    """
    y, x = np.meshgrid(
        np.linspace(0.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(0.0, 1.0, shape[1], dtype=np.float32),
        indexing="ij",
    )
    img = 0.45 + 0.25 * np.sin(2.0 * np.pi * x * 2.5)
    img += 0.15 * np.cos(2.0 * np.pi * y * 1.5)
    img += 0.10 * ((x > 0.55) & (y < 0.45))
    img += 0.08 * (((x - 0.70) ** 2 + (y - 0.35) ** 2) < 0.04)
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def make_clean_3d(shape: tuple[int, int, int] = (12, 12, 12)) -> np.ndarray:
    """Create a deterministic 3D test volume.

    Args:
        shape: Volume shape (depth, height, width).

    Returns:
        Clean float32 volume with values in [0, 1].
    """
    z, y, x = np.meshgrid(
        np.linspace(0.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(0.0, 1.0, shape[1], dtype=np.float32),
        np.linspace(0.0, 1.0, shape[2], dtype=np.float32),
        indexing="ij",
    )
    vol = 0.35 + 0.20 * np.sin(2.0 * np.pi * x * 1.5)
    vol += 0.12 * np.cos(2.0 * np.pi * y * 1.0)
    vol += 0.10 * z
    vol += 0.12 * (((x - 0.55) ** 2 + (y - 0.45) ** 2 + (z - 0.60) ** 2) < 0.05)
    vol += 0.08 * ((x > 0.65) & (y < 0.35) & (z > 0.40))
    return np.clip(vol.astype(np.float32), 0.0, 1.0)


def add_gaussian_noise(clean: np.ndarray, sigma: float, seed: int = 123) -> np.ndarray:
    """Add Gaussian noise to a clean image/volume.

    Args:
        clean: Clean input array.
        sigma: Noise standard deviation.
        seed: Random seed for reproducibility.

    Returns:
        Noisy array with values clipped to [0, 1].
    """
    rng = np.random.default_rng(seed)
    noisy = clean + rng.normal(0.0, sigma, size=clean.shape).astype(clean.dtype)
    return np.clip(noisy, 0.0, 1.0)
