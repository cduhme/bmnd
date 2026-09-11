from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

bm3d = pytest.importorskip("bm3d")

from bmnd import bmnd
from bmnd.profiles import BM3DProfile
from bmnd.psd import scalar_sigma_to_psd

from .fixtures import add_gaussian_noise, compute_metrics_nd, make_clean_2d


def _make_colored_psd(shape: tuple[int, ...], sigma: float) -> NDArray[np.float32]:
    grids = np.meshgrid(*[np.fft.fftfreq(size) for size in shape], indexing="ij")
    radial_energy = np.zeros(shape, dtype=np.float32)
    for grid in grids:
        radial_energy += (grid * grid).astype(np.float32)
    sigma_psd = 1.0 + 4.0 * radial_energy
    sigma_psd *= (
        np.prod(shape) * sigma * sigma / float(np.mean(sigma_psd, dtype=np.float32))
    )
    return sigma_psd.astype(np.float32)


@pytest.mark.parametrize(
    ("sigma", "rmse_limit", "max_abs_limit", "min_psnr_gain", "min_ssim_gain"),
    [(0.05, 0.0032, 0.028, 6.0, 0.050), (0.10, 0.0050, 0.035, 8.0, 0.150)],
)
def test_bmnd_tracks_official_bm3d_on_small_gaussian_inputs(
    sigma: float,
    rmse_limit: float,
    max_abs_limit: float,
    min_psnr_gain: float,
    min_ssim_gain: float,
) -> None:
    clean = make_clean_2d()
    noisy = add_gaussian_noise(clean, sigma=sigma)

    bmnd_out = cast(NDArray[np.float32], bmnd(noisy, BM3DProfile(), sigma=sigma))
    official_out = bm3d.bm3d(noisy, sigma, profile="np")

    bmnd_metrics = compute_metrics_nd(clean, bmnd_out)
    noisy_metrics = compute_metrics_nd(clean, noisy)

    rmse_to_official = float(np.sqrt(np.mean((bmnd_out - official_out) ** 2)))
    max_abs_diff = float(np.max(np.abs(bmnd_out - official_out)))

    assert bmnd_out.shape == clean.shape
    assert bmnd_out.dtype == np.float32
    assert np.isfinite(bmnd_out).all()
    assert bmnd_metrics["psnr"] - noisy_metrics["psnr"] > min_psnr_gain
    assert bmnd_metrics["ssim"] - noisy_metrics["ssim"] > min_ssim_gain
    assert rmse_to_official < rmse_limit, (
        f"RMSE to official BM3D too large for sigma={sigma}: "
        f"{rmse_to_official:.6f} >= {rmse_limit:.6f}"
    )
    assert max_abs_diff < max_abs_limit, (
        f"Max abs diff to official BM3D too large for sigma={sigma}: "
        f"{max_abs_diff:.6f} >= {max_abs_limit:.6f}"
    )


def test_bmnd_bm3d_scalar_sigma_matches_flat_psd_semantics() -> None:
    sigma = 0.05
    clean = make_clean_2d()
    noisy = add_gaussian_noise(clean, sigma=sigma)
    sigma_psd = scalar_sigma_to_psd(noisy.shape, sigma)

    bmnd_sigma = cast(NDArray[np.float32], bmnd(noisy, BM3DProfile(), sigma=sigma))
    bmnd_psd = cast(
        NDArray[np.float32], bmnd(noisy, BM3DProfile(), sigma_psd=sigma_psd)
    )
    official_sigma = bm3d.bm3d(noisy, sigma, profile="np")
    official_psd = bm3d.bm3d(noisy, sigma_psd, profile="np")

    assert np.allclose(official_sigma, official_psd, atol=1e-6)
    assert np.allclose(bmnd_sigma, bmnd_psd, atol=1e-6)


def test_bmnd_tracks_official_bm3d_on_small_colored_psd_input() -> None:
    sigma = 0.05
    clean = make_clean_2d()
    noisy = add_gaussian_noise(clean, sigma=sigma)
    sigma_psd = _make_colored_psd(noisy.shape, sigma)

    bmnd_out = cast(
        NDArray[np.float32], bmnd(noisy, BM3DProfile(), sigma_psd=sigma_psd)
    )
    official_out = bm3d.bm3d(noisy, sigma_psd, profile="np")

    rmse_to_official = float(np.sqrt(np.mean((bmnd_out - official_out) ** 2)))
    max_abs_diff = float(np.max(np.abs(bmnd_out - official_out)))

    assert np.isfinite(bmnd_out).all()
    # The modern profile propagates generalized coefficient variance into patch weights.
    assert rmse_to_official < 0.0061
    assert max_abs_diff < 0.023
