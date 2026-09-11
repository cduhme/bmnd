from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from bmnd.bmndalgo import bmnd
from bmnd.profiles import BMNDProfile
from bmnd.transforms import Transform, TransformMode, TransformType


def _make_profile(
    noise_model: str,
) -> BMNDProfile:
    profile = BMNDProfile(
        noise_model=noise_model,
        ht_block_size=(4, 4),
        ht_step=(4, 4),
        ht_search_window=(1, 1),
        ht_max_stack_size=4,
        wiener_block_size=(4, 4),
        wiener_step=(4, 4),
        wiener_search_window=(1, 1),
        wiener_max_stack_size=4,
    )
    profile.ht_transform = Transform(TransformType.DCT, TransformMode.ND)
    profile.wiener_transform = Transform(TransformType.DCT, TransformMode.ND)
    profile.ref_batch_size = 8
    return profile


@pytest.mark.parametrize(
    "mean, noise_std",
    [(0.0, 0.0), (0.0, 1e-6), (0.5, 0.2), (1e6, 1e3)],
    ids=["zero", "tiny-signed", "ordinary", "large-offset"],
)
def test_bmnd_gaussian_numerical_stability(mean: float, noise_std: float):
    rng = np.random.default_rng(0)
    volume = rng.normal(mean, noise_std, size=(8, 8)).astype(np.float32)
    output = cast(
        NDArray[np.float32],
        bmnd(volume, _make_profile("gaussian"), sigma=noise_std),
    )
    assert output.shape == volume.shape
    assert output.dtype == np.float32
    assert np.isfinite(output).all()


@pytest.mark.parametrize(
    "rate",
    [0.0, 0.05, 5.0, 1e6],
    ids=["zero", "sparse", "ordinary", "high-count"],
)
def test_bmnd_poisson_numerical_stability(rate: float):
    rng = np.random.default_rng(0)
    volume = rng.poisson(rate, size=(8, 8)).astype(np.float32)
    output, sigma_maps = bmnd(
        volume,
        _make_profile("poisson"),
        sigma=np.sqrt(rate),
        return_sigma_map=True,
    )
    assert output.shape == volume.shape
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert np.isfinite(sigma_maps["ht"]).all()
    assert np.isfinite(sigma_maps["wiener"]).all()
