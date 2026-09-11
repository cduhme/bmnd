from typing import cast

import numpy as np
from numpy.typing import NDArray

from bmnd.bmndalgo import bmnd
from bmnd.profiles import BMNDProfile
from bmnd.transforms import Transform, TransformMode, TransformType


def _make_profile() -> BMNDProfile:
    profile = BMNDProfile(
        noise_model="gaussian",
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


def test_bmnd_handles_non_contiguous_input():
    rng = np.random.default_rng(6)
    volume = rng.normal(0.0, 0.1, size=(8, 8)).astype(np.float32).T
    profile = _make_profile()

    result = bmnd(volume, profile, sigma=0.1)
    contiguous = bmnd(np.ascontiguousarray(volume), profile, sigma=0.1)

    np.testing.assert_array_equal(result, contiguous)


def test_bmnd_constant_input_preserves_value_with_zero_sigma():
    volume = np.full((8, 8), 2.0, dtype=np.float32)
    profile = _make_profile()

    result = cast(NDArray[np.float32], bmnd(volume, profile, sigma=0.0))

    np.testing.assert_allclose(result, volume)


def test_bmnd_soft_thresholding_shrinks_constant_patch_dc():
    volume = np.full((8, 8), 2.0, dtype=np.float32)
    profile = _make_profile()
    profile.k = 0
    profile.ht_search_window = (0, 0)
    profile.ht_min_stack_size = 1
    profile.ht_max_stack_size = 1
    profile.ht_use_soft_thresholding = True
    stages = {}

    bmnd(
        volume,
        profile,
        sigma=0.1,
        stage_callback=lambda stage, value: stages.update({stage: value.copy()}),
    )

    # A constant 4x4 orthonormal DCT patch has DC = 4 * pixel value.
    expected = 2.0 - profile.ht_lambda_threshold * 0.1 / 4.0
    np.testing.assert_allclose(stages["ht"], expected, atol=1e-6)


def test_bmnd_deterministic_for_fixed_input():
    rng = np.random.default_rng(10)
    volume = (0.5 + rng.normal(0.0, 0.1, size=(8, 8))).astype(np.float32)
    profile = _make_profile()

    result1 = bmnd(volume, profile, sigma=0.1)
    result2 = bmnd(volume, profile, sigma=0.1)

    np.testing.assert_array_equal(result1, result2)
