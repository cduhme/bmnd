from typing import Callable, cast

from numba import get_num_threads, set_num_threads
import numpy as np
import pytest

import bmnd.variance as variance_module
import bmnd.wiener as wiener_module
from bmnd.cache import get_poisson_cache_limits
from bmnd.poisson import _get_group_mass_functional
from bmnd.psd import (
    get_kernel_from_psd,
    process_psd_for_nf,
    preprocess_psd,
    resolve_nf,
    scalar_sigma_to_psd,
)
from bmnd.transforms import (
    Transform,
    TransformMode,
    TransformType,
    get_transform_matrix,
)
from bmnd.variance import (
    build_exact_poisson_group_plan,
    build_poisson_variance_model,
    build_variance_model,
    get_direct_poisson_noise_std,
    get_exact_noise_std,
    get_exact_poisson_contribution_matrix,
    get_exact_poisson_covariance_factors,
    get_gaussian_blockmatch_noise_ssd,
    get_stationary_gaussian_group_noise_variance,
    limit_poisson_group_variance_planes,
)
from bmnd.wiener import (
    compute_approximate_poisson_wiener_group_risk,
    compute_exact_poisson_wiener_group_risk,
    compute_exact_poisson_wiener_group_risk_from_plan,
    compute_poisson_wiener_gain,
    compute_poisson_wiener_risk,
    compute_poisson_wiener_signal_power,
)


def _dense_exact_poisson_variance(
    variance_volume: np.ndarray,
    selected_abs_positions: np.ndarray,
    block_shape: tuple[int, ...],
    forward: list[np.ndarray | Callable | None],
) -> np.ndarray:
    group_size = selected_abs_positions.shape[0]
    local_coords = np.stack(
        np.meshgrid(
            *[np.arange(size, dtype=np.int64) for size in block_shape], indexing="ij"
        ),
        axis=-1,
    ).reshape(-1, len(block_shape))
    patch_vol = local_coords.shape[0]

    group_matrix = (
        np.eye(group_size, dtype=np.float32)
        if forward[0] is None
        else np.asarray(forward[0], dtype=np.float32)
    )
    spatial_matrix = np.array([[1.0]], dtype=np.float32)
    for axis_size, transform in zip(block_shape, forward[1:], strict=False):
        matrix = (
            np.eye(axis_size, dtype=np.float32)
            if transform is None
            else np.asarray(transform, dtype=np.float32)
        )
        spatial_matrix = np.kron(spatial_matrix, matrix).astype(np.float32, copy=False)
    full_transform = np.kron(group_matrix, spatial_matrix).astype(np.float32, copy=False)

    entry_coords = (
        selected_abs_positions[:, None, :] + local_coords[None, :, :]
    ).reshape(group_size * patch_vol, len(block_shape))
    covariance = np.zeros((group_size * patch_vol, group_size * patch_vol), dtype=np.float32)
    for i in range(entry_coords.shape[0]):
        for j in range(entry_coords.shape[0]):
            if np.array_equal(entry_coords[i], entry_coords[j]):
                covariance[i, j] = variance_volume[tuple(entry_coords[i])]

    transformed = full_transform @ covariance @ full_transform.T
    return np.diag(transformed).reshape((group_size,) + block_shape)


def _dense_exact_poisson_group_risk(
    variance_volume: np.ndarray,
    selected_abs_positions: np.ndarray,
    block_shape: tuple[int, ...],
    forward: list[np.ndarray | Callable | None],
    inverse: list[np.ndarray | Callable | None],
    gain: np.ndarray,
    signal_power: np.ndarray,
    w_patch: np.ndarray,
    exact_plane_count: int | None = None,
    noise_sigma: np.ndarray | None = None,
    mass_functional: np.ndarray | None = None,
) -> float:
    group_size = selected_abs_positions.shape[0]
    local_coords = np.stack(
        np.meshgrid(
            *[np.arange(size, dtype=np.int64) for size in block_shape], indexing="ij"
        ),
        axis=-1,
    ).reshape(-1, len(block_shape))
    patch_vol = local_coords.shape[0]

    group_matrix = (
        np.eye(group_size, dtype=np.float32)
        if forward[0] is None
        else np.asarray(forward[0], dtype=np.float32)
    )
    inv_group_matrix = (
        np.eye(group_size, dtype=np.float32)
        if inverse[0] is None
        else np.asarray(inverse[0], dtype=np.float32)
    )

    spatial_matrix = np.array([[1.0]], dtype=np.float32)
    inv_spatial_matrix = np.array([[1.0]], dtype=np.float32)
    for axis_size, forward_transform, inverse_transform in zip(
        block_shape,
        forward[1:],
        inverse[1:],
        strict=False,
    ):
        matrix = (
            np.eye(axis_size, dtype=np.float32)
            if forward_transform is None
            else np.asarray(forward_transform, dtype=np.float32)
        )
        inverse_matrix = (
            np.eye(axis_size, dtype=np.float32)
            if inverse_transform is None
            else np.asarray(inverse_transform, dtype=np.float32)
        )
        spatial_matrix = np.kron(spatial_matrix, matrix).astype(np.float32, copy=False)
        inv_spatial_matrix = np.kron(inv_spatial_matrix, inverse_matrix).astype(
            np.float32,
            copy=False,
        )

    full_transform = np.kron(group_matrix, spatial_matrix).astype(np.float32, copy=False)
    full_inverse = np.kron(inv_group_matrix, inv_spatial_matrix).astype(
        np.float32,
        copy=False,
    )

    entry_coords = (
        selected_abs_positions[:, None, :] + local_coords[None, :, :]
    ).reshape(group_size * patch_vol, len(block_shape))
    covariance = np.zeros((group_size * patch_vol, group_size * patch_vol), dtype=np.float32)
    for i in range(entry_coords.shape[0]):
        for j in range(entry_coords.shape[0]):
            if np.array_equal(entry_coords[i], entry_coords[j]):
                covariance[i, j] = variance_volume[tuple(entry_coords[i])]

    transformed_covariance = full_transform @ covariance @ full_transform.T
    if exact_plane_count is not None:
        plane_count = min(exact_plane_count, group_size)
        exact_coefficient_count = plane_count * patch_vol
        hybrid_covariance = np.zeros_like(transformed_covariance)
        hybrid_covariance[:exact_coefficient_count, :exact_coefficient_count] = (
            transformed_covariance[:exact_coefficient_count, :exact_coefficient_count]
        )
        if plane_count < group_size:
            assert noise_sigma is not None
            approximate_variance = (
                np.asarray(noise_sigma, dtype=np.float32).reshape(-1) ** 2
            )
            remainder = np.arange(exact_coefficient_count, group_size * patch_vol)
            hybrid_covariance[remainder, remainder] = approximate_variance[remainder]
        transformed_covariance = hybrid_covariance
    gain_flat = np.asarray(gain, dtype=np.float32).reshape(-1)
    signal_power_flat = np.asarray(signal_power, dtype=np.float32).reshape(-1)
    gain_matrix = np.diag(gain_flat)
    if mass_functional is None:
        filter_matrix = gain_matrix
    else:
        mass = np.asarray(mass_functional, dtype=np.float32).reshape(-1)
        projection = np.outer(mass, mass) / np.sum(mass**2)
        filter_matrix = gain_matrix + projection @ (
            np.eye(gain_flat.size, dtype=np.float32) - gain_matrix
        )
    bias_operator = filter_matrix - np.eye(gain_flat.size, dtype=np.float32)
    bias_matrix = bias_operator @ np.diag(signal_power_flat) @ bias_operator.T
    spatial_weights = np.tile((np.asarray(w_patch, dtype=np.float32) ** 2).reshape(-1), group_size)
    window_matrix = np.diag(spatial_weights)
    noise_term = np.trace(
        window_matrix
        @ full_inverse
        @ filter_matrix
        @ transformed_covariance
        @ filter_matrix.T
        @ full_inverse.T
    )
    bias_term = np.trace(window_matrix @ full_inverse @ bias_matrix @ full_inverse.T)
    return float(noise_term + bias_term)


def test_build_variance_model_from_scalar_sigma():
    volume = np.zeros((4, 4), dtype=np.float32)

    model = build_variance_model(volume, sigma=0.2)

    assert model is not None
    assert np.isclose(model.global_sigma, 0.2)
    assert model.sigma_psd.shape == volume.shape
    assert np.allclose(model.sigma_psd, scalar_sigma_to_psd(volume.shape, 0.2))


def test_build_variance_model_estimates_sigma_when_not_provided():
    volume = np.tile(np.arange(4, dtype=np.float32), (8, 2))

    model = build_variance_model(volume)

    assert model is not None
    expected_sigma = 1.0 / 0.6745
    assert model.global_sigma == pytest.approx(expected_sigma)
    assert model.sigma_psd.shape == volume.shape
    assert np.allclose(model.sigma_psd, volume.size * expected_sigma**2)


def test_build_variance_model_from_flat_psd_matches_scalar_sigma():
    volume = np.zeros((4, 4), dtype=np.float32)
    flat_psd = scalar_sigma_to_psd(volume.shape, 0.3)

    model = build_variance_model(volume, sigma_psd=flat_psd)

    assert model is not None
    assert np.isclose(model.global_sigma, 0.3)
    assert np.allclose(model.sigma_psd, flat_psd)
    assert np.allclose(model.processed_sigma_psd, flat_psd)
    assert np.allclose(model.reduced_sigma_psd, flat_psd)


def test_build_variance_model_rejects_sigma_and_sigma_psd():
    volume = np.zeros((4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_variance_model(volume, sigma=0.2, sigma_psd=np.ones_like(volume))


@pytest.mark.parametrize("sigma", [-0.1, np.nan, np.inf, True])
def test_scalar_sigma_to_psd_rejects_invalid_sigma(sigma: object) -> None:
    with pytest.raises(ValueError, match="sigma must be finite and non-negative"):
        scalar_sigma_to_psd((4, 4), sigma)  # type: ignore[arg-type]


@pytest.mark.parametrize("variance_floor", [-0.1, np.nan, np.inf, True])
def test_build_poisson_variance_model_rejects_invalid_floor(
    variance_floor: float,
) -> None:
    with pytest.raises(ValueError, match="variance_floor.*finite and non-negative"):
        build_poisson_variance_model(
            np.ones((4, 4), dtype=np.float32),
            variance_floor=variance_floor,
            source_name="test",
        )


@pytest.mark.parametrize(
    ("nf", "expected", "conventional"),
    [
        (None, None, False),
        ((), None, False),
        ((4, 6), (4, 6), False),
        (0, (8, 16), True),
        ((0, 6), (8, 16), True),
    ],
)
def test_resolve_nf_contract(
    nf: tuple[int, ...] | int | None,
    expected: tuple[int, ...] | None,
    conventional: bool,
):
    assert resolve_nf(nf, (8, 24)) == (expected, conventional)


@pytest.mark.parametrize(
    "nf",
    [
        True,
        4,
        (4,),
        (0,),
        (4, -1),
        (0, -1),
        (4, 1.5),
        (4, False),
    ],
)
def test_resolve_nf_rejects_malformed_values(nf: object):
    with pytest.raises(ValueError, match="nf"):
        resolve_nf(nf, (8, 24))


def test_build_variance_model_resolves_conventional_nf_domain():
    model = build_variance_model(
        np.zeros((8, 24), dtype=np.float32),
        sigma=0.2,
        nf=0,
    )

    assert model is not None
    assert model.variance_domain_shape == (8, 16)


def test_gaussian_blockmatch_noise_ssd_matches_white_noise_bias():
    sigma = 0.2
    block_shape = (2, 3)
    search_radius = (1, 2)
    model = build_variance_model(np.zeros((8, 8), dtype=np.float32), sigma=sigma)
    assert model is not None

    noise_ssd = get_gaussian_blockmatch_noise_ssd(
        model,
        block_shape,
        search_radius,
    )

    expected = 2.0 * np.prod(block_shape) * sigma**2
    assert noise_ssd.shape == (3, 5)
    assert np.isclose(noise_ssd[search_radius], 0.0)
    nonzero_lags = np.ones(noise_ssd.shape, dtype=bool)
    nonzero_lags[search_radius] = False
    assert np.allclose(noise_ssd[nonzero_lags], expected, atol=1e-6)


@pytest.mark.parametrize("exact_plane_count", [1, 4, 8])
@pytest.mark.parametrize("patch_volume", [17, 1025])
def test_gaussian_exact_plane_variance_preserves_serial_reduction(
    exact_plane_count: int,
    patch_volume: int,
) -> None:
    rng = np.random.default_rng(42)
    covariance_maps = rng.normal(size=(patch_volume, 32)).astype(np.float32)
    pair_lags = rng.integers(0, 32, size=(8, 8), dtype=np.int64)
    rows = rng.normal(size=(exact_plane_count, 8)).astype(np.float32)
    rows[rng.random(rows.shape) < 0.3] = 0.0
    if exact_plane_count > 1:
        rows[-1] = 0.0
    indices = np.full(rows.shape, -1, dtype=np.int64)
    counts = np.count_nonzero(rows, axis=1).astype(np.int64)
    for plane_index in range(exact_plane_count):
        indices[plane_index, : counts[plane_index]] = np.flatnonzero(rows[plane_index])

    # Preserve the original plane-major, float32 triangular accumulation as oracle.
    expected = np.empty((exact_plane_count, patch_volume), dtype=np.float32)
    for plane_index in range(exact_plane_count):
        for coefficient_index in range(patch_volume):
            covariance_map = covariance_maps[coefficient_index]
            value = np.float32(0.0)
            for first_offset in range(counts[plane_index]):
                first = indices[plane_index, first_offset]
                first_weight = rows[plane_index, first]
                value += (
                    first_weight * covariance_map[pair_lags[first, first]] * first_weight
                )
                for second_offset in range(first_offset):
                    second = indices[plane_index, second_offset]
                    second_weight = rows[plane_index, second]
                    covariance_pair = (
                        covariance_map[pair_lags[first, second]]
                        + covariance_map[pair_lags[second, first]]
                    )
                    value += first_weight * second_weight * covariance_pair
            expected[plane_index, coefficient_index] = max(value, np.float32(0.0))

    original_threads = get_num_threads()
    try:
        for threads in sorted({1, min(4, original_threads)}):
            set_num_threads(threads)
            actual = variance_module._accumulate_gaussian_exact_plane_variance_numba(
                covariance_maps, pair_lags, rows, indices, counts
            )
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
            assert actual.tobytes() == expected.tobytes()
    finally:
        set_num_threads(original_threads)


@pytest.mark.parametrize("k", [0, 1, 2, 4])
def test_stationary_gaussian_group_variance_matches_dense_overlap_oracle(k: int):
    sigma = 0.2
    group_shape = (2, 3)
    positions = np.array([[0], [1]], dtype=np.int64)
    model = build_variance_model(np.zeros((8,), dtype=np.float32), sigma=sigma)
    assert model is not None
    forward, _ = get_transform_matrix(
        group_shape,
        Transform(TransformType.HAAR, TransformMode.GROUP),
    )

    variance = get_stationary_gaussian_group_noise_variance(
        group_shape,
        forward,
        model,
        positions,
        k,
    )

    entry_positions = (
        positions[:, None, :] + np.arange(group_shape[1], dtype=np.int64)[None, :, None]
    ).reshape(-1)
    covariance = sigma**2 * (
        entry_positions[:, None] == entry_positions[None, :]
    ).astype(np.float32)
    full_transform = np.kron(
        np.asarray(forward[0], dtype=np.float32),
        np.eye(group_shape[1], dtype=np.float32),
    )
    exact = np.diag(full_transform @ covariance @ full_transform.T).reshape(group_shape)

    if k == 0:
        expected = np.full(group_shape, sigma**2, dtype=np.float32)
    elif k == 1:
        expected = np.empty(group_shape, dtype=np.float32)
        expected[0] = exact[0]
        expected[1] = 2.0 * sigma**2 - exact[0]
    else:
        expected = exact
    assert np.allclose(variance, expected, atol=1e-6)
    assert np.allclose(np.sum(variance, axis=0), 2.0 * sigma**2, atol=1e-6)


def test_stationary_gaussian_group_variance_rescales_white_psd_to_nf_domain():
    sigma = 0.2
    group_shape = (2, 2)
    model = build_variance_model(
        np.zeros((12,), dtype=np.float32),
        sigma=sigma,
        nf=(16,),
    )
    assert model is not None
    forward, _ = get_transform_matrix(
        group_shape,
        Transform(TransformType.HAAR, TransformMode.GROUP),
    )

    variance = get_stationary_gaussian_group_noise_variance(
        group_shape,
        forward,
        model,
        np.array([[0], [4]], dtype=np.int64),
        k=2,
    )

    assert model.variance_domain_shape == (16,)
    assert np.allclose(variance, sigma**2, atol=1e-6)


def test_stationary_gaussian_group_variance_matches_colored_dense_oracle():
    group_shape = (2, 2)
    positions = np.array([[0], [2]], dtype=np.int64)
    frequencies = np.arange(8, dtype=np.float32)
    sigma_psd = 0.32 * (
        1.0 + 0.5 * np.cos(2.0 * np.pi * frequencies / frequencies.size)
    )
    model = build_variance_model(
        np.zeros((8,), dtype=np.float32),
        sigma_psd=sigma_psd.astype(np.float32),
    )
    assert model is not None
    forward, _ = get_transform_matrix(
        group_shape,
        Transform(TransformType.HAAR, TransformMode.GROUP),
    )

    variance = get_stationary_gaussian_group_noise_variance(
        group_shape,
        forward,
        model,
        positions,
        k=2,
    )

    entry_positions = (
        positions[:, None, :] + np.arange(group_shape[1], dtype=np.int64)[None, :, None]
    ).reshape(-1)
    lag = (entry_positions[:, None] - entry_positions[None, :]) % 8
    covariance = model.autocovariance[lag]
    full_transform = np.kron(
        np.asarray(forward[0], dtype=np.float32),
        np.eye(group_shape[1], dtype=np.float32),
    )
    expected = np.diag(full_transform @ covariance @ full_transform.T).reshape(group_shape)

    assert np.allclose(variance, expected, atol=1e-6)


def test_limit_poisson_group_variance_planes_respects_k():
    exact_sigma = np.sqrt(np.array([[1.5], [0.5]], dtype=np.float32))
    approximate_sigma = np.ones((2, 1), dtype=np.float32)

    k_zero = limit_poisson_group_variance_planes(exact_sigma, approximate_sigma, k=0)
    k_one = limit_poisson_group_variance_planes(exact_sigma, approximate_sigma, k=1)
    k_full = limit_poisson_group_variance_planes(exact_sigma, approximate_sigma, k=2)
    k_one_partial = limit_poisson_group_variance_planes(
        exact_sigma[:1],
        approximate_sigma,
        k=1,
    )

    assert np.allclose(k_zero, approximate_sigma)
    assert np.allclose(k_one**2, np.array([[1.5], [0.5]], dtype=np.float32))
    assert np.allclose(k_one_partial, k_one)
    assert np.allclose(k_full, exact_sigma)


def test_build_variance_model_processes_colored_psd_for_nf():
    volume = np.zeros((12, 12, 12), dtype=np.float32)
    grids = np.meshgrid(*[np.fft.fftfreq(size) for size in volume.shape], indexing="ij")
    colored_psd = np.zeros_like(volume, dtype=np.float32)
    for grid in grids:
        colored_psd += (grid * grid).astype(np.float32)
    colored_psd = 1.0 + 5.0 * colored_psd

    model = build_variance_model(volume, sigma_psd=colored_psd, nf=(4, 4, 4))

    assert model is not None
    assert model.processed_sigma_psd.shape == volume.shape
    assert model.reduced_sigma_psd.shape == volume.shape
    assert not np.allclose(model.processed_sigma_psd, model.sigma_psd)
    assert model.blur_kernel.ndim == 3
    assert np.isclose(np.sum(model.blur_kernel, dtype=np.float32), 1.0)


def test_process_psd_for_nf_preserves_flat_psd_structure():
    sigma_psd = np.full((8, 8, 8), 3.5, dtype=np.float32)

    processed = process_psd_for_nf(sigma_psd, (4, 4, 4))

    assert processed.shape == sigma_psd.shape
    assert np.allclose(processed, 3.5, atol=1e-6)


def test_psd_preprocessing_uses_shared_anisotropic_reduction_without_aliasing():
    frequencies = np.meshgrid(
        np.fft.fftfreq(48),
        np.fft.fftfreq(18),
        indexing="ij",
    )
    sigma_psd = (
        1.0 + 3.0 * frequencies[0] ** 2 + 7.0 * frequencies[1] ** 2
    ).astype(np.float32)
    original = sigma_psd.copy()

    processed = preprocess_psd(sigma_psd, (2, 6), single_dim_psd=False)
    wrapper_result = process_psd_for_nf(sigma_psd, (2, 6))

    assert processed.reduced_sigma_psd.shape == (16, 18)
    assert processed.blurred_sigma_psd.shape == (16, 18)
    assert np.allclose(wrapper_result, processed.blurred_sigma_psd, atol=1e-6)
    assert np.array_equal(sigma_psd, original)
    assert not np.shares_memory(processed.sigma_psd, sigma_psd)
    assert not np.shares_memory(processed.reduced_sigma_psd, sigma_psd)
    assert not np.shares_memory(processed.blurred_sigma_psd, sigma_psd)


def test_get_kernel_from_psd_matches_white_noise_scalar_level():
    sigma_psd = scalar_sigma_to_psd((8, 8), 0.1)

    kernel = get_kernel_from_psd(sigma_psd)

    assert kernel.shape == sigma_psd.shape
    center = tuple(size // 2 for size in kernel.shape)
    assert np.isclose(kernel[center], 0.1, atol=1e-6)
    zeroed = kernel.copy()
    zeroed[center] = 0.0
    assert np.allclose(zeroed, 0.0, atol=1e-6)


def test_exact_noise_std_is_constant_for_orthonormal_dct():
    shape = (2, 4, 4)
    model = build_variance_model(np.zeros(shape[1:], dtype=np.float32), sigma=0.1)
    assert model is not None
    forward, _ = get_transform_matrix(
        shape, Transform(TransformType.DCT, TransformMode.ND)
    )

    sigma_std = get_exact_noise_std(shape, forward, model)

    assert sigma_std.shape == shape
    assert np.allclose(sigma_std, 0.1, atol=1e-6)


def test_exact_noise_std_reflects_bior15_row_energy():
    shape = (2, 8, 8)
    model = build_variance_model(np.zeros(shape[1:], dtype=np.float32), sigma=0.1)
    assert model is not None
    forward, _ = get_transform_matrix(
        shape,
        [
            Transform(TransformType.HAAR, TransformMode.GROUP),
            Transform(TransformType.BIOR15, TransformMode.SPATIAL),
        ],
    )

    sigma_std = get_exact_noise_std(shape, forward, model)

    assert sigma_std.shape == shape
    row_energies = [np.sum(np.asarray(matrix) ** 2, axis=1) for matrix in forward]
    expected_variance = (
        0.1**2
        * row_energies[0][:, None, None]
        * row_energies[1][None, :, None]
        * row_energies[2][None, None, :]
    )
    assert np.allclose(sigma_std**2, expected_variance, rtol=1e-6, atol=0.0)


def test_build_poisson_variance_model_clips_negative_values():
    volume = np.array([[4.0, -2.0], [0.0, 9.0]], dtype=np.float32)

    model = build_poisson_variance_model(
        volume,
        variance_floor=0.25,
        source_name="observation",
    )

    np.testing.assert_array_equal(model.variance_volume, [[4.0, 0.25], [0.25, 9.0]])
    assert model.source_name == "observation"


def test_build_poisson_variance_model_applies_cache_limits() -> None:
    limits = get_poisson_cache_limits(80 * 1024 * 1024)

    model = build_poisson_variance_model(
        np.ones((2, 2), dtype=np.float32),
        source_name="observation",
        cache_limits=limits,
    )

    assert model.overlap_cache_max_bytes == limits.overlap_max_bytes
    assert model.overlap_cache_max_entries == limits.overlap_max_entries
    assert model.contribution_cache_max_bytes == limits.contribution_max_bytes
    assert model.contribution_cache_max_entries == limits.contribution_max_entries
    assert model.factorized_cache_max_bytes == limits.factorized_max_bytes
    assert model.factorized_cache_max_entries == limits.factorized_max_entries
    assert (
        model.factorized_entry_cache_max_bytes
        == limits.factorized_entry_max_bytes
    )


@pytest.mark.parametrize("scale", [0.0, -1.0, np.inf, np.nan])
def test_build_poisson_variance_model_rejects_invalid_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="scale must be finite and positive"):
        build_poisson_variance_model(
            np.ones((2, 2), dtype=np.float32),
            scale=scale,
            source_name="observation",
        )


def test_build_poisson_variance_model_estimates_sparse_global_sigma_from_positive_counts():
    volume = np.zeros((8, 8), dtype=np.float32)
    volume[0, 0] = 9.0
    volume[1, 1] = 16.0
    volume[2, 2] = 25.0

    model = build_poisson_variance_model(
        volume,
        variance_floor=0.0,
        source_name="observation",
    )

    assert model.global_sigma == pytest.approx(4.0)
    assert np.isclose(model.variance_volume[0, 0], 9.0)
    assert np.isclose(model.variance_volume[1, 1], 16.0)
    assert np.isclose(model.variance_volume[2, 2], 25.0)


def test_direct_poisson_noise_std_matches_spatial_variance_for_identity_transform():
    group_variance = np.array(
        [
            [[1.0, 4.0], [9.0, 16.0]],
            [[25.0, 36.0], [49.0, 64.0]],
        ],
        dtype=np.float32,
    )
    forward = cast(list[np.ndarray | Callable | None], [None, None, None])

    sigma_std = get_direct_poisson_noise_std(group_variance, forward)

    assert sigma_std.shape == group_variance.shape
    assert np.allclose(sigma_std**2, group_variance)


def test_direct_poisson_noise_std_k_zero_does_not_touch_exact_covariance(monkeypatch):
    group_variance = np.array([[1.0, 2.0], [2.0, 5.0]], dtype=np.float32)
    forward = cast(list[np.ndarray | Callable | None], [None, None])

    def reject_exact_covariance(*args, **kwargs):
        raise AssertionError("k=0 must not inspect exact covariance metadata")

    monkeypatch.setattr(
        variance_module,
        "get_exact_poisson_covariance_factors",
        reject_exact_covariance,
    )

    sigma = get_direct_poisson_noise_std(
        group_variance,
        forward,
        k=0,
        selected_abs_positions=None,
        selected_shifted_positions=None,
    )

    assert np.allclose(sigma**2, group_variance)


def test_direct_poisson_noise_std_exact_matches_approximate_without_overlap():
    variance_volume = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    model = build_poisson_variance_model(
        variance_volume,
        source_name="observation",
    )
    selected_abs_positions = np.array([[0], [2]], dtype=np.int64)
    selected_shifted_positions = np.array([[0], [2]], dtype=np.int64)
    group_variance = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    haar = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    dct2 = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    forward = cast(list[np.ndarray | Callable | None], [haar, dct2])

    sigma_approx = get_direct_poisson_noise_std(group_variance, forward)
    sigma_exact = get_direct_poisson_noise_std(
        group_variance,
        forward,
        k=2,
        variance_model=model,
        selected_abs_positions=selected_abs_positions,
        selected_shifted_positions=selected_shifted_positions,
        group_shape=group_variance.shape,
    )

    assert np.allclose(sigma_exact, sigma_approx, atol=1e-6)


@pytest.mark.parametrize("k", [2, 3], ids=["full", "overfull"])
def test_direct_poisson_noise_std_exact_matches_dense_covariance_diagonal(k: int):
    variance_volume = np.array([1.0, 2.0, 5.0], dtype=np.float32)
    model = build_poisson_variance_model(
        variance_volume,
        source_name="observation",
    )
    selected_abs_positions = np.array([[0], [1]], dtype=np.int64)
    selected_shifted_positions = np.array([[0], [1]], dtype=np.int64)
    group_variance = np.array([[1.0, 2.0], [2.0, 5.0]], dtype=np.float32)
    haar = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    forward = cast(list[np.ndarray | Callable | None], [haar, haar])

    sigma_exact = get_direct_poisson_noise_std(
        group_variance,
        forward,
        k=k,
        variance_model=model,
        selected_abs_positions=selected_abs_positions,
        selected_shifted_positions=selected_shifted_positions,
        group_shape=group_variance.shape,
    )
    dense_variance = _dense_exact_poisson_variance(
        variance_volume,
        selected_abs_positions,
        (2,),
        forward,
    )

    assert np.allclose(sigma_exact**2, dense_variance, atol=1e-6)


@pytest.mark.parametrize(
    ("coefficient_count", "voxel_count"),
    [(1, 4), (64, 0), (64, 4), (256, 4), (500, 4)],
)
def test_compact_poisson_variance_preserves_serial_reduction(
    coefficient_count: int,
    voxel_count: int,
) -> None:
    rng = np.random.default_rng(42)
    lengths = rng.integers(1, 5, size=voxel_count, dtype=np.int64)
    starts = np.cumsum(lengths) - lengths
    entry_count = int(np.sum(lengths))
    entry_transform = rng.normal(size=(entry_count, coefficient_count)).astype(np.float32)
    entry_indices = rng.permutation(entry_count).astype(np.int64)
    union_variance = rng.uniform(0.0, 12.0, size=voxel_count).astype(np.float32)
    if voxel_count:
        union_variance[0] = 0.0
    expected = np.zeros(coefficient_count, dtype=np.float32)
    for voxel_id in range(union_variance.size):
        contribution = np.zeros(coefficient_count, dtype=np.float32)
        for pos in range(starts[voxel_id], starts[voxel_id] + lengths[voxel_id]):
            contribution += entry_transform[entry_indices[pos]]
        expected += union_variance[voxel_id] * contribution * contribution

    kernel = variance_module._accumulate_exact_poisson_variance_from_entry_transform_numba
    original_threads = get_num_threads()
    try:
        for threads in sorted({1, min(4, original_threads)}):
            set_num_threads(threads)
            actual = kernel(
                entry_transform, entry_indices, starts, lengths, union_variance
            )
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
            assert actual.tobytes() == expected.tobytes()
    finally:
        set_num_threads(original_threads)


def test_direct_poisson_noise_std_exact_k_uses_compact_entry_transform():
    variance_volume = np.arange(1.0, 7.0, dtype=np.float32)
    model = build_poisson_variance_model(
        variance_volume,
        source_name="observation",
    )
    selected_abs_positions = np.arange(4, dtype=np.int64).reshape(-1, 1)
    selected_shifted_positions = selected_abs_positions.copy()
    group_variance = np.stack(
        [variance_volume[start : start + 2] for start in range(4)],
    )
    forward, _inverse = get_transform_matrix(
        group_variance.shape,
        [
            Transform(TransformType.HAAR, TransformMode.GROUP),
            Transform(TransformType.DCT, TransformMode.SPATIAL),
        ],
    )

    approximate_sigma = get_direct_poisson_noise_std(group_variance, forward)
    full_model = build_poisson_variance_model(
        variance_volume,
        source_name="observation",
    )
    full_exact_sigma = get_direct_poisson_noise_std(
        group_variance,
        forward,
        k=4,
        variance_model=full_model,
        selected_abs_positions=selected_abs_positions,
        selected_shifted_positions=selected_shifted_positions,
        group_shape=group_variance.shape,
    )
    compact_sigma = get_direct_poisson_noise_std(
        group_variance,
        forward,
        k=1,
        variance_model=model,
        selected_abs_positions=selected_abs_positions,
        selected_shifted_positions=selected_shifted_positions,
        group_shape=group_variance.shape,
    )
    expected = limit_poisson_group_variance_planes(
        full_exact_sigma,
        approximate_sigma,
        k=1,
    )

    assert np.allclose(compact_sigma, expected, atol=1e-6)
    compact_transforms = list(model.compact_entry_transform_t_cache.values())
    assert any(transform.shape == (8, 2) for transform in compact_transforms)
    assert all(transform.shape != (8, 8) for transform in compact_transforms)


def test_exact_poisson_wiener_group_risk_matches_diagonal_without_overlap():
    variance_volume = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    model = build_poisson_variance_model(
        variance_volume,
        source_name="observation",
    )
    selected_abs_positions = np.array([[0], [2]], dtype=np.int64)
    selected_shifted_positions = np.array([[0], [2]], dtype=np.int64)
    reference = np.array([[0.0, 0.3], [0.6, 1.5]], dtype=np.float32)
    sigma = np.sqrt(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)).astype(np.float32)
    forward = cast(list[np.ndarray | Callable | None], [None, None])
    inverse = cast(list[np.ndarray | Callable | None], [None, None])
    w_patch = np.ones((2,), dtype=np.float32)

    gain = compute_poisson_wiener_gain(
        reference,
        sigma,
        wiener_variance_scale=1.0,
        mode="classic",
    )
    signal_power = compute_poisson_wiener_signal_power(
        reference,
        sigma,
        wiener_variance_scale=1.0,
        mode="classic",
    )

    exact_risk = compute_exact_poisson_wiener_group_risk(
        gain,
        signal_power,
        model,
        selected_abs_positions,
        selected_shifted_positions,
        reference.shape,
        forward,
        inverse,
        w_patch,
    )
    diagonal_risk = float(
        np.sum(
            compute_poisson_wiener_risk(
                reference,
                sigma,
                wiener_variance_scale=1.0,
                mode="classic",
            ),
            dtype=np.float32,
        )
    )

    assert np.isclose(exact_risk, diagonal_risk, atol=1e-6)


def test_approximate_poisson_wiener_group_risk_matches_dense_metric_oracle():
    variance_volume = np.array([1.0, 2.0, 5.0], dtype=np.float32)
    positions = np.array([[0], [1]], dtype=np.int64)
    forward = cast(list[np.ndarray | Callable | None], [None, None])
    inverse_group = np.array([[1.0, 0.25], [0.0, 1.5]], dtype=np.float32)
    inverse_spatial = np.array([[1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
    inverse = cast(
        list[np.ndarray | Callable | None],
        [inverse_group, inverse_spatial],
    )
    gain = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    signal_power = np.array([[0.3, 0.5], [0.7, 0.9]], dtype=np.float32)
    sigma = np.sqrt(np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32))
    w_patch = np.array([1.0, 0.4], dtype=np.float32)

    risk = compute_approximate_poisson_wiener_group_risk(
        gain,
        signal_power,
        sigma,
        inverse,
        w_patch,
    )
    dense_risk = _dense_exact_poisson_group_risk(
        variance_volume,
        positions,
        (2,),
        forward,
        inverse,
        gain,
        signal_power,
        w_patch,
        exact_plane_count=0,
        noise_sigma=sigma,
    )

    assert np.isclose(risk, dense_risk, atol=1e-6)


def test_mass_projected_approximate_poisson_risk_matches_dense_metric_oracle():
    variance_volume = np.array([1.0, 2.0, 5.0], dtype=np.float32)
    positions = np.array([[0], [1]], dtype=np.int64)
    forward = cast(list[np.ndarray | Callable | None], [None, None])
    inverse = cast(
        list[np.ndarray | Callable | None],
        [
            np.array([[1.0, 0.25], [0.0, 1.5]], dtype=np.float32),
            np.array([[1.0, 0.0], [0.5, 1.0]], dtype=np.float32),
        ],
    )
    gain = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    signal_power = np.array([[0.3, 0.5], [0.7, 0.9]], dtype=np.float32)
    sigma = np.sqrt(np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32))
    w_patch = np.array([1.0, 0.4], dtype=np.float32)
    mass_functional = _get_group_mass_functional(gain.shape, inverse)

    risk = compute_approximate_poisson_wiener_group_risk(
        gain,
        signal_power,
        sigma,
        inverse,
        w_patch,
        mass_functional=mass_functional,
    )
    dense_risk = _dense_exact_poisson_group_risk(
        variance_volume,
        positions,
        (2,),
        forward,
        inverse,
        gain,
        signal_power,
        w_patch,
        exact_plane_count=0,
        noise_sigma=sigma,
        mass_functional=mass_functional,
    )

    assert np.isclose(risk, dense_risk, atol=1e-6)


def test_partial_poisson_wiener_group_risk_matches_dense_hybrid_oracle():
    variance_volume = np.arange(1.0, 7.0, dtype=np.float32)
    model = build_poisson_variance_model(
        variance_volume,
        source_name="observation",
    )
    positions = np.arange(4, dtype=np.int64).reshape(-1, 1)
    group_variance = np.stack(
        [variance_volume[start : start + 2] for start in range(4)],
    )
    forward, inverse = get_transform_matrix(
        group_variance.shape,
        [
            Transform(TransformType.HAAR, TransformMode.GROUP),
            Transform(TransformType.DCT, TransformMode.SPATIAL),
        ],
    )
    inverse[0] = np.array(
        [
            [1.0, 0.3, 0.0, 0.0],
            [0.0, 1.0, 0.2, 0.0],
            [0.0, 0.0, 1.0, 0.4],
            [0.1, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    plan = build_exact_poisson_group_plan(
        model,
        positions,
        positions,
        group_variance.shape,
        forward,
    )
    approximate_sigma = get_direct_poisson_noise_std(group_variance, forward, k=0)
    exact_sigma = plan.prepare_noise_std_for_risk(1)
    limited_sigma = limit_poisson_group_variance_planes(
        exact_sigma,
        approximate_sigma,
        k=1,
    )
    reference = np.linspace(0.2, 1.6, num=8, dtype=np.float32).reshape(4, 2)
    gain = compute_poisson_wiener_gain(
        reference,
        limited_sigma,
        wiener_variance_scale=1.0,
        mode="classic",
    )
    signal_power = compute_poisson_wiener_signal_power(
        reference,
        limited_sigma,
        wiener_variance_scale=1.0,
        mode="classic",
    )
    w_patch = np.array([1.0, 0.4], dtype=np.float32)

    assert not plan.is_group_metric_diagonal(inverse, w_patch)

    risk = compute_exact_poisson_wiener_group_risk_from_plan(
        gain,
        signal_power,
        plan,
        inverse,
        w_patch,
        exact_plane_count=1,
        noise_sigma=limited_sigma,
    )
    dense_risk = _dense_exact_poisson_group_risk(
        variance_volume,
        positions,
        (2,),
        forward,
        inverse,
        gain,
        signal_power,
        w_patch,
        exact_plane_count=1,
        noise_sigma=limited_sigma,
    )
    mass_functional = _get_group_mass_functional(reference.shape, inverse)
    mass_projected_risk = compute_exact_poisson_wiener_group_risk_from_plan(
        gain,
        signal_power,
        plan,
        inverse,
        w_patch,
        exact_plane_count=1,
        noise_sigma=limited_sigma,
        mass_functional=mass_functional,
    )
    dense_mass_projected_risk = _dense_exact_poisson_group_risk(
        variance_volume,
        positions,
        (2,),
        forward,
        inverse,
        gain,
        signal_power,
        w_patch,
        exact_plane_count=1,
        noise_sigma=limited_sigma,
        mass_functional=mass_functional,
    )

    assert np.isclose(risk, dense_risk, atol=1e-5)
    assert np.isclose(mass_projected_risk, dense_mass_projected_risk, atol=1e-5)


def test_exact_poisson_wiener_group_risk_matches_dense_reference_with_overlap():
    variance_volume = np.array([1.0, 2.0, 5.0], dtype=np.float32)
    model = build_poisson_variance_model(
        variance_volume,
        source_name="observation",
    )
    selected_abs_positions = np.array([[0], [1]], dtype=np.int64)
    selected_shifted_positions = np.array([[0], [1]], dtype=np.int64)
    reference = np.array([[0.0, 0.3], [0.6, 1.5]], dtype=np.float32)
    sigma = get_direct_poisson_noise_std(
        np.array([[1.0, 2.0], [2.0, 5.0]], dtype=np.float32),
        cast(list[np.ndarray | Callable | None], [None, None]),
        k=2,
        variance_model=model,
        selected_abs_positions=selected_abs_positions,
        selected_shifted_positions=selected_shifted_positions,
        group_shape=reference.shape,
    )
    forward = cast(list[np.ndarray | Callable | None], [None, None])
    inverse = cast(list[np.ndarray | Callable | None], [None, None])
    w_patch = np.array([1.0, 0.5], dtype=np.float32)

    gain = compute_poisson_wiener_gain(
        reference,
        sigma,
        wiener_variance_scale=1.0,
        mode="variance-scaled",
    )
    signal_power = compute_poisson_wiener_signal_power(
        reference,
        sigma,
        wiener_variance_scale=1.0,
        mode="variance-scaled",
    )

    exact_risk = compute_exact_poisson_wiener_group_risk(
        gain,
        signal_power,
        model,
        selected_abs_positions,
        selected_shifted_positions,
        reference.shape,
        forward,
        inverse,
        w_patch,
    )
    dense_risk = _dense_exact_poisson_group_risk(
        variance_volume,
        selected_abs_positions,
        (2,),
        forward,
        inverse,
        gain,
        signal_power,
        w_patch,
    )
    mass_functional = _get_group_mass_functional(reference.shape, inverse)
    mass_projected_risk = compute_exact_poisson_wiener_group_risk(
        gain,
        signal_power,
        model,
        selected_abs_positions,
        selected_shifted_positions,
        reference.shape,
        forward,
        inverse,
        w_patch,
        mass_functional=mass_functional,
    )
    dense_mass_projected_risk = _dense_exact_poisson_group_risk(
        variance_volume,
        selected_abs_positions,
        (2,),
        forward,
        inverse,
        gain,
        signal_power,
        w_patch,
        mass_functional=mass_functional,
    )

    assert exact_risk >= 0.0
    assert np.isclose(exact_risk, dense_risk, atol=1e-6)
    assert np.isclose(mass_projected_risk, dense_mass_projected_risk, atol=1e-6)


def test_exact_poisson_plan_contributions_and_noise_std_match_split_builders():
    variance_volume = np.arange(1.0, 65.0, dtype=np.float32).reshape(8, 8)
    model = build_poisson_variance_model(
        variance_volume,
        source_name="observation",
    )
    selected_abs_positions = np.array(
        [[2, 2], [2, 3], [3, 2], [3, 3]],
        dtype=np.int64,
    )
    selected_shifted_positions = selected_abs_positions - np.min(
        selected_abs_positions,
        axis=0,
        keepdims=True,
    )
    group_shape = (4, 2, 2)
    forward, _inverse = get_transform_matrix(
        group_shape,
        [
            Transform(TransformType.HAAR, TransformMode.GROUP),
            Transform(TransformType.DCT, TransformMode.SPATIAL),
        ],
    )
    plan = build_exact_poisson_group_plan(
        model,
        selected_abs_positions,
        selected_shifted_positions,
        group_shape,
        forward,
    )

    contributions_from_plan, sigma_from_plan = plan.get_contributions_and_noise_std()
    grouped_from_plan = plan.get_grouped_contributions()

    split_plan = build_exact_poisson_group_plan(
        model,
        selected_abs_positions,
        selected_shifted_positions,
        group_shape,
        forward,
    )
    contributions_split = split_plan.get_contributions()
    sigma_split = split_plan.get_noise_std_from_factors()

    assert contributions_from_plan.shape == contributions_split.shape
    assert grouped_from_plan.shape == (plan.union_variance.shape[0], group_shape[0], 4)
    assert np.allclose(contributions_from_plan, contributions_split, atol=1e-6)
    assert np.allclose(sigma_from_plan, sigma_split, atol=1e-6)
    assert np.allclose(
        sigma_from_plan,
        np.sqrt(
            _dense_exact_poisson_variance(
                variance_volume,
                selected_abs_positions,
                group_shape[1:],
                forward,
            )
        ),
        atol=1e-5,
    )


@pytest.mark.parametrize("k", [2, 3], ids=["full", "overfull"])
def test_exact_poisson_wiener_group_risk_from_plan_matches_overlap_reference(k: int):
    variance_volume = np.array([1.0, 2.0, 5.0], dtype=np.float32)
    model = build_poisson_variance_model(
        variance_volume,
        source_name="observation",
    )
    selected_abs_positions = np.array([[0], [1]], dtype=np.int64)
    selected_shifted_positions = np.array([[0], [1]], dtype=np.int64)
    reference = np.array([[0.0, 0.3], [0.6, 1.5]], dtype=np.float32)
    forward = cast(list[np.ndarray | Callable | None], [None, None])
    inverse = cast(list[np.ndarray | Callable | None], [None, None])
    w_patch = np.array([1.0, 0.5], dtype=np.float32)
    plan = build_exact_poisson_group_plan(
        model,
        selected_abs_positions,
        selected_shifted_positions,
        reference.shape,
        forward,
    )
    sigma = plan.prepare_noise_std_for_risk(k)

    gain = compute_poisson_wiener_gain(
        reference,
        sigma,
        wiener_variance_scale=1.0,
        mode="variance-scaled",
    )
    signal_power = compute_poisson_wiener_signal_power(
        reference,
        sigma,
        wiener_variance_scale=1.0,
        mode="variance-scaled",
    )

    exact_risk_from_plan = compute_exact_poisson_wiener_group_risk_from_plan(
        gain,
        signal_power,
        plan,
        inverse,
        w_patch,
        exact_plane_count=k,
        noise_sigma=sigma,
    )
    exact_risk = compute_exact_poisson_wiener_group_risk(
        gain,
        signal_power,
        model,
        selected_abs_positions,
        selected_shifted_positions,
        reference.shape,
        forward,
        inverse,
        w_patch,
        plan=plan,
    )
    dense_risk = _dense_exact_poisson_group_risk(
        variance_volume,
        selected_abs_positions,
        (2,),
        forward,
        inverse,
        gain,
        signal_power,
        w_patch,
    )

    assert np.isclose(exact_risk_from_plan, exact_risk, atol=1e-6)
    assert np.isclose(exact_risk_from_plan, dense_risk, atol=1e-6)


def test_exact_poisson_plan_prepares_only_requested_planes() -> None:
    variance_volume = np.arange(1.0, 65.0, dtype=np.float32).reshape(8, 8)
    model = build_poisson_variance_model(
        variance_volume,
        source_name="observation",
    )
    selected_abs_positions = np.array(
        [[2, 2], [2, 3], [3, 2], [3, 3]],
        dtype=np.int64,
    )
    selected_shifted_positions = selected_abs_positions - np.min(
        selected_abs_positions,
        axis=0,
        keepdims=True,
    )
    group_shape = (4, 2, 2)
    forward, _ = get_transform_matrix(
        group_shape,
        [
            Transform(TransformType.HAAR, TransformMode.GROUP),
            Transform(TransformType.DCT, TransformMode.SPATIAL),
        ],
    )
    plan = build_exact_poisson_group_plan(
        model,
        selected_abs_positions,
        selected_shifted_positions,
        group_shape,
        forward,
    )

    sigma = plan.prepare_noise_std_for_risk(2)
    assert plan.prepared_risk_contributions_cache is not None
    prepared = plan.prepared_risk_contributions_cache.copy()
    full = plan.get_contribution_chunk(0, plan.union_variance.shape[0])

    assert sigma.shape == (2, 2, 2)
    assert prepared.shape == (plan.union_variance.shape[0], 2, 4)
    assert plan.prepared_risk_plane_count == 2
    assert full.shape == (plan.union_variance.shape[0], 4, 4)
    assert np.array_equal(full[:, :2], prepared)


def test_covariance_aware_variance_risk_supports_zero_exact_planes() -> None:
    variance_volume = np.array([1.0, 2.0, 5.0], dtype=np.float32)
    model = build_poisson_variance_model(
        variance_volume,
        source_name="observation",
    )
    positions = np.array([[0], [1]], dtype=np.int64)
    gain = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    transforms = cast(list[np.ndarray | Callable | None], [None, None])
    window = np.array([1.0, 0.5], dtype=np.float32)
    noise_sigma = np.full_like(gain, 0.75)
    plan = build_exact_poisson_group_plan(
        model,
        positions,
        positions,
        gain.shape,
        transforms,
    )

    actual = wiener_module.compute_covariance_aware_group_risk_from_plan(
        gain,
        plan,
        transforms,
        window,
        exact_plane_count=0,
        noise_sigma=noise_sigma,
    )
    expected = compute_approximate_poisson_wiener_group_risk(
        gain,
        np.zeros_like(gain),
        noise_sigma,
        transforms,
        window,
    )

    assert actual == pytest.approx(expected)


def test_exact_poisson_plan_diagonal_group_risk_matches_dense_reference():
    variance_volume = np.arange(1.0, 65.0, dtype=np.float32).reshape(8, 8)
    model = build_poisson_variance_model(
        variance_volume,
        source_name="observation",
    )
    selected_abs_positions = np.array(
        [[2, 2], [2, 3], [3, 2], [3, 3]],
        dtype=np.int64,
    )
    selected_shifted_positions = selected_abs_positions - np.min(
        selected_abs_positions,
        axis=0,
        keepdims=True,
    )
    reference = np.linspace(-0.6, 1.0, num=16, dtype=np.float32).reshape(4, 2, 2)
    w_patch = np.ones((2, 2), dtype=np.float32)
    forward, inverse = get_transform_matrix(
        reference.shape,
        [
            Transform(TransformType.HAAR, TransformMode.GROUP),
            Transform(TransformType.DCT, TransformMode.SPATIAL),
        ],
    )
    plan = build_exact_poisson_group_plan(
        model,
        selected_abs_positions,
        selected_shifted_positions,
        reference.shape,
        forward,
    )
    _, sigma = plan.get_contributions_and_noise_std()

    gain = compute_poisson_wiener_gain(
        reference,
        sigma,
        wiener_variance_scale=1.0,
        mode="classic",
    )
    signal_power = compute_poisson_wiener_signal_power(
        reference,
        sigma,
        wiener_variance_scale=1.0,
        mode="classic",
    )

    assert plan.is_group_metric_diagonal(inverse, w_patch)

    exact_risk_from_plan = compute_exact_poisson_wiener_group_risk_from_plan(
        gain,
        signal_power,
        plan,
        inverse,
        w_patch,
    )
    exact_risk = compute_exact_poisson_wiener_group_risk(
        gain,
        signal_power,
        model,
        selected_abs_positions,
        selected_shifted_positions,
        reference.shape,
        forward,
        inverse,
        w_patch,
        plan=plan,
    )
    dense_risk = _dense_exact_poisson_group_risk(
        variance_volume,
        selected_abs_positions,
        (2, 2),
        forward,
        inverse,
        gain,
        signal_power,
        w_patch,
    )
    mass_functional = _get_group_mass_functional(reference.shape, inverse)
    partial_mass_risk = compute_exact_poisson_wiener_group_risk_from_plan(
        gain,
        signal_power,
        plan,
        inverse,
        w_patch,
        exact_plane_count=2,
        noise_sigma=sigma,
        mass_functional=mass_functional,
    )
    dense_partial_mass_risk = _dense_exact_poisson_group_risk(
        variance_volume,
        selected_abs_positions,
        (2, 2),
        forward,
        inverse,
        gain,
        signal_power,
        w_patch,
        exact_plane_count=2,
        noise_sigma=sigma,
        mass_functional=mass_functional,
    )

    assert np.isclose(exact_risk_from_plan, exact_risk, atol=1e-6)
    assert np.isclose(exact_risk_from_plan, dense_risk, atol=1e-5)
    assert np.isclose(partial_mass_risk, dense_partial_mass_risk, atol=1e-5)


def test_partial_exact_mass_risk_handles_active_tail_planes_and_chunks(
    monkeypatch,
) -> None:
    variance_volume = np.arange(1.0, 7.0, dtype=np.float32)
    model = build_poisson_variance_model(
        variance_volume,
        source_name="observation",
    )
    positions = np.arange(4, dtype=np.int64)[:, None]
    group_shape = (4, 2)
    transforms = cast(list[np.ndarray | Callable | None], [None, None])
    plan = build_exact_poisson_group_plan(
        model,
        positions,
        positions,
        group_shape,
        transforms,
    )
    plan.prepare_noise_std_for_risk(1)
    full_sigma = plan.get_noise_std_from_factors()
    gain = np.linspace(0.1, 0.8, num=8, dtype=np.float32).reshape(group_shape)
    signal_power = np.linspace(0.2, 1.6, num=8, dtype=np.float32).reshape(
        group_shape
    )
    window = np.array([1.0, 0.5], dtype=np.float32)
    mass_functional = _get_group_mass_functional(group_shape, transforms)
    assert np.flatnonzero(np.any(mass_functional[1:] != 0.0, axis=1)).size > 0
    monkeypatch.setattr(wiener_module, "_EXACT_POISSON_RISK_WORKSPACE_BYTES", 1)

    actual = compute_exact_poisson_wiener_group_risk_from_plan(
        gain,
        signal_power,
        plan,
        transforms,
        window,
        exact_plane_count=1,
        noise_sigma=full_sigma,
        mass_functional=mass_functional,
    )
    expected = _dense_exact_poisson_group_risk(
        variance_volume,
        positions,
        (2,),
        transforms,
        transforms,
        gain,
        signal_power,
        window,
        exact_plane_count=1,
        noise_sigma=full_sigma,
        mass_functional=mass_functional,
    )

    assert actual == pytest.approx(expected, abs=1e-5)
    assert plan.prepared_risk_contributions_cache is None


def test_exact_poisson_plan_diagonal_risk_reuses_metrics_and_streams_chunks(monkeypatch):
    variance_volume = np.arange(1.0, 65.0, dtype=np.float32).reshape(8, 8)
    model = build_poisson_variance_model(
        variance_volume,
        source_name="observation",
    )
    selected_abs_positions = np.array(
        [[2, 2], [2, 3], [3, 2], [3, 3]],
        dtype=np.int64,
    )
    selected_shifted_positions = selected_abs_positions - np.min(
        selected_abs_positions,
        axis=0,
        keepdims=True,
    )
    reference = np.linspace(-0.6, 1.0, num=16, dtype=np.float32).reshape(4, 2, 2)
    w_patch = np.ones((2, 2), dtype=np.float32)
    forward, inverse = get_transform_matrix(
        reference.shape,
        [
            Transform(TransformType.HAAR, TransformMode.GROUP),
            Transform(TransformType.DCT, TransformMode.SPATIAL),
        ],
    )
    plan = build_exact_poisson_group_plan(
        model,
        selected_abs_positions,
        selected_shifted_positions,
        reference.shape,
        forward,
    )
    sigma = plan.prepare_noise_std_for_risk(reference.shape[0])

    gain = compute_poisson_wiener_gain(
        reference,
        sigma,
        wiener_variance_scale=1.0,
        mode="classic",
    )
    signal_power = compute_poisson_wiener_signal_power(
        reference,
        sigma,
        wiener_variance_scale=1.0,
        mode="classic",
    )

    prepared_first = plan.get_prepared_diagonal_risk_metrics(inverse, w_patch)
    prepared_second = plan.get_prepared_diagonal_risk_metrics(inverse, w_patch)

    assert prepared_first is not None
    assert prepared_second is not None
    assert prepared_first[0] is prepared_second[0]
    assert prepared_first[1] is prepared_second[1]
    assert prepared_first[2] is prepared_second[2]
    assert plan.contributions_cache is None
    assert plan.prepared_risk_contributions_cache is not None
    assert all(
        transform.shape != (16, 16)
        for transform in model.compact_entry_transform_t_cache.values()
    )
    explicit_contributions = plan.get_contributions()
    explicit_contributions_before_risk = explicit_contributions.copy()

    exact_risk_first = compute_exact_poisson_wiener_group_risk_from_plan(
        gain,
        signal_power,
        plan,
        inverse,
        w_patch,
    )
    assert plan.prepared_risk_contributions_cache is None
    assert np.array_equal(explicit_contributions, explicit_contributions_before_risk)
    monkeypatch.setattr(
        variance_module,
        "_EXACT_POISSON_PREPARED_CONTRIBUTION_MAX_BYTES",
        1,
    )
    fallback_sigma = plan.prepare_noise_std_for_risk(reference.shape[0])
    assert np.allclose(fallback_sigma, sigma, atol=1e-6)
    assert plan.prepared_risk_contributions_cache is None
    monkeypatch.setattr(wiener_module, "_EXACT_POISSON_RISK_WORKSPACE_BYTES", 1)
    exact_risk_second = compute_exact_poisson_wiener_group_risk_from_plan(
        gain,
        signal_power,
        plan,
        inverse,
        w_patch,
    )
    dense_risk = _dense_exact_poisson_group_risk(
        variance_volume,
        selected_abs_positions,
        (2, 2),
        forward,
        inverse,
        gain,
        signal_power,
        w_patch,
    )
    assert np.isclose(exact_risk_first, exact_risk_second, atol=1e-6)
    assert np.isclose(exact_risk_first, dense_risk, atol=1e-5)
    assert plan.contributions_cache is explicit_contributions


def test_exact_poisson_plan_streams_complete_nondiagonal_group_risk(monkeypatch):
    variance_volume = np.array([1.0, 2.0, 5.0], dtype=np.float32)
    model = build_poisson_variance_model(
        variance_volume,
        source_name="observation",
    )
    selected_abs_positions = np.array([[0], [1]], dtype=np.int64)
    selected_shifted_positions = selected_abs_positions.copy()
    reference = np.array([[0.2, 0.4], [0.7, 1.1]], dtype=np.float32)
    haar = (1.0 / np.sqrt(2.0)) * np.array(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=np.float32,
    )
    forward = cast(list[np.ndarray | Callable | None], [haar, haar])
    inverse_group = np.array([[1.0, 0.25], [0.0, 1.0]], dtype=np.float32)
    inverse = cast(list[np.ndarray | Callable | None], [inverse_group, haar.T])
    w_patch = np.array([1.0, 0.5], dtype=np.float32)
    model.factorized_entry_cache_max_bytes = 0
    plan = build_exact_poisson_group_plan(
        model,
        selected_abs_positions,
        selected_shifted_positions,
        reference.shape,
        forward,
    )
    sigma = plan.get_noise_std_from_factors()
    gain = compute_poisson_wiener_gain(
        reference,
        sigma,
        wiener_variance_scale=1.0,
        mode="classic",
    )
    signal_power = compute_poisson_wiener_signal_power(
        reference,
        sigma,
        wiener_variance_scale=1.0,
        mode="classic",
    )
    monkeypatch.setattr(wiener_module, "_EXACT_POISSON_RISK_WORKSPACE_BYTES", 1)

    assert not plan.is_group_metric_diagonal(inverse, w_patch)
    assert plan.scaled_spatial_entries.size == 0
    with pytest.raises(ValueError, match="outside the union-voxel range"):
        plan.get_contribution_chunk(-1, 1)
    with pytest.raises(ValueError, match="outside the union-voxel range"):
        plan.get_contribution_chunk(0, plan.union_variance.shape[0] + 1)
    risk = compute_exact_poisson_wiener_group_risk_from_plan(
        gain,
        signal_power,
        plan,
        inverse,
        w_patch,
    )
    dense_risk = _dense_exact_poisson_group_risk(
        variance_volume,
        selected_abs_positions,
        (2,),
        forward,
        inverse,
        gain,
        signal_power,
        w_patch,
    )

    assert np.isclose(risk, dense_risk, atol=1e-6)
    assert plan.contributions_cache is None


def test_exact_poisson_caches_are_bounded_and_instrumented():
    variance_volume = np.arange(1.0, 9.0, dtype=np.float32)
    model = build_poisson_variance_model(
        variance_volume,
        source_name="observation",
    )
    model.overlap_cache_max_entries = 1
    model.contribution_cache_max_entries = 1
    forward = cast(list[np.ndarray | Callable | None], [None, None])
    first_positions = np.array([[0], [1]], dtype=np.int64)
    second_positions = np.array([[0], [2]], dtype=np.int64)

    first_factors = get_exact_poisson_covariance_factors(
        model,
        first_positions,
        first_positions,
        (2, 2),
        forward,
    )
    get_exact_poisson_covariance_factors(
        model,
        first_positions + 1,
        first_positions,
        (2, 2),
        forward,
    )
    first_contributions = get_exact_poisson_contribution_matrix(
        model,
        first_positions,
        (2, 2),
        forward,
        covariance_factors=first_factors,
    )
    cached_contributions = get_exact_poisson_contribution_matrix(
        model,
        first_positions,
        (2, 2),
        forward,
        covariance_factors=first_factors,
    )
    second_factors = get_exact_poisson_covariance_factors(
        model,
        second_positions,
        second_positions,
        (2, 2),
        forward,
    )
    get_exact_poisson_contribution_matrix(
        model,
        second_positions,
        (2, 2),
        forward,
        covariance_factors=second_factors,
    )

    stats = model.covariance_cache_stats
    assert cached_contributions is first_contributions
    assert stats.overlap_hits == 1
    assert stats.overlap_misses == 2
    assert stats.overlap_evictions == 1
    assert stats.overlap_bytes <= model.overlap_cache_max_bytes
    assert len(model.overlap_structure_cache) == 1
    assert stats.contribution_hits == 1
    assert stats.contribution_misses == 2
    assert stats.contribution_evictions == 1
    assert stats.contribution_bytes <= model.contribution_cache_max_bytes
    assert len(model.exact_contribution_cache) == 1


def test_exact_poisson_wiener_group_risk_differs_from_diagonal_proxy_with_overlap():
    variance_volume = np.array([1.0, 2.0, 5.0], dtype=np.float32)

    model = build_poisson_variance_model(
        variance_volume,
        source_name="observation",
    )
    selected_abs_positions = np.array([[0], [1]], dtype=np.int64)
    selected_shifted_positions = np.array([[0], [1]], dtype=np.int64)
    reference = np.array([[0.0, 0.3], [0.6, 1.5]], dtype=np.float32)
    haar = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    forward = cast(list[np.ndarray | Callable | None], [haar, haar])
    inverse = cast(list[np.ndarray | Callable | None], [haar.T, haar.T])
    sigma_exact = get_direct_poisson_noise_std(
        np.array([[1.0, 2.0], [2.0, 5.0]], dtype=np.float32),
        forward,
        k=2,
        variance_model=model,
        selected_abs_positions=selected_abs_positions,
        selected_shifted_positions=selected_shifted_positions,
        group_shape=reference.shape,
    )
    w_patch = np.array([1.0, 0.5], dtype=np.float32)

    gain = compute_poisson_wiener_gain(
        reference,
        sigma_exact,
        wiener_variance_scale=1.0,
        mode="classic",
    )
    signal_power = compute_poisson_wiener_signal_power(
        reference,
        sigma_exact,
        wiener_variance_scale=1.0,
        mode="classic",
    )

    exact_risk = compute_exact_poisson_wiener_group_risk(
        gain,
        signal_power,
        model,
        selected_abs_positions,
        selected_shifted_positions,
        reference.shape,
        forward,
        inverse,
        w_patch,
    )
    diagonal_risk = float(
        np.sum(
            compute_poisson_wiener_risk(
                reference,
                sigma_exact,
                wiener_variance_scale=1.0,
                mode="classic",
            )
            * np.tile(w_patch**2, reference.shape[0]).reshape(reference.shape),
            dtype=np.float32,
        )
    )

    assert exact_risk > 0.0
    assert not np.isclose(exact_risk, diagonal_risk, atol=1e-6)


def test_direct_poisson_noise_std_exact_rejects_callable_transforms():
    variance_volume = np.array([1.0, 2.0, 5.0], dtype=np.float32)
    model = build_poisson_variance_model(
        variance_volume,
        source_name="observation",
    )
    selected_abs_positions = np.array([[0], [1]], dtype=np.int64)
    group_variance = np.array([[1.0, 2.0], [2.0, 5.0]], dtype=np.float32)
    forward = cast(list[np.ndarray | Callable | None], [lambda x: x, None])

    with pytest.raises(ValueError, match="requires matrix transforms"):
        get_direct_poisson_noise_std(
            group_variance,
            forward,
            k=2,
            variance_model=model,
            selected_abs_positions=selected_abs_positions,
            group_shape=group_variance.shape,
        )
