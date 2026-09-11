import numpy as np
import pytest

import bmnd.patchaggregation as pa
from bmnd.patchaggregation import (
    _get_flat_patch_offsets,
    compute_origins_numba,
    get_normalized_group_occurrence_weights_prepared,
    lin_to_multi_numba,
    prepare_patch_aggregation,
    scatter_add_patch_numerator_prepared,
    scatter_add_patches,
    scatter_add_patches_prepared,
    scatter_add_patches_with_scalar_sigma_prepared,
    scatter_add_patches_with_sigma_prepared,
)


def _scatter_add_scalar_reference(
    volume_shape: tuple[int, ...],
    counts: tuple[int, ...],
    block_size: tuple[int, ...],
    group_filt: np.ndarray,
    weights: np.ndarray,
    sel_global: np.ndarray,
    window: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    accum_num = np.zeros(volume_shape, dtype=np.float32)
    accum_den = np.zeros(volume_shape, dtype=np.float32)
    group_flat = group_filt.reshape(group_filt.shape[0], -1)
    window_flat = window.reshape(-1)

    for group_idx, selected_idx in enumerate(sel_global):
        origin = np.unravel_index(int(selected_idx), counts)
        for patch_idx, relative_idx in enumerate(np.ndindex(block_size)):
            output_idx = tuple(
                origin[dim] + relative_idx[dim] for dim in range(len(volume_shape))
            )
            weighted_window = weights[group_idx] * window_flat[patch_idx]
            accum_num[output_idx] += group_flat[group_idx, patch_idx] * weighted_window
            accum_den[output_idx] += weighted_window

    return accum_num, accum_den


def test_lin_to_multi_numba_matches_known_mapping():
    counts = np.array([2, 3], dtype=np.int64)
    assert np.array_equal(
        lin_to_multi_numba(0, counts), np.array([0, 0], dtype=np.int64)
    )
    assert np.array_equal(
        lin_to_multi_numba(4, counts), np.array([1, 1], dtype=np.int64)
    )
    assert np.array_equal(
        lin_to_multi_numba(5, counts), np.array([1, 2], dtype=np.int64)
    )


def test_compute_origins_numba_vectorized_mapping():
    sel = np.array([0, 3, 5], dtype=np.int64)
    counts = np.array([2, 3], dtype=np.int64)
    origins = compute_origins_numba(sel, counts)

    expected = np.array([[0, 0], [1, 0], [1, 2]], dtype=np.int64)
    assert np.array_equal(origins, expected)


def test_get_flat_patch_offsets_2d_correctness_and_cache_hit():
    pa._PATCH_OFFSETS_CACHE.clear()
    offsets1 = _get_flat_patch_offsets((2, 2), (3, 3))
    cache_len_after_first = len(pa._PATCH_OFFSETS_CACHE)
    offsets2 = _get_flat_patch_offsets((2, 2), (3, 3))

    # C-order offsets for a 2x2 patch in a 3x3 volume: [0, 1, 3, 4]
    assert np.array_equal(offsets1, np.array([0, 1, 3, 4], dtype=np.int64))
    assert cache_len_after_first == 1
    assert len(pa._PATCH_OFFSETS_CACHE) == 1
    assert offsets1 is offsets2


def test_scatter_add_patches_weighted_overlap_matches_expected():
    accum_num = np.zeros((3, 3), dtype=np.float32)
    accum_den = np.zeros((3, 3), dtype=np.float32)

    group_filt = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
        ],
        dtype=np.float32,
    )
    weights = np.array([1.0, 0.5], dtype=np.float32)
    sel_global = np.array([0, 3], dtype=np.int64)

    scatter_add_patches(
        accum_num=accum_num,
        accum_den=accum_den,
        group_filt=group_filt,
        weights=weights,
        sel_global=sel_global,
        counts=(2, 2),
        block_size=(2, 2),
        w_patch=np.ones((2, 2), dtype=np.float32),
    )

    np.testing.assert_array_equal(accum_num, [[1, 2, 0], [3, 9, 10], [0, 15, 20]])
    np.testing.assert_array_equal(accum_den, [[1, 1, 0], [1, 1.5, 0.5], [0, 0.5, 0.5]])


def test_scatter_add_patches_uses_kaiser_window_once_in_both_accumulators():
    accum_num = np.zeros((2, 2), dtype=np.float32)
    accum_den = np.zeros((2, 2), dtype=np.float32)
    window = np.array([[1.0, 0.5], [0.25, 0.75]], dtype=np.float32)

    scatter_add_patches(
        accum_num=accum_num,
        accum_den=accum_den,
        group_filt=np.full((1, 2, 2), 3.0, dtype=np.float32),
        weights=np.array([2.0], dtype=np.float32),
        sel_global=np.array([0], dtype=np.int64),
        counts=(1, 1),
        block_size=(2, 2),
        w_patch=window,
    )

    assert np.allclose(accum_num, 6.0 * window)
    assert np.allclose(accum_den, 2.0 * window)


def test_normalized_occurrence_weights_reconstruct_final_image_sum() -> None:
    volume_shape = (3, 3)
    counts = (2, 2)
    block_size = (2, 2)
    window = np.array([[1.0, 0.5], [0.25, 0.75]], dtype=np.float32)
    selected_groups = (
        np.array([0, 3], dtype=np.int64),
        np.array([1, 2], dtype=np.int64),
    )
    group_weights = (
        np.array([2.0, 0.5], dtype=np.float32),
        np.array([1.5, 0.75], dtype=np.float32),
    )
    group_values = (
        np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2),
        np.arange(9, 17, dtype=np.float32).reshape(2, 2, 2),
    )
    plan = prepare_patch_aggregation(volume_shape, counts, block_size, window)
    numerator = np.zeros(volume_shape, dtype=np.float32)
    denominator = np.zeros(volume_shape, dtype=np.float32)
    for values, weights, selected in zip(
        group_values,
        group_weights,
        selected_groups,
    ):
        scatter_add_patches_prepared(
            numerator,
            denominator,
            values,
            weights,
            selected,
            plan,
        )

    contribution_sum = 0.0
    rebuilt_numerator = np.zeros_like(numerator)
    for values, weights, selected in zip(
        group_values,
        group_weights,
        selected_groups,
    ):
        occurrence_weights = get_normalized_group_occurrence_weights_prepared(
            denominator,
            weights,
            selected,
            plan,
        ).reshape(values.shape)
        contribution_sum += float(np.sum(occurrence_weights * values))
        scatter_add_patch_numerator_prepared(
            rebuilt_numerator,
            values,
            weights,
            selected,
            plan,
        )

    assert np.array_equal(rebuilt_numerator, numerator)
    assert np.isclose(contribution_sum, np.sum(numerator / denominator), rtol=1e-6)


def test_scatter_add_patches_invalid_ndim_raises():
    with pytest.raises(ValueError, match="group_filt ndim"):
        scatter_add_patches(
            accum_num=np.zeros((3, 3), dtype=np.float32),
            accum_den=np.zeros((3, 3), dtype=np.float32),
            group_filt=np.zeros((1, 2, 2, 2), dtype=np.float32),
            weights=np.array([1.0], dtype=np.float32),
            sel_global=np.array([0], dtype=np.int64),
            counts=(2, 2),
            block_size=(2, 2),
            w_patch=np.ones((2, 2), dtype=np.float32),
        )


def test_scatter_add_patches_empty_group_noop():
    accum_num = np.arange(9, dtype=np.float32).reshape(3, 3)
    accum_den = np.full((3, 3), 2.0, dtype=np.float32)
    original_num = accum_num.copy()
    original_den = accum_den.copy()

    scatter_add_patches(
        accum_num=accum_num,
        accum_den=accum_den,
        group_filt=np.zeros((0, 2, 2), dtype=np.float32),
        weights=np.zeros((0,), dtype=np.float32),
        sel_global=np.zeros((0,), dtype=np.int64),
        counts=(2, 2),
        block_size=(2, 2),
        w_patch=np.ones((2, 2), dtype=np.float32),
    )

    assert np.array_equal(accum_num, original_num)
    assert np.array_equal(accum_den, original_den)


@pytest.mark.parametrize(
    ("volume_shape", "block_size", "selected"),
    [
        ((5, 6), (2, 3), [5, 0, 10, 5]),
        ((4, 5, 4), (2, 2, 2), [17, 0, 35, 17]),
    ],
)
def test_randomized_scatter_matches_scalar_reference(
    volume_shape: tuple[int, ...],
    block_size: tuple[int, ...],
    selected: list[int],
) -> None:
    rng = np.random.default_rng(123)
    counts = tuple(
        volume_shape[dim] - block_size[dim] + 1 for dim in range(len(volume_shape))
    )
    group_shape = (len(selected), *block_size)
    group_filt = rng.normal(size=group_shape).astype(np.float32)
    weights = rng.uniform(0.1, 2.0, size=len(selected)).astype(np.float32)
    sel_global = np.asarray(selected, dtype=np.int64)
    window = rng.uniform(0.2, 1.0, size=block_size).astype(np.float32)
    expected_num, expected_den = _scatter_add_scalar_reference(
        volume_shape,
        counts,
        block_size,
        group_filt,
        weights,
        sel_global,
        window,
    )

    direct_num = np.zeros(volume_shape, dtype=np.float32)
    direct_den = np.zeros(volume_shape, dtype=np.float32)
    scatter_add_patches(
        direct_num,
        direct_den,
        group_filt,
        weights,
        sel_global,
        counts,
        block_size,
        window,
    )

    prepared_num = np.zeros(volume_shape, dtype=np.float32)
    prepared_den = np.zeros(volume_shape, dtype=np.float32)
    prepared = prepare_patch_aggregation(volume_shape, counts, block_size, window)
    scatter_add_patches_prepared(
        prepared_num,
        prepared_den,
        group_filt,
        weights,
        sel_global,
        prepared,
    )

    assert np.array_equal(direct_num, expected_num)
    assert np.array_equal(direct_den, expected_den)
    assert np.array_equal(prepared_num, expected_num)
    assert np.array_equal(prepared_den, expected_den)


@pytest.mark.parametrize("scalar_sigma", [False, True])
def test_fused_sigma_scatter_matches_scalar_references(scalar_sigma: bool) -> None:
    volume_shape = (5, 6)
    block_size = (2, 3)
    counts = (4, 4)
    selected = np.array([5, 0, 10, 5], dtype=np.int64)
    rng = np.random.default_rng(321)
    group_filt = rng.normal(size=(4, *block_size)).astype(np.float32)
    weights = rng.uniform(0.1, 2.0, size=4).astype(np.float32)
    window = rng.uniform(0.2, 1.0, size=block_size).astype(np.float32)
    sigma = np.float32(0.125)
    if scalar_sigma:
        group_sigma = np.full_like(group_filt, sigma)
    else:
        group_sigma = rng.uniform(0.01, 0.5, size=group_filt.shape).astype(np.float32)

    expected_num, expected_den = _scatter_add_scalar_reference(
        volume_shape,
        counts,
        block_size,
        group_filt,
        weights,
        selected,
        window,
    )
    expected_sigma_num, expected_sigma_den = _scatter_add_scalar_reference(
        volume_shape,
        counts,
        block_size,
        group_sigma,
        weights,
        selected,
        window,
    )
    prepared = prepare_patch_aggregation(
        volume_shape,
        counts,
        block_size,
        window,
    )
    actual_num = np.zeros(volume_shape, dtype=np.float32)
    actual_den = np.zeros(volume_shape, dtype=np.float32)
    actual_sigma_num = np.zeros(volume_shape, dtype=np.float32)

    if scalar_sigma:
        scatter_add_patches_with_scalar_sigma_prepared(
            actual_num,
            actual_den,
            actual_sigma_num,
            group_filt,
            sigma,
            weights,
            selected,
            prepared,
        )
    else:
        scatter_add_patches_with_sigma_prepared(
            actual_num,
            actual_den,
            actual_sigma_num,
            group_filt,
            group_sigma,
            weights,
            selected,
            prepared,
        )

    assert np.array_equal(actual_num, expected_num)
    assert np.array_equal(actual_den, expected_den)
    assert np.array_equal(actual_sigma_num, expected_sigma_num)
    assert np.array_equal(actual_den, expected_sigma_den)
