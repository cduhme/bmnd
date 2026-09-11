from dataclasses import dataclass
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

# Small cache for flat patch offsets keyed by (block_size, volume_shape)
_PATCH_OFFSETS_CACHE: dict[
    tuple[tuple[int, ...], tuple[int, ...]], NDArray[np.int64]
] = {}

_VOLUME_STRIDES_CACHE: dict[tuple[int, ...], NDArray[np.int64]] = {}


@dataclass(frozen=True)
class PatchAggregationPlan:
    """Prepared aggregation state reused across repeated group scatters."""

    counts_arr: NDArray[np.int64]
    flat_offsets: NDArray[np.int64]
    vol_strides: NDArray[np.int64]
    w_patch_flat: NDArray[np.float32]
    patch_vol: int


def prepare_patch_aggregation(
    vol_shape: tuple[int, ...],
    counts: tuple[int, ...],
    block_size: tuple[int, ...],
    w_patch: NDArray[np.float32],
) -> PatchAggregationPlan:
    counts_arr = np.asarray(counts, dtype=np.int64)
    flat_offsets = _get_flat_patch_offsets(block_size, vol_shape)
    vol_strides = _get_volume_strides(vol_shape)
    w_patch_flat = np.ascontiguousarray(
        np.asarray(w_patch, dtype=np.float32).reshape(-1)
    )
    patch_vol = int(flat_offsets.shape[0])
    return PatchAggregationPlan(
        counts_arr=counts_arr,
        flat_offsets=flat_offsets,
        vol_strides=vol_strides,
        w_patch_flat=w_patch_flat,
        patch_vol=patch_vol,
    )


def _prepare_group_filt_flat(
    group_filt: NDArray[np.float32],
    ndim: int,
    patch_vol: int,
) -> NDArray[np.float32]:
    if group_filt.ndim == 2:
        if group_filt.shape[1] != patch_vol:
            raise ValueError(
                "group_filt has flat shape incompatible with prepared patch volume"
            )
        return np.ascontiguousarray(group_filt, dtype=np.float32)

    if group_filt.ndim != ndim + 1:
        raise ValueError(
            f"group_filt ndim={group_filt.ndim} is inconsistent with counts ndim={ndim}"
        )

    return np.ascontiguousarray(
        group_filt.reshape(group_filt.shape[0], patch_vol),
        dtype=np.float32,
    )


@njit(cache=True)
def _gather_normalized_occurrence_weights_numba(
    denominator_flat: NDArray[np.float32],
    weights: NDArray[np.float32],
    sel_global: NDArray[np.int64],
    counts: NDArray[np.int64],
    vol_strides: NDArray[np.int64],
    flat_offsets: NDArray[np.int64],
    w_patch_flat: NDArray[np.float32],
) -> NDArray[np.float64]:
    output = np.empty((sel_global.shape[0], flat_offsets.shape[0]), dtype=np.float64)
    for group_idx in range(sel_global.shape[0]):
        idx = sel_global[group_idx]
        origin_lin = np.int64(0)
        for dim in range(counts.shape[0] - 1, -1, -1):
            coordinate = idx % counts[dim]
            idx //= counts[dim]
            origin_lin += coordinate * vol_strides[dim]
        for patch_idx in range(flat_offsets.shape[0]):
            flat_idx = origin_lin + flat_offsets[patch_idx]
            denominator = denominator_flat[flat_idx]
            if denominator <= 0.0:
                raise ValueError(
                    "overlap denominator must be positive at every occurrence"
                )
            output[group_idx, patch_idx] = (
                weights[group_idx] * w_patch_flat[patch_idx] / denominator
            )
    return output


@njit(cache=True)
def lin_to_multi_numba(idx: int, counts: NDArray[np.int64]) -> NDArray[np.int64]:
    ndim = counts.shape[0]
    out = np.empty(ndim, dtype=np.int64)
    for i in range(ndim - 1, -1, -1):
        out[i] = idx % counts[i]
        idx //= counts[i]
    return out


@njit(cache=True, parallel=True)
def compute_origins_numba(
    sel_global: NDArray[np.int64],
    counts: NDArray[np.int64],
) -> NDArray[np.int64]:
    G = sel_global.shape[0]
    ndim = counts.shape[0]
    origins = np.empty((G, ndim), dtype=np.int64)
    for i in prange(G):
        origins[i, :] = lin_to_multi_numba(sel_global[i], counts)
    return origins


def _get_flat_patch_offsets(
    block_size: tuple[int, ...],
    vol_shape: tuple[int, ...],
) -> NDArray[np.int64]:
    key = (tuple(block_size), tuple(vol_shape))
    if key in _PATCH_OFFSETS_CACHE:
        return _PATCH_OFFSETS_CACHE[key]

    ndim = len(block_size)

    # All relative coordinates inside the patch
    grids = np.meshgrid(
        *[np.arange(b, dtype=np.int64) for b in block_size], indexing="ij"
    )
    offsets = np.stack(grids, axis=-1).reshape(-1, ndim)  # (patch_vol, ndim)

    # C-order strides for vol_shape
    strides = np.empty(ndim, dtype=np.int64)
    stride = 1
    for d in range(ndim - 1, -1, -1):
        strides[d] = stride
        stride *= vol_shape[d]

    # Flat offset for each relative coordinate
    flat_offsets = offsets @ strides  # (patch_vol,)

    _PATCH_OFFSETS_CACHE[key] = flat_offsets
    return flat_offsets


def _get_volume_strides(vol_shape: tuple[int, ...]) -> NDArray[np.int64]:
    key = tuple(vol_shape)
    if key in _VOLUME_STRIDES_CACHE:
        return _VOLUME_STRIDES_CACHE[key]

    ndim = len(vol_shape)
    strides = np.empty(ndim, dtype=np.int64)
    stride = 1
    for d in range(ndim - 1, -1, -1):
        strides[d] = stride
        stride *= vol_shape[d]

    _VOLUME_STRIDES_CACHE[key] = strides
    return strides


@njit(cache=True)
def _scatter_add_patches_numba(
    accum_num_flat: NDArray[np.float32],
    accum_den_flat: NDArray[np.float32],
    group_filt_flat: NDArray[np.float32],
    weights: NDArray[np.float32],
    sel_global: NDArray[np.int64],
    counts: NDArray[np.int64],
    vol_strides: NDArray[np.int64],
    flat_offsets: NDArray[np.int64],
    w_patch_flat: NDArray[np.float32],
) -> None:
    ndim = counts.shape[0]
    patch_vol = flat_offsets.shape[0]

    for group_idx in range(sel_global.shape[0]):
        idx = sel_global[group_idx]
        origin_lin = np.int64(0)
        for dim in range(ndim - 1, -1, -1):
            coord = idx % counts[dim]
            idx //= counts[dim]
            origin_lin += coord * vol_strides[dim]
        weight = weights[group_idx]
        for patch_idx in range(patch_vol):
            weighted_window = weight * w_patch_flat[patch_idx]
            flat_idx = origin_lin + flat_offsets[patch_idx]
            accum_num_flat[flat_idx] += (
                group_filt_flat[group_idx, patch_idx] * weighted_window
            )
            accum_den_flat[flat_idx] += weighted_window


@njit(cache=True)
def _scatter_add_patch_numerator_numba(
    accum_num_flat: NDArray[np.float32],
    group_filt_flat: NDArray[np.float32],
    weights: NDArray[np.float32],
    sel_global: NDArray[np.int64],
    counts: NDArray[np.int64],
    vol_strides: NDArray[np.int64],
    flat_offsets: NDArray[np.int64],
    w_patch_flat: NDArray[np.float32],
) -> None:
    patch_vol = flat_offsets.shape[0]
    for group_idx in range(sel_global.shape[0]):
        idx = sel_global[group_idx]
        origin_lin = np.int64(0)
        for dim in range(counts.shape[0] - 1, -1, -1):
            coordinate = idx % counts[dim]
            idx //= counts[dim]
            origin_lin += coordinate * vol_strides[dim]
        weight = weights[group_idx]
        for patch_idx in range(patch_vol):
            flat_idx = origin_lin + flat_offsets[patch_idx]
            accum_num_flat[flat_idx] += group_filt_flat[group_idx, patch_idx] * (
                weight * w_patch_flat[patch_idx]
            )


@njit(cache=True)
def _scatter_add_patches_with_scalar_sigma_numba(
    accum_num_flat: NDArray[np.float32],
    accum_den_flat: NDArray[np.float32],
    sigma_num_flat: NDArray[np.float32],
    group_filt_flat: NDArray[np.float32],
    sigma: np.float32,
    weights: NDArray[np.float32],
    sel_global: NDArray[np.int64],
    counts: NDArray[np.int64],
    vol_strides: NDArray[np.int64],
    flat_offsets: NDArray[np.int64],
    w_patch_flat: NDArray[np.float32],
) -> None:
    patch_vol = flat_offsets.shape[0]
    for group_idx in range(sel_global.shape[0]):
        idx = sel_global[group_idx]
        origin_lin = np.int64(0)
        for dim in range(counts.shape[0] - 1, -1, -1):
            coordinate = idx % counts[dim]
            idx //= counts[dim]
            origin_lin += coordinate * vol_strides[dim]
        weight = weights[group_idx]
        for patch_idx in range(patch_vol):
            weighted_window = weight * w_patch_flat[patch_idx]
            flat_idx = origin_lin + flat_offsets[patch_idx]
            accum_num_flat[flat_idx] += (
                group_filt_flat[group_idx, patch_idx] * weighted_window
            )
            accum_den_flat[flat_idx] += weighted_window
            sigma_num_flat[flat_idx] += sigma * weighted_window


@njit(cache=True)
def _scatter_add_patches_with_sigma_numba(
    accum_num_flat: NDArray[np.float32],
    accum_den_flat: NDArray[np.float32],
    sigma_num_flat: NDArray[np.float32],
    group_filt_flat: NDArray[np.float32],
    group_sigma_flat: NDArray[np.float32],
    weights: NDArray[np.float32],
    sel_global: NDArray[np.int64],
    counts: NDArray[np.int64],
    vol_strides: NDArray[np.int64],
    flat_offsets: NDArray[np.int64],
    w_patch_flat: NDArray[np.float32],
) -> None:
    patch_vol = flat_offsets.shape[0]
    for group_idx in range(sel_global.shape[0]):
        idx = sel_global[group_idx]
        origin_lin = np.int64(0)
        for dim in range(counts.shape[0] - 1, -1, -1):
            coordinate = idx % counts[dim]
            idx //= counts[dim]
            origin_lin += coordinate * vol_strides[dim]
        weight = weights[group_idx]
        for patch_idx in range(patch_vol):
            weighted_window = weight * w_patch_flat[patch_idx]
            flat_idx = origin_lin + flat_offsets[patch_idx]
            accum_num_flat[flat_idx] += (
                group_filt_flat[group_idx, patch_idx] * weighted_window
            )
            accum_den_flat[flat_idx] += weighted_window
            sigma_num_flat[flat_idx] += (
                group_sigma_flat[group_idx, patch_idx] * weighted_window
            )


def _prepare_scatter_inputs(
    group_filt: NDArray[np.float32],
    weights: NDArray[np.float32],
    sel_global: NDArray[np.int64],
    prepared: PatchAggregationPlan,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]:
    group_filt_flat = _prepare_group_filt_flat(
        group_filt,
        int(prepared.counts_arr.size),
        prepared.patch_vol,
    )
    weights_arr = np.ascontiguousarray(weights, dtype=np.float32)
    selected_arr = np.ascontiguousarray(sel_global, dtype=np.int64)
    group_size = group_filt_flat.shape[0]
    if weights_arr.shape != (group_size,):
        raise ValueError("weights length must match the filtered group size")
    if selected_arr.shape != (group_size,):
        raise ValueError("sel_global length must match the filtered group size")
    return group_filt_flat, weights_arr, selected_arr


def get_normalized_group_occurrence_weights_prepared(
    overlap_denominator: NDArray[np.float32],
    weights: NDArray[np.float32],
    sel_global: NDArray[np.int64],
    prepared: PatchAggregationPlan,
) -> NDArray[np.float64]:
    weights_arr = np.ascontiguousarray(weights, dtype=np.float32)
    selected_arr = np.ascontiguousarray(sel_global, dtype=np.int64)
    if weights_arr.shape != selected_arr.shape:
        raise ValueError("weights and sel_global must have the same shape")
    num_patches = int(np.prod(prepared.counts_arr, dtype=np.int64))
    if np.any(selected_arr < 0) or np.any(selected_arr >= num_patches):
        raise ValueError("sel_global contains an index outside the patch grid")
    denominator_flat = np.ascontiguousarray(
        overlap_denominator,
        dtype=np.float32,
    ).reshape(-1)
    return _gather_normalized_occurrence_weights_numba(
        denominator_flat,
        weights_arr,
        selected_arr,
        prepared.counts_arr,
        prepared.vol_strides,
        prepared.flat_offsets,
        prepared.w_patch_flat,
    )


def scatter_add_patch_numerator_prepared(
    accum_num: NDArray[np.float32],
    group_filt: NDArray[np.float32],
    weights: NDArray[np.float32],
    sel_global: NDArray[np.int64],
    prepared: PatchAggregationPlan,
) -> None:
    if group_filt.shape[0] == 0:
        return
    group_filt_flat, weights_arr, selected_arr = _prepare_scatter_inputs(
        group_filt,
        weights,
        sel_global,
        prepared,
    )
    _scatter_add_patch_numerator_numba(
        accum_num.ravel(),
        group_filt_flat,
        weights_arr,
        selected_arr,
        prepared.counts_arr,
        prepared.vol_strides,
        prepared.flat_offsets,
        prepared.w_patch_flat,
    )


def scatter_add_patches_prepared(
    accum_num: NDArray[np.float32],
    accum_den: NDArray[np.float32],
    group_filt: NDArray[np.float32],
    weights: NDArray[np.float32],
    sel_global: NDArray[np.int64],
    prepared: PatchAggregationPlan,
) -> None:
    ndim = int(prepared.counts_arr.size)
    G = group_filt.shape[0]
    if G == 0:
        return

    group_filt_flat = _prepare_group_filt_flat(group_filt, ndim, prepared.patch_vol)
    _scatter_add_patches_numba(
        accum_num.ravel(),
        accum_den.ravel(),
        group_filt_flat,
        np.ascontiguousarray(weights, dtype=np.float32),
        np.ascontiguousarray(sel_global, dtype=np.int64),
        prepared.counts_arr,
        prepared.vol_strides,
        prepared.flat_offsets,
        prepared.w_patch_flat,
    )


def scatter_add_patches_with_scalar_sigma_prepared(
    accum_num: NDArray[np.float32],
    accum_den: NDArray[np.float32],
    sigma_num: NDArray[np.float32],
    group_filt: NDArray[np.float32],
    sigma: float | np.float32,
    weights: NDArray[np.float32],
    sel_global: NDArray[np.int64],
    prepared: PatchAggregationPlan,
) -> None:
    if group_filt.shape[0] == 0:
        return
    group_filt_flat, weights_arr, selected_arr = _prepare_scatter_inputs(
        group_filt,
        weights,
        sel_global,
        prepared,
    )
    _scatter_add_patches_with_scalar_sigma_numba(
        accum_num.ravel(),
        accum_den.ravel(),
        sigma_num.ravel(),
        group_filt_flat,
        np.float32(sigma),
        weights_arr,
        selected_arr,
        prepared.counts_arr,
        prepared.vol_strides,
        prepared.flat_offsets,
        prepared.w_patch_flat,
    )


def scatter_add_patches_with_sigma_prepared(
    accum_num: NDArray[np.float32],
    accum_den: NDArray[np.float32],
    sigma_num: NDArray[np.float32],
    group_filt: NDArray[np.float32],
    group_sigma: NDArray[np.float32],
    weights: NDArray[np.float32],
    sel_global: NDArray[np.int64],
    prepared: PatchAggregationPlan,
) -> None:
    if group_filt.shape[0] == 0:
        return
    group_filt_flat, weights_arr, selected_arr = _prepare_scatter_inputs(
        group_filt,
        weights,
        sel_global,
        prepared,
    )
    group_sigma_flat = _prepare_group_filt_flat(
        group_sigma,
        int(prepared.counts_arr.size),
        prepared.patch_vol,
    )
    if group_sigma_flat.shape != group_filt_flat.shape:
        raise ValueError("group_sigma shape must match group_filt")
    _scatter_add_patches_with_sigma_numba(
        accum_num.ravel(),
        accum_den.ravel(),
        sigma_num.ravel(),
        group_filt_flat,
        group_sigma_flat,
        weights_arr,
        selected_arr,
        prepared.counts_arr,
        prepared.vol_strides,
        prepared.flat_offsets,
        prepared.w_patch_flat,
    )


def scatter_add_patches(
    accum_num: NDArray[np.float32],
    accum_den: NDArray[np.float32],
    group_filt: NDArray[np.float32],  # (G, p1, p2, ..., pn)
    weights: NDArray[np.float32],  # (G,)
    sel_global: NDArray[np.int64],  # (G,)
    counts: tuple[int, ...],  # patch grid counts
    block_size: tuple[int, ...],  # patch size
    w_patch: NDArray[np.float32],  # shape block_size
) -> None:
    prepared = prepare_patch_aggregation(
        accum_num.shape,
        counts,
        block_size,
        w_patch,
    )
    scatter_add_patches_prepared(
        accum_num=accum_num,
        accum_den=accum_den,
        group_filt=group_filt,
        weights=weights,
        sel_global=sel_global,
        prepared=prepared,
    )
