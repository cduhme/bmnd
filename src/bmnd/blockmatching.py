import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from .enums import BlockMatchDistance, NoiseModel, coerce_enum
from .shifts import ReferenceScheduleItem

_DISTANCE_SSD = 0
_DISTANCE_POISSON_DEVIANCE = 1
_DISTANCE_PEARSON = 2
_DISTANCE_ANSCOMBE_SSD = 3
_POISSON_DISTANCE_ROUNDOFF_FACTOR = 32.0
_FLOAT32_EPS = float(np.finfo(np.float32).eps)


_DISTANCE_CODES = {
    BlockMatchDistance.SSD: _DISTANCE_SSD,
    BlockMatchDistance.POISSON_DEVIANCE: _DISTANCE_POISSON_DEVIANCE,
    BlockMatchDistance.PEARSON: _DISTANCE_PEARSON,
    BlockMatchDistance.ANSCOMBE_SSD: _DISTANCE_ANSCOMBE_SSD,
}


@cache
def poisson_null_moment_table(
    distance_measure: BlockMatchDistance,
    max_count: int = 512,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    distance_measure = coerce_enum(
        distance_measure,
        BlockMatchDistance,
        "block-matching distance",
    )
    if distance_measure not in {
        BlockMatchDistance.POISSON_DEVIANCE,
        BlockMatchDistance.PEARSON,
        BlockMatchDistance.ANSCOMBE_SSD,
    }:
        raise ValueError(
            "Poisson null moments require 'poisson_deviance', 'pearson', "
            f"or 'anscombe_ssd', got {distance_measure!r}."
        )
    if max_count < 1:
        raise ValueError(f"max_count must be positive, got {max_count}.")

    means = np.zeros(max_count + 1, dtype=np.float64)
    variances = np.zeros(max_count + 1, dtype=np.float64)
    if distance_measure is BlockMatchDistance.PEARSON:
        counts = np.arange(1, max_count + 1, dtype=np.float64)
        means[1:] = 1.0
        variances[1:] = 2.0 - 2.0 / counts
        return means, variances

    for pooled_count in range(1, max_count + 1):
        expectation = 0.0
        second_moment = 0.0
        probability_scale = 2.0 ** (-pooled_count)
        for reference_count in range(pooled_count + 1):
            candidate_count = pooled_count - reference_count
            probability = math.comb(pooled_count, reference_count) * probability_scale
            if distance_measure is BlockMatchDistance.POISSON_DEVIANCE:
                contribution = 0.0
                if reference_count > 0:
                    contribution += reference_count * math.log(
                        2.0 * reference_count / pooled_count
                    )
                if candidate_count > 0:
                    contribution += candidate_count * math.log(
                        2.0 * candidate_count / pooled_count
                    )
                contribution *= 2.0
            else:
                reference_stabilized = 2.0 * math.sqrt(reference_count + 0.375)
                candidate_stabilized = 2.0 * math.sqrt(candidate_count + 0.375)
                contribution = 0.5 * (
                    candidate_stabilized - reference_stabilized
                ) ** 2
            expectation += probability * contribution
            second_moment += probability * contribution * contribution
        means[pooled_count] = expectation
        variances[pooled_count] = max(second_moment - expectation * expectation, 0.0)
    return means, variances


def compute_poisson_reference_thresholds(
    reference_schedule: Sequence[ReferenceScheduleItem],
    counts: tuple[int, ...],
    patches_flat: NDArray[np.float32],
    distance_measure: BlockMatchDistance,
    poisson_count_scale: float,
    ht_match_threshold: float,
    structure_beta: float,
    intensity_power: float,
    max_count: int = 512,
) -> NDArray[np.float32]:
    distance_measure = coerce_enum(
        distance_measure,
        BlockMatchDistance,
        "block-matching distance",
    )
    count_scale = _validate_poisson_count_scale(poisson_count_scale)
    if not np.isfinite(ht_match_threshold) or ht_match_threshold < 0.0:
        raise ValueError(
            "ht_match_threshold must be finite and non-negative, "
            f"got {ht_match_threshold}."
        )
    if not np.isfinite(structure_beta) or structure_beta < 0.0:
        raise ValueError(
            "structure_beta must be finite and non-negative, "
            f"got {structure_beta}."
        )
    if not np.isfinite(intensity_power) or intensity_power <= 0.0:
        raise ValueError(
            f"intensity_power must be finite and positive, got {intensity_power}."
        )

    means, variances = poisson_null_moment_table(distance_measure, max_count)
    patch_volume = patches_flat.shape[1]
    thresholds = np.empty(len(reference_schedule), dtype=np.float32)
    for index, schedule_item in enumerate(reference_schedule):
        reference_linear = int(np.ravel_multi_index(schedule_item.reference_abs, counts))
        reference_counts = np.clip(
            np.asarray(patches_flat[reference_linear], dtype=np.float64) / count_scale,
            0.0,
            None,
        )
        pooled_estimate = np.rint(2.0 * reference_counts)
        table_indices = np.minimum(pooled_estimate, max_count).astype(np.int64)
        null_means = means[table_indices]
        null_variances = variances[table_indices]
        above_table = pooled_estimate > max_count
        null_means[above_table] = 1.0
        null_variances[above_table] = 2.0
        patch_mean = float(np.mean(null_means))
        patch_variance = float(np.sum(null_variances)) / (patch_volume * patch_volume)
        reference_intensity = float(np.mean(reference_counts))
        threshold = (
            patch_mean
            + float(ht_match_threshold) * math.sqrt(max(patch_variance, 0.0))
            + float(structure_beta) * reference_intensity ** float(intensity_power)
        )
        thresholds[index] = np.float32(threshold)
    return thresholds


@dataclass
class BlockMatchGroup:
    reference_abs: tuple[int, ...]
    reference_shift: tuple[int, ...]
    reference_shifted: tuple[int, ...]
    selected_lin_indices: NDArray[np.int64]
    selected_abs_positions: NDArray[np.int64]
    selected_shifted_positions: NDArray[np.int64]
    accepted_count: int
    final_count: int


@dataclass
class BlockMatchResult:
    groups: list[BlockMatchGroup]


def compute_blockmatch_threshold(
    similarity_threshold: float,
    volume: NDArray[np.floating],
    distance_measure: BlockMatchDistance = BlockMatchDistance.SSD,
) -> float:
    distance_code = _get_distance_code(distance_measure)
    if distance_code != _DISTANCE_SSD:
        return float(similarity_threshold)

    arr = np.asarray(volume, dtype=np.float32)
    if arr.size == 0:
        return float(similarity_threshold)

    data_scale = float(np.max(arr) - np.min(arr))
    if data_scale <= 0.0:
        data_scale = 1.0
    return float(similarity_threshold) * data_scale * data_scale


def resolve_blockmatch_distance(
    distance_measure: BlockMatchDistance,
    noise_model: NoiseModel,
) -> BlockMatchDistance:
    distance_measure = coerce_enum(
        distance_measure,
        BlockMatchDistance,
        "block-matching distance",
    )
    noise_model = coerce_enum(noise_model, NoiseModel, "noise_model")
    resolved = (
        BlockMatchDistance.POISSON_DEVIANCE
        if distance_measure is BlockMatchDistance.AUTO
        and noise_model is NoiseModel.POISSON
        else BlockMatchDistance.SSD
        if distance_measure is BlockMatchDistance.AUTO
        else distance_measure
    )
    _get_distance_code(resolved)
    return resolved


def _get_distance_code(distance_measure: BlockMatchDistance) -> int:
    distance_measure = coerce_enum(
        distance_measure,
        BlockMatchDistance,
        "block-matching distance",
    )
    try:
        return _DISTANCE_CODES[distance_measure]
    except KeyError as exc:
        expected = "', '".join(distance.value for distance in _DISTANCE_CODES)
        raise ValueError(
            f"Unknown block-matching distance: {distance_measure!r} "
            f"(expected '{expected}')."
        ) from exc


def _validate_poisson_count_scale(poisson_count_scale: float) -> float:
    count_scale = float(poisson_count_scale)
    if not np.isfinite(count_scale) or count_scale <= 0.0:
        raise ValueError(
            "poisson_count_scale must be finite and positive, "
            f"got {poisson_count_scale}."
        )
    return count_scale


def _resolve_poisson_count_scale(
    distance_code: int,
    poisson_count_scale: float,
) -> float:
    if distance_code == _DISTANCE_SSD:
        return 1.0
    return _validate_poisson_count_scale(poisson_count_scale)


def _prepare_gamma_correction(
    gaussian_noise_ssd: NDArray[np.floating] | None,
    poisson_gamma: bool,
    gamma: float,
    search_radius: tuple[int, ...],
    distance_code: int,
) -> tuple[NDArray[np.float32], np.float32, bool, bool]:
    gamma_value = float(gamma)
    if not np.isfinite(gamma_value) or gamma_value < 0.0:
        raise ValueError(f"gamma must be finite and non-negative, got {gamma}.")

    apply_poisson_correction = gamma_value != 0.0 and poisson_gamma
    if apply_poisson_correction and distance_code != _DISTANCE_SSD:
        raise ValueError("Poisson gamma correction is only supported for SSD block matching.")

    if gaussian_noise_ssd is None:
        if gamma_value != 0.0 and not apply_poisson_correction:
            raise ValueError(
                "A Gaussian or Poisson SSD noise-bias estimate is required when gamma is nonzero."
            )
        return (
            np.empty(0, dtype=np.float32),
            np.float32(gamma_value),
            False,
            apply_poisson_correction,
        )
    if distance_code != _DISTANCE_SSD and gamma_value != 0.0:
        raise ValueError("Gaussian gamma correction is only supported for SSD block matching.")
    expected_shape = tuple(2 * radius + 1 for radius in search_radius)
    noise_ssd = np.asarray(gaussian_noise_ssd, dtype=np.float32)
    if noise_ssd.shape != expected_shape:
        raise ValueError(
            f"gaussian_noise_ssd must have shape {expected_shape}, "
            f"got {noise_ssd.shape}."
        )
    if not np.all(np.isfinite(noise_ssd)) or np.any(noise_ssd < 0.0):
        raise ValueError("gaussian_noise_ssd values must be finite and non-negative.")
    return (
        np.ascontiguousarray(noise_ssd.reshape(-1)),
        np.float32(gamma_value),
        gamma_value != 0.0,
        False,
    )


def _prepare_poisson_standardization(
    distance_measure: BlockMatchDistance,
    standardize: bool,
    max_count: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if not standardize:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    distance_measure = coerce_enum(
        distance_measure,
        BlockMatchDistance,
        "block-matching distance",
    )
    if distance_measure not in {
        BlockMatchDistance.POISSON_DEVIANCE,
        BlockMatchDistance.PEARSON,
        BlockMatchDistance.ANSCOMBE_SSD,
    }:
        raise ValueError(
            "Candidate standardization requires 'poisson_deviance', 'pearson', "
            f"or 'anscombe_ssd', got {distance_measure!r}."
        )
    if isinstance(max_count, bool) or not isinstance(max_count, (int, np.integer)):
        raise ValueError(
            f"poisson_moment_max_count must be a positive integer, got {max_count!r}."
        )
    if int(max_count) <= 0:
        raise ValueError(
            f"poisson_moment_max_count must be a positive integer, got {max_count!r}."
        )
    means, variances = poisson_null_moment_table(distance_measure, int(max_count))
    return means.astype(np.float32), variances.astype(np.float32)


@njit(cache=True)
def _patch_distance_numba(
    reference_patch: NDArray[np.float32],
    candidate_patch: NDArray[np.float32],
    distance_code: int,
    poisson_count_scale: float,
) -> np.float32:
    patch_vol = reference_patch.shape[0]
    distance = np.float32(0.0)

    if distance_code == _DISTANCE_SSD:
        for patch_idx in range(patch_vol):
            difference = candidate_patch[patch_idx] - reference_patch[patch_idx]
            distance += difference * difference
        return distance

    inverse_count_scale = 1.0 / poisson_count_scale
    for patch_idx in range(patch_vol):
        reference = reference_patch[patch_idx] * inverse_count_scale
        if reference < 0.0:
            reference = np.float32(0.0)
        candidate = candidate_patch[patch_idx] * inverse_count_scale
        if candidate < 0.0:
            candidate = np.float32(0.0)

        if distance_code == _DISTANCE_POISSON_DEVIANCE:
            combined = reference + candidate
            if combined > 0.0:
                contribution = np.float32(0.0)
                if reference > 0.0:
                    contribution += reference * np.log(2.0 * reference / combined)
                if candidate > 0.0:
                    contribution += candidate * np.log(2.0 * candidate / combined)
                distance += 2.0 * contribution
        elif distance_code == _DISTANCE_PEARSON:
            combined = reference + candidate
            if combined > 0.0:
                difference = candidate - reference
                distance += difference * difference / combined
        else:
            reference_stabilized = 2.0 * np.sqrt(reference + 0.375)
            candidate_stabilized = 2.0 * np.sqrt(candidate + 0.375)
            difference = candidate_stabilized - reference_stabilized
            # Each stabilized observation has variance about one, so their
            # difference has variance about two under the equal-rate model.
            distance += 0.5 * difference * difference

    if distance < 0.0:
        distance = np.float32(0.0)
    return np.float32(distance / patch_vol)


@njit(cache=True)
def _standardized_poisson_distance_numba(
    reference_patch: NDArray[np.float32],
    candidate_patch: NDArray[np.float32],
    distance_code: int,
    poisson_count_scale: float,
    null_means: NDArray[np.float32],
    null_variances: NDArray[np.float32],
) -> np.float32:
    inverse_count_scale = 1.0 / poisson_count_scale
    max_count = null_means.shape[0] - 1
    distance_sum = np.float32(0.0)
    null_mean_sum = np.float32(0.0)
    null_variance_sum = np.float32(0.0)

    for patch_idx in range(reference_patch.shape[0]):
        reference = reference_patch[patch_idx] * inverse_count_scale
        if reference < 0.0:
            reference = np.float32(0.0)
        candidate = candidate_patch[patch_idx] * inverse_count_scale
        if candidate < 0.0:
            candidate = np.float32(0.0)
        combined = reference + candidate
        if combined <= 0.0:
            continue

        if distance_code == _DISTANCE_POISSON_DEVIANCE:
            contribution = np.float32(0.0)
            if reference > 0.0:
                contribution += reference * np.log(2.0 * reference / combined)
            if candidate > 0.0:
                contribution += candidate * np.log(2.0 * candidate / combined)
            distance_sum += 2.0 * contribution
        elif distance_code == _DISTANCE_PEARSON:
            difference = candidate - reference
            distance_sum += difference * difference / combined
        else:
            reference_stabilized = 2.0 * np.sqrt(reference + 0.375)
            candidate_stabilized = 2.0 * np.sqrt(candidate + 0.375)
            difference = candidate_stabilized - reference_stabilized
            distance_sum += 0.5 * difference * difference

        pooled_count = int(np.rint(combined))
        if pooled_count <= max_count:
            null_mean_sum += null_means[pooled_count]
            null_variance_sum += null_variances[pooled_count]
        else:
            null_mean_sum += np.float32(1.0)
            null_variance_sum += np.float32(2.0)

    if null_variance_sum <= 0.0:
        return np.float32(0.0)
    return np.float32(
        (distance_sum - null_mean_sum) / np.sqrt(null_variance_sum)
    )


@njit(cache=True)
def _poisson_ssd_noise_bias_numba(
    reference_patch: NDArray[np.float32],
    candidate_patch: NDArray[np.float32],
    poisson_count_scale: float,
) -> np.float32:
    patch_vol = reference_patch.shape[0]
    bias = np.float32(0.0)
    for patch_idx in range(patch_vol):
        pooled = reference_patch[patch_idx] + candidate_patch[patch_idx]
        if pooled > 0.0:
            bias += poisson_count_scale * pooled
    return bias


def linear_index_to_multi(
    lin_index: int,
    counts: tuple[int, ...],
) -> tuple[int, ...]:
    coords = [0] * len(counts)
    tmp = int(lin_index)
    for dim in range(len(counts) - 1, -1, -1):
        coords[dim] = tmp % counts[dim]
        tmp //= counts[dim]
    return tuple(coords)


def multi_index_to_linear(
    multi_index: tuple[int, ...],
    counts: tuple[int, ...],
) -> int:
    lin = int(multi_index[0])
    for dim in range(1, len(counts)):
        lin = lin * counts[dim] + int(multi_index[dim])
    return lin


def _largest_power_of_two_at_most(value: int) -> int:
    if value <= 0:
        return 0
    return 1 << (value.bit_length() - 1)


@njit(cache=True)
def _largest_power_of_two_at_most_numba(value: int) -> int:
    if value <= 0:
        return 0
    power = 1
    while (power << 1) <= value:
        power <<= 1
    return power


@njit(cache=True)
def neighborhood_lin_indices_numba(
    center: NDArray[np.int64],
    counts: NDArray[np.int64],
    radius: NDArray[np.int64],
) -> NDArray[np.int64]:
    ndim = counts.shape[0]

    lo = np.empty(ndim, dtype=np.int64)
    hi = np.empty(ndim, dtype=np.int64)
    extents = np.empty(ndim, dtype=np.int64)

    for d in range(ndim):
        c = counts[d]
        cd = center[d]
        rd = radius[d]
        lo_d = cd - rd
        if lo_d < 0:
            lo_d = 0
        hi_d = cd + rd
        if hi_d > c - 1:
            hi_d = c - 1
        lo[d] = lo_d
        hi[d] = hi_d
        extents[d] = hi_d - lo_d + 1

    size = 1
    for d in range(ndim):
        size *= extents[d]

    out = np.empty(size, dtype=np.int64)

    # Enumerate all combinations via linear index over extents-grid
    for idx_lin in range(size):
        tmp = idx_lin
        coord = np.empty(ndim, dtype=np.int64)
        for d in range(ndim - 1, -1, -1):
            ed = extents[d]
            val = tmp % ed
            tmp //= ed
            coord[d] = lo[d] + val

        # Convert coord (multi) -> linear in counts-grid
        lin = coord[0]
        for d in range(1, ndim):
            lin = lin * counts[d] + coord[d]
        out[idx_lin] = lin

    return out


@njit(cache=True, parallel=True)
def blockmatch_distance_batch_numba(
    ref_multis_arr: NDArray[np.int64],
    counts_arr: NDArray[np.int64],
    patches_flat: NDArray[np.float32],
    search_radius_arr: NDArray[np.int64],
    max_neighborhood_size: int,
    distance_code: int = _DISTANCE_SSD,
    poisson_count_scale: float = 1.0,
) -> tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.int64]]:
    n_refs, ndim = ref_multis_arr.shape

    neighborhoods = -np.ones((n_refs, max_neighborhood_size), dtype=np.int64)
    distances = np.full(
        (n_refs, max_neighborhood_size),
        np.float32(np.inf),
        dtype=np.float32,
    )
    neighborhood_sizes = np.zeros(n_refs, dtype=np.int64)

    for ri in prange(n_refs):
        center = ref_multis_arr[ri]

        center_lin = center[0]
        for d in range(1, ndim):
            center_lin = center_lin * counts_arr[d] + center[d]

        ref_patch = patches_flat[center_lin]

        lo = np.empty(ndim, dtype=np.int64)
        extents = np.empty(ndim, dtype=np.int64)
        neighborhood_size = 1
        for d in range(ndim):
            count_d = counts_arr[d]
            center_d = center[d]
            radius_d = search_radius_arr[d]

            lo_d = center_d - radius_d
            if lo_d < 0:
                lo_d = 0
            hi_d = center_d + radius_d
            if hi_d > count_d - 1:
                hi_d = count_d - 1

            lo[d] = lo_d
            extent_d = hi_d - lo_d + 1
            extents[d] = extent_d
            neighborhood_size *= extent_d

        neighborhood_sizes[ri] = neighborhood_size

        for idx_lin in range(neighborhood_size):
            tmp = idx_lin
            coord = np.empty(ndim, dtype=np.int64)
            for d in range(ndim - 1, -1, -1):
                extent_d = extents[d]
                coord_d = lo[d] + (tmp % extent_d)
                tmp //= extent_d
                coord[d] = coord_d

            lin = coord[0]
            for d in range(1, ndim):
                lin = lin * counts_arr[d] + coord[d]

            neighborhoods[ri, idx_lin] = lin
            cand_patch = patches_flat[lin]
            distances[ri, idx_lin] = _patch_distance_numba(
                ref_patch,
                cand_patch,
                distance_code,
                poisson_count_scale,
            )

    return neighborhoods, distances, neighborhood_sizes


@njit(cache=True, parallel=True)
def blockmatch_full_numba(
    ref_multis_arr: NDArray[np.int64],
    counts_arr: NDArray[np.int64],
    patches_flat: NDArray[np.float32],
    search_radius_arr: NDArray[np.int64],
    max_matches: int,
    distance_threshold: float,
    distance_code: int = _DISTANCE_SSD,
    poisson_count_scale: float = 1.0,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    n_refs, ndim = ref_multis_arr.shape
    n_patches = patches_flat.shape[0]

    # Ensure contiguous for best performance
    patches_flat = np.ascontiguousarray(patches_flat)

    # -------- Precompute patch norms: ||p||^2 for all patches ----------
    patch_norms = np.empty(n_patches, dtype=np.float32)
    for i in prange(n_patches):
        # dot(p, p) is SIMD-friendly
        p = patches_flat[i]
        patch_norms[i] = np.dot(p, p)

    matches_lin = -np.ones((n_refs, max_matches), dtype=np.int64)
    match_counts = np.zeros(n_refs, dtype=np.int64)

    # ----------------- Parallel loop over reference patches -----------------
    for ri in prange(n_refs):
        center = ref_multis_arr[ri]

        # linear index of reference in patch-grid
        center_lin = center[0]
        for d in range(1, ndim):
            center_lin = center_lin * counts_arr[d] + center[d]

        ref_patch = patches_flat[center_lin]
        ref_norm = patch_norms[center_lin]

        # local neighborhood (indices in patch-grid)
        neigh = neighborhood_lin_indices_numba(center, counts_arr, search_radius_arr)
        n_neigh = neigh.shape[0]
        if n_neigh == 0:
            continue

        # local candidate arrays (per-reference, so thread-local)
        cand_lin = np.empty(n_neigh, dtype=np.int64)
        cand_dist = np.empty(n_neigh, dtype=np.float32)
        kept = 0

        for j in range(n_neigh):
            ci = neigh[j]
            cand_patch = patches_flat[ci]

            if distance_code == _DISTANCE_SSD:
                # Fast SSD via ||p-q||^2 = ||p||^2 + ||q||^2 - 2<p,q>.
                dot_pq = np.dot(ref_patch, cand_patch)
                dist2 = ref_norm + patch_norms[ci] - 2.0 * dot_pq
                if dist2 < 0.0:
                    dist2 = 0.0
            else:
                dist2 = _patch_distance_numba(
                    ref_patch,
                    cand_patch,
                    distance_code,
                    poisson_count_scale,
                )

            # prune by similarity threshold
            if dist2 < distance_threshold:
                cand_lin[kept] = ci
                cand_dist[kept] = dist2
                kept += 1

        if kept == 0:
            continue

        # keep up to max_matches best candidates
        k = max_matches if kept > max_matches else kept
        order = np.argsort(cand_dist[:kept])

        for kk in range(k):
            matches_lin[ri, kk] = cand_lin[order[kk]]

        match_counts[ri] = k

    return matches_lin, match_counts


def _batched_distance_blockmatch(
    ref_multis: Sequence[tuple[int, ...]],
    counts: tuple[int, ...],
    patches_flat: NDArray[np.float32],
    search_radius: tuple[int, ...],
    max_matches: int,
    distance_threshold: float,
    batch_size: int | None = None,
    distance_measure: BlockMatchDistance = BlockMatchDistance.SSD,
    poisson_count_scale: float = 1.0,
) -> list[NDArray[np.int64]]:
    reference_schedule = [
        ReferenceScheduleItem(
            shift=tuple(0 for _ in counts),
            reference_abs=tuple(int(x) for x in ref_multi),
            reference_shifted=tuple(int(x) for x in ref_multi),
        )
        for ref_multi in ref_multis
    ]
    result = blockmatch_groups(
        reference_schedule=reference_schedule,
        counts=counts,
        patches_flat=patches_flat,
        search_radius=search_radius,
        max_matches=max_matches,
        min_matches=1,
        distance_threshold=distance_threshold,
        batch_size=batch_size,
        prefer_power_of_two_stack=False,
        distance_measure=distance_measure,
        poisson_count_scale=poisson_count_scale,
    )
    return [group.selected_lin_indices.copy() for group in result.groups]


def _empty_blockmatch_group(
    schedule_item: ReferenceScheduleItem,
    ndim: int,
) -> BlockMatchGroup:
    empty_positions = np.empty((0, ndim), dtype=np.int64)
    return BlockMatchGroup(
        reference_abs=schedule_item.reference_abs,
        reference_shift=schedule_item.shift,
        reference_shifted=schedule_item.reference_shifted,
        selected_lin_indices=np.empty(0, dtype=np.int64),
        selected_abs_positions=empty_positions.copy(),
        selected_shifted_positions=empty_positions,
        accepted_count=0,
        final_count=0,
    )


@njit(cache=True)
def _finalize_selected_positions_numba(
    selected_positions: NDArray[np.int64],
    selected_count: int,
    neighborhood: NDArray[np.int64],
    center_lin: int,
    max_matches: int,
    min_matches: int,
    prefer_power_of_two_stack: bool,
) -> NDArray[np.int64]:
    if max_matches <= 0:
        return np.empty(0, dtype=np.int64)

    reference_pos = -1
    for idx in range(neighborhood.shape[0]):
        if neighborhood[idx] == center_lin:
            reference_pos = idx
            break

    deduped_positions = np.full(max_matches, -1, dtype=np.int64)
    deduped_count = 0
    if reference_pos >= 0 and max_matches > 0:
        deduped_positions[0] = reference_pos
        deduped_count = 1
    seen = np.zeros(neighborhood.shape[0], dtype=np.uint8)
    if reference_pos >= 0:
        seen[reference_pos] = 1
    for idx in range(selected_count):
        pos = selected_positions[idx]
        if pos >= 0 and seen[pos] == 0:
            deduped_positions[deduped_count] = pos
            deduped_count += 1
            seen[pos] = 1
            if deduped_count >= max_matches:
                break

    if prefer_power_of_two_stack and deduped_count > 0:
        target_size = _largest_power_of_two_at_most_numba(deduped_count)
        if min_matches > target_size:
            target_size = min_matches
        if target_size < deduped_count:
            deduped_count = target_size

    return deduped_positions[:deduped_count]


@njit(cache=True)
def _candidate_precedes_numba(
    candidate_pos: int,
    existing_pos: int,
    neighborhood: NDArray[np.int64],
    dist2: NDArray[np.float32],
) -> bool:
    candidate_dist = dist2[candidate_pos]
    existing_dist = dist2[existing_pos]
    if candidate_dist < existing_dist:
        return True
    if candidate_dist > existing_dist:
        return False
    return neighborhood[candidate_pos] < neighborhood[existing_pos]


@njit(cache=True)
def _insert_position_sorted_limited_numba(
    buffer: NDArray[np.int64],
    current_size: int,
    candidate_pos: int,
    neighborhood: NDArray[np.int64],
    dist2: NDArray[np.float32],
) -> int:
    limit = buffer.shape[0]
    if limit == 0:
        return current_size

    if current_size < limit:
        current_size += 1
        insert_at = current_size - 1
    else:
        last_pos = buffer[limit - 1]
        if not _candidate_precedes_numba(candidate_pos, last_pos, neighborhood, dist2):
            return current_size
        insert_at = limit - 1

    while insert_at > 0 and _candidate_precedes_numba(
        candidate_pos,
        buffer[insert_at - 1],
        neighborhood,
        dist2,
    ):
        buffer[insert_at] = buffer[insert_at - 1]
        insert_at -= 1

    buffer[insert_at] = candidate_pos
    return current_size


@njit(cache=True)
def _select_blockmatch_positions_from_distances_numba(
    neighborhood: NDArray[np.int64],
    dist2: NDArray[np.float32],
    center_lin: int,
    max_matches: int,
    min_matches: int,
    prefer_power_of_two_stack: bool,
    distance_threshold: float,
) -> tuple[NDArray[np.int64], int]:
    if neighborhood.shape[0] == 0:
        return np.empty(0, dtype=np.int64), 0

    required_size = min(min_matches, max_matches)
    accepted_positions = np.full(max_matches, -1, dtype=np.int64)
    rejected_positions = np.full(required_size, -1, dtype=np.int64)
    accepted_size = 0
    rejected_size = 0
    accepted_count = 0

    for pos in range(neighborhood.shape[0]):
        if dist2[pos] < distance_threshold:
            accepted_count += 1
            accepted_size = _insert_position_sorted_limited_numba(
                accepted_positions,
                accepted_size,
                pos,
                neighborhood,
                dist2,
            )
        else:
            rejected_size = _insert_position_sorted_limited_numba(
                rejected_positions,
                rejected_size,
                pos,
                neighborhood,
                dist2,
            )

    selected_positions = np.full(max_matches, -1, dtype=np.int64)
    selected_count = accepted_size
    for idx in range(accepted_size):
        selected_positions[idx] = accepted_positions[idx]

    if selected_count < required_size:
        needed = required_size - selected_count
        take = needed if needed < rejected_size else rejected_size
        for idx in range(take):
            selected_positions[selected_count] = rejected_positions[idx]
            selected_count += 1

    return (
        _finalize_selected_positions_numba(
            selected_positions,
            selected_count,
            neighborhood,
            center_lin,
            max_matches,
            min_matches,
            prefer_power_of_two_stack,
        ),
        accepted_count,
    )


@njit(cache=True)
def _finalize_selected_lin_indices_numba(
    selected_lin_indices: NDArray[np.int64],
    selected_count: int,
    center_lin: int,
    max_matches: int,
    min_matches: int,
    prefer_power_of_two_stack: bool,
) -> NDArray[np.int64]:
    if max_matches <= 0:
        return np.empty(0, dtype=np.int64)

    deduped_lin_indices = np.full(max_matches, -1, dtype=np.int64)
    deduped_count = 0
    if max_matches > 0:
        deduped_lin_indices[0] = center_lin
        deduped_count = 1
    for idx in range(selected_count):
        lin = selected_lin_indices[idx]
        if lin < 0 or lin == center_lin:
            continue

        already_seen = False
        for prev_idx in range(deduped_count):
            if deduped_lin_indices[prev_idx] == lin:
                already_seen = True
                break

        if not already_seen:
            deduped_lin_indices[deduped_count] = lin
            deduped_count += 1
            if deduped_count >= max_matches:
                break

    if prefer_power_of_two_stack and deduped_count > 0:
        target_size = _largest_power_of_two_at_most_numba(deduped_count)
        if min_matches > target_size:
            target_size = min_matches
        if target_size < deduped_count:
            deduped_count = target_size

    return deduped_lin_indices[:deduped_count]


@njit(cache=True)
def _canonicalize_poisson_distance_numba(distance: np.float32) -> np.float32:
    if not math.isfinite(float(distance)):
        return distance
    magnitude = max(abs(float(distance)), 1.0)
    exponent = math.floor(math.log2(magnitude))
    quantum = _POISSON_DISTANCE_ROUNDOFF_FACTOR * _FLOAT32_EPS * 2.0**exponent
    return np.float32(round(float(distance) / quantum) * quantum)


@njit(cache=True)
def _candidate_lin_precedes_numba(
    candidate_lin: int,
    candidate_dist2: np.float32,
    existing_lin: int,
    existing_dist2: np.float32,
    stabilize_distance_ties: bool,
) -> bool:
    if stabilize_distance_ties:
        candidate_dist2 = _canonicalize_poisson_distance_numba(candidate_dist2)
        existing_dist2 = _canonicalize_poisson_distance_numba(existing_dist2)
        if candidate_dist2 == existing_dist2:
            return candidate_lin < existing_lin
    if candidate_dist2 < existing_dist2:
        return True
    if candidate_dist2 > existing_dist2:
        return False
    return candidate_lin < existing_lin


@njit(cache=True)
def _insert_lin_sorted_limited_numba(
    lin_buffer: NDArray[np.int64],
    dist_buffer: NDArray[np.float32],
    current_size: int,
    candidate_lin: int,
    candidate_dist2: np.float32,
    stabilize_distance_ties: bool,
) -> int:
    limit = lin_buffer.shape[0]
    if limit == 0:
        return current_size

    if current_size < limit:
        current_size += 1
        insert_at = current_size - 1
    else:
        last_lin = lin_buffer[limit - 1]
        last_dist2 = dist_buffer[limit - 1]
        if not _candidate_lin_precedes_numba(
            candidate_lin,
            candidate_dist2,
            last_lin,
            last_dist2,
            stabilize_distance_ties,
        ):
            return current_size
        insert_at = limit - 1

    while insert_at > 0 and _candidate_lin_precedes_numba(
        candidate_lin,
        candidate_dist2,
        lin_buffer[insert_at - 1],
        dist_buffer[insert_at - 1],
        stabilize_distance_ties,
    ):
        lin_buffer[insert_at] = lin_buffer[insert_at - 1]
        dist_buffer[insert_at] = dist_buffer[insert_at - 1]
        insert_at -= 1

    lin_buffer[insert_at] = candidate_lin
    dist_buffer[insert_at] = candidate_dist2
    return current_size


@njit(cache=True)
def _distance_is_accepted_numba(
    candidate_dist2: np.float32,
    distance_threshold: np.float32,
    stabilize_distance_ties: bool,
) -> bool:
    if not stabilize_distance_ties:
        return candidate_dist2 < distance_threshold
    candidate_key = _canonicalize_poisson_distance_numba(candidate_dist2)
    threshold_key = _canonicalize_poisson_distance_numba(distance_threshold)
    return candidate_key < threshold_key


@njit(cache=True, parallel=True)
def _fused_blockmatch_groups_batch_numba(
    ref_multis_arr: NDArray[np.int64],
    counts_arr: NDArray[np.int64],
    patches_flat: NDArray[np.float32],
    search_radius_arr: NDArray[np.int64],
    max_matches: int,
    min_matches: int,
    prefer_power_of_two_stack: bool,
    distance_thresholds: NDArray[np.float32],
    distance_code: int,
    poisson_count_scale: float,
    poisson_null_means: NDArray[np.float32],
    poisson_null_variances: NDArray[np.float32],
    standardize_poisson_distance: bool,
    gaussian_noise_ssd_flat: NDArray[np.float32],
    gamma: np.float32,
    apply_gaussian_gamma: bool,
    apply_poisson_gamma: bool,
    stabilize_poisson_ties: bool,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    n_refs, ndim = ref_multis_arr.shape
    required_size = min(min_matches, max_matches)
    selected_lin_indices = np.full((n_refs, max_matches), -1, dtype=np.int64)
    selected_counts = np.zeros(n_refs, dtype=np.int64)
    accepted_counts = np.zeros(n_refs, dtype=np.int64)

    for ri in prange(n_refs):
        center = ref_multis_arr[ri]
        distance_threshold = distance_thresholds[ri]

        center_lin = center[0]
        for d in range(1, ndim):
            center_lin = center_lin * counts_arr[d] + center[d]

        ref_patch = patches_flat[center_lin]

        lo = np.empty(ndim, dtype=np.int64)
        extents = np.empty(ndim, dtype=np.int64)
        coord = np.empty(ndim, dtype=np.int64)
        for d in range(ndim):
            count_d = counts_arr[d]
            center_d = center[d]
            radius_d = search_radius_arr[d]

            lo_d = center_d - radius_d
            if lo_d < 0:
                lo_d = 0
            hi_d = center_d + radius_d
            if hi_d > count_d - 1:
                hi_d = count_d - 1

            lo[d] = lo_d
            extents[d] = hi_d - lo_d + 1
            coord[d] = lo_d

        accepted_lins = np.full(max_matches, -1, dtype=np.int64)
        accepted_dist2 = np.full(max_matches, np.float32(np.inf), dtype=np.float32)
        rejected_lins = np.full(required_size, -1, dtype=np.int64)
        rejected_dist2 = np.full(required_size, np.float32(np.inf), dtype=np.float32)
        accepted_size = 0
        rejected_size = 0
        accepted_count = 0
        stabilize_distance_ties = stabilize_poisson_ties and distance_code != _DISTANCE_SSD

        done = False
        while not done:
            lin = coord[0]
            for d in range(1, ndim):
                lin = lin * counts_arr[d] + coord[d]

            cand_patch = patches_flat[lin]
            if standardize_poisson_distance:
                candidate_dist2 = _standardized_poisson_distance_numba(
                    ref_patch,
                    cand_patch,
                    distance_code,
                    poisson_count_scale,
                    poisson_null_means,
                    poisson_null_variances,
                )
            else:
                candidate_dist2 = _patch_distance_numba(
                    ref_patch,
                    cand_patch,
                    distance_code,
                    poisson_count_scale,
                )
            if apply_gaussian_gamma:
                correction_index = coord[0] - center[0] + search_radius_arr[0]
                for d in range(1, ndim):
                    correction_index = correction_index * (
                        2 * search_radius_arr[d] + 1
                    ) + (coord[d] - center[d] + search_radius_arr[d])
                candidate_dist2 -= gamma * gaussian_noise_ssd_flat[correction_index]
            elif apply_poisson_gamma and lin != center_lin:
                candidate_dist2 -= gamma * _poisson_ssd_noise_bias_numba(
                    ref_patch,
                    cand_patch,
                    poisson_count_scale,
                )

            if _distance_is_accepted_numba(
                candidate_dist2,
                distance_threshold,
                stabilize_distance_ties,
            ):
                accepted_count += 1
                accepted_size = _insert_lin_sorted_limited_numba(
                    accepted_lins,
                    accepted_dist2,
                    accepted_size,
                    lin,
                    candidate_dist2,
                    stabilize_distance_ties,
                )
            else:
                rejected_size = _insert_lin_sorted_limited_numba(
                    rejected_lins,
                    rejected_dist2,
                    rejected_size,
                    lin,
                    candidate_dist2,
                    stabilize_distance_ties,
                )

            for d in range(ndim - 1, -1, -1):
                coord[d] += 1
                if coord[d] < lo[d] + extents[d]:
                    break
                coord[d] = lo[d]
                if d == 0:
                    done = True

        selected_lins = np.full(max_matches, -1, dtype=np.int64)
        selected_count = accepted_size
        for idx in range(accepted_size):
            selected_lins[idx] = accepted_lins[idx]

        if selected_count < required_size:
            needed = required_size - selected_count
            take = needed if needed < rejected_size else rejected_size
            for idx in range(take):
                selected_lins[selected_count] = rejected_lins[idx]
                selected_count += 1

        finalized_lin_indices = _finalize_selected_lin_indices_numba(
            selected_lins,
            selected_count,
            center_lin,
            max_matches,
            min_matches,
            prefer_power_of_two_stack,
        )
        final_count = finalized_lin_indices.shape[0]
        selected_counts[ri] = final_count
        accepted_counts[ri] = accepted_count
        for idx in range(final_count):
            selected_lin_indices[ri, idx] = finalized_lin_indices[idx]

    return selected_lin_indices, selected_counts, accepted_counts



@njit(cache=True)
def _linear_indices_to_multi_numba(
    selected_lin_indices: NDArray[np.int64],
    counts_arr: NDArray[np.int64],
) -> NDArray[np.int64]:
    n_selected = selected_lin_indices.shape[0]
    ndim = counts_arr.shape[0]
    coords = np.empty((n_selected, ndim), dtype=np.int64)

    for idx in range(n_selected):
        lin = selected_lin_indices[idx]
        for dim in range(ndim - 1, -1, -1):
            coords[idx, dim] = lin % counts_arr[dim]
            lin //= counts_arr[dim]

    return coords


def _build_blockmatch_group(
    schedule_item: ReferenceScheduleItem,
    counts_arr: NDArray[np.int64],
    selected_lin_indices: NDArray[np.int64],
    accepted_count: int,
    include_position_metadata: bool = True,
) -> BlockMatchGroup:
    selected_lin_indices = selected_lin_indices.astype(np.int64, copy=False)
    ndim = counts_arr.shape[0]
    if include_position_metadata:
        selected_abs_positions = _linear_indices_to_multi_numba(selected_lin_indices, counts_arr)

        if selected_abs_positions.size == 0:
            selected_shifted_positions = selected_abs_positions.copy()
            reference_shifted = schedule_item.reference_shifted
        else:
            group_origin = np.min(selected_abs_positions, axis=0, keepdims=True)
            selected_shifted_positions = selected_abs_positions - group_origin
            reference_shifted = tuple(
                int(x) for x in selected_shifted_positions[0].tolist()
            )
    else:
        selected_abs_positions = np.empty((0, ndim), dtype=np.int64)
        selected_shifted_positions = np.empty((0, ndim), dtype=np.int64)
        reference_shifted = schedule_item.reference_shifted

    return BlockMatchGroup(
        reference_abs=schedule_item.reference_abs,
        reference_shift=schedule_item.shift,
        reference_shifted=reference_shifted,
        selected_lin_indices=selected_lin_indices,
        selected_abs_positions=selected_abs_positions,
        selected_shifted_positions=selected_shifted_positions,
        accepted_count=accepted_count,
        final_count=int(selected_lin_indices.size),
    )


def _assemble_blockmatch_group(
    schedule_item: ReferenceScheduleItem,
    counts: tuple[int, ...],
    counts_arr: NDArray[np.int64],
    neighborhood: NDArray[np.int64],
    dist2: NDArray[np.float32],
    center_lin: int,
    max_matches: int,
    min_matches: int,
    prefer_power_of_two_stack: bool,
    distance_threshold: float,
    include_position_metadata: bool = True,
) -> BlockMatchGroup:
    if neighborhood.size == 0:
        return _empty_blockmatch_group(schedule_item, len(counts))

    selected_positions, accepted_count = _select_blockmatch_positions_from_distances_numba(
        neighborhood,
        dist2,
        center_lin,
        max_matches,
        min_matches,
        prefer_power_of_two_stack,
        distance_threshold,
    )
    selected_lin_indices = neighborhood[selected_positions].astype(np.int64, copy=False)

    return _build_blockmatch_group(
        schedule_item=schedule_item,
        counts_arr=counts_arr,
        selected_lin_indices=selected_lin_indices,
        accepted_count=accepted_count,
        include_position_metadata=include_position_metadata,
    )


def _get_max_neighborhood_size(
    counts: tuple[int, ...],
    search_radius: tuple[int, ...],
) -> int:
    size = 1
    for count, radius in zip(counts, search_radius, strict=False):
        size *= min(count, 2 * radius + 1)
    return int(size)


def blockmatch_group(
    schedule_item: ReferenceScheduleItem,
    counts: tuple[int, ...],
    patches_flat: NDArray[np.float32],
    search_radius: tuple[int, ...],
    max_matches: int,
    min_matches: int,
    distance_threshold: float,
    prefer_power_of_two_stack: bool = False,
    include_position_metadata: bool = True,
    distance_measure: BlockMatchDistance = BlockMatchDistance.SSD,
    poisson_count_scale: float = 1.0,
    gaussian_noise_ssd: NDArray[np.floating] | None = None,
    poisson_moment_max_count: int = 512,
    standardize_poisson_distance: bool = False,
    poisson_gamma: bool = False,
    gamma: float = 0.0,
    stabilize_poisson_ties: bool = False,
) -> BlockMatchGroup:
    counts_arr = np.asarray(counts, dtype=np.int64)
    search_arr = np.asarray(search_radius, dtype=np.int64)
    ref_multis_arr = np.asarray([schedule_item.reference_abs], dtype=np.int64)
    patches = np.ascontiguousarray(patches_flat.astype(np.float32, copy=False))
    distance_code = _get_distance_code(distance_measure)
    count_scale = (
        _validate_poisson_count_scale(poisson_count_scale)
        if poisson_gamma
        else _resolve_poisson_count_scale(distance_code, poisson_count_scale)
    )
    (
        noise_ssd_flat,
        gamma_value,
        apply_gaussian_gamma,
        apply_poisson_gamma,
    ) = _prepare_gamma_correction(
        gaussian_noise_ssd,
        poisson_gamma,
        gamma,
        search_radius,
        distance_code,
    )
    poisson_null_means, poisson_null_variances = _prepare_poisson_standardization(
        distance_measure,
        standardize_poisson_distance,
        poisson_moment_max_count,
    )
    distance_thresholds = np.asarray([distance_threshold], dtype=np.float32)

    selected_lin_indices_batch, selected_counts, accepted_counts = (
        _fused_blockmatch_groups_batch_numba(
            ref_multis_arr,
            counts_arr,
            patches,
            search_arr,
            max_matches,
            min_matches,
            prefer_power_of_two_stack,
            distance_thresholds,
            distance_code,
            count_scale,
            poisson_null_means,
            poisson_null_variances,
            standardize_poisson_distance,
            noise_ssd_flat,
            gamma_value,
            apply_gaussian_gamma,
            apply_poisson_gamma,
            stabilize_poisson_ties,
        )
    )
    selected_count = int(selected_counts[0])
    selected_lin_indices = selected_lin_indices_batch[0, :selected_count]

    return _build_blockmatch_group(
        schedule_item=schedule_item,
        counts_arr=counts_arr,
        selected_lin_indices=selected_lin_indices,
        accepted_count=int(accepted_counts[0]),
        include_position_metadata=include_position_metadata,
    )


def blockmatch_groups(
    reference_schedule: Sequence[ReferenceScheduleItem],
    counts: tuple[int, ...],
    patches_flat: NDArray[np.float32],
    search_radius: tuple[int, ...],
    max_matches: int,
    min_matches: int,
    distance_threshold: float | NDArray[np.floating],
    batch_size: int | None = None,
    prefer_power_of_two_stack: bool = True,
    include_position_metadata: bool = True,
    distance_measure: BlockMatchDistance | str = BlockMatchDistance.SSD,
    poisson_count_scale: float = 1.0,
    gaussian_noise_ssd: NDArray[np.floating] | None = None,
    poisson_moment_max_count: int = 512,
    standardize_poisson_distance: bool = False,
    poisson_gamma: bool = False,
    gamma: float = 0.0,
    stabilize_poisson_ties: bool = False,
) -> BlockMatchResult:
    distance_code = _get_distance_code(distance_measure)
    count_scale = (
        _validate_poisson_count_scale(poisson_count_scale)
        if poisson_gamma
        else _resolve_poisson_count_scale(distance_code, poisson_count_scale)
    )
    (
        noise_ssd_flat,
        gamma_value,
        apply_gaussian_gamma,
        apply_poisson_gamma,
    ) = _prepare_gamma_correction(
        gaussian_noise_ssd,
        poisson_gamma,
        gamma,
        search_radius,
        distance_code,
    )
    poisson_null_means, poisson_null_variances = _prepare_poisson_standardization(
        distance_measure,
        standardize_poisson_distance,
        poisson_moment_max_count,
    )

    if not reference_schedule:
        return BlockMatchResult(groups=[])

    schedule_list = list(reference_schedule)
    n_refs = len(schedule_list)
    if batch_size is None or batch_size <= 0 or batch_size > n_refs:
        batch_size = n_refs

    counts_arr = np.asarray(counts, dtype=np.int64)
    search_arr = np.asarray(search_radius, dtype=np.int64)
    patches = np.ascontiguousarray(patches_flat.astype(np.float32, copy=False))
    threshold_values = np.asarray(distance_threshold, dtype=np.float32)
    if threshold_values.ndim == 0:
        threshold_values = np.full(n_refs, float(threshold_values), dtype=np.float32)
    elif threshold_values.shape != (n_refs,):
        raise ValueError(
            "distance_threshold must be scalar or contain one value per reference "
            f"({n_refs}), got shape {threshold_values.shape}."
        )
    else:
        threshold_values = np.ascontiguousarray(threshold_values)
    if not np.all(np.isfinite(threshold_values)) or np.any(threshold_values < 0.0):
        raise ValueError("distance_threshold values must be finite and non-negative.")

    groups: list[BlockMatchGroup] = []
    for batch_start in range(0, n_refs, batch_size):
        batch_schedule = schedule_list[batch_start : batch_start + batch_size]
        batch_thresholds = threshold_values[
            batch_start : batch_start + len(batch_schedule)
        ]
        ref_multis_arr = np.asarray(
            [item.reference_abs for item in batch_schedule],
            dtype=np.int64,
        )
        (
            selected_lin_indices_batch,
            selected_counts,
            accepted_counts,
        ) = _fused_blockmatch_groups_batch_numba(
            ref_multis_arr,
            counts_arr,
            patches,
            search_arr,
            max_matches,
            min_matches,
            prefer_power_of_two_stack,
            batch_thresholds,
            distance_code,
            count_scale,
            poisson_null_means,
            poisson_null_variances,
            standardize_poisson_distance,
            noise_ssd_flat,
            gamma_value,
            apply_gaussian_gamma,
            apply_poisson_gamma,
            stabilize_poisson_ties,
        )

        for batch_index, schedule_item in enumerate(batch_schedule):
            selected_count = int(selected_counts[batch_index])
            selected_lin_indices = selected_lin_indices_batch[
                batch_index, :selected_count
            ]
            groups.append(
                _build_blockmatch_group(
                    schedule_item=schedule_item,
                    counts_arr=counts_arr,
                    selected_lin_indices=selected_lin_indices,
                    accepted_count=int(accepted_counts[batch_index]),
                    include_position_metadata=include_position_metadata,
                )
            )

    return BlockMatchResult(groups=groups)
