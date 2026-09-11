from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
from typing import Callable

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

from .cache import DEFAULT_POISSON_CACHE_LIMITS, PoissonCacheLimits
from .psd import (
    autocovariance_from_psd,
    normalize_sigma_psd,
    preprocess_psd,
    resolve_nf,
)
from .sigmaestimation import (
    estimate_gaussian_global_sigma,
    estimate_poisson_global_sigma,
)
from .transforms import _apply_matrix_along_axis
from .utils import extract_patches_strided


_EXACT_POISSON_PREPARED_CONTRIBUTION_MAX_BYTES = 512 * 1024 * 1024


@dataclass
class PoissonCovarianceCacheStats:
    overlap_hits: int = 0
    overlap_misses: int = 0
    overlap_evictions: int = 0
    overlap_bytes: int = 0
    contribution_hits: int = 0
    contribution_misses: int = 0
    contribution_evictions: int = 0
    contribution_bytes: int = 0
    factorized_hits: int = 0
    factorized_misses: int = 0
    factorized_evictions: int = 0
    factorized_bytes: int = 0


@dataclass
class VarianceModel:
    global_sigma: float
    sigma_psd: NDArray[np.float32]
    reduced_sigma_psd: NDArray[np.float32]
    processed_sigma_psd: NDArray[np.float32]
    blur_kernel: NDArray[np.float32]
    correlation_kernel: NDArray[np.float32]
    autocovariance: NDArray[np.float32]
    variance_domain_shape: tuple[int, ...]
    patch_covariance_cache: dict[tuple[int, ...], NDArray[np.float32]] = field(
        default_factory=dict
    )
    transformed_patch_variance_cache: dict[
        tuple[tuple[int, ...], tuple[object, ...]],
        NDArray[np.float32],
    ] = field(default_factory=dict)
    blockmatch_noise_ssd_cache: dict[
        tuple[tuple[int, ...], tuple[int, ...]], NDArray[np.float32]
    ] = field(default_factory=dict)
    transformed_patch_cross_covariance_cache: dict[
        tuple[tuple[int, ...], tuple[object, ...]], NDArray[np.float32]
    ] = field(default_factory=dict)
    gaussian_exact_row_structure_cache: dict[
        tuple[int, int, int],
        tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.int64]],
    ] = field(default_factory=dict)
    white_exact_source_checked: bool = False
    white_exact_source_variance: float | None = None
    white_exact_source_model: PoissonVarianceModel | None = None
    poisson_cache_limits: PoissonCacheLimits | None = None


@dataclass
class PoissonVarianceModel:
    global_sigma: float
    variance_volume: NDArray[np.float32]
    variance_floor: float
    source_name: str
    constant_variance: float | None = None
    patch_view_cache: dict[tuple[int, ...], NDArray[np.float32]] = field(
        default_factory=dict
    )
    patch_local_coords_cache: dict[tuple[int, ...], NDArray[np.int64]] = field(
        default_factory=dict
    )
    overlap_structure_cache: OrderedDict[
        tuple[tuple[int, ...], tuple[int, ...], str, bytes],
        tuple[
            NDArray[np.int64],
            NDArray[np.int64],
            NDArray[np.int64],
            NDArray[np.int64],
            NDArray[np.int64],
        ],
    ] = field(default_factory=OrderedDict)
    spatial_transform_cache: dict[
        tuple[tuple[int, ...], tuple[object, ...]],
        NDArray[np.float32],
    ] = field(default_factory=dict)
    compact_entry_transform_t_cache: dict[
        tuple[tuple[int, ...], tuple[object, ...], int],
        NDArray[np.float32],
    ] = field(default_factory=dict)
    exact_risk_metric_cache: dict[
        tuple[tuple[int, ...], tuple[object, ...], tuple[int, ...], bytes],
        tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]],
    ] = field(default_factory=dict)
    exact_contribution_cache: OrderedDict[
        tuple[tuple[int, ...], tuple[object, ...], tuple[int, ...]],
        NDArray[np.float32],
    ] = field(default_factory=OrderedDict)
    exact_factorized_contribution_cache: OrderedDict[
        tuple[object, ...],
        tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]],
    ] = field(default_factory=OrderedDict)
    covariance_cache_stats: PoissonCovarianceCacheStats = field(
        default_factory=PoissonCovarianceCacheStats
    )
    overlap_cache_max_bytes: int = DEFAULT_POISSON_CACHE_LIMITS.overlap_max_bytes
    overlap_cache_max_entries: int = (
        DEFAULT_POISSON_CACHE_LIMITS.overlap_max_entries
    )
    contribution_cache_max_bytes: int = (
        DEFAULT_POISSON_CACHE_LIMITS.contribution_max_bytes
    )
    contribution_cache_max_entries: int = (
        DEFAULT_POISSON_CACHE_LIMITS.contribution_max_entries
    )
    factorized_cache_max_bytes: int = (
        DEFAULT_POISSON_CACHE_LIMITS.factorized_max_bytes
    )
    factorized_cache_max_entries: int = (
        DEFAULT_POISSON_CACHE_LIMITS.factorized_max_entries
    )
    factorized_entry_cache_max_bytes: int = (
        DEFAULT_POISSON_CACHE_LIMITS.factorized_entry_max_bytes
    )


def _apply_poisson_cache_limits(
    model: PoissonVarianceModel,
    cache_limits: PoissonCacheLimits | None,
) -> PoissonVarianceModel:
    if cache_limits is None:
        return model
    model.overlap_cache_max_bytes = cache_limits.overlap_max_bytes
    model.overlap_cache_max_entries = cache_limits.overlap_max_entries
    model.contribution_cache_max_bytes = cache_limits.contribution_max_bytes
    model.contribution_cache_max_entries = cache_limits.contribution_max_entries
    model.factorized_cache_max_bytes = cache_limits.factorized_max_bytes
    model.factorized_cache_max_entries = cache_limits.factorized_max_entries
    model.factorized_entry_cache_max_bytes = cache_limits.factorized_entry_max_bytes
    return model


@dataclass(frozen=True)
class ExactPoissonCovarianceFactors:
    group_matrix: NDArray[np.float32]
    spatial_matrix: NDArray[np.float32]
    union_variance: NDArray[np.float32]
    sorted_entry_indices: NDArray[np.int64]
    segment_starts: NDArray[np.int64]
    segment_lengths: NDArray[np.int64]


@dataclass
class ExactPoissonGroupPlan:
    variance_model: PoissonVarianceModel
    group_shape: tuple[int, ...]
    group_matrix: NDArray[np.float32]
    spatial_matrix: NDArray[np.float32]
    union_variance: NDArray[np.float32]
    sorted_entry_indices: NDArray[np.int64]
    segment_starts: NDArray[np.int64]
    segment_lengths: NDArray[np.int64]
    group_column_starts: NDArray[np.int64]
    group_output_indices: NDArray[np.int64]
    scaled_spatial_entries: NDArray[np.float32]
    group_size: int = field(init=False)
    patch_volume: int = field(init=False)
    contributions_cache: NDArray[np.float32] | None = None
    grouped_contributions_cache: NDArray[np.float32] | None = None
    prepared_risk_contributions_cache: NDArray[np.float32] | None = None
    prepared_risk_plane_count: int = 0
    sigma_from_contributions_cache: NDArray[np.float32] | None = None
    risk_metric_cache: tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
    ] | None = None
    risk_metric_signature: tuple[tuple[object, ...], bytes] | None = None
    risk_group_metric_diag_cache: NDArray[np.float32] | None = None
    risk_group_metric_diagonal_cache: bool | None = None

    def __post_init__(self) -> None:
        self.group_shape = tuple(int(size) for size in self.group_shape)
        self.group_size = int(self.group_shape[0])
        self.patch_volume = int(np.prod(self.group_shape[1:]))

    def covariance_factors(self) -> ExactPoissonCovarianceFactors:
        return ExactPoissonCovarianceFactors(
            group_matrix=self.group_matrix,
            spatial_matrix=self.spatial_matrix,
            union_variance=self.union_variance,
            sorted_entry_indices=self.sorted_entry_indices,
            segment_starts=self.segment_starts,
            segment_lengths=self.segment_lengths,
        )

    def get_contribution_chunk(
        self,
        voxel_start: int,
        voxel_count: int,
        output_plane_count: int | None = None,
    ) -> NDArray[np.float32]:
        total_voxels = int(self.union_variance.shape[0])
        if (
            voxel_start < 0
            or voxel_count < 0
            or voxel_start + voxel_count > total_voxels
        ):
            raise ValueError(
                "Exact Poisson contribution chunk is outside the union-voxel range: "
                f"start={voxel_start}, count={voxel_count}, total={total_voxels}."
            )
        plane_count = (
            self.group_size
            if output_plane_count is None
            else min(max(int(output_plane_count), 0), self.group_size)
        )
        if (
            self.prepared_risk_contributions_cache is not None
            and self.prepared_risk_plane_count >= plane_count
        ):
            return self.prepared_risk_contributions_cache[
                voxel_start : voxel_start + voxel_count,
                :plane_count,
            ]
        if self.scaled_spatial_entries.size == 0:
            return _build_exact_poisson_contribution_chunk_direct_numba(
                self.group_matrix,
                self.spatial_matrix,
                self.sorted_entry_indices,
                self.segment_starts,
                self.segment_lengths,
                voxel_start,
                voxel_count,
                plane_count,
            )
        return _build_exact_poisson_contribution_chunk_sparse_numba(
            self.group_column_starts,
            self.group_output_indices,
            self.scaled_spatial_entries,
            self.sorted_entry_indices,
            self.segment_starts,
            self.segment_lengths,
            voxel_start,
            voxel_count,
            plane_count,
        )

    def prepare_noise_std_for_risk(
        self,
        exact_plane_count: int,
    ) -> NDArray[np.float32]:
        plane_count = min(int(exact_plane_count), self.group_size)
        contribution_bytes = (
            int(self.union_variance.shape[0])
            * plane_count
            * self.patch_volume
            * np.dtype(np.float32).itemsize
        )
        if contribution_bytes > _EXACT_POISSON_PREPARED_CONTRIBUTION_MAX_BYTES:
            return self.get_noise_std_from_factors(exact_plane_count=plane_count)

        self.prepared_risk_contributions_cache = self.get_contribution_chunk(
            0,
            int(self.union_variance.shape[0]),
            output_plane_count=plane_count,
        )
        self.prepared_risk_plane_count = plane_count
        variance = _accumulate_exact_poisson_variance_from_contributions_numba(
            self.prepared_risk_contributions_cache,
            self.union_variance,
            plane_count,
        )
        return np.sqrt(np.maximum(variance, 0.0)).reshape(
            (plane_count,) + self.group_shape[1:]
        ).astype(np.float32, copy=False)

    def release_prepared_risk_contributions(self) -> None:
        self.prepared_risk_contributions_cache = None
        self.prepared_risk_plane_count = 0

    def _cache_contribution_views(self, contributions: NDArray[np.float32]) -> NDArray[np.float32]:
        self.contributions_cache = contributions
        grouped = contributions.reshape(-1, self.group_size, self.patch_volume)
        self.grouped_contributions_cache = grouped
        return grouped

    def get_contributions(self) -> NDArray[np.float32]:
        if self.contributions_cache is None:
            prepared = self.prepared_risk_contributions_cache is not None
            contributions = self.get_contribution_chunk(
                0,
                int(self.union_variance.shape[0]),
            ).reshape(int(self.union_variance.shape[0]), -1)
            if prepared:
                contributions = contributions.copy()
            self._cache_contribution_views(contributions)
        return self.contributions_cache

    def get_grouped_contributions(self) -> NDArray[np.float32]:
        if self.grouped_contributions_cache is None:
            self._cache_contribution_views(self.get_contributions())
        return self.grouped_contributions_cache

    def get_contributions_and_noise_std(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if self.contributions_cache is None or self.sigma_from_contributions_cache is None:
            prepared = self.prepared_risk_contributions_cache is not None
            contributions = self.get_contribution_chunk(
                0,
                int(self.union_variance.shape[0]),
            ).reshape(int(self.union_variance.shape[0]), -1)
            if prepared:
                contributions = contributions.copy()
            self._cache_contribution_views(contributions)
            self.sigma_from_contributions_cache = (
                get_exact_direct_poisson_noise_std_from_factors(
                    self.group_shape,
                    self.covariance_factors(),
                )
            )
        assert self.contributions_cache is not None
        assert self.sigma_from_contributions_cache is not None
        return self.contributions_cache, self.sigma_from_contributions_cache

    def get_noise_std_from_factors(
        self,
        exact_plane_count: int | None = None,
    ) -> NDArray[np.float32]:
        if exact_plane_count is None and self.sigma_from_contributions_cache is not None:
            return self.sigma_from_contributions_cache
        sigma = get_exact_direct_poisson_noise_std_from_factors(
            self.group_shape,
            self.covariance_factors(),
            exact_plane_count=exact_plane_count,
        )
        if exact_plane_count is None:
            self.sigma_from_contributions_cache = sigma
        return sigma

    def get_risk_metrics(
        self,
        inverse_transforms: list[NDArray[np.floating] | Callable | None],
        w_patch: NDArray[np.floating],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        window = np.asarray(w_patch, dtype=np.float32)
        signature = (
            _matrix_transform_signature(list(inverse_transforms)),
            window.tobytes(),
        )
        if self.risk_metric_cache is None or self.risk_metric_signature != signature:
            self.risk_metric_cache = get_exact_poisson_risk_metrics(
                self.variance_model,
                self.group_shape,
                list(inverse_transforms),
                window,
            )
            self.risk_metric_signature = signature
            group_metric = self.risk_metric_cache[0]
            group_metric_diag = np.diag(group_metric).astype(np.float32, copy=False)
            group_metric_offdiag = group_metric - np.diag(group_metric_diag)
            self.risk_group_metric_diag_cache = group_metric_diag
            self.risk_group_metric_diagonal_cache = bool(
                np.all(np.abs(group_metric_offdiag) <= 1e-6)
            )
        return self.risk_metric_cache

    def get_prepared_risk_metrics(
        self,
        inverse_transforms: list[NDArray[np.floating] | Callable | None],
        w_patch: NDArray[np.floating],
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        bool,
    ]:
        group_metric, spatial_metric, coefficient_metric_diag = self.get_risk_metrics(
            inverse_transforms,
            w_patch,
        )
        assert self.risk_group_metric_diag_cache is not None
        assert self.risk_group_metric_diagonal_cache is not None
        return (
            group_metric,
            spatial_metric,
            coefficient_metric_diag,
            self.risk_group_metric_diag_cache,
            bool(self.risk_group_metric_diagonal_cache),
        )

    def get_prepared_diagonal_risk_metrics(
        self,
        inverse_transforms: list[NDArray[np.floating] | Callable | None],
        w_patch: NDArray[np.floating],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]] | None:
        _, spatial_metric, coefficient_metric_diag = self.get_risk_metrics(
            inverse_transforms,
            w_patch,
        )
        assert self.risk_group_metric_diag_cache is not None
        assert self.risk_group_metric_diagonal_cache is not None
        if not self.risk_group_metric_diagonal_cache:
            return None
        return (
            spatial_metric,
            coefficient_metric_diag,
            self.risk_group_metric_diag_cache,
        )

    def get_group_metric_diag(
        self,
        inverse_transforms: list[NDArray[np.floating] | Callable | None],
        w_patch: NDArray[np.floating],
    ) -> NDArray[np.float32]:
        self.get_risk_metrics(inverse_transforms, w_patch)
        assert self.risk_group_metric_diag_cache is not None
        return self.risk_group_metric_diag_cache

    def is_group_metric_diagonal(
        self,
        inverse_transforms: list[NDArray[np.floating] | Callable | None],
        w_patch: NDArray[np.floating],
        tol: float = 1e-6,
    ) -> bool:
        self.get_risk_metrics(inverse_transforms, w_patch)
        if self.risk_group_metric_diagonal_cache is None:
            return False
        return bool(self.risk_group_metric_diagonal_cache)


def build_variance_model(
    volume: NDArray[np.floating],
    sigma: float | None = None,
    sigma_psd: NDArray[np.floating] | float | None = None,
    nf: tuple[int, ...] | int | None = None,
    poisson_cache_limits: PoissonCacheLimits | None = None,
) -> VarianceModel | None:
    if sigma is not None and sigma_psd is not None:
        raise ValueError("sigma and sigma_psd are mutually exclusive.")
    if sigma is None and sigma_psd is None:
        sigma = estimate_gaussian_global_sigma(volume)

    shape = tuple(int(dim) for dim in volume.shape)
    resolved_nf, _ = resolve_nf(nf, shape)
    normalized_psd = normalize_sigma_psd(shape, sigma=sigma, sigma_psd=sigma_psd)
    assert normalized_psd is not None
    if sigma is not None:
        is_scalar_input = True
    elif sigma_psd is None:
        is_scalar_input = True
    else:
        sigma_psd_arr = np.asarray(sigma_psd)
        is_scalar_input = sigma_psd_arr.ndim == 0 or sigma_psd_arr.size == 1

    processed_psd = preprocess_psd(
        normalized_psd,
        resolved_nf,
        single_dim_psd=is_scalar_input,
    )
    processed_sigma_psd = processed_psd.blurred_sigma_psd

    volume_size = float(np.prod(shape))
    global_sigma = float(
        np.sqrt(
            max(float(np.mean(normalized_psd, dtype=np.float32)) / volume_size, 0.0)
        )
    )
    autocovariance = autocovariance_from_psd(processed_sigma_psd)
    variance_domain_shape = tuple(int(size) for size in processed_sigma_psd.shape)
    if resolved_nf is not None:
        variance_domain_shape = resolved_nf

    return VarianceModel(
        global_sigma=max(global_sigma, 0.0),
        sigma_psd=normalized_psd,
        reduced_sigma_psd=processed_psd.reduced_sigma_psd,
        processed_sigma_psd=processed_sigma_psd,
        blur_kernel=processed_psd.blur_kernel,
        correlation_kernel=processed_psd.correlation_kernel,
        autocovariance=autocovariance,
        variance_domain_shape=variance_domain_shape,
        poisson_cache_limits=poisson_cache_limits,
    )


def get_gaussian_blockmatch_noise_ssd(
    variance_model: VarianceModel,
    block_shape: tuple[int, ...],
    search_radius: tuple[int, ...],
) -> NDArray[np.float32]:
    ndim = variance_model.autocovariance.ndim
    if len(block_shape) != ndim or any(size <= 0 for size in block_shape):
        raise ValueError(
            f"block_shape must contain {ndim} positive values, got {block_shape}."
        )
    if len(search_radius) != ndim or any(radius < 0 for radius in search_radius):
        raise ValueError(
            f"search_radius must contain {ndim} non-negative values, "
            f"got {search_radius}."
        )

    cache_key = (block_shape, search_radius)
    cached = variance_model.blockmatch_noise_ssd_cache.get(cache_key)
    if cached is not None:
        return cached

    autocovariance = variance_model.autocovariance
    autocovariance_shape = tuple(int(size) for size in autocovariance.shape)
    output_shape = tuple(2 * radius + 1 for radius in search_radius)
    noise_ssd = np.empty(output_shape, dtype=np.float32)
    zero_lag = (0,) * ndim
    variance = float(autocovariance[zero_lag])
    patch_volume = float(np.prod(block_shape))

    for output_index in np.ndindex(output_shape):
        lag = tuple(output_index[axis] - search_radius[axis] for axis in range(ndim))
        positive_index = tuple(
            lag[axis] % autocovariance_shape[axis] for axis in range(ndim)
        )
        negative_index = tuple(
            (-lag[axis]) % autocovariance_shape[axis] for axis in range(ndim)
        )
        lag_covariance = 0.5 * float(
            autocovariance[positive_index] + autocovariance[negative_index]
        )
        noise_ssd[output_index] = np.float32(
            max(2.0 * patch_volume * (variance - lag_covariance), 0.0)
        )

    noise_ssd[search_radius] = np.float32(0.0)
    variance_model.blockmatch_noise_ssd_cache[cache_key] = noise_ssd
    return noise_ssd


def build_poisson_variance_model(
    volume: NDArray[np.floating],
    *,
    variance_floor: float = 1e-10,
    scale: float = 1.0,
    source_name: str,
    cache_limits: PoissonCacheLimits | None = None,
) -> PoissonVarianceModel:
    if isinstance(variance_floor, (bool, np.bool_)):
        raise ValueError(
            f"variance_floor must be finite and non-negative, got {variance_floor!r}."
        )
    try:
        variance_floor = float(variance_floor)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"variance_floor must be finite and non-negative, got {variance_floor!r}."
        ) from exc
    if not np.isfinite(variance_floor) or variance_floor < 0.0:
        raise ValueError(
            f"variance_floor must be finite and non-negative, got {variance_floor}."
        )
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale}.")
    nonnegative_volume = np.maximum(np.asarray(volume, dtype=np.float32), 0.0)
    variance_volume = np.maximum(
        scale * nonnegative_volume,
        variance_floor,
    ).astype(np.float32, copy=False)
    global_sigma = estimate_poisson_global_sigma(
        variance_volume,
        eps=max(variance_floor, 1e-10),
    )
    return _apply_poisson_cache_limits(
        PoissonVarianceModel(
            global_sigma=max(global_sigma, 0.0),
            variance_volume=variance_volume,
            variance_floor=variance_floor,
            source_name=source_name,
        ),
        cache_limits,
    )


def get_poisson_variance_patches(
    variance_model: PoissonVarianceModel,
    block_size: tuple[int, ...],
) -> NDArray[np.float32]:
    if block_size not in variance_model.patch_view_cache:
        variance_model.patch_view_cache[block_size] = extract_patches_strided(
            variance_model.variance_volume,
            block_size,
        )
    return variance_model.patch_view_cache[block_size]


def _apply_squared_transform_along_axis(
    variance: NDArray[np.float32],
    transform: NDArray[np.floating] | Callable | None,
    axis: int,
) -> NDArray[np.float32]:
    if transform is None or callable(transform):
        return variance

    transform_sq = np.asarray(transform, dtype=np.float32) ** 2
    updated = _apply_matrix_along_axis(variance, transform_sq, axis)
    return np.asarray(updated, dtype=np.float32)


def _require_matrix_transforms(
    forward_transform: list[NDArray[np.floating] | Callable | None],
) -> None:
    for axis, transform in enumerate(forward_transform):
        if callable(transform):
            raise ValueError(
                "Exact overlap-aware Poisson group covariance requires matrix "
                f"transforms, got callable transform on axis {axis}."
            )


def _get_patch_local_coords(
    variance_model: PoissonVarianceModel,
    block_shape: tuple[int, ...],
) -> NDArray[np.int64]:
    if block_shape not in variance_model.patch_local_coords_cache:
        variance_model.patch_local_coords_cache[block_shape] = np.stack(
            np.meshgrid(
                *[np.arange(size, dtype=np.int64) for size in block_shape],
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, len(block_shape))
    return variance_model.patch_local_coords_cache[block_shape]


def _build_poisson_overlap_structure(
    variance_model: PoissonVarianceModel,
    selected_shifted_positions: NDArray[np.int64],
    block_shape: tuple[int, ...],
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    shifted_positions = np.ascontiguousarray(
        selected_shifted_positions,
        dtype=np.int64,
    )
    cache_key = (
        block_shape,
        shifted_positions.shape,
        shifted_positions.dtype.str,
        shifted_positions.tobytes(),
    )
    cached = variance_model.overlap_structure_cache.get(cache_key)
    if cached is not None:
        variance_model.covariance_cache_stats.overlap_hits += 1
        variance_model.overlap_structure_cache.move_to_end(cache_key)
        return cached
    variance_model.covariance_cache_stats.overlap_misses += 1

    local_coords = _get_patch_local_coords(variance_model, block_shape)
    (
        entry_labels,
        unique_coords,
        sorted_entry_indices,
        segment_starts,
        segment_lengths,
    ) = _build_poisson_overlap_structure_numba(
        shifted_positions,
        local_coords,
        np.asarray(block_shape, dtype=np.int64),
    )
    labels = entry_labels.reshape(
        selected_shifted_positions.shape[0], local_coords.shape[0]
    )
    result = (
        labels.astype(np.int64, copy=False),
        unique_coords.astype(np.int64, copy=False),
        sorted_entry_indices,
        segment_starts,
        segment_lengths,
    )
    result_bytes = sum(array.nbytes for array in result)
    if (
        variance_model.overlap_cache_max_entries > 0
        and result_bytes <= variance_model.overlap_cache_max_bytes
    ):
        while variance_model.overlap_structure_cache and (
            len(variance_model.overlap_structure_cache)
            >= variance_model.overlap_cache_max_entries
            or variance_model.covariance_cache_stats.overlap_bytes + result_bytes
            > variance_model.overlap_cache_max_bytes
        ):
            _, evicted = variance_model.overlap_structure_cache.popitem(last=False)
            variance_model.covariance_cache_stats.overlap_bytes -= sum(
                array.nbytes for array in evicted
            )
            variance_model.covariance_cache_stats.overlap_evictions += 1
        variance_model.overlap_structure_cache[cache_key] = result
        variance_model.covariance_cache_stats.overlap_bytes += result_bytes
    return result


def _gather_group_union_variances(
    variance_model: PoissonVarianceModel,
    selected_abs_positions: NDArray[np.int64],
    selected_shifted_positions: NDArray[np.int64],
    unique_shifted_coords: NDArray[np.int64],
) -> NDArray[np.float32]:
    if variance_model.constant_variance is not None:
        return np.full(
            unique_shifted_coords.shape[0],
            variance_model.constant_variance,
            dtype=np.float32,
        )
    group_origin_abs = np.min(selected_abs_positions, axis=0, keepdims=True)
    group_origin_shifted = np.min(selected_shifted_positions, axis=0, keepdims=True)
    unique_abs_coords = unique_shifted_coords - group_origin_shifted + group_origin_abs
    return np.asarray(
        variance_model.variance_volume[tuple(unique_abs_coords.T)],
        dtype=np.float32,
    )


def _full_separable_transform_matrix_for_axes(
    axis_sizes: tuple[int, ...],
    transforms: list[NDArray[np.floating] | Callable | None],
) -> NDArray[np.float32]:
    full: NDArray[np.float32] = np.array([[1.0]], dtype=np.float32)
    for axis_size, transform in zip(axis_sizes, transforms, strict=False):
        if transform is None:
            matrix = np.eye(axis_size, dtype=np.float32)
        else:
            matrix = np.asarray(transform, dtype=np.float32)
        full = np.kron(full, matrix).astype(np.float32, copy=False)
    return full


@njit(cache=True, nogil=True)
def _build_poisson_overlap_structure_numba(
    selected_shifted_positions: NDArray[np.int64],
    local_coords: NDArray[np.int64],
    block_shape: NDArray[np.int64],
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    group_size = selected_shifted_positions.shape[0]
    patch_vol = local_coords.shape[0]
    ndim = selected_shifted_positions.shape[1]
    entry_count = group_size * patch_vol

    union_shape = np.empty(ndim, dtype=np.int64)
    for axis in range(ndim):
        max_shift = 0
        for group_idx in range(group_size):
            value = selected_shifted_positions[group_idx, axis]
            if group_idx == 0 or value > max_shift:
                max_shift = value
        union_shape[axis] = max_shift + block_shape[axis]

    union_strides = np.empty(ndim, dtype=np.int64)
    stride = 1
    for axis in range(ndim - 1, -1, -1):
        union_strides[axis] = stride
        stride *= union_shape[axis]

    flat_labels = np.empty(entry_count, dtype=np.int64)
    cursor = 0
    for group_idx in range(group_size):
        for patch_idx in range(patch_vol):
            flat_label = 0
            for axis in range(ndim):
                coord = (
                    selected_shifted_positions[group_idx, axis]
                    + local_coords[patch_idx, axis]
                )
                flat_label += coord * union_strides[axis]
            flat_labels[cursor] = flat_label
            cursor += 1

    sorted_entry_indices = np.argsort(flat_labels).astype(np.int64)
    sorted_labels = flat_labels[sorted_entry_indices]

    unique_count = 0
    previous_label = np.int64(-1)
    for idx in range(entry_count):
        label = sorted_labels[idx]
        if idx == 0 or label != previous_label:
            unique_count += 1
            previous_label = label

    entry_labels = np.empty(entry_count, dtype=np.int64)
    unique_flat_labels = np.empty(unique_count, dtype=np.int64)
    segment_starts = np.empty(unique_count, dtype=np.int64)
    segment_lengths = np.empty(unique_count, dtype=np.int64)

    segment_idx = -1
    segment_start = 0
    previous_label = np.int64(-1)
    for idx in range(entry_count):
        label = sorted_labels[idx]
        if idx == 0 or label != previous_label:
            if idx > 0:
                segment_lengths[segment_idx] = idx - segment_start
            segment_idx += 1
            unique_flat_labels[segment_idx] = label
            segment_starts[segment_idx] = idx
            segment_start = idx
            previous_label = label
        entry_labels[sorted_entry_indices[idx]] = segment_idx
    if unique_count > 0:
        segment_lengths[segment_idx] = entry_count - segment_start

    unique_coords = np.empty((unique_count, ndim), dtype=np.int64)
    for idx in range(unique_count):
        remainder = unique_flat_labels[idx]
        for axis in range(ndim):
            stride_value = union_strides[axis]
            unique_coords[idx, axis] = remainder // stride_value
            remainder = remainder % stride_value

    return (
        entry_labels,
        unique_coords,
        sorted_entry_indices,
        segment_starts,
        segment_lengths,
    )


def _matrix_transform_signature(
    transforms: list[NDArray[np.floating] | Callable | None],
) -> tuple[object, ...]:
    _require_matrix_transforms(transforms)
    signature: list[object] = []
    for transform in transforms:
        if transform is None:
            signature.append(("none",))
            continue
        matrix = np.asarray(transform, dtype=np.float32)
        signature.append(("matrix", id(transform), matrix.shape, matrix.dtype.str))
    return tuple(signature)


def _matrix_content_transform_signature(
    transforms: list[NDArray[np.floating] | Callable | None],
) -> tuple[object, ...]:
    _require_matrix_transforms(transforms)
    signature: list[object] = []
    for transform in transforms:
        if transform is None:
            signature.append(("none",))
            continue
        matrix = np.ascontiguousarray(transform, dtype=np.float32)
        signature.append(("matrix", matrix.shape, matrix.dtype.str, matrix.tobytes()))
    return tuple(signature)


def _exact_poisson_contribution_cache_key(
    group_shape: tuple[int, ...],
    transforms: list[NDArray[np.floating] | Callable | None],
    selected_shifted_positions: NDArray[np.int64],
) -> tuple[tuple[int, ...], tuple[object, ...], tuple[int, ...]]:
    return (
        tuple(int(size) for size in group_shape),
        _matrix_transform_signature(transforms),
        tuple(int(x) for x in np.asarray(selected_shifted_positions, dtype=np.int64).reshape(-1)),
    )


def _get_cached_spatial_transform_matrix(
    variance_model: PoissonVarianceModel,
    block_shape: tuple[int, ...],
    transforms: list[NDArray[np.floating] | Callable | None],
) -> NDArray[np.float32]:
    cache_key = (block_shape, _matrix_transform_signature(transforms))
    cached = variance_model.spatial_transform_cache.get(cache_key)
    if cached is not None:
        return cached

    spatial_matrix = _full_separable_transform_matrix_for_axes(block_shape, transforms)
    variance_model.spatial_transform_cache[cache_key] = spatial_matrix
    return spatial_matrix


def _get_group_transform_matrix(
    group_shape: tuple[int, ...],
    forward_transform: list[NDArray[np.floating] | Callable | None],
) -> NDArray[np.float32]:
    group_size = int(group_shape[0])
    group_transform = forward_transform[0]
    if group_transform is None:
        return np.eye(group_size, dtype=np.float32)
    return np.ascontiguousarray(group_transform, dtype=np.float32)


def _get_cached_compact_entry_transform_t(
    variance_model: PoissonVarianceModel,
    group_shape: tuple[int, ...],
    forward_transform: list[NDArray[np.floating] | Callable | None],
    exact_plane_count: int,
) -> NDArray[np.float32]:
    cache_key = (
        group_shape,
        _matrix_transform_signature(list(forward_transform)),
        exact_plane_count,
    )
    cached = variance_model.compact_entry_transform_t_cache.get(cache_key)
    if cached is not None:
        return cached

    group_matrix = _get_group_transform_matrix(group_shape, forward_transform)
    spatial_matrix = _get_cached_spatial_transform_matrix(
        variance_model,
        tuple(int(size) for size in group_shape[1:]),
        list(forward_transform[1:]),
    )
    compact_entry_transform_t = np.ascontiguousarray(
        np.kron(group_matrix[:exact_plane_count], spatial_matrix)
        .astype(np.float32, copy=False)
        .T
    )
    variance_model.compact_entry_transform_t_cache[cache_key] = compact_entry_transform_t
    return compact_entry_transform_t


@njit(cache=True, nogil=True)
def _accumulate_exact_poisson_variance_from_entry_transform_numba(
    entry_transform_t: NDArray[np.float32],
    sorted_entry_indices: NDArray[np.int64],
    segment_starts: NDArray[np.int64],
    segment_lengths: NDArray[np.int64],
    union_variance: NDArray[np.float32],
) -> NDArray[np.float32]:
    coefficient_count = entry_transform_t.shape[1]
    variance_flat = np.zeros(coefficient_count, dtype=np.float32)
    contribution = np.empty(coefficient_count, dtype=np.float32)

    for voxel_id in range(union_variance.shape[0]):
        for coefficient_idx in range(coefficient_count):
            contribution[coefficient_idx] = 0.0
        start = segment_starts[voxel_id]
        end = start + segment_lengths[voxel_id]
        for pos in range(start, end):
            row = entry_transform_t[sorted_entry_indices[pos]]
            for coefficient_idx in range(coefficient_count):
                contribution[coefficient_idx] += row[coefficient_idx]
        voxel_variance = union_variance[voxel_id]
        for coefficient_idx in range(coefficient_count):
            value = contribution[coefficient_idx]
            variance_flat[coefficient_idx] += voxel_variance * value * value
    return variance_flat


@njit(cache=True, parallel=True, nogil=True)
def _accumulate_exact_poisson_plane_variance_numba(
    exact_group_rows: NDArray[np.float32],
    spatial_matrix: NDArray[np.float32],
    sorted_entry_indices: NDArray[np.int64],
    segment_starts: NDArray[np.int64],
    segment_lengths: NDArray[np.int64],
    union_variance: NDArray[np.float32],
) -> NDArray[np.float32]:
    exact_plane_count = exact_group_rows.shape[0]
    group_size = exact_group_rows.shape[1]
    patch_volume = spatial_matrix.shape[0]
    coefficient_count = exact_plane_count * patch_volume
    variance_flat = np.empty(coefficient_count, dtype=np.float32)

    for coefficient_idx in prange(coefficient_count):
        plane_idx = coefficient_idx // patch_volume
        spatial_idx = coefficient_idx - plane_idx * patch_volume
        accumulated = np.float32(0.0)
        for voxel_id in range(union_variance.shape[0]):
            contribution = np.float32(0.0)
            start = segment_starts[voxel_id]
            end = start + segment_lengths[voxel_id]
            for pos in range(start, end):
                entry_idx = sorted_entry_indices[pos]
                group_idx = entry_idx // patch_volume
                patch_idx = entry_idx - group_idx * patch_volume
                if group_idx < group_size:
                    contribution += (
                        exact_group_rows[plane_idx, group_idx]
                        * spatial_matrix[spatial_idx, patch_idx]
                    )
            accumulated += union_variance[voxel_id] * contribution * contribution
        variance_flat[coefficient_idx] = accumulated
    variance = variance_flat.reshape(exact_plane_count, patch_volume)
    return variance


@njit(cache=True, parallel=True, nogil=True)
def _accumulate_exact_poisson_variance_from_contributions_numba(
    contributions: NDArray[np.float32],
    union_variance: NDArray[np.float32],
    exact_plane_count: int,
) -> NDArray[np.float32]:
    patch_volume = contributions.shape[2]
    coefficient_count = exact_plane_count * patch_volume
    variance = np.empty(coefficient_count, dtype=np.float32)

    for coefficient_idx in prange(coefficient_count):
        group_idx = coefficient_idx // patch_volume
        patch_idx = coefficient_idx - group_idx * patch_volume
        accumulated = np.float32(0.0)
        for voxel_id in range(union_variance.shape[0]):
            contribution = contributions[voxel_id, group_idx, patch_idx]
            accumulated += union_variance[voxel_id] * contribution * contribution
        variance[coefficient_idx] = accumulated
    return variance


@njit(cache=True, parallel=True, nogil=True)
def _build_exact_poisson_contribution_chunk_sparse_numba(
    group_column_starts: NDArray[np.int64],
    group_output_indices: NDArray[np.int64],
    scaled_spatial_entries: NDArray[np.float32],
    sorted_entry_indices: NDArray[np.int64],
    segment_starts: NDArray[np.int64],
    segment_lengths: NDArray[np.int64],
    voxel_start: int,
    voxel_count: int,
    output_plane_count: int,
) -> NDArray[np.float32]:
    patch_volume = scaled_spatial_entries.shape[1]
    contributions = np.zeros(
        (voxel_count, output_plane_count, patch_volume),
        dtype=np.float32,
    )

    for local_voxel_id in prange(voxel_count):
        voxel_id = voxel_start + local_voxel_id
        start = segment_starts[voxel_id]
        end = start + segment_lengths[voxel_id]
        row_out = contributions[local_voxel_id]
        for pos in range(start, end):
            entry_idx = sorted_entry_indices[pos]
            input_group_idx = entry_idx // patch_volume
            input_patch_idx = entry_idx - input_group_idx * patch_volume
            edge_start = group_column_starts[input_group_idx]
            edge_end = group_column_starts[input_group_idx + 1]
            for edge_idx in range(edge_start, edge_end):
                output_group_idx = group_output_indices[edge_idx]
                if output_group_idx >= output_plane_count:
                    # Each cached input-column list is ordered by output index.
                    break
                scaled_spatial = scaled_spatial_entries[edge_idx]
                for output_patch_idx in range(patch_volume):
                    row_out[output_group_idx, output_patch_idx] += scaled_spatial[
                        input_patch_idx,
                        output_patch_idx,
                    ]
    return contributions


@njit(cache=True, parallel=True, nogil=True)
def _build_exact_poisson_contribution_chunk_direct_numba(
    group_matrix: NDArray[np.float32],
    spatial_matrix: NDArray[np.float32],
    sorted_entry_indices: NDArray[np.int64],
    segment_starts: NDArray[np.int64],
    segment_lengths: NDArray[np.int64],
    voxel_start: int,
    voxel_count: int,
    output_plane_count: int,
) -> NDArray[np.float32]:
    patch_volume = spatial_matrix.shape[0]
    contributions = np.zeros(
        (voxel_count, output_plane_count, patch_volume),
        dtype=np.float32,
    )

    for local_voxel_id in prange(voxel_count):
        voxel_id = voxel_start + local_voxel_id
        start = segment_starts[voxel_id]
        end = start + segment_lengths[voxel_id]
        row_out = contributions[local_voxel_id]
        for pos in range(start, end):
            entry_idx = sorted_entry_indices[pos]
            input_group_idx = entry_idx // patch_volume
            input_patch_idx = entry_idx - input_group_idx * patch_volume
            for output_group_idx in range(output_plane_count):
                group_weight = group_matrix[output_group_idx, input_group_idx]
                if group_weight == 0.0:
                    continue
                for output_patch_idx in range(patch_volume):
                    row_out[output_group_idx, output_patch_idx] += (
                        group_weight
                        * spatial_matrix[output_patch_idx, input_patch_idx]
                    )
    return contributions


def _build_exact_poisson_factorized_entry_cache(
    group_matrix: NDArray[np.float32],
    spatial_matrix: NDArray[np.float32],
    max_bytes: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]] | None:
    group_size = int(group_matrix.shape[0])
    patch_volume = int(spatial_matrix.shape[0])
    columns = [np.flatnonzero(group_matrix[:, column]) for column in range(group_size)]
    edge_count = sum(int(column.shape[0]) for column in columns)
    required_bytes = (
        edge_count
        * patch_volume
        * patch_volume
        * np.dtype(np.float32).itemsize
    )
    if required_bytes > max_bytes:
        return None

    group_column_starts = np.empty(group_size + 1, dtype=np.int64)
    group_output_indices = np.empty(edge_count, dtype=np.int64)
    scaled_spatial_entries = np.empty(
        (edge_count, patch_volume, patch_volume),
        dtype=np.float32,
    )
    edge_idx = 0
    for input_group_idx, output_indices in enumerate(columns):
        group_column_starts[input_group_idx] = edge_idx
        for output_group_idx in output_indices:
            group_output_indices[edge_idx] = output_group_idx
            np.multiply(
                spatial_matrix.T,
                group_matrix[output_group_idx, input_group_idx],
                out=scaled_spatial_entries[edge_idx],
            )
            edge_idx += 1
    group_column_starts[group_size] = edge_idx
    return group_column_starts, group_output_indices, scaled_spatial_entries


def _build_exact_poisson_contribution_chunk_from_factors(
    covariance_factors: ExactPoissonCovarianceFactors,
    voxel_start: int,
    voxel_count: int,
) -> NDArray[np.float32]:
    cached_factors = _build_exact_poisson_factorized_entry_cache(
        covariance_factors.group_matrix,
        covariance_factors.spatial_matrix,
        DEFAULT_POISSON_CACHE_LIMITS.factorized_entry_max_bytes,
    )
    if cached_factors is None:
        return _build_exact_poisson_contribution_chunk_direct_numba(
            covariance_factors.group_matrix,
            covariance_factors.spatial_matrix,
            covariance_factors.sorted_entry_indices,
            covariance_factors.segment_starts,
            covariance_factors.segment_lengths,
            voxel_start,
            voxel_count,
            int(covariance_factors.group_matrix.shape[0]),
        )
    group_column_starts, group_output_indices, scaled_spatial_entries = cached_factors
    return _build_exact_poisson_contribution_chunk_sparse_numba(
        group_column_starts,
        group_output_indices,
        scaled_spatial_entries,
        covariance_factors.sorted_entry_indices,
        covariance_factors.segment_starts,
        covariance_factors.segment_lengths,
        voxel_start,
        voxel_count,
        int(covariance_factors.group_matrix.shape[0]),
    )


def get_exact_direct_poisson_noise_std_from_factors(
    group_shape: tuple[int, ...],
    covariance_factors: ExactPoissonCovarianceFactors,
    exact_plane_count: int | None = None,
    compact_entry_transform_t: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    group_size = int(group_shape[0])
    plane_count = (
        group_size
        if exact_plane_count is None
        else min(int(exact_plane_count), group_size)
    )
    if compact_entry_transform_t is None:
        exact_group_rows = np.ascontiguousarray(
            covariance_factors.group_matrix[:plane_count],
            dtype=np.float32,
        )
        variance_flat = _accumulate_exact_poisson_plane_variance_numba(
            exact_group_rows,
            covariance_factors.spatial_matrix,
            covariance_factors.sorted_entry_indices,
            covariance_factors.segment_starts,
            covariance_factors.segment_lengths,
            covariance_factors.union_variance,
        ).reshape(-1)
    else:
        variance_flat = _accumulate_exact_poisson_variance_from_entry_transform_numba(
            compact_entry_transform_t,
            covariance_factors.sorted_entry_indices,
            covariance_factors.segment_starts,
            covariance_factors.segment_lengths,
            covariance_factors.union_variance,
        )
    variance = variance_flat.reshape((plane_count,) + tuple(group_shape[1:]))
    return np.sqrt(np.maximum(variance, 0.0)).astype(np.float32, copy=False)


def get_exact_poisson_contribution_matrix_from_factors(
    covariance_factors: ExactPoissonCovarianceFactors,
) -> NDArray[np.float32]:
    grouped = _build_exact_poisson_contribution_chunk_from_factors(
        covariance_factors,
        0,
        int(covariance_factors.union_variance.shape[0]),
    )
    return grouped.reshape(grouped.shape[0], -1)


def get_exact_poisson_contribution_chunk_from_factors(
    covariance_factors: ExactPoissonCovarianceFactors,
    voxel_start: int,
    voxel_count: int,
) -> NDArray[np.float32]:
    total_voxels = int(covariance_factors.union_variance.shape[0])
    if voxel_start < 0 or voxel_count < 0 or voxel_start + voxel_count > total_voxels:
        raise ValueError(
            "Exact Poisson contribution chunk is outside the union-voxel range: "
            f"start={voxel_start}, count={voxel_count}, total={total_voxels}."
        )
    return _build_exact_poisson_contribution_chunk_from_factors(
        covariance_factors,
        voxel_start,
        voxel_count,
    )


def get_exact_poisson_contribution_matrix(
    variance_model: PoissonVarianceModel,
    selected_shifted_positions: NDArray[np.int64],
    group_shape: tuple[int, ...],
    transforms: list[NDArray[np.floating] | Callable | None],
    covariance_factors: ExactPoissonCovarianceFactors | None = None,
) -> NDArray[np.float32]:
    cache_key = _exact_poisson_contribution_cache_key(
        group_shape,
        transforms,
        selected_shifted_positions,
    )
    cached = variance_model.exact_contribution_cache.get(cache_key)
    if cached is not None:
        variance_model.covariance_cache_stats.contribution_hits += 1
        variance_model.exact_contribution_cache.move_to_end(cache_key)
        return cached
    variance_model.covariance_cache_stats.contribution_misses += 1

    if covariance_factors is None:
        raise ValueError("covariance_factors are required to build exact contribution matrix")

    contributions = get_exact_poisson_contribution_matrix_from_factors(covariance_factors)
    contribution_bytes = contributions.nbytes
    if (
        variance_model.contribution_cache_max_entries > 0
        and contribution_bytes <= variance_model.contribution_cache_max_bytes
    ):
        while variance_model.exact_contribution_cache and (
            len(variance_model.exact_contribution_cache)
            >= variance_model.contribution_cache_max_entries
            or variance_model.covariance_cache_stats.contribution_bytes
            + contribution_bytes
            > variance_model.contribution_cache_max_bytes
        ):
            _, evicted = variance_model.exact_contribution_cache.popitem(last=False)
            variance_model.covariance_cache_stats.contribution_bytes -= evicted.nbytes
            variance_model.covariance_cache_stats.contribution_evictions += 1
        variance_model.exact_contribution_cache[cache_key] = contributions
        variance_model.covariance_cache_stats.contribution_bytes += contribution_bytes
    return contributions


def get_exact_poisson_covariance_factors(
    variance_model: PoissonVarianceModel,
    selected_abs_positions: NDArray[np.int64],
    selected_shifted_positions: NDArray[np.int64],
    group_shape: tuple[int, ...],
    transforms: list[NDArray[np.floating] | Callable | None],
) -> ExactPoissonCovarianceFactors:
    _require_matrix_transforms(transforms)

    group_size = int(group_shape[0])
    block_shape = tuple(int(size) for size in group_shape[1:])
    if selected_abs_positions.shape != (group_size, len(block_shape)):
        raise ValueError(
            "selected_abs_positions shape does not match group shape: "
            f"got {selected_abs_positions.shape}, expected {(group_size, len(block_shape))}."
        )
    if selected_shifted_positions.shape != selected_abs_positions.shape:
        raise ValueError(
            "selected_shifted_positions shape does not match selected_abs_positions: "
            f"got {selected_shifted_positions.shape}, expected {selected_abs_positions.shape}."
        )

    group_matrix = _get_group_transform_matrix(group_shape, transforms)
    spatial_matrix = _get_cached_spatial_transform_matrix(
        variance_model,
        block_shape,
        list(transforms[1:]),
    )
    (
        _voxel_labels,
        unique_shifted_coords,
        sorted_entry_indices,
        segment_starts,
        segment_lengths,
    ) = _build_poisson_overlap_structure(
        variance_model,
        selected_shifted_positions,
        block_shape,
    )
    union_variance = _gather_group_union_variances(
        variance_model,
        selected_abs_positions,
        selected_shifted_positions,
        unique_shifted_coords,
    )
    return ExactPoissonCovarianceFactors(
        group_matrix=group_matrix,
        spatial_matrix=spatial_matrix,
        union_variance=union_variance,
        sorted_entry_indices=sorted_entry_indices,
        segment_starts=segment_starts,
        segment_lengths=segment_lengths,
    )


def _get_cached_exact_poisson_factorized_entries(
    variance_model: PoissonVarianceModel,
    group_matrix: NDArray[np.float32],
    spatial_matrix: NDArray[np.float32],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]]:
    cache_key = (
        group_matrix.shape,
        group_matrix.dtype.str,
        hashlib.blake2b(group_matrix.tobytes(), digest_size=16).digest(),
        spatial_matrix.shape,
        spatial_matrix.dtype.str,
        hashlib.blake2b(spatial_matrix.tobytes(), digest_size=16).digest(),
    )
    cached = variance_model.exact_factorized_contribution_cache.get(cache_key)
    if cached is not None:
        variance_model.covariance_cache_stats.factorized_hits += 1
        variance_model.exact_factorized_contribution_cache.move_to_end(cache_key)
        return cached
    variance_model.covariance_cache_stats.factorized_misses += 1

    result = _build_exact_poisson_factorized_entry_cache(
        group_matrix,
        spatial_matrix,
        min(
            variance_model.factorized_entry_cache_max_bytes,
            variance_model.factorized_cache_max_bytes,
        ),
    )
    if result is None:
        result = (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 0, 0), dtype=np.float32),
        )
    result_bytes = sum(array.nbytes for array in result)
    if (
        variance_model.factorized_cache_max_entries > 0
        and result_bytes <= variance_model.factorized_cache_max_bytes
    ):
        while variance_model.exact_factorized_contribution_cache and (
            len(variance_model.exact_factorized_contribution_cache)
            >= variance_model.factorized_cache_max_entries
            or variance_model.covariance_cache_stats.factorized_bytes + result_bytes
            > variance_model.factorized_cache_max_bytes
        ):
            _, evicted = variance_model.exact_factorized_contribution_cache.popitem(
                last=False
            )
            variance_model.covariance_cache_stats.factorized_bytes -= sum(
                array.nbytes for array in evicted
            )
            variance_model.covariance_cache_stats.factorized_evictions += 1
        variance_model.exact_factorized_contribution_cache[cache_key] = result
        variance_model.covariance_cache_stats.factorized_bytes += result_bytes
    return result


def build_exact_poisson_group_plan(
    variance_model: PoissonVarianceModel,
    selected_abs_positions: NDArray[np.int64],
    selected_shifted_positions: NDArray[np.int64],
    group_shape: tuple[int, ...],
    transforms: list[NDArray[np.floating] | Callable | None],
) -> ExactPoissonGroupPlan:
    factors = get_exact_poisson_covariance_factors(
        variance_model,
        selected_abs_positions,
        selected_shifted_positions,
        group_shape,
        transforms,
    )
    (
        group_column_starts,
        group_output_indices,
        scaled_spatial_entries,
    ) = _get_cached_exact_poisson_factorized_entries(
        variance_model,
        factors.group_matrix,
        factors.spatial_matrix,
    )
    return ExactPoissonGroupPlan(
        variance_model=variance_model,
        group_shape=tuple(int(size) for size in group_shape),
        group_matrix=factors.group_matrix,
        spatial_matrix=factors.spatial_matrix,
        union_variance=factors.union_variance,
        sorted_entry_indices=factors.sorted_entry_indices,
        segment_starts=factors.segment_starts,
        segment_lengths=factors.segment_lengths,
        group_column_starts=group_column_starts,
        group_output_indices=group_output_indices,
        scaled_spatial_entries=scaled_spatial_entries,
    )


def build_exact_white_gaussian_group_plan(
    variance_model: VarianceModel,
    volume_shape: tuple[int, ...],
    selected_abs_positions: NDArray[np.int64],
    selected_shifted_positions: NDArray[np.int64],
    group_shape: tuple[int, ...],
    transforms: list[NDArray[np.floating] | Callable | None],
) -> ExactPoissonGroupPlan | None:
    if not variance_model.white_exact_source_checked:
        autocovariance = np.asarray(variance_model.autocovariance, dtype=np.float64)
        source_variance = float(autocovariance[(0,) * autocovariance.ndim])
        tolerance = max(1e-12, 1e-6 * max(source_variance, 0.0))
        off_diagonal = autocovariance.reshape(-1)[1:]
        if not np.any(np.abs(off_diagonal) > tolerance):
            variance_model.white_exact_source_variance = source_variance
        variance_model.white_exact_source_checked = True

    source_variance = variance_model.white_exact_source_variance
    if source_variance is None:
        return None

    source_model = variance_model.white_exact_source_model
    shape = tuple(int(size) for size in volume_shape)
    if source_model is None or source_model.variance_volume.shape != shape:
        source_model = _apply_poisson_cache_limits(
            PoissonVarianceModel(
                global_sigma=float(np.sqrt(max(source_variance, 0.0))),
                variance_volume=np.full(
                    shape,
                    max(source_variance, 0.0),
                    dtype=np.float32,
                ),
                variance_floor=0.0,
                source_name="white_gaussian",
                constant_variance=max(source_variance, 0.0),
            ),
            variance_model.poisson_cache_limits,
        )
        variance_model.white_exact_source_model = source_model
    return build_exact_poisson_group_plan(
        source_model,
        selected_abs_positions,
        selected_shifted_positions,
        group_shape,
        transforms,
    )


def get_entry_transform_matrix(
    variance_model: PoissonVarianceModel,
    group_shape: tuple[int, ...],
    transforms: list[NDArray[np.floating] | Callable | None],
) -> NDArray[np.float32]:
    _require_matrix_transforms(transforms)
    group_matrix = _get_group_transform_matrix(group_shape, transforms)
    spatial_matrix = _get_cached_spatial_transform_matrix(
        variance_model,
        tuple(int(size) for size in group_shape[1:]),
        list(transforms[1:]),
    )
    return np.ascontiguousarray(
        np.kron(group_matrix, spatial_matrix).astype(np.float32, copy=False)
    )


def get_exact_poisson_risk_metrics(
    variance_model: PoissonVarianceModel,
    group_shape: tuple[int, ...],
    inverse_transforms: list[NDArray[np.floating] | Callable | None],
    w_patch: NDArray[np.floating],
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    _require_matrix_transforms(inverse_transforms)

    block_shape = tuple(int(size) for size in group_shape[1:])
    window = np.asarray(w_patch, dtype=np.float32)
    cache_key = (
        tuple(int(size) for size in group_shape),
        _matrix_transform_signature(list(inverse_transforms)),
        window.shape,
        window.tobytes(),
    )
    cached = variance_model.exact_risk_metric_cache.get(cache_key)
    if cached is not None:
        return cached

    group_transform = inverse_transforms[0]
    group_matrix = (
        np.eye(int(group_shape[0]), dtype=np.float32)
        if group_transform is None
        else np.asarray(group_transform, dtype=np.float32)
    )
    spatial_matrix = _full_separable_transform_matrix_for_axes(
        block_shape,
        list(inverse_transforms[1:]),
    )
    window_sq = (window**2).reshape(-1)

    group_metric = np.asarray(group_matrix.T @ group_matrix, dtype=np.float32)
    weighted_spatial = window_sq[:, None] * spatial_matrix
    spatial_metric = np.asarray(
        spatial_matrix.T @ weighted_spatial,
        dtype=np.float32,
    )
    coefficient_metric_diag = np.kron(
        np.diag(group_metric),
        np.diag(spatial_metric),
    ).astype(np.float32, copy=False)

    result = (
        np.ascontiguousarray(group_metric),
        np.ascontiguousarray(spatial_metric),
        np.ascontiguousarray(coefficient_metric_diag),
    )
    variance_model.exact_risk_metric_cache[cache_key] = result
    return result


def limit_poisson_group_variance_planes(
    exact_noise_std: NDArray[np.floating],
    approximate_noise_std: NDArray[np.floating],
    k: int,
) -> NDArray[np.float32]:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or int(k) < 0:
        raise ValueError(f"k must be a non-negative integer, got {k!r}.")
    exact_variance = np.maximum(np.asarray(exact_noise_std, dtype=np.float32), 0.0) ** 2
    approximate_variance = (
        np.maximum(np.asarray(approximate_noise_std, dtype=np.float32), 0.0) ** 2
    )
    if (
        exact_variance.ndim == 0
        or exact_variance.ndim != approximate_variance.ndim
        or exact_variance.shape[1:] != approximate_variance.shape[1:]
    ):
        raise ValueError(
            "Exact and approximate Poisson noise arrays must have matching spatial shapes."
        )

    group_size = int(approximate_variance.shape[0])
    exact_plane_count = min(int(k), group_size)
    if exact_variance.shape[0] < exact_plane_count:
        raise ValueError(
            "Exact Poisson noise array does not contain the requested exact planes: "
            f"got {exact_variance.shape[0]}, need {exact_plane_count}."
        )
    if exact_plane_count == 0:
        return np.sqrt(approximate_variance).astype(np.float32, copy=False)
    if exact_plane_count == group_size:
        return np.sqrt(exact_variance).astype(np.float32, copy=False)

    variance = approximate_variance.copy()
    variance[:exact_plane_count] = exact_variance[:exact_plane_count]
    target_remaining = np.maximum(
        np.sum(approximate_variance, axis=0, dtype=np.float32)
        - np.sum(exact_variance[:exact_plane_count], axis=0, dtype=np.float32),
        0.0,
    )
    approximate_remaining = np.sum(
        approximate_variance[exact_plane_count:],
        axis=0,
        dtype=np.float32,
    )
    scale = np.divide(
        target_remaining,
        approximate_remaining,
        out=np.zeros_like(target_remaining),
        where=approximate_remaining > 0.0,
    )
    variance[exact_plane_count:] *= scale
    zero_approximation = approximate_remaining <= 0.0
    if np.any(zero_approximation):
        fallback = target_remaining / float(group_size - exact_plane_count)
        variance[exact_plane_count:] = np.where(
            zero_approximation,
            fallback,
            variance[exact_plane_count:],
        )
    return np.sqrt(np.maximum(variance, 0.0)).astype(np.float32, copy=False)


def get_direct_poisson_noise_std(
    group_variance: NDArray[np.floating] | None,
    forward_transform: list[NDArray[np.floating] | Callable | None],
    *,
    k: int = 0,
    variance_model: PoissonVarianceModel | None = None,
    selected_abs_positions: NDArray[np.int64] | None = None,
    selected_shifted_positions: NDArray[np.int64] | None = None,
    group_shape: tuple[int, ...] | None = None,
    exact_covariance_factors: ExactPoissonCovarianceFactors | None = None,
) -> NDArray[np.float32]:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or int(k) < 0:
        raise ValueError(f"k must be a non-negative integer, got {k!r}.")
    if int(k) == 0:
        if group_variance is None:
            raise ValueError("Approximate Poisson covariance requires group_variance.")
        variance = np.maximum(np.asarray(group_variance, dtype=np.float32), 0.0)
        for axis, transform in enumerate(forward_transform):
            variance = _apply_squared_transform_along_axis(variance, transform, axis)
        return np.sqrt(np.maximum(variance, 0.0)).astype(np.float32, copy=False)

    if variance_model is None:
        raise ValueError(
            "Exact overlap-aware Poisson covariance requires a PoissonVarianceModel."
        )
    if selected_abs_positions is None:
        raise ValueError(
            "Exact overlap-aware Poisson covariance requires selected_abs_positions."
        )
    if group_shape is None:
        raise ValueError("Exact overlap-aware Poisson covariance requires group_shape.")

    selected_abs_positions = np.asarray(selected_abs_positions, dtype=np.int64)
    if selected_shifted_positions is None:
        group_origin = np.min(selected_abs_positions, axis=0, keepdims=True)
        selected_shifted_positions = selected_abs_positions - group_origin
    else:
        selected_shifted_positions = np.asarray(
            selected_shifted_positions,
            dtype=np.int64,
        )
    group_shape = tuple(int(size) for size in group_shape)
    group_size = int(group_shape[0])
    exact_plane_count = min(int(k), group_size)
    if exact_covariance_factors is None:
        exact_covariance_factors = get_exact_poisson_covariance_factors(
            variance_model,
            selected_abs_positions,
            selected_shifted_positions,
            group_shape,
            forward_transform,
        )
    compact_entry_transform_t = _get_cached_compact_entry_transform_t(
        variance_model,
        group_shape,
        forward_transform,
        exact_plane_count,
    )
    exact_noise_std = get_exact_direct_poisson_noise_std_from_factors(
        group_shape,
        exact_covariance_factors,
        exact_plane_count=exact_plane_count,
        compact_entry_transform_t=compact_entry_transform_t,
    )
    if exact_plane_count == group_size:
        return exact_noise_std
    if group_variance is None:
        raise ValueError("Partial exact Poisson covariance requires group_variance.")
    approximate_noise_std = get_direct_poisson_noise_std(
        group_variance,
        forward_transform,
        k=0,
    )
    return limit_poisson_group_variance_planes(
        exact_noise_std,
        approximate_noise_std,
        k,
    )


def _build_patch_covariance(
    block_shape: tuple[int, ...],
    autocovariance: NDArray[np.float32],
) -> NDArray[np.float32]:
    coords = np.stack(
        np.meshgrid(
            *[np.arange(size, dtype=np.int64) for size in block_shape], indexing="ij"
        ),
        axis=-1,
    ).reshape(-1, len(block_shape))
    n = coords.shape[0]
    covariance = np.empty((n, n), dtype=np.float32)
    ac_shape = np.array(autocovariance.shape, dtype=np.int64)

    for i in range(n):
        for j in range(n):
            lag = (coords[i] - coords[j]) % ac_shape
            covariance[i, j] = autocovariance[tuple(int(x) for x in lag)]

    return covariance


def _full_separable_transform_matrix(
    axis_sizes: tuple[int, ...],
    transforms: list[NDArray[np.floating] | Callable | None],
) -> NDArray[np.float32]:
    _require_matrix_transforms(transforms)
    return _full_separable_transform_matrix_for_axes(axis_sizes, transforms)


def _get_cached_exact_patch_variance(
    variance_model: VarianceModel,
    block_shape: tuple[int, ...],
    spatial_forward_transform: list[NDArray[np.floating] | Callable | None],
) -> NDArray[np.float32]:
    cache_key = (block_shape, _matrix_transform_signature(spatial_forward_transform))
    cached = variance_model.transformed_patch_variance_cache.get(cache_key)
    if cached is not None:
        return cached

    if block_shape not in variance_model.patch_covariance_cache:
        variance_model.patch_covariance_cache[block_shape] = _build_patch_covariance(
            block_shape,
            variance_model.autocovariance,
        )

    patch_covariance = variance_model.patch_covariance_cache[block_shape]
    spatial_transform = _full_separable_transform_matrix(
        block_shape,
        spatial_forward_transform,
    )
    transformed_covariance = spatial_transform @ patch_covariance @ spatial_transform.T
    patch_variance = np.clip(np.diag(transformed_covariance), 0.0, None).astype(
        np.float32,
        copy=False,
    ).reshape(block_shape)
    variance_model.transformed_patch_variance_cache[cache_key] = patch_variance
    return patch_variance


def get_exact_noise_std(
    group_shape: tuple[int, ...],
    forward_transform: list[NDArray[np.floating] | Callable | None],
    variance_model: VarianceModel,
) -> NDArray[np.float32]:
    block_shape = tuple(int(size) for size in group_shape[1:])
    patch_variance = _get_cached_exact_patch_variance(
        variance_model,
        block_shape,
        list(forward_transform[1:]),
    )

    variance = np.broadcast_to(patch_variance, group_shape).copy()
    variance = _apply_squared_transform_along_axis(variance, forward_transform[0], 0)
    return np.sqrt(np.maximum(variance, 0.0)).astype(np.float32, copy=False)


def _get_cached_transformed_patch_cross_covariance(
    variance_model: VarianceModel,
    block_shape: tuple[int, ...],
    spatial_forward_transform: list[NDArray[np.floating] | Callable | None],
) -> NDArray[np.float32]:
    identity_cache_key = (
        block_shape,
        ("identity",) + _matrix_transform_signature(spatial_forward_transform),
    )
    cached = variance_model.transformed_patch_cross_covariance_cache.get(
        identity_cache_key
    )
    if cached is not None:
        return cached

    content_cache_key = (
        block_shape,
        ("content",) + _matrix_content_transform_signature(spatial_forward_transform),
    )
    cached = variance_model.transformed_patch_cross_covariance_cache.get(content_cache_key)
    if cached is not None:
        variance_model.transformed_patch_cross_covariance_cache[
            identity_cache_key
        ] = cached
        return cached

    spatial_transform = _full_separable_transform_matrix(
        block_shape,
        spatial_forward_transform,
    )
    psd = np.asarray(variance_model.processed_sigma_psd, dtype=np.float32)
    domain_shape = variance_model.variance_domain_shape
    if len(domain_shape) != len(block_shape):
        raise ValueError(
            "Processed PSD rank does not match block rank: "
            f"got {domain_shape} and {block_shape}."
        )
    if psd.shape != domain_shape:
        source_shape = tuple(int(size) for size in psd.shape)
        sample_axes = [
            np.arange(target_size, dtype=np.float64)
            * (source_size / float(target_size))
            for source_size, target_size in zip(source_shape, domain_shape, strict=False)
        ]
        sample_coords = np.stack(
            np.meshgrid(*sample_axes, indexing="ij"),
            axis=0,
        )
        psd = map_coordinates(
            psd,
            sample_coords,
            order=1,
            mode="grid-wrap",
        ).astype(np.float32, copy=False)
        psd *= np.float32(np.prod(domain_shape) / float(np.prod(source_shape)))
        psd = np.maximum(psd, 0.0)

    patch_volume = int(np.prod(block_shape))
    covariance_maps = np.empty((patch_volume,) + domain_shape, dtype=np.float32)
    local_coords = np.stack(
        np.meshgrid(
            *[np.arange(size, dtype=np.int64) for size in block_shape],
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, len(block_shape))

    for coefficient_index in range(patch_volume):
        folded_basis = np.zeros(domain_shape, dtype=np.float32)
        basis = spatial_transform[coefficient_index]
        for local_index, coord in enumerate(local_coords):
            folded_index = tuple(
                int(coord[axis] % domain_shape[axis])
                for axis in range(len(domain_shape))
            )
            folded_basis[folded_index] += basis[local_index]

        basis_spectrum = np.fft.fftn(folded_basis)
        covariance_map = np.real(
            np.fft.ifftn(psd * (np.abs(basis_spectrum) ** 2)) / float(psd.size)
        ).astype(np.float32, copy=False)
        reverse_indices = tuple(
            (-np.arange(size, dtype=np.int64)) % size for size in domain_shape
        )
        covariance_maps[coefficient_index] = 0.5 * (
            covariance_map + covariance_map[np.ix_(*reverse_indices)]
        )

    variance_model.transformed_patch_cross_covariance_cache[
        content_cache_key
    ] = covariance_maps
    variance_model.transformed_patch_cross_covariance_cache[
        identity_cache_key
    ] = covariance_maps
    return covariance_maps


@njit(cache=True)
def _build_pair_lag_indices_numba(
    positions: NDArray[np.int64],
    domain_shape: NDArray[np.int64],
) -> NDArray[np.int64]:
    group_size = positions.shape[0]
    ndim = positions.shape[1]
    pair_lag_indices = np.empty((group_size, group_size), dtype=np.int64)

    for first in range(group_size):
        for second in range(group_size):
            flat_index = np.int64(0)
            for axis in range(ndim):
                lag = (
                    positions[first, axis] - positions[second, axis]
                ) % domain_shape[axis]
                flat_index = flat_index * domain_shape[axis] + lag
            pair_lag_indices[first, second] = flat_index
    return pair_lag_indices


# Cluster HT/Wiener timings motivate a cutoff between 22k and 85k covariance terms.
_GAUSSIAN_VARIANCE_PARALLEL_MIN_TERMS = 32768


@njit(cache=True, inline="always")
def _accumulate_gaussian_coefficient_variance_numba(
    covariance_maps_flat: NDArray[np.float32],
    pair_lag_indices: NDArray[np.int64],
    exact_rows: NDArray[np.float32],
    nonzero_indices: NDArray[np.int64],
    nonzero_counts: NDArray[np.int64],
    exact_variance: NDArray[np.float32],
    coefficient_index: int,
) -> None:
    covariance_map = covariance_maps_flat[coefficient_index]
    for plane_index in range(exact_rows.shape[0]):
        group_row = exact_rows[plane_index]
        nonzero_count = nonzero_counts[plane_index]
        value = np.float32(0.0)
        for first_offset in range(nonzero_count):
            first = nonzero_indices[plane_index, first_offset]
            first_weight = group_row[first]
            value += (
                first_weight
                * covariance_map[pair_lag_indices[first, first]]
                * first_weight
            )
            for second_offset in range(first_offset):
                second = nonzero_indices[plane_index, second_offset]
                second_weight = group_row[second]
                covariance_pair = (
                    covariance_map[pair_lag_indices[first, second]]
                    + covariance_map[pair_lag_indices[second, first]]
                )
                value += first_weight * second_weight * covariance_pair
        exact_variance[plane_index, coefficient_index] = max(value, np.float32(0.0))


@njit(cache=True, parallel=True)
def _accumulate_gaussian_exact_plane_variance_parallel_numba(
    covariance_maps_flat: NDArray[np.float32],
    pair_lag_indices: NDArray[np.int64],
    exact_rows: NDArray[np.float32],
    nonzero_indices: NDArray[np.int64],
    nonzero_counts: NDArray[np.int64],
) -> NDArray[np.float32]:
    exact_plane_count = exact_rows.shape[0]
    patch_volume = covariance_maps_flat.shape[0]
    exact_variance = np.empty((exact_plane_count, patch_volume), dtype=np.float32)

    # Coefficients are independent; keep each covariance reduction in serial order.
    for coefficient_index in prange(patch_volume):
        _accumulate_gaussian_coefficient_variance_numba(
            covariance_maps_flat,
            pair_lag_indices,
            exact_rows,
            nonzero_indices,
            nonzero_counts,
            exact_variance,
            coefficient_index,
        )
    return exact_variance


@njit(cache=True)
def _accumulate_gaussian_exact_plane_variance_numba(
    covariance_maps_flat: NDArray[np.float32],
    pair_lag_indices: NDArray[np.int64],
    exact_rows: NDArray[np.float32],
    nonzero_indices: NDArray[np.int64],
    nonzero_counts: NDArray[np.int64],
) -> NDArray[np.float32]:
    patch_volume = covariance_maps_flat.shape[0]
    terms_per_coefficient = 0
    for count in nonzero_counts:
        terms_per_coefficient += count * (count + 1) // 2
    if patch_volume * terms_per_coefficient >= _GAUSSIAN_VARIANCE_PARALLEL_MIN_TERMS:
        return _accumulate_gaussian_exact_plane_variance_parallel_numba(
            covariance_maps_flat,
            pair_lag_indices,
            exact_rows,
            nonzero_indices,
            nonzero_counts,
        )

    exact_variance = np.empty((exact_rows.shape[0], patch_volume), dtype=np.float32)
    for coefficient_index in range(patch_volume):
        _accumulate_gaussian_coefficient_variance_numba(
            covariance_maps_flat,
            pair_lag_indices,
            exact_rows,
            nonzero_indices,
            nonzero_counts,
            exact_variance,
            coefficient_index,
        )
    return exact_variance


def _get_cached_gaussian_exact_row_structure(
    variance_model: VarianceModel,
    group_matrix: NDArray[np.float32],
    transform_identity: int,
    exact_plane_count: int,
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.int64]]:
    cache_key = (transform_identity, group_matrix.shape[0], exact_plane_count)
    cached = variance_model.gaussian_exact_row_structure_cache.get(cache_key)
    if cached is not None:
        return cached

    exact_rows = np.ascontiguousarray(group_matrix[:exact_plane_count])
    nonzero_counts = np.count_nonzero(exact_rows, axis=1).astype(np.int64)
    max_nonzero_count = int(np.max(nonzero_counts))
    nonzero_indices = np.full(
        (exact_plane_count, max_nonzero_count),
        -1,
        dtype=np.int64,
    )
    for plane_index in range(exact_plane_count):
        indices = np.flatnonzero(exact_rows[plane_index]).astype(np.int64, copy=False)
        nonzero_indices[plane_index, : indices.size] = indices

    result = (exact_rows, nonzero_indices, nonzero_counts)
    variance_model.gaussian_exact_row_structure_cache[cache_key] = result
    return result


def get_stationary_gaussian_group_noise_variance(
    group_shape: tuple[int, ...],
    forward_transform: list[NDArray[np.floating] | Callable | None],
    variance_model: VarianceModel,
    selected_positions: NDArray[np.integer],
    k: int,
) -> NDArray[np.float32]:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or int(k) < 0:
        raise ValueError(f"k must be a non-negative integer, got {k!r}.")

    group_size = int(group_shape[0])
    block_shape = tuple(int(size) for size in group_shape[1:])
    if group_size <= 0:
        return np.empty(group_shape, dtype=np.float32)

    covariance_maps = _get_cached_transformed_patch_cross_covariance(
        variance_model,
        block_shape,
        list(forward_transform[1:]),
    )
    patch_variance = covariance_maps[(slice(None),) + (0,) * len(block_shape)]
    exact_plane_count = min(int(k), group_size)
    if exact_plane_count == 0:
        return np.broadcast_to(patch_variance.reshape(block_shape), group_shape).copy()

    positions = np.asarray(selected_positions, dtype=np.int64)
    expected_position_shape = (group_size, len(block_shape))
    if positions.shape != expected_position_shape:
        raise ValueError(
            "selected_positions shape does not match the group geometry: "
            f"got {positions.shape}, expected {expected_position_shape}."
        )
    group_transform = forward_transform[0]
    if group_transform is None or callable(group_transform):
        group_matrix = np.eye(group_size, dtype=np.float32)
    else:
        group_matrix = np.asarray(group_transform, dtype=np.float32)
    if group_matrix.shape != (group_size, group_size):
        raise ValueError(
            f"Group transform must have shape {(group_size, group_size)}, "
            f"got {group_matrix.shape}."
        )

    domain_shape = np.asarray(covariance_maps.shape[1:], dtype=np.int64)
    pair_lag_indices = _build_pair_lag_indices_numba(positions, domain_shape)
    covariance_maps_flat = covariance_maps.reshape(patch_variance.size, -1)
    exact_rows, nonzero_indices, nonzero_counts = (
        _get_cached_gaussian_exact_row_structure(
            variance_model,
            group_matrix,
            id(group_transform),
            exact_plane_count,
        )
    )
    exact_variance = _accumulate_gaussian_exact_plane_variance_numba(
        covariance_maps_flat,
        pair_lag_indices,
        exact_rows,
        nonzero_indices,
        nonzero_counts,
    )

    variance = np.empty((group_size, patch_variance.size), dtype=np.float32)
    variance[:exact_plane_count] = exact_variance
    if exact_plane_count < group_size:
        remaining = (
            group_size * patch_variance
            - np.sum(exact_variance, axis=0, dtype=np.float32)
        ) / float(group_size - exact_plane_count)
        variance[exact_plane_count:] = np.maximum(remaining, 0.0)
    return variance.reshape(group_shape)
