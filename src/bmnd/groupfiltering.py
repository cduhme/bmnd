from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import local
from typing import Callable

from numba import get_num_threads, set_num_threads
import numpy as np
from numpy.typing import NDArray

from .aggregationweights import SynthesisRiskMetricFactors
from .blockmatching import BlockMatchGroup
from .poisson import _FrozenAggregationGroup
from .variance import PoissonVarianceModel


_PARALLEL_POISSON_GROUPS = True


@dataclass
class _GroupFilterState:
    poisson_model: PoissonVarianceModel | None
    transforms: dict[
        tuple[tuple[int, ...], str],
        tuple[list[NDArray | Callable | None], list[NDArray | Callable | None]],
    ] = field(default_factory=dict)
    constant_directions: dict[
        tuple[tuple[int, ...], str], NDArray[np.generic]
    ] = field(default_factory=dict)
    risk_metrics: dict[
        tuple[tuple[int, ...], str], SynthesisRiskMetricFactors
    ] = field(default_factory=dict)


@dataclass(frozen=True)
class _FilteredGroup:
    selected: NDArray[np.int64]
    patches: NDArray
    weights: NDArray[np.float32]
    sigma: NDArray | float | None
    frozen: _FrozenAggregationGroup | None


def _copy_group_filter_state(
    state: _GroupFilterState, cache_shares: int
) -> _GroupFilterState:
    source = state.poisson_model
    assert source is not None
    model = PoissonVarianceModel(
        global_sigma=source.global_sigma,
        variance_volume=source.variance_volume,
        variance_floor=source.variance_floor,
        source_name=source.source_name,
        constant_variance=source.constant_variance,
        overlap_cache_max_bytes=source.overlap_cache_max_bytes // cache_shares,
        overlap_cache_max_entries=source.overlap_cache_max_entries // cache_shares,
        contribution_cache_max_bytes=source.contribution_cache_max_bytes // cache_shares,
        contribution_cache_max_entries=(
            source.contribution_cache_max_entries // cache_shares
        ),
        factorized_cache_max_bytes=source.factorized_cache_max_bytes // cache_shares,
        factorized_cache_max_entries=source.factorized_cache_max_entries // cache_shares,
        factorized_entry_cache_max_bytes=source.factorized_entry_cache_max_bytes,
    )
    # Geometry arrays are read-only during filtering; LRU caches and plans stay private.
    model.patch_view_cache = source.patch_view_cache.copy()
    model.patch_local_coords_cache = source.patch_local_coords_cache.copy()
    model.spatial_transform_cache = source.spatial_transform_cache.copy()
    model.compact_entry_transform_t_cache = source.compact_entry_transform_t_cache.copy()
    model.exact_risk_metric_cache = source.exact_risk_metric_cache.copy()
    if cache_shares == 1:
        model.exact_factorized_contribution_cache = (
            source.exact_factorized_contribution_cache.copy()
        )
        model.covariance_cache_stats.factorized_bytes = sum(
            array.nbytes
            for entry in model.exact_factorized_contribution_cache.values()
            for array in entry
        )
    return _GroupFilterState(
        model,
        state.transforms.copy(),
        state.constant_directions.copy(),
        state.risk_metrics.copy(),
    )


def _iter_filtered_groups(
    groups: list[BlockMatchGroup],
    filter_group: Callable[[BlockMatchGroup, _GroupFilterState], _FilteredGroup | None],
    state: _GroupFilterState,
    *,
    parallel: bool,
) -> Iterator[_FilteredGroup]:
    workers = min(get_num_threads(), len(groups))
    if (
        not parallel
        or not _PARALLEL_POISSON_GROUPS
        or state.poisson_model is None
        or workers < 2
    ):
        for group in groups:
            result = filter_group(group, state)
            if result is not None:
                yield result
        return

    # Warm common transform geometry once and split retained-cache limits, including
    # this seed context. Each worker holds at most one live group's risk workspace.
    seed = _copy_group_filter_state(state, workers + 1)
    first_result = filter_group(groups[0], seed)
    if first_result is not None:
        yield first_result
    worker_state = local()

    def initialize_worker() -> None:
        set_num_threads(1)
        worker_state.value = _copy_group_filter_state(seed, 1)

    def filter_chunk(chunk: list[BlockMatchGroup]) -> list[_FilteredGroup]:
        results = []
        for group in chunk:
            result = filter_group(group, worker_state.value)
            if result is not None:
                results.append(result)
        return results

    chunk_size = 8
    batch_size = workers * chunk_size
    with ThreadPoolExecutor(max_workers=workers, initializer=initialize_worker) as pool:
        for start in range(1, len(groups), batch_size):
            batch = groups[start : start + batch_size]
            chunks = [batch[i : i + chunk_size] for i in range(0, len(batch), chunk_size)]
            # map yields chunks in submission order, never completion order.
            for results in pool.map(filter_chunk, chunks):
                yield from results
