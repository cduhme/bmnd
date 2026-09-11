from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on Windows.
    resource = None  # type: ignore[assignment]

import numpy as np
from numba import get_num_threads
from numpy.typing import NDArray

from .enums import (
    AggregationWeightDomain,
    AggregationWeightModel,
    AggregationWeightScope,
    CacheMode,
    PoissonVarianceSource,
    coerce_enum,
)
from .profiles import BMNDProfile

CacheBudget = int | CacheMode | None

_MIB = 1024 * 1024
_AUTO_CACHE_MINIMUM_HEADROOM_BYTES = 512 * _MIB


class _ResolvedProfileLike(Protocol):
    is_poisson: bool
    effective_nf: tuple[int, ...] | None
    variance_k: int


@dataclass(frozen=True)
class PoissonCacheLimits:
    """Limits for retained overlap-aware covariance caches."""

    overlap_max_bytes: int
    overlap_max_entries: int
    contribution_max_bytes: int
    contribution_max_entries: int
    factorized_max_bytes: int
    factorized_max_entries: int
    factorized_entry_max_bytes: int


DEFAULT_POISSON_CACHE_LIMITS = PoissonCacheLimits(
    overlap_max_bytes=256 * _MIB,
    overlap_max_entries=2048,
    contribution_max_bytes=128 * _MIB,
    contribution_max_entries=64,
    factorized_max_bytes=256 * _MIB,
    factorized_max_entries=64,
    factorized_entry_max_bytes=64 * _MIB,
)


def _validate_max_cache_bytes(max_cache_bytes: CacheBudget) -> CacheBudget:
    if isinstance(max_cache_bytes, str):
        max_cache_bytes = coerce_enum(
            max_cache_bytes,
            CacheMode,
            "max_cache_bytes mode",
        )
    if max_cache_bytes is None or max_cache_bytes is CacheMode.AUTO:
        return max_cache_bytes
    if (
        isinstance(max_cache_bytes, bool)
        or not isinstance(max_cache_bytes, (int, np.integer))
        or int(max_cache_bytes) < 0
    ):
        raise ValueError(
            "max_cache_bytes must be a non-negative integer, 'auto', or None, "
            f"got {max_cache_bytes!r}."
        )
    return int(max_cache_bytes)


def _read_memory_integer(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _current_process_memory_bytes() -> tuple[int, int]:
    try:
        fields = Path("/proc/self/statm").read_text(encoding="utf-8").split()
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return int(fields[1]) * page_size, int(fields[0]) * page_size
    except (OSError, ValueError, IndexError):
        return 0, 0


def _linux_mem_available_bytes() -> int | None:
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2:
                return int(fields[1]) * 1024
    return None


def _cgroup_memory_headroom_bytes() -> int | None:
    roots = [Path("/sys/fs/cgroup")]
    try:
        cgroup_lines = (
            Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
        )
    except OSError:
        cgroup_lines = []
    for line in cgroup_lines:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        controllers, relative = fields[1], fields[2].lstrip("/")
        if controllers == "":
            mount_root = Path("/sys/fs/cgroup")
        elif "memory" in controllers.split(","):
            mount_root = Path("/sys/fs/cgroup/memory")
        else:
            continue
        current = mount_root / relative
        while current == mount_root or mount_root in current.parents:
            roots.append(current)
            if current == mount_root:
                break
            current = current.parent

    headrooms: list[int] = []
    for root in dict.fromkeys(roots):
        limit = _read_memory_integer(root / "memory.max")
        current = _read_memory_integer(root / "memory.current")
        if limit is None:
            limit = _read_memory_integer(root / "memory.limit_in_bytes")
            current = _read_memory_integer(root / "memory.usage_in_bytes")
        if limit is not None and current is not None and limit < (1 << 62):
            headrooms.append(max(limit - current, 0))
    return min(headrooms) if headrooms else None


def _slurm_memory_headroom_bytes(current_rss: int) -> int | None:
    def parse_mebibytes(value: str | None) -> float | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        multiplier = 1.0
        if normalized.endswith("G"):
            normalized = normalized[:-1]
            multiplier = 1024.0
        elif normalized.endswith("M"):
            normalized = normalized[:-1]
        elif normalized.endswith("K"):
            normalized = normalized[:-1]
            multiplier = 1.0 / 1024.0
        try:
            return float(normalized) * multiplier
        except ValueError:
            return None

    def local_task_count() -> int:
        configured = os.environ.get("SLURM_NTASKS_PER_NODE")
        if configured is not None:
            try:
                return max(int(configured), 1)
            except ValueError:
                pass
        distribution = os.environ.get("SLURM_TASKS_PER_NODE")
        if distribution:
            counts = [int(value) for value in re.findall(r"\d+", distribution)]
            if counts:
                return max(counts)
        try:
            return max(int(os.environ.get("SLURM_NTASKS", "1")), 1)
        except ValueError:
            return 1

    allocated_mib = parse_mebibytes(os.environ.get("SLURM_MEM_PER_NODE"))
    if allocated_mib is not None:
        allocated_mib /= local_task_count()
    if allocated_mib is None:
        per_cpu_mib = parse_mebibytes(os.environ.get("SLURM_MEM_PER_CPU"))
        cpu_count = os.environ.get("SLURM_CPUS_PER_TASK", "1")
        if per_cpu_mib is not None and cpu_count is not None:
            try:
                allocated_mib = per_cpu_mib * int(cpu_count)
            except ValueError:
                allocated_mib = None
    if allocated_mib is None or allocated_mib <= 0.0:
        return None
    return max(int(allocated_mib * _MIB) - current_rss, 0)


def _effective_available_memory_bytes() -> int:
    current_rss, current_virtual = _current_process_memory_bytes()
    candidates = [
        _linux_mem_available_bytes(),
        _cgroup_memory_headroom_bytes(),
        _slurm_memory_headroom_bytes(current_rss),
    ]
    if resource is not None:
        try:
            address_limit, _ = resource.getrlimit(resource.RLIMIT_AS)
        except (AttributeError, OSError, ValueError):
            address_limit = resource.RLIM_INFINITY
        if address_limit != resource.RLIM_INFINITY and current_virtual > 0:
            candidates.append(max(int(address_limit) - current_virtual, 0))
    available = [int(value) for value in candidates if value is not None]
    if not available:
        try:
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, OSError, ValueError):
            available_pages = 0
            page_size = 0
        if available_pages > 0 and page_size > 0:
            available.append(available_pages * page_size)
    return min(available) if available else 0


def _estimate_noncache_peak_bytes(
    volume: NDArray[np.floating],
    profile: BMNDProfile,
    resolved_profile: _ResolvedProfileLike,
    *,
    return_sigma_map: bool,
    has_stage_callback: bool,
) -> int:
    shape = tuple(int(size) for size in volume.shape)
    ndim = len(shape)
    voxel_count = int(np.prod(shape, dtype=np.int64))

    def stage_geometry(stage: str) -> tuple[int, int, int]:
        block_shape = tuple(
            int(size) for size in getattr(profile, f"{stage}_block_size")
        )
        patch_counts = tuple(
            shape[axis] - block_shape[axis] + 1 for axis in range(ndim)
        )
        patch_count = int(np.prod(patch_counts, dtype=np.int64))
        patch_volume = int(np.prod(block_shape, dtype=np.int64))
        max_group = int(getattr(profile, f"{stage}_max_stack_size"))
        return patch_count, patch_volume, max_group

    def group_sizes(min_group: int, max_group: int) -> list[int]:
        sizes: list[int] = []
        size = 1
        while size <= max_group:
            sizes.append(size)
            size *= 2
        if min_group not in sizes:
            sizes.append(min_group)
        return sorted(sizes)

    ht_count, ht_volume, ht_group = stage_geometry("ht")
    wiener_count, wiener_volume, wiener_group = stage_geometry("wiener")
    same_patch_geometry = profile.ht_block_size == profile.wiener_block_size

    volume_peak = (58 if return_sigma_map else 33) * voxel_count
    image_patch_bytes = 4 * (
        ht_count * ht_volume
        + wiener_count * wiener_volume
        + (0 if same_patch_geometry else wiener_count * wiener_volume)
    )
    blockmatch_bytes = 0
    for patch_count, max_group in (
        (ht_count, ht_group),
        (wiener_count, wiener_group),
    ):
        blockmatch_bytes += (8 + 16 * ndim) * patch_count * max_group
        blockmatch_bytes += patch_count * (1024 + 128 * ndim)

    if resolved_profile.is_poisson:
        variance_patch_bytes = 4 * (ht_count * ht_volume + wiener_count * wiener_volume)
        variance_model_bytes = (
            8
            if profile.poisson_variance_source_wiener is PoissonVarianceSource.PILOT
            else 4
        ) * voxel_count
        model_construction_bytes = 21 * voxel_count
        group_workspace_bytes = (
            128 * ht_group * ht_volume + 160 * wiener_group * wiener_volume
        )
        overlap_build_bytes = max(
            32 * ht_group * ht_volume + 8 * ht_group * ht_volume * (ndim + 3),
            32 * wiener_group * wiener_volume
            + 8 * wiener_group * wiener_volume * (ndim + 3),
        )
        unmanaged_cache_bytes = 0
        if resolved_profile.variance_k > 0:
            for stage, patch_volume, max_group in (
                ("ht", ht_volume, ht_group),
                ("wiener", wiener_volume, wiener_group),
            ):
                unmanaged_cache_bytes += 4 * patch_volume * patch_volume
                unmanaged_cache_bytes += 8 * patch_volume * ndim
                min_group = int(getattr(profile, f"{stage}_min_stack_size"))
                for group_size in group_sizes(min_group, max_group):
                    plane_count = min(resolved_profile.variance_k, group_size)
                    unmanaged_cache_bytes += (
                        4 * group_size * plane_count * patch_volume * patch_volume
                    )
        exact_risk_bytes = 0
        for stage, patch_volume, max_group in (
            ("ht", ht_volume, ht_group),
            ("wiener", wiener_volume, wiener_group),
        ):
            exact_group_risk = (
            getattr(profile, f"{stage}_weight_model")
            in {AggregationWeightModel.VARIANCE, AggregationWeightModel.RISK}
            and getattr(profile, f"{stage}_weight_domain")
            is AggregationWeightDomain.WINDOWED_SYNTHESIS
            and getattr(profile, f"{stage}_weight_scope")
            is AggregationWeightScope.GROUP
                and resolved_profile.variance_k > 0
            )
            if exact_group_risk:
                plane_count = min(resolved_profile.variance_k, max_group)
                union_voxels = max_group * patch_volume
                exact_risk_bytes = max(
                    exact_risk_bytes,
                    4 * union_voxels * plane_count * patch_volume
                    + 64 * _MIB
                    + 8 * union_voxels,
                )
        noise_model_bytes = (
            variance_patch_bytes
            + variance_model_bytes
            + model_construction_bytes
            + group_workspace_bytes
            + overlap_build_bytes
            + unmanaged_cache_bytes
            + exact_risk_bytes
        )
    else:
        gaussian_model_and_fft_bytes = 60 * voxel_count
        group_workspace_bytes = (
            96 * ht_group * ht_volume + 128 * wiener_group * wiener_volume
        )
        covariance_cache_bytes = 0
        if resolved_profile.variance_k > 0:
            variance_domain_shape = resolved_profile.effective_nf or shape
            variance_domain_size = int(np.prod(variance_domain_shape, dtype=np.int64))
            covariance_cache_bytes = (
                4 * (ht_volume + wiener_volume) * variance_domain_size
            )
        noise_model_bytes = (
            gaussian_model_and_fft_bytes
            + group_workspace_bytes
            + covariance_cache_bytes
        )

    callback_bytes = 8 * voxel_count if has_stage_callback else 0
    return int(
        volume_peak
        + image_patch_bytes
        + blockmatch_bytes
        + noise_model_bytes
        + callback_bytes
    )


def _resolve_cache_budget_bytes(
    request: CacheBudget,
    volume: NDArray[np.floating],
    profile: BMNDProfile,
    resolved_profile: _ResolvedProfileLike,
    *,
    return_sigma_map: bool,
    has_stage_callback: bool,
) -> int | None:
    if request is None or isinstance(request, int):
        return request
    available_bytes = _effective_available_memory_bytes()
    noncache_bytes = _estimate_noncache_peak_bytes(
        volume,
        profile,
        resolved_profile,
        return_sigma_map=return_sigma_map,
        has_stage_callback=has_stage_callback,
    )
    uncertainty_bytes = max(
        _AUTO_CACHE_MINIMUM_HEADROOM_BYTES,
        noncache_bytes // 4,
        16 * _MIB * get_num_threads(),
    )
    return max(available_bytes - noncache_bytes - uncertainty_bytes, 0)


def _uses_managed_poisson_caches(
    profile: BMNDProfile,
    resolved_profile: _ResolvedProfileLike,
) -> bool:
    if resolved_profile.variance_k <= 0:
        return False
    if resolved_profile.is_poisson:
        return True
    return any(
        getattr(profile, f"{stage}_weight_model")
        in (
            {AggregationWeightModel.VARIANCE, AggregationWeightModel.RISK}
            if stage == "wiener"
            else {AggregationWeightModel.VARIANCE}
        )
        and getattr(profile, f"{stage}_weight_domain")
        is AggregationWeightDomain.WINDOWED_SYNTHESIS
        and getattr(profile, f"{stage}_weight_scope") is AggregationWeightScope.GROUP
        for stage in ("ht", "wiener")
    )


def get_poisson_cache_limits(max_cache_bytes: int | None) -> PoissonCacheLimits:
    if max_cache_bytes is None:
        return PoissonCacheLimits(
            overlap_max_bytes=sys.maxsize,
            overlap_max_entries=sys.maxsize,
            contribution_max_bytes=sys.maxsize,
            contribution_max_entries=sys.maxsize,
            factorized_max_bytes=sys.maxsize,
            factorized_max_entries=sys.maxsize,
            factorized_entry_max_bytes=sys.maxsize,
        )

    total_bytes = int(max_cache_bytes)
    if total_bytes <= 0:
        return PoissonCacheLimits(0, 0, 0, 0, 0, 0, 0)

    defaults = DEFAULT_POISSON_CACHE_LIMITS
    baseline_total = (
        defaults.overlap_max_bytes
        + defaults.contribution_max_bytes
        + defaults.factorized_max_bytes
    )
    overlap_bytes = total_bytes * defaults.overlap_max_bytes // baseline_total
    contribution_bytes = total_bytes * defaults.contribution_max_bytes // baseline_total
    factorized_bytes = total_bytes - overlap_bytes - contribution_bytes

    def scaled_entries(base_entries: int, cache_bytes: int, base_bytes: int) -> int:
        if cache_bytes == 0:
            return 0
        return max(1, (base_entries * cache_bytes + base_bytes - 1) // base_bytes)

    factorized_entry_bytes = min(
        factorized_bytes,
        total_bytes * defaults.factorized_entry_max_bytes // baseline_total,
    )
    return PoissonCacheLimits(
        overlap_max_bytes=overlap_bytes,
        overlap_max_entries=scaled_entries(
            defaults.overlap_max_entries,
            overlap_bytes,
            defaults.overlap_max_bytes,
        ),
        contribution_max_bytes=contribution_bytes,
        contribution_max_entries=scaled_entries(
            defaults.contribution_max_entries,
            contribution_bytes,
            defaults.contribution_max_bytes,
        ),
        factorized_max_bytes=factorized_bytes,
        factorized_max_entries=scaled_entries(
            defaults.factorized_max_entries,
            factorized_bytes,
            defaults.factorized_max_bytes,
        ),
        factorized_entry_max_bytes=factorized_entry_bytes,
    )


def _resolve_cache_limits(
    request: CacheBudget,
    volume: NDArray[np.floating],
    profile: BMNDProfile,
    resolved_profile: _ResolvedProfileLike,
    *,
    return_sigma_map: bool,
    has_stage_callback: bool,
) -> tuple[PoissonCacheLimits, PoissonCacheLimits]:
    if request is CacheMode.AUTO and not _uses_managed_poisson_caches(
        profile,
        resolved_profile,
    ):
        cache_budget_bytes = 0
    else:
        cache_budget_bytes = _resolve_cache_budget_bytes(
            request,
            volume,
            profile,
            resolved_profile,
            return_sigma_map=return_sigma_map,
            has_stage_callback=has_stage_callback,
        )

    if cache_budget_bytes is None:
        ht_cache_budget = None
        wiener_cache_budget = None
    elif (
        resolved_profile.is_poisson
        and profile.poisson_variance_source_wiener is PoissonVarianceSource.PILOT
    ):
        ht_cache_budget = cache_budget_bytes // 2
        wiener_cache_budget = cache_budget_bytes - ht_cache_budget
    else:
        ht_cache_budget = cache_budget_bytes
        wiener_cache_budget = cache_budget_bytes
    return (
        get_poisson_cache_limits(ht_cache_budget),
        get_poisson_cache_limits(wiener_cache_budget),
    )
