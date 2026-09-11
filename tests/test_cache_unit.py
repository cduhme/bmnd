from types import SimpleNamespace

import numpy as np
import pytest

import bmnd.cache as cache_module
from bmnd.cache import get_poisson_cache_limits
from bmnd.profiles import BMNDProfile


def _make_profile(noise_model: str) -> BMNDProfile:
    return BMNDProfile(
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


def test_auto_cache_budget_reserves_noncache_memory(monkeypatch) -> None:
    profile = _make_profile("poisson")
    volume = np.ones((8, 8), dtype=np.float32)
    resolved = SimpleNamespace(
        is_poisson=True,
        effective_nf=None,
        variance_k=profile.k,
    )
    monkeypatch.setattr(
        cache_module,
        "_effective_available_memory_bytes",
        lambda: 2 * 1024**3,
    )
    monkeypatch.setattr(
        cache_module,
        "_estimate_noncache_peak_bytes",
        lambda *args, **kwargs: 256 * 1024**2,
    )
    monkeypatch.setattr(cache_module, "get_num_threads", lambda: 8)

    budget = cache_module._resolve_cache_budget_bytes(
        "auto",
        volume,
        profile,
        resolved,
        return_sigma_map=False,
        has_stage_callback=False,
    )

    assert budget == 1280 * 1024**2


@pytest.mark.parametrize("budget", [0, 1, 5, 79, 1024, 640 * 1024**2])
def test_poisson_cache_limits_preserve_aggregate_budgets(budget: int) -> None:
    limits = get_poisson_cache_limits(budget)

    assert (
        limits.overlap_max_bytes
        + limits.contribution_max_bytes
        + limits.factorized_max_bytes
        == budget
    )
    assert 0 <= limits.factorized_entry_max_bytes <= limits.factorized_max_bytes
    for byte_limit, entry_limit in (
        (limits.overlap_max_bytes, limits.overlap_max_entries),
        (limits.contribution_max_bytes, limits.contribution_max_entries),
        (limits.factorized_max_bytes, limits.factorized_max_entries),
    ):
        assert byte_limit >= 0
        assert entry_limit == 0 if byte_limit == 0 else entry_limit > 0
