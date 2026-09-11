from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import get_ident

from numba import get_num_threads, set_num_threads
import numpy as np
from numpy.typing import NDArray
import pytest

import bmnd.bmndalgo as algorithm
import bmnd.groupfiltering as groupfiltering
from bmnd.profiles import BMNDProfile
from bmnd.transforms import Transform, TransformMode, TransformType


@pytest.mark.parametrize(
    ("weighting", "mass", "cache_bytes", "k"),
    [
        ("default", "none", 0, 2),
        ("default", "both", 0, 2),
        ("risk", "none", 16 * 1024 * 1024, 2),
        ("risk", "both", 16 * 1024 * 1024, 2),
        ("risk", "both", 0, 2),
        ("variance", "none", 16 * 1024 * 1024, 2),
        ("risk", "none", 0, 0),
    ],
)
@pytest.mark.parametrize("workers", [2, 4])
def test_parallel_poisson_groups_preserve_images_sigma_maps_and_callbacks(
    monkeypatch: pytest.MonkeyPatch,
    weighting: str,
    mass: str,
    cache_bytes: int,
    workers: int,
    k: int,
) -> None:
    original_threads = get_num_threads()
    if original_threads < workers:
        pytest.skip(f"Requires at least {workers} available Numba threads.")
    profile = BMNDProfile(
        noise_model="poisson",
        ht_block_size=(4, 4),
        wiener_block_size=(4, 4),
        ht_step=(4, 4),
        wiener_step=(4, 4),
        ht_search_window=(2, 2),
        wiener_search_window=(2, 2),
        ht_max_stack_size=4,
        wiener_max_stack_size=4,
        k=k,
        poisson_group_mass_conservation=mass,
    )
    profile.ht_transform = Transform(TransformType.DCT, TransformMode.ND)
    profile.wiener_transform = Transform(TransformType.DCT, TransformMode.ND)
    if weighting != "default":
        profile.wiener_weight_model = weighting
        profile.wiener_weight_domain = "windowed_synthesis"
        profile.wiener_weight_scope = "group"
    volume = np.random.default_rng(42).poisson(6.0, size=(20, 20)).astype(np.float32)
    original_volume = volume.tobytes()
    caller = get_ident()
    events: list[str] = []
    pool_events: list[tuple[str, ...]] = []

    def make_pool(
        *, max_workers: int, initializer: Callable[[], None]
    ) -> ThreadPoolExecutor:
        assert get_ident() == caller
        assert max_workers == workers
        pool_events.append(tuple(events))
        return ThreadPoolExecutor(max_workers=max_workers, initializer=initializer)

    monkeypatch.setattr(groupfiltering, "ThreadPoolExecutor", make_pool)

    def on_stage(stage: str, estimate: NDArray[np.float32]) -> None:
        assert get_ident() == caller
        events.append(stage)

    try:
        set_num_threads(workers)
        monkeypatch.setattr(groupfiltering, "_PARALLEL_POISSON_GROUPS", False)
        expected = algorithm.bmnd(
            volume,
            profile,
            return_sigma_map=True,
            max_cache_bytes=cache_bytes,
            stage_callback=on_stage,
        )
        monkeypatch.setattr(groupfiltering, "_PARALLEL_POISSON_GROUPS", True)
        actual = algorithm.bmnd(
            volume,
            profile,
            return_sigma_map=True,
            max_cache_bytes=cache_bytes,
            stage_callback=on_stage,
        )
        assert isinstance(expected, tuple) and isinstance(actual, tuple)
        np.testing.assert_array_equal(actual[0], expected[0], strict=True)
        assert actual[0].tobytes() == expected[0].tobytes()
        assert actual[1].keys() == expected[1].keys() == {"global", "ht", "wiener"}
        assert actual[1]["global"] == expected[1]["global"]
        for stage in ("ht", "wiener"):
            np.testing.assert_array_equal(actual[1][stage], expected[1][stage], strict=True)
            assert actual[1][stage].tobytes() == expected[1][stage].tobytes()
        assert events == ["ht", "wiener", "ht", "wiener"]
        assert pool_events == (
            [("ht", "wiener", "ht")]
            if weighting != "default" and k > 0
            else []
        )
        assert get_num_threads() == workers
        assert volume.tobytes() == original_volume
    finally:
        set_num_threads(original_threads)
