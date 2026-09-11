from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

bm4d = pytest.importorskip("bm4d")

from bm4d.profiles import BM4DStages

from bmnd.blockmatching import (
    BlockMatchResult,
    blockmatch_groups,
    compute_blockmatch_threshold,
)
from bmnd.profiles import BM4DProfile
from bmnd.shifts import get_reference_schedule
from bmnd.utils import extract_patches_strided


def _to_official_storage_coords(coords: tuple[int, ...]) -> tuple[int, ...]:
    if len(coords) in (2, 3):
        reordered = list(coords)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        return tuple(int(x) for x in reordered)
    return tuple(int(x) for x in coords)


def _make_clean_volume(
    shape: tuple[int, int, int] = (12, 12, 12)
) -> NDArray[np.float32]:
    z, y, x = np.meshgrid(
        np.linspace(0.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(0.0, 1.0, shape[1], dtype=np.float32),
        np.linspace(0.0, 1.0, shape[2], dtype=np.float32),
        indexing="ij",
    )
    vol = 0.35 + 0.20 * np.sin(2.0 * np.pi * x * 1.5)
    vol += 0.12 * np.cos(2.0 * np.pi * y * 1.0)
    vol += 0.10 * z
    vol += 0.12 * (((x - 0.55) ** 2 + (y - 0.45) ** 2 + (z - 0.60) ** 2) < 0.05)
    vol += 0.08 * ((x > 0.65) & (y < 0.35) & (z > 0.40))
    return np.clip(vol.astype(np.float32), 0.0, 1.0)


def _make_noisy(
    clean: NDArray[np.float32], sigma: float, seed: int = 123
) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    noisy = clean + rng.normal(0.0, sigma, size=clean.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def _run_bmnd_blockmatching(
    volume: NDArray[np.float32],
    *,
    block_size: tuple[int, int, int],
    step: tuple[int, int, int],
    search_window: tuple[int, int, int],
    max_stack_size: int,
    min_stack_size: int,
    match_threshold: float,
) -> list[dict[str, object]]:
    patches_view = extract_patches_strided(volume, block_size)
    counts = cast(
        tuple[int, int, int], tuple(int(x) for x in patches_view.shape[: volume.ndim])
    )
    patches_flat = patches_view.reshape(
        int(np.prod(counts)), int(np.prod(block_size))
    ).astype(
        np.float32,
        copy=False,
    )
    threshold = compute_blockmatch_threshold(match_threshold, volume)
    result: BlockMatchResult = blockmatch_groups(
        reference_schedule=get_reference_schedule(counts, block_size, step),
        counts=counts,
        patches_flat=patches_flat,
        search_radius=search_window,
        max_matches=max_stack_size,
        min_matches=min_stack_size,
        distance_threshold=threshold,
        batch_size=None,
    )

    groups: list[dict[str, object]] = []
    for group in result.groups:
        if group.final_count == 0:
            continue
        groups.append(
            {
                "reference_abs": _to_official_storage_coords(group.reference_abs),
                "reference_shifted": _to_official_storage_coords(
                    group.reference_shifted
                ),
                "size": group.final_count,
                "positions": [
                    _to_official_storage_coords(tuple(int(x) for x in pos))
                    for pos in group.selected_abs_positions.tolist()
                ],
                "shifted_positions": [
                    _to_official_storage_coords(tuple(int(x) for x in pos))
                    for pos in group.selected_shifted_positions.tolist()
                ],
            }
        )
    return groups


def _extract_official_groups(storage: Any) -> list[dict[str, object]]:
    sizes, positions, shifted_positions = storage.get_python_structs()
    max_stack_size = int(storage.maxStackSize)
    groups: list[dict[str, object]] = []
    for group_index in range(int(storage.currentStack)):
        size = int(sizes[group_index])
        if size <= 0:
            continue
        start = group_index * max_stack_size
        positions_group = [
            tuple(int(x) for x in pos) for pos in positions[start : start + size]
        ]
        shifted_group = [
            tuple(int(x) for x in pos)
            for pos in shifted_positions[start : start + size]
        ]
        groups.append(
            {
                "reference_abs": positions_group[0],
                "reference_shifted": shifted_group[0],
                "size": size,
                "positions": positions_group,
                "shifted_positions": shifted_group,
            }
        )
    return groups


def test_bmnd_ht_blockmatch_layout_tracks_official_bm4d_storage() -> None:
    sigma = 0.05
    noisy = _make_noisy(_make_clean_volume(), sigma=sigma)
    profile = BM4DProfile()

    _, (official_ht, _) = bm4d.bm4d(
        noisy,
        sigma,
        profile="np",
        stage_arg=BM4DStages.HARD_THRESHOLDING,
        blockmatches=(True, False),
    )
    official_groups = _extract_official_groups(official_ht)
    bmnd_groups = _run_bmnd_blockmatching(
        noisy,
        block_size=profile.ht_block_size,
        step=profile.ht_step,
        search_window=profile.ht_search_window,
        max_stack_size=profile.ht_max_stack_size,
        min_stack_size=profile.ht_min_stack_size,
        match_threshold=profile.ht_match_threshold,
    )

    assert official_groups
    assert bmnd_groups
    assert [group["reference_abs"] for group in bmnd_groups[:12]] == [
        group["reference_abs"] for group in official_groups[:12]
    ]
    assert [group["reference_shifted"] for group in bmnd_groups[:12]] == [
        group["reference_shifted"] for group in official_groups[:12]
    ]
    assert [group["size"] for group in bmnd_groups[:12]] == [
        group["size"] for group in official_groups[:12]
    ]


def test_bmnd_wiener_blockmatch_layout_tracks_official_bm4d_storage() -> None:
    sigma = 0.05
    clean = _make_clean_volume()
    noisy = _make_noisy(clean, sigma=sigma)
    pilot = np.clip(0.65 * clean + 0.35 * noisy, 0.0, 1.0).astype(np.float32)
    profile = BM4DProfile()

    _, (_, official_wiener) = bm4d.bm4d(
        noisy,
        sigma,
        profile="np",
        stage_arg=pilot,
        blockmatches=(False, True),
    )
    official_groups = _extract_official_groups(official_wiener)
    bmnd_groups = _run_bmnd_blockmatching(
        pilot,
        block_size=profile.wiener_block_size,
        step=profile.wiener_step,
        search_window=profile.wiener_search_window,
        max_stack_size=profile.wiener_max_stack_size,
        min_stack_size=profile.wiener_min_stack_size,
        match_threshold=profile.wiener_match_threshold,
    )

    assert official_groups
    assert bmnd_groups
    assert [group["reference_abs"] for group in bmnd_groups[:12]] == [
        group["reference_abs"] for group in official_groups[:12]
    ]
    assert [group["reference_shifted"] for group in bmnd_groups[:12]] == [
        group["reference_shifted"] for group in official_groups[:12]
    ]
    assert [group["size"] for group in bmnd_groups[:12]] == [
        group["size"] for group in official_groups[:12]
    ]
