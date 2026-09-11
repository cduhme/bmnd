import numpy as np
import pytest

from bmnd.blockmatching import (
    _batched_distance_blockmatch,
    compute_blockmatch_threshold,
    resolve_blockmatch_distance,
)


def test_blockmatch_self_only_with_zero_radius() -> None:
    patches_flat = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    ref_multis = [(0,), (1,), (2,)]
    counts = (3,)

    matches = _batched_distance_blockmatch(
        ref_multis=ref_multis,
        counts=counts,
        patches_flat=patches_flat,
        search_radius=(0,),
        max_matches=2,
        distance_threshold=1.0,
        batch_size=None,
    )

    assert len(matches) == 3
    assert np.array_equal(matches[0], np.array([0], dtype=np.int64))
    assert np.array_equal(matches[1], np.array([1], dtype=np.int64))
    assert np.array_equal(matches[2], np.array([2], dtype=np.int64))


def test_blockmatch_includes_self_in_neighbors() -> None:
    patches_flat = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    ref_multis = [(1,)]
    counts = (3,)

    matches = _batched_distance_blockmatch(
        ref_multis=ref_multis,
        counts=counts,
        patches_flat=patches_flat,
        search_radius=(1,),
        max_matches=2,
        distance_threshold=100.0,
        batch_size=None,
    )

    assert matches[0].tolist() == [1, 0]


def test_blockmatch_batch_size_equivalence() -> None:
    patches_flat = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [1.1, 1.1]], dtype=np.float32
    )
    ref_multis = [(0,), (1,), (2,), (3,)]
    counts = (4,)

    matches_single = _batched_distance_blockmatch(
        ref_multis=ref_multis,
        counts=counts,
        patches_flat=patches_flat,
        search_radius=(1,),
        max_matches=3,
        distance_threshold=100.0,
        batch_size=None,
    )
    matches_batched = _batched_distance_blockmatch(
        ref_multis=ref_multis,
        counts=counts,
        patches_flat=patches_flat,
        search_radius=(1,),
        max_matches=3,
        distance_threshold=100.0,
        batch_size=2,
    )

    assert len(matches_single) == len(matches_batched) == len(ref_multis)
    for a, b in zip(matches_single, matches_batched):
        assert np.array_equal(a, b)


def test_blockmatch_distance_auto_resolution_tracks_noise_model() -> None:
    assert resolve_blockmatch_distance("auto", "gaussian") == "ssd"
    assert resolve_blockmatch_distance("auto", "poisson") == "poisson_deviance"
    assert resolve_blockmatch_distance("pearson", "poisson") == "pearson"


def test_poisson_distance_threshold_does_not_depend_on_data_range() -> None:
    threshold = compute_blockmatch_threshold(
        2.5,
        np.array([0.0, 100.0], dtype=np.float32),
        distance_measure="poisson_deviance",
    )

    assert threshold == 2.5


@pytest.mark.parametrize(
    "distance_measure",
    ["poisson_deviance", "pearson", "anscombe_ssd"],
)
def test_poisson_distances_account_for_count_dependent_variance(
    distance_measure: str,
) -> None:
    patches_flat = np.array(
        [
            [0.0, 2.0],
            [2.0, 2.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    ssd_matches = _batched_distance_blockmatch(
        ref_multis=[(2,)],
        counts=(3,),
        patches_flat=patches_flat,
        search_radius=(2,),
        max_matches=3,
        distance_threshold=100.0,
        distance_measure="ssd",
    )
    poisson_matches = _batched_distance_blockmatch(
        ref_multis=[(2,)],
        counts=(3,),
        patches_flat=patches_flat,
        search_radius=(2,),
        max_matches=3,
        distance_threshold=100.0,
        distance_measure=distance_measure,
    )

    assert ssd_matches[0].tolist() == [2, 0, 1]
    assert poisson_matches[0].tolist() == [2, 1, 0]


def test_unknown_blockmatch_distance_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown block-matching distance"):
        resolve_blockmatch_distance("unknown", "poisson")


@pytest.mark.parametrize(
    "distance_measure",
    ["poisson_deviance", "pearson", "anscombe_ssd"],
)
def test_poisson_distances_are_invariant_to_representation_scale(
    distance_measure: str,
) -> None:
    count_patches = np.array(
        [
            [11.0, 9.0],
            [10.0, 10.0],
            [20.0, 20.0],
        ],
        dtype=np.float32,
    )
    representation_scale = 7.0

    count_matches = _batched_distance_blockmatch(
        ref_multis=[(1,)],
        counts=(3,),
        patches_flat=count_patches,
        search_radius=(1,),
        max_matches=3,
        distance_threshold=0.2,
        distance_measure=distance_measure,
        poisson_count_scale=1.0,
    )
    scaled_matches = _batched_distance_blockmatch(
        ref_multis=[(1,)],
        counts=(3,),
        patches_flat=representation_scale * count_patches,
        search_radius=(1,),
        max_matches=3,
        distance_threshold=0.2,
        distance_measure=distance_measure,
        poisson_count_scale=representation_scale,
    )

    assert count_matches[0].tolist() == [1, 0]
    assert scaled_matches[0].tolist() == count_matches[0].tolist()


@pytest.mark.parametrize("poisson_count_scale", [0.0, -1.0, np.inf, np.nan])
def test_blockmatching_rejects_invalid_poisson_count_scale(
    poisson_count_scale: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="poisson_count_scale must be finite and positive",
    ):
        _batched_distance_blockmatch(
            ref_multis=[(0,)],
            counts=(1,),
            patches_flat=np.ones((1, 1), dtype=np.float32),
            search_radius=(0,),
            max_matches=1,
            distance_threshold=1.0,
            distance_measure="poisson_deviance",
            poisson_count_scale=poisson_count_scale,
        )
