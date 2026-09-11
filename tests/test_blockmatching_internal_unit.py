import math

import numpy as np
import pytest

from bmnd.blockmatching import (
    _batched_distance_blockmatch,
    _get_distance_code,
    _standardized_poisson_distance_numba,
    blockmatch_group,
    blockmatch_groups,
    blockmatch_full_numba,
    compute_blockmatch_threshold,
    linear_index_to_multi,
    neighborhood_lin_indices_numba,
    poisson_null_moment_table,
)
from bmnd.shifts import ReferenceScheduleItem


def test_neighborhood_lin_indices_center_and_edge():
    counts = np.array([3, 3], dtype=np.int64)
    radius = np.array([1, 1], dtype=np.int64)

    center = np.array([1, 1], dtype=np.int64)
    neigh_center = neighborhood_lin_indices_numba(center, counts, radius)
    assert set(neigh_center.tolist()) == set(range(9))

    edge = np.array([0, 0], dtype=np.int64)
    neigh_edge = neighborhood_lin_indices_numba(edge, counts, radius)
    assert set(neigh_edge.tolist()) == {0, 1, 3, 4}


def test_blockmatch_full_numba_respects_max_matches():
    ref_multis = np.array([[1]], dtype=np.int64)
    counts = np.array([4], dtype=np.int64)
    patches = np.array([[0.0], [1.0], [3.0], [4.0]], dtype=np.float32)
    radius = np.array([3], dtype=np.int64)

    matches, counts_out = blockmatch_full_numba(
        ref_multis_arr=ref_multis,
        counts_arr=counts,
        patches_flat=patches,
        search_radius_arr=radius,
        max_matches=2,
        distance_threshold=100.0,
    )

    assert counts_out[0] == 2
    assert matches[0, :2].tolist() == [1, 0]


def test_blockmatch_full_numba_threshold_prunes_candidates():
    ref_multis = np.array([[1]], dtype=np.int64)
    counts = np.array([3], dtype=np.int64)
    patches = np.array([[0.0], [10.0], [20.0]], dtype=np.float32)
    radius = np.array([1], dtype=np.int64)

    matches, counts_out = blockmatch_full_numba(
        ref_multis_arr=ref_multis,
        counts_arr=counts,
        patches_flat=patches,
        search_radius_arr=radius,
        max_matches=3,
        distance_threshold=1e-6,
    )

    assert counts_out[0] == 1
    assert matches[0, 0] == 1


def test_batched_distance_blockmatch_empty_refs_returns_empty():
    matches = _batched_distance_blockmatch(
        ref_multis=[],
        counts=(3,),
        patches_flat=np.zeros((3, 2), dtype=np.float32),
        search_radius=(1,),
        max_matches=2,
        distance_threshold=1.0,
        batch_size=2,
    )
    assert matches == []


def test_compute_blockmatch_threshold_scales_with_input_magnitude():
    base = np.array([[0.0, 0.5], [0.25, 1.0]], dtype=np.float32)
    scaled = 2.0 * base

    base_threshold = compute_blockmatch_threshold(2.0, base)
    scaled_threshold = compute_blockmatch_threshold(2.0, scaled)

    assert np.isclose(base_threshold, 2.0)
    assert np.isclose(scaled_threshold, 8.0)


def test_blockmatch_group_fills_to_min_stack_size_when_threshold_is_tight():
    group = blockmatch_group(
        schedule_item=ReferenceScheduleItem(
            shift=(0,),
            reference_abs=(1,),
            reference_shifted=(1,),
        ),
        counts=(4,),
        patches_flat=np.array([[0.0], [10.0], [20.0], [30.0]], dtype=np.float32),
        search_radius=(3,),
        max_matches=4,
        min_matches=3,
        distance_threshold=1e-8,
    )

    assert group.accepted_count == 1
    assert group.final_count == 3
    assert group.selected_lin_indices.tolist() == [1, 0, 2]


def test_blockmatch_group_stores_shifted_positions():
    group = blockmatch_group(
        schedule_item=ReferenceScheduleItem(
            shift=(2, 1),
            reference_abs=(2, 1),
            reference_shifted=(0, 0),
        ),
        counts=(4, 4),
        patches_flat=np.arange(16, dtype=np.float32).reshape(16, 1),
        search_radius=(0, 0),
        max_matches=1,
        min_matches=1,
        distance_threshold=1.0,
    )

    assert group.selected_abs_positions.tolist() == [[2, 1]]
    assert group.selected_shifted_positions.tolist() == [[0, 0]]


def test_blockmatch_group_breaks_distance_ties_by_linear_index():
    group = blockmatch_group(
        schedule_item=ReferenceScheduleItem(
            shift=(0,),
            reference_abs=(2,),
            reference_shifted=(2,),
        ),
        counts=(5,),
        patches_flat=np.array([[1.0], [3.0], [2.0], [1.0], [3.0]], dtype=np.float32),
        search_radius=(2,),
        max_matches=5,
        min_matches=1,
        distance_threshold=10.0,
        prefer_power_of_two_stack=False,
    )

    assert group.selected_lin_indices.tolist() == [2, 0, 1, 3, 4]


def test_blockmatch_group_keeps_reference_first_when_threshold_accepts_nothing():
    group = blockmatch_group(
        schedule_item=ReferenceScheduleItem(
            shift=(0,),
            reference_abs=(1,),
            reference_shifted=(1,),
        ),
        counts=(4,),
        patches_flat=np.array([[0.0], [10.0], [20.0], [30.0]], dtype=np.float32),
        search_radius=(3,),
        max_matches=2,
        min_matches=1,
        distance_threshold=-1.0,
        prefer_power_of_two_stack=False,
    )

    assert group.accepted_count == 0
    assert group.selected_lin_indices.tolist() == [1]


def test_blockmatch_group_gamma_corrects_acceptance_and_keeps_reference_first():
    common_args = {
        "schedule_item": ReferenceScheduleItem(
            shift=(0,),
            reference_abs=(1,),
            reference_shifted=(1,),
        ),
        "counts": (3,),
        "patches_flat": np.array([[2.0], [0.0], [3.0]], dtype=np.float32),
        "search_radius": (1,),
        "max_matches": 3,
        "min_matches": 1,
        "distance_threshold": 1.0,
        "prefer_power_of_two_stack": False,
    }

    uncorrected = blockmatch_group(**common_args)
    corrected = blockmatch_group(
        **common_args,
        gaussian_noise_ssd=np.array([5.0, 0.0, 5.0], dtype=np.float32),
        gamma=1.0,
    )

    assert uncorrected.accepted_count == 1
    assert uncorrected.selected_lin_indices.tolist() == [1]
    assert corrected.accepted_count == 2
    assert corrected.selected_lin_indices.tolist() == [1, 0]


def test_blockmatch_group_gamma_corrects_poisson_ssd_noise_bias():
    common_args = {
        "schedule_item": ReferenceScheduleItem(
            shift=(0,),
            reference_abs=(0,),
            reference_shifted=(0,),
        ),
        "counts": (2,),
        "patches_flat": np.array([[0.0], [2.0]], dtype=np.float32),
        "search_radius": (1,),
        "max_matches": 2,
        "min_matches": 1,
        "distance_threshold": 1.0,
        "prefer_power_of_two_stack": False,
        "distance_measure": "ssd",
        "poisson_count_scale": 1.0,
    }

    uncorrected = blockmatch_group(**common_args)
    corrected = blockmatch_group(
        **common_args,
        poisson_gamma=True,
        gamma=2.0,
    )

    assert uncorrected.selected_lin_indices.tolist() == [0]
    assert corrected.selected_lin_indices.tolist() == [0, 1]


@pytest.mark.parametrize(
    "distance_measure",
    ["poisson_deviance", "pearson", "anscombe_ssd"],
)
def test_standardized_poisson_distance_has_unit_conditional_null_moments(
    distance_measure: str,
) -> None:
    pooled_count = 4
    means, variances = poisson_null_moment_table(distance_measure, 8)
    scores = []
    probabilities = []
    for reference_count in range(pooled_count + 1):
        candidate_count = pooled_count - reference_count
        scores.append(
            float(
                _standardized_poisson_distance_numba(
                    np.array([reference_count], dtype=np.float32),
                    np.array([candidate_count], dtype=np.float32),
                    _get_distance_code(distance_measure),
                    1.0,
                    means.astype(np.float32),
                    variances.astype(np.float32),
                )
            )
        )
        probabilities.append(math.comb(pooled_count, reference_count) / 2**pooled_count)

    scores_array = np.asarray(scores)
    probabilities_array = np.asarray(probabilities)
    mean = float(np.sum(probabilities_array * scores_array))
    variance = float(np.sum(probabilities_array * (scores_array - mean) ** 2))

    assert np.isclose(mean, 0.0, atol=1e-6)
    assert np.isclose(variance, 1.0, atol=1e-6)


def test_blockmatch_group_prefers_power_of_two_stack_after_reference_insertion():
    group = blockmatch_group(
        schedule_item=ReferenceScheduleItem(
            shift=(0,),
            reference_abs=(2,),
            reference_shifted=(2,),
        ),
        counts=(5,),
        patches_flat=np.array([[1.0], [3.0], [2.0], [1.0], [3.0]], dtype=np.float32),
        search_radius=(2,),
        max_matches=5,
        min_matches=2,
        distance_threshold=10.0,
        prefer_power_of_two_stack=True,
    )

    assert group.final_count == 4
    assert group.selected_lin_indices.tolist() == [2, 0, 1, 3]


def test_blockmatch_groups_is_batch_size_invariant():
    reference_schedule = [
        ReferenceScheduleItem(
            shift=(0, 0),
            reference_abs=(0, 0),
            reference_shifted=(0, 0),
        ),
        ReferenceScheduleItem(
            shift=(0, 0),
            reference_abs=(1, 1),
            reference_shifted=(1, 1),
        ),
        ReferenceScheduleItem(
            shift=(0, 0),
            reference_abs=(2, 2),
            reference_shifted=(2, 2),
        ),
    ]
    counts = (3, 3)
    patches_flat = np.arange(9, dtype=np.float32).reshape(9, 1)

    full = blockmatch_groups(
        reference_schedule=reference_schedule,
        counts=counts,
        patches_flat=patches_flat,
        search_radius=(1, 1),
        max_matches=4,
        min_matches=2,
        distance_threshold=100.0,
        batch_size=None,
    )
    batched = blockmatch_groups(
        reference_schedule=reference_schedule,
        counts=counts,
        patches_flat=patches_flat,
        search_radius=(1, 1),
        max_matches=4,
        min_matches=2,
        distance_threshold=100.0,
        batch_size=2,
    )

    assert len(full.groups) == len(batched.groups) == len(reference_schedule)
    for group_full, group_batched in zip(full.groups, batched.groups, strict=False):
        assert group_full.reference_abs == group_batched.reference_abs
        assert group_full.reference_shifted == group_batched.reference_shifted
        assert group_full.accepted_count == group_batched.accepted_count
        assert group_full.final_count == group_batched.final_count
        assert (
            group_full.selected_lin_indices.tolist()
            == group_batched.selected_lin_indices.tolist()
        )
        assert (
            group_full.selected_abs_positions.tolist()
            == group_batched.selected_abs_positions.tolist()
        )
        assert (
            group_full.selected_shifted_positions.tolist()
            == group_batched.selected_shifted_positions.tolist()
        )


def test_linear_index_to_multi_round_trips_c_order():
    assert linear_index_to_multi(0, (3, 4, 5)) == (0, 0, 0)
    assert linear_index_to_multi(19, (3, 4, 5)) == (0, 3, 4)
    assert linear_index_to_multi(59, (3, 4, 5)) == (2, 3, 4)
