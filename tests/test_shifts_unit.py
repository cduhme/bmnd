import itertools
import math

from bmnd.shifts import (
    get_reference_schedule,
    get_reference_shifts,
    get_reference_starts,
)


def test_reference_shifts_match_official_bm3d_case():
    shifts = get_reference_shifts((8, 8), (3, 3))
    assert [s.tolist() for s in shifts] == [[0, 1, 2], [0]]


def test_reference_shifts_match_official_bm4d_ht_case():
    shifts = get_reference_shifts((4, 4, 4), (3, 3, 3))
    assert [s.tolist() for s in shifts] == [[0, 1, 2], [0, 2], [0]]


def test_reference_shifts_match_official_bm4d_wiener_case():
    shifts = get_reference_shifts((5, 5, 5), (3, 3, 3))
    assert [s.tolist() for s in shifts] == [[0, 1, 2], [0, 4], [0]]


def test_reference_shifts_support_additional_official_cases():
    assert [s.tolist() for s in get_reference_shifts((6, 6, 6), (3, 3, 3))] == [
        [0, 2, 1],
        [0, 2, 4],
        [0],
    ]
    assert [s.tolist() for s in get_reference_shifts((7, 7, 7), (3, 3, 3))] == [
        [0, 1, 2],
        [0, 2, 1],
        [0],
    ]
    assert [s.tolist() for s in get_reference_shifts((8, 8, 8), (3, 3, 3))] == [
        [0, 2, 1],
        [0, 5, 1],
        [0],
    ]


def test_reference_starts_expand_shifts_into_origins():
    starts = get_reference_starts((9, 9, 9), (4, 4, 4), (3, 3, 3))
    assert starts == [[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 2, 3, 5, 6, 8], [0, 3, 6]]


def test_reference_starts_fallback_to_zero_for_unsupported_dimensions():
    starts = get_reference_starts((5, 5, 5, 5), (4, 4, 4, 4), (3, 3, 3, 3))
    assert starts == [[0, 3], [0, 3], [0, 3], [0, 3]]


def test_reference_schedule_preserves_shift_pass_metadata():
    schedule = get_reference_schedule((9, 9, 9), (4, 4, 4), (3, 3, 3))

    assert schedule[0].shift == (0, 0, 0)
    assert schedule[0].reference_abs == (0, 0, 0)
    assert schedule[0].reference_shifted == (0, 0, 0)

    assert schedule[5].shift == (1, 0, 0)
    assert schedule[5].reference_abs == (2, 3, 0)
    assert schedule[5].reference_shifted == (3, 3, 0)


def test_reference_schedule_keeps_official_shift_order():
    schedule = get_reference_schedule((9, 9, 9), (4, 4, 4), (3, 3, 3))
    first_reference_abs = [schedule[index].reference_abs for index in range(12)]

    assert first_reference_abs == [
        (0, 0, 0),
        (0, 3, 0),
        (0, 6, 0),
        (0, 8, 0),
        (3, 0, 0),
        (2, 3, 0),
        (1, 6, 0),
        (3, 8, 0),
        (6, 0, 0),
        (5, 3, 0),
        (4, 6, 0),
        (6, 8, 0),
    ]


def test_generated_reference_shifts_use_sparse_asymmetric_nd_pattern():
    shifts = get_reference_shifts((8, 8), (3, 3), mode="generated")
    assert [s.tolist() for s in shifts] == [[0, 4, 7], [0]]


def test_generated_reference_schedule_covers_far_boundary_without_duplicates():
    schedule = get_reference_schedule((9, 9), (8, 8), (3, 3), mode="generated")
    reference_abs = [item.reference_abs for item in schedule]

    assert reference_abs[0] == (0, 0)
    assert reference_abs[-1] == (8, 8)
    assert len(reference_abs) == len(set(reference_abs))


def test_generated_reference_schedule_extends_hierarchically_to_4d():
    schedule = get_reference_schedule(
        (9, 9, 9, 9),
        (5, 5, 5, 5),
        (3, 3, 3, 3),
        mode="generated",
    )

    assert schedule
    assert schedule[0].reference_abs == (0, 0, 0, 0)
    assert schedule[-1].reference_abs == (8, 8, 8, 8)
    assert all(len(item.reference_abs) == 4 for item in schedule)


def test_balanced_reference_shifts_include_every_non_singleton_axis() -> None:
    shifts = get_reference_shifts(
        (8, 1, 8, 1, 8),
        (3, 1, 3, 1, 3),
        mode="balanced",
        shift_density=2.0,
    )

    assert [shift.tolist() for shift in shifts] == [
        [0, 2, 4, 5, 7],
        [0],
        [0, 2, 4, 5, 7],
        [0],
        [0, 2, 4, 5, 7],
    ]


def test_balanced_reference_schedule_ignores_inserted_singleton_axes() -> None:
    schedule_3d = get_reference_schedule(
        (9, 8, 7),
        (8, 8, 8),
        (3, 3, 3),
        mode="balanced",
        shift_density=2.0,
        schedule_density=2,
    )
    schedule_5d = get_reference_schedule(
        (9, 1, 8, 1, 7),
        (8, 1, 8, 1, 8),
        (3, 1, 3, 1, 3),
        mode="balanced",
        shift_density=2.0,
        schedule_density=2,
    )

    active_axes = (0, 2, 4)
    squeezed_5d = [
        (
            tuple(item.shift[axis] for axis in active_axes),
            tuple(item.reference_abs[axis] for axis in active_axes),
            tuple(item.reference_shifted[axis] for axis in active_axes),
        )
        for item in schedule_5d
    ]
    metadata_3d = [
        (item.shift, item.reference_abs, item.reference_shifted) for item in schedule_3d
    ]

    assert squeezed_5d == metadata_3d


def test_balanced_reference_schedule_is_axis_permutation_equivariant() -> None:
    counts = (7, 8, 9)
    block_size = (4, 5, 6)
    step_size = (2, 3, 4)
    permutation = (2, 0, 1)
    schedule = get_reference_schedule(
        counts,
        block_size,
        step_size,
        mode="balanced",
        shift_density=1.5,
        schedule_density=2,
    )
    permuted_schedule = get_reference_schedule(
        tuple(counts[axis] for axis in permutation),
        tuple(block_size[axis] for axis in permutation),
        tuple(step_size[axis] for axis in permutation),
        mode="balanced",
        shift_density=1.5,
        schedule_density=2,
    )

    def restore_axis_order(values: tuple[int, ...]) -> tuple[int, ...]:
        restored = [0] * len(permutation)
        for new_axis, old_axis in enumerate(permutation):
            restored[old_axis] = values[new_axis]
        return tuple(restored)

    reference_abs = {item.reference_abs for item in schedule}
    restored_reference_abs = {
        restore_axis_order(item.reference_abs) for item in permuted_schedule
    }

    assert restored_reference_abs == reference_abs


def test_balanced_reference_schedule_has_fixed_phase_budget_and_boundaries() -> None:
    counts = (9, 1, 9, 1, 9)
    step_size = (3, 1, 3, 1, 3)
    schedule_density = 2
    schedule = get_reference_schedule(
        counts,
        (8, 1, 8, 1, 8),
        step_size,
        mode="balanced",
        shift_density=2.0,
        schedule_density=schedule_density,
    )
    reference_abs = [item.reference_abs for item in schedule]
    last_valid = tuple(count - 1 for count in counts)
    slot_counts = tuple(
        (last + step - 1) // step + 1
        for last, step in zip(last_valid, step_size)
    )

    assert reference_abs[0] == (0, 0, 0, 0, 0)
    assert reference_abs[-1] == last_valid
    assert len(reference_abs) == len(set(reference_abs))
    assert len(schedule) <= schedule_density * int(math.prod(slot_counts))


def test_sparse_reference_shifts_ignore_singleton_axis_placement() -> None:
    cases = [
        ((8, 8), (3, 3), [[0, 2, 4, 5, 7], [0]]),
        ((1, 8, 8), (1, 3, 3), [[0], [0, 2, 4, 5, 7], [0]]),
        ((8, 1, 8), (3, 1, 3), [[0, 2, 4, 5, 7], [0], [0]]),
        ((8, 8, 1), (3, 3, 1), [[0, 2, 4, 5, 7], [0], [0]]),
    ]

    for block_size, step_size, expected in cases:
        shifts = get_reference_shifts(
            block_size,
            step_size,
            mode="sparse",
            shift_density=2.0,
        )
        assert [shift.tolist() for shift in shifts] == expected


def test_sparse_reference_schedule_ignores_singleton_axis_placement() -> None:
    schedule_2d = get_reference_schedule(
        (9, 8),
        (8, 8),
        (3, 3),
        mode="sparse",
        shift_density=2.0,
        schedule_density=2,
    )
    metadata_2d = [
        (item.shift, item.reference_abs, item.reference_shifted) for item in schedule_2d
    ]
    cases = [
        ((1, 9, 8), (1, 8, 8), (1, 3, 3), (1, 2)),
        ((9, 1, 8), (8, 1, 8), (3, 1, 3), (0, 2)),
        ((9, 8, 1), (8, 8, 1), (3, 3, 1), (0, 1)),
    ]

    for counts, block_size, step_size, active_axes in cases:
        schedule = get_reference_schedule(
            counts,
            block_size,
            step_size,
            mode="sparse",
            shift_density=2.0,
            schedule_density=2,
        )
        squeezed_metadata = [
            (
                tuple(item.shift[axis] for axis in active_axes),
                tuple(item.reference_abs[axis] for axis in active_axes),
                tuple(item.reference_shifted[axis] for axis in active_axes),
            )
            for item in schedule
        ]
        assert squeezed_metadata == metadata_2d


def test_sparse_reference_schedule_scales_to_5d_with_two_singletons() -> None:
    schedule_3d = get_reference_schedule(
        (9, 8, 7),
        (8, 8, 8),
        (3, 3, 3),
        mode="sparse",
        shift_density=2.0,
        schedule_density=2,
    )
    counts_5d = (9, 1, 8, 1, 7)
    step_size_5d = (3, 1, 3, 1, 3)
    schedule_5d = get_reference_schedule(
        counts_5d,
        (8, 1, 8, 1, 8),
        step_size_5d,
        mode="sparse",
        shift_density=2.0,
        schedule_density=2,
    )

    active_axes = (0, 2, 4)
    squeezed_5d = [
        (
            tuple(item.shift[axis] for axis in active_axes),
            tuple(item.reference_abs[axis] for axis in active_axes),
            tuple(item.reference_shifted[axis] for axis in active_axes),
        )
        for item in schedule_5d
    ]
    metadata_3d = [
        (item.shift, item.reference_abs, item.reference_shifted) for item in schedule_3d
    ]
    last_valid = tuple(count - 1 for count in counts_5d)
    slot_counts = tuple(
        (last + step - 1) // step + 1
        for last, step in zip(last_valid, step_size_5d)
    )

    assert squeezed_5d == metadata_3d
    assert schedule_5d[-1].reference_abs == last_valid
    assert len(schedule_5d) <= 2 * int(math.prod(slot_counts))


def test_off_reference_shifts_are_zero_in_every_dimension() -> None:
    shifts = get_reference_shifts(
        (8, 1, 8, 1, 8),
        (3, 1, 3, 1, 3),
        mode="off",
        shift_density=100.0,
    )

    assert [shift.tolist() for shift in shifts] == [[0], [0], [0], [0], [0]]


def test_off_reference_schedule_is_boundary_inclusive_and_density_independent() -> None:
    schedule = get_reference_schedule(
        (9, 8),
        (8, 8),
        (3, 3),
        mode="off",
        shift_density=100.0,
        schedule_density=100,
    )
    expected_positions = list(
        itertools.product(
            (0, 3, 6, 8),
            (0, 3, 6, 7),
        )
    )

    assert [item.reference_abs for item in schedule] == expected_positions
    assert all(item.shift == (0, 0) for item in schedule)
    assert all(item.reference_shifted == item.reference_abs for item in schedule)
    assert get_reference_starts(
        (9, 8),
        (8, 8),
        (3, 3),
        mode="off",
        shift_density=100.0,
    ) == [[0, 3, 6, 8], [0, 3, 6, 7]]
