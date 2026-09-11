from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .patchaggregation import (
    PatchAggregationPlan,
    get_normalized_group_occurrence_weights_prepared,
    scatter_add_patch_numerator_prepared,
)
from .transforms import apply_transform_nd

TransformEntry = NDArray[np.generic] | Callable | None

_POISSON_THRESHOLD_ROUNDOFF_FACTOR = 128.0


@dataclass(frozen=True)
class _FrozenAggregationGroup:
    selected: NDArray[np.int64]
    weights: NDArray[np.float32]
    coefficients: NDArray[np.generic]
    inverse_transforms: list[NDArray | Callable | None]
    correction_direction: NDArray[np.generic]


def _poisson_hard_threshold_mask(
    coefficient_magnitude: NDArray[np.floating],
    threshold: NDArray[np.floating],
    count_scale: float,
) -> NDArray[np.bool_]:
    coefficient_counts = np.asarray(coefficient_magnitude, dtype=np.float64) / count_scale
    threshold_counts = np.asarray(threshold, dtype=np.float64) / count_scale
    comparison_scale = np.maximum(
        np.maximum(coefficient_counts, threshold_counts),
        1.0,
    )
    tolerance = (
        _POISSON_THRESHOLD_ROUNDOFF_FACTOR
        * np.finfo(np.float32).eps
        * comparison_scale
    )
    return coefficient_counts > threshold_counts + tolerance


def _get_group_mass_functional(
    shape: tuple[int, ...],
    inverse_transforms: Sequence[TransformEntry],
) -> NDArray[np.generic]:
    if len(inverse_transforms) != len(shape):
        raise ValueError(
            "inverse_transforms must contain one entry per group axis, "
            f"got {len(inverse_transforms)} entries for shape {shape}."
        )

    axis_functionals: list[NDArray[np.generic]] = []
    for axis, (size, inverse_transform) in enumerate(zip(shape, inverse_transforms)):
        if inverse_transform is None:
            axis_functionals.append(np.ones(size, dtype=np.float64))
            continue
        if callable(inverse_transform):
            raise ValueError(
                "Poisson group-mass conservation requires matrix-based transforms; "
                f"axis {axis} uses an operator transform."
            )

        matrix = np.asarray(inverse_transform)
        if matrix.shape != (size, size):
            raise ValueError(
                f"Inverse transform on axis {axis} has shape {matrix.shape}, "
                f"expected {(size, size)}."
            )
        axis_functionals.append(np.sum(matrix, axis=0, dtype=matrix.dtype))

    mass_functional = axis_functionals[0]
    for axis_functional in axis_functionals[1:]:
        mass_functional = np.multiply.outer(mass_functional, axis_functional)
    return np.asarray(mass_functional)


def _get_group_synthesis_functional(
    output_functional: NDArray[np.floating],
    inverse_transforms: Sequence[TransformEntry],
) -> NDArray[np.generic]:
    functional = np.asarray(output_functional)
    if len(inverse_transforms) != functional.ndim:
        raise ValueError(
            "inverse_transforms must contain one entry per group axis, "
            f"got {len(inverse_transforms)} entries for shape {functional.shape}."
        )

    result = functional
    for axis, inverse_transform in enumerate(inverse_transforms):
        if inverse_transform is None:
            continue
        if callable(inverse_transform):
            raise ValueError(
                "Poisson group-mass conservation requires matrix-based transforms; "
                f"axis {axis} uses an operator transform."
            )
        matrix = np.asarray(inverse_transform)
        size = functional.shape[axis]
        if matrix.shape != (size, size):
            raise ValueError(
                f"Inverse transform on axis {axis} has shape {matrix.shape}, "
                f"expected {(size, size)}."
            )
        moved = np.moveaxis(result, axis, 0)
        transformed = (matrix.T @ moved.reshape(size, -1)).reshape(moved.shape)
        result = np.moveaxis(transformed, 0, axis)
    return np.asarray(result)


def _get_group_constant_synthesis_direction(
    shape: tuple[int, ...],
    inverse_transforms: Sequence[TransformEntry],
) -> NDArray[np.generic]:
    if len(inverse_transforms) != len(shape):
        raise ValueError(
            "inverse_transforms must contain one entry per group axis, "
            f"got {len(inverse_transforms)} entries for shape {shape}."
        )

    axis_directions: list[NDArray[np.generic]] = []
    for axis, (size, inverse_transform) in enumerate(zip(shape, inverse_transforms)):
        if inverse_transform is None:
            axis_directions.append(np.ones(size, dtype=np.float64))
            continue
        if callable(inverse_transform):
            raise ValueError(
                "Poisson group-mass conservation requires matrix-based transforms; "
                f"axis {axis} uses an operator transform."
            )
        matrix = np.asarray(inverse_transform)
        if matrix.shape != (size, size):
            raise ValueError(
                f"Inverse transform on axis {axis} has shape {matrix.shape}, "
                f"expected {(size, size)}."
            )
        solve_dtype = np.complex128 if np.iscomplexobj(matrix) else np.float64
        try:
            axis_direction = np.linalg.solve(
                np.asarray(matrix, dtype=solve_dtype),
                np.ones(size, dtype=solve_dtype),
            )
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"Inverse transform on axis {axis} has no constant synthesis mode."
            ) from exc
        axis_directions.append(axis_direction)

    direction = axis_directions[0]
    for axis_direction in axis_directions[1:]:
        direction = np.multiply.outer(direction, axis_direction)
    return np.asarray(direction)


def _preserve_group_functional(
    coefficients: NDArray[np.generic],
    source_group: NDArray[np.floating],
    output_functional: NDArray[np.floating],
    inverse_transforms: Sequence[TransformEntry],
    eps: float,
    correction_direction: NDArray[np.generic],
) -> NDArray[np.generic]:
    coefficient_array = np.asarray(coefficients)
    source_array = np.asarray(source_group)
    output_array = np.asarray(output_functional)
    if not (coefficient_array.shape == source_array.shape == output_array.shape):
        raise ValueError(
            "coefficients, source_group, and output_functional must have the same "
            f"shape, got {coefficient_array.shape}, {source_array.shape}, and "
            f"{output_array.shape}."
        )

    coefficient_functional = _get_group_synthesis_functional(
        output_array,
        inverse_transforms,
    )
    target = np.sum(output_array * source_array, dtype=np.float64)
    denominator = float(np.sum(np.abs(coefficient_functional) ** 2, dtype=np.float64))
    if denominator <= eps:
        functional_scale = float(np.max(np.abs(coefficient_functional)))
        if not np.isfinite(functional_scale) or functional_scale == 0.0:
            raise RuntimeError("Inverse transforms have no nonzero weighted functional.")
        coefficient_functional = coefficient_functional / functional_scale
        target = target / functional_scale
    use_complex = (
        np.iscomplexobj(coefficient_functional)
        or np.iscomplexobj(coefficient_array)
        or np.iscomplexobj(correction_direction)
    )
    current = np.sum(
        coefficient_functional * coefficient_array,
        dtype=np.complex128 if use_complex else np.float64,
    )
    residual = target - current
    if residual == 0.0:
        return coefficient_array.copy()
    direction_array = np.asarray(correction_direction)
    if direction_array.shape != coefficient_array.shape:
        raise ValueError(
            "correction_direction and coefficients must have the same shape, "
            f"got {direction_array.shape} and {coefficient_array.shape}."
        )
    direction_scale = float(np.max(np.abs(direction_array)))
    if not np.isfinite(direction_scale) or direction_scale == 0.0:
        raise RuntimeError("Group-mass correction direction must be finite and nonzero.")
    direction = direction_array / direction_scale
    response = np.sum(
        coefficient_functional * direction,
        dtype=np.complex128 if use_complex else np.float64,
    )
    if not np.isfinite(response) or response == 0.0:
        raise RuntimeError(
            "Group-mass correction direction cannot change the weighted mass."
        )
    correction = residual / response
    corrected = coefficient_array + correction * direction
    if not np.iscomplexobj(coefficient_array):
        if np.iscomplexobj(corrected):
            raise ValueError(
                "A complex mass correction cannot be represented by real coefficients."
            )
        corrected = np.real(corrected)
    return np.asarray(corrected, dtype=coefficient_array.dtype)


def _reaggregate_with_group_functionals(
    frozen_groups: list[_FrozenAggregationGroup],
    source_patches: NDArray[np.float32],
    block_size: tuple[int, ...],
    denominator: NDArray[np.float32],
    aggregation_plan: PatchAggregationPlan,
    eps: float,
) -> NDArray[np.float32]:
    numerator = np.zeros_like(denominator, dtype=np.float32)
    for frozen in frozen_groups:
        source_group = source_patches[frozen.selected].reshape(-1, *block_size)
        output_functional = get_normalized_group_occurrence_weights_prepared(
            denominator,
            frozen.weights,
            frozen.selected,
            aggregation_plan,
        ).reshape(source_group.shape)
        corrected = _preserve_group_functional(
            frozen.coefficients,
            source_group,
            output_functional,
            frozen.inverse_transforms,
            eps,
            frozen.correction_direction,
        )
        group_filt = apply_transform_nd(corrected, frozen.inverse_transforms)
        scatter_add_patch_numerator_prepared(
            numerator,
            group_filt,
            frozen.weights,
            frozen.selected,
            aggregation_plan,
        )
    return numerator
