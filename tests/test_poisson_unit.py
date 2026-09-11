import numpy as np
import pytest
from numpy.typing import NDArray

from bmnd.poisson import (
    _get_group_constant_synthesis_direction,
    _get_group_mass_functional,
    _preserve_group_functional,
)
from bmnd.transforms import (
    Transform,
    TransformMode,
    TransformType,
    apply_transform_nd,
    get_transform_matrix,
)


def test_group_mass_functional_rejects_operator_transform() -> None:
    with pytest.raises(ValueError, match="matrix-based transforms"):
        _get_group_mass_functional((2,), [lambda value: value])


@pytest.mark.parametrize("transform_type", [TransformType.DCT, TransformType.BIOR15])
@pytest.mark.parametrize(
    "functional_scale, direction_scale", [(1e-12, 1.0), (1.0, 1e-12)]
)
def test_dc_group_functional_is_invariant_to_scaling(
    transform_type: TransformType,
    functional_scale: float,
    direction_scale: float,
) -> None:
    source = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
    output_functional = np.linspace(0.1, 0.8, source.size).reshape(source.shape)
    forward, inverse = get_transform_matrix(
        source.shape,
        Transform(transform_type, TransformMode.ND),
    )
    filtered = 0.5 * apply_transform_nd(source, forward)
    direction = _get_group_constant_synthesis_direction(source.shape, inverse)

    corrected = _preserve_group_functional(
        filtered,
        source,
        output_functional,
        inverse,
        eps=1e-10,
        correction_direction=direction,
    )
    corrected_tiny = _preserve_group_functional(
        filtered,
        source,
        output_functional * functional_scale,
        inverse,
        eps=1e-10,
        correction_direction=direction * direction_scale,
    )

    assert np.allclose(corrected_tiny, corrected, rtol=1e-6)


@pytest.mark.parametrize("transform_type", [TransformType.DCT, TransformType.BIOR15])
def test_group_constant_direction_synthesizes_ones(
    transform_type: TransformType,
) -> None:
    shape = (8,)
    _, inverse = get_transform_matrix(
        shape,
        Transform(transform_type, TransformMode.ND),
    )

    direction = _get_group_constant_synthesis_direction(shape, inverse)

    assert np.allclose(apply_transform_nd(direction, inverse), 1.0, atol=1e-6)


@pytest.mark.parametrize(
    "transform_type", [TransformType.DCT, TransformType.BIOR15, TransformType.FFT]
)
def test_dc_only_group_functional_correction_is_constant_and_exact(
    transform_type: TransformType,
) -> None:
    source = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
    output_functional = np.linspace(0.1, 0.8, source.size).reshape(source.shape)
    forward, inverse = get_transform_matrix(
        source.shape,
        Transform(transform_type, TransformMode.ND),
    )
    filtered = 0.5 * apply_transform_nd(source, forward)
    direction = _get_group_constant_synthesis_direction(source.shape, inverse)

    corrected = _preserve_group_functional(
        filtered,
        source,
        output_functional,
        inverse,
        eps=1e-10,
        correction_direction=direction,
    )
    synthesized = apply_transform_nd(corrected, inverse)

    spatial_correction = synthesized - apply_transform_nd(filtered, inverse)
    assert corrected.dtype == filtered.dtype
    assert np.allclose(spatial_correction, spatial_correction.reshape(-1)[0], atol=1e-5)
    assert np.isclose(
        np.sum(output_functional * synthesized),
        np.sum(output_functional * source),
        rtol=1e-6,
    )


def test_dc_group_functional_accepts_zero_neutral_group() -> None:
    source = np.zeros((2, 4), dtype=np.float32)
    output_functional = np.ones_like(source)
    _, inverse = get_transform_matrix(
        source.shape,
        Transform(TransformType.DCT, TransformMode.ND),
    )
    direction = _get_group_constant_synthesis_direction(source.shape, inverse)

    corrected = _preserve_group_functional(
        np.zeros_like(source),
        source,
        output_functional,
        inverse,
        eps=1e-10,
        correction_direction=direction,
    )

    assert np.array_equal(corrected, source)


@pytest.mark.parametrize(
    "direction, error, match",
    [
        (np.ones(3), ValueError, "same shape"),
        (np.zeros(2), RuntimeError, "finite and nonzero"),
        (np.array([1.0, np.nan]), RuntimeError, "finite and nonzero"),
        (np.array([1.0, np.inf]), RuntimeError, "finite and nonzero"),
        (np.array([1.0, -1.0]), RuntimeError, "cannot change the weighted mass"),
    ],
)
def test_group_functional_rejects_invalid_correction_direction(
    direction: NDArray[np.float64],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        _preserve_group_functional(
            np.zeros(2),
            np.ones(2),
            np.ones(2),
            [None],
            eps=1e-10,
            correction_direction=direction,
        )
