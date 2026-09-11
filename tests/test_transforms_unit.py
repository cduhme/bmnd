import numpy as np

from bmnd.transforms import (
    Transform,
    TransformMode,
    TransformType,
    apply_transform_nd,
    get_transform_matrix,
)


def test_dct_round_trip_nd():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(2, 3)).astype(np.float32)

    forward, inverse = get_transform_matrix(
        data.shape, Transform(TransformType.DCT, TransformMode.ND)
    )

    coeffs = apply_transform_nd(data, forward)
    recon = apply_transform_nd(coeffs, inverse)

    assert np.allclose(recon, data, atol=1e-6)


def test_transform_mode_group_targets_group_axis_only():
    forward, inverse = get_transform_matrix(
        (2, 3, 4), Transform(TransformType.DCT, TransformMode.GROUP)
    )

    assert forward[0] is not None
    assert inverse[0] is not None
    assert forward[1] is None
    assert inverse[1] is None
    assert forward[2] is None
    assert inverse[2] is None


def test_transform_mode_spatial_targets_spatial_axes_only():
    forward, inverse = get_transform_matrix(
        (2, 3, 4), Transform(TransformType.DCT, TransformMode.SPATIAL)
    )

    assert forward[0] is None
    assert inverse[0] is None
    assert forward[1] is not None
    assert inverse[1] is not None
    assert forward[2] is not None
    assert inverse[2] is not None
