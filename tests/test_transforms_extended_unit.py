import numpy as np
import pytest

from bmnd.transforms import (
    Transform,
    TransformMode,
    TransformType,
    apply_transform_nd,
    get_transform_matrix,
)


def test_apply_transform_nd_all_none_is_identity():
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = apply_transform_nd(data, [None, None])
    assert np.array_equal(out, data)


def test_dst_round_trip_nd():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(4, 4)).astype(np.float32)
    fw, inv = get_transform_matrix(
        data.shape, Transform(TransformType.DST, TransformMode.ND)
    )
    coeff = apply_transform_nd(data, fw)
    recon = apply_transform_nd(coeff, inv)
    assert np.allclose(recon, data, atol=1e-5)


def test_fft_round_trip_nd_complex():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(4, 4)).astype(np.float32)
    fw, inv = get_transform_matrix(
        data.shape, Transform(TransformType.FFT, TransformMode.ND)
    )
    coeff = apply_transform_nd(data, fw)
    recon = apply_transform_nd(coeff, inv)
    assert np.allclose(recon, data, atol=1e-5)


def test_haar_round_trip_nd():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(4, 4)).astype(np.float32)
    fw, inv = get_transform_matrix(
        data.shape, Transform(TransformType.HAAR, TransformMode.ND)
    )
    coeff = apply_transform_nd(data, fw)
    recon = apply_transform_nd(coeff, inv)
    assert np.allclose(recon, data, atol=1e-5)


def test_hadamard_round_trip_nd_power_of_two():
    rng = np.random.default_rng(4)
    data = rng.normal(size=(4, 4)).astype(np.float32)
    fw, inv = get_transform_matrix(
        data.shape, Transform(TransformType.HADAMARD, TransformMode.ND)
    )
    coeff = apply_transform_nd(data, fw)
    recon = apply_transform_nd(coeff, inv)
    assert np.allclose(recon, data, atol=1e-5)


def test_hadamard_non_power_of_two_raises():
    with pytest.raises(ValueError, match="power of two"):
        get_transform_matrix(
            (3, 4), Transform(TransformType.HADAMARD, TransformMode.ND)
        )


def test_mixed_transform_list_group_and_spatial_round_trip():
    rng = np.random.default_rng(5)
    data = rng.normal(size=(2, 4, 4)).astype(np.float32)
    transform = [
        Transform(TransformType.HAAR, TransformMode.GROUP),
        Transform(TransformType.DCT, TransformMode.SPATIAL),
    ]
    fw, inv = get_transform_matrix(data.shape, transform)
    coeff = apply_transform_nd(data, fw)
    recon = apply_transform_nd(coeff, inv)
    assert np.allclose(recon, data, atol=1e-5)


@pytest.mark.parametrize("group_size", [2, 3, 4, 5, 7, 8, 15, 16])
def test_group_haar_is_orthonormal(group_size: int):
    fw, inv = get_transform_matrix(
        (group_size, 4, 4), Transform(TransformType.HAAR, TransformMode.GROUP)
    )

    group_forward = np.asarray(fw[0])
    group_inverse = np.asarray(inv[0])
    identity = np.eye(group_size, dtype=np.float64)

    assert np.allclose(group_forward.T @ group_forward, identity, atol=1e-6)
    assert np.allclose(group_forward @ group_forward.T, identity, atol=1e-6)
    assert np.allclose(group_inverse, group_forward.T, atol=1e-6)


def test_group_haar_size_four_matches_official_hierarchical_order():
    fw, _ = get_transform_matrix(
        (4, 2), Transform(TransformType.HAAR, TransformMode.GROUP)
    )
    expected = np.array(
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5, -0.5],
            [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0, 0.0],
            [0.0, 0.0, 1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)],
        ],
        dtype=np.float32,
    )

    assert np.allclose(np.asarray(fw[0]), expected, atol=1e-7)
