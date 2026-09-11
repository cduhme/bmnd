import numpy as np
import pytest

from bmnd.profiles import BM3DProfile, BM4DProfile
from bmnd.transforms import (
    Transform,
    TransformMode,
    TransformType,
    get_transform_matrix,
)
from bmnd.wiener import (
    compute_poisson_wiener_gain,
    compute_poisson_wiener_risk,
    compute_poisson_wiener_signal_power,
    compute_wiener_gain,
)


def test_dct_transform_matches_official_reference_matrix_for_n8() -> None:
    bm4d = pytest.importorskip("bm4d")
    forward, _ = get_transform_matrix(
        (8,), Transform(TransformType.DCT, TransformMode.ND)
    )
    official_forward, _ = bm4d._get_transf_matrix(8, "dct")

    assert forward[0] is not None
    assert not callable(forward[0])
    assert np.allclose(np.asarray(forward[0]), official_forward, atol=1e-8)


@pytest.mark.parametrize("size", [4, 8])
def test_bior15_transform_matches_official_reference_matrix(size: int) -> None:
    bm4d = pytest.importorskip("bm4d")
    forward, _ = get_transform_matrix(
        (size,), Transform(TransformType.BIOR15, TransformMode.ND)
    )
    official_forward, _ = bm4d._get_transf_matrix(size, "bior1.5")

    assert forward[0] is not None
    assert not callable(forward[0])
    assert np.allclose(np.asarray(forward[0]), official_forward, atol=1e-8)


def test_wiener_gain_uses_variance_scaling_not_exponentiation() -> None:
    reference = np.array([0.0, 0.25, 1.0, 2.0], dtype=np.float32)
    gain_small = compute_wiener_gain(reference, 0.2, wiener_variance_scale=0.4)
    gain_large = compute_wiener_gain(reference, 0.2, wiener_variance_scale=2.0)

    power = reference**2
    np.testing.assert_allclose(gain_small, power / (power + 0.4 * 0.2**2))
    np.testing.assert_allclose(gain_large, power / (power + 2.0 * 0.2**2))


def test_poisson_wiener_gain_variance_scaled_shrinks_weak_coefficients_more() -> None:
    reference = np.array([0.0, 0.1, 0.4, 1.5], dtype=np.float32)
    sigma = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32)

    gain_classic = compute_poisson_wiener_gain(
        reference,
        sigma,
        wiener_variance_scale=1.0,
        mode="classic",
    )
    gain_variance_scaled = compute_poisson_wiener_gain(
        reference,
        sigma,
        wiener_variance_scale=1.0,
        mode="variance-scaled",
    )

    power = reference**2
    signal = np.maximum(power - sigma**2, 0.0)
    np.testing.assert_allclose(gain_classic, power / (power + sigma**2))
    np.testing.assert_allclose(
        gain_variance_scaled, signal / (signal + sigma**2)
    )


def test_poisson_wiener_gain_noise_floor_sits_between_classic_and_variance_scaled() -> None:
    reference = np.array([0.0, 0.3, 0.6, 1.5], dtype=np.float32)
    sigma = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32)

    gain_classic = compute_poisson_wiener_gain(
        reference,
        sigma,
        wiener_variance_scale=2.0,
        mode="classic",
    )
    gain_noise_floor = compute_poisson_wiener_gain(
        reference,
        sigma,
        wiener_variance_scale=2.0,
        mode="noise-floor",
    )
    gain_variance_scaled = compute_poisson_wiener_gain(
        reference,
        sigma,
        wiener_variance_scale=2.0,
        mode="variance-scaled",
    )

    for gain, subtraction in (
        (gain_classic, 0.0),
        (gain_noise_floor, 1.0),
        (gain_variance_scaled, 2.0),
    ):
        signal = np.maximum(reference**2 - subtraction * sigma**2, 0.0)
        np.testing.assert_allclose(gain, signal / (signal + 2.0 * sigma**2))


def test_poisson_wiener_signal_power_tracks_gain_mode_definition() -> None:
    reference = np.array([0.0, 0.3, 0.6, 1.5], dtype=np.float32)
    sigma = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32)

    signal_classic = compute_poisson_wiener_signal_power(
        reference,
        sigma,
        wiener_variance_scale=2.0,
        mode="classic",
    )
    signal_noise_floor = compute_poisson_wiener_signal_power(
        reference,
        sigma,
        wiener_variance_scale=2.0,
        mode="noise-floor",
    )
    signal_variance_scaled = compute_poisson_wiener_signal_power(
        reference,
        sigma,
        wiener_variance_scale=2.0,
        mode="variance-scaled",
    )

    np.testing.assert_allclose(signal_classic, reference**2)
    np.testing.assert_allclose(
        signal_noise_floor, np.maximum(reference**2 - sigma**2, 0.0)
    )
    np.testing.assert_allclose(
        signal_variance_scaled, np.maximum(reference**2 - 2.0 * sigma**2, 0.0)
    )


def test_poisson_wiener_risk_adds_bias_term_beyond_heteroscedastic_noise_energy() -> None:
    reference = np.array([0.0, 0.3, 0.6, 1.5], dtype=np.float32)
    sigma = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32)

    gain = compute_poisson_wiener_gain(
        reference,
        sigma,
        wiener_variance_scale=1.0,
        mode="classic",
    )
    heteroscedastic = (gain**2) * (sigma**2)
    risk = compute_poisson_wiener_risk(
        reference,
        sigma,
        wiener_variance_scale=1.0,
        mode="classic",
    )

    np.testing.assert_allclose(
        risk, heteroscedastic + (1.0 - gain) ** 2 * reference**2
    )


def test_poisson_wiener_risk_follows_signal_power_debiasing_strength() -> None:
    reference = np.array([0.0, 0.3, 0.6, 1.5], dtype=np.float32)
    sigma = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32)

    risk_classic = compute_poisson_wiener_risk(
        reference,
        sigma,
        wiener_variance_scale=2.0,
        mode="classic",
    )
    risk_noise_floor = compute_poisson_wiener_risk(
        reference,
        sigma,
        wiener_variance_scale=2.0,
        mode="noise-floor",
    )
    risk_variance_scaled = compute_poisson_wiener_risk(
        reference,
        sigma,
        wiener_variance_scale=2.0,
        mode="variance-scaled",
    )

    for risk, subtraction in (
        (risk_classic, 0.0),
        (risk_noise_floor, 1.0),
        (risk_variance_scaled, 2.0),
    ):
        signal = np.maximum(reference**2 - subtraction * sigma**2, 0.0)
        gain = signal / (signal + 2.0 * sigma**2)
        np.testing.assert_allclose(
            risk, gain**2 * sigma**2 + (1.0 - gain) ** 2 * signal
        )


def test_gaussian_compat_profiles_use_official_white_defaults() -> None:
    bm3d_profile = BM3DProfile()
    bm4d_profile = BM4DProfile()

    assert np.isclose(bm3d_profile.gamma, 3.0)
    assert np.isclose(bm4d_profile.gamma, 3.0)
    assert np.isclose(bm3d_profile.wiener_variance_scale, 0.4)
    assert np.isclose(bm4d_profile.wiener_variance_scale, 0.4)


def test_bm3d_and_bm4d_profiles_use_official_transform_defaults() -> None:
    bm3d_profile = BM3DProfile()
    bm4d_profile = BM4DProfile()

    assert bm3d_profile.ht_transform[0] == Transform(
        TransformType.HAAR, TransformMode.GROUP
    )
    assert bm3d_profile.ht_transform[1] == Transform(
        TransformType.BIOR15, TransformMode.SPATIAL
    )
    assert bm3d_profile.wiener_transform[0] == Transform(
        TransformType.HAAR, TransformMode.GROUP
    )
    assert bm3d_profile.wiener_transform[1] == Transform(
        TransformType.DCT, TransformMode.SPATIAL
    )

    assert bm4d_profile.ht_transform[0] == Transform(
        TransformType.HAAR, TransformMode.GROUP
    )
    assert bm4d_profile.ht_transform[1] == Transform(
        TransformType.BIOR15, TransformMode.SPATIAL
    )
    assert bm4d_profile.wiener_transform[0] == Transform(
        TransformType.HAAR, TransformMode.GROUP
    )
    assert bm4d_profile.wiener_transform[1] == Transform(
        TransformType.DCT, TransformMode.SPATIAL
    )
