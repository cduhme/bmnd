import numpy as np
import pytest

from bmnd.aggregationweights import (
    compute_diagonal_filter_group_risk,
    compute_diagonal_filter_patch_risks,
    compute_mass_projected_diagonal_filter_risks,
    get_synthesis_risk_metric_factors,
    invert_aggregation_risks,
)


def _orthonormal_matrix(size: int, seed: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(size, size))
    orthonormal, _ = np.linalg.qr(matrix)
    return orthonormal.astype(np.float32)


@pytest.mark.parametrize("model", ["classic", "variance", "risk"])
def test_diagonal_group_risk_matches_dense_windowed_metric(model: str) -> None:
    group_size = 3
    patch_size = 2
    inverse_group = _orthonormal_matrix(group_size, 1)
    inverse_spatial = np.array([[1.0, 0.3], [0.2, 0.8]], dtype=np.float32)
    window = np.array([0.5, 1.25], dtype=np.float32)
    attenuation = np.array(
        [[0.0, 0.2], [0.7, 1.0], [0.4, 0.9]],
        dtype=np.float32,
    )
    variance = np.array(
        [[0.5, 0.7], [1.1, 1.3], [1.7, 1.9]],
        dtype=np.float32,
    )
    signal = np.array(
        [[2.0, 2.2], [2.4, 2.6], [2.8, 3.0]],
        dtype=np.float32,
    )
    factors = get_synthesis_risk_metric_factors(
        attenuation.shape,
        [inverse_group, inverse_spatial],
        window,
    )

    if model == "classic":
        coefficient_error = attenuation.astype(np.float64) ** 2
    elif model == "variance":
        coefficient_error = attenuation.astype(np.float64) ** 2 * variance
    else:
        coefficient_error = (
            attenuation.astype(np.float64) ** 2 * variance
            + (1.0 - attenuation.astype(np.float64)) ** 2 * signal
        )
    synthesis = np.kron(
        inverse_group.astype(np.float64),
        inverse_spatial.astype(np.float64),
    )
    repeated_window = np.tile(window, group_size).astype(np.float64)
    metric = synthesis.T @ ((repeated_window**2)[:, None] * synthesis)
    expected = float(np.sum(np.diag(metric) * coefficient_error.reshape(-1)))

    actual = compute_diagonal_filter_group_risk(
        attenuation,
        model=model,
        domain="windowed_synthesis",
        noise_variance=None if model == "classic" else variance,
        signal_power=signal if model == "risk" else None,
        metric_factors=factors,
    )

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("domain", ["coefficient", "windowed_synthesis"])
def test_patch_risks_match_dense_patch_selection_metrics(domain: str) -> None:
    group_size = 2
    patch_size = 3
    inverse_group = _orthonormal_matrix(group_size, 2)
    inverse_spatial = np.array(
        [[1.0, 0.1, 0.0], [0.2, 0.8, 0.2], [0.0, 0.3, 0.7]],
        dtype=np.float32,
    )
    window = np.array([0.4, 1.0, 0.7], dtype=np.float32)
    attenuation = np.array(
        [[0.2, 0.5, 0.8], [0.1, 0.6, 0.9]],
        dtype=np.float32,
    )
    variance = np.array(
        [[0.7, 0.9, 1.1], [1.3, 1.5, 1.7]],
        dtype=np.float32,
    )
    coefficient_error = attenuation.astype(np.float64) ** 2 * variance
    factors = get_synthesis_risk_metric_factors(
        attenuation.shape,
        [inverse_group, inverse_spatial],
        window,
    )

    actual = compute_diagonal_filter_patch_risks(
        attenuation,
        model="variance",
        domain=domain,
        noise_variance=variance,
        metric_factors=factors,
    )
    spatial = (
        np.eye(patch_size, dtype=np.float64)
        if domain == "coefficient"
        else inverse_spatial.astype(np.float64).T
        @ ((window.astype(np.float64) ** 2)[:, None] * inverse_spatial)
    )
    expected = np.empty(group_size, dtype=np.float64)
    for patch_index in range(group_size):
        group_row = inverse_group[patch_index].astype(np.float64)
        group_metric = np.outer(group_row, group_row)
        metric_diag = np.diag(np.kron(group_metric, spatial))
        expected[patch_index] = np.sum(metric_diag * coefficient_error.reshape(-1))

    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_unitary_coefficient_group_risk_equals_sum_of_patch_risks() -> None:
    attenuation = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    variance = np.array([[0.5, 0.7], [1.1, 1.3]], dtype=np.float32)
    inverse_group = _orthonormal_matrix(2, 3)
    factors = get_synthesis_risk_metric_factors(
        attenuation.shape,
        [inverse_group, np.eye(2, dtype=np.float32)],
        np.ones(2, dtype=np.float32),
    )

    group_risk = compute_diagonal_filter_group_risk(
        attenuation,
        model="variance",
        domain="coefficient",
        noise_variance=variance,
    )
    patch_risks = compute_diagonal_filter_patch_risks(
        attenuation,
        model="variance",
        domain="coefficient",
        noise_variance=variance,
        metric_factors=factors,
    )

    assert np.sum(patch_risks) == pytest.approx(group_risk, rel=1e-7)


def test_unitary_synthesis_with_constant_window_matches_coefficient_domain() -> None:
    attenuation = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    variance = np.array([[0.5, 0.7], [1.1, 1.3]], dtype=np.float32)
    factors = get_synthesis_risk_metric_factors(
        attenuation.shape,
        [_orthonormal_matrix(2, 5), _orthonormal_matrix(2, 6)],
        np.ones(2, dtype=np.float32),
    )

    coefficient_group = compute_diagonal_filter_group_risk(
        attenuation,
        model="variance",
        domain="coefficient",
        noise_variance=variance,
    )
    windowed_group = compute_diagonal_filter_group_risk(
        attenuation,
        model="variance",
        domain="windowed_synthesis",
        noise_variance=variance,
        metric_factors=factors,
    )
    coefficient_patches = compute_diagonal_filter_patch_risks(
        attenuation,
        model="variance",
        domain="coefficient",
        noise_variance=variance,
        metric_factors=factors,
    )
    windowed_patches = compute_diagonal_filter_patch_risks(
        attenuation,
        model="variance",
        domain="windowed_synthesis",
        noise_variance=variance,
        metric_factors=factors,
    )

    assert windowed_group == pytest.approx(coefficient_group, rel=1e-7)
    assert np.allclose(windowed_patches, coefficient_patches, rtol=1e-7, atol=1e-9)


def test_classic_and_variance_risks_are_proportional_for_constant_variance() -> None:
    attenuation = np.arange(1, 7, dtype=np.float32).reshape(2, 3) / 7.0
    variance = np.full_like(attenuation, 2.5)

    classic = compute_diagonal_filter_group_risk(
        attenuation,
        model="classic",
        domain="coefficient",
    )
    variance_risk = compute_diagonal_filter_group_risk(
        attenuation,
        model="variance",
        domain="coefficient",
        noise_variance=variance,
    )

    assert variance_risk == pytest.approx(2.5 * classic)


@pytest.mark.parametrize("retained", [False, True], ids=["all-rejected", "all-retained"])
def test_ht_extreme_masks_have_expected_variance_risk(retained: bool) -> None:
    attenuation = np.full((2, 3), float(retained), dtype=np.float32)
    variance = np.arange(1, 7, dtype=np.float32).reshape(2, 3)

    risk = compute_diagonal_filter_group_risk(
        attenuation,
        model="variance",
        domain="coefficient",
        noise_variance=variance,
    )

    expected = float(np.sum(variance, dtype=np.float64)) if retained else 0.0
    assert risk == expected


@pytest.mark.parametrize("scope", ["group", "patch"])
@pytest.mark.parametrize("domain", ["coefficient", "windowed_synthesis"])
@pytest.mark.parametrize("model", ["variance", "risk"])
def test_mass_projected_risk_matches_dense_operator(
    domain: str, scope: str, model: str,
) -> None:
    group_size = 2
    patch_size = 2
    inverse_group = _orthonormal_matrix(group_size, 4).astype(np.float64)
    inverse_spatial = np.array([[1.0, 0.2], [0.1, 0.9]], dtype=np.float64)
    window = np.array([0.5, 1.0], dtype=np.float32)
    attenuation = np.array([[0.2, 0.7], [0.4, 0.9]], dtype=np.float32)
    variance = np.array([[0.5, 0.8], [1.1, 1.4]], dtype=np.float32)
    signal = np.array([[2.0, 2.3], [2.6, 2.9]], dtype=np.float32)
    mass = np.array([[1.0, 0.5], [0.25, 0.75]], dtype=np.float32)
    factors = get_synthesis_risk_metric_factors(
        attenuation.shape,
        [inverse_group, inverse_spatial],
        window,
    )

    actual = compute_mass_projected_diagonal_filter_risks(
        attenuation,
        model=model,
        domain=domain,
        scope=scope,
        noise_variance=variance,
        signal_power=signal if model == "risk" else None,
        mass_functional=mass,
        metric_factors=factors,
    )

    attenuation_flat = attenuation.astype(np.float64).reshape(-1)
    mass_flat = mass.astype(np.float64).reshape(-1)
    diagonal_filter = np.diag(attenuation_flat)
    projection = np.outer(mass_flat, mass_flat) / np.dot(mass_flat, mass_flat)
    projected_filter = diagonal_filter + projection @ (
        np.eye(attenuation_flat.size) - diagonal_filter
    )
    noise_covariance = np.diag(variance.astype(np.float64).reshape(-1))
    signal_covariance = np.diag(signal.astype(np.float64).reshape(-1))
    error_noise = projected_filter @ noise_covariance @ projected_filter.T
    bias_filter = projected_filter - np.eye(attenuation_flat.size)
    error_bias = bias_filter @ signal_covariance @ bias_filter.T
    if model == "variance":
        error_bias = np.zeros_like(error_noise)

    spatial_metric = (
        np.eye(patch_size, dtype=np.float64)
        if domain == "coefficient"
        else factors.spatial_metric
    )
    if scope == "group":
        group_metric = (
            np.eye(group_size, dtype=np.float64)
            if domain == "coefficient"
            else factors.group_metric
        )
        expected = np.array(
            [np.trace(np.kron(group_metric, spatial_metric) @ (error_noise + error_bias))]
        )
    else:
        expected = np.empty(group_size, dtype=np.float64)
        for patch_index in range(group_size):
            row = inverse_group[patch_index]
            metric = np.kron(np.outer(row, row), spatial_metric)
            expected[patch_index] = np.trace(metric @ (error_noise + error_bias))

    assert np.allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_mass_projected_patch_rejects_reshaped_variance() -> None:
    attenuation = np.array([[0.2, 0.7], [0.4, 0.9]], dtype=np.float32)
    factors = get_synthesis_risk_metric_factors(
        attenuation.shape,
        [_orthonormal_matrix(2, 9), np.eye(2, dtype=np.float32)],
        np.ones(2, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="same shape"):
        compute_mass_projected_diagonal_filter_risks(
            attenuation,
            model="variance",
            domain="coefficient",
            scope="patch",
            noise_variance=np.ones(4, dtype=np.float32),
            mass_functional=np.ones_like(attenuation),
            metric_factors=factors,
        )


def test_invert_aggregation_risks_uses_fixed_epsilon() -> None:
    weights = invert_aggregation_risks(
        np.array([0.0, 2.0]),
        group_size=2,
        scope="patch",
        eps=1e-10,
    )

    assert weights[0] == pytest.approx(1e10)
    assert weights[1] == pytest.approx(0.5)


def test_invert_aggregation_risks_rejects_negative_epsilon_tie() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        invert_aggregation_risks(
            np.array([-1e-10]),
            group_size=1,
            scope="patch",
            eps=1e-10,
        )


def test_patch_risks_support_float16() -> None:
    attenuation = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float16)
    variance = np.array([[0.5, 0.7], [1.1, 1.3]], dtype=np.float16)
    factors = get_synthesis_risk_metric_factors(
        attenuation.shape,
        [_orthonormal_matrix(2, 7), np.eye(2, dtype=np.float32)],
        np.ones(2, dtype=np.float32),
    )

    risks = compute_diagonal_filter_patch_risks(
        attenuation,
        model="variance",
        domain="coefficient",
        metric_factors=factors,
        noise_variance=variance,
    )
    error = attenuation.astype(np.float64) ** 2 * variance.astype(np.float64)
    expected = factors.group_mixing_energy @ np.sum(error, axis=1)

    assert np.allclose(risks, expected, rtol=1e-12, atol=1e-12)
