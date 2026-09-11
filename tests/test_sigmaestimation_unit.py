import numpy as np
import pytest

from bmnd.sigmaestimation import (
    estimate_gaussian_global_sigma,
    estimate_poisson_global_sigma,
)


def test_estimate_gaussian_global_sigma_uses_mad_for_simple_case():
    volume = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

    sigma = estimate_gaussian_global_sigma(volume)

    assert sigma == pytest.approx(1.0 / 0.6745)


def test_estimate_gaussian_global_sigma_falls_back_to_eps_for_constant_input():
    volume = np.full((4, 4), 2.0, dtype=np.float32)

    sigma = estimate_gaussian_global_sigma(volume)

    assert sigma == pytest.approx(np.sqrt(1e-10))


def test_estimate_poisson_global_sigma_uses_positive_mean_for_dense_data():
    volume = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)

    sigma = estimate_poisson_global_sigma(volume)

    assert sigma == pytest.approx(np.sqrt(7.5))


def test_estimate_poisson_global_sigma_uses_trimmed_positive_mean_for_sparse_data():
    volume = np.zeros((10, 10), dtype=np.float32)
    volume[0, 0] = 1.0
    volume[1, 1] = 9.0
    volume[2, 2] = 100.0

    sigma = estimate_poisson_global_sigma(volume)

    assert sigma == pytest.approx(3.0)


def test_estimate_poisson_global_sigma_ignores_negative_and_nonfinite_values():
    volume = np.array([np.nan, -5.0, np.inf, 4.0, 9.0], dtype=np.float32)

    sigma = estimate_poisson_global_sigma(volume)

    assert sigma == pytest.approx(np.sqrt(6.5))


def test_estimate_poisson_global_sigma_falls_back_to_eps_when_no_positive_counts():
    volume = np.array([[0.0, -2.0], [0.0, 0.0]], dtype=np.float32)

    sigma = estimate_poisson_global_sigma(volume)

    assert sigma == pytest.approx(np.sqrt(1e-10))
