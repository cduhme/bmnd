from __future__ import annotations

from typing import Callable, cast

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from .enums import PoissonWienerGainMode, coerce_enum
from .transforms import apply_transform_nd
from .variance import (
    ExactPoissonCovarianceFactors,
    ExactPoissonGroupPlan,
    PoissonVarianceModel,
    get_exact_poisson_contribution_matrix,
    get_exact_poisson_covariance_factors,
    get_exact_poisson_risk_metrics,
)


_EXACT_POISSON_RISK_WORKSPACE_BYTES = 64 * 1024 * 1024


def compute_poisson_wiener_signal_power(
    reference_coefficients: NDArray[np.floating],
    noise_sigma: float | NDArray[np.floating],
    wiener_variance_scale: float,
    mode: PoissonWienerGainMode = PoissonWienerGainMode.VARIANCE_SCALED,
) -> NDArray[np.float32]:
    mode = coerce_enum(mode, PoissonWienerGainMode, "poisson_wiener_gain_mode")
    power_reference = np.abs(reference_coefficients) ** 2
    noise_variance = np.asarray(noise_sigma) ** 2
    effective_variance = wiener_variance_scale * noise_variance

    if mode is PoissonWienerGainMode.CLASSIC:
        signal_power = power_reference
    elif mode is PoissonWienerGainMode.VARIANCE_SCALED:
        signal_power = np.maximum(
            power_reference - effective_variance,
            0.0,
        )
    elif mode is PoissonWienerGainMode.NOISE_FLOOR:
        signal_power = np.maximum(
            power_reference - noise_variance,
            0.0,
        )
    else:
        raise ValueError(
            "Unknown Poisson Wiener gain mode: "
            f"{mode} (expected 'classic', 'noise-floor', or 'variance-scaled')."
        )

    return np.asarray(signal_power, dtype=np.float32)


def compute_wiener_gain(
    reference_coefficients: NDArray[np.floating],
    noise_sigma: float | NDArray[np.floating],
    wiener_variance_scale: float,
    eps: float = 1e-12,
) -> NDArray[np.float32]:
    power_reference = np.abs(reference_coefficients) ** 2
    noise_variance = np.asarray(noise_sigma) ** 2
    effective_variance = wiener_variance_scale * np.maximum(
        noise_variance,
        max(float(eps), 1e-8),
    )
    denominator = power_reference + effective_variance
    gain = np.divide(
        power_reference,
        denominator,
        out=np.zeros_like(power_reference, dtype=np.float32),
        where=denominator > 0.0,
    )
    return np.clip(gain, 0.0, 1.0).astype(np.float32, copy=False)


def compute_poisson_wiener_gain(
    reference_coefficients: NDArray[np.floating],
    noise_sigma: float | NDArray[np.floating],
    wiener_variance_scale: float,
    mode: PoissonWienerGainMode = PoissonWienerGainMode.VARIANCE_SCALED,
    eps: float = 1e-12,
) -> NDArray[np.float32]:
    noise_variance = np.asarray(noise_sigma) ** 2
    effective_variance = wiener_variance_scale * noise_variance
    signal_power = compute_poisson_wiener_signal_power(
        reference_coefficients,
        noise_sigma,
        wiener_variance_scale,
        mode=mode,
    )

    gain = signal_power / (signal_power + effective_variance + eps)
    return np.clip(gain, 0.0, 1.0).astype(np.float32, copy=False)


def compute_poisson_wiener_risk_from_precomputed(
    gain: NDArray[np.floating],
    signal_power: NDArray[np.floating],
    noise_sigma: float | NDArray[np.floating],
) -> NDArray[np.float32]:
    noise_variance = np.asarray(noise_sigma, dtype=np.float32) ** 2
    gain_arr = np.asarray(gain, dtype=np.float32)
    signal_power_arr = np.asarray(signal_power, dtype=np.float32)
    risk = (gain_arr**2) * noise_variance + ((1.0 - gain_arr) ** 2) * signal_power_arr
    return np.asarray(risk, dtype=np.float32)


def _poisson_wiener_coefficient_metric_diag(
    coefficient_shape: tuple[int, ...],
    inverse_transform: list[NDArray[np.floating] | Callable | None],
    w_patch: NDArray[np.floating],
) -> NDArray[np.float32]:
    window_sq = np.asarray(w_patch, dtype=np.float32).reshape(-1) ** 2
    if not any(callable(transform) for transform in inverse_transform):
        group_size = int(coefficient_shape[0])
        inverse_group = inverse_transform[0]
        group_matrix = (
            np.eye(group_size, dtype=np.float32)
            if inverse_group is None
            else np.asarray(inverse_group, dtype=np.float32)
        )
        group_metric_diag = np.sum(group_matrix**2, axis=0, dtype=np.float32)

        spatial_matrix = np.array([[1.0]], dtype=np.float32)
        for axis_size, transform in zip(
            coefficient_shape[1:],
            inverse_transform[1:],
            strict=False,
        ):
            matrix = (
                np.eye(axis_size, dtype=np.float32)
                if transform is None
                else np.asarray(transform, dtype=np.float32)
            )
            spatial_matrix = np.kron(spatial_matrix, matrix).astype(
                np.float32,
                copy=False,
            )
        spatial_metric_diag = np.sum(
            window_sq[:, None] * spatial_matrix**2,
            axis=0,
            dtype=np.float32,
        )
        return np.kron(group_metric_diag, spatial_metric_diag).astype(
            np.float32,
            copy=False,
        )

    coefficient_count = int(np.prod(coefficient_shape))
    output_weights = np.tile(window_sq, int(coefficient_shape[0]))
    metric_diag = np.empty(coefficient_count, dtype=np.float32)
    for coefficient_index in range(coefficient_count):
        basis = np.zeros(coefficient_shape, dtype=np.float32)
        basis.reshape(-1)[coefficient_index] = 1.0
        restored = np.asarray(
            apply_transform_nd(basis, inverse_transform),
            dtype=np.float32,
        ).reshape(-1)
        metric_diag[coefficient_index] = np.sum(
            output_weights * restored**2,
            dtype=np.float32,
        )
    return metric_diag


def _poisson_wiener_separable_risk_metrics(
    coefficient_shape: tuple[int, ...],
    inverse_transform: list[NDArray[np.floating] | Callable | None],
    w_patch: NDArray[np.floating],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    if any(callable(transform) for transform in inverse_transform):
        raise ValueError(
            "Mass-projected Poisson Wiener risk requires matrix-based transforms."
        )
    if any(
        transform is not None and np.iscomplexobj(transform)
        for transform in inverse_transform
    ):
        raise ValueError(
            "Mass-projected Poisson Wiener risk requires real matrix transforms."
        )

    group_size = int(coefficient_shape[0])
    inverse_group = inverse_transform[0]
    group_matrix = (
        np.eye(group_size, dtype=np.float32)
        if inverse_group is None
        else np.asarray(inverse_group, dtype=np.float32)
    )
    group_metric = np.asarray(group_matrix.T @ group_matrix, dtype=np.float32)

    spatial_matrix = np.array([[1.0]], dtype=np.float32)
    for axis_size, transform in zip(
        coefficient_shape[1:],
        inverse_transform[1:],
        strict=False,
    ):
        matrix = (
            np.eye(axis_size, dtype=np.float32)
            if transform is None
            else np.asarray(transform, dtype=np.float32)
        )
        spatial_matrix = np.kron(spatial_matrix, matrix).astype(
            np.float32,
            copy=False,
        )
    window_sq = np.asarray(w_patch, dtype=np.float32).reshape(-1) ** 2
    spatial_metric = np.asarray(
        spatial_matrix.T @ (window_sq[:, None] * spatial_matrix),
        dtype=np.float32,
    )
    coefficient_metric_diag = np.kron(
        np.diag(group_metric),
        np.diag(spatial_metric),
    ).astype(np.float32, copy=False)
    return group_metric, spatial_metric, coefficient_metric_diag


def _mass_projected_wiener_metric_energies(
    gain: NDArray[np.floating],
    mass_functional: NDArray[np.floating],
    group_metric: NDArray[np.floating],
    spatial_metric: NDArray[np.floating],
    coefficient_metric_diag: NDArray[np.floating],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if np.iscomplexobj(mass_functional):
        raise ValueError(
            "Mass-projected Poisson Wiener risk requires a real mass functional."
        )
    gain_arr = np.asarray(gain, dtype=np.float32)
    mass = np.asarray(mass_functional, dtype=np.float32)
    if mass.size != gain_arr.size:
        raise ValueError(
            "mass_functional and gain must have the same number of coefficients, "
            f"got {mass.shape} and {gain_arr.shape}."
        )
    mass = mass.reshape(gain_arr.shape)

    denominator = float(np.sum(mass * mass, dtype=np.float64))
    if denominator <= 0.0:
        raise ValueError("mass_functional must have nonzero energy.")
    group_size = int(gain_arr.shape[0])
    patch_volume = int(np.prod(gain_arr.shape[1:]))
    mass_group = mass.reshape(group_size, patch_volume)
    projection_direction = mass_group / denominator
    metric_projection = (
        np.asarray(group_metric, dtype=np.float32)
        @ projection_direction
        @ np.asarray(spatial_metric, dtype=np.float32).T
    )
    cross_metric = mass_group * metric_projection
    projection_energy = float(
        np.sum(projection_direction * metric_projection, dtype=np.float64)
    )

    gain_flat = gain_arr.reshape(-1)
    residual_gain = 1.0 - gain_flat
    metric_diag = np.asarray(coefficient_metric_diag, dtype=np.float32).reshape(-1)
    cross_flat = cross_metric.reshape(-1)
    mass_sq = (mass.reshape(-1) ** 2).astype(np.float32, copy=False)
    noise_energy = (
        gain_flat**2 * metric_diag
        + 2.0 * gain_flat * residual_gain * cross_flat
        + residual_gain**2 * mass_sq * projection_energy
    )
    bias_energy = residual_gain**2 * (
        metric_diag - 2.0 * cross_flat + mass_sq * projection_energy
    )
    return (
        np.maximum(noise_energy, 0.0).astype(np.float32, copy=False),
        np.maximum(bias_energy, 0.0).astype(np.float32, copy=False),
    )


def _apply_mass_projected_wiener_filter(
    contributions: NDArray[np.floating],
    gain_group: NDArray[np.floating],
    mass_functional: NDArray[np.floating],
    input_plane_count: int,
) -> NDArray[np.float32]:
    if np.iscomplexobj(mass_functional):
        raise ValueError(
            "Mass-projected Poisson Wiener risk requires a real mass functional."
        )
    gain_arr = np.asarray(gain_group, dtype=np.float32)
    mass = np.asarray(mass_functional, dtype=np.float32).reshape(gain_arr.shape)
    denominator = float(np.sum(mass * mass, dtype=np.float64))
    if denominator <= 0.0:
        raise ValueError("mass_functional must have nonzero energy.")

    plane_count = min(max(int(input_plane_count), 0), int(gain_arr.shape[0]))
    contribution_arr = np.asarray(contributions, dtype=np.float32)
    expected_shape = (plane_count, int(gain_arr.shape[1]))
    if contribution_arr.shape[1:] != expected_shape:
        raise ValueError(
            "Contribution shape does not match the selected coefficient planes: "
            f"got {contribution_arr.shape[1:]}, expected {expected_shape}."
        )

    filtered = np.zeros(
        (contribution_arr.shape[0],) + gain_arr.shape,
        dtype=np.float32,
    )
    filtered[:, :plane_count] = contribution_arr * gain_arr[None, :plane_count]
    rejected_mass = np.sum(
        contribution_arr
        * (mass[:plane_count] * (1.0 - gain_arr[:plane_count]))[None, :, :],
        axis=(1, 2),
        dtype=np.float32,
    )
    filtered += rejected_mass[:, None, None] * (mass / denominator)[None, :, :]
    return filtered


def compute_approximate_poisson_wiener_group_risk(
    gain: NDArray[np.floating],
    signal_power: NDArray[np.floating] | None,
    noise_sigma: float | NDArray[np.floating],
    inverse_transform: list[NDArray[np.floating] | Callable | None],
    w_patch: NDArray[np.floating],
    coefficient_metric_diag: NDArray[np.floating] | None = None,
    mass_functional: NDArray[np.floating] | None = None,
    group_metric: NDArray[np.floating] | None = None,
    spatial_metric: NDArray[np.floating] | None = None,
) -> float:
    if mass_functional is not None:
        if (
            group_metric is None
            or spatial_metric is None
            or coefficient_metric_diag is None
        ):
            group_metric, spatial_metric, coefficient_metric_diag = (
                _poisson_wiener_separable_risk_metrics(
                    tuple(int(size) for size in np.shape(gain)),
                    inverse_transform,
                    w_patch,
                )
            )
        assert group_metric is not None
        assert spatial_metric is not None
        assert coefficient_metric_diag is not None
        noise_energy, bias_energy = _mass_projected_wiener_metric_energies(
            gain,
            mass_functional,
            group_metric,
            spatial_metric,
            coefficient_metric_diag,
        )
        noise_variance = np.asarray(noise_sigma, dtype=np.float32).reshape(-1) ** 2
        risk = float(np.sum(noise_variance * noise_energy, dtype=np.float64))
        if signal_power is not None:
            signal_power_flat = np.asarray(signal_power, dtype=np.float32).reshape(-1)
            risk += float(
                np.sum(signal_power_flat * bias_energy, dtype=np.float64)
            )
        return risk

    if signal_power is None:
        risk = (
            np.abs(np.asarray(gain, dtype=np.float32)) ** 2
            * np.asarray(noise_sigma, dtype=np.float32) ** 2
        ).reshape(-1)
    else:
        risk = compute_poisson_wiener_risk_from_precomputed(
            gain,
            signal_power,
            noise_sigma,
        ).reshape(-1)
    metric_diag = (
        _poisson_wiener_coefficient_metric_diag(
            tuple(int(size) for size in np.shape(gain)),
            inverse_transform,
            w_patch,
        )
        if coefficient_metric_diag is None
        else np.asarray(coefficient_metric_diag, dtype=np.float32).reshape(-1)
    )
    if metric_diag.shape != risk.shape:
        raise ValueError(
            "Poisson Wiener coefficient metric shape does not match the risk shape: "
            f"got {metric_diag.shape}, expected {risk.shape}."
        )
    return float(np.sum(metric_diag * risk, dtype=np.float64))


def compute_poisson_wiener_risk(
    reference_coefficients: NDArray[np.floating],
    noise_sigma: float | NDArray[np.floating],
    wiener_variance_scale: float,
    mode: PoissonWienerGainMode = PoissonWienerGainMode.VARIANCE_SCALED,
    eps: float = 1e-12,
) -> NDArray[np.float32]:
    gain = compute_poisson_wiener_gain(
        reference_coefficients,
        noise_sigma,
        wiener_variance_scale,
        mode=mode,
        eps=eps,
    )
    signal_power = compute_poisson_wiener_signal_power(
        reference_coefficients,
        noise_sigma,
        wiener_variance_scale,
        mode=mode,
    )
    return compute_poisson_wiener_risk_from_precomputed(gain, signal_power, noise_sigma)


@njit(cache=True, nogil=True)
def _accumulate_exact_poisson_noise_risk_numba(
    contributions: NDArray[np.float32],
    union_variance: NDArray[np.float32],
    gain_group: NDArray[np.float32],
    group_metric: NDArray[np.float32],
    spatial_metric: NDArray[np.float32],
) -> np.float64:
    group_size = gain_group.shape[0]
    patch_vol = gain_group.shape[1]
    filtered = np.zeros((group_size, patch_vol), dtype=np.float32)
    spatial_filtered = np.zeros((group_size, patch_vol), dtype=np.float32)
    noise_risk = np.float64(0.0)

    for voxel_id in range(union_variance.shape[0]):
        contribution = contributions[voxel_id]

        for group_idx in range(group_size):
            base = group_idx * patch_vol
            for patch_idx in range(patch_vol):
                filtered[group_idx, patch_idx] = (
                    gain_group[group_idx, patch_idx]
                    * contribution[base + patch_idx]
                )

        for group_idx in range(group_size):
            for out_patch_idx in range(patch_vol):
                spatial_value = np.float32(0.0)
                for in_patch_idx in range(patch_vol):
                    spatial_value += (
                        filtered[group_idx, in_patch_idx]
                        * spatial_metric[in_patch_idx, out_patch_idx]
                    )
                spatial_filtered[group_idx, out_patch_idx] = spatial_value

        voxel_risk = np.float64(0.0)
        for left_group_idx in range(group_size):
            for right_group_idx in range(group_size):
                group_weight = group_metric[left_group_idx, right_group_idx]
                if group_weight == 0.0:
                    continue
                for patch_idx in range(patch_vol):
                    voxel_risk += (
                        group_weight
                        * spatial_filtered[left_group_idx, patch_idx]
                        * filtered[right_group_idx, patch_idx]
                    )

        noise_risk += union_variance[voxel_id] * voxel_risk

    return noise_risk


@njit(cache=True, parallel=True, nogil=True)
def _compute_exact_poisson_noise_risk_rows_numba(
    filtered_contributions: NDArray[np.float32],
    group_metric: NDArray[np.float32],
    spatial_metric: NDArray[np.float32],
) -> NDArray[np.float64]:
    voxel_count = filtered_contributions.shape[0]
    group_size = filtered_contributions.shape[1]
    patch_vol = filtered_contributions.shape[2]
    row_risk = np.empty(voxel_count, dtype=np.float64)

    for voxel_id in prange(voxel_count):
        spatial_filtered = np.empty((group_size, patch_vol), dtype=np.float32)
        filtered = filtered_contributions[voxel_id]

        for group_idx in range(group_size):
            for out_patch_idx in range(patch_vol):
                spatial_value = np.float32(0.0)
                for in_patch_idx in range(patch_vol):
                    spatial_value += (
                        filtered[group_idx, in_patch_idx]
                        * spatial_metric[in_patch_idx, out_patch_idx]
                    )
                spatial_filtered[group_idx, out_patch_idx] = spatial_value

        voxel_risk = np.float64(0.0)
        for left_group_idx in range(group_size):
            for right_group_idx in range(group_size):
                group_weight = group_metric[left_group_idx, right_group_idx]
                if group_weight == 0.0:
                    continue
                for patch_idx in range(patch_vol):
                    voxel_risk += (
                        group_weight
                        * spatial_filtered[left_group_idx, patch_idx]
                        * filtered[right_group_idx, patch_idx]
                    )
        row_risk[voxel_id] = voxel_risk
    return row_risk


def _compute_exact_poisson_noise_quadratics_diagonal_group_numpy(
    filtered_contributions: NDArray[np.float32],
    spatial_metric: NDArray[np.float32],
) -> NDArray[np.float64]:
    voxel_count = filtered_contributions.shape[0]
    group_size = filtered_contributions.shape[1]
    patch_vol = filtered_contributions.shape[2]
    weighted_flat = filtered_contributions.reshape(-1, patch_vol)

    projected_flat = np.empty_like(weighted_flat)
    np.matmul(weighted_flat, spatial_metric, out=projected_flat)

    quadratic_flat = np.einsum(
        "ij,ij->i",
        weighted_flat,
        projected_flat,
        dtype=np.float64,
    )
    return quadratic_flat.reshape(voxel_count, group_size)


def _compute_exact_poisson_noise_risk_rows_diagonal_group_numpy(
    filtered_contributions: NDArray[np.float32],
    group_metric_diag: NDArray[np.float32],
    spatial_metric: NDArray[np.float32],
) -> NDArray[np.float64]:
    quadratic = _compute_exact_poisson_noise_quadratics_diagonal_group_numpy(
        filtered_contributions,
        spatial_metric,
    )
    return np.asarray(quadratic @ group_metric_diag, dtype=np.float64)


def _compute_exact_mass_projected_noise_risk_rows_diagonal_group(
    contributions: NDArray[np.float32],
    gain_group: NDArray[np.float32],
    mass_functional: NDArray[np.floating],
    spatial_metric: NDArray[np.float32],
    group_metric_diag: NDArray[np.float32],
) -> NDArray[np.float64]:
    if np.iscomplexobj(mass_functional):
        raise ValueError(
            "Mass-projected Poisson Wiener risk requires a real mass functional."
        )
    exact_plane_count = contributions.shape[1]
    mass = np.asarray(mass_functional, dtype=np.float32).reshape(gain_group.shape)
    denominator = float(np.sum(mass * mass, dtype=np.float64))
    if denominator <= 0.0:
        raise ValueError("mass_functional must have nonzero energy.")

    projection_direction = mass / denominator
    rejected_mass = np.sum(
        contributions
        * (
            mass[:exact_plane_count]
            * (np.float32(1.0) - gain_group[:exact_plane_count])
        )[None, :, :],
        axis=(1, 2),
        dtype=np.float32,
    )
    filtered_exact = np.asarray(
        contributions * gain_group[None, :exact_plane_count, :],
        dtype=np.float32,
    )
    filtered_exact += (
        rejected_mass[:, None, None]
        * projection_direction[None, :exact_plane_count, :]
    )
    quadratic = np.zeros(
        (contributions.shape[0], gain_group.shape[0]),
        dtype=np.float64,
    )
    quadratic[:, :exact_plane_count] = (
        _compute_exact_poisson_noise_quadratics_diagonal_group_numpy(
            filtered_exact,
            spatial_metric,
        )
    )

    tail_indices = np.flatnonzero(
        np.any(projection_direction[exact_plane_count:] != 0.0, axis=1)
    ) + exact_plane_count
    if tail_indices.size:
        filtered_tail = np.asarray(
            rejected_mass[:, None, None]
            * projection_direction[None, tail_indices, :],
            dtype=np.float32,
        )
        quadratic[:, tail_indices] = (
            _compute_exact_poisson_noise_quadratics_diagonal_group_numpy(
                filtered_tail,
                spatial_metric,
            )
        )
    return np.asarray(quadratic @ group_metric_diag, dtype=np.float64)


def _accumulate_exact_poisson_noise_risk_diagonal_group_numpy(
    contributions_group: NDArray[np.float32],
    union_variance: NDArray[np.float32],
    gain_group: NDArray[np.float32],
    group_metric_diag: NDArray[np.float32],
    spatial_metric: NDArray[np.float32],
) -> float:
    row_weighted_risk = _compute_exact_poisson_noise_risk_rows_diagonal_group_numpy(
        np.asarray(contributions_group * gain_group[None, :, :], dtype=np.float32),
        group_metric_diag,
        spatial_metric,
    )
    return float(np.dot(row_weighted_risk, union_variance))


def _exact_poisson_risk_chunk_size(
    group_size: int,
    patch_volume: int,
    mass_projected: bool = False,
) -> int:
    array_count = 4 if mass_projected else 3
    bytes_per_voxel = 4 * (array_count * group_size * patch_volume + group_size)
    return max(1, _EXACT_POISSON_RISK_WORKSPACE_BYTES // bytes_per_voxel)


@njit(cache=True, nogil=True)
def _sum_exact_poisson_voxel_risk_numba(
    row_risk: NDArray[np.float64],
    union_variance: NDArray[np.float32],
) -> np.float64:
    noise_risk = np.float64(0.0)
    for voxel_id in range(row_risk.shape[0]):
        noise_risk += union_variance[voxel_id] * row_risk[voxel_id]
    return noise_risk


def _accumulate_exact_poisson_noise_risk_from_plan(
    plan: ExactPoissonGroupPlan,
    gain_group: NDArray[np.float32],
    group_metric: NDArray[np.float32],
    spatial_metric: NDArray[np.float32],
    group_metric_diag: NDArray[np.float32],
    group_metric_diagonal: bool,
    exact_plane_count: int,
    mass_functional: NDArray[np.floating] | None = None,
) -> float:
    voxel_count = int(plan.union_variance.shape[0])
    row_risk = np.empty(voxel_count, dtype=np.float64)
    chunk_size = _exact_poisson_risk_chunk_size(
        plan.group_size,
        plan.patch_volume,
        mass_projected=mass_functional is not None,
    )

    for voxel_start in range(0, voxel_count, chunk_size):
        count = min(chunk_size, voxel_count - voxel_start)
        contributions = plan.get_contribution_chunk(
            voxel_start,
            count,
            output_plane_count=exact_plane_count,
        )
        if mass_functional is not None and group_metric_diagonal:
            row_risk[voxel_start : voxel_start + count] = (
                _compute_exact_mass_projected_noise_risk_rows_diagonal_group(
                    contributions,
                    gain_group,
                    mass_functional,
                    spatial_metric,
                    group_metric_diag,
                )
            )
            continue
        if mass_functional is None:
            np.multiply(
                contributions,
                gain_group[None, :exact_plane_count, :],
                out=contributions,
            )
            filtered_contributions = contributions
            active_group_metric = group_metric[:exact_plane_count, :exact_plane_count]
            active_group_metric_diag = group_metric_diag[:exact_plane_count]
        else:
            filtered_contributions = _apply_mass_projected_wiener_filter(
                contributions,
                gain_group,
                mass_functional,
                exact_plane_count,
            )
            active_group_metric = group_metric
            active_group_metric_diag = group_metric_diag
        if group_metric_diagonal:
            row_risk[voxel_start : voxel_start + count] = (
                _compute_exact_poisson_noise_risk_rows_diagonal_group_numpy(
                    filtered_contributions,
                    active_group_metric_diag,
                    spatial_metric,
                )
            )
        else:
            row_risk[voxel_start : voxel_start + count] = (
                _compute_exact_poisson_noise_risk_rows_numba(
                    filtered_contributions,
                    active_group_metric,
                    spatial_metric,
                )
            )
    if group_metric_diagonal:
        return float(np.dot(row_risk, plan.union_variance))
    return float(_sum_exact_poisson_voxel_risk_numba(row_risk, plan.union_variance))


def compute_exact_poisson_wiener_group_risk_from_plan(
    gain: NDArray[np.floating],
    signal_power: NDArray[np.floating] | None,
    plan: ExactPoissonGroupPlan,
    inverse_transform: list[NDArray[np.floating] | Callable | None],
    w_patch: NDArray[np.floating],
    exact_plane_count: int | None = None,
    noise_sigma: float | NDArray[np.floating] | None = None,
    mass_functional: NDArray[np.floating] | None = None,
) -> float:
    inverse_transform_list = cast(
        list[NDArray[np.floating] | Callable | None],
        list(inverse_transform),
    )
    window = np.asarray(w_patch, dtype=np.float32)
    try:
        gain_group = np.asarray(gain, dtype=np.float32).reshape(
            int(plan.group_shape[0]),
            -1,
        )
        plane_count = (
            plan.group_size
            if exact_plane_count is None
            else min(max(int(exact_plane_count), 0), plan.group_size)
        )
        if plane_count == 0:
            if noise_sigma is None:
                raise ValueError("noise_sigma is required when no exact planes are used.")
            return compute_approximate_poisson_wiener_group_risk(
                gain,
                signal_power,
                noise_sigma,
                inverse_transform_list,
                window,
                mass_functional=mass_functional,
            )
        signal_power_flat = (
            None
            if signal_power is None
            else np.asarray(signal_power, dtype=np.float32).reshape(-1)
        )
        (
            group_metric,
            spatial_metric,
            coefficient_metric_diag,
            group_metric_diag,
            group_metric_diagonal,
        ) = plan.get_prepared_risk_metrics(inverse_transform_list, window)
        noise_risk = _accumulate_exact_poisson_noise_risk_from_plan(
            plan,
            gain_group,
            group_metric,
            spatial_metric,
            group_metric_diag,
            group_metric_diagonal,
            plane_count,
            mass_functional=mass_functional,
        )
        if plane_count < plan.group_size:
            if noise_sigma is None:
                raise ValueError("noise_sigma is required for partial exact Poisson risk.")
            noise_variance = np.asarray(noise_sigma, dtype=np.float32).reshape(
                plan.group_size,
                -1,
            ) ** 2
            if mass_functional is None:
                metric_diag_group = coefficient_metric_diag.reshape(plan.group_size, -1)
                noise_risk += float(
                    np.sum(
                        metric_diag_group[plane_count:]
                        * gain_group[plane_count:] ** 2
                        * noise_variance[plane_count:],
                        dtype=np.float64,
                    )
                )
        if mass_functional is None:
            if signal_power_flat is None:
                bias_risk = 0.0
            else:
                bias_power = ((1.0 - gain_group.reshape(-1)) ** 2) * signal_power_flat
                bias_risk = float(
                    np.sum(coefficient_metric_diag * bias_power, dtype=np.float64)
                )
        else:
            noise_energy, bias_energy = _mass_projected_wiener_metric_energies(
                gain_group,
                mass_functional,
                group_metric,
                spatial_metric,
                coefficient_metric_diag,
            )
            if plane_count < plan.group_size:
                noise_risk += float(
                    np.sum(
                        noise_energy.reshape(plan.group_size, -1)[plane_count:]
                        * noise_variance[plane_count:],
                        dtype=np.float64,
                    )
                )
            bias_risk = (
                0.0
                if signal_power_flat is None
                else float(
                    np.sum(bias_energy * signal_power_flat, dtype=np.float64)
                )
            )
        return float(noise_risk) + bias_risk
    finally:
        plan.release_prepared_risk_contributions()


def compute_covariance_aware_group_risk_from_plan(
    attenuation: NDArray[np.floating],
    plan: ExactPoissonGroupPlan,
    inverse_transform: list[NDArray[np.floating] | Callable | None],
    w_patch: NDArray[np.floating],
    *,
    exact_plane_count: int,
    noise_sigma: float | NDArray[np.floating],
    signal_power: NDArray[np.floating] | None = None,
    mass_functional: NDArray[np.floating] | None = None,
) -> float:
    return compute_exact_poisson_wiener_group_risk_from_plan(
        attenuation,
        signal_power,
        plan,
        inverse_transform,
        w_patch,
        exact_plane_count=exact_plane_count,
        noise_sigma=noise_sigma,
        mass_functional=mass_functional,
    )


def compute_exact_poisson_diagonal_filter_group_risk_from_plan(
    attenuation: NDArray[np.floating],
    plan: ExactPoissonGroupPlan,
    inverse_transform: list[NDArray[np.floating] | Callable | None],
    w_patch: NDArray[np.floating],
    *,
    exact_plane_count: int,
    noise_sigma: float | NDArray[np.floating],
    signal_power: NDArray[np.floating] | None = None,
    mass_functional: NDArray[np.floating] | None = None,
) -> float:
    return compute_covariance_aware_group_risk_from_plan(
        attenuation,
        plan,
        inverse_transform,
        w_patch,
        exact_plane_count=exact_plane_count,
        noise_sigma=noise_sigma,
        signal_power=signal_power,
        mass_functional=mass_functional,
    )


def compute_exact_poisson_wiener_group_risk(
    gain: NDArray[np.floating],
    signal_power: NDArray[np.floating],
    variance_model: PoissonVarianceModel,
    selected_abs_positions: NDArray[np.int64],
    selected_shifted_positions: NDArray[np.int64],
    group_shape: tuple[int, ...],
    forward_transform: list[NDArray[np.floating] | Callable | None],
    inverse_transform: list[NDArray[np.floating] | Callable | None],
    w_patch: NDArray[np.floating],
    covariance_factors: ExactPoissonCovarianceFactors | None = None,
    contributions: NDArray[np.float32] | None = None,
    plan: ExactPoissonGroupPlan | None = None,
    mass_functional: NDArray[np.floating] | None = None,
) -> float:
    if plan is not None:
        return compute_exact_poisson_wiener_group_risk_from_plan(
            gain,
            signal_power,
            plan,
            inverse_transform,
            w_patch,
            mass_functional=mass_functional,
        )

    forward_transform_list = cast(
        list[NDArray[np.floating] | Callable | None],
        list(forward_transform),
    )
    inverse_transform_list = cast(
        list[NDArray[np.floating] | Callable | None],
        list(inverse_transform),
    )
    group_shape = tuple(int(size) for size in group_shape)
    if covariance_factors is None:
        covariance_factors = get_exact_poisson_covariance_factors(
            variance_model,
            np.asarray(selected_abs_positions, dtype=np.int64),
            np.asarray(selected_shifted_positions, dtype=np.int64),
            group_shape,
            forward_transform_list,
        )
    union_variance = covariance_factors.union_variance
    if contributions is None:
        contributions = get_exact_poisson_contribution_matrix(
            variance_model,
            np.asarray(selected_shifted_positions, dtype=np.int64),
            group_shape,
            forward_transform_list,
            covariance_factors=covariance_factors,
        )
    group_metric, spatial_metric, coefficient_metric_diag = get_exact_poisson_risk_metrics(
        variance_model,
        group_shape,
        inverse_transform_list,
        np.asarray(w_patch, dtype=np.float32),
    )
    grouped_contributions = np.asarray(
        contributions,
        dtype=np.float32,
    ).reshape(contributions.shape[0], int(group_shape[0]), -1)

    gain_group = np.asarray(gain, dtype=np.float32).reshape(int(group_shape[0]), -1)
    signal_power_flat = np.asarray(signal_power, dtype=np.float32).reshape(-1)

    group_metric_diag = np.diag(group_metric).astype(np.float32, copy=False)
    group_metric_offdiag = group_metric - np.diag(group_metric_diag)
    group_metric_diagonal = bool(np.all(np.abs(group_metric_offdiag) <= 1e-6))
    if mass_functional is not None:
        filtered_contributions = _apply_mass_projected_wiener_filter(
            grouped_contributions,
            gain_group,
            mass_functional,
            int(group_shape[0]),
        )
        if group_metric_diagonal:
            row_risk = _compute_exact_poisson_noise_risk_rows_diagonal_group_numpy(
                filtered_contributions,
                group_metric_diag,
                spatial_metric,
            )
        else:
            row_risk = _compute_exact_poisson_noise_risk_rows_numba(
                filtered_contributions,
                group_metric,
                spatial_metric,
            )
        noise_risk = float(np.dot(row_risk, union_variance))
    elif group_metric_diagonal:
        noise_risk = _accumulate_exact_poisson_noise_risk_diagonal_group_numpy(
            grouped_contributions,
            union_variance,
            gain_group,
            group_metric_diag,
            spatial_metric,
        )
    else:
        noise_risk = _accumulate_exact_poisson_noise_risk_numba(
            contributions,
            union_variance,
            gain_group,
            group_metric,
            spatial_metric,
        )
    if mass_functional is None:
        bias_power = ((1.0 - gain_group.reshape(-1)) ** 2) * signal_power_flat
        bias_risk = float(
            np.sum(coefficient_metric_diag * bias_power, dtype=np.float64)
        )
    else:
        _, bias_energy = _mass_projected_wiener_metric_energies(
            gain_group,
            mass_functional,
            group_metric,
            spatial_metric,
            coefficient_metric_diag,
        )
        bias_risk = float(np.sum(bias_energy * signal_power_flat, dtype=np.float64))
    return float(noise_risk) + bias_risk
