from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .enums import (
    AggregationWeightDomain,
    AggregationWeightModel,
    AggregationWeightScope,
    coerce_enum,
)

TransformEntry = NDArray[np.generic] | Callable | None
_REAL_AGGREGATION_WEIGHT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))
_WEIGHT_MODEL_CLASSIC = 0
_WEIGHT_MODEL_VARIANCE = 1
_WEIGHT_MODEL_RISK = 2
_AGGREGATION_WEIGHT_MODEL_CODES = {
    AggregationWeightModel.CLASSIC: _WEIGHT_MODEL_CLASSIC,
    AggregationWeightModel.VARIANCE: _WEIGHT_MODEL_VARIANCE,
    AggregationWeightModel.RISK: _WEIGHT_MODEL_RISK,
}


def _get_aggregation_weight_model_code(model: AggregationWeightModel) -> int:
    try:
        return _AGGREGATION_WEIGHT_MODEL_CODES[model]
    except KeyError as exc:
        expected = "', '".join(
            weight_model.value for weight_model in _AGGREGATION_WEIGHT_MODEL_CODES
        )
        raise ValueError(
            f"Unknown weight model: {model!r} (expected '{expected}')."
        ) from exc


@dataclass(frozen=True)
class SynthesisRiskMetricFactors:
    """Separable factors used to evaluate synthesized-patch risk."""

    inverse_group: NDArray[np.float64]
    group_mixing_energy: NDArray[np.float64]
    group_metric: NDArray[np.float64]
    spatial_metric: NDArray[np.float64]
    spatial_metric_diag: NDArray[np.float64]

    @property
    def coefficient_metric_diag(self) -> NDArray[np.float64]:
        return np.kron(
            np.diag(self.group_metric),
            self.spatial_metric_diag,
        )


def _as_real_matrix(
    transform: TransformEntry,
    size: int,
    axis: int,
) -> NDArray[np.float64]:
    if transform is None:
        return np.eye(size, dtype=np.float64)
    if callable(transform):
        raise ValueError("Aggregation risk metrics require matrix-based transforms.")
    matrix = np.asarray(transform)
    if np.iscomplexobj(matrix):
        raise ValueError("Aggregation risk metrics require real matrix transforms.")
    if matrix.shape != (size, size):
        raise ValueError(
            f"Inverse transform on axis {axis} has shape {matrix.shape}, "
            f"expected {(size, size)}."
        )
    return np.asarray(matrix, dtype=np.float64)


def get_synthesis_risk_metric_factors(
    coefficient_shape: tuple[int, ...],
    inverse_transform: Sequence[TransformEntry],
    w_patch: NDArray[np.floating],
) -> SynthesisRiskMetricFactors:
    shape = tuple(int(size) for size in coefficient_shape)
    if len(shape) < 2 or any(size <= 0 for size in shape):
        raise ValueError(f"coefficient_shape must describe a nonempty group, got {shape}.")
    if len(inverse_transform) != len(shape):
        raise ValueError(
            "inverse_transform must contain one entry per coefficient axis, "
            f"got {len(inverse_transform)} entries for shape {shape}."
        )

    inverse_group = _as_real_matrix(inverse_transform[0], shape[0], 0)
    spatial_synthesis = np.array([[1.0]], dtype=np.float64)
    for axis, (size, transform) in enumerate(
        zip(shape[1:], inverse_transform[1:], strict=True),
        start=1,
    ):
        spatial_synthesis = np.kron(
            spatial_synthesis,
            _as_real_matrix(transform, size, axis),
        )

    window = np.asarray(w_patch, dtype=np.float64).reshape(-1)
    patch_volume = int(np.prod(shape[1:]))
    if window.size != patch_volume:
        raise ValueError(
            f"w_patch has {window.size} entries, expected {patch_volume} for shape {shape}."
        )
    if not np.all(np.isfinite(window)):
        raise ValueError("w_patch must contain only finite values.")

    spatial_metric = spatial_synthesis.T @ ((window**2)[:, None] * spatial_synthesis)
    return SynthesisRiskMetricFactors(
        inverse_group=inverse_group,
        group_mixing_energy=np.abs(inverse_group) ** 2,
        group_metric=inverse_group.T @ inverse_group,
        spatial_metric=spatial_metric,
        spatial_metric_diag=np.diag(spatial_metric).copy(),
    )


def _diagonal_filter_error(
    attenuation: NDArray[np.floating],
    model: AggregationWeightModel,
    noise_variance: NDArray[np.floating] | None,
    signal_power: NDArray[np.floating] | None,
) -> NDArray[np.float64]:
    model = coerce_enum(model, AggregationWeightModel, "weight_model")
    attenuation_arr = np.asarray(attenuation)
    if np.iscomplexobj(attenuation_arr):
        attenuation_energy = np.abs(attenuation_arr.astype(np.complex128)) ** 2
    else:
        attenuation_arr = np.asarray(attenuation_arr, dtype=np.float64)
        attenuation_energy = attenuation_arr**2

    if model is AggregationWeightModel.CLASSIC:
        error = attenuation_energy
    else:
        if noise_variance is None:
            raise ValueError(f"{model} weighting requires coefficient noise variance.")
        variance = np.asarray(noise_variance, dtype=np.float64)
        if variance.shape != attenuation_arr.shape:
            raise ValueError(
                "noise_variance and attenuation must have the same shape, "
                f"got {variance.shape} and {attenuation_arr.shape}."
            )
        error = attenuation_energy
        np.multiply(error, variance, out=error)
        if model is AggregationWeightModel.RISK:
            if signal_power is None:
                raise ValueError("risk weighting requires coefficient signal power.")
            signal = np.asarray(signal_power, dtype=np.float64)
            if signal.shape != attenuation_arr.shape:
                raise ValueError(
                    "signal_power and attenuation must have the same shape, "
                    f"got {signal.shape} and {attenuation_arr.shape}."
                )
            if np.iscomplexobj(attenuation_arr):
                rejected_error = np.abs(1.0 - attenuation_arr) ** 2
            else:
                rejected_error = np.subtract(1.0, attenuation_arr)
                np.square(rejected_error, out=rejected_error)
            np.multiply(rejected_error, signal, out=rejected_error)
            np.add(error, rejected_error, out=error)
        elif model is not AggregationWeightModel.VARIANCE:
            raise ValueError(
                f"Unknown weight model: {model!r} "
                "(expected 'classic', 'variance', or 'risk')."
            )

    error = np.asarray(error, dtype=np.float64)
    if not np.all(np.isfinite(error)) or np.any(error < 0.0):
        raise ValueError("Aggregation risks must be finite and non-negative.")
    return error


def compute_diagonal_filter_group_risk(
    attenuation: NDArray[np.floating],
    *,
    model: AggregationWeightModel,
    domain: AggregationWeightDomain,
    noise_variance: NDArray[np.floating] | None = None,
    signal_power: NDArray[np.floating] | None = None,
    metric_factors: SynthesisRiskMetricFactors | None = None,
) -> float:
    model = coerce_enum(model, AggregationWeightModel, "weight_model")
    domain = coerce_enum(domain, AggregationWeightDomain, "weight_domain")
    error = _diagonal_filter_error(attenuation, model, noise_variance, signal_power)
    if domain is AggregationWeightDomain.COEFFICIENT:
        return float(np.sum(error, dtype=np.float64))
    if domain is not AggregationWeightDomain.WINDOWED_SYNTHESIS:
        raise ValueError(
            f"Unknown weight domain: {domain!r} "
            "(expected 'coefficient' or 'windowed_synthesis')."
        )
    if metric_factors is None:
        raise ValueError("windowed_synthesis weighting requires synthesis metric factors.")

    group_size = int(error.shape[0])
    error_group = error.reshape(group_size, -1)
    if error_group.shape[1] != metric_factors.spatial_metric_diag.size:
        raise ValueError("Synthesis metric factors do not match the attenuation shape.")
    risk = np.sum(
        error_group
        * np.diag(metric_factors.group_metric)[:, None]
        * metric_factors.spatial_metric_diag[None, :],
        dtype=np.float64,
    )
    return float(risk)


@njit(cache=True, nogil=True)
def _compute_real_diagonal_filter_patch_risks_numba(
    attenuation_group: NDArray[np.floating],
    noise_variance_group: NDArray[np.floating],
    signal_power_group: NDArray[np.floating],
    group_mixing_energy: NDArray[np.float64],
    spatial_metric_diag: NDArray[np.float64],
    model_code: int,
    windowed_synthesis: bool,
) -> NDArray[np.float64]:
    group_size, patch_volume = attenuation_group.shape
    plane_risk = np.empty(group_size, dtype=np.float64)
    for group_index in range(group_size):
        risk = np.float64(0.0)
        for coefficient_index in range(patch_volume):
            attenuation = np.float64(
                attenuation_group[group_index, coefficient_index]
            )
            error = attenuation * attenuation
            if model_code != _WEIGHT_MODEL_CLASSIC:
                error *= np.float64(
                    noise_variance_group[group_index, coefficient_index]
                )
            if model_code == _WEIGHT_MODEL_RISK:
                rejected = np.float64(1.0) - attenuation
                error += (
                    rejected
                    * rejected
                    * np.float64(
                        signal_power_group[group_index, coefficient_index]
                    )
                )
            if windowed_synthesis:
                error *= spatial_metric_diag[coefficient_index]
            risk += error
        plane_risk[group_index] = risk

    risks = np.empty(group_size, dtype=np.float64)
    for output_index in range(group_size):
        risk = np.float64(0.0)
        for input_index in range(group_size):
            risk += (
                group_mixing_energy[output_index, input_index]
                * plane_risk[input_index]
            )
        risks[output_index] = risk
    return risks


def compute_diagonal_filter_patch_risks(
    attenuation: NDArray[np.floating],
    *,
    model: AggregationWeightModel,
    domain: AggregationWeightDomain,
    metric_factors: SynthesisRiskMetricFactors,
    noise_variance: NDArray[np.floating] | None = None,
    signal_power: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    attenuation_arr = np.asarray(attenuation)
    group_size = int(attenuation_arr.shape[0])
    attenuation_group = attenuation_arr.reshape(group_size, -1)
    patch_volume = attenuation_group.shape[1]
    if patch_volume != metric_factors.spatial_metric_diag.size:
        raise ValueError("Synthesis metric factors do not match the attenuation shape.")

    model = coerce_enum(model, AggregationWeightModel, "weight_model")
    domain = coerce_enum(domain, AggregationWeightDomain, "weight_domain")
    model_code = _get_aggregation_weight_model_code(model)
    if domain not in {
        AggregationWeightDomain.COEFFICIENT,
        AggregationWeightDomain.WINDOWED_SYNTHESIS,
    }:
        raise ValueError(
            f"Unknown weight domain: {domain!r} "
            "(expected 'coefficient' or 'windowed_synthesis')."
        )

    variance_arr = attenuation_arr
    if model_code != _WEIGHT_MODEL_CLASSIC:
        if noise_variance is None:
            raise ValueError(f"{model} weighting requires coefficient noise variance.")
        variance_arr = np.asarray(noise_variance)
        if variance_arr.shape != attenuation_arr.shape:
            raise ValueError(
                "noise_variance and attenuation must have the same shape, "
                f"got {variance_arr.shape} and {attenuation_arr.shape}."
            )
    signal_arr = attenuation_arr
    if model_code == _WEIGHT_MODEL_RISK:
        if signal_power is None:
            raise ValueError("risk weighting requires coefficient signal power.")
        signal_arr = np.asarray(signal_power)
        if signal_arr.shape != attenuation_arr.shape:
            raise ValueError(
                "signal_power and attenuation must have the same shape, "
                f"got {signal_arr.shape} and {attenuation_arr.shape}."
            )

    use_fast_path = not np.iscomplexobj(attenuation_arr) and np.all(
        np.isfinite(attenuation_arr)
    )
    use_fast_path = (
        use_fast_path and attenuation_arr.dtype in _REAL_AGGREGATION_WEIGHT_DTYPES
    )
    if model_code != _WEIGHT_MODEL_CLASSIC:
        use_fast_path = (
            use_fast_path
            and not np.iscomplexobj(variance_arr)
            and variance_arr.dtype in _REAL_AGGREGATION_WEIGHT_DTYPES
            and np.all(np.isfinite(variance_arr))
            and np.all(variance_arr >= 0.0)
        )
    if model_code == _WEIGHT_MODEL_RISK:
        use_fast_path = (
            use_fast_path
            and not np.iscomplexobj(signal_arr)
            and signal_arr.dtype in _REAL_AGGREGATION_WEIGHT_DTYPES
            and np.all(np.isfinite(signal_arr))
            and np.all(signal_arr >= 0.0)
        )
    if use_fast_path:
        risks = _compute_real_diagonal_filter_patch_risks_numba(
            attenuation_group,
            variance_arr.reshape(group_size, -1),
            signal_arr.reshape(group_size, -1),
            metric_factors.group_mixing_energy,
            metric_factors.spatial_metric_diag,
            model_code,
            domain is AggregationWeightDomain.WINDOWED_SYNTHESIS,
        )
        if not np.all(np.isfinite(risks)) or np.any(risks < 0.0):
            raise ValueError("Aggregation risks must be finite and non-negative.")
        return risks

    error = _diagonal_filter_error(attenuation, model, noise_variance, signal_power)
    error_group = error.reshape(group_size, -1)

    if domain is AggregationWeightDomain.COEFFICIENT:
        plane_risk = np.sum(error_group, axis=1, dtype=np.float64)
    elif domain is AggregationWeightDomain.WINDOWED_SYNTHESIS:
        plane_risk = np.sum(
            error_group * metric_factors.spatial_metric_diag[None, :],
            axis=1,
            dtype=np.float64,
        )
    risks = metric_factors.group_mixing_energy @ plane_risk
    if not np.all(np.isfinite(risks)) or np.any(risks < 0.0):
        raise ValueError("Aggregation risks must be finite and non-negative.")
    return np.asarray(risks, dtype=np.float64)


def _mass_projected_metric_energies(
    attenuation_flat: NDArray[np.float64],
    attenuation_sq: NDArray[np.float64],
    rejected: NDArray[np.float64],
    rejected_sq: NDArray[np.float64],
    mass_group: NDArray[np.float64],
    mass_sq: NDArray[np.float64],
    projection_direction: NDArray[np.float64],
    group_metric: NDArray[np.floating],
    spatial_metric: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    group_metric_arr = np.asarray(group_metric, dtype=np.float64)
    spatial_metric_arr = np.asarray(spatial_metric, dtype=np.float64)
    metric_projection = group_metric_arr @ projection_direction @ spatial_metric_arr.T
    cross_metric = mass_group * metric_projection
    projection_energy = float(
        np.sum(projection_direction * metric_projection, dtype=np.float64)
    )

    metric_diag = np.kron(
        np.diag(group_metric_arr),
        np.diag(spatial_metric_arr),
    )
    cross_flat = cross_metric.reshape(-1)
    noise_energy = (
        attenuation_sq * metric_diag
        + 2.0 * attenuation_flat * rejected * cross_flat
        + rejected_sq * mass_sq * projection_energy
    )
    bias_energy = rejected_sq * (
        metric_diag - 2.0 * cross_flat + mass_sq * projection_energy
    )
    return np.maximum(noise_energy, 0.0), np.maximum(bias_energy, 0.0)


@njit(cache=True, nogil=True)
def _compute_mass_projected_patch_risks_numba(
    attenuation_group: NDArray[np.float64],
    mass_group: NDArray[np.float64],
    projection_direction: NDArray[np.float64],
    inverse_group: NDArray[np.float64],
    spatial_metric: NDArray[np.float64],
    noise_variance_group: NDArray[np.float64],
    signal_power_group: NDArray[np.float64],
    include_signal_bias: bool,
    coefficient_domain: bool,
) -> NDArray[np.float64]:
    group_size, patch_volume = attenuation_group.shape
    spatial_metric_diag = np.empty(patch_volume, dtype=np.float64)
    projected_direction = np.empty(patch_volume, dtype=np.float64)
    projected_spatial = np.empty(patch_volume, dtype=np.float64)
    risks = np.empty(group_size, dtype=np.float64)
    for patch_index in range(patch_volume):
        spatial_metric_diag[patch_index] = (
            np.float64(1.0)
            if coefficient_domain
            else spatial_metric[patch_index, patch_index]
        )

    for output_index in range(group_size):
        for patch_index in range(patch_volume):
            value = np.float64(0.0)
            for group_index in range(group_size):
                value += (
                    inverse_group[output_index, group_index]
                    * projection_direction[group_index, patch_index]
                )
            projected_direction[patch_index] = value

        if coefficient_domain:
            for patch_index in range(patch_volume):
                projected_spatial[patch_index] = projected_direction[patch_index]
        else:
            for output_patch_index in range(patch_volume):
                value = np.float64(0.0)
                for input_patch_index in range(patch_volume):
                    value += (
                        projected_direction[input_patch_index]
                        * spatial_metric[output_patch_index, input_patch_index]
                    )
                projected_spatial[output_patch_index] = value

        projection_energy = np.float64(0.0)
        for patch_index in range(patch_volume):
            projection_energy += (
                projected_direction[patch_index] * projected_spatial[patch_index]
            )

        risk = np.float64(0.0)
        for group_index in range(group_size):
            synthesis = inverse_group[output_index, group_index]
            synthesis_energy = synthesis * synthesis
            for patch_index in range(patch_volume):
                attenuation = attenuation_group[group_index, patch_index]
                rejected = np.float64(1.0) - attenuation
                mass = mass_group[group_index, patch_index]
                mass_energy = mass * mass
                cross_metric = (
                    mass * synthesis * projected_spatial[patch_index]
                )
                metric_diag = (
                    synthesis_energy * spatial_metric_diag[patch_index]
                )
                noise_energy = (
                    attenuation * attenuation * metric_diag
                    + np.float64(2.0)
                    * attenuation
                    * rejected
                    * cross_metric
                    + rejected * rejected * mass_energy * projection_energy
                )
                risk += noise_variance_group[group_index, patch_index] * max(
                    noise_energy,
                    np.float64(0.0),
                )
                if include_signal_bias:
                    bias_energy = rejected * rejected * (
                        metric_diag
                        - np.float64(2.0) * cross_metric
                        + mass_energy * projection_energy
                    )
                    risk += signal_power_group[group_index, patch_index] * max(
                        bias_energy,
                        np.float64(0.0),
                    )
        risks[output_index] = risk
    return risks


def compute_mass_projected_diagonal_filter_risks(
    attenuation: NDArray[np.floating],
    *,
    model: AggregationWeightModel,
    domain: AggregationWeightDomain,
    scope: AggregationWeightScope,
    noise_variance: NDArray[np.floating],
    mass_functional: NDArray[np.floating],
    metric_factors: SynthesisRiskMetricFactors,
    signal_power: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    model = coerce_enum(model, AggregationWeightModel, "weight_model")
    domain = coerce_enum(domain, AggregationWeightDomain, "weight_domain")
    scope = coerce_enum(scope, AggregationWeightScope, "weight_scope")
    if model not in {
        AggregationWeightModel.VARIANCE,
        AggregationWeightModel.RISK,
    }:
        raise ValueError("Mass-projected exact risks support variance or risk weighting.")
    variance_arr = np.asarray(noise_variance, dtype=np.float64)
    signal = None
    if model is AggregationWeightModel.RISK:
        if signal_power is None:
            raise ValueError("risk weighting requires coefficient signal power.")
        signal = np.asarray(signal_power, dtype=np.float64)

    attenuation_arr = np.asarray(attenuation, dtype=np.float64)
    mass = np.asarray(mass_functional, dtype=np.float64)
    if attenuation_arr.shape != mass.shape:
        raise ValueError(
            "mass_functional and attenuation must have the same shape, "
            f"got {mass.shape} and {attenuation_arr.shape}."
        )
    if variance_arr.shape != attenuation_arr.shape:
        raise ValueError(
            "noise_variance and attenuation must have the same shape, "
            f"got {variance_arr.shape} and {attenuation_arr.shape}."
        )
    if signal is not None and signal.shape != attenuation_arr.shape:
        raise ValueError(
            "signal_power and attenuation must have the same shape, "
            f"got {signal.shape} and {attenuation_arr.shape}."
        )
    mass_energy = float(np.sum(mass**2, dtype=np.float64))
    if mass_energy <= 0.0:
        raise ValueError("mass_functional must have nonzero energy.")

    group_size = int(attenuation_arr.shape[0])
    patch_volume = int(np.prod(attenuation_arr.shape[1:]))
    if metric_factors.inverse_group.shape != (group_size, group_size):
        raise ValueError("Synthesis metric factors do not match the attenuation shape.")
    if metric_factors.spatial_metric.shape != (patch_volume, patch_volume):
        raise ValueError("Synthesis metric factors do not match the attenuation shape.")
    variance = variance_arr.reshape(-1)
    if signal is not None:
        signal = signal.reshape(-1)
    mass_group = mass.reshape(group_size, patch_volume)
    projection_direction = mass_group / mass_energy

    if scope is AggregationWeightScope.GROUP:
        attenuation_flat = attenuation_arr.reshape(-1)
        attenuation_sq = attenuation_flat**2
        rejected = 1.0 - attenuation_flat
        rejected_sq = rejected**2
        mass_sq = mass_group.reshape(-1) ** 2

        def evaluate(
            group_metric: NDArray[np.floating],
            spatial_metric: NDArray[np.floating],
        ) -> float:
            noise_energy, bias_energy = _mass_projected_metric_energies(
                attenuation_flat,
                attenuation_sq,
                rejected,
                rejected_sq,
                mass_group,
                mass_sq,
                projection_direction,
                group_metric,
                spatial_metric,
            )
            risk = float(np.sum(variance * noise_energy, dtype=np.float64))
            if signal is not None:
                risk += float(np.sum(signal * bias_energy, dtype=np.float64))
            return risk

        if domain is AggregationWeightDomain.COEFFICIENT:
            risk = evaluate(
                np.eye(group_size, dtype=np.float64),
                np.eye(patch_volume, dtype=np.float64),
            )
        elif domain is AggregationWeightDomain.WINDOWED_SYNTHESIS:
            risk = evaluate(metric_factors.group_metric, metric_factors.spatial_metric)
        else:
            raise ValueError(f"Unknown weight domain: {domain!r}.")
        return np.asarray([risk], dtype=np.float64)
    if scope is not AggregationWeightScope.PATCH:
        raise ValueError(f"Unknown weight scope: {scope!r}.")

    if domain not in {
        AggregationWeightDomain.COEFFICIENT,
        AggregationWeightDomain.WINDOWED_SYNTHESIS,
    }:
        raise ValueError(f"Unknown weight domain: {domain!r}.")
    spatial_metric = metric_factors.spatial_metric
    signal_group = (
        attenuation_arr.reshape(group_size, patch_volume)
        if signal is None
        else signal.reshape(group_size, patch_volume)
    )
    return _compute_mass_projected_patch_risks_numba(
        attenuation_arr.reshape(group_size, patch_volume),
        mass_group,
        projection_direction,
        metric_factors.inverse_group,
        np.asarray(spatial_metric, dtype=np.float64),
        variance.reshape(group_size, patch_volume),
        signal_group,
        signal is not None,
        domain is AggregationWeightDomain.COEFFICIENT,
    )


@njit(cache=True, nogil=True)
def _invert_patch_aggregation_risks_numba(
    risks: NDArray[np.float64],
    eps: float,
) -> tuple[NDArray[np.float32], bool]:
    weights = np.empty(risks.size, dtype=np.float32)
    valid = True
    for index in range(risks.size):
        risk = risks[index]
        if not np.isfinite(risk) or risk < 0.0:
            valid = False
            weights[index] = np.float32(0.0)
        else:
            weights[index] = np.float32(1.0 / (risk + eps))
    return weights, valid


@njit(cache=True, nogil=True)
def _compute_real_diagonal_filter_patch_weights_numba(
    attenuation_group: NDArray[np.floating],
    noise_variance_group: NDArray[np.floating],
    signal_power_group: NDArray[np.floating],
    group_mixing_energy: NDArray[np.float64],
    spatial_metric_diag: NDArray[np.float64],
    model_code: int,
    windowed_synthesis: bool,
    eps: float,
) -> tuple[NDArray[np.float32], bool]:
    risks = _compute_real_diagonal_filter_patch_risks_numba(
        attenuation_group,
        noise_variance_group,
        signal_power_group,
        group_mixing_energy,
        spatial_metric_diag,
        model_code,
        windowed_synthesis,
    )
    return _invert_patch_aggregation_risks_numba(risks, eps)


def invert_aggregation_risks(
    risks: float | NDArray[np.floating],
    *,
    group_size: int,
    scope: AggregationWeightScope,
    eps: float = 1e-10,
) -> NDArray[np.float32]:
    risk = np.asarray(risks, dtype=np.float64).reshape(-1)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps}.")
    scope = coerce_enum(scope, AggregationWeightScope, "weight_scope")
    if scope is AggregationWeightScope.GROUP:
        if risk.size != 1:
            raise ValueError(f"Group scope requires one risk, got {risk.size}.")
        if not np.isfinite(risk.item()) or risk.item() < 0.0:
            raise ValueError("Aggregation risks must be finite and non-negative.")
        return np.full(
            group_size,
            np.float32(1.0 / (risk.item() + eps)),
            dtype=np.float32,
        )
    elif scope is AggregationWeightScope.PATCH:
        if risk.size != group_size:
            raise ValueError(
                f"Patch scope requires {group_size} risks, got {risk.size}."
            )
    else:
        raise ValueError(f"Unknown weight scope: {scope!r}.")
    weights, valid = _invert_patch_aggregation_risks_numba(risk, eps)
    if not valid:
        raise ValueError("Aggregation risks must be finite and non-negative.")
    return weights


def compute_aggregation_weights(
    attenuation: NDArray[np.floating],
    *,
    model: AggregationWeightModel,
    domain: AggregationWeightDomain,
    scope: AggregationWeightScope,
    noise_variance: NDArray[np.floating] | None,
    signal_power: NDArray[np.floating] | None,
    inverse_transform: Sequence[TransformEntry],
    w_patch: NDArray[np.floating],
    metric_factors: SynthesisRiskMetricFactors | None,
    mass_functional: NDArray[np.floating] | None,
    eps: float,
) -> NDArray[np.float32]:
    model = coerce_enum(model, AggregationWeightModel, "weight_model")
    domain = coerce_enum(domain, AggregationWeightDomain, "weight_domain")
    scope = coerce_enum(scope, AggregationWeightScope, "weight_scope")
    group_size = int(attenuation.shape[0])
    needs_metric = (
        domain is AggregationWeightDomain.WINDOWED_SYNTHESIS
        or scope is AggregationWeightScope.PATCH
    )
    if mass_functional is not None and model in {
        AggregationWeightModel.VARIANCE,
        AggregationWeightModel.RISK,
    }:
        needs_metric = True
    if needs_metric and metric_factors is None:
        metric_factors = get_synthesis_risk_metric_factors(
            tuple(int(size) for size in attenuation.shape),
            inverse_transform,
            w_patch,
        )

    if (
        mass_functional is None
        and scope is AggregationWeightScope.PATCH
        and metric_factors is not None
    ):
        model_code = _get_aggregation_weight_model_code(model)
        attenuation_arr = attenuation
        variance_arr = attenuation_arr if noise_variance is None else noise_variance
        signal_arr = attenuation_arr if signal_power is None else signal_power
        if (
            domain in {
                AggregationWeightDomain.COEFFICIENT,
                AggregationWeightDomain.WINDOWED_SYNTHESIS,
            }
            and (model_code == _WEIGHT_MODEL_CLASSIC or noise_variance is not None)
            and (model_code != _WEIGHT_MODEL_RISK or signal_power is not None)
            and attenuation_arr.dtype in _REAL_AGGREGATION_WEIGHT_DTYPES
            and variance_arr.dtype in _REAL_AGGREGATION_WEIGHT_DTYPES
            and signal_arr.dtype in _REAL_AGGREGATION_WEIGHT_DTYPES
            and variance_arr.shape == attenuation_arr.shape
            and signal_arr.shape == attenuation_arr.shape
        ):
            weights, valid = _compute_real_diagonal_filter_patch_weights_numba(
                attenuation_arr.reshape(group_size, -1),
                variance_arr.reshape(group_size, -1),
                signal_arr.reshape(group_size, -1),
                metric_factors.group_mixing_energy,
                metric_factors.spatial_metric_diag,
                model_code,
                domain is AggregationWeightDomain.WINDOWED_SYNTHESIS,
                eps,
            )
            if valid:
                return weights

    if mass_functional is not None and model in {
        AggregationWeightModel.VARIANCE,
        AggregationWeightModel.RISK,
    }:
        assert noise_variance is not None
        assert metric_factors is not None
        risks = compute_mass_projected_diagonal_filter_risks(
            attenuation,
            model=model,
            domain=domain,
            scope=scope,
            noise_variance=noise_variance,
            signal_power=signal_power,
            mass_functional=mass_functional,
            metric_factors=metric_factors,
        )
    elif scope is AggregationWeightScope.GROUP:
        risk = compute_diagonal_filter_group_risk(
            attenuation,
            model=model,
            domain=domain,
            noise_variance=noise_variance,
            signal_power=signal_power,
            metric_factors=metric_factors,
        )
        risks = np.asarray([risk], dtype=np.float64)
    else:
        assert metric_factors is not None
        risks = compute_diagonal_filter_patch_risks(
            attenuation,
            model=model,
            domain=domain,
            noise_variance=noise_variance,
            signal_power=signal_power,
            metric_factors=metric_factors,
        )
    return invert_aggregation_risks(
        risks,
        group_size=group_size,
        scope=scope,
        eps=eps,
    )
