from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .aggregationweights import (
    SynthesisRiskMetricFactors,
    compute_aggregation_weights,
    get_synthesis_risk_metric_factors,
    invert_aggregation_risks,
)
from .blockmatching import (
    BlockMatchGroup,
    BlockMatchResult,
    blockmatch_groups,
    compute_blockmatch_threshold,
    compute_poisson_reference_thresholds,
)
from .cache import CacheBudget, _resolve_cache_limits, _validate_max_cache_bytes
from .enums import (
    AggregationWeightDomain,
    AggregationWeightModel,
    AggregationWeightScope,
    BlockMatchDistance,
    CacheMode,
    PoissonGroupMassConservation,
    PoissonHTMatchPolicy,
    PoissonVarianceSource,
)
from .groupfiltering import _FilteredGroup, _GroupFilterState, _iter_filtered_groups
from .patchaggregation import (
    prepare_patch_aggregation,
    scatter_add_patches_prepared,
    scatter_add_patches_with_scalar_sigma_prepared,
    scatter_add_patches_with_sigma_prepared,
)
from .poisson import (
    _FrozenAggregationGroup,
    _get_group_constant_synthesis_direction,
    _poisson_hard_threshold_mask,
    _reaggregate_with_group_functionals,
)
from .profilevalidation import resolve_profile
from .profilevalidation import resolve_profile
from .psd import _validate_sigma, _validate_sigma_psd
from .sharpening import sharpen_group_dc, spatial_sharpen
from .shifts import get_reference_schedule
from .transforms import apply_transform_nd, get_transform_matrix
from .utils import extract_patches_strided, kaiser_window_nd
from .variance import (
    build_exact_poisson_group_plan,
    build_exact_white_gaussian_group_plan,
    build_poisson_variance_model,
    build_variance_model,
    get_direct_poisson_noise_std,
    get_gaussian_blockmatch_noise_ssd,
    get_poisson_variance_patches,
    get_stationary_gaussian_group_noise_variance,
    limit_poisson_group_variance_planes,
)
from .wiener import (
    compute_covariance_aware_group_risk_from_plan,
    compute_poisson_wiener_gain,
    compute_poisson_wiener_signal_power,
    compute_wiener_gain,
)


def bmnd(
    volume: NDArray[np.floating],
    profile: BMNDProfile,
    sigma: float | None = None,
    sigma_psd: NDArray[np.floating] | float | None = None,
    return_sigma_map: bool = False,
    stage_callback: Callable[[str, NDArray[np.float32]], None] | None = None,
    blockmatch_callback: Callable[[str, BlockMatchResult], None] | None = None,
    max_cache_bytes: CacheBudget = CacheMode.AUTO,
) -> (
    NDArray[np.float32]
    | tuple[NDArray[np.float32], dict[str, NDArray[np.float32] | float]]
):
    """
    Apply BMND denoising to a noisy volume.

    Parameters
    ----------
    volume : ndarray, shape (D1, D2, ..., Dn)
        Noisy input n-D volume. It is internally converted to
        ``float32``.
    profile : BMNDProfile
        Profile object containing all algorithmic hyperparameters.
    sigma : float or None, optional
        Global noise standard deviation estimate for stationary additive
        noise. If omitted, BMND only uses noise levels supplied by the
        active variance model.
    sigma_psd : ndarray, float, or None, optional
        Stationary noise PSD, or a scalar noise standard deviation. This is
        mutually exclusive with ``sigma`` and unsupported for direct Poisson noise.
    stage_callback : callable or None, optional
        Diagnostic callback receiving copies of the HT and Wiener estimates
        as ``("ht", estimate)`` and ``("wiener", estimate)``.
    blockmatch_callback : callable or None, optional
        Diagnostic callback receiving the HT and Wiener block-match results
        as ``("ht", result)`` and ``("wiener", result)``.
    max_cache_bytes : non-negative int, ``"auto"``, or None, default ``"auto"``
        Aggregate retained Poisson covariance-cache budget for this invocation.
        ``None`` leaves those caches unlimited, while ``"auto"`` subtracts an
        estimated non-cache peak and safety reserve from available system,
        cgroup, and Slurm memory. It disables retained covariance caching when
        memory headroom cannot be detected. Temporary filtering workspaces are
        excluded.

    Returns
    -------
    ndarray of float32, shape (D1, D2, ..., Dn)
        Denoised output volume, with the same shape as ``volume``.
    (ndarray, dict) if return_sigma_map is True
        A tuple containing the denoised output and a dict of sigma maps
        ``{"ht": sigma_map_ht, "wiener": sigma_map_wiener}``, each with the
        same shape as ``volume``.
    """
    if sigma is not None and sigma_psd is not None:
        raise ValueError("sigma and sigma_psd are mutually exclusive.")
    if sigma is not None:
        _validate_sigma(sigma)
    cache_request = _validate_max_cache_bytes(max_cache_bytes)

    resolved_profile = resolve_profile(
        profile,
        tuple(int(size) for size in volume.shape),
    )
    noise_model = profile.noise_model
    is_poisson = resolved_profile.is_poisson
    effective_nf = resolved_profile.effective_nf
    conventional_nf = resolved_profile.conventional_nf
    if is_poisson:
        if sigma_psd is not None:
            raise ValueError(
                "sigma_psd is only supported for stationary additive noise; "
                "direct Poisson mode uses a variance volume instead."
            )

    volume_array = np.asarray(volume)
    if not np.all(np.isfinite(volume_array)):
        raise ValueError("volume must contain only finite values.")
    with np.errstate(over="ignore", invalid="ignore"):
        vol = volume_array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(vol)):
        raise ValueError(
            "volume values must be representable as finite float32 values."
        )
    if is_poisson and np.any(vol < 0.0):
        raise ValueError("Direct Poisson input must contain only non-negative values.")

    if sigma_psd is not None:
        _validate_sigma_psd(tuple(int(size) for size in vol.shape), sigma_psd)

    if not np.any(vol != 0.0):
        zero_estimate = vol
        if stage_callback is not None:
            stage_callback("ht", zero_estimate.copy())
            stage_callback("wiener", zero_estimate.copy())
        if return_sigma_map:
            sigma_map = np.zeros_like(volume, dtype=np.float32)
            return zero_estimate, {
                "ht": sigma_map,
                "wiener": sigma_map,
            }
        return zero_estimate

    ht_cache_limits, wiener_cache_limits = _resolve_cache_limits(
        cache_request,
        volume,
        profile,
        resolved_profile,
        return_sigma_map=return_sigma_map,
        has_stage_callback=stage_callback is not None,
    )

    if not vol.flags["C_CONTIGUOUS"]:
        vol = np.ascontiguousarray(vol)

    transform_cache: dict[
        tuple[tuple[int, ...], str],
        tuple[list[NDArray | Callable | None], list[NDArray | Callable | None]],
    ] = {}
    group_constant_direction_cache: dict[
        tuple[tuple[int, ...], str], NDArray[np.generic]
    ] = {}
    synthesis_risk_metric_cache: dict[
        tuple[tuple[int, ...], str], SynthesisRiskMetricFactors
    ] = {}
    eps = 1e-10

    # --- Noise model selection ---
    blockmatch_distance = resolved_profile.blockmatch_distance
    variance_k = resolved_profile.variance_k

    poisson_variance_model_ht = None
    poisson_variance_model_wiener = None
    variance_model = None
    if not is_poisson:
        variance_model = build_variance_model(
            vol,
            sigma=sigma,
            sigma_psd=sigma_psd,
            nf=effective_nf,
            poisson_cache_limits=ht_cache_limits,
        )

    if is_poisson:
        poisson_variance_model_ht = build_poisson_variance_model(
            vol,
            variance_floor=profile.poisson_variance_floor,
            scale=profile.poisson_count_scale,
            source_name="observation",
            cache_limits=ht_cache_limits,
        )

    global_sigma: float
    if poisson_variance_model_ht is not None:
        global_sigma = poisson_variance_model_ht.global_sigma
    elif variance_model is not None:
        global_sigma = variance_model.global_sigma
    else:
        raise RuntimeError("BMND expected a Gaussian or Poisson variance model.")

    # -------------------- Stage 1: Hard-thresholding ---------------------
    w_patch_ht = kaiser_window_nd(profile.ht_block_size, beta=profile.ht_kaiser_beta)

    patches_view = extract_patches_strided(vol, profile.ht_block_size)
    counts = patches_view.shape[: vol.ndim]
    num_patches = int(np.prod(counts))
    patch_vol = int(np.prod(profile.ht_block_size))

    patches_flat = patches_view.reshape(num_patches, patch_vol).astype(
        np.float32, copy=False
    )
    reference_schedule_ht = get_reference_schedule(
        counts,
        profile.ht_block_size,
        profile.ht_step,
        mode=profile.reference_schedule_mode,
        shift_density=profile.reference_shift_density,
        schedule_density=profile.reference_schedule_density,
    )

    step1_numerator = np.zeros_like(vol, dtype=np.float32)
    step1_denominator = np.zeros_like(vol, dtype=np.float32)

    if return_sigma_map:
        sigma_num_ht = np.zeros_like(vol, dtype=np.float32)
    else:
        sigma_num_ht = None
    ht_aggregation_state = prepare_patch_aggregation(
        step1_numerator.shape,
        tuple(counts),
        profile.ht_block_size,
        w_patch_ht,
    )
    aggregation_aware_ht = is_poisson and profile.poisson_group_mass_conservation in {
        PoissonGroupMassConservation.HT,
        PoissonGroupMassConservation.BOTH,
    }
    frozen_ht_groups: list[_FrozenAggregationGroup] = []

    poisson_ht_match_policy = (
        profile.poisson_ht_match_policy if is_poisson else PoissonHTMatchPolicy.FIXED
    )
    if poisson_ht_match_policy is PoissonHTMatchPolicy.REFERENCE_FINITE_COUNT:
        threshold_ht = compute_poisson_reference_thresholds(
            reference_schedule_ht,
            tuple(counts),
            patches_flat,
            blockmatch_distance,
            profile.poisson_count_scale,
            profile.ht_match_threshold,
            profile.poisson_ht_match_structure_beta,
            profile.poisson_ht_match_intensity_power,
            profile.poisson_ht_match_moment_max_count,
        )
    else:
        threshold_ht = compute_blockmatch_threshold(
            profile.ht_match_threshold,
            vol,
            blockmatch_distance,
        )
    need_position_metadata_ht = variance_k > 0
    gaussian_noise_ssd_ht = None
    gamma_ht = 0.0
    if is_poisson and blockmatch_distance is BlockMatchDistance.SSD:
        gamma_ht = profile.gamma
    elif blockmatch_distance is BlockMatchDistance.SSD:
        assert variance_model is not None
        gamma_ht = 0.0 if conventional_nf else profile.gamma
        if gamma_ht != 0.0:
            gaussian_noise_ssd_ht = get_gaussian_blockmatch_noise_ssd(
                variance_model,
                profile.ht_block_size,
                profile.ht_search_window,
            )
    matches_stage1 = blockmatch_groups(
        reference_schedule=reference_schedule_ht,
        counts=counts,
        patches_flat=patches_flat,
        search_radius=profile.ht_search_window,
        max_matches=profile.ht_max_stack_size,
        min_matches=profile.ht_min_stack_size,
        distance_threshold=threshold_ht,
        batch_size=profile.ref_batch_size,
        include_position_metadata=need_position_metadata_ht,
        distance_measure=blockmatch_distance,
        poisson_count_scale=profile.poisson_count_scale,
        gaussian_noise_ssd=gaussian_noise_ssd_ht,
        poisson_moment_max_count=profile.poisson_ht_match_moment_max_count,
        standardize_poisson_distance=(
            poisson_ht_match_policy is PoissonHTMatchPolicy.CANDIDATE_STANDARDIZED
        ),
        poisson_gamma=is_poisson and blockmatch_distance is BlockMatchDistance.SSD,
        gamma=gamma_ht,
        stabilize_poisson_ties=is_poisson,
    )
    if blockmatch_callback is not None:
        blockmatch_callback("ht", matches_stage1)

    variance_patches_ht_flat = None
    if poisson_variance_model_ht is not None:
        variance_patches_ht_flat = get_poisson_variance_patches(
            poisson_variance_model_ht,
            profile.ht_block_size,
        ).reshape(num_patches, *profile.ht_block_size)

    def filter_ht_group(
        match_group: BlockMatchGroup, state: _GroupFilterState
    ) -> _FilteredGroup | None:
        if match_group.final_count == 0:
            return None

        poisson_variance_model_ht = state.poisson_model
        transform_cache = state.transforms
        group_constant_direction_cache = state.constant_directions
        synthesis_risk_metric_cache = state.risk_metrics

        sel_global = match_group.selected_lin_indices
        group = patches_flat[sel_global].reshape(-1, *profile.ht_block_size)

        transform_key = (group.shape, "ht")
        if transform_key not in transform_cache:
            transform_cache[transform_key] = get_transform_matrix(
                group.shape, profile.ht_transform
            )
        ht_transform, inverse_ht_transform = transform_cache[transform_key]

        group_transformed = apply_transform_nd(group, ht_transform)

        # ----------------- Local sigma / variance (noise-model aware) -----------------
        exact_source_plan_ht = None
        if is_poisson and poisson_variance_model_ht is not None:
            assert variance_patches_ht_flat is not None
            group_variance_ht = variance_patches_ht_flat[sel_global]
            exact_ht_weighting = (
                profile.ht_weight_model is AggregationWeightModel.VARIANCE
                and profile.ht_weight_domain
                is AggregationWeightDomain.WINDOWED_SYNTHESIS
                and profile.ht_weight_scope is AggregationWeightScope.GROUP
                and variance_k > 0
            )
            if exact_ht_weighting:
                exact_source_plan_ht = build_exact_poisson_group_plan(
                    poisson_variance_model_ht,
                    match_group.selected_abs_positions,
                    match_group.selected_shifted_positions,
                    group.shape,
                    ht_transform,
                )
                exact_sigma_ht = exact_source_plan_ht.prepare_noise_std_for_risk(
                    variance_k
                )
                if variance_k >= group.shape[0]:
                    sigma_ht = exact_sigma_ht
                else:
                    approximate_sigma_ht = get_direct_poisson_noise_std(
                        group_variance_ht,
                        ht_transform,
                        k=0,
                    )
                    sigma_ht = limit_poisson_group_variance_planes(
                        exact_sigma_ht,
                        approximate_sigma_ht,
                        variance_k,
                    )
            else:
                sigma_ht = get_direct_poisson_noise_std(
                    group_variance_ht,
                    ht_transform,
                    k=variance_k,
                    variance_model=poisson_variance_model_ht,
                    selected_abs_positions=match_group.selected_abs_positions,
                    selected_shifted_positions=match_group.selected_shifted_positions,
                    group_shape=group.shape,
                )
        elif variance_model is not None:
            exact_ht_weighting = (
                profile.ht_weight_model is AggregationWeightModel.VARIANCE
                and profile.ht_weight_domain
                is AggregationWeightDomain.WINDOWED_SYNTHESIS
                and profile.ht_weight_scope is AggregationWeightScope.GROUP
                and variance_k > 0
            )
            if exact_ht_weighting:
                exact_source_plan_ht = build_exact_white_gaussian_group_plan(
                    variance_model,
                    tuple(int(size) for size in vol.shape),
                    match_group.selected_abs_positions,
                    match_group.selected_shifted_positions,
                    group.shape,
                    ht_transform,
                )
            if exact_source_plan_ht is not None:
                exact_sigma_ht = exact_source_plan_ht.prepare_noise_std_for_risk(
                    variance_k
                )
                if variance_k >= group.shape[0]:
                    sigma_ht = exact_sigma_ht
                else:
                    approximate_variance_ht = (
                        get_stationary_gaussian_group_noise_variance(
                            group.shape,
                            ht_transform,
                            variance_model,
                            match_group.selected_abs_positions,
                            0,
                        )
                    )
                    sigma_ht = limit_poisson_group_variance_planes(
                        exact_sigma_ht,
                        np.sqrt(np.maximum(approximate_variance_ht, 0.0)),
                        variance_k,
                    )
                variance_ht = np.asarray(sigma_ht, dtype=np.float32) ** 2
            else:
                variance_ht = get_stationary_gaussian_group_noise_variance(
                    group.shape,
                    ht_transform,
                    variance_model,
                    match_group.selected_abs_positions,
                    variance_k,
                )
                sigma_ht = np.sqrt(np.maximum(variance_ht, 0.0)).astype(
                    np.float32,
                    copy=False,
                )
        else:
            raise RuntimeError(
                "BMND expected HT-stage noise statistics from a variance model."
            )

        thr = profile.ht_lambda_threshold * sigma_ht
        coefficient_magnitude = np.abs(group_transformed)
        if is_poisson:
            maskG = _poisson_hard_threshold_mask(
                coefficient_magnitude,
                thr,
                profile.poisson_count_scale,
            )
        else:
            maskG = coefficient_magnitude > thr
        ht_attenuation = np.asarray(maskG, dtype=np.float32)
        if not is_poisson:
            maskG = coefficient_magnitude >= thr
            maskG.reshape(-1)[0] = True
            ht_attenuation = np.asarray(maskG, dtype=np.float32)
        if profile.ht_use_soft_thresholding:
            shrunk_magnitude = np.maximum(coefficient_magnitude - thr, 0.0)
            ht_attenuation = np.divide(
                shrunk_magnitude,
                coefficient_magnitude,
                out=np.zeros_like(coefficient_magnitude, dtype=np.float32),
                where=coefficient_magnitude > 0.0,
            ).astype(np.float32, copy=False)
            group_transformed = np.sign(group_transformed) * shrunk_magnitude
        else:
            group_transformed *= maskG

        group_filt = apply_transform_nd(group_transformed, inverse_ht_transform)

        n_group = group_filt.shape[0]
        assert profile.ht_weight_domain is not None
        assert profile.ht_weight_scope is not None
        attenuation_for_weight = ht_attenuation
        if exact_source_plan_ht is not None:
            group_risk_ht = compute_covariance_aware_group_risk_from_plan(
                attenuation_for_weight,
                exact_source_plan_ht,
                inverse_ht_transform,
                w_patch_ht,
                exact_plane_count=variance_k,
                noise_sigma=sigma_ht,
            )
            weights_valid = invert_aggregation_risks(
                group_risk_ht,
                group_size=n_group,
                scope=AggregationWeightScope.GROUP,
                eps=eps,
            )
        else:
            noise_variance_ht = np.asarray(sigma_ht, dtype=np.float32) ** 2
            needs_metric = (
                profile.ht_weight_domain is AggregationWeightDomain.WINDOWED_SYNTHESIS
                or profile.ht_weight_scope is AggregationWeightScope.PATCH
            )
            if needs_metric and transform_key not in synthesis_risk_metric_cache:
                synthesis_risk_metric_cache[transform_key] = (
                    get_synthesis_risk_metric_factors(
                        group.shape,
                        inverse_ht_transform,
                        w_patch_ht,
                    )
                )
            weights_valid = compute_aggregation_weights(
                attenuation_for_weight,
                model=profile.ht_weight_model,
                domain=profile.ht_weight_domain,
                scope=profile.ht_weight_scope,
                noise_variance=noise_variance_ht,
                signal_power=None,
                inverse_transform=inverse_ht_transform,
                w_patch=w_patch_ht,
                mass_functional=None,
                metric_factors=synthesis_risk_metric_cache.get(transform_key),
                eps=eps,
            )

        frozen = None
        if aggregation_aware_ht:
            if transform_key not in group_constant_direction_cache:
                group_constant_direction_cache[transform_key] = (
                    _get_group_constant_synthesis_direction(
                        group.shape,
                        inverse_ht_transform,
                    )
                )
            frozen = _FrozenAggregationGroup(
                selected=np.asarray(sel_global, dtype=np.int64).copy(),
                weights=np.asarray(weights_valid, dtype=np.float32).copy(),
                coefficients=np.asarray(group_transformed).copy(),
                inverse_transforms=inverse_ht_transform,
                correction_direction=group_constant_direction_cache[transform_key],
            )

        return _FilteredGroup(
            sel_global,
            group_filt,
            weights_valid,
            sigma_ht if return_sigma_map else None,
            frozen,
        )

    ht_state = _GroupFilterState(
        poisson_variance_model_ht,
        transform_cache,
        group_constant_direction_cache,
        synthesis_risk_metric_cache,
    )
    for filtered in _iter_filtered_groups(
        matches_stage1.groups, filter_ht_group, ht_state, parallel=False
    ):
        sel_global = filtered.selected
        group_filt = filtered.patches
        weights_valid = filtered.weights
        sigma_ht = filtered.sigma
        if filtered.frozen is not None:
            frozen_ht_groups.append(filtered.frozen)
        if return_sigma_map:
            assert sigma_ht is not None
            if np.ndim(sigma_ht) == 0:
                scatter_add_patches_with_scalar_sigma_prepared(
                    accum_num=step1_numerator,
                    accum_den=step1_denominator,
                    sigma_num=sigma_num_ht,
                    group_filt=group_filt,
                    sigma=np.asarray(sigma_ht, dtype=np.float32).item(),
                    weights=weights_valid,
                    sel_global=sel_global,
                    prepared=ht_aggregation_state,
                )
            else:
                scatter_add_patches_with_sigma_prepared(
                    accum_num=step1_numerator,
                    accum_den=step1_denominator,
                    sigma_num=sigma_num_ht,
                    group_filt=group_filt,
                    group_sigma=np.asarray(sigma_ht, dtype=np.float32),
                    weights=weights_valid,
                    sel_global=sel_global,
                    prepared=ht_aggregation_state,
                )
        else:
            scatter_add_patches_prepared(
                accum_num=step1_numerator,
                accum_den=step1_denominator,
                group_filt=group_filt,
                weights=weights_valid,
                sel_global=sel_global,
                prepared=ht_aggregation_state,
            )

    if aggregation_aware_ht:
        step1_numerator = _reaggregate_with_group_functionals(
            frozen_ht_groups,
            patches_flat,
            profile.ht_block_size,
            step1_denominator,
            ht_aggregation_state,
            eps,
        )
        step1_est = vol.copy()
        np.divide(
            step1_numerator,
            step1_denominator,
            out=step1_est,
            where=step1_denominator > 0,
        )
    else:
        step1_est = np.where(
            step1_denominator > 0,
            step1_numerator / (step1_denominator + eps),
            vol,
        )
    if stage_callback is not None:
        stage_callback("ht", step1_est.copy())

    if is_poisson:
        if profile.poisson_variance_source_wiener is PoissonVarianceSource.PILOT:
            wiener_variance_source = step1_est
            poisson_variance_model_wiener = build_poisson_variance_model(
                wiener_variance_source,
                variance_floor=profile.poisson_variance_floor,
                scale=profile.poisson_count_scale,
                source_name=profile.poisson_variance_source_wiener.value,
                cache_limits=wiener_cache_limits,
            )
        else:
            poisson_variance_model_wiener = poisson_variance_model_ht

    # -------------------- Stage 2: Wiener filtering ---------------------
    same_patch_geometry = profile.wiener_block_size == profile.ht_block_size
    same_aggregation_geometry = (
        same_patch_geometry and profile.wiener_kaiser_beta == profile.ht_kaiser_beta
    )
    if same_aggregation_geometry:
        w_patch_wiener = w_patch_ht
    else:
        w_patch_wiener = kaiser_window_nd(
            profile.wiener_block_size, beta=profile.wiener_kaiser_beta
        )

    patches_view1 = extract_patches_strided(step1_est, profile.wiener_block_size)
    counts2 = patches_view1.shape[: vol.ndim]
    num_patches2 = int(np.prod(counts2))
    patch_vol2 = int(np.prod(profile.wiener_block_size))

    patches_flat1 = patches_view1.reshape(num_patches2, patch_vol2).astype(
        np.float32, copy=False
    )
    if same_patch_geometry:
        patches_flat_noisy2 = patches_flat
    else:
        patches_view_noisy2 = extract_patches_strided(vol, profile.wiener_block_size)
        patches_flat_noisy2 = patches_view_noisy2.reshape(
            int(np.prod(patches_view_noisy2.shape[: vol.ndim])), patch_vol2
        ).astype(np.float32, copy=False)

    if same_patch_geometry and profile.wiener_step == profile.ht_step:
        reference_schedule_wiener = reference_schedule_ht
    else:
        reference_schedule_wiener = get_reference_schedule(
            counts2,
            profile.wiener_block_size,
            profile.wiener_step,
            mode=profile.reference_schedule_mode,
            shift_density=profile.reference_shift_density,
            schedule_density=profile.reference_schedule_density,
        )

    step2_numerator = np.zeros_like(vol, dtype=np.float32)
    step2_denominator = np.zeros_like(vol, dtype=np.float32)

    if return_sigma_map:
        sigma_num_wiener = np.zeros_like(vol, dtype=np.float32)
    else:
        sigma_num_wiener = None
    if same_aggregation_geometry:
        wiener_aggregation_state = ht_aggregation_state
    else:
        wiener_aggregation_state = prepare_patch_aggregation(
            step2_numerator.shape,
            tuple(counts2),
            profile.wiener_block_size,
            w_patch_wiener,
        )
    aggregation_aware_wiener = (
        is_poisson
        and profile.poisson_group_mass_conservation
        in {
            PoissonGroupMassConservation.WIENER,
            PoissonGroupMassConservation.BOTH,
        }
    )
    frozen_wiener_groups: list[_FrozenAggregationGroup] = []

    wiener_match_on_observation = (
        is_poisson
        and profile.poisson_wiener_match_source is PoissonVarianceSource.OBSERVATION
    )
    wiener_match_patches = (
        patches_flat_noisy2 if wiener_match_on_observation else patches_flat1
    )
    wiener_match_volume = vol if wiener_match_on_observation else step1_est
    threshold_wiener = compute_blockmatch_threshold(
        profile.wiener_match_threshold,
        wiener_match_volume,
        blockmatch_distance,
    )
    need_position_metadata_wiener = variance_k > 0
    matches_stage2 = blockmatch_groups(
        reference_schedule=reference_schedule_wiener,
        counts=counts2,
        patches_flat=wiener_match_patches,
        search_radius=profile.wiener_search_window,
        max_matches=profile.wiener_max_stack_size,
        min_matches=profile.wiener_min_stack_size,
        distance_threshold=threshold_wiener,
        batch_size=profile.ref_batch_size,
        include_position_metadata=need_position_metadata_wiener,
        distance_measure=blockmatch_distance,
        poisson_count_scale=profile.poisson_count_scale,
        stabilize_poisson_ties=is_poisson,
    )
    if blockmatch_callback is not None:
        blockmatch_callback("wiener", matches_stage2)

    variance_patches_wiener_flat = None
    if poisson_variance_model_wiener is not None:
        variance_patches_wiener_flat = get_poisson_variance_patches(
            poisson_variance_model_wiener,
            profile.wiener_block_size,
        ).reshape(num_patches2, *profile.wiener_block_size)

    exact_wiener_weighting = (
        profile.wiener_weight_model
        in {AggregationWeightModel.VARIANCE, AggregationWeightModel.RISK}
        and profile.wiener_weight_domain is AggregationWeightDomain.WINDOWED_SYNTHESIS
        and profile.wiener_weight_scope is AggregationWeightScope.GROUP
        and variance_k > 0
    )

    def filter_wiener_group(
        match_group: BlockMatchGroup, state: _GroupFilterState
    ) -> _FilteredGroup | None:
        if match_group.final_count == 0:
            return None

        poisson_variance_model_wiener = state.poisson_model
        transform_cache = state.transforms
        group_constant_direction_cache = state.constant_directions
        synthesis_risk_metric_cache = state.risk_metrics

        sel_global = match_group.selected_lin_indices
        group_ref = patches_flat1[sel_global].reshape(-1, *profile.wiener_block_size)
        group_noisy = patches_flat_noisy2[sel_global].reshape(
            -1, *profile.wiener_block_size
        )

        transform_key = (group_ref.shape, "wiener")
        if transform_key not in transform_cache:
            transform_cache[transform_key] = get_transform_matrix(
                group_ref.shape, profile.wiener_transform
            )
        wiener_transform, inverse_wiener_transform = transform_cache[transform_key]

        Rcoef = apply_transform_nd(group_ref, wiener_transform)
        Ncoef = apply_transform_nd(group_noisy, wiener_transform)

        # ----------------- Local variance term for Wiener (noise-model aware) --------
        exact_source_plan_wiener = None
        if is_poisson and poisson_variance_model_wiener is not None:
            assert variance_patches_wiener_flat is not None
            group_variance_wiener = variance_patches_wiener_flat[sel_global]
            if exact_wiener_weighting:
                exact_source_plan_wiener = build_exact_poisson_group_plan(
                    poisson_variance_model_wiener,
                    match_group.selected_abs_positions,
                    match_group.selected_shifted_positions,
                    group_ref.shape,
                    wiener_transform,
                )
                exact_sigma_w = exact_source_plan_wiener.prepare_noise_std_for_risk(
                    variance_k
                )
                if variance_k >= group_ref.shape[0]:
                    sigma_w = exact_sigma_w
                else:
                    approximate_sigma_w = get_direct_poisson_noise_std(
                        group_variance_wiener,
                        wiener_transform,
                        k=0,
                    )
                    sigma_w = limit_poisson_group_variance_planes(
                        exact_sigma_w,
                        approximate_sigma_w,
                        variance_k,
                    )
            else:
                sigma_w = get_direct_poisson_noise_std(
                    group_variance_wiener,
                    wiener_transform,
                    k=variance_k,
                    variance_model=poisson_variance_model_wiener,
                    selected_abs_positions=match_group.selected_abs_positions,
                    selected_shifted_positions=match_group.selected_shifted_positions,
                    group_shape=group_ref.shape,
                )
        elif variance_model is not None:
            if exact_wiener_weighting:
                exact_source_plan_wiener = build_exact_white_gaussian_group_plan(
                    variance_model,
                    tuple(int(size) for size in vol.shape),
                    match_group.selected_abs_positions,
                    match_group.selected_shifted_positions,
                    group_ref.shape,
                    wiener_transform,
                )
            if exact_source_plan_wiener is not None:
                exact_sigma_w = exact_source_plan_wiener.prepare_noise_std_for_risk(
                    variance_k
                )
                if variance_k >= group_ref.shape[0]:
                    sigma_w = exact_sigma_w
                else:
                    approximate_variance_w = (
                        get_stationary_gaussian_group_noise_variance(
                            group_ref.shape,
                            wiener_transform,
                            variance_model,
                            match_group.selected_abs_positions,
                            0,
                        )
                    )
                    sigma_w = limit_poisson_group_variance_planes(
                        exact_sigma_w,
                        np.sqrt(np.maximum(approximate_variance_w, 0.0)),
                        variance_k,
                    )
                variance_w = np.asarray(sigma_w, dtype=np.float32) ** 2
            else:
                variance_w = get_stationary_gaussian_group_noise_variance(
                    group_ref.shape,
                    wiener_transform,
                    variance_model,
                    match_group.selected_abs_positions,
                    variance_k,
                )
                sigma_w = np.sqrt(np.maximum(variance_w, 0.0)).astype(
                    np.float32,
                    copy=False,
                )
        else:
            raise RuntimeError(
                "BMND expected Wiener-stage noise statistics from a variance model."
            )
        if return_sigma_map:
            sigma_wiener = float(np.mean(np.asarray(sigma_w, dtype=np.float32)))

        # -------- Per-coefficient Wiener gain --------
        if is_poisson:
            gain = compute_poisson_wiener_gain(
                Rcoef,
                sigma_w,
                profile.wiener_variance_scale,
                mode=profile.poisson_wiener_gain_mode,
                eps=eps,
            )
        else:
            gain = compute_wiener_gain(
                Rcoef, sigma_w, profile.wiener_variance_scale, eps
            )

        C_wien = Ncoef * gain
        sharpen_group_dc(C_wien, profile.sharpen_alpha_3d)
        group_filt = apply_transform_nd(C_wien, inverse_wiener_transform)
        spatial_sharpen(group_filt, profile.sharpen_alpha)

        n_group = group_filt.shape[0]

        assert profile.wiener_weight_domain is not None
        assert profile.wiener_weight_scope is not None
        attenuation_for_weight = gain
        signal_power = None
        if profile.wiener_weight_model is AggregationWeightModel.RISK:
            if is_poisson:
                signal_power = compute_poisson_wiener_signal_power(
                    Rcoef,
                    sigma_w,
                    profile.wiener_variance_scale,
                    mode=profile.poisson_wiener_gain_mode,
                )
            else:
                signal_power = np.asarray(np.abs(Rcoef) ** 2, dtype=np.float32)

        if exact_source_plan_wiener is not None:
            group_risk_wiener = compute_covariance_aware_group_risk_from_plan(
                attenuation_for_weight,
                exact_source_plan_wiener,
                inverse_wiener_transform,
                w_patch_wiener,
                exact_plane_count=variance_k,
                noise_sigma=sigma_w,
                signal_power=signal_power,
            )
            weights_valid = invert_aggregation_risks(
                group_risk_wiener,
                group_size=n_group,
                scope=AggregationWeightScope.GROUP,
                eps=eps,
            )
        else:
            noise_variance_w = np.asarray(sigma_w, dtype=np.float32) ** 2
            needs_metric = (
                profile.wiener_weight_domain
                is AggregationWeightDomain.WINDOWED_SYNTHESIS
                or profile.wiener_weight_scope is AggregationWeightScope.PATCH
            )
            if needs_metric and transform_key not in synthesis_risk_metric_cache:
                synthesis_risk_metric_cache[transform_key] = (
                    get_synthesis_risk_metric_factors(
                        group_ref.shape,
                        inverse_wiener_transform,
                        w_patch_wiener,
                    )
                )
            weights_valid = compute_aggregation_weights(
                attenuation_for_weight,
                model=profile.wiener_weight_model,
                domain=profile.wiener_weight_domain,
                scope=profile.wiener_weight_scope,
                noise_variance=noise_variance_w,
                signal_power=signal_power,
                inverse_transform=inverse_wiener_transform,
                w_patch=w_patch_wiener,
                mass_functional=None,
                metric_factors=synthesis_risk_metric_cache.get(transform_key),
                eps=eps,
            )

        frozen = None
        if aggregation_aware_wiener:
            if transform_key not in group_constant_direction_cache:
                group_constant_direction_cache[transform_key] = (
                    _get_group_constant_synthesis_direction(
                        group_noisy.shape,
                        inverse_wiener_transform,
                    )
                )
            frozen = _FrozenAggregationGroup(
                selected=np.asarray(sel_global, dtype=np.int64).copy(),
                weights=np.asarray(weights_valid, dtype=np.float32).copy(),
                coefficients=np.asarray(C_wien).copy(),
                inverse_transforms=inverse_wiener_transform,
                correction_direction=group_constant_direction_cache[transform_key],
            )

        return _FilteredGroup(
            sel_global,
            group_filt,
            weights_valid,
            sigma_wiener if return_sigma_map else None,
            frozen,
        )

    wiener_state = _GroupFilterState(
        poisson_variance_model_wiener,
        transform_cache,
        group_constant_direction_cache,
        synthesis_risk_metric_cache,
    )
    for filtered in _iter_filtered_groups(
        matches_stage2.groups,
        filter_wiener_group,
        wiener_state,
        parallel=is_poisson and exact_wiener_weighting,
    ):
        sel_global = filtered.selected
        group_filt = filtered.patches
        weights_valid = filtered.weights
        sigma_wiener = filtered.sigma
        if filtered.frozen is not None:
            frozen_wiener_groups.append(filtered.frozen)
        if return_sigma_map:
            assert sigma_wiener is not None
            scatter_add_patches_with_scalar_sigma_prepared(
                accum_num=step2_numerator,
                accum_den=step2_denominator,
                sigma_num=sigma_num_wiener,
                group_filt=group_filt,
                sigma=sigma_wiener,
                weights=weights_valid,
                sel_global=sel_global,
                prepared=wiener_aggregation_state,
            )
        else:
            scatter_add_patches_prepared(
                accum_num=step2_numerator,
                accum_den=step2_denominator,
                group_filt=group_filt,
                weights=weights_valid,
                sel_global=sel_global,
                prepared=wiener_aggregation_state,
            )

    if aggregation_aware_wiener:
        step2_numerator = _reaggregate_with_group_functionals(
            frozen_wiener_groups,
            patches_flat_noisy2,
            profile.wiener_block_size,
            step2_denominator,
            wiener_aggregation_state,
            eps,
        )
        step2_est = vol.copy()
        np.divide(
            step2_numerator,
            step2_denominator,
            out=step2_est,
            where=step2_denominator > 0,
        )
    else:
        step2_est = np.where(
            step2_denominator > 0,
            step2_numerator / (step2_denominator + eps),
            vol,
        )
    if stage_callback is not None:
        stage_callback("wiener", step2_est.copy())

    if return_sigma_map:
        sigma_map_ht = np.full_like(vol, global_sigma, dtype=np.float32)
        valid_ht = step1_denominator > 0
        sigma_map_ht[valid_ht] = sigma_num_ht[valid_ht] / step1_denominator[valid_ht]

        sigma_map_wiener = np.full_like(vol, global_sigma, dtype=np.float32)
        valid_wiener = step2_denominator > 0
        sigma_map_wiener[valid_wiener] = (
            sigma_num_wiener[valid_wiener] / step2_denominator[valid_wiener]
        )
        return step2_est, {
            "global": global_sigma,
            "ht": sigma_map_ht,
            "wiener": sigma_map_wiener,
        }
    return step2_est
