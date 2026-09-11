from dataclasses import dataclass

import numpy as np

from .blockmatching import resolve_blockmatch_distance
from .enums import (
    AggregationWeightDomain,
    AggregationWeightModel,
    AggregationWeightScope,
    BlockMatchDistance,
    NoiseModel,
    PoissonGroupMassConservation,
    PoissonHTMatchPolicy,
    PoissonVarianceSource,
    PoissonWienerGainMode,
    ReferenceScheduleMode,
    coerce_enum,
)
from .profiles import BMNDProfile
from .psd import resolve_nf
from .transforms import Transform, TransformMode, TransformType


@dataclass(frozen=True)
class ResolvedProfile:
    is_poisson: bool
    effective_nf: tuple[int, ...] | None
    conventional_nf: bool
    blockmatch_distance: BlockMatchDistance
    variance_k: int


def resolve_profile(
    profile: BMNDProfile,
    volume_shape: tuple[int, ...],
) -> ResolvedProfile:
    def validate_finite_nonnegative(
        name: str,
        value: object,
    ) -> float:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                f"{name} must be finite and non-negative, got {value!r}."
            )
        try:
            resolved_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be finite and non-negative, got {value!r}."
            ) from exc
        if not np.isfinite(resolved_value) or resolved_value < 0.0:
            raise ValueError(
                f"{name} must be finite and non-negative, got {value!r}."
            )
        return resolved_value

    def validate_axis_values(
        name: str,
        values: tuple[int, ...],
        *,
        minimum: int,
        maximum: tuple[int, ...] | None = None,
    ) -> None:
        try:
            resolved_values = tuple(values)
        except TypeError as exc:
            raise ValueError(
                f"{name} must contain {len(volume_shape)} integer values, got {values!r}."
            ) from exc
        if len(resolved_values) != len(volume_shape):
            raise ValueError(
                f"{name} must contain {len(volume_shape)} values for shape "
                f"{volume_shape}, got {resolved_values}."
            )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in resolved_values
        ):
            raise ValueError(f"{name} values must be integers, got {resolved_values}.")
        if any(int(value) < minimum for value in resolved_values):
            qualifier = "positive" if minimum == 1 else "non-negative"
            raise ValueError(
                f"{name} values must be {qualifier}, got {resolved_values}."
            )
        if maximum is not None and any(
            int(value) > maximum[axis] for axis, value in enumerate(resolved_values)
        ):
            raise ValueError(
                f"{name} values cannot exceed volume shape {maximum}, "
                f"got {resolved_values}."
            )

    def validate_stack_sizes(stage: str, minimum: int, maximum: int) -> None:
        for name, value in (
            (f"{stage}_min_stack_size", minimum),
            (f"{stage}_max_stack_size", maximum),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) <= 0
            ):
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if int(minimum) > int(maximum):
            raise ValueError(
                f"{stage}_min_stack_size cannot exceed {stage}_max_stack_size, "
                f"got {minimum} > {maximum}."
            )

    def validate_transform(name: str, transform: Transform | list[Transform]) -> None:
        if type(transform) is Transform:
            transforms = [transform]
        elif type(transform) is list:
            transforms = transform
        else:
            raise ValueError(
                f"{name} must be a Transform or list of Transform objects."
            )
        for configured_transform in transforms:
            if type(configured_transform) is not Transform:
                raise ValueError(f"{name} must contain only Transform objects.")
            if not isinstance(configured_transform.type, TransformType):
                raise ValueError(
                    f"{name} contains an unknown transform type: "
                    f"{configured_transform.type!r}."
                )
            if not isinstance(configured_transform.mode, TransformMode):
                raise ValueError(
                    f"{name} contains an unknown transform mode: "
                    f"{configured_transform.mode!r}."
                )

    noise_model = coerce_enum(profile.noise_model, NoiseModel, "noise_model")
    profile.noise_model = noise_model
    reference_schedule_mode = coerce_enum(
        profile.reference_schedule_mode,
        ReferenceScheduleMode,
        "reference_schedule_mode",
    )
    profile.reference_schedule_mode = reference_schedule_mode
    if reference_schedule_mode in {
        ReferenceScheduleMode.GENERATED,
        ReferenceScheduleMode.BALANCED,
        ReferenceScheduleMode.SPARSE,
    }:
        validate_finite_nonnegative(
            "reference_shift_density",
            profile.reference_shift_density,
        )
        schedule_density = profile.reference_schedule_density
        if (
            isinstance(schedule_density, (bool, np.bool_))
            or not isinstance(schedule_density, (int, np.integer))
            or int(schedule_density) <= 0
        ):
            raise ValueError(
                "reference_schedule_density must be a positive integer, "
                f"got {schedule_density!r}."
            )

    for stage in ("ht", "wiener"):
        validate_axis_values(
            f"{stage}_block_size",
            getattr(profile, f"{stage}_block_size"),
            minimum=1,
            maximum=volume_shape,
        )
        validate_axis_values(
            f"{stage}_step",
            getattr(profile, f"{stage}_step"),
            minimum=1,
        )
        validate_axis_values(
            f"{stage}_search_window",
            getattr(profile, f"{stage}_search_window"),
            minimum=0,
        )
        validate_stack_sizes(
            stage,
            getattr(profile, f"{stage}_min_stack_size"),
            getattr(profile, f"{stage}_max_stack_size"),
        )
        validate_transform(
            f"{stage}_transform",
            getattr(profile, f"{stage}_transform"),
        )
        validate_finite_nonnegative(
            f"{stage}_match_threshold",
            getattr(profile, f"{stage}_match_threshold"),
        )
        validate_finite_nonnegative(
            f"{stage}_kaiser_beta",
            getattr(profile, f"{stage}_kaiser_beta"),
        )

    validate_finite_nonnegative(
        "ht_lambda_threshold",
        profile.ht_lambda_threshold,
    )
    validate_finite_nonnegative(
        "wiener_variance_scale",
        profile.wiener_variance_scale,
    )
    ref_batch_size = profile.ref_batch_size
    if (
        isinstance(ref_batch_size, (bool, np.bool_))
        or not isinstance(ref_batch_size, (int, np.integer))
        or int(ref_batch_size) <= 0
    ):
        raise ValueError(
            f"ref_batch_size must be a positive integer, got {ref_batch_size!r}."
        )

    configured_blockmatch_distance = coerce_enum(
        profile.blockmatch_distance,
        BlockMatchDistance,
        "blockmatch_distance",
    )
    profile.blockmatch_distance = configured_blockmatch_distance
    blockmatch_distance = resolve_blockmatch_distance(
        configured_blockmatch_distance,
        noise_model,
    )
    is_poisson = noise_model is NoiseModel.POISSON
    effective_nf = None
    conventional_nf = False

    if isinstance(profile.k, bool) or not isinstance(profile.k, (int, np.integer)):
        raise ValueError(f"k must be a non-negative integer, got {profile.k!r}.")
    if int(profile.k) < 0:
        raise ValueError(f"k must be a non-negative integer, got {profile.k!r}.")
    variance_k = int(profile.k)

    operator_transform_types = {
        TransformType.STARLET,
        TransformType.ATROUS,
        TransformType.B3SPLINE,
        TransformType.CURVELET,
    }
    for stage in ("ht", "wiener"):
        model_name = f"{stage}_weight_model"
        domain_name = f"{stage}_weight_domain"
        scope_name = f"{stage}_weight_scope"
        model = getattr(profile, model_name)
        domain = getattr(profile, domain_name)
        scope = getattr(profile, scope_name)
        model = coerce_enum(model, AggregationWeightModel, model_name)
        domain = coerce_enum(domain, AggregationWeightDomain, domain_name)
        scope = coerce_enum(scope, AggregationWeightScope, scope_name)
        setattr(profile, model_name, model)
        setattr(profile, domain_name, domain)
        setattr(profile, scope_name, scope)
        valid_models = {
            AggregationWeightModel.CLASSIC,
            AggregationWeightModel.VARIANCE,
        }
        if stage == "wiener":
            valid_models.add(AggregationWeightModel.RISK)
        if model not in valid_models:
            expected = "', '".join(sorted(valid_models))
            raise ValueError(
                f"Unknown {model_name}: {model!r} (expected '{expected}')."
            )
        if domain not in {
            AggregationWeightDomain.COEFFICIENT,
            AggregationWeightDomain.WINDOWED_SYNTHESIS,
        }:
            raise ValueError(
                f"{domain_name} must be 'coefficient' or 'windowed_synthesis' "
                f"for {model_name}, got {domain!r}."
            )
        if scope not in {
            AggregationWeightScope.GROUP,
            AggregationWeightScope.PATCH,
        }:
            raise ValueError(
                f"{scope_name} must be 'group' or 'patch' for {model_name}, "
                f"got {scope!r}."
            )

        configured = getattr(profile, f"{stage}_transform")
        transforms = [configured] if type(configured) is Transform else configured
        needs_all_synthesis = domain is AggregationWeightDomain.WINDOWED_SYNTHESIS
        needs_group_synthesis = scope is AggregationWeightScope.PATCH
        for transform in transforms:
            applies_to_group = transform.mode in {TransformMode.ND, TransformMode.GROUP}
            if not needs_all_synthesis and not (
                needs_group_synthesis and applies_to_group
            ):
                continue
            if transform.type in operator_transform_types:
                raise ValueError(
                    f"{stage.upper()} {domain}/{scope} weighting requires "
                    "matrix-based transforms."
                )
            if transform.type is TransformType.FFT:
                raise ValueError(
                    f"{stage.upper()} {domain}/{scope} weighting requires "
                    "real matrix transforms."
                )

    if profile.sharpen_alpha != 1.0 or profile.sharpen_alpha_3d != 1.0:
        raise ValueError("Aggregation weighting requires sharpening exponents of 1.")

    gamma = float(profile.gamma)
    if not np.isfinite(gamma) or gamma < 0.0:
        raise ValueError(f"gamma must be finite and non-negative, got {profile.gamma}.")

    if is_poisson:
        poisson_mode_types = {
            "poisson_ht_match_policy": PoissonHTMatchPolicy,
            "poisson_wiener_gain_mode": PoissonWienerGainMode,
            "poisson_group_mass_conservation": PoissonGroupMassConservation,
        }
        for name, enum_type in poisson_mode_types.items():
            value = coerce_enum(getattr(profile, name), enum_type, name)
            setattr(profile, name, value)
        profile.poisson_variance_source_wiener = coerce_enum(
            profile.poisson_variance_source_wiener,
            PoissonVarianceSource,
            "poisson_variance_source_wiener",
        )
        profile.poisson_wiener_match_source = coerce_enum(
            profile.poisson_wiener_match_source,
            PoissonVarianceSource,
            "poisson_wiener_match_source",
        )
        count_scale = float(profile.poisson_count_scale)
        if not np.isfinite(count_scale) or count_scale <= 0.0:
            raise ValueError(
                "poisson_count_scale must be finite and positive, "
                f"got {profile.poisson_count_scale}."
            )
        validate_finite_nonnegative(
            "poisson_variance_floor",
            profile.poisson_variance_floor,
        )
        group_mass_mode = profile.poisson_group_mass_conservation
        if group_mass_mode in {
            PoissonGroupMassConservation.WIENER,
            PoissonGroupMassConservation.BOTH,
        }:
            if profile.sharpen_alpha != 1.0 or profile.sharpen_alpha_3d != 1.0:
                raise ValueError(
                    "Wiener group-mass conservation requires sharpening exponents of 1."
                )
            configured_wiener = profile.wiener_transform
            wiener_transforms = (
                [configured_wiener]
                if type(configured_wiener) is Transform
                else configured_wiener
            )
            if profile.wiener_weight_model in {
                AggregationWeightModel.VARIANCE,
                AggregationWeightModel.RISK,
            } and any(
                transform.type is TransformType.FFT for transform in wiener_transforms
            ):
                raise ValueError(
                    "Mass-projected Poisson Wiener weighting requires real matrix transforms."
                )
        for stage, enabled in (
            (
                "ht",
                group_mass_mode
                in {
                    PoissonGroupMassConservation.HT,
                    PoissonGroupMassConservation.BOTH,
                },
            ),
            (
                "wiener",
                group_mass_mode
                in {
                    PoissonGroupMassConservation.WIENER,
                    PoissonGroupMassConservation.BOTH,
                },
            ),
        ):
            if not enabled:
                continue
            configured = getattr(profile, f"{stage}_transform")
            transforms = [configured] if type(configured) is Transform else configured
            if any(
                transform.type in operator_transform_types for transform in transforms
            ):
                raise ValueError(
                    f"{stage.upper()} group-mass conservation requires matrix-based transforms."
                )
            projected_risk = getattr(
                profile,
                f"{stage}_weight_model",
            ) in {
                AggregationWeightModel.VARIANCE,
                AggregationWeightModel.RISK,
            }
            if projected_risk and any(
                transform.type is TransformType.FFT for transform in transforms
            ):
                raise ValueError(
                    f"Mass-projected Poisson {stage.upper()} weighting requires "
                    "real matrix transforms."
                )
        max_count = profile.poisson_ht_match_moment_max_count
        if (
            isinstance(max_count, bool)
            or not isinstance(max_count, (int, np.integer))
            or int(max_count) <= 0
        ):
            raise ValueError(
                "poisson_ht_match_moment_max_count must be a positive integer, "
                f"got {max_count!r}."
            )
        match_policy = profile.poisson_ht_match_policy
        if match_policy is not PoissonHTMatchPolicy.FIXED:
            if blockmatch_distance not in {
                BlockMatchDistance.POISSON_DEVIANCE,
                BlockMatchDistance.PEARSON,
                BlockMatchDistance.ANSCOMBE_SSD,
            }:
                raise ValueError(
                    f"Poisson HT match policy {match_policy!r} requires "
                    "'poisson_deviance', "
                    f"'pearson', or 'anscombe_ssd', got {blockmatch_distance!r}."
                )
        if match_policy is PoissonHTMatchPolicy.REFERENCE_FINITE_COUNT:
            finite_count_values = (
                (
                    "poisson_ht_match_structure_beta",
                    float(profile.poisson_ht_match_structure_beta),
                    False,
                ),
                (
                    "poisson_ht_match_intensity_power",
                    profile.poisson_ht_match_intensity_power,
                    True,
                ),
            )
            for name, value, positive in finite_count_values:
                if not np.isfinite(value) or value < 0.0 or (positive and value == 0.0):
                    qualifier = "positive" if positive else "non-negative"
                    raise ValueError(
                        f"{name} must be finite and {qualifier}, got {value}."
                    )
    else:
        effective_nf, conventional_nf = resolve_nf(profile.nf, volume_shape)
        if conventional_nf:
            variance_k = 0

    return ResolvedProfile(
        is_poisson=is_poisson,
        effective_nf=effective_nf,
        conventional_nf=conventional_nf,
        blockmatch_distance=blockmatch_distance,
        variance_k=variance_k,
    )
