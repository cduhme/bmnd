from dataclasses import dataclass, field

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
)
from .transforms import Transform, TransformMode, TransformType


@dataclass
class BMNDProfile:
    """
    BMNDProfile object, containing the default settings for BMND.
    """

    # ---------- General ----------
    # "gaussian" or "poisson"
    noise_model: NoiseModel = NoiseModel.GAUSSIAN

    # ---------- Shift parameters ----------
    #   "official"          : use the official lookup-table-compatible reference schedule
    #   "generated"         : generate an ND reference schedule from the configured step sizes
    #   "balanced"          : couple shifts symmetrically across every axis with fixed phase cost
    #   "sparse"            : run the asymmetric hierarchy over non-singleton axes only
    #   "off"               : use one unshifted stepped lattice with explicit boundary coverage
    reference_schedule_mode: ReferenceScheduleMode = ReferenceScheduleMode.GENERATED
    # Density multiplier for generated shift locations; higher values add more per-axis offsets
    reference_shift_density: float = 2.0
    # Density multiplier for generated hierarchical schedule phases; higher values emit more references
    reference_schedule_density: int = 2

    # ---------- Poisson ----------
    # Wiener variance can be estimated from the observation or denoised HT pilot.
    poisson_variance_source_wiener: PoissonVarianceSource = PoissonVarianceSource.PILOT
    # Minimum variance used to stabilize near-zero Poisson intensities
    poisson_variance_floor: float = 1e-10
    # Gain g in Var(Y)=g*E[Y]; equivalent raw counts are Y/g
    poisson_count_scale: float = 1.0
    #   "fixed"                  : apply ht_match_threshold directly to the raw distance
    #   "reference_finite_count" : derive a reference-null threshold in raw-distance units
    #   "candidate_standardized" : standardize every candidate distance by its null moments
    poisson_ht_match_policy: PoissonHTMatchPolicy = PoissonHTMatchPolicy.FIXED
    # Wiener matching source; the BM3D/BM4D baseline matches on the HT pilot.
    poisson_wiener_match_source: PoissonVarianceSource = PoissonVarianceSource.PILOT
    # Structural allowance for HT reference-finite-count matching
    poisson_ht_match_structure_beta: float = 0.1
    # Exponent applied to reference mean equivalent counts in that allowance
    poisson_ht_match_intensity_power: float = 1.0
    # Largest pooled count evaluated exactly before using asymptotic moments
    poisson_ht_match_moment_max_count: int = 512
    #   "classic"           : use the standard plug-in Wiener gain from pilot power
    #   "noise-floor"       : subtract raw Poisson noise power before shrinkage
    #   "variance-scaled"   : subtract scaled Wiener noise power before shrinkage
    poisson_wiener_gain_mode: PoissonWienerGainMode = (
        PoissonWienerGainMode.VARIANCE_SCALED
    )
    # Preserve normalized aggregation contributions with constant group corrections.
    #   "none"              : do not preserve noisy mass within transformed groups
    #   "ht"                : preserve group mass after HT shrinkage
    #   "wiener"            : preserve group mass after Wiener shrinkage
    #   "both"              : preserve group mass in both filtering stages
    poisson_group_mass_conservation: PoissonGroupMassConservation = (
        PoissonGroupMassConservation.NONE
    )

    # ---------- Noise variance and covariance ----------
    # Per-axis local-variance domain; None/() disables it and zero selects conventional mode
    nf: tuple[int, ...] | int | None = field(default_factory=tuple)
    # Exact nonlocal covariance planes; zero is fully approximate, group size is fully exact
    k: int = 4

    # ---------- Aggregation weighting ----------
    #   "classic"           : weight by retained filter energy
    #   "variance"          : weight by transmitted coefficient noise variance
    ht_weight_model: AggregationWeightModel = AggregationWeightModel.VARIANCE
    #   "coefficient"       : exclude inverse spatial synthesis and Kaiser window energy
    #   "windowed_synthesis": evaluate after inverse synthesis and Kaiser windowing
    ht_weight_domain: AggregationWeightDomain = AggregationWeightDomain.COEFFICIENT
    #   "group"             : share one HT reliability weight across the matched group
    #   "patch"             : compute one marginal HT reliability weight per output patch
    ht_weight_scope: AggregationWeightScope = AggregationWeightScope.PATCH
    #   "classic"           : weight by Wiener gain energy
    #   "variance"          : weight by transmitted coefficient noise variance
    #   "risk"              : include transmitted noise and rejected-signal error
    wiener_weight_model: AggregationWeightModel = AggregationWeightModel.VARIANCE
    #   "coefficient"       : exclude inverse spatial synthesis and Kaiser window energy
    #   "windowed_synthesis": evaluate after inverse synthesis and Kaiser windowing
    wiener_weight_domain: AggregationWeightDomain = AggregationWeightDomain.COEFFICIENT
    #   "group"             : share one Wiener reliability weight across the matched group
    #   "patch"             : compute one marginal Wiener reliability weight per output patch
    wiener_weight_scope: AggregationWeightScope = AggregationWeightScope.PATCH

    # ---------- Refiltering ----------
    # Perform residual thresholding and re-denoising
    denoise_residual: bool = False
    # Threshold multiplier used when denoising the residual signal
    residual_threshold: float = 3.0
    # Maximum required pad size (= half of the kernel size), or 0 -> use image size
    max_pad_size: int = 0

    # ---------- Block matching ----------
    # HT SSD block-matching noise-bias correction factor
    gamma: float = 1.0
    #   "auto"              : SSD for Gaussian, Poisson deviance for Poisson
    #   "ssd"               : raw SSD (BM3D/BM4D-compatible)
    #   "poisson_deviance"  : mean symmetric Poisson likelihood-ratio deviance
    #   "pearson"           : mean Poisson Pearson chi-square distance
    #   "anscombe_ssd"      : mean SSD after the Anscombe variance-stabilizing transform
    blockmatch_distance: BlockMatchDistance = BlockMatchDistance.AUTO

    # ---------- Hard-thresholding (HT) parameters ----------
    # If True, use soft-thresholding instead of hard-thresholding in the HT stage
    ht_use_soft_thresholding: bool = False
    # Patch size used in the hard-thresholding stage
    ht_block_size: tuple[int, ...] = field(default_factory=tuple)
    # Stride between reference patches in the hard-thresholding stage
    ht_step: tuple[int, ...] = field(default_factory=tuple)
    # Search neighborhood size for block matching in the hard-thresholding stage
    ht_search_window: tuple[int, ...] = field(default_factory=tuple)
    # Maximum number of matched patches retained per HT group
    ht_max_stack_size: int = field(default_factory=int)
    # Minimum number of patches retained per HT group once block matching completes
    ht_min_stack_size: int = 2
    # Dimensionless HT block-matching threshold; SSD scales it by the squared data range
    ht_match_threshold: float = 2.9527
    # Threshold multiplier applied to HT transform coefficients before shrinkage
    ht_lambda_threshold: float = 1.0
    # Kaiser window beta used during HT-stage aggregation
    ht_kaiser_beta: float = 2.0

    # ---------- Wiener filtering parameters ----------
    # Patch size used in the Wiener stage
    wiener_block_size: tuple[int, ...] = field(default_factory=tuple)
    # Stride between reference patches in the Wiener stage
    wiener_step: tuple[int, ...] = field(default_factory=tuple)
    # Search neighborhood size for block matching in the Wiener stage
    wiener_search_window: tuple[int, ...] = field(default_factory=tuple)
    # Maximum number of matched patches retained per Wiener group
    wiener_max_stack_size: int = field(default_factory=int)
    # Minimum number of patches retained per Wiener group once block matching completes
    wiener_min_stack_size: int = 2
    # Dimensionless Wiener block-matching threshold; SSD scales it by the squared data range
    wiener_match_threshold: float = 0.7689
    # Scales the noise-variance term in the Wiener gain denominator
    wiener_variance_scale: float = 1.0
    # Kaiser window beta used during Wiener-stage aggregation
    wiener_kaiser_beta: float = 2.0

    # ---------- Transforms ----------
    # Transform configuration used in the HT stage (typically local + group transforms)
    ht_transform: Transform | list[Transform] = field(
        default_factory=lambda: Transform(TransformType.BIOR15, TransformMode.ND)
    )
    # Transform configuration used in the Wiener stage (typically local + group transforms)
    wiener_transform: Transform | list[Transform] = field(
        default_factory=lambda: Transform(TransformType.DCT, TransformMode.ND)
    )

    # ---------- Other stuff ----------
    # If not equal to 1, sharpening will be done with power 1/sharpen_alpha
    sharpen_alpha: float = 1.0
    # Sharpening parameter for the "3rd dimension DC" of the group spectrum
    sharpen_alpha_3d: float = 1.0

    # Number of reference patches to process in a single vectorized batch
    ref_batch_size: int = 64


@dataclass
class BM2DProfile(BMNDProfile):
    """Template profile for one-dimensional BMND (BM2D) denoising."""

    # ---------- General ----------
    noise_model: NoiseModel = NoiseModel.GAUSSIAN

    # ---------- Block matching ----------
    gamma: float = 3.0
    nf: tuple[int] | int | None = (32,)
    k: int = 4

    # ---------- Hard-thresholding (HT) parameters ----------
    ht_use_soft_thresholding: bool = False
    ht_block_size: tuple[int] = (64,)
    ht_step: tuple[int] = (16,)
    ht_search_window: tuple[int] = (128,)
    ht_max_stack_size: int = 16
    ht_min_stack_size: int = 2
    ht_match_threshold: float = 2.9527
    ht_lambda_threshold: float = 3.0
    ht_kaiser_beta: float = 2.0

    # ---------- Wiener filtering parameters ----------
    wiener_block_size: tuple[int] = (64,)
    wiener_step: tuple[int] = (16,)
    wiener_search_window: tuple[int] = (128,)
    wiener_max_stack_size: int = 32
    wiener_min_stack_size: int = 2
    wiener_match_threshold: float = 0.3937
    wiener_variance_scale: float = 0.4
    wiener_kaiser_beta: float = 2.0

    # ---------- Transforms ----------
    ht_transform: Transform | list[Transform] = field(
        default_factory=lambda: [
            Transform(TransformType.HAAR, TransformMode.GROUP),
            Transform(TransformType.BIOR15, TransformMode.SPATIAL),
        ]
    )
    wiener_transform: Transform | list[Transform] = field(
        default_factory=lambda: [
            Transform(TransformType.HAAR, TransformMode.GROUP),
            Transform(TransformType.DCT, TransformMode.SPATIAL),
        ]
    )


@dataclass
class BM3DProfile(BMNDProfile):
    """Profile using the official BM3D compatibility parameter values."""

    # ---------- General ----------
    noise_model: NoiseModel = NoiseModel.GAUSSIAN

    # ---------- Block matching ----------
    gamma: float = 3.0
    nf: tuple[int, int] | int | None = (32, 32)
    k: int = 4

    # ---------- Aggregation weighting ----------
    # Modern BM3D propagates coefficient variance to each synthesized patch
    ht_weight_model: AggregationWeightModel = AggregationWeightModel.VARIANCE
    ht_weight_domain: AggregationWeightDomain = AggregationWeightDomain.COEFFICIENT
    ht_weight_scope: AggregationWeightScope = AggregationWeightScope.PATCH
    wiener_weight_model: AggregationWeightModel = AggregationWeightModel.VARIANCE
    wiener_weight_domain: AggregationWeightDomain = AggregationWeightDomain.COEFFICIENT
    wiener_weight_scope: AggregationWeightScope = AggregationWeightScope.PATCH

    # ---------- Hard-thresholding (HT) parameters ----------
    ht_use_soft_thresholding: bool = False
    ht_block_size: tuple[int, int] = (8, 8)
    ht_step: tuple[int, int] = (3, 3)
    ht_search_window: tuple[int, int] = (19, 19)
    ht_max_stack_size: int = 16
    ht_min_stack_size: int = 2
    # Equivalent to the official BM3D 8-bit threshold 3000 after scaling by 8x8 patch area and 255^2
    ht_match_threshold: float = 2.9527
    # Official scalar-sigma BM3D/BM4D wrappers resolve to lambda=3.0
    ht_lambda_threshold: float = 3.0
    ht_kaiser_beta: float = 2.0

    # ---------- Wiener filtering parameters ----------
    wiener_block_size: tuple[int, int] = (8, 8)
    wiener_step: tuple[int, int] = (3, 3)
    wiener_search_window: tuple[int, int] = (19, 19)
    wiener_max_stack_size: int = 32
    wiener_min_stack_size: int = 2
    # Equivalent to the official BM3D 8-bit threshold 400 after scaling by 8x8 patch area and 255^2
    wiener_match_threshold: float = 0.3937
    # Official white-Gaussian mu2 value; scales noise variance directly
    wiener_variance_scale: float = 0.4
    wiener_kaiser_beta: float = 2.0

    # ---------- Transforms ----------
    ht_transform: Transform | list[Transform] = field(
        default_factory=lambda: [
            Transform(TransformType.HAAR, TransformMode.GROUP),
            Transform(TransformType.BIOR15, TransformMode.SPATIAL),
        ]
    )
    wiener_transform: Transform | list[Transform] = field(
        default_factory=lambda: [
            Transform(TransformType.HAAR, TransformMode.GROUP),
            Transform(TransformType.DCT, TransformMode.SPATIAL),
        ]
    )

    # ---------- Other stuff ----------
    sharpen_alpha = 1.0
    sharpen_alpha_3d = 1.0
    ref_batch_size: int = 64


@dataclass
class BM4DProfile(BMNDProfile):
    """Profile using the official BM4D compatibility parameter values."""

    # ---------- General ----------
    noise_model: NoiseModel = NoiseModel.GAUSSIAN

    # ---------- Block matching ----------
    gamma: float = 3.0
    nf: tuple[int, int, int] | int | None = (16, 16, 16)
    k: int = 4

    # ---------- Aggregation weighting ----------
    # Modern BM4D propagates coefficient variance to each synthesized patch
    ht_weight_model: AggregationWeightModel = AggregationWeightModel.VARIANCE
    ht_weight_domain: AggregationWeightDomain = AggregationWeightDomain.COEFFICIENT
    ht_weight_scope: AggregationWeightScope = AggregationWeightScope.PATCH
    wiener_weight_model: AggregationWeightModel = AggregationWeightModel.VARIANCE
    wiener_weight_domain: AggregationWeightDomain = AggregationWeightDomain.COEFFICIENT
    wiener_weight_scope: AggregationWeightScope = AggregationWeightScope.PATCH

    # ---------- Hard-thresholding (HT) parameters ----------
    ht_use_soft_thresholding: bool = False
    ht_block_size: tuple[int, int, int] = (4, 4, 4)
    ht_step: tuple[int, int, int] = (3, 3, 3)
    ht_search_window: tuple[int, int, int] = (7, 7, 7)
    ht_max_stack_size: int = 16
    ht_min_stack_size: int = 2
    ht_match_threshold: float = 2.9527
    # Official scalar-sigma BM4D wrapper resolves to lambda=3.0
    ht_lambda_threshold: float = 3.0
    ht_kaiser_beta: float = 2.0

    # ---------- Wiener filtering parameters ----------
    wiener_block_size: tuple[int, int, int] = (5, 5, 5)
    wiener_step: tuple[int, int, int] = (3, 3, 3)
    wiener_search_window: tuple[int, int, int] = (7, 7, 7)
    wiener_max_stack_size: int = 32
    wiener_min_stack_size: int = 2
    wiener_match_threshold: float = 0.7689
    # Official white-Gaussian mu2 value; scales noise variance directly
    wiener_variance_scale: float = 0.4
    wiener_kaiser_beta: float = 2.0

    # ---------- Transforms ----------
    ht_transform: Transform | list[Transform] = field(
        default_factory=lambda: [
            Transform(TransformType.HAAR, TransformMode.GROUP),
            Transform(TransformType.BIOR15, TransformMode.SPATIAL),
        ]
    )
    wiener_transform: Transform | list[Transform] = field(
        default_factory=lambda: [
            Transform(TransformType.HAAR, TransformMode.GROUP),
            Transform(TransformType.DCT, TransformMode.SPATIAL),
        ]
    )

    # ---------- Other stuff ----------
    sharpen_alpha = 1.0
    sharpen_alpha_3d = 1.0
    ref_batch_size: int = 64


@dataclass
class BM5DProfile(BMNDProfile):
    """Template profile for four-dimensional BMND (BM5D) denoising."""

    # ---------- General ----------
    noise_model: NoiseModel = NoiseModel.GAUSSIAN

    # ---------- Block matching ----------
    gamma: float = 3.0
    nf: tuple[int, int, int, int] | int | None = (16, 16, 16, 16)

    # ---------- Hard-thresholding (HT) parameters ----------
    ht_use_soft_thresholding: bool = False
    ht_block_size: tuple[int, int, int, int] = (4, 4, 4, 4)
    ht_step: tuple[int, int, int, int] = (3, 3, 3, 3)
    ht_search_window: tuple[int, int, int, int] = (3, 3, 7, 7)
    ht_max_stack_size: int = 16
    ht_min_stack_size: int = 2
    ht_match_threshold: float = 2.9527
    # Official scalar-sigma BM4D wrapper resolves to lambda=3.0
    ht_lambda_threshold: float = 3.0
    ht_kaiser_beta: float = 2.0

    # ---------- Wiener filtering parameters ----------
    wiener_block_size: tuple[int, int, int, int] = (4, 4, 4, 4)
    wiener_step: tuple[int, int, int, int] = (3, 3, 3, 3)
    wiener_search_window: tuple[int, int, int, int] = (3, 3, 7, 7)
    wiener_max_stack_size: int = 32
    wiener_min_stack_size: int = 2
    wiener_match_threshold: float = 0.7689
    wiener_variance_scale: float = 1.0
    wiener_kaiser_beta: float = 2.0

    # ---------- Transforms ----------
    ht_transform: Transform | list[Transform] = field(
        default_factory=lambda: [
            Transform(TransformType.HAAR, TransformMode.GROUP),
            Transform(TransformType.BIOR15, TransformMode.SPATIAL),
        ]
    )
    wiener_transform: Transform | list[Transform] = field(
        default_factory=lambda: [
            Transform(TransformType.HAAR, TransformMode.GROUP),
            Transform(TransformType.DCT, TransformMode.SPATIAL),
        ]
    )

    # ---------- Other stuff ----------
    sharpen_alpha = 1.0
    sharpen_alpha_3d = 1.0
    ref_batch_size: int = 64
