from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import bmnd.bmndalgo as bmndalgo_module
import bmnd.cache as cache_module
from bmnd import BM2DProfile
from bmnd.bmndalgo import bmnd
from bmnd.profiles import BMNDProfile
from bmnd.psd import scalar_sigma_to_psd
from bmnd.transforms import Transform, TransformMode, TransformType


def _make_profile(
    noise_model: str,
) -> BMNDProfile:
    profile = BMNDProfile(
        noise_model=noise_model,
        ht_block_size=(4, 4),
        ht_step=(4, 4),
        ht_search_window=(1, 1),
        ht_max_stack_size=4,
        wiener_block_size=(4, 4),
        wiener_step=(4, 4),
        wiener_search_window=(1, 1),
        wiener_max_stack_size=4,
    )
    profile.ht_transform = Transform(TransformType.DCT, TransformMode.ND)
    profile.wiener_transform = Transform(TransformType.DCT, TransformMode.ND)
    profile.ref_batch_size = 8
    return profile


def test_bm2d_default_transforms_reconstruct_noiseless_signal() -> None:
    profile = BM2DProfile()
    # Disable the Wiener variance floor as well as the supplied noise.
    profile.wiener_variance_scale = 0.0
    signal = (2.0 + np.sin(np.linspace(0.0, 4.0 * np.pi, 128))).astype(np.float32)
    result = bmnd(signal, profile, sigma=0.0)
    assert result.shape == signal.shape
    np.testing.assert_allclose(result, signal, atol=1e-6)


@pytest.mark.parametrize("value", [-1, 1.5, "large", True])
def test_bmnd_rejects_invalid_max_cache_bytes(value: object) -> None:
    profile = _make_profile("gaussian")

    with pytest.raises(ValueError, match="max_cache_bytes"):
        bmnd(
            np.ones((8, 8), dtype=np.float32),
            profile,
            sigma=0.1,
            max_cache_bytes=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("noise_model", ["gaussian", "poisson"])
def test_auto_cache_skips_detection_when_managed_caches_are_inactive(
    monkeypatch,
    noise_model: str,
) -> None:
    profile = _make_profile(noise_model)
    profile.k = 0
    volume = np.ones((8, 8), dtype=np.float32)

    def reject_detection() -> int:
        raise AssertionError("inactive managed caches must not inspect system memory")

    monkeypatch.setattr(
        cache_module,
        "_effective_available_memory_bytes",
        reject_detection,
    )

    output = bmnd(volume, profile, sigma=0.1 if noise_model == "gaussian" else None)

    assert output.shape == volume.shape


def test_bmnd_poisson_models_share_one_cache_budget(monkeypatch) -> None:
    profile = _make_profile("poisson")
    profile.k = 0
    volume = np.ones((8, 8), dtype=np.float32)
    captured_limits = []
    original = bmndalgo_module.build_poisson_variance_model

    def capture_limits(*args, **kwargs):
        captured_limits.append(kwargs["cache_limits"])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        bmndalgo_module,
        "build_poisson_variance_model",
        capture_limits,
    )

    bmnd(volume, profile, max_cache_bytes=200 * 1024**2)

    assert len(captured_limits) == 2
    assert sum(
        limits.overlap_max_bytes
        + limits.contribution_max_bytes
        + limits.factorized_max_bytes
        for limits in captured_limits
    ) <= 200 * 1024**2


def test_poisson_cache_budget_does_not_change_output() -> None:
    profile = _make_profile("poisson")
    volume = np.random.default_rng(109).poisson(6.0, size=(8, 8)).astype(np.float32)

    uncached = bmnd(volume, profile, max_cache_bytes=0)
    unlimited = bmnd(volume, profile, max_cache_bytes=None)

    assert np.array_equal(uncached, unlimited)


@pytest.mark.parametrize("stage", ["ht", "wiener"])
@pytest.mark.parametrize("missing_axis", ["domain", "scope"])
def test_bmnd_weighting_requires_both_axes(
    stage: str,
    missing_axis: str,
) -> None:
    profile = _make_profile("gaussian")
    setattr(profile, f"{stage}_weight_model", "variance")
    setattr(
        profile,
        f"{stage}_weight_domain",
        None if missing_axis == "domain" else "coefficient",
    )
    setattr(
        profile,
        f"{stage}_weight_scope",
        None if missing_axis == "scope" else "group",
    )

    with pytest.raises(ValueError, match=f"{stage}_weight_{missing_axis}"):
        bmnd(np.zeros((8, 8), dtype=np.float32), profile, sigma=0.1)


def test_bmnd_rejects_ht_risk_model() -> None:
    profile = _make_profile("gaussian")
    profile.ht_weight_model = "risk"
    profile.ht_weight_domain = "coefficient"
    profile.ht_weight_scope = "group"

    with pytest.raises(ValueError, match="ht_weight_model"):
        bmnd(np.zeros((8, 8), dtype=np.float32), profile, sigma=0.1)


def test_bmnd_rejects_sharpening_with_aggregation_weighting() -> None:
    profile = _make_profile("gaussian")
    profile.wiener_weight_model = "variance"
    profile.wiener_weight_domain = "coefficient"
    profile.wiener_weight_scope = "group"
    profile.sharpen_alpha_3d = 0.9

    with pytest.raises(ValueError, match="requires sharpening exponents of 1"):
        bmnd(np.zeros((8, 8), dtype=np.float32), profile, sigma=0.1)


@pytest.mark.parametrize("noise_model", ["gaussian", "poisson"])
def test_bmnd_return_contract_with_sigma_map_toggle(noise_model: str):
    rng = np.random.default_rng(11)
    volume = (
        rng.poisson(6.0, size=(8, 8))
        if noise_model == "poisson"
        else 0.5 + rng.normal(0.0, 0.1, size=(8, 8))
    ).astype(np.float32)
    profile = _make_profile(noise_model)
    sigma = None if noise_model == "poisson" else 0.1

    out = bmnd(volume, profile, sigma=sigma, return_sigma_map=False)
    out_w_map = bmnd(volume, profile, sigma=sigma, return_sigma_map=True)

    assert isinstance(out, np.ndarray)
    assert out.shape == volume.shape
    assert isinstance(out_w_map, tuple)
    denoised, sigma_maps = out_w_map
    assert denoised.shape == volume.shape
    assert np.array_equal(denoised, out)
    assert set(sigma_maps.keys()) == {"global", "ht", "wiener"}
    assert np.isscalar(sigma_maps["global"])
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    for stage in ("ht", "wiener"):
        assert sigma_maps[stage].shape == volume.shape
        assert np.isfinite(sigma_maps[stage]).all()


@pytest.mark.parametrize("noise_model", ["gaussian", "poisson"])
@pytest.mark.parametrize(
    ("stage", "model"),
    [
        ("ht", "classic"),
        ("ht", "variance"),
        ("wiener", "classic"),
        ("wiener", "variance"),
        ("wiener", "risk"),
    ],
)
@pytest.mark.parametrize("domain", ["coefficient", "windowed_synthesis"])
@pytest.mark.parametrize("scope", ["group", "patch"])
def test_bmnd_diagonal_weight_combinations_reduce_noise(
    noise_model: str,
    stage: str,
    model: str,
    domain: str,
    scope: str,
) -> None:
    rng = np.random.default_rng(71)
    if noise_model == "poisson":
        clean = 5.0
        volume = rng.poisson(5.0, size=(8, 8)).astype(np.float32)
        sigma = None
    else:
        clean = 0.5
        volume = (0.5 + rng.normal(0.0, 0.1, size=(8, 8))).astype(np.float32)
        sigma = 0.1
    profile = _make_profile(noise_model)
    profile.k = 0
    setattr(profile, f"{stage}_weight_model", model)
    setattr(profile, f"{stage}_weight_domain", domain)
    setattr(profile, f"{stage}_weight_scope", scope)

    output = bmnd(volume, profile, sigma=sigma)

    assert output.shape == volume.shape
    assert np.isfinite(output).all()
    assert np.mean((output - clean) ** 2) < np.mean((volume - clean) ** 2)


@pytest.mark.parametrize("model", ["variance", "risk"])
def test_bmnd_poisson_covariance_saturates_at_full_stack_size(model: str):
    rng = np.random.default_rng(29)
    volume = rng.poisson(6.0, size=(8, 8)).astype(np.float32)
    profile = _make_profile("poisson")
    profile.k = 4
    profile.wiener_weight_model = model
    profile.wiener_weight_domain = "windowed_synthesis"
    profile.wiener_weight_scope = "group"

    out, sigma_maps = cast(
        tuple[NDArray[np.float32], dict[str, NDArray[np.float32] | float]],
        bmnd(volume, profile, sigma=None, return_sigma_map=True),
    )

    assert out.shape == volume.shape
    assert np.isfinite(out).all()
    assert np.isfinite(np.asarray(sigma_maps["ht"])).all()
    assert np.isfinite(np.asarray(sigma_maps["wiener"])).all()
    profile.k = 8
    overfull, overfull_maps = bmnd(volume, profile, return_sigma_map=True)
    np.testing.assert_array_equal(overfull, out)
    for stage in ("ht", "wiener"):
        np.testing.assert_array_equal(overfull_maps[stage], sigma_maps[stage])


def test_bmnd_poisson_partial_nonrisk_does_not_build_group_plan(monkeypatch):
    rng = np.random.default_rng(41)
    volume = rng.poisson(6.0, size=(8, 8)).astype(np.float32)
    profile = _make_profile("poisson")
    profile.wiener_weight_model = "variance"
    profile.wiener_weight_domain = "coefficient"
    profile.wiener_weight_scope = "group"
    profile.k = 1

    def reject_plan(*args, **kwargs):
        raise AssertionError("non-risk exact covariance must use the compact k-plane path")

    monkeypatch.setattr(
        bmndalgo_module,
        "build_exact_poisson_group_plan",
        reject_plan,
    )

    output = bmnd(volume, profile, sigma=None)

    assert output.shape == volume.shape
    assert np.isfinite(output).all()


@pytest.mark.parametrize("stage", ["ht", "wiener"])
def test_bmnd_poisson_windowed_group_variance_uses_exact_plan(
    monkeypatch,
    stage: str,
) -> None:
    rng = np.random.default_rng(73)
    volume = rng.poisson(6.0, size=(8, 8)).astype(np.float32)
    profile = _make_profile("poisson")
    setattr(profile, f"{stage}_weight_model", "variance")
    setattr(profile, f"{stage}_weight_domain", "windowed_synthesis")
    setattr(profile, f"{stage}_weight_scope", "group")
    profile.k = 1
    original_plan = bmndalgo_module.build_exact_poisson_group_plan
    observed_shapes = []

    def capture_plan(*args, **kwargs):
        observed_shapes.append(tuple(args[3]))
        return original_plan(*args, **kwargs)

    monkeypatch.setattr(
        bmndalgo_module,
        "build_exact_poisson_group_plan",
        capture_plan,
    )

    output = bmnd(volume, profile, sigma=None)

    assert output.shape == volume.shape
    assert observed_shapes


def test_bmnd_poisson_windowed_patch_uses_k_refined_marginals_without_exact_plan(
    monkeypatch,
) -> None:
    rng = np.random.default_rng(79)
    volume = rng.poisson(6.0, size=(8, 8)).astype(np.float32)
    profile = _make_profile("poisson")
    profile.wiener_weight_model = "variance"
    profile.wiener_weight_domain = "windowed_synthesis"
    profile.wiener_weight_scope = "patch"
    profile.k = 1

    def reject_plan(*args, **kwargs):
        raise AssertionError("Patch scope must use k-refined diagonal marginals")

    monkeypatch.setattr(
        bmndalgo_module,
        "build_exact_poisson_group_plan",
        reject_plan,
    )

    output = bmnd(volume, profile, sigma=None)

    assert output.shape == volume.shape


@pytest.mark.parametrize("noise_input", ["scalar", "flat_psd"])
def test_bmnd_white_gaussian_windowed_group_variance_uses_exact_source_plan(
    monkeypatch,
    noise_input: str,
) -> None:
    rng = np.random.default_rng(89)
    volume = (0.5 + rng.normal(0.0, 0.1, size=(8, 8))).astype(np.float32)
    profile = _make_profile("gaussian")
    profile.wiener_weight_model = "variance"
    profile.wiener_weight_domain = "windowed_synthesis"
    profile.wiener_weight_scope = "group"
    profile.k = 1
    original_plan = bmndalgo_module.build_exact_white_gaussian_group_plan
    original_risk = bmndalgo_module.compute_covariance_aware_group_risk_from_plan
    observed_shapes = []
    observed_risks = []

    def capture_plan(*args, **kwargs):
        observed_shapes.append(tuple(args[4]))
        return original_plan(*args, **kwargs)

    def capture_risk(*args, **kwargs):
        observed_risks.append(kwargs["exact_plane_count"])
        return original_risk(*args, **kwargs)

    monkeypatch.setattr(
        bmndalgo_module,
        "build_exact_white_gaussian_group_plan",
        capture_plan,
    )
    monkeypatch.setattr(
        bmndalgo_module,
        "compute_covariance_aware_group_risk_from_plan",
        capture_risk,
    )

    if noise_input == "scalar":
        output = bmnd(volume, profile, sigma=0.1)
    else:
        output = bmnd(
            volume,
            profile,
            sigma_psd=scalar_sigma_to_psd(volume.shape, 0.1),
        )

    assert output.shape == volume.shape
    assert observed_shapes
    assert observed_risks and set(observed_risks) == {1}


def test_bmnd_colored_gaussian_windowed_group_uses_diagonal_covariance(
    monkeypatch,
) -> None:
    rng = np.random.default_rng(97)
    volume = (0.5 + rng.normal(0.0, 0.1, size=(8, 8))).astype(np.float32)
    profile = _make_profile("gaussian")
    profile.wiener_weight_model = "variance"
    profile.wiener_weight_domain = "windowed_synthesis"
    profile.wiener_weight_scope = "group"
    profile.k = 1
    frequencies = np.fft.fftfreq(8)
    fy, fx = np.meshgrid(frequencies, frequencies, indexing="ij")
    sigma_psd = np.asarray(1.0 + 4.0 * (fx**2 + fy**2), dtype=np.float32)

    def reject_exact_risk(*args, **kwargs):
        raise AssertionError("Colored Gaussian weighting must use diagonal covariance")

    monkeypatch.setattr(
        bmndalgo_module,
        "compute_covariance_aware_group_risk_from_plan",
        reject_exact_risk,
    )

    output = bmnd(volume, profile, sigma_psd=sigma_psd)

    assert output.shape == volume.shape


def test_bmnd_poisson_partial_risk_uses_limited_sigma_and_group_plan(monkeypatch):
    rng = np.random.default_rng(43)
    volume = rng.poisson(6.0, size=(8, 8)).astype(np.float32)
    profile = _make_profile("poisson")
    profile.wiener_weight_model = "risk"
    profile.wiener_weight_domain = "windowed_synthesis"
    profile.wiener_weight_scope = "group"
    profile.k = 1
    original_build_plan = bmndalgo_module.build_exact_poisson_group_plan
    original_gain = bmndalgo_module.compute_poisson_wiener_gain
    original_risk = bmndalgo_module.compute_covariance_aware_group_risk_from_plan
    plans = []
    gain_sigmas = []
    risk_sigmas = []

    def capture_plan(*args, **kwargs):
        plan = original_build_plan(*args, **kwargs)
        plans.append(plan)
        return plan

    def capture_gain(*args, **kwargs):
        gain_sigmas.append(np.asarray(args[1], dtype=np.float32).copy())
        return original_gain(*args, **kwargs)

    def capture_risk(*args, **kwargs):
        risk_sigmas.append(np.asarray(kwargs["noise_sigma"], dtype=np.float32).copy())
        assert kwargs["exact_plane_count"] == 1
        return original_risk(*args, **kwargs)

    monkeypatch.setattr(
        bmndalgo_module,
        "build_exact_poisson_group_plan",
        capture_plan,
    )
    monkeypatch.setattr(bmndalgo_module, "compute_poisson_wiener_gain", capture_gain)
    monkeypatch.setattr(
        bmndalgo_module,
        "compute_covariance_aware_group_risk_from_plan",
        capture_risk,
    )

    output = bmnd(volume, profile, sigma=None)

    assert output.shape == volume.shape
    assert np.isfinite(output).all()
    assert plans
    assert gain_sigmas
    assert len(gain_sigmas) == len(risk_sigmas)
    assert all(
        np.allclose(gain_sigma, risk_sigma)
        for gain_sigma, risk_sigma in zip(gain_sigmas, risk_sigmas, strict=True)
    )
    assert all(plan.contributions_cache is None for plan in plans)


def test_bmnd_poisson_k_zero_omits_exact_metadata_and_plan(monkeypatch):
    rng = np.random.default_rng(47)
    volume = rng.poisson(6.0, size=(8, 8)).astype(np.float32)
    profile = _make_profile("poisson")
    profile.wiener_weight_model = "risk"
    profile.wiener_weight_domain = "windowed_synthesis"
    profile.wiener_weight_scope = "group"
    profile.k = 0
    metadata_flags = []
    observed_k = []
    original_blockmatch = bmndalgo_module.blockmatch_groups
    original_variance = bmndalgo_module.get_direct_poisson_noise_std

    def capture_blockmatch(*args, **kwargs):
        metadata_flags.append(kwargs["include_position_metadata"])
        return original_blockmatch(*args, **kwargs)

    def capture_variance(*args, **kwargs):
        observed_k.append(kwargs["k"])
        return original_variance(*args, **kwargs)

    def reject_plan(*args, **kwargs):
        raise AssertionError("k=0 must not build an exact Poisson plan")

    monkeypatch.setattr(bmndalgo_module, "blockmatch_groups", capture_blockmatch)
    monkeypatch.setattr(
        bmndalgo_module,
        "get_direct_poisson_noise_std",
        capture_variance,
    )
    monkeypatch.setattr(
        bmndalgo_module,
        "build_exact_poisson_group_plan",
        reject_plan,
    )

    output = bmnd(volume, profile, sigma=None)

    assert output.shape == volume.shape
    assert metadata_flags == [False, False]
    assert observed_k and set(observed_k) == {0}


def test_bmnd_poisson_direct_path_rejects_sigma_psd():
    rng = np.random.default_rng(18)
    volume = rng.poisson(4.0, size=(8, 8)).astype(np.float32)
    profile = _make_profile("poisson")

    with pytest.raises(ValueError, match="sigma_psd is only supported"):
        bmnd(volume, profile, sigma_psd=scalar_sigma_to_psd(volume.shape, 1.0))


def test_bmnd_rejects_sigma_and_sigma_psd_before_zero_volume_fast_path():
    volume = np.zeros((8, 8), dtype=np.float32)
    profile = _make_profile("gaussian")

    with pytest.raises(ValueError, match="mutually exclusive"):
        bmnd(volume, profile, sigma=0.1, sigma_psd=np.ones_like(volume))


@pytest.mark.parametrize("sigma", [-0.1, np.nan, np.inf, True])
def test_bmnd_rejects_invalid_sigma_before_zero_volume_fast_path(
    sigma: object,
) -> None:
    profile = _make_profile("gaussian")

    with pytest.raises(ValueError, match="sigma must be finite and non-negative"):
        bmnd(
            np.zeros((8, 8), dtype=np.float32),
            profile,
            sigma=sigma,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_bmnd_rejects_invalid_sigma_psd_before_zero_volume_fast_path(
    value: float,
) -> None:
    profile = _make_profile("gaussian")
    sigma_psd = np.zeros((8, 8), dtype=np.float32)
    sigma_psd[0, 0] = value

    with pytest.raises(ValueError, match="sigma_psd.*finite non-negative"):
        bmnd(np.zeros((8, 8), dtype=np.float32), profile, sigma_psd=sigma_psd)


def test_bmnd_rejects_malformed_sigma_psd_before_zero_volume_fast_path() -> None:
    profile = _make_profile("gaussian")

    with pytest.raises(ValueError, match="sigma_psd.*same shape as volume"):
        bmnd(
            np.zeros((8, 8), dtype=np.float32),
            profile,
            sigma_psd=np.ones((4, 4), dtype=np.float32),
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_bmnd_rejects_nonfinite_volume(value: float) -> None:
    profile = _make_profile("gaussian")
    volume = np.ones((8, 8), dtype=np.float32)
    volume[0, 0] = value

    with pytest.raises(ValueError, match="volume must contain only finite values"):
        bmnd(volume, profile, sigma=0.1)


def test_bmnd_rejects_negative_direct_poisson_input() -> None:
    profile = _make_profile("poisson")
    volume = np.ones((8, 8), dtype=np.float32)
    volume[0, 0] = -1.0

    with pytest.raises(ValueError, match="Poisson input.*non-negative"):
        bmnd(volume, profile)


def test_bmnd_rejects_malformed_nf_before_zero_volume_fast_path():
    volume = np.zeros((8, 8), dtype=np.float32)
    profile = _make_profile("gaussian")
    profile.nf = (4,)

    with pytest.raises(ValueError, match="nf must contain 2 values"):
        bmnd(volume, profile, sigma=0.1)


@pytest.mark.parametrize("fill_value", [0.0, 1.0], ids=["zero", "nonzero"])
@pytest.mark.parametrize(
    ("noise_model", "field", "value", "error"),
    [
        ("gaussian", "noise_model", "rician", "Unknown noise_model"),
        ("gaussian", "reference_schedule_mode", "invalid", "reference_schedule_mode"),
        (
            "gaussian",
            "reference_schedule_density",
            0,
            "reference_schedule_density must be a positive integer",
        ),
        ("gaussian", "ht_block_size", (4,), "ht_block_size must contain 2 values"),
        ("gaussian", "ref_batch_size", 0, "ref_batch_size must be a positive integer"),
        ("gaussian", "k", -1, "k must be a non-negative integer"),
        ("gaussian", "ht_weight_model", "legacy", "ht_weight_model"),
        ("poisson", "wiener_weight_model", "invalid", "wiener_weight_model"),
    ],
)
def test_bmnd_validates_active_profile_before_processing(
    fill_value: float,
    noise_model: str,
    field: str,
    value: object,
    error: str,
) -> None:
    volume = np.full((8, 8), fill_value, dtype=np.float32)
    profile = _make_profile(noise_model)
    setattr(profile, field, value)

    with pytest.raises(ValueError, match=error):
        bmnd(volume, profile, sigma=0.1)


@pytest.mark.parametrize("stage", ["ht", "wiener"])
def test_bmnd_rejects_min_stack_size_above_maximum(stage: str) -> None:
    profile = _make_profile("gaussian")
    setattr(profile, f"{stage}_min_stack_size", 3)
    setattr(profile, f"{stage}_max_stack_size", 2)

    with pytest.raises(ValueError, match=f"{stage}_min_stack_size cannot exceed"):
        bmnd(np.zeros((8, 8), dtype=np.float32), profile, sigma=0.1)


@pytest.mark.parametrize(
    ("noise_model", "field", "value"),
    [
        ("gaussian", "reference_shift_density", -1.0),
        ("gaussian", "ht_match_threshold", np.nan),
        ("gaussian", "wiener_match_threshold", -1.0),
        ("gaussian", "ht_lambda_threshold", np.inf),
        ("gaussian", "ht_kaiser_beta", -1.0),
        ("gaussian", "wiener_kaiser_beta", np.nan),
        ("gaussian", "wiener_variance_scale", -1.0),
        ("poisson", "poisson_variance_floor", np.nan),
    ],
)
def test_bmnd_rejects_invalid_nonnegative_profile_numbers(
    noise_model: str,
    field: str,
    value: float,
) -> None:
    profile = _make_profile(noise_model)
    setattr(profile, field, value)

    with pytest.raises(ValueError, match=f"{field} must be finite and non-negative"):
        bmnd(np.zeros((8, 8), dtype=np.float32), profile, sigma=0.1)


@pytest.mark.parametrize("stages", ["ht", "wiener", "both"])
def test_dc_only_aggregation_aware_projection_preserves_active_stage_totals(
    stages: str,
) -> None:
    rng = np.random.default_rng(43)
    volume = rng.poisson(3.0, size=(8, 8)).astype(np.float32)
    profile = _make_profile("poisson")
    profile.poisson_group_mass_conservation = stages
    stage_results: dict[str, NDArray[np.float32]] = {}

    result = cast(
        NDArray[np.float32],
        bmnd(
            volume,
            profile,
            sigma=None,
            stage_callback=lambda stage, value: stage_results.update({stage: value}),
        ),
    )

    observed_total = np.sum(volume, dtype=np.float64)
    if stages in {"ht", "both"}:
        assert np.isclose(
            np.sum(stage_results["ht"], dtype=np.float64),
            observed_total,
            rtol=2e-6,
        )
    if stages in {"wiener", "both"}:
        assert np.isclose(
            np.sum(result, dtype=np.float64),
            observed_total,
            rtol=2e-6,
        )


@pytest.mark.parametrize("k", [0, 1, 4], ids=["approximate", "partial-exact", "full-exact"])
def test_bmnd_supports_risk_weighting_with_wiener_group_mass_conservation(
    k: int,
) -> None:
    rng = np.random.default_rng(53)
    volume = rng.poisson(4.0, size=(8, 8)).astype(np.float32)
    profile = _make_profile("poisson")
    profile.poisson_group_mass_conservation = "wiener"
    profile.wiener_weight_model = "risk"
    profile.wiener_weight_domain = "windowed_synthesis"
    profile.wiener_weight_scope = "group"
    profile.k = k
    result = cast(NDArray[np.float32], bmnd(volume, profile, sigma=None))

    assert result.shape == volume.shape
    assert np.isfinite(result).all()
    assert np.isclose(
        np.sum(result, dtype=np.float64),
        np.sum(volume, dtype=np.float64),
        rtol=2e-6,
    )


def test_bmnd_rejects_sharpening_with_wiener_group_mass_conservation() -> None:
    volume = np.ones((8, 8), dtype=np.float32)
    profile = _make_profile("poisson")
    profile.poisson_group_mass_conservation = "wiener"
    profile.sharpen_alpha = 0.9

    with pytest.raises(ValueError, match="requires sharpening exponents of 1"):
        bmnd(volume, profile, sigma=None)


def test_bmnd_rejects_complex_transform_for_mass_projected_risk() -> None:
    volume = np.ones((8, 8), dtype=np.float32)
    profile = _make_profile("poisson")
    profile.poisson_group_mass_conservation = "wiener"
    profile.wiener_weight_model = "risk"
    profile.wiener_weight_domain = "windowed_synthesis"
    profile.wiener_weight_scope = "group"
    profile.wiener_transform = Transform(TransformType.FFT, TransformMode.ND)

    with pytest.raises(ValueError, match="requires real matrix transforms"):
        bmnd(volume, profile, sigma=None)


def test_bmnd_rejects_complex_ht_transform_for_mass_projected_variance() -> None:
    volume = np.ones((8, 8), dtype=np.float32)
    profile = _make_profile("poisson")
    profile.poisson_group_mass_conservation = "ht"
    profile.ht_weight_model = "variance"
    profile.ht_weight_domain = "coefficient"
    profile.ht_weight_scope = "group"
    profile.ht_transform = Transform(TransformType.FFT, TransformMode.ND)

    with pytest.raises(ValueError, match="requires real matrix transforms"):
        bmnd(volume, profile, sigma=None)


@pytest.mark.parametrize(
    ("domain", "scope", "transform_type", "message"),
    [
        ("windowed_synthesis", "group", TransformType.STARLET, "matrix-based"),
        ("windowed_synthesis", "group", TransformType.FFT, "real matrix"),
        ("coefficient", "patch", TransformType.FFT, "real matrix"),
    ],
)
def test_bmnd_rejects_unsupported_synthesis_weight_transforms(
    domain: str,
    scope: str,
    transform_type: TransformType,
    message: str,
) -> None:
    profile = _make_profile("gaussian")
    profile.wiener_weight_model = "variance"
    profile.wiener_weight_domain = domain
    profile.wiener_weight_scope = scope
    profile.wiener_transform = Transform(transform_type, TransformMode.ND)

    with pytest.raises(ValueError, match=message):
        bmnd(np.zeros((8, 8), dtype=np.float32), profile, sigma=0.1)


def test_bmnd_rejects_operator_transform_with_group_mass_conservation() -> None:
    volume = np.ones((8, 8), dtype=np.float32)
    profile = _make_profile("poisson")
    profile.poisson_group_mass_conservation = "ht"
    profile.ht_transform = Transform(TransformType.STARLET, TransformMode.ND)

    with pytest.raises(ValueError, match="requires matrix-based transforms"):
        bmnd(volume, profile, sigma=None)


def test_bmnd_ignores_inactive_poisson_options_for_gaussian_zero_volume() -> None:
    volume = np.zeros((8, 8), dtype=np.float32)
    profile = _make_profile("gaussian")
    profile.poisson_wiener_gain_mode = "invalid"

    result = cast(NDArray[np.float32], bmnd(volume, profile, sigma=0.1))

    assert np.array_equal(result, volume)


def test_bmnd_poisson_specific_distance_uses_local_zero_gamma(monkeypatch):
    rng = np.random.default_rng(37)
    volume = rng.poisson(5.0, size=(8, 8)).astype(np.float32)
    profile = _make_profile("poisson")
    profile.gamma = 2.5
    observed_blockmatch = []
    original_blockmatch = bmndalgo_module.blockmatch_groups

    def capture_blockmatch(*args, **kwargs):
        observed_blockmatch.append(
            (
                kwargs.get("distance_measure"),
                kwargs.get("gamma"),
                kwargs.get("poisson_gamma"),
            )
        )
        return original_blockmatch(*args, **kwargs)

    monkeypatch.setattr(bmndalgo_module, "blockmatch_groups", capture_blockmatch)

    output = bmnd(volume, profile, sigma=None)

    assert output.shape == volume.shape
    assert observed_blockmatch[0] == ("poisson_deviance", 0.0, False)
    assert profile.gamma == 2.5


@pytest.mark.parametrize(
    "distance_measure",
    ["poisson_deviance", "pearson", "anscombe_ssd"],
)
@pytest.mark.parametrize("wiener_match_source", ["pilot", "observation"])
def test_bmnd_candidate_standardization_is_ht_only(
    monkeypatch,
    distance_measure: str,
    wiener_match_source: str,
) -> None:
    rng = np.random.default_rng(41)
    volume = rng.poisson(5.0, size=(8, 8)).astype(np.float32)
    profile = _make_profile("poisson")
    profile.blockmatch_distance = distance_measure
    profile.poisson_ht_match_policy = "candidate_standardized"
    profile.poisson_wiener_match_source = wiener_match_source
    profile.ht_match_threshold = 2.0
    observed_blockmatch = []
    original_blockmatch = bmndalgo_module.blockmatch_groups

    def capture_blockmatch(*args, **kwargs):
        observed_blockmatch.append(
            (
                kwargs.get("distance_measure"),
                kwargs.get("standardize_poisson_distance", False),
            )
        )
        return original_blockmatch(*args, **kwargs)

    monkeypatch.setattr(bmndalgo_module, "blockmatch_groups", capture_blockmatch)

    output = bmnd(volume, profile, sigma=None)

    assert output.shape == volume.shape
    assert observed_blockmatch == [
        (distance_measure, True),
        (distance_measure, False),
    ]


@pytest.mark.parametrize(
    "match_policy",
    ["reference_finite_count", "candidate_standardized"],
)
def test_bmnd_rejects_non_fixed_poisson_ht_policy_with_ssd(match_policy: str) -> None:
    profile = _make_profile("poisson")
    profile.blockmatch_distance = "ssd"
    profile.poisson_ht_match_policy = match_policy

    with pytest.raises(ValueError, match="requires 'poisson_deviance'"):
        bmnd(np.ones((8, 8), dtype=np.float32), profile, sigma=None)


@pytest.mark.parametrize("wiener_match_source", ["pilot", "observation"])
def test_bmnd_finite_count_uses_ht_match_threshold(monkeypatch, wiener_match_source):
    rng = np.random.default_rng(43)
    volume = rng.poisson(5.0, size=(8, 8)).astype(np.float32)
    profile = _make_profile("poisson")
    profile.poisson_ht_match_policy = "reference_finite_count"
    profile.poisson_wiener_match_source = wiener_match_source
    profile.ht_match_threshold = 1.75
    observed_thresholds = []
    original_thresholds = bmndalgo_module.compute_poisson_reference_thresholds

    def capture_thresholds(*args, **kwargs):
        observed_thresholds.append(args[5])
        return original_thresholds(*args, **kwargs)

    monkeypatch.setattr(
        bmndalgo_module,
        "compute_poisson_reference_thresholds",
        capture_thresholds,
    )

    output = bmnd(volume, profile, sigma=None)

    assert output.shape == volume.shape
    assert observed_thresholds == [1.75]


def test_bmnd_wiener_can_match_on_poisson_observation(monkeypatch) -> None:
    rng = np.random.default_rng(47)
    volume = rng.poisson(5.0, size=(8, 8)).astype(np.float32)
    profile = _make_profile("poisson")
    profile.poisson_wiener_match_source = "observation"
    observed_patches = []
    original_blockmatch = bmndalgo_module.blockmatch_groups

    def capture_blockmatch(*args, **kwargs):
        observed_patches.append(kwargs["patches_flat"].copy())
        return original_blockmatch(*args, **kwargs)

    monkeypatch.setattr(bmndalgo_module, "blockmatch_groups", capture_blockmatch)

    output = bmnd(volume, profile, sigma=None)

    expected = bmndalgo_module.extract_patches_strided(
        volume,
        profile.wiener_block_size,
    ).reshape(-1, int(np.prod(profile.wiener_block_size)))
    assert output.shape == volume.shape
    assert np.array_equal(observed_patches[1], expected)


def test_bmnd_conventional_nf_uses_local_zero_k_and_gamma(monkeypatch):
    rng = np.random.default_rng(31)
    volume = (0.5 + rng.normal(0.0, 0.1, size=(8, 8))).astype(np.float32)
    profile = _make_profile("gaussian")
    profile.nf = 0
    profile.k = 3
    profile.gamma = 2.5
    observed_k = []
    observed_blockmatch = []
    original_variance = bmndalgo_module.get_stationary_gaussian_group_noise_variance
    original_blockmatch = bmndalgo_module.blockmatch_groups

    def capture_variance(*args, **kwargs):
        observed_k.append(args[-1])
        return original_variance(*args, **kwargs)

    def capture_blockmatch(*args, **kwargs):
        observed_blockmatch.append(
            (kwargs.get("gamma"), kwargs["include_position_metadata"])
        )
        return original_blockmatch(*args, **kwargs)

    monkeypatch.setattr(
        bmndalgo_module,
        "get_stationary_gaussian_group_noise_variance",
        capture_variance,
    )
    monkeypatch.setattr(bmndalgo_module, "blockmatch_groups", capture_blockmatch)

    output = bmnd(volume, profile, sigma=0.1)

    assert output.shape == volume.shape
    assert observed_k and set(observed_k) == {0}
    assert observed_blockmatch[0] == (0.0, False)
    assert observed_blockmatch[1] == (None, False)
    assert profile.nf == 0
    assert profile.k == 3
    assert profile.gamma == 2.5


def test_bmnd_float64_input_coerces_to_float32_output():
    rng = np.random.default_rng(14)
    volume = (0.5 + rng.normal(0.0, 0.1, size=(8, 8))).astype(np.float64)
    profile = _make_profile("gaussian")

    out = cast(NDArray[np.float32], bmnd(volume, profile, sigma=0.1))

    assert out.dtype == np.float32
    assert out.shape == volume.shape
    expected = bmnd(volume.astype(np.float32), profile, sigma=0.1)
    np.testing.assert_array_equal(out, expected)


def test_bmnd_3d_reconstructs_noiseless_volume():
    rng = np.random.default_rng(15)
    volume = (0.5 + rng.normal(0.0, 0.1, size=(4, 4, 4))).astype(np.float32)
    profile = BMNDProfile(
        noise_model="gaussian",
        ht_block_size=(2, 2, 2),
        ht_step=(2, 2, 2),
        ht_search_window=(1, 1, 1),
        ht_max_stack_size=4,
        wiener_block_size=(2, 2, 2),
        wiener_step=(2, 2, 2),
        wiener_search_window=(1, 1, 1),
        wiener_max_stack_size=4,
    )
    profile.ht_transform = Transform(TransformType.DCT, TransformMode.ND)
    profile.wiener_transform = Transform(TransformType.DCT, TransformMode.ND)
    profile.ref_batch_size = 8
    profile.wiener_variance_scale = 0.0

    out = cast(NDArray[np.float32], bmnd(volume, profile, sigma=0.0))

    assert out.shape == volume.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, volume, atol=1e-6)


def test_bmnd_scalar_sigma_and_flat_psd_agree():
    rng = np.random.default_rng(16)
    volume = (0.5 + rng.normal(0.0, 0.1, size=(8, 8))).astype(np.float32)
    profile = _make_profile("gaussian")
    sigma = 0.1

    out_sigma = cast(NDArray[np.float32], bmnd(volume, profile, sigma=sigma))
    out_psd = cast(
        NDArray[np.float32],
        bmnd(volume, profile, sigma_psd=scalar_sigma_to_psd(volume.shape, sigma)),
    )

    assert out_sigma.shape == volume.shape
    assert out_psd.shape == volume.shape
    assert out_sigma.dtype == np.float32
    assert out_psd.dtype == np.float32
    assert np.isfinite(out_psd).all()
    assert np.allclose(out_sigma, out_psd, atol=1e-6)


def test_bmnd_gaussian_global_sigma_is_estimated_without_explicit_sigma():
    volume = np.tile(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), 16).reshape(8, 8)

    profile = _make_profile("gaussian")

    out, sigma_maps = cast(
        tuple[NDArray[np.float32], dict[str, NDArray[np.float32] | float]],
        bmnd(volume, profile, sigma=None, return_sigma_map=True),
    )

    assert out.shape == volume.shape
    assert np.isfinite(out).all()
    expected_sigma = 1.0 / 0.6745
    assert float(sigma_maps["global"]) == pytest.approx(expected_sigma)
    explicit = bmnd(volume, profile, sigma=expected_sigma)
    np.testing.assert_allclose(out, explicit, atol=1e-6)
