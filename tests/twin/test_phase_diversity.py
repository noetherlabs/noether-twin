"""Analytical fit/action oracles; these fixtures are not development results."""

import numpy as np
import pytest

from noether_optics.twin.phase_diversity import (
    FitCriteria,
    choose_correction,
    fit_multistart,
    fit_start,
    initial_points,
    summarize_starts,
    surface_gram,
    validate_counts,
)


def quadratic(center=(0.0, 0.0), curvature=1.0):
    def objective(theta):
        delta = theta - center
        return float(curvature * delta @ delta / 2), curvature * delta

    return objective


def test_initial_stationarity_requires_two_charged_confirmations_without_motion():
    result = fit_start(quadratic(), (0, 0))
    assert result["status"] == "stationary_confirmed"
    assert result["accepted_steps"] == 0
    assert result["calls"] == 3
    assert [row["role"] for row in result["trace"]] == ["initial", "stationary_confirmation", "stationary_confirmation"]
    assert all(row["theta"] == [0, 0] for row in result["trace"])


def test_projected_stationarity_at_box_boundary_has_nonzero_raw_gradient():
    result = fit_start(quadratic((10, 0)), (4, 0))
    assert result["status"] == "stationary_confirmed"
    assert result["trace"][0]["gradient"] == [-6, 0]
    assert result["projected_gradient_inf"] == 0


def test_stationary_but_changing_objective_is_not_confirmed():
    calls = []

    def nondeterministic(theta):
        calls.append(1)
        return float(len(calls)), np.zeros(2)

    result = fit_start(nondeterministic, (0, 0), FitCriteria(5))
    assert result["status"] == "proposal_cap_exhausted"
    assert result["calls"] == 5


def test_clipping_dedup_cannot_supply_two_independent_starts():
    starts = initial_points((1000, 1000))
    assert starts == [(4, 4)]
    fit = fit_multistart(quadratic((4, 4)), starts=starts)
    assert fit["agreeing_distinct_starts"] == 1
    assert fit["status"] == "unresolved" and fit["estimate_nm"] is None


def test_duplicate_explicit_starts_cannot_supply_independent_agreement():
    fit = fit_multistart(quadratic(), starts=[(0, 0), (0, 0)])
    assert len(fit["starts"]) == 1 and fit["status"] == "unresolved"
    with pytest.raises(ValueError, match="duplicate"):
        summarize_starts([fit["starts"][0], fit["starts"][0]])


def test_distinct_initial_points_recover_analytic_convex_minimum():
    fit = fit_multistart(quadratic((0.25, -0.75)), starts=[(-1, 0), (1, 0)])
    assert fit["status"] == "locally_resolved"
    np.testing.assert_allclose(fit["estimate_nm"], [5, -15], atol=1e-10)
    assert fit["agreeing_distinct_starts"] == 2


def test_equal_objective_distinct_basins_are_explicitly_ambiguous():
    def double_well(theta):
        x, y = theta
        return (x * x - 1) ** 2 + y * y, np.array([4 * x * (x * x - 1), 2 * y])

    fit = fit_multistart(double_well, starts=[(-1, 0), (1, 0)])
    assert fit["status"] == "unresolved"
    assert fit["estimate_nm"] is None
    assert len(fit["starts"]) == 2


def test_rejected_line_search_proposals_are_charged_at_exact_cap():
    fit = fit_start(quadratic(curvature=10), (1, -1), FitCriteria(3))
    assert fit["status"] == "proposal_cap_exhausted"
    assert fit["calls"] == 3 and fit["accepted_steps"] == 0
    assert [row["accepted"] for row in fit["trace"][1:]] == [False, False]


def test_nonfinite_fit_stops_instead_of_becoming_missing_row():
    with pytest.raises(FloatingPointError):
        fit_start(lambda theta: (float("nan"), np.zeros(2)), (0, 0))


def test_surface_gram_removes_piston_and_keeps_cross_axis_covariance():
    profiles = np.array([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 1.0], [4.0, 3.0]]])
    gram = surface_gram(profiles, np.ones((2, 2), dtype=bool))
    np.testing.assert_allclose(gram, [[1.25, 0.75], [0.75, 1.25]], rtol=0, atol=0)
    shifted = surface_gram(profiles + np.array([7.0, -2.0])[:, None, None], np.ones((2, 2), dtype=bool))
    np.testing.assert_array_equal(gram, shifted)


def test_action_uses_full_gram_and_stroke_limit_not_per_axis_rounding():
    gram = np.array([[1.0, 0.9], [0.9, 1.0]])
    choice = choose_correction([4, -2], gram)
    # Independent enumeration of physical double-pass variance.
    possible = [(x, y) for x in (-5, 0, 5) for y in (-5, 0, 5)]

    def loss(d):
        return 4 * ((4 + d[0]) ** 2 + (-2 + d[1]) ** 2 + 1.8 * (4 + d[0]) * (-2 + d[1]))

    assert tuple(choice["increment_nm"]) == min(possible, key=loss)
    bounded = choose_correction([-30, 30], gram, applied_nm=[40, -40])
    assert all(row["increment_nm"][0] <= 0 and row["increment_nm"][1] >= 0 for row in bounded["scores"])
    assert "increment-4" in {row["id"] for row in bounded["scores"]}


def test_photon_validation_does_not_renormalize_detector_counts():
    raw = np.arange(2 * 65 * 65, dtype=np.int64).reshape(2, 65, 65)
    np.testing.assert_array_equal(validate_counts(raw, 10000), raw)
    with pytest.raises(ValueError, match="integer"):
        validate_counts(raw.astype(float), 10000)
    with pytest.raises(ValueError, match="positive"):
        validate_counts(raw, float("inf"))


def test_diversity_factories_do_not_share_cache_and_preserve_v1_default():
    pytest.importorskip("hcipy")
    from noether_optics.bench.twin_intervention import reference_backend
    from noether_optics.twin.phase_diversity import surface_command

    command = surface_command([17, -9])
    plus, minus = reference_backend(), reference_backend(defocus_opd_m=-30e-9)
    first = plus(16, command, 1.0)["power"]
    negative = minus(16, command, 1.0)["power"]
    again = plus(16, command, 1.0)["power"]
    explicit = reference_backend(defocus_opd_m=30e-9)(16, command, 1.0)["power"]
    np.testing.assert_array_equal(first, again)
    np.testing.assert_array_equal(first, explicit)
    assert np.sum(np.abs(first - negative)) > 1e-5


def test_joint_likelihood_and_gradient_have_independent_hcipy_oracle():
    jax = pytest.importorskip("jax")
    pytest.importorskip("dLux")
    from noether_optics.bench.twin_intervention import reference_backend
    from noether_optics.twin.phase_diversity import make_candidate, surface_command

    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        refs = [reference_backend(defocus_opd_m=d) for d in (30e-9, -30e-9)]
        photons = 1000000
        truth_nm = np.array([13.0, -7.0])

        def means(theta):
            return np.stack([photons / 2 * ref(64, surface_command(theta * 20), 1.0)["power"] + 0.1 for ref in refs])

        counts = np.rint(means(truth_nm / 20)).astype(np.int64)
        model = make_candidate(64)
        theta = np.array([0.4, -0.2])
        value, gradient = model["objective"](theta, validate_counts(counts, photons), photons)

        def loss(point):
            mu = means(point)
            return np.sum(mu - counts * np.log(mu)) / photons

        assert value == pytest.approx(loss(theta), rel=2e-13, abs=1e-12)
        step = 1e-5
        derivative = [
            (loss(theta + step * direction) - loss(theta - step * direction)) / (2 * step) for direction in np.eye(2)
        ]
        np.testing.assert_allclose(gradient, derivative, rtol=1e-5, atol=2e-9)
        assert model["compile_seconds"] > 0 and model["setup_seconds"] >= 0
    finally:
        jax.config.update("jax_enable_x64", previous)
