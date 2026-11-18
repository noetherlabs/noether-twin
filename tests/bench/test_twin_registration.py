"""Analytical and injected wiring checks; these are not optical study outcomes."""

import json

import numpy as np
import pytest

from noether_optics.bench import twin_registration as study
from noether_optics.twin import registration as fit


def quadratic(center):
    return lambda point: ((point - center) ** 2, 2 * (point - center))


def test_scalar_fit_recovers_quadratic_and_initial_stationarity_needs_no_movement():
    result = fit.fit_registration(quadratic(0.073))
    assert result["status"] == "locally_resolved"
    assert result["estimate_pitch"] == pytest.approx(0.073, abs=1e-6)
    stationary = fit.fit_start(quadratic(0.0), 0.0)
    assert stationary["calls"] == 3 and stationary["status"] == "stationary_confirmed"
    assert [row["role"] for row in stationary["trace"]] == [
        "initial",
        "stationary_confirmation",
        "stationary_confirmation",
    ]
    assert all(row["point_pitch"] == 0 for row in stationary["trace"])


@pytest.mark.parametrize("center", [-0.4, 0.4])
def test_constrained_boundary_minimum_is_explicit(center):
    result = fit.fit_registration(quadratic(center))
    assert result["status"] == "locally_resolved" and result["boundary_solution"]
    assert result["estimate_pitch"] == np.sign(center) * 0.2


def test_flat_likelihood_and_separated_equal_minima_abstain_to_nominal():
    flat = fit.fit_registration(lambda point: (1.0, 0.0))
    double = fit.fit_registration(lambda point: ((point * point - 0.01) ** 2, 4 * point * (point * point - 0.01)))
    for result in (flat, double):
        assert result["status"] == "unresolved"
        assert result["prediction_pitch"] == 0.0 and result["estimate_pitch"] is None
        assert result["near_best_spread_pitch"] > fit.CRITERIA.agreement_pitch


def test_clipped_duplicate_starts_cannot_supply_independent_agreement():
    result = fit.fit_registration(quadratic(0.4), starts=(1, 2, 3))
    assert len(result["starts"]) == 1
    assert result["near_best_distinct_starts"] == 1
    assert result["status"] == "unresolved"


def test_every_rejected_proposal_charged_and_exhausted_fit_unavailable():
    calls = []

    def contradictory(point):
        calls.append(point)
        return 1.0, 1.0  # No Armijo decrease is possible; deliberately inconsistent oracle.

    result = fit.fit_registration(contradictory, criteria=fit.RegistrationCriteria(max_calls_per_start=3))
    assert result["objective_calls"] == len(calls) == 15
    assert result["status"] == "unresolved"
    assert any(row["status"] == "call_cap_exhausted" for row in result["starts"])
    assert all(row["calls"] <= 3 for row in result["starts"])


def test_nonfinite_estimator_does_not_return_nominal_success():
    with pytest.raises(FloatingPointError):
        fit.fit_registration(lambda point: (np.nan, 0.0))


def test_expected_means_and_counts_have_separate_likelihood_contracts():
    means = np.full((4, 65, 65), 1.5)
    assert fit.prediction_loss(means, means, expected=True) == 0
    with pytest.raises(ValueError, match="integer"):
        fit.prediction_loss(means, means)
    counts = np.ones((4, 65, 65), dtype=np.int64)
    expected_nll = 4 * 65**2 * (1.5 - np.log(1.5)) / (2 * fit.PHOTONS_PER_PAIR)
    assert fit.prediction_loss(means, counts) == pytest.approx(expected_nll)
    assert fit.prediction_loss(means * 2, means, expected=True) > 0
    with pytest.raises(ValueError):
        fit.prediction_loss(np.zeros_like(means), counts)


def test_fixed_schedule_photons_and_noise_streams_are_disjoint():
    contract = study.protocol()
    assert len(contract["calibration_schedule"]) == 8
    assert len(contract["prediction_schedule"]) == 4
    assert {row[0] for row in contract["calibration_schedule"]} == {7, 12}
    assert {row[0] for row in contract["prediction_schedule"]} == {18}
    assert contract["calibration_photons_per_unit"] == 4_000_000
    assert contract["scoring_photons_per_unit"] == 2_000_000
    seeds = [
        unit[key]
        for unit in study.units()
        if unit["kind"] == "poisson"
        for key in ("calibration_seed", "prediction_seed")
    ]
    assert len(seeds) == len(set(seeds)) == 96
    assert len(study.units()) == 51 and contract["selection_or_evaluation_cohort"] is False
    assert study.LIMITS["objective"] == 51 * 3 * 5 * 64
    assert study.LIMITS["prediction"] == 51 * 3 * 3


def test_reference_sensitivity_does_not_convert_a_reversed_order_into_benefit():
    coarse, fine = np.full((4, 65, 65), 1.0), np.full((4, 65, 65), 2.0)
    predictions = {
        f"{arm}{grid}": coarse.copy() for grid in study.GRIDS for arm in ("nominal", "estimated", "true_registration")
    }
    predictions["estimated128"] = fine.copy()
    result = study.score_predictions(predictions, coarse, coarse, fine, expected=True)
    primary = result["contrasts"]["estimated128_minus_nominal256"]
    assert primary["expected_difference_512"] > 0 and primary["expected_difference_1024"] < 0
    assert primary["reference_status"] == "unresolved"


def test_compiled_likelihood_and_prediction_match_separately_assembled_exposures():
    jax = pytest.importorskip("jax")
    pytest.importorskip("dLux")
    from noether_optics.twin.static_mirror import StaticMirrorSpec, pixel_power

    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        model = fit.make_registration_candidate(64)
        # Asymmetric integer targets make exposure/sign/spatial permutations visible.
        counts = (1 + np.arange(8 * 65**2).reshape(8, 65, 65) % 17).astype(np.int64)
        observations = fit.images(counts, 8)

        def assemble(shift, actuators):
            values = []
            # This explicit physical schedule does not call the candidate's
            # vectorized powers or its schedule helper.
            for actuator in actuators:
                for sign in (1, -1):
                    command = np.zeros(25)
                    command[actuator] = sign * 10e-9
                    for diversity in (30e-9, -30e-9):
                        values.append(
                            np.asarray(
                                pixel_power(
                                    command,
                                    1.0,
                                    spec=StaticMirrorSpec(defocus_opd_m=diversity),
                                    pupil_samples=64,
                                    registration_x_pitch=shift,
                                )
                            )
                        )
            return np.stack(values)

        def manual_loss(shift):
            means = 500_000 * assemble(shift, (12, 7)) + 0.1
            return float(np.sum(means - observations * np.log(means)) / 4_000_000)

        shift, step = 0.037, 1e-5
        value, gradient = model["objective"](shift, observations)
        assert value == pytest.approx(manual_loss(shift), rel=1e-12, abs=1e-12)
        manual_means = 500_000 * assemble(shift, (12, 7)) + 0.1
        swapped = manual_means[[1, 0, 2, 3, 4, 5, 6, 7]]
        swapped_value = float(np.sum(swapped - observations * np.log(swapped)) / 4_000_000)
        assert abs(swapped_value - manual_loss(shift)) > 1e-10
        difference = (manual_loss(shift + step) - manual_loss(shift - step)) / (2 * step)
        assert abs(difference) > 1e-8
        assert abs(gradient - difference) <= 2e-4 * abs(difference)
        np.testing.assert_allclose(model["prediction"](shift), assemble(shift, (18,)), rtol=1e-11, atol=2e-14)
        assert model["objective_compile_seconds"] > 0 and model["prediction_compile_seconds"] > 0
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def injected(tmp_path, monkeypatch):
    source = tmp_path / "source"
    for name in study.SOURCE_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture source\n")
    from noether_optics.bench import twin_development

    monkeypatch.setattr(twin_development.importlib.metadata, "version", lambda name: "fixture")
    monkeypatch.setattr(twin_development.os, "fsync", lambda fd: None)
    root = tmp_path / "run"
    seen = {grid: [] for grid in study.GRIDS}

    def reference_factory(*, defocus_opd_m, registration_x_pitch):
        def reference(n, command, gain):
            if np.any(command[18]):
                journal = [json.loads(row) for row in (root / "events.jsonl").read_text().splitlines()]
                assert any(row["kind"] == "all_predictions_frozen" for row in journal)
                assert len(list(root.glob("*-fit.json"))) == 51
            return {"power": np.full((65, 65), (1 + registration_x_pitch) * 1e-5), "incident": np.asarray(1.0)}

        return reference

    def candidate(grid):
        def objective(shift, calibration):
            assert calibration.shape == (8, 65, 65)
            assert not calibration.flags.writeable
            seen[grid].append(calibration)
            return (shift - 0.05) ** 2, 2 * (shift - 0.05)

        return {
            "objective": objective,
            "prediction": lambda shift: np.full((4, 65, 65), (1 + shift) * 1e-5),
            "objective_compile_seconds": 0.01,
            "prediction_compile_seconds": 0.02,
        }

    monkeypatch.setattr(study, "reference_backend", reference_factory)
    monkeypatch.setattr(study, "make_registration_candidate", candidate)
    return root, source, seen


def test_full_injected_study_freezes_every_prediction_before_scoring_and_counts_all_units(injected):
    root, source, seen = injected
    result = study.run_registration(root, source_root=source)
    assert result["units"] == 51 and result["noisy_units"] == 48
    assert result["counts"]["reference"] == 72 and result["counts"]["prediction"] == 459
    assert result["counts"]["objective"] <= 48960
    assert all(row["units"] == 16 for row in result["stratified_summary"])
    assert "withheld" in result["promotion"]
    assert (root / "manifest.json").exists()
    for grid in study.GRIDS:
        assert len({id(array) for array in seen[grid]}) == 51
    assert {id(array) for array in seen[64]} == {id(array) for array in seen[256]}
    journal = [json.loads(row) for row in (root / "events.jsonl").read_text().splitlines()]
    barrier = next(row["event"] for row in journal if row["kind"] == "all_predictions_frozen")
    for row in journal:
        context = row.get("context", {})
        if (
            context.get("stage") in ("scoring", "refinement_scoring", "refinement_calibration")
            or context.get("arm") == "evaluator_true_registration"
        ):
            assert row["event"] > barrier


def test_unresolved_fallback_stays_in_every_stratified_denominator(injected, monkeypatch):
    root, source, _ = injected
    original = study.make_registration_candidate

    def flat(grid):
        model = original(grid)
        model["objective"] = lambda shift, calibration: (1.0, 0.0)
        return model

    monkeypatch.setattr(study, "make_registration_candidate", flat)
    result = study.run_registration(root, source_root=source)
    assert result["counts"]["objective"] == 51 * 3 * 5 * 3
    assert all(row["unresolved_fit_counts"] == {"64": 16, "128": 16, "256": 16} for row in result["stratified_summary"])
    for row in result["stratified_summary"]:
        assert row["paired_contrasts"]["estimated128_minus_nominal256"]["values"] == [0.0] * 16


def test_nonfinite_prediction_retained_stops_before_any_scoring(injected, monkeypatch):
    root, source, _ = injected
    original = study.make_registration_candidate

    def bad(grid):
        model = original(grid)
        model["prediction"] = lambda shift: np.full((4, 65, 65), np.nan)
        return model

    monkeypatch.setattr(study, "make_registration_candidate", bad)
    with pytest.raises(FloatingPointError):
        study.run_registration(root, source_root=source)
    journal = [json.loads(row) for row in (root / "events.jsonl").read_text().splitlines()]
    failed = next(row for row in journal if row.get("status") == "nonfinite")
    with np.load(root / failed["raw"]["path"], allow_pickle=False) as raw:
        assert np.isnan(raw["power"]).all()
    assert not any(row["kind"] == "all_predictions_frozen" for row in journal)
    assert not (root / "manifest.json").exists()


def test_changed_source_cannot_publish_complete_manifest(injected, monkeypatch):
    root, source, _ = injected
    original = study.summarize

    def mutate(rows):
        (source / study.SOURCE_FILES[0]).write_text("modified fixture source\n")
        return original(rows)

    monkeypatch.setattr(study, "summarize", mutate)
    with pytest.raises(RuntimeError, match="Source changed"):
        study.run_registration(root, source_root=source)
    assert not (root / "manifest.json").exists()
