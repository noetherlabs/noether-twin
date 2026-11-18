"""Fresh-only registration-versus-resolution development diagnostic.

No controller is selected or applied. Predictions freeze before scoring images
and evaluator-only true-registration predictions; independent HCIPy supplies truth.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from noether_optics.bench.twin_development import DevelopmentPacket
from noether_optics.bench.twin_intervention import (
    _array_bytes,
    canonical,
    file_hash,
    physical_contract,
    reference_backend,
)
from noether_optics.twin.registration import (
    BOUND_PITCH,
    CALIBRATION_ACTUATORS,
    CRITERIA,
    PHOTONS_PER_PAIR,
    PREDICTION_ACTUATORS,
    STARTS_PITCH,
    command_for,
    expected_means,
    fit_registration,
    images,
    make_registration_candidate,
    prediction_loss,
    schedule,
)

SCHEMA = "noether-twin-registration-development-v1"
SHIFTS = (-0.10, 0.0, 0.10)
GRIDS = (64, 128, 256)
REPLICATES = 16
LIMITS = {"objective": 48960, "reference": 72, "prediction": 459, "packet_bytes": 512 * 1024**2}
SOURCE_FILES = (
    "src/noether_optics/twin/static_mirror.py",
    "src/noether_optics/twin/phase_diversity.py",
    "src/noether_optics/twin/registration.py",
    "src/noether_optics/bench/twin_intervention.py",
    "src/noether_optics/bench/twin_development.py",
    "src/noether_optics/bench/twin_registration.py",
    "pyproject.toml",
    "uv.lock",
)


def units():
    return [
        {
            "id": f"shift-{state}-mean",
            "state_index": state,
            "kind": "expected_mean",
            "calibration_seed": None,
            "prediction_seed": None,
        }
        for state in range(3)
    ] + [
        {
            "id": f"shift-{state}-noise-{repeat:02d}",
            "state_index": state,
            "kind": "poisson",
            "calibration_seed": 911110000 + state * 100 + repeat,
            "prediction_seed": 921110000 + state * 100 + repeat,
        }
        for state in range(3)
        for repeat in range(REPLICATES)
    ]


def protocol():
    return {
        "schema": SCHEMA,
        "purpose": "known_flat_registration_development_only",
        "physical": physical_contract(),
        "registration_x_pitch": list(SHIFTS),
        "registration_interpretation": "fixed_sensitivity_challenge_not_measured_uncertainty",
        "plant_grid": 512,
        "reference_diagnostic_grid": 1024,
        "candidate_grids": list(GRIDS),
        "command_gain": 1.0,
        "incidental_wavefront": "known_flat",
        "calibration_schedule": list(schedule(CALIBRATION_ACTUATORS)),
        "prediction_schedule": list(schedule(PREDICTION_ACTUATORS)),
        "source_photons_per_diversity_pair": PHOTONS_PER_PAIR,
        "calibration_photons_per_unit": 4 * PHOTONS_PER_PAIR,
        "scoring_photons_per_unit": 2 * PHOTONS_PER_PAIR,
        "source_command_surface_nm": 10.0,
        "background_per_pixel_per_exposure": 0.1,
        "fit_box_pitch": [-BOUND_PITCH, BOUND_PITCH],
        "fit_starts_pitch": list(STARTS_PITCH),
        "fit_criteria": dataclasses.asdict(CRITERIA),
        "fallback": "nominal_registration_zero",
        "units": units(),
        "primary_contrast": "estimated128_minus_nominal256",
        "reference_rule": "report_coarse_fine_expected_loss_contrasts; no_continuum_or_statistical_certificate",
        "oracle_scope": "true_registration_not_necessarily_a_finite_grid_loss_lower_bound",
        "independent_physical_states": 3,
        "noise_replicates_per_state": REPLICATES,
        "selection_or_evaluation_cohort": False,
        "limits": LIMITS,
    }


def reference_block(packet, state_index, grid, actuators, *, stage):
    """Acquire each physical exposure, without retaining a grid/shift global cache."""
    shift = SHIFTS[state_index]
    factories = {}
    powers = []
    for exposure, (actuator, sign, diversity) in enumerate(schedule(actuators)):
        if diversity not in factories:
            factories[diversity] = reference_backend(defocus_opd_m=diversity, registration_x_pitch=shift)
        backend = factories[diversity]

        def acquire(backend=backend, actuator=actuator, sign=sign):
            output = backend(grid, command_for(actuator, sign), 1.0)
            return {"power": output["power"], "incident": output["incident"]}

        value = packet.call(
            "reference",
            {
                "stage": stage,
                "state_index": state_index,
                "grid": grid,
                "exposure": exposure,
                "actuator": actuator,
                "sign": sign,
                "defocus_opd_m": diversity,
            },
            acquire,
            arrays=True,
        )
        if (
            np.asarray(value["power"]).dtype != np.float64
            or np.asarray(value["incident"]).dtype != np.float64
            or np.shape(value["incident"]) != ()
            or float(value["incident"]) <= 0
        ):
            raise ValueError("Reference must retain float64 physical power and positive scalar incident flux")
        power = images(np.asarray(value["power"])[None], 1, expected=True)[0]
        powers.append(power)
    return expected_means(np.stack(powers))


def predict(packet, model, shift, *, unit, grid, arm):
    value = packet.call(
        "prediction",
        {"unit": unit, "grid": grid, "arm": arm, "shift_pitch": shift},
        lambda: {"power": model["prediction"](shift)},
        arrays=True,
    )
    if np.asarray(value["power"]).dtype != np.float64:
        raise ValueError("Candidate prediction must retain float64 power")
    return expected_means(images(value["power"], 4, expected=True))


def score_predictions(predictions, target, coarse_means, fine_means, *, expected):
    observed = {
        name: prediction_loss(prediction, target, expected=expected) for name, prediction in predictions.items()
    }
    coarse = {
        name: prediction_loss(prediction, coarse_means, expected=True) for name, prediction in predictions.items()
    }
    fine = {name: prediction_loss(prediction, fine_means, expected=True) for name, prediction in predictions.items()}
    pairs = {
        "estimated128_minus_nominal256": ("estimated128", "nominal256"),
        "estimated128_minus_estimated256": ("estimated128", "estimated256"),
    }
    for grid in GRIDS:
        pairs[f"estimated{grid}_minus_nominal{grid}"] = (f"estimated{grid}", f"nominal{grid}")
        pairs[f"estimated{grid}_minus_true_registration{grid}"] = (f"estimated{grid}", f"true_registration{grid}")
    contrasts = {}
    for name, (left, right) in pairs.items():
        c, f = coarse[left] - coarse[right], fine[left] - fine[right]
        sensitivity = abs(c - f)
        # This deliberately supplies no pass/certification flag. Small or reversed
        # contrasts are labelled unresolved; larger ones remain descriptive only.
        unresolved = c * f <= 0 or min(abs(c), abs(f)) <= sensitivity
        contrasts[name] = {
            "observed_difference": observed[left] - observed[right],
            "expected_difference_512": c,
            "expected_difference_1024": f,
            "reference_grid_sensitivity": sensitivity,
            "reference_status": "unresolved" if unresolved else "descriptive_separation_only",
        }
    return {
        "loss_kind": "expected_poisson_kl" if expected else "poisson_nll_without_factorial",
        "per_incident_scoring_photon": True,
        "losses": observed,
        "expected_kl_512": coarse,
        "expected_kl_1024": fine,
        "contrasts": contrasts,
    }


def summarize(rows):
    """Descriptive paired reductions within each fixed offset; all fallback rows remain."""
    summaries = []
    for state_index, shift in enumerate(SHIFTS):
        cohort = [row for row in rows if row["state_index"] == state_index and row["kind"] == "poisson"]
        contrasts = {
            name: [row["scores"]["contrasts"][name]["observed_difference"] for row in cohort]
            for name in cohort[0]["scores"]["contrasts"]
        }
        summaries.append(
            {
                "state_index": state_index,
                "true_registration_pitch": shift,
                "units": len(cohort),
                "paired_contrasts": {
                    name: {
                        "values": values,
                        "mean": float(np.mean(values)),
                        "minimum": min(values),
                        "maximum": max(values),
                    }
                    for name, values in contrasts.items()
                },
                "unresolved_fit_counts": {
                    str(grid): sum(row["fits"][str(grid)]["status"] != "locally_resolved" for row in cohort)
                    for grid in GRIDS
                },
            }
        )
    return summaries


def run_registration(root, *, source_root):
    packet = DevelopmentPacket(root, source_root, study_protocol=protocol(), source_files=SOURCE_FILES, limits=LIMITS)
    try:
        source_before = {name: file_hash(Path(source_root) / name) for name in SOURCE_FILES}
        if source_before != json.loads((packet.root / "protocol.json").read_text())["source"]:
            raise RuntimeError("Source changed during packet construction")
        # Evaluator-owned means are not arguments of any candidate objective.
        calibration_means = [
            reference_block(packet, state, 512, CALIBRATION_ACTUATORS, stage="calibration") for state in range(3)
        ]
        models = {}
        for grid in GRIDS:
            packet.event("candidate_compile_dispatch", grid=grid)
            models[grid] = make_registration_candidate(grid)
            packet.event(
                "candidate_compile_complete",
                grid=grid,
                objective_seconds=models[grid]["objective_compile_seconds"],
                prediction_seconds=models[grid]["prediction_compile_seconds"],
            )
        frozen = []
        for unit in units():
            means = calibration_means[unit["state_index"]]
            expected = unit["kind"] == "expected_mean"
            calibration = means.copy() if expected else np.random.default_rng(unit["calibration_seed"]).poisson(means)
            raw = packet.publish(f"arrays/{unit['id']}-calibration.npz", _array_bytes({"images": calibration}))
            packet.event(
                "calibration_images",
                unit=unit["id"],
                seed=unit["calibration_seed"],
                kind_of_images=unit["kind"],
                raw=raw,
            )
            observations = images(calibration, 8, expected=expected)
            observations.setflags(write=False)
            fits, predictions = {}, {}
            for grid in GRIDS:
                model = models[grid]

                def objective(shift, model=model, grid=grid, unit=unit, observations=observations):
                    return packet.call(
                        "objective",
                        {"unit": unit["id"], "grid": grid, "shift_pitch": shift},
                        lambda: model["objective"](shift, observations),
                    )

                before = time.perf_counter()
                fitted = fit_registration(objective)
                fitted["fit_with_journal_seconds"] = time.perf_counter() - before
                fits[str(grid)] = fitted
                for arm, shift in (("nominal", 0.0), ("estimated", fitted["prediction_pitch"])):
                    predictions[f"{arm}{grid}"] = predict(packet, model, shift, unit=unit["id"], grid=grid, arm=arm)
            prediction_raw = packet.publish(f"arrays/{unit['id']}-predictions.npz", _array_bytes(predictions))
            receipt = packet.publish(
                f"{unit['id']}-fit.json", canonical({"unit": unit, "fits": fits, "predictions": prediction_raw})
            )
            packet.event("predictions_frozen", unit=unit["id"], raw=receipt)
            frozen.append((unit, fits, predictions))
        packet.event("all_predictions_frozen", units=len(frozen), grid_fits=len(frozen) * len(GRIDS))

        # All declared predictions/statuses exist before these evaluator calls.
        scoring_means, fine_scoring = [], []
        for state in range(3):
            scoring_means.append(reference_block(packet, state, 512, PREDICTION_ACTUATORS, stage="scoring"))
            fine_calibration = reference_block(
                packet, state, 1024, CALIBRATION_ACTUATORS, stage="refinement_calibration"
            )
            fine_scoring.append(reference_block(packet, state, 1024, PREDICTION_ACTUATORS, stage="refinement_scoring"))
            packet.event(
                "calibration_reference_sensitivity",
                state_index=state,
                max_absolute_expected_count_difference=float(
                    np.max(np.abs(fine_calibration - calibration_means[state]))
                ),
                l1_expected_count_difference=float(np.sum(np.abs(fine_calibration - calibration_means[state]))),
            )
        rows = []
        for unit, fits, predictions in frozen:
            state, expected = unit["state_index"], unit["kind"] == "expected_mean"
            target = (
                scoring_means[state].copy()
                if expected
                else np.random.default_rng(unit["prediction_seed"]).poisson(scoring_means[state])
            )
            raw = packet.publish(f"arrays/{unit['id']}-scoring.npz", _array_bytes({"images": target}))
            packet.event("scoring_images", unit=unit["id"], seed=unit["prediction_seed"], raw=raw)
            predictions = dict(predictions)
            for grid in GRIDS:
                predictions[f"true_registration{grid}"] = predict(
                    packet, models[grid], SHIFTS[state], unit=unit["id"], grid=grid, arm="evaluator_true_registration"
                )
            scores = score_predictions(
                predictions, target, scoring_means[state], fine_scoring[state], expected=expected
            )
            row = {
                **unit,
                "true_registration_pitch": SHIFTS[state],
                "fits": fits,
                "scores": scores,
                "registration_errors_pitch": {
                    str(grid): fits[str(grid)]["prediction_pitch"] - SHIFTS[state] for grid in GRIDS
                },
            }
            packet.publish(f"{unit['id']}-outcome.json", canonical(row))
            rows.append(row)
        result = {
            "schema": SCHEMA,
            "status": "complete_development",
            "units": len(rows),
            "noisy_units": 48,
            "mean_diagnostics": 3,
            "physical_states": 3,
            "stratified_summary": summarize(rows),
            "counts": packet.counts,
            "compile_seconds": {
                str(grid): {key: value for key, value in model.items() if key.endswith("seconds")}
                for grid, model in models.items()
            },
            "total_seconds": time.perf_counter() - packet.started,
            "promotion": "withheld; no_selection_evaluation_or_closed_loop_experiment",
            "interpretation": "Fixed known-flat simulation; no hardware, global-fit or useful-control claim.",
        }
        if packet.counts["reference"] != 72 or packet.counts["prediction"] != 459:
            raise RuntimeError("Declared callback coverage is incomplete")
        packet.publish("analysis.json", canonical(result))
        source_after = {name: file_hash(Path(source_root) / name) for name in SOURCE_FILES}
        if source_after != source_before:
            raise RuntimeError("Source changed during development; complete manifest withheld")
        packet.event("complete", counts=packet.counts, source=source_after)
        packet.close()
        files = {
            str(path.relative_to(packet.root)): file_hash(path)
            for path in sorted(packet.root.rglob("*"))
            if path.is_file()
        }
        packet.publish("manifest.json", canonical({"schema": SCHEMA, "status": "complete_development", "files": files}))
        return result
    except BaseException as error:
        try:
            packet.event("run_stopped", exception_type=type(error).__name__, message=str(error), counts=packet.counts)
        finally:
            packet.close()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    run_registration(args.output, source_root=Path(__file__).resolve().parents[3])


if __name__ == "__main__":
    main()
