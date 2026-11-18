"""Fresh-only development pilot for phase-diversity decision discrimination.

This module contains no selection/evaluation seeds or feedback trajectory. Fits
and choices are published before evaluator-only action consequences are acquired.
An interrupted/failed directory cannot be resumed or presented as complete.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import itertools
import math
import os
import platform
import time
from pathlib import Path

import numpy as np

from noether_optics.bench.twin_intervention import (
    _array_bytes,
    _publish,
    canonical,
    core_mask,
    file_hash,
    physical_contract,
    reference_backend,
)
from noether_optics.twin.phase_diversity import (
    ACTIVE_AXES,
    BACKGROUND,
    DIVERSITIES_M,
    SCALE_NM,
    FitCriteria,
    choose_correction,
    coordinates,
    fit_multistart,
    initial_points,
    make_candidate,
    projected_gradient,
    surface_command,
    surface_gram,
    validate_counts,
)

SCHEMA = "noether-twin-development-v1"
RESOLUTIONS = (64, 128, 256)
# Four fixed signed/mixed states crossed with both fluxes. Not selected for a
# winning resolution. Half-action-step anchors examine the decision boundary;
# large anchors examine correction saturation. Redesigns retain this outcome.
STATES_NM = ((-25.0, -15.0), (-2.5, 2.5), (2.5, -2.5), (25.0, 15.0))
FLUXES = (10000, 1000000)
ANCHORS = (0, 7)
LIMITS = {"objective": 100000, "reference": 1500, "packet_bytes": 512 * 1024**2}
SOURCE_FILES = (
    "src/noether_optics/twin/static_mirror.py",
    "src/noether_optics/twin/phase_diversity.py",
    "src/noether_optics/bench/twin_intervention.py",
    "src/noether_optics/bench/twin_development.py",
    "pyproject.toml",
    "uv.lock",
)


def protocol():
    return {
        "schema": SCHEMA,
        "purpose": "development_discrimination_and_convergence_only",
        "physical": physical_contract(),
        "diversities_opd_m": list(DIVERSITIES_M),
        "active_surface_axes": list(ACTIVE_AXES),
        "plant_grid": 512,
        "diagnostic_grid": 1024,
        "candidate_grids": list(RESOLUTIONS),
        "states_nm": [list(x) for x in STATES_NM],
        "source_photons_per_pair": list(FLUXES),
        "background_per_pixel_per_exposure": BACKGROUND,
        "noise": "independent_poisson",
        "command_gain": 1.0,
        "damping": 1.0,
        "applied_command_nm": [0.0, 0.0],
        "detector_seed_namespace": "twin-development-v1-20261110",
        "detector_seeds": [6111000 + i for i in range(8)],
        "adequacy_anchors": list(ANCHORS),
        "fit": dataclasses.asdict(FitCriteria()),
        "expanded_fit": dataclasses.asdict(FitCriteria(128)),
        "initial_starts": initial_points(),
        "expanded_starts": initial_points(dense=True),
        "stationary_confirmation_calls": 2,
        "timing_repetitions_per_grid": 32,
        "timing_scope": "gradient_proposal_and_command_component_only; decision_budget_not_frozen",
        "reference_ranking": {
            "roundoff_tolerance_nm2": "64*float64_epsilon*max(1,abs(each_of_four_action_losses))",
            "resolved_pair": "absolute_contrast_exceeds_roundoff_on_both_grids",
            "gate": "at_least_one_resolved_pair_and_no_resolved_sign_reversal; retain_unresolved_pairs",
            "near_best_rule": "same_four_loss_roundoff_relative_to_each_grid_minimum; union_grids_then_same_states",
            "state_dependence": "disjoint_near_best_unions_for_two_distinct_physical_states",
            "separation_is_distinct": "abs(coarse_contrast)>abs(coarse_minus_fine_contrast)+roundoff",
        },
        "limits": LIMITS,
        "held_out_generation": False,
    }


def _json_values(value):
    """Retain nonfinite callback scalars explicitly in failed journal entries."""
    if isinstance(value, np.ndarray):
        return _json_values(value.tolist())
    if isinstance(value, np.generic):
        return _json_values(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_values(item) for item in value]
    return value


class DevelopmentPacket:
    """One fresh attempt; immutable raw outputs and append-only dispatch journal.

    External process supervision remains mandatory for wall/RSS/storage reserve.
    Local checks bound callbacks and bytes; this is not a resumable experiment.
    """

    def __init__(self, root, source_root, *, study_protocol=None, source_files=None, limits=None):
        self.limits = LIMITS if limits is None else dict(limits)
        self.root = Path(root)
        self.root.mkdir(exist_ok=False)
        (self.root / "arrays").mkdir()
        self.counts = {name: 0 for name in self.limits if name != "packet_bytes"}
        self.events = 0
        self.bytes = 0
        self.started = time.perf_counter()
        self.stream = (self.root / "events.jsonl").open("xb")
        header = {
            "protocol": protocol() if study_protocol is None else study_protocol,
            "source": {
                name: file_hash(Path(source_root) / name)
                for name in (SOURCE_FILES if source_files is None else source_files)
            },
            "runtime": {
                "python": platform.python_version(),
                "machine": platform.machine(),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "scipy", "hcipy", "dLux", "jax", "jaxlib")
                },
            },
        }
        self.publish("protocol.json", canonical(header))

    def publish(self, name, payload):
        if self.bytes + len(payload) > self.limits["packet_bytes"]:
            raise RuntimeError("Development packet byte cap would be exceeded")
        _publish(self.root / name, payload)
        self.bytes += len(payload)
        return {"path": name, "sha256": hashlib.sha256(payload).hexdigest()}

    def event(self, kind, **fields):
        payload = canonical(_json_values({"event": self.events, "kind": kind, **fields})) + b"\n"
        if self.bytes + len(payload) > self.limits["packet_bytes"]:
            raise RuntimeError("Development journal byte cap would be exceeded")
        self.stream.write(payload)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.bytes += len(payload)
        self.events += 1

    def call(self, kind, context, callback, *, arrays=False):
        if kind not in self.counts or self.counts[kind] >= self.limits[kind]:
            raise RuntimeError("Development callback cap would be exceeded")
        identity = f"{kind}-{self.counts[kind]:06d}"
        self.event("dispatch", identity=identity, callback_kind=kind, context=context)
        self.counts[kind] += 1
        before = time.perf_counter()
        try:
            value = callback()
            elapsed = time.perf_counter() - before
            if arrays:
                # Publish exact returned arrays even when they are nonfinite.
                retained = {name: np.asarray(item) for name, item in value.items()}
                receipt = self.publish(f"arrays/{identity}.npz", _array_bytes(retained))
                finite = all(np.isfinite(item).all() for item in retained.values())
                self.event(
                    "result",
                    identity=identity,
                    seconds=elapsed,
                    raw=receipt,
                    status="complete" if finite else "nonfinite",
                )
            else:
                loss, gradient = value
                finite = math.isfinite(float(loss)) and np.isfinite(gradient).all()
                self.event(
                    "result",
                    identity=identity,
                    seconds=elapsed,
                    output={"objective": loss, "gradient": gradient},
                    status="complete" if finite else "nonfinite",
                )
            if not finite:
                raise FloatingPointError("Nonfinite callback is retained and stops this development attempt")
            return value
        except BaseException as error:
            self.event(
                "exception",
                identity=identity,
                seconds=time.perf_counter() - before,
                exception_type=type(error).__name__,
                message=str(error),
            )
            raise

    def close(self):
        if not self.stream.closed:
            self.stream.close()


def reference_profiles(packet, backend, n):
    profiles = []
    aperture = None
    for axis in range(2):
        unit = np.eye(2)[axis]
        values = packet.call(
            "reference",
            {"phase": "physical_basis", "grid": n, "axis": axis},
            lambda unit=unit: backend(n, surface_command(unit), 1.0, component=True),
            arrays=True,
        )
        profiles.append(values["surface"] / 1e-9)
        if aperture is not None and not np.array_equal(aperture, values["aperture"]):
            raise ValueError("Reference basis aperture changed")
        aperture = values["aperture"]
    return surface_gram(np.asarray(profiles), aperture)


def reference_image(packet, backend, n, coefficients_nm, context):
    def acquire():
        values = backend(n, surface_command(coefficients_nm), 1.0)
        return {"power": values["power"], "incident": values["incident"]}

    values = packet.call(
        "reference", {**context, "grid": n, "residual_nm": list(coefficients_nm)}, acquire, arrays=True
    )
    power = values["power"]
    if power.shape != (65, 65) or power.dtype != np.float64 or np.any(power < 0):
        raise ValueError("Reference image violates the fixed focal power contract")
    return power


def independent_fit(objective, starts):
    """A separate bounded SciPy optimizer on the identical likelihood/box.

    Its status and projected gradient are retained. It is an adequacy diagnostic,
    not a substitute that turns a failed primary fit into a successful arm.
    """
    from scipy.optimize import minimize

    rows = []
    for start in starts:
        calls = []

        def function(theta, calls=calls):
            if len(calls) >= 128:
                raise RuntimeError("independent_optimizer_call_cap")
            value, gradient = objective(theta)
            calls.append({"theta": theta.tolist(), "objective": float(value), "gradient": gradient.tolist()})
            return value, gradient

        before = time.perf_counter()
        try:
            result = minimize(
                function,
                np.asarray(start),
                jac=True,
                method="L-BFGS-B",
                bounds=[(-4, 4)] * 2,
                options={"maxiter": 128, "maxfun": 128, "ftol": 1e-15, "gtol": 1e-8, "maxls": 30},
            )
            pg = float(np.max(np.abs(projected_gradient(result.x, result.jac))))
            rows.append(
                {
                    "start_theta": list(start),
                    "theta": result.x.tolist(),
                    "objective": float(result.fun),
                    "status": "stationary" if result.success and pg <= 1e-6 else "unresolved",
                    "solver_message": str(result.message),
                    "projected_gradient_inf": pg,
                    "seconds": time.perf_counter() - before,
                    "trace": calls,
                }
            )
        except RuntimeError as error:
            if str(error) != "independent_optimizer_call_cap":
                raise
            rows.append(
                {
                    "start_theta": list(start),
                    "status": "proposal_cap_exhausted",
                    "seconds": time.perf_counter() - before,
                    "trace": calls,
                }
            )
    return rows


def adequacy(primary, expanded, independent, gram):
    if primary["status"] != "locally_resolved" or expanded["status"] != "locally_resolved":
        return {"passed": False, "reason": "primary_or_expanded_fit_unresolved"}
    candidates = [row for row in independent if row["status"] == "stationary"]
    if len(candidates) != 2:
        return {"passed": False, "reason": "independent_optimizer_unresolved"}
    estimates = [primary["estimate_nm"], expanded["estimate_nm"]]
    estimates.extend((np.asarray(row["theta"]) * SCALE_NM).tolist() for row in candidates)
    objectives = [primary["best_objective"], expanded["best_objective"]]
    objectives.extend(row["objective"] for row in candidates)
    spread = max(float(np.max(np.abs(np.asarray(point) - estimates[0]))) for point in estimates)
    actions = [choose_correction(point, gram)["chosen_id"] for point in estimates]
    passed = spread <= 1e-3 and max(objectives) - min(objectives) <= 1e-8 and len(set(actions)) == 1
    return {
        "passed": bool(passed),
        "reason": "checked" if passed else "fit_or_action_disagreement",
        "coefficient_spread_nm": spread,
        "objective_spread": max(objectives) - min(objectives),
        "chosen_ids": actions,
    }


def action_scores(state_nm, gram):
    return {
        f"increment-{i}": float(4 * (np.asarray(state_nm) + delta) @ gram @ (np.asarray(state_nm) + delta))
        for i, delta in enumerate(itertools.product((-5.0, 0.0, 5.0), repeat=2))
    }


def discrimination(cases):
    """Separate roundoff-qualified ranking from finite-grid sensitivity.

    The ranking gate requires at least one resolved pair and no reversal among
    pairs resolved on BOTH grids. Near-zero pairs remain unresolved; passing the
    gate does not assert their ordering or certify continuum accuracy.
    """
    nonzero_contrast = False
    best_by_case = []
    best_by_state = {}
    separated_disagreement = False
    contrasts = []
    agreed_pairs, reversed_pairs, unresolved_pairs = [], [], []
    for case in cases:
        coarse, fine = case["reference_wfe_squared_nm2"], case["diagnostic_wfe_squared_nm2"]
        ids = sorted(coarse)
        state = tuple(coordinates(case["evaluator_state_nm"]).tolist())
        per_grid = {}
        near_best_union = set()
        for grid, losses in ((512, coarse), (1024, fine)):
            best_id = min(ids, key=lambda name: (losses[name], name))
            minimum = losses[best_id]
            tolerances = {
                name: 64
                * np.finfo(np.float64).eps
                * max(1.0, abs(coarse[name]), abs(coarse[best_id]), abs(fine[name]), abs(fine[best_id]))
                for name in ids
            }
            near_best = [name for name in ids if losses[name] - minimum <= tolerances[name]]
            per_grid[str(grid)] = {
                "minimum_loss_nm2": minimum,
                "near_best_ids": near_best,
                "roundoff_tolerance_nm2_by_action": tolerances,
            }
            near_best_union.update(near_best)
        best_by_case.append(
            {
                "case": case["id"],
                "physical_state_nm": list(state),
                "grids": per_grid,
                "near_best_union": sorted(near_best_union),
            }
        )
        group = best_by_state.setdefault(state, {"cases": [], "near_best_union": set()})
        group["cases"].append(case["id"])
        group["near_best_union"].update(near_best_union)
        for left, right in itertools.combinations(ids, 2):
            c, f = coarse[left] - coarse[right], fine[left] - fine[right]
            scale = max(1.0, abs(coarse[left]), abs(coarse[right]), abs(fine[left]), abs(fine[right]))
            roundoff = 64 * np.finfo(np.float64).eps * scale
            pair = {
                "case": case["id"],
                "actions": [left, right],
                "reference_difference_nm2": c,
                "diagnostic_difference_nm2": f,
                "roundoff_tolerance_nm2": roundoff,
            }
            # This test does not use |c-f|: genuine sign reversals must be
            # detectable independently of the subsequent separation criterion.
            if abs(c) <= roundoff or abs(f) <= roundoff:
                pair["unresolved_grids"] = [grid for grid, value in ((512, c), (1024, f)) if abs(value) <= roundoff]
                unresolved_pairs.append(pair)
            elif (c > 0) == (f > 0):
                agreed_pairs.append(pair)
            else:
                reversed_pairs.append(pair)
            if abs(c) > abs(c - f) + roundoff:
                nonzero_contrast = True
        available = [arm for arm in case["arms"] if arm["fit"]["status"] == "locally_resolved"]
        for left, right in itertools.combinations(available, 2):
            a, b = left["choice"]["chosen_id"], right["choice"]["chosen_id"]
            if a == b:
                continue
            c, f = coarse[a] - coarse[b], fine[a] - fine[b]
            scale = max(1.0, abs(coarse[a]), abs(coarse[b]), abs(fine[a]), abs(fine[b]))
            numerical = abs(c - f) + 64 * np.finfo(np.float64).eps * scale
            separated = abs(c) > numerical and (c > 0) == (f > 0)
            separated_disagreement |= bool(separated)
            contrasts.append(
                {
                    "case": case["id"],
                    "grids": [left["grid"], right["grid"]],
                    "actions": [a, b],
                    "reference_difference_nm2": c,
                    "diagnostic_difference_nm2": f,
                    "sensitivity_nm2": numerical,
                    "separated": bool(separated),
                }
            )
    state_sets = [
        {
            "physical_state_nm": list(state),
            "cases": values["cases"],
            "near_best_union": sorted(values["near_best_union"]),
        }
        for state, values in sorted(best_by_state.items())
    ]
    separated_states = [
        {
            "physical_states_nm": [left["physical_state_nm"], right["physical_state_nm"]],
            "near_best_unions": [left["near_best_union"], right["near_best_union"]],
        }
        for left, right in itertools.combinations(state_sets, 2)
        if set(left["near_best_union"]).isdisjoint(right["near_best_union"])
    ]
    state_dependent = bool(separated_states)
    ranking_passed = bool(agreed_pairs) and not reversed_pairs
    ranking = {
        "agreed_pairs": agreed_pairs,
        "reversed_pairs": reversed_pairs,
        "unresolved_pairs": unresolved_pairs,
        "resolved_pair_ordering_stable": ranking_passed,
        "all_pairs_resolved": not unresolved_pairs,
        "gate_requires": "at_least_one_resolved_pair_and_no_resolved_sign_reversal",
    }
    return {
        "reference_action_contrast": nonzero_contrast,
        "reference_ranking": ranking,
        "state_dependent_best_actions": state_dependent,
        "near_best_actions_by_case": best_by_case,
        "near_best_actions_by_state": state_sets,
        "separated_physical_states": separated_states,
        "cross_model_separated_disagreement": separated_disagreement,
        "contrasts": contrasts,
        "passed": nonzero_contrast and ranking_passed and state_dependent and separated_disagreement,
    }


def run_development(root, *, source_root):
    packet = DevelopmentPacket(root, source_root)
    try:
        # Separate factories prevent diversity/grid cache aliasing. Reference
        # arrays are never passed into a candidate model or fitting objective.
        references = [reference_backend(defocus_opd_m=value) for value in DIVERSITIES_M]
        science = reference_backend(defocus_opd_m=0.0)
        reference_gram = reference_profiles(packet, science, 512)
        diagnostic_gram = reference_profiles(packet, science, 1024)
        models = {}
        for n in RESOLUTIONS:
            packet.event("candidate_setup_dispatch", grid=n)
            models[n] = make_candidate(n)
            packet.event(
                "candidate_setup_complete",
                grid=n,
                compile_seconds=models[n]["compile_seconds"],
                gram_setup_seconds=models[n]["setup_seconds"],
                gram=models[n]["gram"],
            )
        cases = []
        timing_inputs = None
        for i, (photons, state) in enumerate(itertools.product(FLUXES, STATES_NM)):
            identity = f"development-{i:02d}"
            means = np.stack(
                [
                    photons
                    * 0.5
                    * reference_image(
                        packet, backend, 512, state, {"phase": "observation", "case": identity, "exposure": j}
                    )
                    + BACKGROUND
                    for j, backend in enumerate(references)
                ]
            )
            counts = np.random.default_rng(6111000 + i).poisson(means)
            raw = packet.publish(
                f"arrays/{identity}-observations.npz", _array_bytes({"counts": counts, "means": means})
            )
            packet.event(
                "observations",
                case=identity,
                seed=6111000 + i,
                raw=raw,
                photons=photons,
                evaluator_state_nm=list(state),
            )
            counts_float = validate_counts(counts, photons)
            if i == 0:
                timing_inputs = counts.copy(), photons
            arms = []
            for n in RESOLUTIONS:
                model = models[n]

                def objective(theta, model=model, n=n, identity=identity, counts_float=counts_float, photons=photons):
                    return packet.call(
                        "objective",
                        {"case": identity, "grid": n, "theta": theta.tolist()},
                        lambda: model["objective"](theta, counts_float, photons),
                    )

                before = time.perf_counter()
                fitted = fit_multistart(objective)
                fit_seconds = time.perf_counter() - before
                before = time.perf_counter()
                choice = (
                    choose_correction(fitted["estimate_nm"], model["gram"])
                    if fitted["estimate_nm"] is not None
                    else {"chosen_id": "increment-4", "increment_nm": [0.0, 0.0], "status": "hold_unresolved_fit"}
                )
                choice_seconds = time.perf_counter() - before
                arm = {
                    "grid": n,
                    "fit": fitted,
                    "fit_including_journal_seconds": fit_seconds,
                    "choice_seconds": choice_seconds,
                    "choice": choice,
                }
                if i in ANCHORS:
                    expanded = fit_multistart(objective, starts=initial_points(dense=True), criteria=FitCriteria(128))
                    independent = independent_fit(objective, ((0.0, 0.0), (1.0, -1.0)))
                    arm["adequacy"] = adequacy(fitted, expanded, independent, model["gram"])
                    arm["expanded_fit"], arm["independent_fit"] = expanded, independent
                arms.append(arm)
            frozen = packet.publish(f"{identity}-choices.json", canonical({"id": identity, "arms": arms}))
            packet.event("choices_frozen", case=identity, raw=frozen)
            # Only now score all legal actions on the evaluator's physical grids.
            losses512, losses1024 = action_scores(state, reference_gram), action_scores(state, diagnostic_gram)
            core = {}
            for index, delta in enumerate(itertools.product((-5.0, 0.0, 5.0), repeat=2)):
                corrected = np.asarray(state) + delta
                image = reference_image(
                    packet,
                    science,
                    512,
                    corrected,
                    {"phase": "counterfactual", "case": identity, "action": f"increment-{index}"},
                )
                core[f"increment-{index}"] = float(np.sum(image[core_mask()]))
            case = {
                "id": identity,
                "source_photons": photons,
                "evaluator_state_nm": list(state),
                "arms": arms,
                "reference_wfe_squared_nm2": losses512,
                "diagnostic_wfe_squared_nm2": losses1024,
                "reference_core_power": core,
            }
            cases.append(case)
            packet.publish(f"{identity}-outcomes.json", canonical(case))
        timing = {}
        for n, model in models.items():
            durations = []
            raw_counts, photons = timing_inputs
            for repeat in range(32):
                # Preprocessing, transfer, synchronized derivative and command
                # construction included; journal I/O explicitly outside interval.
                measured = {}

                def timed_iteration(raw_counts=raw_counts, photons=photons, model=model, measured=measured):
                    before = time.perf_counter()
                    counts_float = validate_counts(raw_counts, photons)
                    theta = np.zeros(2)
                    value, gradient = model["objective"](theta, counts_float, photons)
                    proposal = np.clip(theta - gradient, -4, 4)
                    command = choose_correction(proposal * SCALE_NM, model["gram"])
                    measured.update(seconds=time.perf_counter() - before, command=command)
                    return value, gradient

                packet.call("objective", {"phase": "timing", "grid": n, "repetition": repeat}, timed_iteration)
                durations.append(measured["seconds"])
                packet.event("timing_complete", grid=n, repetition=repeat, **measured)
            timing[str(n)] = {
                "seconds": durations,
                "p95_seconds": float(np.quantile(durations, 0.95)),
                "scope": "one_gradient_proposal_plus_command_not_complete_converged_fit",
            }
        gate = discrimination(cases)
        checked = all(arm["adequacy"]["passed"] for i in ANCHORS for arm in cases[i]["arms"])
        all_fits_resolved = all(arm["fit"]["status"] == "locally_resolved" for case in cases for arm in case["arms"])
        result = {
            "schema": SCHEMA,
            "status": "complete_development",
            "cases": cases,
            "discrimination": gate,
            "anchor_adequacy_passed": checked,
            "all_primary_fits_resolved": all_fits_resolved,
            "timing": timing,
            "decision_budget_frozen": False,
            "development_discrimination_passed": bool(gate["passed"] and checked and all_fits_resolved),
            "counts": packet.counts,
            "timing_objective_calls": 96,
            "total_seconds": time.perf_counter() - packet.started,
            "interpretation": "Development only; no equal-resource or closed-loop outcome has been measured.",
        }
        packet.publish("analysis.json", canonical(result))
        packet.event("complete", counts=packet.counts)
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
    run_development(args.output, source_root=Path(__file__).resolve().parents[3])


if __name__ == "__main__":
    main()
