"""Two-coordinate phase-diversity fitting for a fixed development experiment.

The estimator sees counts and nominal physics only. It returns declared local
stationarity evidence, never a global optimum or an optical accuracy certificate.
HCIPy and evaluator state deliberately do not enter this module.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass

import numpy as np

ACTIVE_AXES = (7, 18)
SCALE_NM = 20.0
BOUND_THETA = 4.0
DIVERSITIES_M = (30e-9, -30e-9)
BACKGROUND = 0.1


@dataclass(frozen=True)
class FitCriteria:
    max_calls_per_start: int = 64
    gradient_tolerance: float = 1e-6
    objective_tolerance: float = 1e-10
    agreement_objective: float = 1e-8
    agreement_nm: float = 1e-3

    def __post_init__(self):
        if type(self.max_calls_per_start) is not int or not 3 <= self.max_calls_per_start <= 128:
            raise ValueError("max_calls_per_start must be an integer in [3, 128]")
        for name in ("gradient_tolerance", "objective_tolerance", "agreement_objective", "agreement_nm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


DEFAULT_FIT_CRITERIA = FitCriteria()


def coordinates(value, *, bound=None):
    value = np.asarray(value)
    if value.shape != (2,) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError("Expected two finite real coordinates")
    value = value.astype(np.float64)
    if bound is not None and np.any(np.abs(value) > bound):
        raise ValueError("Coordinates exceed the declared box")
    return value


def surface_command(coefficients_nm):
    command = np.zeros(25, dtype=np.float64)
    command[list(ACTIVE_AXES)] = coordinates(coefficients_nm) * 1e-9
    return command


def initial_points(center_nm=(0.0, 0.0), *, dense=False):
    """Clip first, then deduplicate; duplicate clipped starts are not evidence."""
    center = coordinates(center_nm) / SCALE_NM
    axis = (-2.0, -1.0, 0.0, 1.0, 2.0) if dense else (-1.0, 0.0, 1.0)
    offsets = [(0.0, 0.0)] + [x for x in itertools.product(axis, repeat=2) if x != (0.0, 0.0)]
    points = []
    for offset in offsets:
        point = tuple(np.clip(center + offset, -BOUND_THETA, BOUND_THETA).tolist())
        if point not in points:
            points.append(point)
    return points


def validate_counts(counts, source_photons):
    counts = np.asarray(counts)
    if counts.shape != (2, 65, 65) or counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise ValueError("Counts must be two labelled 65x65 nonnegative integer arrays")
    if np.any(counts > 2**53):
        raise ValueError("Counts must be exactly representable in float64")
    if isinstance(source_photons, bool) or not isinstance(source_photons, int | float):
        raise ValueError("Source photons must be finite and positive")
    if not math.isfinite(source_photons) or source_photons <= 0:
        raise ValueError("Source photons must be finite and positive")
    return counts.astype(np.float64)


def surface_gram(profiles, aperture):
    """Piston-removed surface Gram; coefficients and WFE are in nanometres."""
    profiles, aperture = np.asarray(profiles), np.asarray(aperture)
    if profiles.ndim != 3 or profiles.shape[0] != 2 or aperture.shape != profiles.shape[1:]:
        raise ValueError("Two physical profiles and matching aperture are required")
    if aperture.dtype != bool or not aperture.any() or not np.isfinite(profiles).all():
        raise ValueError("Finite profiles and a nonempty boolean aperture are required")
    values = profiles[:, aperture]
    values = values - np.mean(values, axis=1, keepdims=True)
    return values @ values.T / values.shape[1]


def choose_correction(residual_nm, gram, *, applied_nm=(0.0, 0.0), damping=1.0):
    """Nine bounded increments, scored by predicted physical OPD variance.

    Exact score ties choose the lexicographically first increment. Scores are
    retained rather than replacing differences with a configurable near-tie rule.
    """
    residual, applied = coordinates(residual_nm), coordinates(applied_nm, bound=40)
    gram = np.asarray(gram, dtype=np.float64)
    if gram.shape != (2, 2) or not np.isfinite(gram).all() or not np.array_equal(gram, gram.T):
        raise ValueError("Expected a finite symmetric two-mode Gram matrix")
    if np.min(np.linalg.eigvalsh(gram)) <= 0:
        raise ValueError("The declared two-mode Gram matrix must be positive definite")
    if damping not in (0.25, 0.5, 1.0):
        raise ValueError("Damping must belong to the declared finite set")
    scores = []
    for i, increment in enumerate(itertools.product((-5.0, 0.0, 5.0), repeat=2)):
        delta = np.asarray(increment)
        if np.any(np.abs(applied + delta) > 40):
            continue
        predicted = damping * residual + delta
        scores.append(
            {
                "id": f"increment-{i}",
                "increment_nm": list(increment),
                "predicted_wfe_squared_nm2": float(4 * predicted @ gram @ predicted),
            }
        )
    best = min(row["predicted_wfe_squared_nm2"] for row in scores)
    ties = [row["id"] for row in scores if row["predicted_wfe_squared_nm2"] == best]
    chosen = next(row for row in scores if row["id"] == ties[0])
    return {"chosen_id": chosen["id"], "increment_nm": chosen["increment_nm"], "ties": ties, "scores": scores}


def projected_gradient(theta, gradient):
    return theta - np.clip(theta - gradient, -BOUND_THETA, BOUND_THETA)


def fit_start(objective, start, criteria=DEFAULT_FIT_CRITERIA):
    """Bounded projected BFGS with charged Armijo proposals and confirmations.

    Stationarity -> two unchanged-point objective/gradient confirmations. These
    are additional calls, not accepted steps or independent initializations.
    An initially stationary point therefore needs no artificial movement.
    """
    theta = coordinates(start, bound=BOUND_THETA)
    start_value = theta.copy()
    inverse = np.eye(2)
    trace = []
    accepted = 0

    def evaluate(point, role):
        before = time.perf_counter()
        value, gradient = objective(point.copy())
        gradient = coordinates(gradient)
        value = float(value)
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite objective; stop the run without substituting a fit")
        row = {
            "call": len(trace),
            "theta": point.tolist(),
            "objective": value,
            "gradient": gradient.tolist(),
            "role": role,
            "seconds": time.perf_counter() - before,
        }
        trace.append(row)
        return value, gradient

    value, gradient = evaluate(theta, "initial")
    status = "proposal_cap_exhausted"
    while len(trace) < criteria.max_calls_per_start:
        if np.max(np.abs(projected_gradient(theta, gradient))) <= criteria.gradient_tolerance:
            stable = 0
            while stable < 2 and len(trace) < criteria.max_calls_per_start:
                new_value, new_gradient = evaluate(theta, "stationary_confirmation")
                if (
                    abs(new_value - value) <= criteria.objective_tolerance
                    and np.max(np.abs(projected_gradient(theta, new_gradient))) <= criteria.gradient_tolerance
                ):
                    stable += 1
                else:
                    stable = 0
                    value, gradient = new_value, new_gradient
                    break
                value, gradient = new_value, new_gradient
            if stable == 2:
                status = "stationary_confirmed"
                break
            continue
        direction = -inverse @ gradient
        if float(gradient @ direction) >= 0:
            inverse = np.eye(2)
            direction = -gradient
        alpha = 1.0
        did_accept = False
        while len(trace) < criteria.max_calls_per_start:
            proposal = np.clip(theta + alpha * direction, -BOUND_THETA, BOUND_THETA)
            delta = proposal - theta
            if not np.any(delta):
                status = "step_underflow"
                break
            new_value, new_gradient = evaluate(proposal, "line_search")
            decrease = float(gradient @ delta)
            if decrease < 0 and new_value <= value + 1e-4 * decrease:
                difference = new_gradient - gradient
                curvature = float(difference @ delta)
                if curvature > 1e-14 * np.linalg.norm(difference) * np.linalg.norm(delta):
                    transform = np.eye(2) - np.outer(delta, difference) / curvature
                    inverse = transform @ inverse @ transform.T + np.outer(delta, delta) / curvature
                else:
                    inverse = np.eye(2)
                trace[-1]["accepted"] = True
                trace[-1]["objective_change"] = new_value - value
                trace[-1]["theta_change_inf"] = float(np.max(np.abs(delta)))
                theta, value, gradient = proposal, new_value, new_gradient
                accepted += 1
                did_accept = True
                break
            trace[-1]["accepted"] = False
            alpha *= 0.5
        if not did_accept:
            break
    return {
        "start_theta": start_value.tolist(),
        "theta": theta.tolist(),
        "objective": value,
        "projected_gradient_inf": float(np.max(np.abs(projected_gradient(theta, gradient)))),
        "status": status,
        "accepted_steps": accepted,
        "calls": len(trace),
        "trace": trace,
    }


def summarize_starts(results, criteria=DEFAULT_FIT_CRITERIA):
    if len({tuple(row["start_theta"]) for row in results}) != len(results):
        raise ValueError("Multistart evidence contains duplicate initial vectors")
    if not results:
        raise ValueError("Multistart evidence is empty")
    best = min(results, key=lambda row: row["objective"])
    close = [row for row in results if abs(row["objective"] - best["objective"]) <= criteria.agreement_objective]
    stationary = [row for row in close if row["status"] == "stationary_confirmed"]
    available = len(stationary) >= 2 and all(
        np.max(np.abs(np.asarray(row["theta"]) - best["theta"])) * SCALE_NM <= criteria.agreement_nm for row in close
    )
    selected = min(stationary, key=lambda row: row["objective"]) if available else None
    return {
        "status": "locally_resolved" if available else "unresolved",
        "estimate_nm": (np.asarray(selected["theta"]) * SCALE_NM).tolist() if selected is not None else None,
        "best_objective": best["objective"],
        "agreeing_distinct_starts": len(stationary),
        "starts": results,
        "objective_calls": sum(row["calls"] for row in results),
    }


def fit_multistart(objective, *, starts=None, criteria=DEFAULT_FIT_CRITERIA):
    starts = initial_points() if starts is None else starts
    unique = []
    for point in starts:
        point = tuple(coordinates(point, bound=BOUND_THETA).tolist())
        if point not in unique:
            unique.append(point)
    results = [fit_start(objective, point, criteria) for point in unique]
    return summarize_starts(results, criteria)


def make_candidate(pupil_samples):
    """Compile one shared joint likelihood and return synchronized host outputs.

    Compilation, physical-input/Gram setup and execution are separate costs.
    The only fitted parameters are residual surface coefficients on axes7/18.
    """
    if type(pupil_samples) is not int or pupil_samples not in (64, 128, 256):
        raise ValueError("Development candidates use exactly 64, 128 and 256 pupil samples")
    import jax
    import jax.numpy as jnp

    from noether_optics.twin.static_mirror import StaticMirrorSpec, pixel_power, pupil_inputs

    if not jax.config.x64_enabled:
        raise RuntimeError("Phase-diversity fitting requires JAX float64")
    specs = tuple(StaticMirrorSpec(defocus_opd_m=value) for value in DIVERSITIES_M)

    def likelihood(theta, counts, photons):
        command = jnp.zeros(25, dtype=jnp.float64).at[jnp.asarray(ACTIVE_AXES)].set(theta * SCALE_NM * 1e-9)
        powers = jnp.stack([pixel_power(command, 1.0, spec=spec, pupil_samples=pupil_samples) for spec in specs])
        means = photons * 0.5 * powers + BACKGROUND
        return jnp.sum(means - counts * jnp.log(means)) / photons

    before = time.perf_counter()
    compiled = (
        jax.jit(jax.value_and_grad(likelihood))
        .lower(
            jax.ShapeDtypeStruct((2,), np.float64),
            jax.ShapeDtypeStruct((2, 65, 65), np.float64),
            jax.ShapeDtypeStruct((), np.float64),
        )
        .compile()
    )
    compile_seconds = time.perf_counter() - before
    before = time.perf_counter()
    inputs = pupil_inputs(specs[0], pupil_samples)
    gram = surface_gram(np.asarray(inputs["influence_functions"])[list(ACTIVE_AXES)], np.asarray(inputs["aperture"]))
    setup_seconds = time.perf_counter() - before

    def objective(theta, counts, photons):
        # All arrays return to the host before the caller's clock stops.
        value, gradient = compiled(coordinates(theta, bound=BOUND_THETA), counts, np.asarray(photons, dtype=np.float64))
        return float(np.asarray(value)), np.asarray(gradient, dtype=np.float64)

    return {
        "objective": objective,
        "gram": gram,
        "compile_seconds": compile_seconds,
        "setup_seconds": setup_seconds,
        "pupil_samples": pupil_samples,
    }
