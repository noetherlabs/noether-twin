"""One-axis registration fitting for a known-flat focal-plane diagnostic.

Only calibration images enter the estimator. Multistart agreement records local
stationarity evidence, not global uniqueness, physical metrology or useful control.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from noether_optics.twin.phase_diversity import BACKGROUND, DIVERSITIES_M

BOUND_PITCH = 0.20
STARTS_PITCH = (-0.20, -0.10, 0.0, 0.10, 0.20)
PHOTONS_PER_PAIR = 1_000_000
CALIBRATION_ACTUATORS = (12, 7)
PREDICTION_ACTUATORS = (18,)
SURFACE_COMMAND_M = 10e-9


@dataclass(frozen=True)
class RegistrationCriteria:
    max_calls_per_start: int = 64
    gradient_tolerance: float = 1e-6
    objective_tolerance: float = 1e-10
    agreement_objective: float = 1e-8
    agreement_pitch: float = 1e-4

    def __post_init__(self):
        if type(self.max_calls_per_start) is not int or not 3 <= self.max_calls_per_start <= 128:
            raise ValueError("Call cap must be an integer in [3, 128]")
        for name in ("gradient_tolerance", "objective_tolerance", "agreement_objective", "agreement_pitch"):
            scalar(getattr(self, name), name, positive=True)


def scalar(value, name="registration", *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return value


CRITERIA = RegistrationCriteria()


def schedule(actuators):
    """Actuator/sign/diversity order; coordinates and signs are not radial bins."""
    return tuple(
        (actuator, sign, diversity) for actuator in actuators for sign in (1, -1) for diversity in DIVERSITIES_M
    )


def command_for(actuator, sign):
    if type(actuator) is not int or not 0 <= actuator < 25 or sign not in (-1, 1):
        raise ValueError("Expected a physical actuator and signed surface command")
    result = np.zeros(25, dtype=np.float64)
    result[actuator] = sign * SURFACE_COMMAND_M
    return result


def images(value, exposures, *, expected=False):
    """Separate real expected means from integer observations; never cast means to counts."""
    value = np.asarray(value)
    kinds = "fiu" if expected else "iu"
    if value.shape != (exposures, 65, 65) or value.dtype.kind not in kinds:
        raise ValueError("Expected labelled 65x65 means" if expected else "Expected labelled 65x65 integer counts")
    if not np.isfinite(value).all() or np.any(value < 0) or (not expected and np.any(value > 2**53)):
        raise ValueError("Images must be finite, nonnegative and counts exactly representable")
    return value.astype(np.float64)


def projected_gradient(point, gradient):
    return point - float(np.clip(point - gradient, -BOUND_PITCH, BOUND_PITCH))


def fit_start(objective, start, criteria=CRITERIA):
    """Projected scalar secant steps with Armijo search and charged confirmation.

    Every proposal, rejected step and stationary confirmation is charged. The
    inverse secant curvature resets to one when curvature is nonpositive.
    """
    point = float(np.clip(scalar(start), -BOUND_PITCH, BOUND_PITCH))
    initial = point
    inverse = 1.0
    trace = []

    def evaluate(x, role):
        before = time.perf_counter()
        value, gradient = objective(x)
        value, gradient = float(value), float(gradient)
        if not math.isfinite(value) or not math.isfinite(gradient):
            raise FloatingPointError("Nonfinite objective/gradient; retain callback and stop")
        trace.append(
            {
                "point_pitch": x,
                "objective": value,
                "gradient_per_pitch": gradient,
                "role": role,
                "seconds": time.perf_counter() - before,
            }
        )
        return value, gradient

    value, gradient = evaluate(point, "initial")
    status = "call_cap_exhausted"
    while len(trace) < criteria.max_calls_per_start:
        if abs(projected_gradient(point, gradient)) <= criteria.gradient_tolerance:
            stable = 0
            while stable < 2 and len(trace) < criteria.max_calls_per_start:
                next_value, next_gradient = evaluate(point, "stationary_confirmation")
                if (
                    abs(next_value - value) <= criteria.objective_tolerance
                    and abs(projected_gradient(point, next_gradient)) <= criteria.gradient_tolerance
                ):
                    stable += 1
                else:
                    stable = 0
                    value, gradient = next_value, next_gradient
                    break
                value, gradient = next_value, next_gradient
            if stable == 2:
                status = "stationary_confirmed"
                break
            continue
        alpha, accepted = 1.0, False
        while len(trace) < criteria.max_calls_per_start:
            proposal = float(np.clip(point - alpha * inverse * gradient, -BOUND_PITCH, BOUND_PITCH))
            step = proposal - point
            if step == 0:
                status = "step_underflow"
                break
            next_value, next_gradient = evaluate(proposal, "line_search")
            accepted = gradient * step < 0 and next_value <= value + 1e-4 * gradient * step
            trace[-1]["accepted"] = bool(accepted)
            if accepted:
                difference = next_gradient - gradient
                inverse = step / difference if step * difference > 0 else 1.0
                point, value, gradient = proposal, next_value, next_gradient
                break
            alpha *= 0.5
        if not accepted:
            break
    return {
        "start_pitch": initial,
        "point_pitch": point,
        "objective": value,
        "gradient_per_pitch": gradient,
        "projected_gradient": projected_gradient(point, gradient),
        "at_boundary": abs(point) == BOUND_PITCH,
        "status": status,
        "calls": len(trace),
        "trace": trace,
    }


def fit_registration(objective, *, starts=STARTS_PITCH, criteria=CRITERIA):
    """All starts must finish; separated near-best points and flat fits abstain."""
    unique = sorted({float(np.clip(scalar(x), -BOUND_PITCH, BOUND_PITCH)) for x in starts})
    if not unique:
        raise ValueError("At least one start is required")
    rows = [fit_start(objective, x, criteria) for x in unique]
    best = min(rows, key=lambda row: row["objective"])
    close = [row for row in rows if row["objective"] - best["objective"] <= criteria.agreement_objective]
    spread = max(row["point_pitch"] for row in close) - min(row["point_pitch"] for row in close)
    all_stationary = all(row["status"] == "stationary_confirmed" for row in rows)
    resolved = all_stationary and len(close) >= 2 and spread <= criteria.agreement_pitch
    return {
        "status": "locally_resolved" if resolved else "unresolved",
        "estimate_pitch": best["point_pitch"] if resolved else None,
        "prediction_pitch": best["point_pitch"] if resolved else 0.0,
        "fallback": None if resolved else "nominal_registration",
        "near_best_spread_pitch": spread,
        "near_best_distinct_starts": len(close),
        "all_starts_stationary": all_stationary,
        "boundary_solution": resolved and best["at_boundary"],
        "objective_calls": sum(row["calls"] for row in rows),
        "starts": rows,
    }


def expected_means(power):
    power = images(power, len(power), expected=True)
    return PHOTONS_PER_PAIR * 0.5 * power + BACKGROUND


def prediction_loss(predicted_means, target, *, expected=False):
    """Held-out NLL (no factorial) or expected Poisson KL, per incident photon."""
    prediction = images(predicted_means, 4, expected=True)
    observed = images(target, 4, expected=expected)
    if np.any(prediction <= 0):
        raise ValueError("Predicted means must be strictly positive")
    if expected:
        # x log x/y is zero at x=0. Keep deterministic means distinct from counts.
        terms = prediction - observed
        nonzero = observed > 0
        terms[nonzero] += observed[nonzero] * np.log(observed[nonzero] / prediction[nonzero])
    else:
        terms = prediction - observed * np.log(prediction)
    value = float(np.sum(terms) / (2 * PHOTONS_PER_PAIR))
    if not math.isfinite(value):
        raise FloatingPointError("Nonfinite prediction loss")
    return value


def make_registration_candidate(pupil_samples):
    """Compile the same eight-image likelihood and four-image predictor per grid.

    Registration is a dynamic argument; no static geometry cache may reuse a
    previous shift. JAX execution includes host transfer and synchronization.
    """
    if type(pupil_samples) is not int or pupil_samples not in (64, 128, 256):
        raise ValueError("Registration candidates use 64, 128 or 256 samples")
    import jax
    import jax.numpy as jnp

    from noether_optics.twin.static_mirror import StaticMirrorSpec, pixel_power

    if not jax.config.x64_enabled:
        raise RuntimeError("Registration study requires JAX float64")
    specs = [StaticMirrorSpec(defocus_opd_m=x) for x in DIVERSITIES_M]

    def powers(shift, actuators):
        commands = jnp.asarray([command_for(actuator, sign) for actuator in actuators for sign in (1, -1)])
        # Keep exposure order actuator -> sign -> labelled diversity.
        pairs = [
            jax.vmap(
                lambda c, spec=spec: pixel_power(
                    c, 1.0, spec=spec, pupil_samples=pupil_samples, registration_x_pitch=shift
                )
            )(commands)
            for spec in specs
        ]
        return jnp.stack(pairs, axis=1).reshape((-1, 65, 65))

    def likelihood(shift, calibration):
        means = PHOTONS_PER_PAIR * 0.5 * powers(shift, CALIBRATION_ACTUATORS) + BACKGROUND
        return jnp.sum(means - calibration * jnp.log(means)) / (4 * PHOTONS_PER_PAIR)

    before = time.perf_counter()
    objective = (
        jax.jit(jax.value_and_grad(likelihood))
        .lower(jax.ShapeDtypeStruct((), np.float64), jax.ShapeDtypeStruct((8, 65, 65), np.float64))
        .compile()
    )
    objective_compile_seconds = time.perf_counter() - before
    before = time.perf_counter()
    predict = (
        jax.jit(lambda shift: powers(shift, PREDICTION_ACTUATORS)).lower(jax.ShapeDtypeStruct((), np.float64)).compile()
    )
    prediction_compile_seconds = time.perf_counter() - before

    def evaluate(shift, calibration):
        shift = scalar(shift)
        if abs(shift) > BOUND_PITCH:
            raise ValueError("Registration exceeds fitted box")
        value, derivative = objective(np.asarray(shift), calibration)
        return float(np.asarray(value)), float(np.asarray(derivative))

    def prediction(shift):
        return np.asarray(predict(np.asarray(scalar(shift))), dtype=np.float64)

    return {
        "objective": evaluate,
        "prediction": prediction,
        "objective_compile_seconds": objective_compile_seconds,
        "prediction_compile_seconds": prediction_compile_seconds,
    }
