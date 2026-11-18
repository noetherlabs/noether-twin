"""Static mirror response on a fixed physical focal grid, using dLux.

The pupil is point sampled; focal ``pixel_power`` is midpoint quadrature,
not an integrated detector response. A continuous unit-flux illumination is
sampled without renormalizing either the pupil or the finite focal image.

Enable JAX float64 before using this module. ``spec`` and ``pupil_samples``
must be static arguments under JIT. Concrete commands, gains and directions
are validated eagerly; callers must validate their values before passing
them into a compiled function. Traced values are never coerced to Python
booleans or silently clipped. This is a trusted numerical API, not a sandbox.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import jax
import jax.numpy as jnp
import numpy as np
from dLux import Wavefront


@dataclass(frozen=True)
class StaticMirrorSpec:
    """Nominal geometry and illumination; no plant gain or fitted parameters.

    Array axes are ``[y, x]`` with both physical coordinates increasing with
    index. Actuators have row-major ordering, with x varying fastest. The
    aperture includes its outer boundary and excludes its inner boundary,
    matching a closed outer disk minus a closed obscuration disk.
    """

    diameter_m: float = 8.0
    obscuration_ratio: float = 0.14
    wavelength_m: float = 551e-9
    focal_length_m: float = 64.0
    actuators_across: int = 5
    coupling: float = 0.45
    cutoff_spacings: float = 3.0
    output_samples: int = 65
    pixels_per_lambda_d: float = 4.0
    defocus_opd_m: float = 30e-9

    def __post_init__(self) -> None:
        for name in (
            "diameter_m",
            "obscuration_ratio",
            "wavelength_m",
            "focal_length_m",
            "coupling",
            "cutoff_spacings",
            "pixels_per_lambda_d",
            "defocus_opd_m",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        for name in ("diameter_m", "wavelength_m", "focal_length_m", "cutoff_spacings", "pixels_per_lambda_d"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.obscuration_ratio < 1:
            raise ValueError("obscuration_ratio must lie in [0, 1)")
        if not 0 < self.coupling < 1:
            raise ValueError("coupling must lie in (0, 1)")
        _positive_integer(self.actuators_across, "actuators_across")
        _positive_integer(self.output_samples, "output_samples")
        if self.output_samples % 2 != 1:
            raise ValueError("output_samples must be odd, so the focal grid contains the origin")
        area = math.pi * (self.diameter_m / 2) ** 2 * (1 - self.obscuration_ratio**2)
        if not math.isfinite(area) or area <= 0:
            raise ValueError("physical annulus area must be finite and positive")
        if not math.isfinite(self.output_pixel_scale_m) or self.output_pixel_scale_m <= 0:
            raise ValueError("output pixel scale must be finite and positive")

    @property
    def actuator_spacing_m(self) -> float:
        return self.diameter_m / self.actuators_across

    @property
    def output_pixel_scale_m(self) -> float:
        return self.wavelength_m * self.focal_length_m / (self.pixels_per_lambda_d * self.diameter_m)

    @property
    def actuator_count(self) -> int:
        return self.actuators_across**2


def _positive_integer(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def _require_x64() -> None:
    if not jax.config.x64_enabled:
        raise RuntimeError("Static mirror qualification requires JAX float64; enable x64 before constructing arrays")


def _real_array(value: object, name: str, shape: tuple[int, ...]) -> jax.Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.number) or jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise ValueError(f"{name} must contain real numbers")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    array = array.astype(jnp.float64)
    if not isinstance(array, jax.core.Tracer) and not np.isfinite(np.asarray(array)).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _command_and_gain(command_m: object, gain: object, spec: StaticMirrorSpec) -> tuple[jax.Array, jax.Array]:
    _require_x64()
    command = _real_array(command_m, "command_m", (spec.actuator_count,))
    scalar_gain = _real_array(gain, "gain", ())
    if not isinstance(scalar_gain, jax.core.Tracer) and float(scalar_gain) <= 0:
        raise ValueError("gain must be positive")
    return command, scalar_gain


def pupil_inputs(
    spec: StaticMirrorSpec, pupil_samples: int, *, registration_x_pitch: object = 0.0
) -> dict[str, jax.Array]:
    """Return the independently constructed physical pupil inputs.

    ``x_m``, ``y_m``, ``aperture``, ``diversity_opd_m`` and ``input_field``
    have shape (N, N). ``influence_functions`` has shape (actuator, N, N),
    ``actuator_xy_m`` has shape (actuator, 2), and ``pixel_scale_m`` is scalar.
    Influences describe surface displacement per metre of actuator command.
    They are not masked by the aperture, normalized to their peak, or imported
    from the reference implementation. ``registration_x_pitch`` translates only
    the actuator centres along +x in projected actuator-pitch units; pupil,
    influence width, focal coordinates and illumination remain fixed.
    """
    _require_x64()
    _positive_integer(pupil_samples, "pupil_samples", minimum=2)
    registration = _real_array(registration_x_pitch, "registration_x_pitch", ())
    dx = spec.diameter_m / pupil_samples
    axis = (jnp.arange(pupil_samples, dtype=jnp.float64) - (pupil_samples - 1) / 2) * dx
    x, y = jnp.meshgrid(axis, axis, indexing="xy")
    radius_squared = x**2 + y**2
    outer_radius_squared = (spec.diameter_m / 2) ** 2
    aperture = (radius_squared <= outer_radius_squared) & (
        radius_squared > outer_radius_squared * spec.obscuration_ratio**2
    )
    area = math.pi * outer_radius_squared * (1 - spec.obscuration_ratio**2)
    field = aperture.astype(jnp.float64) / math.sqrt(area)
    diversity = jnp.where(aperture, spec.defocus_opd_m * (2 * radius_squared / outer_radius_squared - 1), 0.0)

    spacing = spec.actuator_spacing_m
    actuator_axis = (jnp.arange(spec.actuators_across, dtype=jnp.float64) - (spec.actuators_across - 1) / 2) * spacing
    actuator_x, actuator_y = jnp.meshgrid(actuator_axis, actuator_axis, indexing="xy")
    positions = jnp.stack((actuator_x.ravel() + registration * spacing, actuator_y.ravel()), axis=1)
    distance_squared = (x[None, :, :] - positions[:, 0, None, None]) ** 2 + (
        y[None, :, :] - positions[:, 1, None, None]
    ) ** 2
    sigma = spacing / math.sqrt(-2 * math.log(spec.coupling))
    cutoff_sigma = spacing / sigma * spec.cutoff_spacings
    floor = math.exp(-0.5 * cutoff_sigma**2)
    influence = jnp.where(
        distance_squared <= (cutoff_sigma * sigma) ** 2,
        jnp.exp(-0.5 * distance_squared / sigma**2) - floor,
        0.0,
    )
    return {
        "x_m": x,
        "y_m": y,
        "aperture": aperture,
        "diversity_opd_m": diversity,
        "input_field": field,
        "influence_functions": influence,
        "actuator_xy_m": positions,
        "pixel_scale_m": jnp.asarray(dx, dtype=jnp.float64),
    }


def _amplitude(
    command: jax.Array, gain: jax.Array, spec: StaticMirrorSpec, pupil_samples: int, registration_x_pitch: object = 0.0
) -> jax.Array:
    inputs = pupil_inputs(spec, pupil_samples, registration_x_pitch=registration_x_pitch)
    surface = jnp.tensordot(command, inputs["influence_functions"], axes=1)
    wavefront = Wavefront.from_phasor(
        inputs["input_field"] * inputs["pixel_scale_m"],
        spec.wavelength_m,
        diameter=spec.diameter_m,
    )
    wavefront = wavefront.add_opd(inputs["diversity_opd_m"] + 2 * gain * surface)
    return wavefront.propagate(
        spec.output_samples,
        spec.output_pixel_scale_m,
        focal_length=spec.focal_length_m,
    ).phasor


def pixel_amplitude(
    command_m: object, gain: object, *, spec: StaticMirrorSpec, pupil_samples: int, registration_x_pitch: object = 0.0
) -> jax.Array:
    """Return complex square-root pixel power on the fixed focal grid.

    For the same grid and gain this is ``1j * dy * E_HCIPy``. The fixed phase
    factor is a convention, not a fitted alignment. Commands are physical
    surface metres before the positive scalar command-response gain.
    """
    command, scalar_gain = _command_and_gain(command_m, gain, spec)
    return _amplitude(command, scalar_gain, spec, pupil_samples, registration_x_pitch)


def pixel_power(
    command_m: object, gain: object, *, spec: StaticMirrorSpec, pupil_samples: int, registration_x_pitch: object = 0.0
) -> jax.Array:
    """Expected pixel-power quadrature, without image normalization or noise."""
    amplitude = pixel_amplitude(
        command_m, gain, spec=spec, pupil_samples=pupil_samples, registration_x_pitch=registration_x_pitch
    )
    return jnp.abs(amplitude) ** 2


def directional_derivative(
    command_m: object,
    direction: object,
    gain: object,
    *,
    spec: StaticMirrorSpec,
    pupil_samples: int,
) -> jax.Array:
    """Return d(pixel_power(command_m + epsilon * direction))/d(epsilon).

    Direction entries are dimensionless and epsilon is measured in metres;
    the result has units of power per metre. Gain is held fixed. No numerical
    finite-difference step or automatic direction normalization is applied.
    """
    command, scalar_gain = _command_and_gain(command_m, gain, spec)
    tangent = _real_array(direction, "direction", (spec.actuator_count,))

    def power_at(value: jax.Array) -> jax.Array:
        return jnp.abs(_amplitude(value, scalar_gain, spec, pupil_samples)) ** 2

    return jax.jvp(power_at, (command,), (tangent,))[1]
