"""Small independent optical oracles; no atmosphere or study acquisition."""

from dataclasses import FrozenInstanceError, replace

import hcipy as hp
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("dLux")

from noether_optics.twin import static_mirror  # noqa: E402
from noether_optics.twin.static_mirror import (  # noqa: E402
    StaticMirrorSpec,
    directional_derivative,
    pixel_amplitude,
    pixel_power,
    pupil_inputs,
)


@pytest.fixture(autouse=True)
def float64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def spec():
    return StaticMirrorSpec(actuators_across=3, output_samples=9)


def _independent_pupil(spec, n):
    """Physical formulas in NumPy, without calling candidate pupil helpers."""
    dx = spec.diameter_m / n
    axis = (np.arange(n) + 0.5) * dx - spec.diameter_m / 2
    x, y = np.meshgrid(axis, axis, indexing="xy")
    r2 = x * x + y * y
    radius = spec.diameter_m / 2
    aperture = (r2 <= radius * radius) & (r2 > (radius * spec.obscuration_ratio) ** 2)
    area = np.pi * radius**2 * (1 - spec.obscuration_ratio**2)
    field = aperture / np.sqrt(area)
    opd = np.where(aperture, spec.defocus_opd_m * (2 * r2 / radius**2 - 1), 0)
    pitch = spec.diameter_m / spec.actuators_across
    positions = np.array(
        [
            ((column + 0.5) * pitch - spec.diameter_m / 2, (row + 0.5) * pitch - spec.diameter_m / 2)
            for row in range(spec.actuators_across)
            for column in range(spec.actuators_across)
        ]
    )
    basis = []
    for cx, cy in positions:
        distance2 = (x - cx) ** 2 + (y - cy) ** 2
        untruncated = spec.coupling ** (distance2 / pitch**2)
        influence = untruncated - spec.coupling ** (spec.cutoff_spacings**2)
        basis.append(np.where(distance2 <= (spec.cutoff_spacings * pitch) ** 2, influence, 0))
    return x, y, aperture, field, opd, np.asarray(basis), positions


def _direct_transform(spec, field):
    """Finite Fourier sum with physical area weights and fixed pixel amplitude."""
    n = field.shape[0]
    dx = spec.diameter_m / n
    pupil_axis = (np.arange(n) + 0.5) * dx - spec.diameter_m / 2
    dy = spec.wavelength_m * spec.focal_length_m / (spec.pixels_per_lambda_d * spec.diameter_m)
    focal_axis = (np.arange(spec.output_samples) - spec.output_samples // 2) * dy
    kernel = np.exp(-2j * np.pi * np.outer(focal_axis, pupil_axis) / (spec.wavelength_m * spec.focal_length_m))
    return (kernel @ field @ kernel.T) * (dx**2 * dy / (spec.wavelength_m * spec.focal_length_m))


def _reference(spec, n, command, gain):
    """HCIPy plant, constructing its own grid, aperture, influences and lens."""
    grid = hp.make_pupil_grid(n, spec.diameter_m)
    aperture = hp.make_obstructed_circular_aperture(spec.diameter_m, spec.obscuration_ratio)(grid)
    area = np.pi * (spec.diameter_m / 2) ** 2 * (1 - spec.obscuration_ratio**2)
    wavefront = hp.Wavefront(aperture / np.sqrt(area), spec.wavelength_m)
    diversity = spec.defocus_opd_m * (2 * (grid.x**2 + grid.y**2) / (spec.diameter_m / 2) ** 2 - 1)
    wavefront.electric_field *= np.exp(2j * np.pi * diversity / spec.wavelength_m)
    basis = hp.make_gaussian_influence_functions(
        grid,
        spec.actuators_across,
        spec.actuator_spacing_m,
        crosstalk=spec.coupling,
        cutoff=spec.cutoff_spacings,
    )
    mirror = hp.DeformableMirror(basis)
    mirror.actuators = np.asarray(command) * gain
    output_grid = hp.make_uniform_grid(
        spec.output_samples,
        [spec.output_samples * spec.output_pixel_scale_m] * 2,
    )
    output = hp.FraunhoferPropagator(grid, output_grid, focal_length=spec.focal_length_m).forward(
        mirror.forward(wavefront)
    )
    return grid, mirror.surface.shaped, output


def _asymmetric_command(spec):
    command = np.zeros(spec.actuator_count)
    command[[0, 4, 7]] = [7e-9, -13e-9, 19e-9]
    return command


def test_default_contract_is_frozen_and_contains_no_gain():
    contract = StaticMirrorSpec()
    assert contract.actuator_count == 25
    assert contract.actuator_spacing_m == pytest.approx(1.6)
    assert contract.output_pixel_scale_m == pytest.approx(551e-9 * 64 / 32)
    assert "gain" not in contract.__dataclass_fields__
    with pytest.raises(FrozenInstanceError):
        contract.diameter_m = 10


def test_physical_inputs_ordering_and_flux_are_not_grid_normalized(spec):
    n = 12
    actual = pupil_inputs(spec, n)
    x, y, aperture, field, diversity, basis, positions = _independent_pupil(spec, n)
    np.testing.assert_allclose(actual["x_m"], x, rtol=0, atol=2e-15)
    np.testing.assert_allclose(actual["y_m"], y, rtol=0, atol=2e-15)
    np.testing.assert_array_equal(actual["aperture"], aperture)
    np.testing.assert_allclose(actual["actuator_xy_m"], positions, rtol=0, atol=2e-15)
    np.testing.assert_allclose(actual["input_field"], field, rtol=2e-15, atol=0)
    np.testing.assert_allclose(actual["diversity_opd_m"], diversity, rtol=1e-13, atol=1e-23)
    np.testing.assert_allclose(actual["influence_functions"], basis, rtol=1e-12, atol=2e-15)
    incident_power = np.sum(np.asarray(actual["input_field"]) ** 2) * (spec.diameter_m / n) ** 2
    expected = np.count_nonzero(aperture) * (spec.diameter_m / n) ** 2
    expected /= np.pi * (spec.diameter_m / 2) ** 2 * (1 - spec.obscuration_ratio**2)
    assert incident_power == pytest.approx(expected, rel=2e-15)
    assert abs(incident_power - 1) > 1e-4
    assert actual["input_field"].dtype == np.float64


def test_finite_cutoff_subtracts_floor_without_peak_renormalization():
    contract = StaticMirrorSpec(diameter_m=6, actuators_across=3, cutoff_spacings=0.6, output_samples=9)
    inputs = pupil_inputs(contract, 9)
    central = np.asarray(inputs["influence_functions"])[4]
    distance2 = np.asarray(inputs["x_m"]) ** 2 + np.asarray(inputs["y_m"]) ** 2
    assert central[4, 4] == pytest.approx(1 - 0.45**0.36, rel=1e-14)
    assert central[4, 4] != pytest.approx(1)
    np.testing.assert_array_equal(central[distance2 > 1.2**2], 0)
    assert np.all(central >= 0)


def test_flat_wavefront_matches_independent_weighted_fourier_sum(spec):
    contract = replace(spec, defocus_opd_m=0)
    _, _, _, field, _, _, _ = _independent_pupil(contract, 12)
    expected = _direct_transform(contract, field)
    actual = pixel_amplitude(np.zeros(contract.actuator_count), 1, spec=contract, pupil_samples=12)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=2e-14)
    center = contract.output_samples // 2
    expected_center = field.sum() * (contract.diameter_m / 12) ** 2
    expected_center *= contract.output_pixel_scale_m / (contract.wavelength_m * contract.focal_length_m)
    assert actual[center, center] == pytest.approx(expected_center, abs=2e-14)


@pytest.mark.parametrize("n", [12, 15])
@pytest.mark.parametrize("gain", [1.0, 1.1])
def test_same_grid_and_same_gain_match_hcipy_with_fixed_phase(spec, n, gain):
    command = _asymmetric_command(spec)
    grid, surface, output = _reference(spec, n, command, gain)
    inputs = pupil_inputs(spec, n)
    np.testing.assert_allclose(inputs["x_m"], grid.x.shaped, rtol=0, atol=2e-15)
    np.testing.assert_allclose(inputs["y_m"], grid.y.shaped, rtol=0, atol=2e-15)
    candidate_surface = gain * np.tensordot(command, inputs["influence_functions"], axes=1)
    np.testing.assert_allclose(candidate_surface, surface, rtol=1e-11, atol=5e-18)
    amplitude = pixel_amplitude(command, gain, spec=spec, pupil_samples=n)
    expected_amplitude = 1j * spec.output_pixel_scale_m * output.electric_field.shaped
    np.testing.assert_allclose(amplitude, expected_amplitude, rtol=1e-10, atol=2e-14)
    power = pixel_power(command, gain, spec=spec, pupil_samples=n)
    assert np.sum(np.abs(np.asarray(power) - output.power.shaped)) <= 1e-10


def _analytic_basis(monkeypatch, spec, n, profile):
    """Replace only the surface basis with a known piston or tilt for an oracle."""
    inputs = pupil_inputs(spec, n)
    basis = np.zeros((spec.actuator_count, n, n))
    basis[0] = profile
    inputs["influence_functions"] = jax.numpy.asarray(basis)
    monkeypatch.setattr(static_mirror, "pupil_inputs", lambda spec, pupil_samples, **kwargs: inputs)


@pytest.mark.parametrize("shift", [-0.1, 0.0, 0.1])
def test_registered_centres_surface_and_power_match_independent_reference(shift):
    from noether_optics.bench.twin_intervention import reference_backend

    contract, n = StaticMirrorSpec(), 16
    command = _asymmetric_command(contract)
    baseline = pupil_inputs(contract, n)
    shifted = pupil_inputs(contract, n, registration_x_pitch=shift)
    expected_positions = np.asarray(baseline["actuator_xy_m"]).copy()
    expected_positions[:, 0] += shift * contract.actuator_spacing_m
    np.testing.assert_allclose(shifted["actuator_xy_m"], expected_positions, rtol=0, atol=2e-15)
    np.testing.assert_array_equal(shifted["aperture"], baseline["aperture"])
    np.testing.assert_array_equal(shifted["input_field"], baseline["input_field"])
    # Independent scalar continuous Gaussian formula, including the cutoff floor.
    x, y = np.asarray(baseline["x_m"]), np.asarray(baseline["y_m"])
    expected_surface = np.zeros_like(x)
    for amount, (cx, cy) in zip(command, expected_positions, strict=True):
        radius_pitch_squared = ((x - cx) ** 2 + (y - cy) ** 2) / contract.actuator_spacing_m**2
        influence = np.where(
            radius_pitch_squared <= 9, contract.coupling**radius_pitch_squared - contract.coupling**9, 0
        )
        expected_surface += amount * influence
    actual_surface = np.tensordot(command, shifted["influence_functions"], axes=1)
    reference = reference_backend(registration_x_pitch=shift)(n, command, 1.0, component=True)
    np.testing.assert_allclose(actual_surface, expected_surface, rtol=1e-11, atol=5e-18)
    np.testing.assert_allclose(reference["surface"], expected_surface, rtol=1e-11, atol=5e-18)
    power = pixel_power(command, 1.0, spec=contract, pupil_samples=n, registration_x_pitch=shift)
    assert np.sum(np.abs(np.asarray(power) - reference["power"])) <= 1e-10


def test_registration_default_zero_and_factory_caches_preserve_existing_physics():
    from noether_optics.bench.twin_intervention import reference_backend

    contract, command = StaticMirrorSpec(), np.zeros(25)
    command[7] = 10e-9
    default = pixel_power(command, 1.0, spec=contract, pupil_samples=16)
    zero = pixel_power(command, 1.0, spec=contract, pupil_samples=16, registration_x_pitch=0.0)
    np.testing.assert_array_equal(default, zero)
    default_ref, zero_ref = reference_backend(), reference_backend(registration_x_pitch=0.0)
    np.testing.assert_array_equal(default_ref(16, command, 1.0)["power"], zero_ref(16, command, 1.0)["power"])
    minus, plus = reference_backend(registration_x_pitch=-0.1), reference_backend(registration_x_pitch=0.1)
    first = minus(16, command, 1.0)["power"].copy()
    positive = plus(16, command, 1.0)["power"]
    minus(17, command, 1.0)
    np.testing.assert_array_equal(first, minus(16, command, 1.0)["power"])
    assert np.max(np.abs(first - positive)) > 1e-8


def test_dynamic_registration_derivative_matches_symmetric_physical_difference():
    contract, command = StaticMirrorSpec(), np.zeros(25)
    command[18] = 10e-9

    def power(shift):
        return pixel_power(command, 1.0, spec=contract, pupil_samples=16, registration_x_pitch=shift)

    center, step = 0.037, 1e-5
    derivative = np.asarray(jax.jacfwd(power)(center))
    difference = (np.asarray(power(center + step)) - np.asarray(power(center - step))) / (2 * step)
    assert np.linalg.norm(derivative) > 1e-6
    assert np.linalg.norm(derivative - difference) <= 2e-4 * np.linalg.norm(difference)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), [0.0]])
def test_invalid_registration_is_rejected_before_optical_propagation(bad):
    with pytest.raises(ValueError):
        pupil_inputs(StaticMirrorSpec(), 16, registration_x_pitch=bad)


def test_piston_has_exact_double_pass_phase_and_preserves_power(monkeypatch, spec):
    contract = replace(spec, defocus_opd_m=0)
    _analytic_basis(monkeypatch, contract, 12, np.ones((12, 12)))
    command = np.zeros(contract.actuator_count)
    flat = pixel_amplitude(command, 1.1, spec=contract, pupil_samples=12)
    command[0] = 17e-9
    amplitude = pixel_amplitude(command, 1.1, spec=contract, pupil_samples=12)
    phase = np.exp(4j * np.pi * 1.1 * 17e-9 / contract.wavelength_m)
    np.testing.assert_allclose(amplitude, np.asarray(flat) * phase, rtol=1e-11, atol=2e-14)
    np.testing.assert_allclose(np.abs(amplitude) ** 2, np.abs(flat) ** 2, rtol=1e-11, atol=2e-14)


@pytest.mark.parametrize("sign", [-1, 1])
def test_physical_x_tilt_moves_focal_peak_along_x_with_correct_sign(monkeypatch, spec, sign):
    contract = replace(spec, defocus_opd_m=0)
    x, _, _, field, _, _, _ = _independent_pupil(contract, 12)
    radius = contract.diameter_m / 2
    _analytic_basis(monkeypatch, contract, 12, x / radius)
    command = np.zeros(contract.actuator_count)
    command[0] = sign * radius * contract.output_pixel_scale_m / (2 * 1.1 * contract.focal_length_m)
    amplitude = pixel_amplitude(command, 1.1, spec=contract, pupil_samples=12)
    opd = 2 * 1.1 * command[0] * x / radius
    expected = _direct_transform(contract, field * np.exp(2j * np.pi * opd / contract.wavelength_m))
    np.testing.assert_allclose(amplitude, expected, rtol=1e-11, atol=2e-14)
    center = contract.output_samples // 2
    assert np.unravel_index(np.argmax(np.abs(amplitude) ** 2), amplitude.shape) == (center, center + sign)


def test_command_jvp_and_scalar_gain_gradient_have_independent_analytic_oracles(spec):
    n, gain = 12, 1.1
    command = _asymmetric_command(spec)
    direction = np.zeros(spec.actuator_count)
    direction[[1, 5]] = [1, -1]
    _, _, _, field, diversity, basis, _ = _independent_pupil(spec, n)
    surface = np.tensordot(command, basis, axes=1)
    tangent_surface = np.tensordot(direction, basis, axes=1)
    field = field * np.exp(2j * np.pi * (diversity + 2 * gain * surface) / spec.wavelength_m)
    amplitude = _direct_transform(spec, field)
    tangent_amplitude = _direct_transform(spec, field * (4j * np.pi * gain / spec.wavelength_m) * tangent_surface)
    expected = 2 * np.real(np.conjugate(amplitude) * tangent_amplitude)
    actual = directional_derivative(command, direction, gain, spec=spec, pupil_samples=n)
    np.testing.assert_allclose(np.asarray(actual) * 1e-9, expected * 1e-9, rtol=1e-9, atol=1e-13)

    gain_amplitude = _direct_transform(spec, field * (4j * np.pi / spec.wavelength_m) * surface)
    gain_pixels = 2 * np.real(np.conjugate(amplitude) * gain_amplitude)
    weights = np.arange(1, spec.output_samples**2 + 1).reshape(spec.output_samples, spec.output_samples)

    def weighted_power(value):
        return jax.numpy.sum(pixel_power(command, value, spec=spec, pupil_samples=n) * weights)

    derivative = jax.grad(weighted_power)(gain)
    assert derivative == pytest.approx(np.sum(weights * gain_pixels), rel=1e-9, abs=2e-12)


def test_command_jvp_matches_hcipy_central_differences(spec):
    command = _asymmetric_command(spec)
    direction = np.zeros(spec.actuator_count)
    direction[[1, 5]] = [1, -1]
    actual = np.asarray(directional_derivative(command, direction, 1.1, spec=spec, pupil_samples=12)) * 1e-9
    for step_nm in (0.01, 0.1):
        step = step_nm * 1e-9
        _, _, plus = _reference(spec, 12, command + step * direction, 1.1)
        _, _, minus = _reference(spec, 12, command - step * direction, 1.1)
        expected = (plus.power.shaped - minus.power.shaped) / (2 * step) * 1e-9
        absolute = np.sum(np.abs(actual - expected))
        relative = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        assert absolute <= 1e-10 or relative <= 1e-5


def test_jit_accepts_traced_commands_gains_and_directions(spec):
    command = _asymmetric_command(spec)
    direction = np.eye(spec.actuator_count)[1]
    compiled = jax.jit(lambda u, g: pixel_power(u, g, spec=spec, pupil_samples=12))
    compiled_jvp = jax.jit(lambda u, v, g: directional_derivative(u, v, g, spec=spec, pupil_samples=12))
    np.testing.assert_allclose(
        compiled(command, 1.1), pixel_power(command, 1.1, spec=spec, pupil_samples=12), rtol=1e-11, atol=2e-14
    )
    np.testing.assert_allclose(
        compiled_jvp(command, direction, 1.1),
        directional_derivative(command, direction, 1.1, spec=spec, pupil_samples=12),
        rtol=1e-9,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"diameter_m": 0},
        {"wavelength_m": float("nan")},
        {"focal_length_m": float("inf")},
        {"obscuration_ratio": 1},
        {"obscuration_ratio": -0.1},
        {"coupling": 0},
        {"coupling": 1},
        {"cutoff_spacings": -1},
        {"pixels_per_lambda_d": False},
        {"actuators_across": 3.5},
        {"actuators_across": True},
        {"output_samples": 8},
        {"defocus_opd_m": 1j},
    ],
)
def test_invalid_physical_contract_is_rejected(changes):
    with pytest.raises(ValueError):
        StaticMirrorSpec(**changes)


@pytest.mark.parametrize("n", [0, 1, 3.5, True])
def test_invalid_pupil_sampling_is_rejected(spec, n):
    with pytest.raises(ValueError, match="pupil_samples"):
        pupil_inputs(spec, n)


@pytest.mark.parametrize("gain", [0, -1, float("nan"), float("inf"), 1j, True, [1]])
def test_invalid_concrete_gain_is_rejected(spec, gain):
    with pytest.raises(ValueError, match="gain"):
        pixel_power(np.zeros(spec.actuator_count), gain, spec=spec, pupil_samples=12)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1j])
def test_nonfinite_or_complex_command_and_direction_are_rejected(spec, value):
    invalid = np.full(spec.actuator_count, value)
    with pytest.raises(ValueError, match="command_m"):
        pixel_power(invalid, 1, spec=spec, pupil_samples=12)
    with pytest.raises(ValueError, match="direction"):
        directional_derivative(np.zeros(spec.actuator_count), invalid, 1, spec=spec, pupil_samples=12)


def test_command_and_direction_dimensions_are_checked_before_propagation(spec):
    with pytest.raises(ValueError, match="command_m must have shape"):
        pixel_power(np.zeros((3, 3)), 1, spec=spec, pupil_samples=12)
    with pytest.raises(ValueError, match="direction must have shape"):
        directional_derivative(np.zeros(9), np.zeros(8), 1, spec=spec, pupil_samples=12)


def test_float32_mode_is_refused_instead_of_silently_downcasting(spec):
    jax.config.update("jax_enable_x64", False)
    with pytest.raises(RuntimeError, match="float64"):
        pupil_inputs(spec, 12)
    with pytest.raises(RuntimeError, match="float64"):
        pixel_power(np.zeros(9), 1, spec=spec, pupil_samples=12)


# The study/packet checks below use analytic arrays or injected callbacks. They
# establish contracts and failure behavior, not results from the optical study.
from noether_optics.bench import twin_intervention as study  # noqa: E402


def test_reference_and_derivative_ledgers_match_declared_budget():
    ledger = study.reference_ledger()
    assert len(ledger) == len({item["id"] for item in ledger}) == 378
    from collections import Counter

    assert Counter(row["purpose"] for row in ledger) == {
        "primary": 182,
        "derivative": 144,
        "component": 4,
        "component-derivative": 24,
        "convergence": 24,
    }
    assert len(study.calibration_commands()) == 18
    assert len(study.evaluation_commands()) == 72
    anchors = study.derivative_anchors()
    assert len(anchors) == 12
    assert len(anchors) * 2 * 3 * 2 + 4 == 148
    commands = dict(study.evaluation_commands())
    for name, command, direction in anchors:
        np.testing.assert_array_equal(command, commands[name])
        assert name.endswith("-tpos")
        assert set(np.flatnonzero(direction)).issubset(study.ACTIVE)
        assert np.sum(direction) == 0
    baseline = dict(study.calibration_commands())
    assert set(baseline).isdisjoint(commands)
    for row in ledger:
        assert np.asarray(row["command_m"]).shape == (25,)
        assert row["grid"] == (512 if row["purpose"] == "convergence" else 256)


def test_discrete_core_contains_exact_boundary_without_extra_area():
    axis = study.focal_axis()
    assert study.core_mask().sum() == 49
    assert study.core_power(np.ones((65, 65), dtype=np.float64), axis, axis) == 49
    inside, outside = np.zeros((65, 65)), np.zeros((65, 65))
    inside[32, 36], outside[33, 36] = 1, 1
    assert study.core_power(inside, axis, axis) == 1
    assert study.core_power(outside, axis, axis) == 0
    with pytest.raises(ValueError, match="coordinates"):
        study.core_power(inside, axis + 1e-20, axis)
    with pytest.raises(ValueError, match="coordinates"):
        study.core_power(inside, axis, axis[::-1])
    with pytest.raises(ValueError, match="float64"):
        study.core_power(inside.astype(np.float32), axis, axis)
    inside[0, 0] = -1e-30
    with pytest.raises(ValueError, match="nonnegative"):
        study.core_power(inside, axis, axis)


def test_predicted_near_tie_does_not_erase_reference_regret():
    names = sorted(name for name, _ in study.evaluation_commands())
    predicted = dict.fromkeys(names, 0.0)
    predicted[names[0]], predicted[names[1]] = 1.0 - 0.5e-10, 1.0
    choice = study.choose_action(predicted)
    assert choice["chosen_id"] == names[0]
    assert choice["predicted_ties"] == names[:2]
    reference = dict.fromkeys(names, 0.0)
    reference[names[0]], reference[names[1]] = 0.7, 0.7 + 0.25e-10
    regret = study.action_regret(choice, reference)
    assert regret["regret"] == reference[names[1]] - reference[names[0]]
    assert regret["regret"] > 0
    del predicted[names[-1]]
    with pytest.raises(ValueError, match="72"):
        study.choose_action(predicted)


def test_zero_response_keeps_absolute_errors_and_undefined_relative_metric():
    for scale in (0, 0.5e-12, 1e-12):
        result = study.error_metrics(np.array([1.0]), np.array([scale]))
        assert result["relative_l2"] is None
        assert result["absolute_l1"] > 0
    assert study.error_metrics(np.array([1.0]), np.array([1.1e-12]))["relative_l2"] is not None


def test_scalar_search_is_bounded_and_preserves_local_and_boundary_status():
    seen = []

    def objective(gain, sequence):
        assert sequence == len(seen)
        seen.append(gain)
        return (gain - 1.071234) ** 2

    fitted = study.fit_scalar_gain(objective)
    assert fitted["status"] == "converged_local"
    assert fitted["claim"] == "local_search_only"
    assert fitted["gain"] == pytest.approx(1.071234, abs=1e-6)
    assert 81 < len(seen) <= 121
    assert fitted["bracket_width"] <= 1e-6
    assert all(0.8 <= gain <= 1.2 for gain in seen)
    endpoint = study.fit_scalar_gain(lambda gain, sequence: gain)
    assert endpoint["gain"] == 0.8 and endpoint["boundary"]
    assert endpoint["status"] == "converged_local"


def test_exact_multiple_calibration_minima_leave_no_substitute_gain():
    fitted = study.fit_scalar_gain(lambda gain, sequence: 3.0)
    assert fitted["status"] == "ambiguous_grid_minimum"
    assert fitted["gain"] is None
    assert len(fitted["trace"]) == len(fitted["grid_minima"]) == 81
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        study.fit_scalar_gain(lambda gain, sequence: float("nan"))


def test_refinement_exhaustion_cannot_be_reported_as_a_fitted_gain(monkeypatch):
    # Inject a broken/no-progress contraction, independently of real objective
    # noise, to establish the finite hard limit and explicit unavailable outcome.
    monkeypatch.setattr(study.math, "sqrt", lambda value: 3.0)
    fitted = study.fit_scalar_gain(lambda gain, sequence: (gain - 1.001) ** 2)
    assert fitted["status"] == "refinement_exhausted"
    assert fitted["gain"] is None
    assert len(fitted["trace"]) == 121


@pytest.fixture
def packet(tmp_path, monkeypatch):
    # Avoid using test-process RSS/free space as a scientific admission receipt.
    monkeypatch.setattr(study.EvidencePacket, "checkpoint", lambda self, *args, **kwargs: None)
    value = study.EvidencePacket(
        tmp_path / "packet",
        study.protocol(source={"fixture": "synthetic"}),
        {"fixture": "no optical runtime"},
        {"device": tmp_path.stat().st_dev},
    )
    try:
        yield value
    finally:
        value.close()


def test_terminal_records_are_immutable_and_resumed_without_reexecution(packet):
    calls = []

    def callback():
        calls.append(1)
        return {"power": np.array([[0.5, -0.0]], dtype=np.float64)}

    actual = packet.perform("forward", "fixture", callback, inputs={"gain": 1})
    np.testing.assert_array_equal(actual["power"], [[0.5, -0.0]])
    second = packet.perform("forward", "fixture", callback, inputs={"gain": 1})
    assert len(calls) == packet.counts["forward"] == 1
    assert np.signbit(second["power"][0, 1])  # original signed float64 preserved
    with pytest.raises(ValueError, match="changed inputs|input identity"):
        packet.perform("forward", "fixture", callback, inputs={"gain": 2})
    before = (packet.root / "records/fixture.json").read_bytes()
    with pytest.raises(FileExistsError):
        study._publish_json(packet.root / "records/fixture.json", {"changed": True})
    assert (packet.root / "records/fixture.json").read_bytes() == before


def test_nonfinite_raw_output_is_retained_and_terminal(packet):
    values = np.array([float("nan"), float("inf"), -0.0], dtype=np.float64)
    with pytest.raises(RuntimeError, match="numerical failure"):
        packet.perform("jvp", "bad-jvp", lambda: {"jvp": values}, inputs={})
    record, arrays = packet.load_record("bad-jvp")
    assert record["status"] == "nonfinite"
    assert np.isnan(arrays["jvp"][0]) and np.isinf(arrays["jvp"][1])
    assert np.signbit(arrays["jvp"][2])
    with pytest.raises(RuntimeError, match="do not retry"):
        packet.perform("jvp", "bad-jvp", lambda: {"jvp": np.zeros(3)}, inputs={})
    assert packet.counts["jvp"] == 1


def test_raw_artifact_corruption_is_refused(packet):
    packet.perform("forward", "array", lambda: {"power": np.ones((2, 2))}, inputs={})
    path = packet.root / packet.load_record("array")[0]["array_file"]
    with path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        packet.load_record("array")


def test_interrupted_dispatch_consumes_budget_and_resumes_only_nonterminal_work(packet):
    class Interrupted(BaseException):
        pass

    def interrupted():
        raise Interrupted()

    with pytest.raises(Interrupted):
        packet.perform("reference", "interrupted", interrupted, inputs={"grid": 256})
    assert packet.counts["reference"] == 1
    assert not (packet.root / "records/interrupted.json").exists()
    packet.perform("reference", "interrupted", lambda: {"power": np.ones((1, 1))}, inputs={"grid": 256})
    assert packet.counts["reference"] == 2
    assert packet.load_record("interrupted")[0]["attempt"] == 1


def test_call_limit_is_checked_before_callback_dispatch(packet):
    packet.counts["reference"] = study.LIMITS["reference"]
    calls = []
    with pytest.raises(RuntimeError, match="budget"):
        packet.perform("reference", "over-budget", lambda: calls.append(1), inputs={})
    assert calls == []
    assert list((packet.root / "attempts").glob("*.json")) == []


def test_frozen_receipts_cannot_change_after_more_work(packet):
    initial = packet.freeze("models", "model", {"gain": 1.0})
    packet.perform("forward", "later", lambda: {"loss": np.array(1.0)}, inputs={}, scalar_only=True)
    assert packet.freeze("models", "model", {"gain": 1.0}) == initial
    with pytest.raises(ValueError, match="Frozen evidence"):
        packet.freeze("models", "model", {"gain": 1.1})
    assert study._read_json(packet.root / "models/model.json")["frozen_after_attempt"] == -1


def test_interrupted_array_publication_recovers_original_output_without_new_callback(packet, monkeypatch):
    # Publication uses packet.write_json; fail exactly at the terminal record,
    # after its raw archive exists and after a durable dispatch was retained.
    writer = packet.write_json

    def stop(path, payload, **kwargs):
        if path.parent.name == "records":
            raise KeyboardInterrupt()
        return writer(path, payload, **kwargs)

    monkeypatch.setattr(packet, "write_json", stop)
    first = {"power": np.ones((2, 2)), "compile_seconds": np.array(2.75)}
    with pytest.raises(KeyboardInterrupt):
        packet.perform("forward", "orphan", lambda: first, inputs={})
    raw_path = packet.root / "arrays/orphan-a000000.npz"
    raw_hash = study.file_hash(raw_path)
    monkeypatch.setattr(packet, "write_json", writer)
    packet.close()
    resumed = study.EvidencePacket(
        packet.root, packet.contract, packet.header["runtime"], packet.header["admission"], resume=True
    )
    calls = []

    def later_process():
        calls.append(1)
        return {"power": np.ones((2, 2)), "compile_seconds": np.array(0.125)}

    try:
        result = resumed.perform("forward", "orphan", later_process, inputs={})
        assert calls == []
        assert float(result["compile_seconds"]) == 2.75
        assert study.file_hash(raw_path) == raw_hash
        assert resumed.counts["forward"] == 1
        assert resumed.load_record("orphan")[0]["attempt"] == 0
    finally:
        resumed.close()


def test_descriptor_without_published_array_cannot_claim_callback_output(packet, monkeypatch):
    publish = study._publish

    def interrupt(path, payload):
        if path.parent.name == "arrays":
            raise KeyboardInterrupt()
        return publish(path, payload)

    monkeypatch.setattr(study, "_publish", interrupt)
    with pytest.raises(KeyboardInterrupt):
        packet.perform("forward", "incomplete", lambda: {"power": np.ones((2, 2))}, inputs={})
    old_descriptor = (packet.root / "prepared/000000.json").read_bytes()
    monkeypatch.setattr(study, "_publish", publish)
    calls = []

    def retry():
        calls.append(1)
        return {"power": np.full((2, 2), 0.5)}

    packet.perform("forward", "incomplete", retry, inputs={})
    assert calls == [1] and packet.counts["forward"] == 2
    record, arrays = packet.load_record("incomplete")
    assert record["attempt"] == 1 and record["array_file"].endswith("-a000001.npz")
    assert np.all(arrays["power"] == 0.5)
    assert (packet.root / "prepared/000000.json").read_bytes() == old_descriptor
    packet._audit_journal()


def test_interrupted_nonfinite_raw_publication_cannot_be_retried_as_finite(packet, monkeypatch):
    publish = study._publish
    first = next(
        row for row in packet.contract["reference_ledger"] if row["purpose"] in ("component", "component-derivative")
    )

    def interrupt(path, payload):
        if path.parent.name == "arrays":
            raise KeyboardInterrupt()
        return publish(path, payload)

    monkeypatch.setattr(study, "_publish", interrupt)
    with pytest.raises(KeyboardInterrupt):
        packet.perform("reference", first["id"], lambda: {"power": np.array([float("nan")])}, inputs=first)
    descriptor = (packet.root / "prepared/000000.json").read_bytes()
    assert study._read_json(packet.root / "prepared/000000.json")["status"] == "nonfinite"
    assert not list((packet.root / "arrays").glob("*.npz"))
    packet.close()
    monkeypatch.setattr(study, "_publish", publish)
    resumed = study.EvidencePacket(
        packet.root, packet.contract, packet.header["runtime"], packet.header["admission"], resume=True
    )
    calls = []

    def finite_retry(*args, **kwargs):
        calls.append(1)
        return {"power": np.ones((65, 65))}

    try:
        with pytest.raises(RuntimeError, match="prepared numerical failure"):
            resumed.perform("reference", first["id"], finite_retry, inputs=first)
        with pytest.raises(RuntimeError, match="prepared numerical failure"):
            study.acquire(resumed, reference=finite_retry, candidate=finite_retry)
        assert calls == [] and resumed.counts["reference"] == 1
        assert len(list((resumed.root / "attempts").glob("*.json"))) == 1
        assert (resumed.root / "prepared/000000.json").read_bytes() == descriptor
        assert not (resumed.root / "records" / (first["id"] + ".json")).exists()
        assert not (resumed.root / "manifest.json").exists()
    finally:
        resumed.close()


def test_finite_later_terminal_cannot_erase_an_earlier_prepared_failure(packet, monkeypatch):
    first = next(
        row for row in packet.contract["reference_ledger"] if row["purpose"] in ("component", "component-derivative")
    )
    publish = study._publish

    def interrupt(path, payload):
        if path.parent.name == "arrays":
            raise KeyboardInterrupt()
        return publish(path, payload)

    monkeypatch.setattr(study, "_publish", interrupt)
    with pytest.raises(KeyboardInterrupt):
        packet.perform("reference", first["id"], lambda: {"power": np.array([float("nan")])}, inputs=first)
    monkeypatch.setattr(study, "_publish", publish)
    # Forensic fixture for the old bug: keep the immutable negative descriptor
    # on disk but simulate a writer that forgot it before accepting a retry.
    # The fresh audit below must discover both attempts from disk, regardless
    # of this temporary in-memory omission or the finite terminal result.
    packet.prepared_by_id = {}
    finite = {
        "power": np.full((65, 65), 1 / 65**2),
        "field": np.ones((65, 65), dtype=np.complex128),
        "incident": np.array(1.0),
        "surface": np.zeros((256, 256)),
        "pupil_x": np.zeros((256, 256)),
        "pupil_y": np.zeros((256, 256)),
        "aperture": np.ones((256, 256), dtype=bool),
    }
    packet.perform("reference", first["id"], lambda: finite, inputs=first)
    assert packet.load_record(first["id"])[0]["status"] == "complete"
    assert study._read_json(packet.root / "prepared/000000.json")["status"] == "nonfinite"
    packet.close()
    resumed = study.EvidencePacket(
        packet.root, packet.contract, packet.header["runtime"], packet.header["admission"], resume=True
    )
    try:
        with pytest.raises(RuntimeError, match="prepared numerical failure"):
            resumed.perform("reference", first["id"], lambda: pytest.fail("must not dispatch"), inputs=first)
        with pytest.raises(RuntimeError, match="prepared numerical failure"):
            study.scientific_preflight(resumed)
        with pytest.raises(RuntimeError, match="prepared numerical failure"):
            study.finalize(resumed)
        assert resumed.counts["reference"] == 2
        assert not (resumed.root / "manifest.json").exists()
    finally:
        resumed.close()


@pytest.mark.parametrize(
    "corruption", ["undeclared_record", "foreign_array", "changed_later_array", "changed_descriptor"]
)
def test_resume_rejects_every_unbound_or_corrupt_artifact_before_callbacks(packet, corruption):
    packet.perform("forward", "retained", lambda: {"power": np.ones((2, 2))}, inputs={})
    if corruption == "undeclared_record":
        study._publish_json(packet.root / "records/undeclared.json", {"id": "undeclared"})
    elif corruption == "foreign_array":
        study._publish(packet.root / "arrays/foreign.npz", study._array_bytes({"power": np.ones((2, 2))}))
    elif corruption == "changed_later_array":
        with (packet.root / packet.load_record("retained")[0]["array_file"]).open("ab") as stream:
            stream.write(b"corrupt")
    else:
        path = packet.root / "prepared/000000.json"
        record = study._read_json(path)
        record["inputs"] = {"changed": True}
        path.write_bytes(study.canonical(record))
    packet.close()
    before = len(list((packet.root / "attempts").glob("*.json")))
    calls = []
    with pytest.raises(ValueError):
        resumed = study.EvidencePacket(
            packet.root, packet.contract, packet.header["runtime"], packet.header["admission"], resume=True
        )
        try:
            resumed.perform("forward", "next", lambda: calls.append(1), inputs={})
        finally:
            resumed.close()
    assert calls == []
    assert len(list((packet.root / "attempts").glob("*.json"))) == before


def test_interrupted_heldout_dispatch_before_freezes_rejects_acquisition_without_backend_calls(packet):
    heldout = next(row for row in packet.contract["reference_ledger"] if row["command_id"].startswith("eval-"))

    def interrupted():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        packet.perform("reference", heldout["id"], interrupted, inputs=heldout)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        pytest.fail("backend called after premature held-out dispatch")

    with pytest.raises(ValueError, match="acquisition prefix.*held-out"):
        study.acquire(packet, reference=forbidden, candidate=forbidden)
    assert calls == [] and packet.counts["reference"] == 1


def test_scientific_preflight_rejects_wrong_inputs_on_an_interrupted_declared_call(packet):
    first = next(
        row for row in packet.contract["reference_ledger"] if row["purpose"] in ("component", "component-derivative")
    )
    changed = {**first, "gain": 1.123}

    def interrupted():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        packet.perform("reference", first["id"], interrupted, inputs=changed)
    with pytest.raises(ValueError, match="order or inputs"):
        study.scientific_preflight(packet)


def test_read_only_packet_reader_rejects_unsafe_paths_and_changed_protocol(packet, tmp_path):
    with pytest.raises(ValueError, match="identity"):
        packet.perform("forward", "../escape", lambda: {}, inputs={})
    outside = tmp_path / "outside"
    outside.write_bytes(b"must not be read")
    (packet.root / "arrays/unsafe.npz").symlink_to(outside)
    with pytest.raises(ValueError, match="Symlinks"):
        study.EvidencePacket.reader(packet.root)
    (packet.root / "arrays/unsafe.npz").unlink()
    path = packet.root / "protocol.json"
    value = study._read_json(path)
    value["contract"]["decision"]["centres"] = 48
    path.write_bytes(study.canonical(value))
    with pytest.raises(ValueError, match="Protocol hash"):
        study.EvidencePacket.reader(packet.root)


def test_incomplete_packet_cannot_publish_a_complete_manifest(packet):
    packet.perform("forward", "unrelated", lambda: {"power": np.ones((1, 1))}, inputs={})
    with pytest.raises(ValueError, match="acquisition prefix"):
        study.finalize(packet)
    assert not (packet.root / "manifest.json").exists()


def test_calibration_adapter_does_not_pass_truth_or_heldout_data(packet, monkeypatch):
    for name, _ in study.calibration_commands():
        packet.perform("reference", f"ref-case-a-{name}", lambda: {"power": np.full((2, 2), 0.25)}, inputs={})
    seen = []

    def backend(kind, resolution, command, model_gain):
        seen.append((kind, resolution, command.copy(), model_gain))
        return {"power": np.full((2, 2), model_gain), "compile_seconds": np.array(0.0)}

    def short_search(objective):
        # Wiring-only fixture: the real bounded search has separate tests.
        assert objective(0.8, 0) > 0
        assert objective(1.2, 1) > 0
        return {"status": "ambiguous_grid_minimum", "gain": None, "trace": []}

    monkeypatch.setattr(study, "fit_scalar_gain", short_search)
    fitted, receipt = study._fit_arm(packet, backend, "case-a", 64)
    assert fitted["gain"] is None and receipt
    assert len(seen) == 36
    assert {value[3] for value in seen} == {0.8, 1.2}
    assert all(kind == "forward" and n == 64 and command.shape == (25,) for kind, n, command, _ in seen)
    assert set(fitted["calibration_records"]) == {f"ref-case-a-{name}" for name, _ in study.calibration_commands()}


def test_runtime_identity_change_refuses_resume_before_new_dispatch(packet):
    packet.close()
    with pytest.raises(ValueError, match="runtime identity"):
        study.EvidencePacket(
            packet.root, packet.contract, {"fixture": "different runtime"}, packet.header["admission"], resume=True
        )
    assert list((packet.root / "attempts").glob("*.json")) == []


def test_unmodified_resume_recovers_consumed_dispatches(packet):
    packet.perform("forward", "retained", lambda: {"loss": np.array(3.0)}, inputs={}, scalar_only=True)
    packet.close()
    resumed = study.EvidencePacket(
        packet.root, packet.contract, packet.header["runtime"], packet.header["admission"], resume=True
    )
    try:
        assert resumed.counts == {"reference": 0, "forward": 1, "jvp": 0}
        values = resumed.perform(
            "forward", "retained", lambda: pytest.fail("must not execute"), inputs={}, scalar_only=True
        )
        assert float(values["loss"]) == 3.0
    finally:
        resumed.close()


def test_resource_gate_checks_peak_rss_and_backing_volume_identity(tmp_path, monkeypatch):
    from types import SimpleNamespace

    writer = study.EvidencePacket.__new__(study.EvidencePacket)
    writer.root, writer.complete, writer.packet_bytes = tmp_path, False, 0
    device = tmp_path.stat().st_dev
    writer.header = {"admission": {"device": device}, "started_unix": 100.0}
    writer.storage_roots = ((tmp_path, device),)
    monkeypatch.setattr(study.time, "time", lambda: 101.0)
    monkeypatch.setattr(study.os, "statvfs", lambda p: SimpleNamespace(f_bavail=10 * 1024**3, f_frsize=1))
    rss_scale = 1 if study.sys.platform == "darwin" else 1024
    monkeypatch.setattr(study.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=2 * 1024**3 / rss_scale))
    with pytest.raises(RuntimeError, match="RSS"):
        writer.checkpoint()
    # A produced terminal record can still be preserved after the RSS threshold,
    # but never after losing its real storage identity.
    writer.checkpoint(1024, terminal_publication=True)
    writer.storage_roots = ((tmp_path, device + 1),)
    with pytest.raises(RuntimeError, match="filesystem identity"):
        writer.checkpoint(1024, terminal_publication=True)


@pytest.mark.parametrize("ambiguous", [False, True])
def test_full_injected_pipeline_retains_all_arms_and_reconstructs_from_raw(packet, monkeypatch, ambiguous):
    """Full protocol/cardinality fixture with synthetic optics, never scientific evidence."""
    monkeypatch.setattr(study.os, "fsync", lambda fd: None)  # crash tests above cover publication boundaries
    calls = {"reference": 0, "forward": 0, "jvp": 0}
    scale = study.focal_axis()[33]

    def power(command, gain):
        gain = 1.0 if ambiguous else gain
        return np.full((65, 65), (0.2 + 0.01 * gain * np.dot(command, command) / 1e-16) / 65**2)

    def geometry():
        return {
            "surface": np.zeros((256, 256)),
            "pupil_x": np.zeros((256, 256)),
            "pupil_y": np.zeros((256, 256)),
            "aperture": np.ones((256, 256), dtype=bool),
        }

    def reference(n, command, true_gain, *, component=False):
        calls["reference"] += 1
        image = power(command, true_gain)
        result = {
            "power": image,
            "field": np.asarray(-1j * np.sqrt(image) / scale, dtype=np.complex128),
            "incident": np.array(1.0),
        }
        if component:
            result.update(geometry())
        return result

    def candidate(kind, n, command, model_gain, *, direction=None, component=False):
        calls[kind] += 1
        if kind == "jvp":
            model_gain = 1.0 if ambiguous else model_gain
            derivative = 0.02 * model_gain * np.dot(command, direction) / (1e-16 * 65**2)
            return {"jvp": np.full((65, 65), derivative), "compile_seconds": np.array(0.0)}
        image = power(command, model_gain)
        result = {
            "power": image,
            "amplitude": np.sqrt(image).astype(np.complex128),
            "incident": np.array(1.0),
            "compile_seconds": np.array(0.0),
        }
        if component:
            result.update(geometry())
        return result

    result = study.acquire(packet, reference=reference, candidate=candidate)
    assert result["status"] == ("complete_with_unavailable_arms" if ambiguous else "complete_success")
    assert len(result["arms"]) == 12 and result["unavailable_arms"] == (6 if ambiguous else 0)
    assert calls["reference"] == 378 and calls["jvp"] == (76 if ambiguous else 148)
    assert all(row["status"] == "available" for row in result["arms"] if row["arm"] == "nominal")
    assert calls["forward"] <= 14056
    assert result["counts"] == calls
    before = calls.copy()
    assert study.verify_packet(packet.root) == result
    assert calls == before  # report reconstruction never calls either optical implementation
    path = packet.root / "analysis.json"
    analysis = study._read_json(path)
    analysis["arms"][0]["decision"]["regret"] += 1.0
    path.write_bytes(study.canonical(analysis))
    manifest_path = packet.root / "manifest.json"
    manifest = study._read_json(manifest_path)
    manifest["files"]["analysis.json"] = study.file_hash(path)
    manifest_path.write_bytes(study.canonical(manifest))
    with pytest.raises(ValueError, match="Analysis does not reproduce"):
        study.verify_packet(packet.root)
