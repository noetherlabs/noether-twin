"""Physical feedback conventions and causal predictive-control regressions."""

from dataclasses import replace
from types import SimpleNamespace

import hcipy as hp
import numpy as np
import pytest

from noether_optics.bench.metrics import residual_rms
from noether_optics.bench.refsim.calibration import build_reconstructor, measure_poke_matrix
from noether_optics.bench.refsim.config import LoopConfig, petscale_config
from noether_optics.bench.refsim.run_loop import run_closed_loop
from noether_optics.bench.refsim.system import RefSimSystem
from noether_optics.control.classical.eof import EOFController
from noether_optics.control.classical.idealized import IdealizedController
from noether_optics.control.classical.integrator import IntegratorController


@pytest.fixture
def static_system(monkeypatch):
    """Real pupil, DM and geometric sensor, with an analytic static OPD screen."""
    cfg = petscale_config()
    cfg = replace(
        cfg,
        wfs=replace(cfg.wfs, num_subapertures=4, pixels_per_subaperture=8),
        dm=replace(cfg.dm, num_actuators_across=5),
    )
    opd = np.zeros(32 * 32)

    class Atmosphere:
        t = 0.0
        scintillation = False

        def phase_for(self, wavelength):
            if self.scintillation:
                raise ValueError("Cannot get unwrapped phase with scintillation")
            return hp.Field(2 * np.pi * opd / wavelength, system.pupil_grid)

        def forward(self, wavefront):
            result = wavefront.copy()
            result.electric_field *= np.exp(1j * self.phase_for(wavefront.wavelength))
            return result

    def atmosphere(system, seed=None):
        system.atmosphere = Atmosphere()

    monkeypatch.setattr(RefSimSystem, "_build_atmosphere", atmosphere)
    monkeypatch.setattr(RefSimSystem, "_build_science_path", lambda system: None)
    system = RefSimSystem(cfg, seed=17)
    system._slope_noise_std = 0.0
    return system, opd


def test_physical_retained_mode_contracts_under_integrator(static_system):
    system, opd = static_system
    recon = build_reconstructor(measure_poke_matrix(system, poke_amplitude=1e-9), 0.5)
    system.dm.actuators = 1e-9 * recon.B[:, 0]
    opd[:] = system.dm.opd
    system.dm.actuators = np.zeros(system.n_actuators)
    baseline = residual_rms(system.residual_phase(), system.aperture_mask)
    controller = IntegratorController(recon, gain=0.3)
    controller.reset(system.n_actuators)
    for step in range(1, 7):
        system.dm.actuators = controller.compute(system.measure_wfs())
        measured = residual_rms(system.residual_phase(), system.aperture_mask)
        assert measured / baseline == pytest.approx(0.7**step, rel=1e-10)


def test_geometric_sensor_and_physical_rms_are_piston_invariant(static_system):
    system, opd = static_system
    phase = 2.0 * np.asarray(system.pupil_grid.x)
    expected = system.slopes_from_phase(phase, add_noise=False)
    for piston in (0.0, 0.7, 10 * np.pi):
        opd[:] = (phase + piston) * system.config.wfs.wavelength / (2 * np.pi)
        np.testing.assert_allclose(system.measure_wfs(), expected, atol=1e-18, rtol=1e-11)
        expected_science = (phase + piston) * system.config.wfs.wavelength / system.config.science_wavelength
        np.testing.assert_allclose(system.residual_phase(), expected_science, atol=1e-13, rtol=1e-13)
        physical = residual_rms(expected_science, system.aperture_mask)
        assert residual_rms(system.residual_phase(), system.aperture_mask) == pytest.approx(physical)


def test_calibration_is_invariant_to_poke_phase_wrapping(static_system):
    system, _ = static_system
    small = measure_poke_matrix(system, poke_amplitude=1e-9)
    large = measure_poke_matrix(system, poke_amplitude=1e-6)
    np.testing.assert_allclose(large, small, atol=1e-12, rtol=1e-11)


def test_idealized_fit_cancels_large_unwrapped_native_opd(static_system):
    system, opd = static_system
    recon = build_reconstructor(measure_poke_matrix(system, poke_amplitude=1e-9), 0.5)
    system.dm.actuators = 2e-6 * recon.B[:, 0]
    opd[:] = system.dm.opd
    system.dm.actuators = np.zeros(system.n_actuators)
    controller = IdealizedController(system, recon)
    controller.reset(system.n_actuators)
    system.dm.actuators = controller.compute(None)
    native_residual = system.atmosphere.phase_for(system.config.science_wavelength)
    native_residual += system.dm.phase_for(system.config.science_wavelength)
    assert residual_rms(np.asarray(native_residual), system.aperture_mask) < 1e-10


def test_native_phase_matches_complex_propagation_and_refuses_scintillation(static_system):
    system, opd = static_system
    opd[:] = np.asarray(system.pupil_grid.x) * 1e-6
    system.dm.actuators = np.linspace(-1e-6, 1e-6, system.n_actuators)
    wavefront = system._residual_wavefront(system.config.science_wavelength)
    np.testing.assert_allclose(
        np.exp(1j * system.residual_phase())[system.aperture_mask],
        np.exp(1j * np.asarray(wavefront.phase))[system.aperture_mask],
        atol=1e-13,
    )
    system.atmosphere.scintillation = True
    with pytest.raises(ValueError, match="scintillation"):
        system.residual_phase()


def test_step_returns_independent_actual_command_snapshot(static_system):
    system, _ = static_system
    supplied = np.linspace(-1e-9, 1e-9, system.n_actuators)
    expected = supplied.copy()
    result = system.step(supplied)
    supplied[:] = 0
    system.dm.actuators[:] = 1e-9
    np.testing.assert_array_equal(result.applied_command, expected)
    result.applied_command[:] = 2e-9
    assert np.all(system.dm.actuators == 1e-9)


def test_noise_reset_is_local_and_reproducible(static_system):
    system, _ = static_system
    system._slope_noise_std = 1e-7
    system.reset(seed=23)
    first = system.measure_wfs()
    second = system.measure_wfs()
    np.random.seed(999)
    np.random.normal(size=100)
    system.reset(seed=23)
    np.testing.assert_array_equal(system.measure_wfs(), first)
    np.testing.assert_array_equal(system.measure_wfs(), second)
    system.reset(seed=24)
    assert not np.array_equal(system.measure_wfs(), first)


@pytest.mark.parametrize("delay", [0, -1, True, False, 1.5, "2"])
def test_invalid_delay_is_rejected(delay):
    with pytest.raises(ValueError, match="delay"):
        LoopConfig(frame_rate_hz=500.0, delay_frames=delay)
    with pytest.raises(ValueError, match="delay"):
        EOFController(scalar_reconstructor(), delay_frames=delay)


def scalar_reconstructor():
    return build_reconstructor(np.ones((1, 1)), 1.0)


def test_eof_requires_actual_applied_context_and_subtracts_it():
    controller = EOFController(scalar_reconstructor(), delay_frames=2)
    controller.reset(1)
    controller.v[:] = 7.0  # latest issued is deliberately not the applied command
    with pytest.raises(ValueError, match="applied"):
        controller.compute(np.array([5.0]))
    np.testing.assert_allclose(controller.compute(np.array([5.0]), applied_command=np.array([2.0])), [-3.0])
    np.testing.assert_allclose(controller._pol_history[-1], [3.0])


def test_unfitted_eof_uses_persistence_through_history_fill_and_reset():
    controller = EOFController(scalar_reconstructor(), delay_frames=2, history_length=3)
    for _ in range(2):
        controller.reset(1)
        for _ in range(8):
            np.testing.assert_allclose(
                controller.compute(np.array([1.0]), applied_command=np.array([0.0])),
                [-1.0],
            )


def test_eof_first_fit_respects_both_minimum_and_refit_cadence():
    controller = EOFController(
        scalar_reconstructor(),
        delay_frames=2,
        history_length=1,
        min_fit_frames=7,
        refit_every=5,
        ridge=0.1,
    )
    controller.reset(1)
    for frame in range(1, 11):
        command = controller.compute(np.array([1e-8]), applied_command=np.array([0.0]))
        expected = -1e-8 if frame < 10 else -1e-8 / 1.1
        assert command[0] == pytest.approx(expected, rel=1e-12, abs=1e-22)
        assert bool(controller._fitted[0]) == (frame == 10)


def fitted_controller(history, *, delay=2, ridge=1e-3, h=1):
    controller = EOFController(scalar_reconstructor(), delay_frames=delay, history_length=h, ridge=ridge)
    controller.reset(1)
    controller._pol_history.extend(np.array([value]) for value in history)
    controller._refit()
    return controller


def test_eof_ridge_is_dimensionless_and_constant_nanometre_signal_remains_predictable():
    history = 1.02 ** np.arange(24)
    unit = fitted_controller(history)
    for scale in (1e-8, 1e-200, 1e200):
        physical = fitted_controller(history * scale)
        np.testing.assert_allclose(physical.weights, unit.weights, rtol=1e-12, atol=0)
    constant = fitted_controller(np.full(20, 1e-8))
    assert constant.weights[0, 0] == pytest.approx(1 / (1 + 1e-3), rel=1e-12)


def test_zero_power_fit_keeps_persistence_for_a_new_signal():
    controller = fitted_controller(np.zeros(20))
    np.testing.assert_allclose(
        controller.compute(np.array([1e-8]), applied_command=np.array([0.0])),
        [-1e-8],
        atol=1e-20,
    )


def test_zero_power_fallback_is_per_mode():
    controller = EOFController(build_reconstructor(np.eye(2), 1.0), delay_frames=2, history_length=1)
    controller.reset(2)
    controller._pol_history.extend(np.array([1e-8, 0.0]) for _ in range(20))
    controller._refit()
    np.testing.assert_allclose(
        controller.compute(np.array([1e-8, 2e-8]), applied_command=np.zeros(2)),
        [-1e-8 / 1.001, -2e-8],
        rtol=1e-12,
        atol=1e-22,
    )


@pytest.mark.parametrize("delay", [1, 2, 3])
def test_eof_fit_preserves_exact_declared_forecast_horizon(delay):
    rho = 1.02
    controller = fitted_controller(rho ** np.arange(24), delay=delay, ridge=0)
    assert controller.weights[0, 0] == pytest.approx(rho**delay, rel=1e-12)


@pytest.mark.parametrize("delay", [2, 3])
def test_actual_runner_and_eof_cancel_predictable_delayed_disturbance(delay):
    rho = 1.02

    class Plant:
        n_actuators = 1
        config = SimpleNamespace(loop=LoopConfig(frame_rate_hz=500, delay_frames=delay))

        def reset(self, seed=None):
            self.t = 0
            self.applied = []

        def step(self, command):
            disturbance = rho**self.t
            self.applied.append(command.copy())
            self.t += 1
            residual = disturbance + command[0]
            return SimpleNamespace(
                slopes=np.array([residual]),
                residual_phase=np.array([-residual, residual]),
                aperture_mask=np.ones(2, dtype=bool),
                applied_command=command.copy(),
            )

    class TrainedEOF(EOFController):
        def reset(self, n_actuators):
            super().reset(n_actuators)
            self._pol_history.extend(np.array([rho**t]) for t in range(24))
            self._refit()

    plant = Plant()
    controller = TrainedEOF(scalar_reconstructor(), delay_frames=delay, history_length=1, ridge=0)
    telemetry = run_closed_loop(plant, controller, n_frames=10)
    np.testing.assert_allclose(telemetry.residual_rms_rad[delay:], 0, atol=1e-12)
    np.testing.assert_array_equal(np.array(plant.applied[:delay]), 0)
    np.testing.assert_allclose(np.array(plant.applied[delay:]).ravel(), -(rho ** np.arange(delay, 10)))


def test_runner_refuses_a_predictor_with_a_different_delay():
    plant = SimpleNamespace(config=SimpleNamespace(loop=LoopConfig(frame_rate_hz=500, delay_frames=2)))
    controller = EOFController(scalar_reconstructor(), delay_frames=3)
    with pytest.raises(ValueError, match="delay"):
        run_closed_loop(plant, controller, n_frames=1)
