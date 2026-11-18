"""Science images must describe the same state as their reported loop frame."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from noether_optics.bench.refsim.config import petscale_config
from noether_optics.bench.refsim.run_loop import run_closed_loop
from noether_optics.bench.refsim.system import RefSimSystem


class FrameSystem:
    config = SimpleNamespace(loop=SimpleNamespace(delay_frames=2))
    n_actuators = 1

    def reset(self, seed):
        self.t = 0
        self.image_frames = []

    def science_image(self):
        self.image_frames.append(self.t)
        return np.array([float(self.t)])

    def step(self, command, *, record_science_image=False):
        result = SimpleNamespace(
            slopes=np.array([self.t]),
            residual_phase=np.array([-self.t, self.t]),
            aperture_mask=np.ones(2, dtype=bool),
            applied_command=command.copy(),
            t=self.t / 500,
            science_image=self.science_image() if record_science_image else None,
        )
        self.t += 1
        return result


class ZeroController:
    name = "zero"

    def reset(self, n_actuators):
        self.n_actuators = n_actuators

    def compute(self, slopes, *, applied_command=None):
        return np.zeros(self.n_actuators)


def test_runner_image_matches_scored_frame_after_warmup():
    system = FrameSystem()
    telemetry = run_closed_loop(system, ZeroController(), 6, warmup_frames=1, record_science_image_every=2)
    np.testing.assert_array_equal(telemetry.residual_rms_rad, np.arange(1, 6))
    np.testing.assert_array_equal(telemetry.final_science_image, [4.0])
    assert telemetry.final_science_image_frame == 4
    assert telemetry.final_science_image_time_s == 4 / 500
    assert system.image_frames == [2, 4]


@pytest.mark.parametrize("cadence", [None, 10])
def test_no_selected_image_has_no_frame_identity(cadence):
    system = FrameSystem()
    telemetry = run_closed_loop(system, ZeroController(), 3, warmup_frames=1, record_science_image_every=cadence)
    assert telemetry.final_science_image is None
    assert telemetry.final_science_image_frame is None
    assert telemetry.final_science_image_time_s is None
    assert system.image_frames == []


@pytest.mark.parametrize("cadence", [0, -1, True, 1.5])
def test_invalid_image_cadence_rejected_before_reset(cadence):
    system = FrameSystem()
    with pytest.raises(ValueError, match="positive integer"):
        run_closed_loop(system, ZeroController(), 3, record_science_image_every=cadence)
    assert not hasattr(system, "t")


def test_step_image_uses_pre_advance_atmosphere_and_applied_command(monkeypatch):
    config = petscale_config()
    config = replace(
        config,
        wfs=replace(config.wfs, num_subapertures=4, pixels_per_subaperture=8),
        dm=replace(config.dm, num_actuators_across=5),
    )

    class Atmosphere:
        t = 0.0
        scintillation = False

        def phase_for(self, wavelength):
            return 2 * np.pi * system.pupil_grid.x * 1e-7 * (1 + 100 * self.t) / wavelength

        def forward(self, wavefront):
            result = wavefront.copy()
            result.electric_field *= np.exp(1j * self.phase_for(wavefront.wavelength))
            return result

    monkeypatch.setattr(RefSimSystem, "_build_atmosphere", lambda self, seed: setattr(self, "atmosphere", Atmosphere()))
    system = RefSimSystem(config, seed=17)
    system._slope_noise_std = 0.0
    command = np.linspace(-1e-8, 1e-8, system.n_actuators)
    system.dm.actuators = command.copy()
    expected_phase = system.residual_phase().copy()
    expected_slopes = system.measure_wfs().copy()
    expected_image = system.science_image().copy()
    result = system.step(command, record_science_image=True)
    np.testing.assert_array_equal(result.residual_phase, expected_phase)
    np.testing.assert_array_equal(result.slopes, expected_slopes)
    np.testing.assert_array_equal(result.science_image, expected_image)
    np.testing.assert_array_equal(result.applied_command, command)
    assert result.t == 0
    assert system.atmosphere.t == system.dt
    assert not np.allclose(system.science_image(), expected_image)
    retained = result.science_image.copy()
    system.step(np.zeros(system.n_actuators))
    np.testing.assert_array_equal(result.science_image, retained)
