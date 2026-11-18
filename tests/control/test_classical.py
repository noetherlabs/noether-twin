import numpy as np
import pytest

from noether_optics.bench.metrics import residual_rms
from noether_optics.bench.refsim.calibration import build_reconstructor, measure_poke_matrix
from noether_optics.bench.refsim.config import petscale_config
from noether_optics.bench.refsim.run_loop import run_closed_loop
from noether_optics.bench.refsim.system import RefSimSystem
from noether_optics.control.classical.eof import EOFController
from noether_optics.control.classical.idealized import IdealizedController
from noether_optics.control.classical.integrator import IntegratorController


@pytest.fixture(scope="module")
def system_and_reconstructor():
    system = RefSimSystem(petscale_config(magnitude=0.0), seed=42)
    D = measure_poke_matrix(system)
    recon = build_reconstructor(D, dof_fraction=0.85)
    return system, recon


def test_idealized_beats_open_loop(system_and_reconstructor):
    system, recon = system_and_reconstructor
    system.reset(seed=42)
    system.dm.actuators = np.zeros(system.n_actuators)
    open_loop_rms = residual_rms(system.residual_phase(), system.aperture_mask.astype(float))

    ideal = IdealizedController(system, recon)
    ideal.reset(system.n_actuators)
    system.dm.actuators = ideal.compute(None)
    ideal_rms = residual_rms(system.residual_phase(), system.aperture_mask.astype(float))

    assert ideal_rms < open_loop_rms


def test_integrator_runs_and_produces_finite_telemetry(system_and_reconstructor):
    system, recon = system_and_reconstructor
    ctrl = IntegratorController(recon, gain=0.3)
    telemetry = run_closed_loop(system, ctrl, n_frames=50, seed=42, warmup_frames=20)
    assert len(telemetry.residual_rms_rad) == 30
    assert np.all(np.isfinite(telemetry.residual_rms_rad))
    assert np.all(telemetry.strehl >= 0.0) and np.all(telemetry.strehl <= 1.0)


def test_eof_runs_and_produces_finite_telemetry(system_and_reconstructor):
    system, recon = system_and_reconstructor
    ctrl = EOFController(
        recon, delay_frames=system.config.loop.delay_frames, history_length=5, refit_every=50, min_fit_frames=60
    )
    telemetry = run_closed_loop(system, ctrl, n_frames=100, seed=42, warmup_frames=20)
    assert len(telemetry.residual_rms_rad) == 80
    assert np.all(np.isfinite(telemetry.residual_rms_rad))
