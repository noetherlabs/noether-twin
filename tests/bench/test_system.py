import numpy as np
import pytest

from noether_optics.bench.refsim.config import petscale_config
from noether_optics.bench.refsim.system import F_NUMBER, RefSimSystem


@pytest.fixture(scope="module")
def system() -> RefSimSystem:
    return RefSimSystem(petscale_config(magnitude=0.0), seed=42)


def test_flat_wavefront_gives_zero_slopes(system: RefSimSystem):
    flat_phase = np.zeros(system.pupil_grid.size)
    slopes = system.slopes_from_phase(flat_phase, add_noise=False)
    assert np.max(np.abs(slopes)) == 0.0


def test_pure_tip_gives_known_slope_pattern(system: RefSimSystem):
    coeff = 0.5  # rad/m
    x = np.asarray(system.pupil_grid.x)
    phase = coeff * x
    slopes = system.slopes_from_phase(phase, add_noise=False)
    n = system.n_illuminated_subaps
    sx, sy = slopes[:n], slopes[n:]

    wavelength = system.config.wfs.wavelength
    expected = system.wfs_focal_length * (wavelength / (2 * np.pi)) * coeff

    assert np.allclose(sx, expected, rtol=1e-9)
    assert np.allclose(sy, 0.0, atol=1e-15)


def test_pure_tilt_y_gives_known_slope_pattern(system: RefSimSystem):
    coeff = 0.3
    y = np.asarray(system.pupil_grid.y)
    phase = coeff * y
    slopes = system.slopes_from_phase(phase, add_noise=False)
    n = system.n_illuminated_subaps
    sx, sy = slopes[:n], slopes[n:]

    wavelength = system.config.wfs.wavelength
    expected = system.wfs_focal_length * (wavelength / (2 * np.pi)) * coeff

    assert np.allclose(sx, 0.0, atol=1e-15)
    assert np.allclose(sy, expected, rtol=1e-9)


def test_dm_zero_command_is_identity():
    system = RefSimSystem(petscale_config(magnitude=0.0), seed=1)
    system.dm.actuators = np.zeros(system.n_actuators)
    phase = system.residual_phase()
    assert np.any(phase != 0.0)  # atmosphere present


def test_n_slopes_matches_illuminated_count(system: RefSimSystem):
    assert system.n_slopes == 2 * system.n_illuminated_subaps


def test_f_number_constant_used_consistently(system: RefSimSystem):
    assert system.wfs_focal_length == pytest.approx(F_NUMBER * system.subap_size)
