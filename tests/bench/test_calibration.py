import numpy as np
import pytest

from noether_optics.bench.refsim.calibration import build_reconstructor, measure_poke_matrix
from noether_optics.bench.refsim.config import petscale_config
from noether_optics.bench.refsim.system import RefSimSystem


@pytest.fixture(scope="module")
def system() -> RefSimSystem:
    return RefSimSystem(petscale_config(magnitude=0.0), seed=42)


@pytest.fixture(scope="module")
def poke_matrix(system: RefSimSystem) -> np.ndarray:
    return measure_poke_matrix(system)


def test_poke_matrix_shape(system: RefSimSystem, poke_matrix: np.ndarray):
    assert poke_matrix.shape == (system.n_slopes, system.n_actuators)


def test_most_actuators_are_sensed(poke_matrix: np.ndarray):
    col_norms = np.linalg.norm(poke_matrix, axis=0)
    sensed = col_norms > 1e-6 * col_norms.max()
    assert sensed.sum() > 0.9 * len(col_norms)


def test_reconstructor_truncation_matches_dof_fraction(poke_matrix: np.ndarray):
    for dof in [0.85, 0.5, 0.2]:
        recon = build_reconstructor(poke_matrix, dof_fraction=dof)
        assert recon.n_modes_kept == max(1, round(dof * recon.n_modes_total))
        assert recon.B.shape[1] == recon.n_modes_kept
        assert recon.C.shape == (poke_matrix.shape[1], poke_matrix.shape[0])


def test_reconstructor_recovers_single_actuator_poke(system: RefSimSystem, poke_matrix: np.ndarray):
    recon = build_reconstructor(poke_matrix, dof_fraction=0.85)
    test_cmd = np.zeros(system.n_actuators)
    test_cmd[300] = 1e-8

    import hcipy as hp

    wf = hp.Wavefront(system.aperture, system.config.wfs.wavelength)
    wf.total_power = 1.0
    system.dm.actuators = test_cmd
    wf = system.dm.forward(wf)
    slopes = system.slopes_from_phase(np.asarray(wf.phase), add_noise=False)

    recovered = recon.C @ slopes
    cosine = np.dot(recovered, test_cmd) / (np.linalg.norm(recovered) * np.linalg.norm(test_cmd))
    # A single-actuator poke is high-spatial-frequency content that a per-subaperture
    # mean-tilt (geometric) sensor, combined with 85%-DoF modal truncation, only
    # partially localizes -- this is a coarse sanity bound, not a precision target.
    assert cosine > 0.35


def test_command_scale_reduces_wfs_domain_residual(system: RefSimSystem, poke_matrix: np.ndarray):
    """Applying the reconstructed correction should not increase the sensed
    residual (regression test for the earlier undersampled-SHS bug where
    corrections had ~zero or negative effect on the WFS-domain residual)."""
    recon = build_reconstructor(poke_matrix, dof_fraction=0.85)
    system.reset(seed=42)
    system.dm.actuators = np.zeros(system.n_actuators)
    slopes0 = system.measure_wfs()
    norm0 = np.linalg.norm(slopes0)

    system.dm.actuators = -(recon.C @ slopes0)
    slopes1 = system.measure_wfs()
    norm1 = np.linalg.norm(slopes1)

    assert norm1 < norm0
