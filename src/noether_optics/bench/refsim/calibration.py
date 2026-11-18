"""Interaction-matrix calibration and truncated-SVD reconstruction.

Actuator modes are ranked by the singular values of the poke matrix D.
The reconstructor does not whiten the actuator basis by a turbulence
covariance matrix. It is therefore a truncated pseudo-inverse, not the
covariance-whitened Karhunen-Loeve construction of Gendron (1994).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from noether_optics.bench.refsim.system import RefSimSystem

ACTUATOR_RESPONSE_FLOOR = 1e-6  # fraction of max column norm; below this, actuator is unsensed


@dataclass
class Reconstructor:
    D: NDArray[np.float64]  # interaction matrix, n_slopes x n_act
    B: NDArray[np.float64]  # modal basis (controllable actuator subspace), n_act x n_modes
    M: NDArray[np.float64]  # modal demodulator, n_modes x n_slopes: mode_coeffs = M @ slopes
    C: NDArray[np.float64]  # reconstructor, n_act x n_slopes: command = C @ slopes = B @ (M @ slopes)
    BtB: NDArray[np.float64]  # projector onto controllable subspace, n_act x n_act
    singular_values: NDArray[np.float64]
    n_modes_total: int
    n_modes_kept: int

    def modal_coeffs(self, slopes: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.M @ slopes

    def modal_coeffs_of_command(self, command: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project an actuator-space command onto the modal basis (B is orthonormal)."""
        return self.B.T @ command


def measure_poke_matrix(system: RefSimSystem, poke_amplitude: float = 1e-7) -> NDArray[np.float64]:
    """Measure the interaction matrix with no atmosphere (flat incoming wavefront),
    noise-free, via central differences."""
    n_act = system.n_actuators
    D = np.zeros((system.n_slopes, n_act))
    wavelength = system.config.wfs.wavelength
    original_actuators = np.array(system.dm.actuators)
    try:
        for i in range(n_act):
            plus = np.zeros(n_act)
            plus[i] = poke_amplitude
            minus = np.zeros(n_act)
            minus[i] = -poke_amplitude

            system.dm.actuators = plus
            s_plus = system.slopes_from_phase(np.asarray(system.dm.phase_for(wavelength)), add_noise=False)

            system.dm.actuators = minus
            s_minus = system.slopes_from_phase(np.asarray(system.dm.phase_for(wavelength)), add_noise=False)

            D[:, i] = (s_plus - s_minus) / (2 * poke_amplitude)
    finally:
        system.dm.actuators = original_actuators
    return D


def build_reconstructor(D: NDArray[np.float64], dof_fraction: float = 0.85) -> Reconstructor:
    """SVD-of-D truncated pseudo-inverse reconstructor (see module docstring)."""
    col_norms = np.linalg.norm(D, axis=0)
    sensed = col_norms > ACTUATOR_RESPONSE_FLOOR * col_norms.max()

    D_sensed = D[:, sensed]
    U, S, Vt = np.linalg.svd(D_sensed, full_matrices=False)
    n_total = len(S)
    n_keep = max(1, int(round(dof_fraction * n_total)))

    n_act = D.shape[1]
    V_full = np.zeros((n_act, n_total))
    V_full[sensed, :] = Vt.T

    B = V_full[:, :n_keep]
    S_keep = S[:n_keep]
    U_keep = U[:, :n_keep]

    # command = C @ slopes recovers modal coefficients projected back to actuator space:
    # mode_coeffs = (1/S) * U_keep^T @ slopes ; command = B @ mode_coeffs
    M = np.diag(1.0 / S_keep) @ U_keep.T
    C = B @ M
    BtB = B @ B.T

    return Reconstructor(D=D, B=B, M=M, C=C, BtB=BtB, singular_values=S, n_modes_total=n_total, n_modes_kept=n_keep)
