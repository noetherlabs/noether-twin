"""Ground-truth spatial fitting oracle for the reference simulator.

Uses ground-truth knowledge only available in the reference simulator: it reads
the actual atmospheric phase directly (bypassing the WFS entirely) and drives
the DM to cancel it exactly, subject only to the DM's own influence-function
fitting error. This oracle fits all DM influence functions, not the
reconstructor's truncated sensing subspace. A standalone fit is a spatial
reference; the generic runner still applies its configured command delay.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from noether_optics.bench.refsim.calibration import Reconstructor
from noether_optics.bench.refsim.system import RefSimSystem


class IdealizedController:
    name = "idealized"

    def __init__(self, system: RefSimSystem, reconstructor: Reconstructor):
        self.system = system
        self.reconstructor = reconstructor
        self.v: NDArray[np.float64] | None = None

    def reset(self, n_actuators: int) -> None:
        self.v = np.zeros(n_actuators)

    def compute(
        self,
        slopes: NDArray[np.float64] | None,
        *,
        applied_command: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Ignores `slopes`; reads ground-truth atmosphere phase and fits the DM
        to it directly via a least-squares projection onto the DM influence
        functions (available in refsim only -- not a realizable controller)."""
        phase = np.asarray(self.system.atmosphere.phase_for(self.system.config.science_wavelength))
        opd = phase * self.system.config.science_wavelength / (2 * np.pi)

        influence = self.system.dm.influence_functions.transformation_matrix.toarray()
        mask = self.system.aperture_mask
        A = influence[mask, :]
        # HCIPy's DeformableMirror.forward applies phase = 2 * wavenumber * surface
        # (reflective double-pass convention), i.e. surface = influence @ actuators
        # produces an *OPD-equivalent* contribution of 2 * surface, not 1 * surface.
        # Solve for surface = -opd / 2 so that 2 * surface cancels -opd.
        b = -opd[mask] / 2.0
        command, *_ = np.linalg.lstsq(A, b, rcond=None)
        self.v = command
        return self.v.copy()
