"""Negative feedback for a positive empirical DM response: v_t = v_{t-1} - g C s_t."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from noether_optics.bench.refsim.calibration import Reconstructor


class IntegratorController:
    name = "integrator"

    def __init__(self, reconstructor: Reconstructor, gain: float, filter_to_controllable: bool = True):
        self.reconstructor = reconstructor
        self.gain = gain
        self.filter_to_controllable = filter_to_controllable
        self.v: NDArray[np.float64] | None = None

    def reset(self, n_actuators: int) -> None:
        self.v = np.zeros(n_actuators)

    def compute(
        self,
        slopes: NDArray[np.float64],
        *,
        applied_command: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        assert self.v is not None, "call reset() before compute()"
        delta = self.gain * (self.reconstructor.C @ slopes)
        if self.filter_to_controllable:
            delta = self.reconstructor.BtB @ delta
        self.v = self.v - delta
        return self.v.copy()
