"""Per-mode linear prediction from pseudo-open-loop telemetry.

Following the temporal-prediction approach of Guyon and Males (2017),
arXiv:1707.00570, each mode is fitted by regularized least squares over
its own sliding history. This independent AR(h) approximation omits the
cross-modal history used by the full empirical-orthogonal-function method.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from noether_optics.bench.refsim.calibration import Reconstructor
from noether_optics.bench.refsim.config import validate_delay_frames


class EOFController:
    name = "eof"

    def __init__(
        self,
        reconstructor: Reconstructor,
        delay_frames: int,
        history_length: int = 10,
        ridge: float = 1e-3,
        refit_every: int = 200,
        min_fit_frames: int = 400,
        fit_window: int = 2000,
    ):
        validate_delay_frames(delay_frames)
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        self.reconstructor = reconstructor
        self.delay_frames = delay_frames
        self.h = history_length
        self.ridge = ridge
        self.refit_every = refit_every
        self.min_fit_frames = min_fit_frames
        self.fit_window = fit_window

        self.n_modes = reconstructor.n_modes_kept
        self.weights: NDArray[np.float64] | None = None  # (n_modes, h)
        self._fitted = np.zeros(self.n_modes, dtype=bool)
        self.v: NDArray[np.float64] | None = None
        self._pol_history: deque[NDArray[np.float64]] = deque(maxlen=fit_window)
        self._t = 0

    def reset(self, n_actuators: int) -> None:
        self.v = np.zeros(n_actuators)
        self.weights = np.zeros((self.n_modes, self.h))
        self._fitted[:] = False
        self._pol_history.clear()
        self._t = 0

    def _pseudo_open_loop(
        self,
        slopes: NDArray[np.float64],
        applied_command: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        assert self.v is not None, "call reset() before compute()"
        residual_modal = self.reconstructor.modal_coeffs(slopes)
        applied_modal = self.reconstructor.modal_coeffs_of_command(applied_command)
        return residual_modal - applied_modal

    def _refit(self) -> None:
        assert self.weights is not None, "call reset() before compute()"
        history = np.array(self._pol_history)  # (T, n_modes)
        T = history.shape[0]
        target_idx = np.arange(self.h - 1 + self.delay_frames, T)
        if len(target_idx) < self.h + 5:
            return
        ridge_rows = np.sqrt(self.ridge) * np.eye(self.h)
        for mode in range(self.n_modes):
            series = history[:, mode]
            X = np.empty((len(target_idx), self.h))
            for k in range(self.h):
                X[:, k] = series[target_idx - self.delay_frames - k]
            y = series[target_idx]
            # Scale from this fit's regressors only, with a stable RMS calculation.
            # Input/output use the same scale, so fitted weights act on physical
            # modal coefficients unchanged. Minimize mean error + ridge * ||w||^2.
            peak = np.max(np.abs(X))
            self._fitted[mode] = False
            if peak == 0:
                self.weights[mode] = 0
                continue
            scaled_X = X / peak
            rms = np.sqrt(np.mean(scaled_X**2))
            normalization = np.sqrt(len(y))
            design = np.vstack((scaled_X / rms / normalization, ridge_rows))
            target = np.concatenate((y / peak / rms / normalization, np.zeros(self.h)))
            if not np.all(np.isfinite(target)):
                raise ValueError("EOF training target cannot be represented at the fitted input scale")
            weights, *_ = np.linalg.lstsq(design, target, rcond=None)
            if not np.all(np.isfinite(weights)):
                raise ValueError("EOF fit produced nonfinite weights")
            self.weights[mode] = weights
            self._fitted[mode] = True

    def compute(
        self,
        slopes: NDArray[np.float64],
        *,
        applied_command: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        assert self.v is not None and self.weights is not None
        if applied_command is None:
            raise ValueError("EOF requires the actual applied command for the current measurement")
        applied_command = np.asarray(applied_command, dtype=np.float64)
        if applied_command.shape != self.v.shape or not np.all(np.isfinite(applied_command)):
            raise ValueError("The applied command must have one finite value per actuator")
        pol = self._pseudo_open_loop(slopes, applied_command)
        if not np.all(np.isfinite(pol)):
            raise ValueError("EOF requires finite pseudo-open-loop observations")
        self._pol_history.append(pol)
        self._t += 1

        if self._t % self.refit_every == 0 and self._t >= self.min_fit_frames:
            self._refit()

        predicted_modal = pol.copy()  # Persistence until this mode has a usable fit.
        if len(self._pol_history) >= self.h and np.any(self._fitted):
            recent = np.array(self._pol_history)[-self.h :]  # (h, n_modes), most recent last
            recent = recent[::-1]  # most recent first, matches weight ordering
            prediction = np.einsum("mh,hm->m", self.weights, recent)
            predicted_modal[self._fitted] = prediction[self._fitted]

        command = -(self.reconstructor.B @ predicted_modal)
        self.v = command
        return self.v.copy()
