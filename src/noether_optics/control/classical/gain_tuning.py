"""Grid-search gain tuning for the integrator, selected on mean Strehl."""

from __future__ import annotations

import numpy as np

from noether_optics.bench.refsim.calibration import Reconstructor
from noether_optics.bench.refsim.run_loop import run_closed_loop
from noether_optics.bench.refsim.system import RefSimSystem
from noether_optics.control.classical.integrator import IntegratorController

DEFAULT_GAIN_GRID = tuple(np.arange(0.1, 0.95, 0.1).round(2))


def tune_integrator_gain(
    system: RefSimSystem,
    reconstructor: Reconstructor,
    n_frames: int = 400,
    warmup_frames: int = 200,
    seed: int = 0,
    gain_grid: tuple[float, ...] = DEFAULT_GAIN_GRID,
) -> dict:
    """Return {"best_gain": ..., "results": {gain: mean_strehl}}."""
    results = {}
    for gain in gain_grid:
        ctrl = IntegratorController(reconstructor, gain=gain)
        telemetry = run_closed_loop(system, ctrl, n_frames=n_frames, seed=seed, warmup_frames=warmup_frames)
        results[float(gain)] = float(telemetry.strehl.mean()) if len(telemetry.strehl) else float("nan")
    best_gain = max(results, key=lambda g: results[g])
    return {"best_gain": best_gain, "results": results}
