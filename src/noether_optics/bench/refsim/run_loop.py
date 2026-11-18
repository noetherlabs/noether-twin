"""Minimal closed-loop runner for Phase 1 (petscale). The full scenario harness
with trace persistence lands in Phase 3 (`bench/harness`); this module is
deliberately small: apply a hard frame delay, drive the controller, and
collect the telemetry needed for the four canonical metrics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from noether_optics.bench.metrics import residual_rms, strehl_from_phase
from noether_optics.bench.refsim.config import validate_delay_frames
from noether_optics.bench.refsim.system import RefSimSystem


class Controller(Protocol):
    name: str

    def reset(self, n_actuators: int) -> None: ...
    def compute(
        self,
        slopes: NDArray[np.float64],
        *,
        applied_command: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]: ...


@dataclass
class LoopTelemetry:
    residual_rms_rad: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    strehl: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    final_science_image: NDArray[np.float64] | None = None
    final_science_image_frame: int | None = None
    final_science_image_time_s: float | None = None


def run_closed_loop(
    system: RefSimSystem,
    controller: Controller,
    n_frames: int,
    seed: int | None = None,
    warmup_frames: int = 0,
    record_science_image_every: int | None = None,
) -> LoopTelemetry:
    if record_science_image_every is not None and (
        isinstance(record_science_image_every, bool)
        or not isinstance(record_science_image_every, int)
        or record_science_image_every < 1
    ):
        raise ValueError("Science-image cadence must be a positive integer or None")
    delay = system.config.loop.delay_frames
    validate_delay_frames(delay)
    if getattr(controller, "delay_frames", delay) != delay:
        raise ValueError("Controller forecast delay must match the plant's command delay")
    system.reset(seed)
    controller.reset(system.n_actuators)

    pending_commands: deque[NDArray[np.float64]] = deque(
        [np.zeros(system.n_actuators) for _ in range(delay)], maxlen=delay + 1
    )

    residual_rms_list = []
    strehl_list = []
    final_science_image = None
    final_science_image_frame = None
    final_science_image_time_s = None

    for t in range(n_frames):
        command_to_apply = pending_commands.popleft() if pending_commands else np.zeros(system.n_actuators)
        record_image = (
            record_science_image_every is not None and t >= warmup_frames and t % record_science_image_every == 0
        )
        result = (
            system.step(command_to_apply, record_science_image=True) if record_image else system.step(command_to_apply)
        )
        new_command = controller.compute(result.slopes, applied_command=result.applied_command.copy())
        pending_commands.append(new_command.copy())

        if t >= warmup_frames:
            rms = residual_rms(result.residual_phase, result.aperture_mask.astype(np.float64))
            residual_rms_list.append(rms)
            strehl_list.append(strehl_from_phase(result.residual_phase, result.aperture_mask.astype(np.float64)))
            if record_image:
                final_science_image = result.science_image
                final_science_image_frame = t
                final_science_image_time_s = result.t

    telemetry = LoopTelemetry(
        residual_rms_rad=np.array(residual_rms_list),
        strehl=np.array(strehl_list),
        final_science_image=final_science_image,
        final_science_image_frame=final_science_image_frame,
        final_science_image_time_s=final_science_image_time_s,
    )
    return telemetry
