"""HCIPy reference propagation with a geometric wavefront-sensor model.

Lenslet slopes use per-subaperture least-squares fits to the local physical
phase. Photon and read noise are applied to the slope measurements using
a centroid-noise approximation. The sensor does not propagate detector
images through lenslet optics, model spot elongation or establish nonlinear
pyramid-sensor optical gain. Analytic tilt checks validate this geometric
measurement model rather than an independently diffractive sensor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import hcipy as hp
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from noether_optics.bench.refsim.config import NO_MR, MisregistrationConfig, SystemConfig
from noether_optics.bench.refsim.photometry import magnitude_to_total_power

SUBAPERTURE_ILLUMINATION_THRESHOLD = 0.75  # fraction of subaperture area that must be illuminated
F_NUMBER = 8.0


@dataclass
class StepResult:
    slopes: NDArray[np.float64]
    residual_phase: NDArray[np.float64]
    aperture_mask: NDArray[np.bool_]
    t: float
    applied_command: NDArray[np.float64]
    science_image: NDArray[np.float64] | None = None


class RefSimSystem:
    def __init__(self, config: SystemConfig, misregistration: MisregistrationConfig = NO_MR, seed: int | None = None):
        self.config = config
        self.misregistration = misregistration
        self.dt = 1.0 / config.loop.frame_rate_hz
        self._reset_noise(seed)

        n_grid = config.effective_pupil_grid_size
        self.pupil_grid = hp.make_pupil_grid(n_grid, config.aperture.diameter)
        self.aperture = hp.make_obstructed_circular_aperture(
            config.aperture.diameter, config.aperture.central_obscuration_ratio
        )(self.pupil_grid)
        self.aperture_mask = np.asarray(self.aperture) > 0.5

        self._build_atmosphere(seed)
        self._build_dm()
        self._build_geometric_wfs()
        self._build_science_path()

        self.dm.actuators = np.zeros(self.dm.num_actuators)

    # -- construction -------------------------------------------------

    def _build_atmosphere(self, seed: int | None) -> None:
        cfg = self.config.atmosphere
        cn2_total = hp.Cn_squared_from_fried_parameter(cfg.r0, cfg.r0_wavelength)
        layers = []
        for i, layer in enumerate(cfg.layers):
            cn2 = cn2_total * layer.fraction
            velocity = layer.wind_speed * np.array(
                [np.cos(np.radians(layer.wind_direction_deg)), np.sin(np.radians(layer.wind_direction_deg))]
            )
            layer_seed = None if seed is None else seed + i
            layers.append(
                hp.InfiniteAtmosphericLayer(
                    self.pupil_grid, cn2, cfg.outer_scale, velocity, layer.altitude, seed=layer_seed
                )
            )
        self.atmosphere = hp.MultiLayerAtmosphere(layers, scintillation=False)
        self.atmosphere.t = 0.0

    def _build_dm(self) -> None:
        cfg = self.config.dm
        spacing = self.config.aperture.diameter / cfg.num_actuators_across
        influence_functions = hp.make_gaussian_influence_functions(
            self.pupil_grid, cfg.num_actuators_across, spacing, crosstalk=cfg.coupling
        )
        self.dm = hp.DeformableMirror(influence_functions)

    def _misregistration_offset(self) -> NDArray[np.float64]:
        if self.misregistration.shift_fraction == 0.0:
            return np.zeros(2)
        subap_size = self.config.aperture.diameter / self.config.wfs.num_subapertures
        shift = self.misregistration.shift_fraction * subap_size
        angle = np.radians(self.misregistration.shift_angle_deg)
        return np.array([shift * np.cos(angle), shift * np.sin(angle)])

    def _build_geometric_wfs(self) -> None:
        """Build a sparse operator G (n_slopes x n_pixels) such that
        slopes = G @ phase gives the geometric-optics local tilt of `phase`
        (in radians on self.pupil_grid) within each illuminated subaperture,
        converted to a physical centroid-shift-equivalent via the WFS
        wavelength and focal length. Misregistration shifts which pupil
        pixels each subaperture "sees" relative to the true aperture mask."""
        cfg = self.config.wfs
        n = cfg.num_subapertures
        D = self.config.aperture.diameter
        subap_size = D / n
        self.subap_size = subap_size
        self.wfs_focal_length = F_NUMBER * subap_size

        x = np.asarray(self.pupil_grid.x)
        y = np.asarray(self.pupil_grid.y)
        ap = self.aperture_mask

        offset = self._misregistration_offset()
        centers_1d = -D / 2 + (np.arange(n) + 0.5) * subap_size
        half = subap_size / 2

        rows_b: list[NDArray[np.float64]] = []
        rows_c: list[NDArray[np.float64]] = []
        cols_idx: list[NDArray[np.intp]] = []
        n_pixels = self.pupil_grid.size
        illuminated_count = 0

        explicit = cfg.channel_ids is not None
        requested = cfg.channel_ids if explicit else tuple((iy, ix) for iy in range(n) for ix in range(n))
        channel_ids = []
        for iy, ix in requested:
            sx, sy = centers_1d[ix] + offset[0], centers_1d[iy] + offset[1]
            in_box = (np.abs(x - sx) <= half) & (np.abs(y - sy) <= half)
            idx = np.flatnonzero(in_box & ap)
            total_box = int(np.count_nonzero(in_box))
            if len(idx) < 6:
                if explicit:
                    raise ValueError(f"Under-resolved physical WFS channel {(iy, ix)}: {len(idx)} illuminated samples")
                continue
            if not explicit and (total_box == 0 or len(idx) < 0.75 * max(1, total_box)):
                continue
            xs, ys = x[idx], y[idx]
            xc, yc = xs.mean(), ys.mean()
            dx, dy = xs - xc, ys - yc
            A = np.stack([np.ones_like(dx), dx, dy], axis=1)
            if explicit and np.linalg.matrix_rank(A) != 3:
                raise ValueError(f"Under-resolved physical WFS channel {(iy, ix)}: singular plane fit")
            pinv = np.linalg.pinv(A)  # (3, n_pix_in_subap)
            rows_b.append(pinv[1])
            rows_c.append(pinv[2])
            cols_idx.append(idx)
            channel_ids.append((iy, ix))
            illuminated_count += 1

        self.n_illuminated_subaps = illuminated_count
        self.wfs_channel_ids = tuple(channel_ids)
        if not illuminated_count:
            raise ValueError("The configured WFS has no resolved illuminated channels")

        def _build_sparse(rows: list[NDArray[np.float64]]) -> sparse.csr_matrix:
            data, row_ind, col_ind = [], [], []
            for r, (weights, idx) in enumerate(zip(rows, cols_idx, strict=True)):
                data.append(weights)
                row_ind.append(np.full(len(idx), r))
                col_ind.append(idx)
            return sparse.csr_matrix(
                (np.concatenate(data), (np.concatenate(row_ind), np.concatenate(col_ind))),
                shape=(illuminated_count, n_pixels),
            )

        G_b = _build_sparse(rows_b)
        G_c = _build_sparse(rows_c)
        self._gradient_operator = sparse.vstack([G_b, G_c]).tocsr()

        wavelength = cfg.wavelength
        self._tilt_to_shift = self.wfs_focal_length * wavelength / (2 * np.pi)

        if explicit:
            self.photons_per_subaperture = np.asarray(cfg.photons_per_channel, dtype=np.float64)
            self.wfs_total_power = float(self.photons_per_subaperture.sum())
        else:
            aperture_area = np.pi * (D / 2) ** 2 * (1 - self.config.aperture.central_obscuration_ratio**2)
            self.wfs_total_power = magnitude_to_total_power(
                self.config.guide_star.magnitude, aperture_area, self.config.loop.frame_rate_hz
            )
            self.photons_per_subaperture = self.wfs_total_power / illuminated_count
        self._update_slope_noise()

    def _update_slope_noise(self) -> None:
        """Physical detector inputs do not depend on pupil quadrature."""
        self._slope_noise_std = self._noise_for_photons(self.photons_per_subaperture)

    def _noise_for_photons(self, photons):
        """Validate proposed noise before changing any runtime/configuration state."""
        cfg = self.config.wfs
        if not np.all(np.isfinite(photons)) or np.any(np.asarray(photons) <= 0):
            raise ValueError("WFS photon inputs must be finite and positive")
        if not np.isfinite(np.sum(photons)):
            raise ValueError("WFS total photon input must be finite")
        spot_size = F_NUMBER * cfg.wavelength
        pixels_per_subap = cfg.pixels_per_subaperture**2
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            photon_term = 1.0 / photons
            read_term = (pixels_per_subap * cfg.read_noise_e**2) / photons**2
            std = spot_size * np.sqrt(photon_term + read_term)
        if not np.all(np.isfinite(std)):
            raise ValueError("WFS photon inputs give nonfinite noise")
        # The geometric observation concatenates all x slopes, then all y slopes.
        return np.concatenate((std, std)) if cfg.channel_ids is not None else std

    def _build_science_path(self) -> None:
        wl = self.config.science_wavelength
        q = 4
        num_airy = 16
        self.science_focal_grid = hp.make_focal_grid(
            q,
            num_airy,
            pupil_diameter=self.config.aperture.diameter,
            reference_wavelength=wl,
            focal_length=1,
        )
        self.science_propagator = hp.FraunhoferPropagator(self.pupil_grid, self.science_focal_grid, focal_length=1)
        perfect_wf = hp.Wavefront(self.aperture, wl)
        perfect_wf.total_power = 1.0
        perfect_img = self.science_propagator.forward(perfect_wf).power
        self.science_psf_peak = float(np.asarray(perfect_img).max())
        self.lambda_over_d = wl / self.config.aperture.diameter

    # -- runtime --------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        self._build_atmosphere(seed)
        self.dm.actuators = np.zeros(self.dm.num_actuators)
        self._reset_noise(seed)

    def _reset_noise(self, seed: int | None) -> None:
        # Separate the noise stream from the atmosphere's integer layer seeds.
        self._noise_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))

    def _residual_phase_at(self, wavelength: float) -> NDArray[np.float64]:
        """Unwrapped physical phase for the phase-only, no-scintillation model."""
        return np.asarray(self.atmosphere.phase_for(wavelength) + self.dm.phase_for(wavelength))

    def _residual_wavefront(self, wavelength: float) -> hp.Wavefront:
        wf = hp.Wavefront(self.aperture, wavelength)
        wf.total_power = 1.0
        wf = self.atmosphere.forward(wf)
        wf = self.dm.forward(wf)
        return wf

    def slopes_from_phase(self, phase: NDArray[np.float64], add_noise: bool = True) -> NDArray[np.float64]:
        """Geometric-optics slopes (centroid-shift-equivalent, meters) from a
        pupil-plane phase map (radians, at the WFS wavelength)."""
        tilts = self._gradient_operator @ phase  # radians/meter, concatenated [b...; c...]
        slopes = tilts * self._tilt_to_shift
        if add_noise:
            slopes = slopes + self._noise_rng.normal(0.0, self._slope_noise_std, size=slopes.shape)
        return slopes

    def measure_wfs(self) -> NDArray[np.float64]:
        """Geometric-optics WFS slopes from the current atmosphere+DM state,
        with photon+read noise added."""
        return self.slopes_from_phase(self._residual_phase_at(self.config.wfs.wavelength), add_noise=True)

    def science_image(self) -> NDArray[np.float64]:
        wf = self._residual_wavefront(self.config.science_wavelength)
        img = self.science_propagator.forward(wf).power
        return np.asarray(img)

    def residual_phase(self) -> NDArray[np.float64]:
        return self._residual_phase_at(self.config.science_wavelength)

    def step(self, dm_command: NDArray[np.float64] | None = None, *, record_science_image: bool = False) -> StepResult:
        """Apply absolute actuator surface displacements in meters and measure.

        Slopes, physical residual phase, optional science image, time and copied
        applied command describe the same pre-advance state. The atmosphere advances only after measurement;
        the upstream queue determines which issued command reaches this frame.
        """
        if dm_command is not None:
            self.dm.actuators = np.array(dm_command, dtype=np.float64, copy=True)
        applied_command = np.array(self.dm.actuators, copy=True)
        slopes = self.measure_wfs()
        residual_phase = self.residual_phase()
        science_image = self.science_image().copy() if record_science_image else None
        t = self.atmosphere.t
        self.atmosphere.t = t + self.dt
        return StepResult(
            slopes=slopes,
            residual_phase=residual_phase,
            aperture_mask=self.aperture_mask,
            t=t,
            applied_command=applied_command,
            science_image=science_image,
        )

    @property
    def n_actuators(self) -> int:
        return int(self.dm.num_actuators)

    @property
    def n_slopes(self) -> int:
        return 2 * self.n_illuminated_subaps

    # -- failure injection ------------------------------------------------

    def inject_wind_switch(self, speed_multiplier: float = 1.0, direction_shift_deg: float = 90.0) -> None:
        """Rotate and/or rescale every layer's wind vector mid-run (S6-style
        scenario: the frozen-flow assumption a predictive controller relies on
        is suddenly violated)."""
        for layer in self.atmosphere.layers:
            speed = np.linalg.norm(layer.velocity) * speed_multiplier
            angle = np.arctan2(layer.velocity[1], layer.velocity[0]) + np.radians(direction_shift_deg)
            layer.velocity = speed * np.array([np.cos(angle), np.sin(angle)])

    def inject_magnitude_switch(self, new_magnitude: float) -> None:
        """Change the guide-star magnitude mid-run (S7-style scenario), updating
        WFS photon flux and noise accordingly."""
        if not np.isfinite(new_magnitude):
            raise ValueError("Guide-star magnitude must be finite")
        if self.config.wfs.channel_ids is not None:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                factor = np.power(10.0, -0.4 * (new_magnitude - self.config.guide_star.magnitude))
                photons = self.photons_per_subaperture * factor
            if not np.all(np.isfinite(photons)) or np.any(photons <= 0) or not np.isfinite(photons.sum()):
                raise ValueError("Magnitude change gives invalid physical channel photon inputs")
            noise = self._noise_for_photons(photons)
            config = replace(
                self.config,
                guide_star=replace(self.config.guide_star, magnitude=float(new_magnitude)),
                wfs=replace(self.config.wfs, photons_per_channel=tuple(float(value) for value in photons)),
            )
            self.photons_per_subaperture = photons
            self.wfs_total_power = float(photons.sum())
            self._slope_noise_std = noise
            self.config = config
            return
        aperture_area = (
            np.pi * (self.config.aperture.diameter / 2) ** 2 * (1 - self.config.aperture.central_obscuration_ratio**2)
        )
        try:
            total_power = magnitude_to_total_power(new_magnitude, aperture_area, self.config.loop.frame_rate_hz)
            if not np.isfinite(total_power) or total_power <= 0:
                raise ValueError("Magnitude change gives invalid legacy total photon input")
            photons = total_power / max(1, self.n_illuminated_subaps)
            noise = self._noise_for_photons(photons)
        except (OverflowError, ZeroDivisionError, FloatingPointError) as exc:
            raise ValueError("Magnitude change gives invalid legacy photon or noise inputs") from exc
        config = replace(self.config, guide_star=replace(self.config.guide_star, magnitude=float(new_magnitude)))
        self.wfs_total_power = total_power
        self.photons_per_subaperture = photons
        self._slope_noise_std = noise
        self.config = config
