"""System configuration dataclasses for the geometric reference simulator.

The petscale and xaoscale configurations follow the parameter sets discussed
in Nousiainen et al. 2021 (arXiv:2104.13685) and 2022 (A&A 664:A71).
Guide-star photon flux is computed from the photometric zero point and
bandwidth in photometry.py rather than copied from a tabulated flux.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral


@dataclass(frozen=True)
class AtmosphereLayerConfig:
    fraction: float  # fraction of total Cn^2
    wind_speed: float  # m/s
    wind_direction_deg: float  # degrees, 0 = +x
    altitude: float  # m


@dataclass(frozen=True)
class AtmosphereConfig:
    r0: float  # Fried parameter at reference wavelength, m
    r0_wavelength: float  # m
    outer_scale: float  # L0, m
    layers: tuple[AtmosphereLayerConfig, ...]

    def __post_init__(self) -> None:
        total = sum(layer.fraction for layer in self.layers)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"layer fractions must sum to 1.0, got {total}")


@dataclass(frozen=True)
class ApertureConfig:
    diameter: float  # m
    central_obscuration_ratio: float


@dataclass(frozen=True)
class WFSConfig:
    kind: str  # "shs" or "pyramid"
    num_subapertures: int  # per side (shs) or per pupil diameter (pyramid)
    pixels_per_subaperture: int  # shs only
    read_noise_e: float
    wavelength: float  # m
    modulated: bool = False  # pyramid only
    # Physical row/column identities and photons per exposure, independent of
    # pupil quadrature. Only valid with SystemConfig physical_v2 sampling.
    channel_ids: tuple[tuple[int, int], ...] | None = None
    photons_per_channel: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.channel_ids is None and self.photons_per_channel is None:
            return
        if self.channel_ids is None or self.photons_per_channel is None:
            raise ValueError("Explicit channels require both channel_ids and photons_per_channel")
        for name, value in (
            ("num_subapertures", self.num_subapertures),
            ("pixels_per_subaperture", self.pixels_per_subaperture),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        channels = []
        for pair in self.channel_ids:
            if not isinstance(pair, tuple | list) or len(pair) != 2:
                raise ValueError("Each channel identity is a physical (row, column) pair")
            if any(
                isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= v < self.num_subapertures for v in pair
            ):
                raise ValueError("Channel indices must lie inside the physical subaperture layout")
            channels.append(tuple(int(v) for v in pair))
        if not channels or len(channels) != len(set(channels)):
            raise ValueError("Explicit channel identities must be nonempty and unique")
        photons = tuple(float(value) for value in self.photons_per_channel)
        if len(photons) != len(channels) or any(not isfinite(value) or value <= 0 for value in photons):
            raise ValueError("Each declared channel requires finite positive photons per exposure")
        if not isfinite(self.read_noise_e) or self.read_noise_e < 0:
            raise ValueError("Detector read noise must be finite and nonnegative")
        if not isfinite(self.wavelength) or self.wavelength <= 0:
            raise ValueError("Wavelength must be finite and positive")
        object.__setattr__(self, "channel_ids", tuple(channels))
        object.__setattr__(self, "photons_per_channel", photons)


@dataclass(frozen=True)
class DMConfig:
    num_actuators_across: int
    coupling: float  # fractional influence at neighboring actuator


@dataclass(frozen=True)
class LoopConfig:
    frame_rate_hz: float
    delay_frames: int

    def __post_init__(self) -> None:
        validate_delay_frames(self.delay_frames)


def validate_delay_frames(delay_frames: int) -> None:
    """A measured frame can first affect a subsequent frame in this runner."""
    if isinstance(delay_frames, bool) or not isinstance(delay_frames, Integral) or delay_frames < 1:
        raise ValueError("delay_frames must be an integer of at least one frame")


@dataclass(frozen=True)
class GuideStarConfig:
    magnitude: float


@dataclass(frozen=True)
class SystemConfig:
    name: str
    aperture: ApertureConfig
    atmosphere: AtmosphereConfig
    wfs: WFSConfig
    dm: DMConfig
    loop: LoopConfig
    guide_star: GuideStarConfig
    science_wavelength: float  # m
    pupil_grid_size: int = 128
    sampling_contract_version: str = "legacy_v1"

    def __post_init__(self) -> None:
        if self.sampling_contract_version not in {"legacy_v1", "physical_v2"}:
            raise ValueError("Unknown sampling contract; use legacy_v1 or physical_v2")
        explicit = self.wfs.channel_ids is not None or self.wfs.photons_per_channel is not None
        if self.sampling_contract_version == "legacy_v1":
            if explicit:
                raise ValueError("Explicit detector fields require sampling_contract_version='physical_v2'")
        else:
            if not explicit:
                raise ValueError("physical_v2 requires frozen channel_ids and photons_per_channel")
            value = self.pupil_grid_size
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 2:
                raise ValueError("physical_v2 pupil_grid_size must be an integer of at least two")

    @property
    def effective_pupil_grid_size(self) -> int:
        """Legacy serialized values were ignored; migration is explicitly versioned."""
        if self.sampling_contract_version == "legacy_v1":
            return self.wfs.num_subapertures * self.wfs.pixels_per_subaperture
        return int(self.pupil_grid_size)


def _petscale_atmosphere() -> AtmosphereConfig:
    return AtmosphereConfig(
        r0=0.15,
        r0_wavelength=550e-9,
        outer_scale=30.0,
        layers=(
            AtmosphereLayerConfig(fraction=0.70, wind_speed=15.0, wind_direction_deg=0.0, altitude=0.0),
            AtmosphereLayerConfig(fraction=0.25, wind_speed=3.0, wind_direction_deg=45.0, altitude=4000.0),
            AtmosphereLayerConfig(fraction=0.05, wind_speed=7.5, wind_direction_deg=90.0, altitude=10000.0),
        ),
    )


def petscale_config(magnitude: float = 0.0, pupil_grid_size: int = 128) -> SystemConfig:
    """8 m, 23x23 SHS, 24x24 DM, 500 Hz, 2-frame delay (Nousiainen 2021)."""
    return SystemConfig(
        name="petscale",
        aperture=ApertureConfig(diameter=8.0, central_obscuration_ratio=0.14),
        atmosphere=_petscale_atmosphere(),
        wfs=WFSConfig(
            kind="shs",
            num_subapertures=23,
            pixels_per_subaperture=10,
            read_noise_e=5.0,
            wavelength=551e-9,
        ),
        dm=DMConfig(num_actuators_across=24, coupling=0.45),
        loop=LoopConfig(frame_rate_hz=500.0, delay_frames=2),
        guide_star=GuideStarConfig(magnitude=magnitude),
        science_wavelength=1.65e-6,
        pupil_grid_size=pupil_grid_size,
    )


def xaoscale_config(magnitude: float = 0.0, pupil_grid_size: int = 128) -> SystemConfig:
    """8 m, unmodulated PWFS, 41x41 actuators, 1 kHz (PO4AO Sect. 6). Slow; nightly."""
    return SystemConfig(
        name="xaoscale",
        aperture=ApertureConfig(diameter=8.0, central_obscuration_ratio=0.14),
        atmosphere=AtmosphereConfig(
            r0=0.16,
            r0_wavelength=550e-9,
            outer_scale=30.0,
            layers=_petscale_atmosphere().layers,
        ),
        wfs=WFSConfig(
            kind="pyramid",
            num_subapertures=41,
            pixels_per_subaperture=6,
            read_noise_e=5.0,
            wavelength=551e-9,
            modulated=False,
        ),
        dm=DMConfig(num_actuators_across=41, coupling=0.45),
        loop=LoopConfig(frame_rate_hz=1000.0, delay_frames=2),
        guide_star=GuideStarConfig(magnitude=magnitude),
        science_wavelength=1.65e-6,
        pupil_grid_size=pupil_grid_size,
    )


@dataclass(frozen=True)
class MisregistrationConfig:
    """Fractional-subaperture shift of the WFS grid relative to the DM grid."""

    shift_fraction: float = 0.0  # e.g. 0.14 or 0.28
    shift_angle_deg: float = 0.0


MR_14 = MisregistrationConfig(shift_fraction=0.14, shift_angle_deg=0.0)
MR_28 = MisregistrationConfig(shift_fraction=0.28, shift_angle_deg=0.0)
NO_MR = MisregistrationConfig()
