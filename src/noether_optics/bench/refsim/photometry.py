"""Guide-star photon flux from magnitude, aperture and WFS bandwidth.

Flux is computed from an assumed V-band zero point and WFS-arm bandwidth.
It is not a transcription of the individual guide-star fluxes tabulated
by Nousiainen et al. (2021).
"""

from __future__ import annotations

import numpy as np

# Standard V-band zero-point flux for a magnitude-0 star (photons / s / m^2 / nm).
# See e.g. Bessell 1979; commonly used value in AO simulation codes (YAO, COMPASS).
V_BAND_ZERO_POINT_PHOTON_FLUX = 1.0e10  # photons / s / m^2 / nm at V=0

# Effective bandwidth of a typical visible WFS arm.
DEFAULT_WFS_BANDWIDTH_NM = 300.0


def guide_star_photon_flux_per_m2_per_s(
    magnitude: float,
    bandwidth_nm: float = DEFAULT_WFS_BANDWIDTH_NM,
    zero_point: float = V_BAND_ZERO_POINT_PHOTON_FLUX,
) -> float:
    """Photon flux (photons / s / m^2) incident on the telescope aperture."""
    return zero_point * bandwidth_nm * 10.0 ** (-magnitude / 2.5)


def photons_per_frame(
    magnitude: float,
    aperture_area_m2: float,
    frame_rate_hz: float,
    throughput: float = 0.5,
    bandwidth_nm: float = DEFAULT_WFS_BANDWIDTH_NM,
) -> float:
    """Total photons collected by the WFS arm in one frame."""
    flux = guide_star_photon_flux_per_m2_per_s(magnitude, bandwidth_nm)
    return float(flux * aperture_area_m2 * throughput / frame_rate_hz)


def magnitude_to_total_power(
    magnitude: float,
    aperture_area_m2: float,
    frame_rate_hz: float,
    throughput: float = 0.5,
    bandwidth_nm: float = DEFAULT_WFS_BANDWIDTH_NM,
) -> float:
    """Total power (in HCIPy wavefront.total_power units, i.e. photon count per
    frame) to assign to a Wavefront so that propagating it and reading out with
    an exposure of 1/frame_rate_hz yields the correct photon flux."""
    return float(np.asarray(photons_per_frame(magnitude, aperture_area_m2, frame_rate_hz, throughput, bandwidth_nm)))
