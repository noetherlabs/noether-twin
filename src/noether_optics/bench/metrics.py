"""Optical residual, image-quality and temporal-speckle metrics.

Includes residual RMS and modal reward, a Marechal Strehl approximation,
coronagraphic contrast by separation, wind-driven-halo sector diagnostics,
and post-coronagraphic temporal speckle variance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def residual_rms(
    phase: NDArray[np.float64],
    aperture_mask: NDArray[np.float64],
) -> float:
    """Aperture-weighted RMS of a pupil-plane phase map, in radians."""
    weights = aperture_mask.astype(np.float64)
    total_weight = weights.sum()
    if total_weight <= 0:
        raise ValueError("aperture_mask has no illuminated points")
    mean = np.sum(phase * weights) / total_weight
    variance = np.sum(weights * (phase - mean) ** 2) / total_weight
    return float(np.sqrt(variance))


def marechal_strehl(rms_rad: float) -> float:
    """Marechal approximation: S = exp(-sigma^2), sigma in radians."""
    return float(np.exp(-(rms_rad**2)))


def strehl_from_phase(
    phase: NDArray[np.float64],
    aperture_mask: NDArray[np.float64],
) -> float:
    return marechal_strehl(residual_rms(phase, aperture_mask))


def reward_from_observation(observation: NDArray[np.float64]) -> float:
    """PO4AO-style reward: r = -||o||^2, o = C . w (modal residual)."""
    return float(-np.sum(observation**2))


def _radial_bins(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    bin_width: float,
    r_max: float,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    r = np.hypot(x, y)
    edges = np.arange(0.0, r_max + bin_width, bin_width)
    bin_index = np.digitize(r.ravel(), edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, bin_index


def contrast_curve(
    image: NDArray[np.float64],
    x_lod: NDArray[np.float64],
    y_lod: NDArray[np.float64],
    peak_normalization: float,
    bin_width: float = 1.0,
    r_max: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Azimuthally-averaged raw-PSF contrast vs separation (Cavarroc 2006 convention):
    the coronagraphic residual intensity normalized by the peak of the
    unocculted (perfect, no-aberration) PSF of the same system.

    image: coronagraphic focal-plane intensity (2D array).
    x_lod, y_lod: coordinate grids in units of lambda/D, same shape as image.
    peak_normalization: peak intensity of the unaberrated, non-coronagraphic PSF.
    Returns (separations_lod, contrast).
    """
    if r_max is None:
        r_max = float(np.hypot(x_lod, y_lod).max())
    centers, bin_index = _radial_bins(x_lod, y_lod, bin_width, r_max)
    flat_image = image.ravel()
    contrast = np.full(centers.shape, np.nan)
    for i in range(len(centers)):
        sel = bin_index == i
        if np.any(sel):
            contrast[i] = np.mean(flat_image[sel]) / peak_normalization
    return centers, contrast


def wdh_sector_metric(
    image: NDArray[np.float64],
    x_lod: NDArray[np.float64],
    y_lod: NDArray[np.float64],
    peak_normalization: float,
    wind_angle_deg: float,
    sector_deg: float = 30.0,
    r_min: float = 2.0,
    r_max: float = 8.0,
) -> dict[str, float]:
    """Wind-driven-halo asymmetry: mean contrast in a +/-sector_deg wedge along
    the dominant wind direction vs the orthogonal wedge, over [r_min, r_max]
    (units of lambda/D). Returns both means and the asymmetry in dex
    (log10 of the ratio)."""
    r = np.hypot(x_lod, y_lod)
    theta = np.degrees(np.arctan2(y_lod, x_lod))

    def _angular_distance(a: NDArray[np.float64], b: float) -> NDArray[np.float64]:
        d = np.abs(a - b) % 360.0
        return np.minimum(d, 360.0 - d)

    annulus = (r >= r_min) & (r <= r_max)
    along_mask = annulus & (_angular_distance(theta, wind_angle_deg) <= sector_deg)
    ortho_mask = annulus & (_angular_distance(theta, wind_angle_deg + 90.0) <= sector_deg)

    contrast_along = float(np.mean(image[along_mask]) / peak_normalization) if np.any(along_mask) else float("nan")
    contrast_ortho = float(np.mean(image[ortho_mask]) / peak_normalization) if np.any(ortho_mask) else float("nan")
    asymmetry_dex = float(np.log10(contrast_along / contrast_ortho)) if contrast_ortho > 0 else float("nan")

    return {
        "contrast_along_wind": contrast_along,
        "contrast_orthogonal": contrast_ortho,
        "asymmetry_dex": asymmetry_dex,
    }


def temporal_speckle_variance(image_stack: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pixel-wise temporal variance of post-coronagraphic images.
    image_stack: (n_frames, ny, nx)."""
    return np.var(image_stack, axis=0)
