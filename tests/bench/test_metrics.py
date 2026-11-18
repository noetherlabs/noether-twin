import numpy as np

from noether_optics.bench.metrics import (
    contrast_curve,
    marechal_strehl,
    residual_rms,
    reward_from_observation,
    strehl_from_phase,
    temporal_speckle_variance,
    wdh_sector_metric,
)


def test_residual_rms_flat_is_zero():
    phase = np.zeros((10, 10))
    mask = np.ones((10, 10))
    assert residual_rms(phase, mask) == 0.0


def test_marechal_strehl_matches_injected_rms():
    rng = np.random.default_rng(0)
    mask = np.ones((64, 64))
    raw = rng.normal(size=(64, 64))
    raw -= raw.mean()
    raw /= residual_rms(raw, mask)  # unit RMS

    for target_sigma in [0.05, 0.2, 0.5, 1.0]:
        phase = raw * target_sigma
        measured_sigma = residual_rms(phase, mask)
        assert abs(measured_sigma - target_sigma) / target_sigma < 0.01

        expected_strehl = np.exp(-(target_sigma**2))
        assert abs(strehl_from_phase(phase, mask) - expected_strehl) < 1e-12
        assert abs(marechal_strehl(target_sigma) - expected_strehl) < 1e-12


def test_strehl_is_one_for_zero_rms():
    assert marechal_strehl(0.0) == 1.0


def test_reward_from_observation():
    obs = np.array([3.0, 4.0])
    assert reward_from_observation(obs) == -25.0


def test_contrast_curve_uniform_image():
    n = 41
    coords = np.linspace(-5, 5, n)
    x, y = np.meshgrid(coords, coords)
    image = np.ones_like(x)
    centers, contrast = contrast_curve(image, x, y, peak_normalization=2.0, bin_width=1.0)
    valid = ~np.isnan(contrast)
    assert np.allclose(contrast[valid], 0.5)


def test_wdh_sector_metric_detects_asymmetry():
    n = 81
    coords = np.linspace(-10, 10, n)
    x, y = np.meshgrid(coords, coords)
    theta = np.degrees(np.arctan2(y, x))
    # bright wedge along wind (0 deg), dim elsewhere
    image = np.where(np.abs(theta) < 20, 1.0, 0.01)
    result = wdh_sector_metric(image, x, y, peak_normalization=1.0, wind_angle_deg=0.0, r_min=2.0, r_max=8.0)
    assert result["contrast_along_wind"] > result["contrast_orthogonal"]
    assert result["asymmetry_dex"] > 0.3


def test_temporal_speckle_variance_shape():
    stack = np.random.default_rng(1).normal(size=(20, 8, 8))
    var = temporal_speckle_variance(stack)
    assert var.shape == (8, 8)
    assert np.all(var >= 0)
