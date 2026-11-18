"""Quadrature changes must not change the physical detector contract."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from noether_optics.bench.refsim.config import (
    ApertureConfig,
    AtmosphereConfig,
    AtmosphereLayerConfig,
    DMConfig,
    GuideStarConfig,
    LoopConfig,
    SystemConfig,
    WFSConfig,
    petscale_config,
)
from noether_optics.bench.refsim.system import F_NUMBER, RefSimSystem

CHANNELS = ((2, 1), (1, 2), (1, 1), (2, 2))
PHOTONS = (1e5, 2e5, 3e5, 4e5)


def physical_config(samples=32):
    base = petscale_config()
    return replace(
        base,
        sampling_contract_version="physical_v2",
        pupil_grid_size=samples,
        wfs=replace(
            base.wfs,
            num_subapertures=4,
            pixels_per_subaperture=8,
            channel_ids=CHANNELS,
            photons_per_channel=PHOTONS,
        ),
        dm=replace(base.dm, num_actuators_across=5),
    )


def legacy_config():
    base = petscale_config()
    return replace(
        base,
        wfs=replace(base.wfs, num_subapertures=4, pixels_per_subaperture=8),
        dm=replace(base.dm, num_actuators_across=5),
    )


@pytest.fixture
def static_atmosphere(monkeypatch):
    def build(system, seed):
        system.atmosphere = SimpleNamespace(t=0.0, phase_for=lambda wavelength: np.zeros(system.pupil_grid.size))

    monkeypatch.setattr(RefSimSystem, "_build_atmosphere", build)


def test_unversioned_serialized_128_remains_legacy():
    payload = json.loads(json.dumps(asdict(petscale_config())))
    del payload["sampling_contract_version"]
    # An actual old JSON payload predates these detector fields as well.
    payload["wfs"].pop("channel_ids")
    payload["wfs"].pop("photons_per_channel")
    payload["atmosphere"]["layers"] = tuple(AtmosphereLayerConfig(**item) for item in payload["atmosphere"]["layers"])
    for field, constructor in (
        ("aperture", ApertureConfig),
        ("atmosphere", AtmosphereConfig),
        ("wfs", WFSConfig),
        ("dm", DMConfig),
        ("loop", LoopConfig),
        ("guide_star", GuideStarConfig),
    ):
        payload[field] = constructor(**payload[field])
    config = SystemConfig(**payload)
    assert config.pupil_grid_size == 128
    assert config.sampling_contract_version == "legacy_v1"
    assert config.effective_pupil_grid_size == 230


def test_partial_or_unknown_migration_is_rejected():
    with pytest.raises(ValueError, match="frozen channel_ids"):
        replace(petscale_config(), sampling_contract_version="physical_v2")
    with pytest.raises(ValueError, match="require sampling_contract_version"):
        replace(physical_config(), sampling_contract_version="legacy_v1")
    with pytest.raises(ValueError, match="Unknown sampling"):
        replace(petscale_config(), sampling_contract_version="v3")


@pytest.mark.parametrize("samples", [0, 1, -1, True, 32.5])
def test_invalid_physical_grid_is_rejected(samples):
    with pytest.raises(ValueError, match="pupil_grid_size"):
        physical_config(samples)


@pytest.mark.parametrize(
    "channels, photons",
    [
        (((1, 1), (1, 1)), (1e5, 1e5)),
        (((4, 1),), (1e5,)),
        (((True, 1),), (1e5,)),
        (((1, 1),), (0.0,)),
        (((1, 1),), (float("nan"),)),
        (((1, 1),), (1e5, 2e5)),
    ],
)
def test_invalid_physical_channel_contract_is_rejected(channels, photons):
    with pytest.raises(ValueError):
        replace(physical_config().wfs, channel_ids=channels, photons_per_channel=photons)


def test_actual_quadrature_changes_without_changing_detector(static_atmosphere):
    systems = [RefSimSystem(physical_config(n), seed=19) for n in (32, 48, 64)]
    reference = systems[0]
    photons = np.asarray(PHOTONS)
    wavelength = reference.config.wfs.wavelength
    expected_noise = F_NUMBER * wavelength * np.sqrt(1 / photons + 64 * 5.0**2 / photons**2)
    for n, system in zip((32, 48, 64), systems, strict=True):
        assert tuple(system.pupil_grid.shape) == (n, n)
        assert system.wfs_channel_ids == CHANNELS
        assert system.n_slopes == 8
        assert system.config.wfs.pixels_per_subaperture == 8
        np.testing.assert_array_equal(system.photons_per_subaperture, photons)
        np.testing.assert_allclose(system._slope_noise_std, np.r_[expected_noise, expected_noise], rtol=1e-15)
        np.testing.assert_array_equal(system.science_focal_grid.points, reference.science_focal_grid.points)
        # Independent analytic plane, with unequal x/y slopes to catch axis ordering.
        phase = 2 * np.asarray(system.pupil_grid.x) - 3 * np.asarray(system.pupil_grid.y) + 17
        expected = np.r_[np.full(4, 2.0), np.full(4, -3.0)] * (8 * 2) * wavelength / (2 * np.pi)
        np.testing.assert_allclose(system.slopes_from_phase(phase, add_noise=False), expected, atol=1e-18, rtol=1e-12)


def test_underresolved_channel_is_not_silently_removed(static_atmosphere):
    with pytest.raises(ValueError, match="Under-resolved physical WFS channel"):
        RefSimSystem(physical_config(4), seed=19)


def test_brightness_preserves_channel_proportions_and_slope_order(static_atmosphere):
    system = RefSimSystem(physical_config(), seed=19)
    old_noise = system._slope_noise_std.copy()
    system.inject_magnitude_switch(2.5)
    np.testing.assert_allclose(system.photons_per_subaperture, np.asarray(PHOTONS) / 10, rtol=1e-15)
    expected = (
        F_NUMBER
        * system.config.wfs.wavelength
        * np.sqrt(1 / system.photons_per_subaperture + 64 * 25 / system.photons_per_subaperture**2)
    )
    np.testing.assert_allclose(system._slope_noise_std, np.r_[expected, expected], rtol=1e-15)
    assert np.all(system._slope_noise_std > old_noise)
    system.inject_magnitude_switch(0)
    np.testing.assert_allclose(system.photons_per_subaperture, PHOTONS, rtol=1e-15)
    before = system.photons_per_subaperture.copy()
    with pytest.raises(ValueError, match="finite"):
        system.inject_magnitude_switch(float("nan"))
    np.testing.assert_array_equal(system.photons_per_subaperture, before)
    assert system.config.guide_star.magnitude == 0


@pytest.mark.parametrize("config_factory", [physical_config, legacy_config], ids=["physical", "legacy"])
def test_brightness_change_is_local_and_serializable(static_atmosphere, config_factory):
    original = config_factory()
    first, second = RefSimSystem(original), RefSimSystem(original)
    initial_photons = np.asarray(first.photons_per_subaperture).copy()
    first.inject_magnitude_switch(2.5)
    assert original.guide_star.magnitude == 0
    np.testing.assert_array_equal(second.photons_per_subaperture, initial_photons)
    second.inject_magnitude_switch(2.5)
    np.testing.assert_allclose(second.photons_per_subaperture, initial_photons / 10)
    reconstructed = RefSimSystem(first.config)
    np.testing.assert_array_equal(reconstructed.photons_per_subaperture, first.photons_per_subaperture)
    first.inject_magnitude_switch(5.0)
    np.testing.assert_allclose(first.photons_per_subaperture, initial_photons / 100)
    np.testing.assert_allclose(second.photons_per_subaperture, initial_photons / 10)


@pytest.mark.parametrize(
    "config_factory,magnitude",
    [
        pytest.param(factory, magnitude, id=f"{contract}-{label}")
        for contract, factory in (("physical", physical_config), ("legacy", legacy_config))
        for label, magnitude in (
            ("infinite", float("inf")),
            ("nan", float("nan")),
            ("overflow", -1e300),
            ("underflow", 1e300),
            ("noise-overflow", 420.0),
        )
    ],
)
def test_invalid_brightness_has_no_partial_state_change(static_atmosphere, config_factory, magnitude):
    system = RefSimSystem(config_factory())
    config, total = system.config, system.wfs_total_power
    initial_magnitude = config.guide_star.magnitude
    photons, noise = np.asarray(system.photons_per_subaperture).copy(), np.asarray(system._slope_noise_std).copy()
    with pytest.raises(ValueError):
        system.inject_magnitude_switch(magnitude)
    assert system.config is config
    assert system.config.guide_star.magnitude == initial_magnitude
    assert system.wfs_total_power == total
    np.testing.assert_array_equal(system.photons_per_subaperture, photons)
    np.testing.assert_array_equal(system._slope_noise_std, noise)


def test_legacy_brightness_switch_matches_fresh_photometry(static_atmosphere):
    original = legacy_config()
    switched = RefSimSystem(original)
    for magnitude in (2.5, 9.0, 10.0, -2.0, 0.0):
        fresh_config = replace(original, guide_star=replace(original.guide_star, magnitude=magnitude))
        fresh = RefSimSystem(fresh_config)
        switched.inject_magnitude_switch(magnitude)
        assert switched.wfs_total_power == fresh.wfs_total_power
        assert switched.photons_per_subaperture == fresh.photons_per_subaperture
        np.testing.assert_array_equal(switched._slope_noise_std, fresh._slope_noise_std)
