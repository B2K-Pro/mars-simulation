import math

import pytest

from mars import physics
from mars.experiment import Experiment
from mars.parameters import Parameters
from mars.reference import load_local

REFS = load_local()


def params(**overrides):
    return Parameters(REFS, overrides)


def test_gravity_matches_nasa_mean():
    assert physics.gravity(params()) == pytest.approx(3.73, rel=0.01)


def test_gravity_decreases_with_altitude():
    p = params()
    assert physics.gravity(p, 20000) < physics.gravity(p, 0)


def test_scale_height_matches_nasa():
    assert physics.scale_height(params()) / 1e3 == pytest.approx(11.0, rel=0.02)


def test_surface_density_within_nasa_uncertainty():
    d = float(physics.density(params()))
    ref = REFS["surface_density"]
    assert ref.min_value <= d <= ref.max_value


def test_equilibrium_temperature_matches_nasa_blackbody():
    assert physics.equilibrium_temperature(params()) == pytest.approx(209.8, abs=0.5)


def test_pressure_at_zero_is_surface_pressure_and_decreases():
    p = params()
    assert float(physics.pressure(p, 0)) == pytest.approx(636.0)
    assert physics.pressure(p, 10000) < physics.pressure(p, 0) < physics.pressure(p, -5000)


def test_colder_atmosphere_has_smaller_scale_height():
    assert physics.scale_height(params(mean_temperature=180)) < physics.scale_height(params())


def test_declination_at_solstice_equals_obliquity():
    dec = physics.solar_declination(params(), 90.0)
    assert math.degrees(dec) == pytest.approx(25.19, abs=1e-6)


def test_equinox_noon_at_equator_sun_overhead():
    el = float(physics.solar_elevation(params(), 0.0, 12.0, 0.0))
    assert math.degrees(el) == pytest.approx(90.0, abs=1e-6)


def test_no_sun_at_midnight_on_equator():
    assert float(physics.insolation_toa(params(), 0.0, 0.0, 0.0)) == 0.0


def test_polar_night_has_zero_daily_insolation():
    assert physics.daily_mean_insolation(params(), -80.0, 90.0) == 0.0


def test_levels_and_validity():
    assert Experiment("ref", REFS).validity() == "reference"
    assert Experiment("incert", REFS, overrides={"surface_pressure_mean": 800}).validity() == "within_uncertainty"
    exp = Experiment("sandbox", REFS, overrides={"mean_temperature": 300})
    assert exp.validity() == "sandbox"
    assert exp.levels()["mean_temperature"] == "sandbox"


def test_unknown_parameter_rejected():
    with pytest.raises(KeyError):
        params(inexistant=1)


def test_seasonal_factor_min_max_and_mean():
    p = params()
    assert physics.seasonal_pressure_factor(p, 150.0) == pytest.approx(0.875)
    assert physics.seasonal_pressure_factor(p, 260.0) == pytest.approx(1.125)
    mean = sum(physics.seasonal_pressure_factor(p, ls) for ls in range(360)) / 360.0
    assert mean == pytest.approx(1.0, abs=0.01)


def test_seasonal_pressure_changes_with_ls():
    p = params()
    assert physics.pressure(p, 0.0, 150.0) < physics.pressure(p, 0.0) < physics.pressure(p, 0.0, 260.0)


def test_new_reference_keys_levels():
    assert Experiment("t", REFS, overrides={"seasonal_pressure_ptp_fraction": 0.30}).validity() == "within_uncertainty"


def test_surface_temperature_is_periodic_and_physical():
    times, ts = physics.surface_temperature_cycle(params(), 0.0, 0.0)
    assert ts.min() > 100 and ts.max() < 350
    assert ts.max() > ts.min()
    assert times[ts.argmax()] == pytest.approx(13.0, abs=2.0)   # maximum en début d'après-midi


def test_higher_thermal_inertia_gives_smaller_temperature_swing():
    _, low = physics.surface_temperature_cycle(params(thermal_inertia=100.0), 0.0, 0.0)
    _, high = physics.surface_temperature_cycle(params(thermal_inertia=1000.0), 0.0, 0.0)
    assert (high.max() - high.min()) < (low.max() - low.min())


def test_mean_surface_temperature_close_to_equilibrium_value():
    _, ts = physics.surface_temperature_cycle(params(), 0.0, 0.0)
    flux_mean = physics.daily_mean_insolation(params(), 0.0, 0.0) * (1 - 0.25)
    t_eq = (flux_mean / physics.SIGMA) ** 0.25
    assert ts.mean() == pytest.approx(t_eq, abs=15.0)


def test_thermal_inertia_levels():
    assert Experiment("t", REFS, overrides={"thermal_inertia": 400.0}).validity() == "within_uncertainty"
    assert Experiment("t", REFS, overrides={"thermal_inertia": 50.0}).validity() == "sandbox"
