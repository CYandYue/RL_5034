from math import pi, sin

import pytest

from hvac_env.weather import internal_gain, outdoor_temperature


@pytest.mark.parametrize("name, mean, amplitude, low, high", [
    ("cold", 5, 6, 20, 23.5), ("hot", 31, 6, 23, 26), ("transition", 19, 8, 20, 23.5),
])
def test_scenario_extrema_and_targets(config, name, mean, amplitude, low, high):
    scenario = config.weather.scenario(name)
    assert (scenario.comfort_low_c, scenario.comfort_high_c) == (low, high)
    for hour, expected in ((3, mean - amplitude), (9, mean), (15, mean + amplitude)):
        for day in (0, 1, 6):
            temperature = outdoor_temperature(hour + 24 * day, scenario, 1.25, config.weather.phase_h)
            assert temperature == pytest.approx(expected + 1.25)


@pytest.mark.parametrize("hour, expected_kw", [
    (0, 0.05), (6, 0.05), (7.75, 0.05 + 0.15 * sin(pi * 1.75 / 12)),
    (8, 0.275), (12, 0.35), (18, 0.05), (23.75, 0.05),
])
def test_gain_schedule_edges_and_daily_repeat(config, hour, expected_kw):
    assert internal_gain(hour, config.gains) == pytest.approx(expected_kw)
    assert internal_gain(hour + 24, config.gains) == internal_gain(hour, config.gains)
