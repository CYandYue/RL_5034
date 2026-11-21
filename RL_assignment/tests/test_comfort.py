import pytest

from hvac_env import HVAC2R2CEnv
from hvac_env.comfort import comfort_deviation, cvr_pct, degree_hours, electricity_kwh


def test_hand_calculated_trajectory():
    distances = [comfort_deviation(t, 20, 23.5) for t in [19, 20, 23.5, 25]]
    assert distances == [1, 0, 0, 1.5]
    assert cvr_pct(distances) == 50
    assert degree_hours(distances, 0.25) == 0.625
    assert electricity_kwh([0, 1, 0.5, 0.5], 0.25) == 0.5
    assert cvr_pct([0, 1e-8, 2e-8]) == pytest.approx(100 / 3)


def test_empty_and_invalid_series():
    with pytest.raises(ValueError, match="at least one"):
        cvr_pct([])
    assert degree_hours([], 0.25) == electricity_kwh([], 0.25) == 0
    with pytest.raises(ValueError, match="nonnegative"):
        electricity_kwh([-0.1], 0.25)
    with pytest.raises(ValueError, match="finite"):
        degree_hours([float("nan")], 0.25)
    with pytest.raises(ValueError, match="low"):
        comfort_deviation(20, 23, 18)


def test_reward_components_at_known_equilibrium():
    # Equal node and outdoor temperatures; cooling exactly removes the 0.5 kW gain.
    env = HVAC2R2CEnv({
        "weather": {"cold": {"mean_c": 18, "amplitude_c": 0}},
        "gains": {"base_kw": 0.5},
        "reward": {"w_c": 2, "w_e": 0.3},
    })
    env.reset(options=dict(scenario="cold", indoor_init_c=18, envelope_init_c=18, weather_delta_c=0))
    _, reward, _, _, info = env.step([-0.5])
    assert info["comfort_violation"] is True
    assert info["comfort_deviation_c"] == pytest.approx(2)
    assert info["reward_comfort"] == pytest.approx(-1.0)
    assert info["reward_energy"] == pytest.approx(-0.0125)
    assert info["reward_total"] == reward == pytest.approx(-1.0125)
