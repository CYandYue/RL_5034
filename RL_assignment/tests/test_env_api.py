import numpy as np
import pytest
from gymnasium.error import ResetNeeded
from gymnasium.utils.env_checker import check_env

from hvac_env import OBSERVATION_FIELDS, HVAC2R2CEnv


def test_gymnasium_checker():
    check_env(HVAC2R2CEnv())


def test_public_info_keys_for_downstream_metrics():
    env = HVAC2R2CEnv()
    _, initial = env.reset(seed=0)
    common = {"scenario", "weather_delta_c", "step_index", "indoor_temp_c",
              "envelope_temp_c", "comfort_low_c", "comfort_high_c"}
    assert common | {"time_h", "outdoor_temp_c", "internal_gain_kw"} <= initial.keys()
    _, _, _, _, step = env.step([0])
    interval_fields = {
        "time_h_start", "time_h_end", "indoor_temp_c_before", "outdoor_temp_c_applied",
        "internal_gain_kw_applied", "action_requested", "action_applied", "q_hvac_kw_thermal",
        "heating_electric_kw", "cooling_electric_kw", "p_electric_kw", "energy_kwh_step",
        "comfort_deviation_c", "comfort_violation", "reward_comfort", "reward_energy", "reward_total",
    }
    assert common | interval_fields <= step.keys()


@pytest.mark.parametrize("name, low, high, one_hot", [
    ("cold", 20, 23.5, [1, 0, 0]), ("hot", 23, 26, [0, 1, 0]),
    ("transition", 20, 23.5, [0, 0, 1]),
])
def test_default_full_episode(name, low, high, one_hot):
    env = HVAC2R2CEnv()
    assert env.action_space.shape == (1,) and env.action_space.dtype == np.float32
    np.testing.assert_array_equal(env.action_space.low, [-1])
    np.testing.assert_array_equal(env.action_space.high, [1])
    assert len(OBSERVATION_FIELDS) == 11
    obs, initial = env.reset(seed=12, options={"scenario": name})
    assert env.observation_space.contains(obs)
    assert initial["time_h"] == 0 and initial["step_index"] == 0
    for index in range(1, 673):
        obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        assert env.observation_space.contains(obs)
        assert obs.shape == (11,) and obs.dtype == np.float32 and np.isfinite(obs).all()
        np.testing.assert_array_equal(obs[4:6], [low, high])
        np.testing.assert_array_equal(obs[8:], one_hot)
        assert info["scenario"] == name and info["step_index"] == index
        assert info["time_h_start"] == (index - 1) * 0.25
        assert info["time_h_end"] == index * 0.25
        assert reward == info["reward_total"] == info["reward_comfort"] + info["reward_energy"]
        assert isinstance(reward, float)
        assert terminated is False and truncated is (index == 672)
    with pytest.raises(ResetNeeded, match="reset"):
        env.step([0])
    obs, info = env.reset(seed=12)
    assert info["step_index"] == 0 and env.observation_space.contains(obs)


def test_weather_at_each_end_of_the_occupied_start_interval():
    env = HVAC2R2CEnv({"dt_h": 1.0})
    obs, last = env.reset(options=dict(scenario="cold", weather_delta_c=1, indoor_init_c=21))
    for _ in range(7):
        obs, _, _, _, last = env.step([0])
    before = [last["indoor_temp_c"], last["envelope_temp_c"]]
    new_obs, _, _, _, info = env.step([0])

    outdoor_at_7 = 5 + 1 + 6 * np.sin(2 * np.pi * (7 - 9) / 24)
    gain_at_7 = 0.05 + 0.15 * np.sin(np.pi / 12)
    outdoor_at_8 = 5 + 1 + 6 * np.sin(2 * np.pi * (8 - 9) / 24)
    gain_at_8 = 0.05 + 0.15 + 0.15 * np.sin(np.pi * 2 / 12)
    assert info["time_h_start"] == 7 and info["time_h_end"] == 8
    assert info["indoor_temp_c_before"] == before[0]
    assert info["outdoor_temp_c_applied"] == pytest.approx(outdoor_at_7)
    assert info["internal_gain_kw_applied"] == pytest.approx(gain_at_7)
    np.testing.assert_allclose(obs[2:4], [outdoor_at_7, gain_at_7], rtol=1e-6)
    expected = env.thermal.advance(before, [0, gain_at_7, outdoor_at_7])
    np.testing.assert_allclose([info["indoor_temp_c"], info["envelope_temp_c"]], expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(new_obs[:4], [*expected, outdoor_at_8, gain_at_8], rtol=1e-6)
    np.testing.assert_allclose(new_obs[6:8], [np.sin(2*np.pi/3), np.cos(2*np.pi/3)], rtol=1e-6)


def test_reward_uses_post_step_indoor_at_comfort_crossing():
    env = HVAC2R2CEnv()
    env.reset(options=dict(scenario="cold", indoor_init_c=20, envelope_init_c=20, weather_delta_c=0))
    _, reward, _, _, info = env.step([-0.5])
    distance = 20 - info["indoor_temp_c"]
    assert info["indoor_temp_c_before"] == 20 and distance > 0
    assert info["comfort_violation"] is True and info["comfort_deviation_c"] == distance
    assert info["reward_comfort"] == pytest.approx(-0.25 * distance)
    assert reward == pytest.approx(-0.25 * (distance + 0.1 * 0.5 / 3))


@pytest.mark.parametrize("action", [0.5, [], [0, 1], [[0]], [float("nan")], [float("inf")], ["1"]])
def test_invalid_action_does_not_consume_a_step(action):
    env = HVAC2R2CEnv()
    env.reset(seed=42)
    with pytest.raises(ValueError, match="action"):
        env.step(action)
    assert env.step([0])[-1]["step_index"] == 1


@pytest.mark.parametrize("requested", [-1.7, 2.1])
def test_action_clipping_is_visible_and_applied(requested):
    env = HVAC2R2CEnv()
    env.reset(seed=123)
    clipped = env.step([requested])
    env.reset(seed=123)
    limit = float(np.sign(requested))
    at_limit = env.step([limit])
    np.testing.assert_array_equal(clipped[0], at_limit[0])
    assert clipped[1] == at_limit[1]
    assert clipped[-1]["action_requested"] == requested
    assert clipped[-1]["action_applied"] == limit


@pytest.mark.parametrize("options, message", [
    ({"typo": 1}, "unknown"), ({"scenario": "rain"}, "scenario"),
    ({"weather_delta_c": float("inf")}, "weather_delta_c"),
    ({"envelope_init_c": float("nan")}, "envelope_init_c"),
    ({"indoor_init_c": 1e40}, "float32"),
])
def test_reset_rejects_invalid_options(options, message):
    with pytest.raises(ValueError, match=message):
        HVAC2R2CEnv().reset(options=options)


def test_step_before_reset():
    with pytest.raises(ResetNeeded, match="reset"):
        HVAC2R2CEnv().step([0])
