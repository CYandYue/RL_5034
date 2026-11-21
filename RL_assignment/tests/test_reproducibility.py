import numpy as np
import pytest

from hvac_env import HVAC2R2CEnv


def rollout(env, seed, options=None):
    obs, info = env.reset(seed=seed, options=options)
    observations = [obs]
    records = [info]
    for action in [-0.5, 0.25, 0, 1] * 24:
        obs, reward, terminated, truncated, info = env.step([action])
        observations.append(obs)
        records.append((reward, terminated, truncated, info))
    return np.stack(observations), records


def test_seed_replays_the_whole_trajectory():
    env = HVAC2R2CEnv()
    first, records = rollout(env, 123)
    for other in (env, HVAC2R2CEnv()):
        repeated, replay = rollout(other, 123)
        np.testing.assert_array_equal(first, repeated)
        assert records == replay


def test_fixed_conditions_override_the_seed():
    options = dict(scenario="hot", weather_delta_c=1.0, indoor_init_c=24, envelope_init_c=23)
    env = HVAC2R2CEnv()
    a, records_a = rollout(env, 2, options)
    b, records_b = rollout(env, 987, options)
    np.testing.assert_array_equal(a, b)
    assert records_a == records_b
    assert all(record[-1]["weather_delta_c"] == 1.0 for record in records_a[1:])


def test_weather_can_be_fixed_independently_of_initial_temperature():
    env = HVAC2R2CEnv()
    options = dict(scenario="transition", weather_delta_c=-0.75)
    a, _ = rollout(env, 3, options)
    b, _ = rollout(env, 9, options)
    assert a[0, 0] != b[0, 0]
    np.testing.assert_array_equal(a[:, 2:], b[:, 2:])


def test_random_initialization_and_envelope_balance(config):
    env = HVAC2R2CEnv(config)
    offsets = []
    for seed in (1, 2, 3, 4):
        _, info = env.reset(seed=seed)
        offsets.append(info["weather_delta_c"])
        assert -2 <= info["weather_delta_c"] <= 2
        midpoint = (info["comfort_low_c"] + info["comfort_high_c"]) / 2
        assert abs(info["indoor_temp_c"] - midpoint) <= 2
        indoor, envelope, outdoor = (info[k] for k in ("indoor_temp_c", "envelope_temp_c", "outdoor_temp_c"))
        flux = (indoor - envelope) / config.thermal.R_ie + (outdoor - envelope) / config.thermal.R_eo
        assert flux == pytest.approx(0, abs=1e-12)
    assert len(set(offsets)) > 1
