import pytest

from hvac_env import HVAC2R2CEnv, load_config
from hvac_env.thermal import ThermalModel


@pytest.mark.parametrize("overrides, message", [
    ({"thermal": {"R_ie": 0}}, "R_ie"), ({"thermal": {"R_eo": -1}}, "R_eo"),
    ({"thermal": {"C_i": 0}}, "C_i"), ({"thermal": {"C_e": -1}}, "C_e"),
    ({"hvac": {"cop_heat": 0}}, "cop_heat"), ({"hvac": {"cop_cool": -2}}, "cop_cool"),
    ({"hvac": {"q_cool_max_kw": -1}}, "q_cool_max_kw"), ({"dt_h": 0}, "dt_h"),
    ({"episode_days": -1}, "episode_days"), ({"dt_h": 5}, "integer"),
    ({"dt_h": float("nan")}, "dt_h"), ({"dt_h": True}, "dt_h"),
    ({"weather": {"hot": {"comfort_high_c": 20}}}, "comfort_high_c"),
    ({"gains": {"sunset_h": 5}}, "sunset_h"), ({"reward": {"w_e": -0.1}}, "w_e"),
    ({"weather": {"humid": {}}}, "weather.humid"), ({"hvac": 3}, "hvac"),
])
def test_invalid_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        HVAC2R2CEnv(overrides)


def test_packaged_defaults_and_partial_yaml(tmp_path, config):
    assert (config.thermal.R_eo, config.thermal.R_ie, config.thermal.C_e, config.thermal.C_i) == (
        3.02, 27.81, 2.11, 1.11,
    )
    assert config.episode_steps == 672
    path = tmp_path / "settings.yaml"
    path.write_text("dt_h: 0.5\nweather:\n  cold:\n    mean_c: 12\n", encoding="utf-8")
    env = HVAC2R2CEnv(path)
    assert env.config == load_config({"dt_h": 0.5, "weather": {"cold": {"mean_c": 12}}})
    assert env.config.episode_steps == 336
    assert env.config.weather.cold.amplitude_c == 6
    assert load_config().weather.cold.mean_c == 5


def test_thermal_model_checks_its_own_dt(config):
    with pytest.raises(ValueError, match="dt_h"):
        ThermalModel(config.thermal, 0)
