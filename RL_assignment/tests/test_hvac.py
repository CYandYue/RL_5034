import pytest

from hvac_env import HVAC2R2CEnv
from hvac_env.hvac import HVACParameters, apply_action


@pytest.mark.parametrize("requested, applied, thermal, heat_el, cool_el", [
    (-3, -1, -3, 0, 1.5), (-0.5, -0.5, -1.5, 0, 0.75), (0, 0, 0, 0, 0),
    (0.5, 0.5, 1, 0.25, 0), (2, 1, 2, 0.5, 0),
])
def test_asymmetric_capacity_and_cop(requested, applied, thermal, heat_el, cool_el):
    params = HVACParameters(q_heat_max_kw=2, q_cool_max_kw=3, cop_heat=4, cop_cool=2)
    result = apply_action(requested, params, 0.25)
    assert result.action_requested == requested
    assert result.action_applied == applied
    assert result.q_hvac_kw_thermal == thermal
    assert result.heating_electric_kw == heat_el
    assert result.cooling_electric_kw == cool_el
    assert result.p_electric_kw == heat_el + cool_el
    assert result.energy_kwh_step == result.p_electric_kw * 0.25


def test_single_step_action_direction():
    env = HVAC2R2CEnv()
    options = dict(scenario="transition", weather_delta_c=0, indoor_init_c=22, envelope_init_c=18)
    temperatures = []
    for action in (-0.5, 0, 0.5):
        env.reset(options=options)
        _, _, _, _, info = env.step([action])
        temperatures.append(info["indoor_temp_c"])
        assert info["q_hvac_kw_thermal"] == action
        assert info["p_electric_kw"] == pytest.approx(abs(action) / 3)
        assert info["energy_kwh_step"] == pytest.approx(abs(action) / 3 * 0.25)
    print(f"next indoor degC for actions -0.5, 0, +0.5: {temperatures}")
    assert temperatures[0] < temperatures[1] < temperatures[2]
