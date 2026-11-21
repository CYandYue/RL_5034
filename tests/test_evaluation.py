import numpy as np
import pytest

from hvac_ppo.evaluation import evaluate_cases, off_policy, thermostat_policy
from hvac_ppo.scenarios import EvalCase


def constant_conditions(temperature):
    return {"episode_days": 0.25, "gains": {"base_kw": 0, "occupied_kw": 0, "daylight_peak_kw": 0},
            "weather": {"cold": {"mean_c": temperature, "amplitude_c": 0}}}


def test_constant_one_degree_violation_has_six_degree_hours():
    case = EvalCase("constant", "cold", 0, 19, 19, 0)
    rows, traces = evaluate_cases(off_policy, [case], constant_conditions(19), collect_traces=True)
    row = rows[0]
    assert row["steps"] == 24
    assert row["cvr_pct"] == 100
    assert row["degree_hours"] == pytest.approx(6)
    assert row["return_raw"] == pytest.approx(-6)
    assert row["energy_kwh"] == 0
    assert len(traces) == 25  # Initial point is plotted, but not counted as an interval.


def test_half_power_for_six_hours_uses_one_kwh():
    case = EvalCase("constant", "cold", 0, 21.75, 21.75, 0)
    rows, _ = evaluate_cases(lambda obs: np.array([0.5], dtype=np.float32), [case],
                             constant_conditions(21.75))
    assert rows[0]["energy_kwh"] == pytest.approx(1.0)


def test_fixed_conditions_replay_independent_of_call_order():
    a = EvalCase("a", "cold", 1, 22, 7, 17)
    b = EvalCase("b", "hot", -1, 25, 29, 31)
    first, _ = evaluate_cases(thermostat_policy, [a, b], {"episode_days": 1 / 24})
    second, _ = evaluate_cases(thermostat_policy, [b, a], {"episode_days": 1 / 24})
    assert first == second[::-1]
