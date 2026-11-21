"""One raw-environment evaluator for rules and learned policies."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from hvac_env import HVAC2R2CEnv
from hvac_env.comfort import VIOLATION_EPS_C

from .io import write_csv
from .scenarios import EvalCase

METRICS = ("return_raw", "cvr_pct", "degree_hours", "energy_kwh")


def off_policy(observation):
    return np.array([0.0], dtype=np.float32)


def thermostat_policy(observation, deadband_c=0.5):
    """Full-power relay around the comfort midpoint, with a +/-0.5 C deadband."""
    indoor, low, high = float(observation[0]), float(observation[4]), float(observation[5])
    target = (low + high) / 2
    action = 1.0 if indoor < target - deadband_c else -1.0 if indoor > target + deadband_c else 0.0
    return np.array([action], dtype=np.float32)


@dataclass
class PPOController:
    model: object
    normalizer: object

    def __call__(self, observation):
        normalized = self.normalizer.normalize_obs(observation.copy())
        action, _ = self.model.predict(normalized, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(1)


@contextmanager
def frozen_normalization(normalizer):
    training, norm_reward = normalizer.training, normalizer.norm_reward
    normalizer.training, normalizer.norm_reward = False, False
    try:
        yield
    finally:
        normalizer.training, normalizer.norm_reward = training, norm_reward


def evaluate_cases(
    controller: Callable, cases: list[EvalCase], environment: dict | None = None,
    collect_traces: bool = False,
) -> tuple[list[dict], list[dict]]:
    env = HVAC2R2CEnv(environment)
    rows, traces = [], []
    try:
        for case in cases:
            observation, initial = env.reset(seed=case.reset_seed, options=case.options())
            reward_sum = energy_sum = deviation_sum = 0.0
            violations = steps = 0
            if collect_traces:
                traces.append({"case_id": case.case_id, "scenario": case.scenario, "step": 0,
                               "time_h": 0.0, "indoor_c": initial["indoor_temp_c"],
                               "envelope_c": initial["envelope_temp_c"], "outdoor_c": initial["outdoor_temp_c"],
                               "comfort_low_c": initial["comfort_low_c"], "comfort_high_c": initial["comfort_high_c"],
                               "action": 0.0, "electric_kw": 0.0, "cumulative_kwh": 0.0,
                               "reward": 0.0, "deviation_c": 0.0})
            while True:
                action = controller(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                if not np.isfinite(observation).all() or not np.isfinite(reward):
                    raise FloatingPointError(f"nonfinite trajectory in {case.case_id}")
                steps += 1
                reward_sum += reward
                energy_sum += info["energy_kwh_step"]
                deviation_sum += info["comfort_deviation_c"]
                violations += int(info["comfort_deviation_c"] > VIOLATION_EPS_C)
                if collect_traces:
                    traces.append({"case_id": case.case_id, "scenario": case.scenario, "step": steps,
                                   "time_h": info["time_h_end"], "indoor_c": info["indoor_temp_c"],
                                   "envelope_c": info["envelope_temp_c"], "outdoor_c": float(observation[2]),
                                   "comfort_low_c": info["comfort_low_c"], "comfort_high_c": info["comfort_high_c"],
                                   "action": info["action_applied"], "electric_kw": info["p_electric_kw"],
                                   "cumulative_kwh": energy_sum, "reward": reward,
                                   "deviation_c": info["comfort_deviation_c"]})
                if terminated or truncated:
                    if steps != env.config.episode_steps:
                        raise RuntimeError("unexpected physical termination or episode length")
                    break
            degree_hours = deviation_sum * env.config.dt_h
            expected = -env.config.reward.w_c * degree_hours - env.config.reward.w_e * energy_sum
            if not np.isclose(reward_sum, expected, rtol=1e-10, atol=1e-9):
                raise RuntimeError("raw reward and physical metrics disagree")
            rows.append({"case_id": case.case_id, "scenario": case.scenario, "steps": steps,
                         "return_raw": reward_sum, "cvr_pct": 100 * violations / steps,
                         "degree_hours": degree_hours, "energy_kwh": energy_sum})
    finally:
        env.close()
    return rows, traces


def mean_metrics(rows: list[dict]) -> dict:
    return {key: float(np.mean([row[key] for row in rows])) for key in METRICS}


def evaluate_rules(cases: list[EvalCase], output_dir: Path, environment=None) -> list[dict]:
    all_rows = []
    for name, policy in (("off", off_policy), ("thermostat", thermostat_policy)):
        rows, traces = evaluate_cases(policy, cases, environment, collect_traces=True)
        all_rows += [{"controller": name, **row} for row in rows]
        representative = [row for row in traces if row["case_id"].endswith("_00")]
        write_csv(output_dir / f"{name}_traces.csv", representative)
    write_csv(output_dir / "rule_episodes.csv", all_rows)
    return all_rows
