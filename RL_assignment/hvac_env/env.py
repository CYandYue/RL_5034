"""Gymnasium environment for the indoor/envelope 2R2C model."""

from collections.abc import Mapping
from dataclasses import asdict
from math import cos, sin, tau

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import ResetNeeded

from .comfort import VIOLATION_EPS_C, comfort_deviation
from .config import ConfigSource, load_config
from .hvac import apply_action
from .thermal import ThermalModel
from .validation import finite_number, finite_vector
from .weather import HOURS_PER_DAY, SCENARIOS, internal_gain, outdoor_temperature


OBSERVATION_FIELDS = (
    "T_i", "T_e", "T_o", "Q_g", "comfort_low", "comfort_high", "sin_hour",
    "cos_hour", "scenario_cold", "scenario_hot", "scenario_transition",
)
_RESET_KEYS = {"scenario", "weather_delta_c", "indoor_init_c", "envelope_init_c"}
_FLOAT32_MAX = float(np.finfo(np.float32).max)


class HVAC2R2CEnv(gym.Env):
    """One action holds heating/cooling power constant for one time interval."""

    metadata = {"render_modes": []}

    def __init__(self, config: ConfigSource = None):
        super().__init__()
        self.config = load_config(config)
        self.thermal = ThermalModel(self.config.thermal, self.config.dt_h)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        # Temperature bounds reflect the transport dtype, not physical clipping.
        limit = _FLOAT32_MAX
        self.observation_space = spaces.Box(
            low=np.array([-limit, -limit, -limit, 0, -limit, -limit, -1, -1, 0, 0, 0], dtype=np.float32),
            high=np.array([limit] * 6 + [1] * 5, dtype=np.float32),
        )
        self._state: np.ndarray | None = None
        self._step_index = 0

    def _forcing(self, time_h: float, scenario_name: str, delta_c: float) -> tuple[float, float]:
        weather = self.config.weather
        scenario = weather.scenario(scenario_name)
        outdoor_c = outdoor_temperature(time_h, scenario, delta_c, weather.phase_h)
        return outdoor_c, internal_gain(time_h, self.config.gains)

    def _observation(
        self, state: np.ndarray, time_h: float, scenario_name: str, forcing: tuple[float, float]
    ) -> np.ndarray:
        scenario = self.config.weather.scenario(scenario_name)
        outdoor_c, gain_kw = forcing
        angle = tau * (time_h % HOURS_PER_DAY) / HOURS_PER_DAY
        values = np.array([
            *state, outdoor_c, gain_kw, scenario.comfort_low_c, scenario.comfort_high_c,
            sin(angle), cos(angle), *(name == scenario_name for name in SCENARIOS),
        ], dtype=np.float64)
        if not np.all(np.abs(values) <= _FLOAT32_MAX):
            raise ValueError("physical observation must fit in finite float32 values")
        return values.astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        options = {} if options is None else options
        if not isinstance(options, Mapping):
            raise ValueError("reset options must be a mapping")
        unknown = options.keys() - _RESET_KEYS
        if unknown:
            raise ValueError(f"unknown reset options: {unknown}")

        if "scenario" in options:
            name = options["scenario"]
        else:
            name = str(self.np_random.choice(SCENARIOS))
        scenario = self.config.weather.scenario(name)
        if "weather_delta_c" in options:
            delta_c = finite_number(options["weather_delta_c"], "weather_delta_c")
        else:
            radius = self.config.weather.delta_radius_c
            delta_c = float(self.np_random.uniform(-radius, radius))

        outdoor_c, gain_kw = self._forcing(0.0, name, delta_c)
        if "indoor_init_c" in options:
            indoor_c = finite_number(options["indoor_init_c"], "indoor_init_c")
        else:
            midpoint = (scenario.comfort_low_c + scenario.comfort_high_c) / 2.0
            spread = self.config.indoor_init_spread_c
            indoor_c = midpoint + float(self.np_random.uniform(-spread, spread))
        if "envelope_init_c" in options:
            envelope_c = finite_number(options["envelope_init_c"], "envelope_init_c")
        else:
            envelope_c = self.config.thermal.equilibrium_envelope_c(indoor_c, outdoor_c)

        state = np.array([indoor_c, envelope_c], dtype=np.float64)
        forcing = (outdoor_c, gain_kw)
        obs = self._observation(state, 0.0, name, forcing)
        self._state, self._forcing_now = state, forcing
        self._scenario_name, self._delta_c = name, delta_c
        self._step_index = 0
        return obs, {
            "scenario": name,
            "weather_delta_c": delta_c,
            "step_index": 0,
            "time_h": 0.0,
            "indoor_temp_c": indoor_c,
            "envelope_temp_c": envelope_c,
            "outdoor_temp_c": outdoor_c,
            "internal_gain_kw": gain_kw,
            "comfort_low_c": scenario.comfort_low_c,
            "comfort_high_c": scenario.comfort_high_c,
        }

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._state is None or self._step_index >= self.config.episode_steps:
            raise ResetNeeded("call reset() before starting a new episode")
        dt_h = self.config.dt_h
        requested = finite_vector(action, "action", 1)[0]
        hvac = apply_action(requested, self.config.hvac, dt_h)

        # The interval uses the forcing at its left endpoint, in float64.
        outdoor_c, gain_kw = self._forcing_now
        inputs = [hvac.q_hvac_kw_thermal, gain_kw, outdoor_c]
        next_state = self.thermal.advance(self._state, inputs)
        indoor_c, envelope_c = map(float, next_state)
        scenario = self.config.weather.scenario(self._scenario_name)
        deviation = comfort_deviation(indoor_c, scenario.comfort_low_c, scenario.comfort_high_c)
        reward_comfort = -dt_h * self.config.reward.w_c * deviation
        reward_energy = -self.config.reward.w_e * hvac.energy_kwh_step
        reward = reward_comfort + reward_energy

        next_index = self._step_index + 1
        time_h_end = next_index * dt_h
        next_forcing = self._forcing(time_h_end, self._scenario_name, self._delta_c)
        obs = self._observation(next_state, time_h_end, self._scenario_name, next_forcing)
        info = {
            "scenario": self._scenario_name,
            "weather_delta_c": self._delta_c,
            "step_index": next_index,
            "time_h_start": self._step_index * dt_h,
            "time_h_end": time_h_end,
            "indoor_temp_c_before": float(self._state[0]),
            "indoor_temp_c": indoor_c,
            "envelope_temp_c": envelope_c,
            "outdoor_temp_c_applied": outdoor_c,
            "internal_gain_kw_applied": gain_kw,
            **asdict(hvac),
            "comfort_low_c": scenario.comfort_low_c,
            "comfort_high_c": scenario.comfort_high_c,
            "comfort_deviation_c": deviation,
            "comfort_violation": deviation > VIOLATION_EPS_C,
            "reward_comfort": reward_comfort,
            "reward_energy": reward_energy,
            "reward_total": reward,
        }
        self._state, self._forcing_now = next_state, next_forcing
        self._step_index = next_index
        truncated = next_index == self.config.episode_steps
        return obs, reward, False, truncated, info
