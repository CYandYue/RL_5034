"""Load the packaged YAML defaults and validate optional overrides."""

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from math import isclose, isfinite
from pathlib import Path

import yaml

from .hvac import HVACParameters
from .thermal import ThermalParameters
from .validation import finite_number, nonnegative, positive
from .weather import HOURS_PER_DAY, SCENARIOS, InternalGains, Scenario, WeatherParameters


@dataclass(frozen=True)
class RewardWeights:
    w_c: float
    w_e: float

    def __post_init__(self):
        nonnegative(self.w_c, "w_c")
        nonnegative(self.w_e, "w_e")


@dataclass(frozen=True)
class EnvConfig:
    thermal: ThermalParameters
    hvac: HVACParameters
    weather: WeatherParameters
    gains: InternalGains
    reward: RewardWeights
    dt_h: float
    episode_days: float
    indoor_init_spread_c: float

    def __post_init__(self):
        positive(self.dt_h, "dt_h")
        positive(self.episode_days, "episode_days")
        nonnegative(self.indoor_init_spread_c, "indoor_init_spread_c")
        steps = self.episode_days * HOURS_PER_DAY / self.dt_h
        if not isfinite(steps) or round(steps) < 1 or not isclose(
            steps, round(steps), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("episode_days * 24 / dt_h must be a positive integer")

    @property
    def episode_steps(self) -> int:
        return round(self.episode_days * HOURS_PER_DAY / self.dt_h)


ConfigSource = EnvConfig | Mapping | str | Path | None


def _apply_overrides(defaults: dict, overrides: Mapping, prefix: str = "") -> None:
    if not isinstance(overrides, Mapping):
        raise ValueError(f"{prefix.rstrip('.') or 'config'} must be a mapping")
    for key, value in overrides.items():
        name = f"{prefix}{key}"
        if key not in defaults:
            raise ValueError(f"unknown config key: {name}")
        if isinstance(defaults[key], dict):
            _apply_overrides(defaults[key], value, f"{name}.")
        else:
            defaults[key] = finite_number(value, name)


def load_config(source: ConfigSource = None) -> EnvConfig:
    """Accept a validated config, partial mapping or YAML path; defaults travel in the wheel."""
    if isinstance(source, EnvConfig):
        return source
    default_file = files("hvac_env").joinpath("configs").joinpath("default.yaml")
    raw = yaml.safe_load(default_file.read_text(encoding="utf-8"))
    if source is not None:
        if isinstance(source, (str, Path)):
            source = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
        _apply_overrides(raw, source)

    weather = raw["weather"]
    for name in SCENARIOS:
        weather[name] = Scenario(**weather[name])
    return EnvConfig(
        thermal=ThermalParameters(**raw["thermal"]),
        hvac=HVACParameters(**raw["hvac"]),
        weather=WeatherParameters(**weather),
        gains=InternalGains(**raw["gains"]),
        reward=RewardWeights(**raw["reward"]),
        dt_h=raw["dt_h"],
        episode_days=raw["episode_days"],
        indoor_init_spread_c=raw["indoor_init_spread_c"],
    )
