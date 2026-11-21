"""Deterministic daily outdoor temperature and internal heat gain profiles."""

from dataclasses import dataclass
from math import pi, sin, tau

from .validation import finite_number, nonnegative


SCENARIOS = ("cold", "hot", "transition")
HOURS_PER_DAY = 24.0


@dataclass(frozen=True)
class Scenario:
    mean_c: float
    amplitude_c: float
    comfort_low_c: float
    comfort_high_c: float

    def __post_init__(self):
        finite_number(self.mean_c, "mean_c")
        nonnegative(self.amplitude_c, "amplitude_c")
        low = finite_number(self.comfort_low_c, "comfort_low_c")
        high = finite_number(self.comfort_high_c, "comfort_high_c")
        if low >= high:
            raise ValueError("comfort_low_c must be below comfort_high_c")


@dataclass(frozen=True)
class WeatherParameters:
    phase_h: float
    delta_radius_c: float
    cold: Scenario
    hot: Scenario
    transition: Scenario

    def __post_init__(self):
        if not 0 <= finite_number(self.phase_h, "phase_h") < HOURS_PER_DAY:
            raise ValueError("phase_h must be in [0, 24)")
        nonnegative(self.delta_radius_c, "delta_radius_c")

    def scenario(self, name: str) -> Scenario:
        if not isinstance(name, str) or name not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}, got {name!r}")
        return getattr(self, name)


@dataclass(frozen=True)
class InternalGains:
    base_kw: float
    occupied_kw: float
    occupied_start_h: float
    occupied_end_h: float
    daylight_peak_kw: float
    sunrise_h: float
    sunset_h: float

    def __post_init__(self):
        for name in ("base_kw", "occupied_kw", "daylight_peak_kw"):
            nonnegative(getattr(self, name), name)
        for start, end in (("occupied_start_h", "occupied_end_h"), ("sunrise_h", "sunset_h")):
            start_h = finite_number(getattr(self, start), start)
            end_h = finite_number(getattr(self, end), end)
            if not 0 <= start_h < end_h <= HOURS_PER_DAY:
                raise ValueError(f"expected 0 <= {start} < {end} <= 24")


def outdoor_temperature(time_h: float, scenario: Scenario, delta_c: float, phase_h: float) -> float:
    hour = time_h % HOURS_PER_DAY
    angle = tau * (hour - phase_h) / HOURS_PER_DAY
    return scenario.mean_c + delta_c + scenario.amplitude_c * sin(angle)


def internal_gain(time_h: float, gains: InternalGains) -> float:
    hour = time_h % HOURS_PER_DAY
    gain_kw = gains.base_kw
    if gains.occupied_start_h <= hour < gains.occupied_end_h:
        gain_kw += gains.occupied_kw
    if gains.sunrise_h < hour < gains.sunset_h:
        daylight_fraction = (hour - gains.sunrise_h) / (gains.sunset_h - gains.sunrise_h)
        gain_kw += gains.daylight_peak_kw * sin(pi * daylight_fraction)
    return gain_kw
