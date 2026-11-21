"""End-of-step comfort distance and equal-duration trajectory metrics."""

import numpy as np

from .validation import finite_number, finite_vector, nonnegative, positive


VIOLATION_EPS_C = 1e-8


def comfort_deviation(temperature: float, low: float, high: float) -> float:
    """Distance in degC to the closed comfort interval [low, high]."""
    temperature = finite_number(temperature, "temperature")
    low = finite_number(low, "low")
    high = finite_number(high, "high")
    if low > high:
        raise ValueError("low must not exceed high")
    return max(0.0, low - temperature, temperature - high)


def _nonnegative_series(values, name: str) -> np.ndarray:
    series = finite_vector(values, name)
    if np.any(series < 0):
        raise ValueError(f"{name} must be nonnegative")
    return series


def cvr_pct(deviations_c, eps: float = VIOLATION_EPS_C) -> float:
    """Percentage of interval endpoints with a comfort distance above eps."""
    series = _nonnegative_series(deviations_c, "deviations_c")
    eps = nonnegative(eps, "eps")
    if series.size == 0:
        raise ValueError("CVR needs at least one time step")
    return float(100.0 * np.mean(series > eps))


def degree_hours(deviations_c, dt_h: float) -> float:
    """Integral of sampled comfort distance, in degC h."""
    series = _nonnegative_series(deviations_c, "deviations_c")
    return float(series.sum()) * positive(dt_h, "dt_h")


def electricity_kwh(p_electric_kw, dt_h: float) -> float:
    """Total electrical energy from the interval powers, in kWh."""
    series = _nonnegative_series(p_electric_kw, "p_electric_kw")
    return float(series.sum()) * positive(dt_h, "dt_h")
