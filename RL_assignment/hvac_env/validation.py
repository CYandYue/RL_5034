"""Numeric checks shared by configuration and public physics functions."""

from math import isfinite
from numbers import Real

import numpy as np


def finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def positive(value, name: str) -> float:
    number = finite_number(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def nonnegative(value, name: str) -> float:
    number = finite_number(value, name)
    if number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def finite_vector(values, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or (length is not None and array.size != length):
        expected = "one dimension" if length is None else f"shape ({length},)"
        raise ValueError(f"{name} must have {expected}")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array
