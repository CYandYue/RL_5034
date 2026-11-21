"""Indoor/envelope heat balance. Time is in hours and heat flow in kW."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import cont2discrete

from .validation import finite_vector, positive


@dataclass(frozen=True)
class ThermalParameters:
    R_eo: float
    R_ie: float
    C_e: float
    C_i: float

    def __post_init__(self):
        for name, value in vars(self).items():
            positive(value, name)

    def equilibrium_envelope_c(self, indoor_c: float, outdoor_c: float) -> float:
        """Envelope temperature at which its two conductive flows balance."""
        return (self.R_eo * indoor_c + self.R_ie * outdoor_c) / (self.R_ie + self.R_eo)


def continuous_matrices(p: ThermalParameters) -> tuple[np.ndarray, np.ndarray]:
    """State [indoor, envelope], input [HVAC heat, internal gain, outdoor]."""
    g_ie = 1.0 / p.R_ie
    g_eo = 1.0 / p.R_eo
    A = np.array([
        [-g_ie / p.C_i, g_ie / p.C_i],
        [g_ie / p.C_e, -(g_ie + g_eo) / p.C_e],
    ])
    B = np.array([
        [1.0 / p.C_i, 1.0 / p.C_i, 0.0],
        [0.0, 0.0, g_eo / p.C_e],
    ])
    return A, B


class ThermalModel:
    """Cache one ZOH discretization for a fixed parameter set and time step."""

    def __init__(self, parameters: ThermalParameters, dt_h: float):
        self.dt_h = positive(dt_h, "dt_h")
        self.A, self.B = continuous_matrices(parameters)
        # Both temperatures are outputs; there is no direct input-to-output term.
        system = (self.A, self.B, np.eye(2), np.zeros((2, 3)))
        self.Ad, self.Bd, _, _, _ = cont2discrete(system, self.dt_h, method="zoh")

    def derivative(self, state, inputs) -> np.ndarray:
        """Continuous rate of change of both temperatures, in degC/h."""
        x = finite_vector(state, "state", 2)
        v = finite_vector(inputs, "inputs", 3)
        return self.A @ x + self.B @ v

    def advance(self, state, inputs) -> np.ndarray:
        """End temperatures when all three inputs are held for one interval."""
        x = finite_vector(state, "state", 2)
        v = finite_vector(inputs, "inputs", 3)
        return self.Ad @ x + self.Bd @ v
