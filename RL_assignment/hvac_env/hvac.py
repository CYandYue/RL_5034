"""Bidirectional sensible heat delivery and fixed-COP electricity use."""

from dataclasses import dataclass

from .validation import finite_number, nonnegative, positive


@dataclass(frozen=True)
class HVACParameters:
    q_heat_max_kw: float
    q_cool_max_kw: float
    cop_heat: float
    cop_cool: float

    def __post_init__(self):
        nonnegative(self.q_heat_max_kw, "q_heat_max_kw")
        nonnegative(self.q_cool_max_kw, "q_cool_max_kw")
        positive(self.cop_heat, "cop_heat")
        positive(self.cop_cool, "cop_cool")


@dataclass(frozen=True)
class HVACOutput:
    action_requested: float
    action_applied: float
    q_hvac_kw_thermal: float
    heating_electric_kw: float
    cooling_electric_kw: float
    p_electric_kw: float
    energy_kwh_step: float


def apply_action(requested: float, p: HVACParameters, dt_h: float) -> HVACOutput:
    requested = finite_number(requested, "action")
    dt_h = positive(dt_h, "dt_h")
    applied = min(1.0, max(-1.0, requested))
    heat_kw = p.q_heat_max_kw * max(applied, 0.0)
    cool_kw = p.q_cool_max_kw * max(-applied, 0.0)
    heat_electric_kw = heat_kw / p.cop_heat
    cool_electric_kw = cool_kw / p.cop_cool
    electric_kw = heat_electric_kw + cool_electric_kw
    return HVACOutput(
        action_requested=requested,
        action_applied=applied,
        q_hvac_kw_thermal=heat_kw - cool_kw,
        heating_electric_kw=heat_electric_kw,
        cooling_electric_kw=cool_electric_kw,
        p_electric_kw=electric_kw,
        energy_kwh_step=electric_kw * dt_h,
    )
