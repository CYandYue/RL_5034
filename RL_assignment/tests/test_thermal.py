import numpy as np
import pytest
from scipy.integrate import solve_ivp

from hvac_env.thermal import ThermalModel, ThermalParameters, continuous_matrices


def test_matrices_for_hand_calculated_parameters():
    parameters = ThermalParameters(R_eo=4, R_ie=2, C_e=2, C_i=1)
    A, B = continuous_matrices(parameters)
    np.testing.assert_array_equal(A, [[-0.5, 0.5], [0.25, -0.375]])
    np.testing.assert_array_equal(B, [[1, 1, 0], [0, 0, 0.125]])


def test_default_matrices_and_stability(config):
    model = ThermalModel(config.thermal, config.dt_h)
    for matrix, shape in ((model.A, (2, 2)), (model.B, (2, 3)),
                          (model.Ad, (2, 2)), (model.Bd, (2, 3))):
        assert matrix.shape == shape
        assert np.isfinite(matrix).all()
    continuous_poles = np.linalg.eigvals(model.A)
    discrete_poles = np.linalg.eigvals(model.Ad)
    assert np.all(continuous_poles.real < 0)
    assert np.all(np.abs(discrete_poles) < 1)
    print(f"poles per hour: {continuous_poles}; discrete poles: {discrete_poles}")


def test_zoh_against_independent_heat_flux_ode(config):
    model = ThermalModel(config.thermal, config.dt_h)
    p = config.thermal
    initial = [24.0, 16.0]
    hvac_kw, gain_kw, outdoor_c = -0.6, 0.27, 5.0

    def heat_balance(_time, temperatures):
        indoor_c, envelope_c = temperatures
        to_indoor_kw = (envelope_c - indoor_c) / p.R_ie
        from_outdoors_kw = (outdoor_c - envelope_c) / p.R_eo
        return [(to_indoor_kw + hvac_kw + gain_kw) / p.C_i,
                (from_outdoors_kw - to_indoor_kw) / p.C_e]

    times = np.arange(25) * config.dt_h
    reference = solve_ivp(heat_balance, (0, times[-1]), initial, t_eval=times,
                          method="DOP853", rtol=1e-12, atol=1e-13)
    assert reference.success
    states = [initial]
    for _ in times[1:]:
        states.append(model.advance(states[-1], [hvac_kw, gain_kw, outdoor_c]))
    error_c = np.max(np.abs(np.array(states) - reference.y.T), axis=0)
    print(f"ZOH vs ODE maximum error (indoor, envelope) degC: {error_c}")
    assert np.all(error_c < 1e-9)


@pytest.mark.parametrize("state, inputs", [
    ([24, 16], [-0.6, 0.27, 5]), ([18, 27], [0.8, 0.1, 32]), ([20, 20], [0, 0, 20]),
])
def test_continuous_energy_balance(config, state, inputs):
    p = config.thermal
    model = ThermalModel(p, config.dt_h)
    rates = model.derivative(state, inputs)
    stored_kw = p.C_i * rates[0] + p.C_e * rates[1]
    external_kw = (inputs[2] - state[1]) / p.R_eo + inputs[0] + inputs[1]
    print(f"stored={stored_kw:.15g} kW, external={external_kw:.15g} kW, "
          f"residual={stored_kw - external_kw:.3e} kW")
    assert abs(stored_kw - external_kw) < 1e-12


def test_passive_convergence(config):
    model = ThermalModel(config.thermal, config.dt_h)
    state = np.array([25.0, 18.0])
    outdoor_c = 5.0
    # Both initial nodes are warmer than outside. Their maximum excess must decay.
    for _ in range(round(40 * 24 / model.dt_h)):
        next_state = model.advance(state, [0, 0, outdoor_c])
        assert next_state.min() >= outdoor_c - 1e-12
        assert next_state.max() <= state.max() + 1e-12
        state = next_state
    error_c = np.max(np.abs(state - outdoor_c))
    print(f"passive temperatures after 40 days: {state} degC, max error: {error_c:.3e} degC")
    np.testing.assert_allclose(state, outdoor_c, atol=1e-9, rtol=0)


@pytest.mark.parametrize("outdoor_c", [4.0, 34.0])
def test_steady_state_heat_requirement(config, outdoor_c):
    p = config.thermal
    model = ThermalModel(p, config.dt_h)
    indoor_c, gain_kw = 22.0, 0.12
    net_heat_kw = (indoor_c - outdoor_c) / (p.R_ie + p.R_eo)
    envelope_c = outdoor_c + net_heat_kw * p.R_eo
    state = [indoor_c, envelope_c]
    inputs = [net_heat_kw - gain_kw, gain_kw, outdoor_c]
    np.testing.assert_allclose(model.derivative(state, inputs), 0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.advance(state, inputs), state, rtol=0, atol=1e-12)
    assert p.equilibrium_envelope_c(indoor_c, outdoor_c) == pytest.approx(envelope_c)


@pytest.mark.parametrize("state, inputs", [([1], [0, 0, 1]), ([1, 2], [0, np.inf, 1])])
def test_thermal_rejects_invalid_vectors(config, state, inputs):
    with pytest.raises(ValueError):
        ThermalModel(config.thermal, config.dt_h).advance(state, inputs)
