from __future__ import annotations

import numpy as np

from OOP.problems.shock_shear_layer import (
    NonPeriodicViscousRHS2D,
    ShockShearBoundary2D,
    ShockShearLayerConfig,
    hyperviscosity_enabled,
    run_shock_shear_layer,
)


def small_config(**overrides) -> ShockShearLayerConfig:
    values = dict(
        nx=40,
        ny=20,
        tfinal=0.02,
        cfl=0.1,
        backend="numpy",
        guard_cells=2,
        boundary_guard=2,
        sensor_width=1,
        mn=0.0,
        progress_every=0,
    )
    values.update(overrides)
    return ShockShearLayerConfig(**values)


def test_reference_parameters_reproduce_reported_mach_numbers():
    config = ShockShearLayerConfig(backend="numpy")
    assert np.isclose(config.upper_inflow_mach, 5.6250, rtol=5.0e-5)
    assert np.isclose(config.lower_inflow_mach, 1.7647, rtol=5.0e-5)
    assert np.isclose(config.forcing_period, 30.0 / 2.68)


def test_time_dependent_inlet_and_boundary_states():
    config = small_config()
    boundary = ShockShearBoundary2D(config)
    time = 0.37 * config.forcing_period
    q = boundary.apply(config.initial_state(), time)
    rho, u, v, pressure = config.equation.primitive_from_conservative(q)
    g = config.guard_cells

    # Inlet matches the prescribed mixing-layer profile away from corners.
    rho_i, u_i, v_i, p_i = config.inlet_primitive(time, xp=np)
    assert np.allclose(rho[g:-g, 0], rho_i[g:-g])
    assert np.allclose(u[g:-g, 0], u_i[g:-g])
    assert np.allclose(v[g:-g, 0], v_i[g:-g])
    assert np.allclose(pressure[g:-g, 0], p_i[g:-g])

    # The complete top guard has the prescribed post-shock state.
    assert np.allclose(rho[-g:, :], config.postshock_density)
    assert np.allclose(u[-g:, :], config.postshock_u)
    assert np.allclose(v[-g:, :], config.postshock_v)
    assert np.allclose(pressure[-g:, :], config.postshock_pressure)

    # Slip-wall reflection makes the bottom normal velocity antisymmetric.
    assert np.allclose(v[:g, :], -v[g : 2 * g, :][::-1, :])


def test_viscous_rhs_is_conservative_in_mass_and_finite():
    config = small_config(viscosity_model="sutherland")
    q = ShockShearBoundary2D(config).apply(config.initial_state(), 0.0)
    rhs = NonPeriodicViscousRHS2D(config)(q)
    assert np.allclose(rhs[0], 0.0)
    assert np.all(np.isfinite(rhs))


def test_weno_only_disables_hyperviscosity():
    hybrid = small_config(scheme="hybrid", mn=1.0e-3)
    weno = small_config(scheme="weno", mn=1.0e-3)
    assert hyperviscosity_enabled(hybrid)
    assert not hyperviscosity_enabled(weno)


def test_short_numpy_run_remains_physical():
    config = small_config()
    _q, diagnostics = run_shock_shear_layer(config)
    assert diagnostics.time == config.tfinal
    assert diagnostics.density_positive
    assert diagnostics.pressure_positive
    assert not diagnostics.has_nonfinite
    assert diagnostics.steps >= 1


def test_centerline_is_interpolated_to_exact_y_zero():
    from OOP.problems.shock_shear_layer import interpolate_to_y_zero

    config = small_config()
    field = np.broadcast_to(config.domain.y[:, None], (config.ny, config.nx))
    line = interpolate_to_y_zero(field, config.domain.y)
    assert np.allclose(line, 0.0)
