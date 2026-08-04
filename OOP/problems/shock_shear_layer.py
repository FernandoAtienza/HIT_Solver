from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np

from OOP.domain import Domain2D
from OOP.parallel.backend import array_module, select_array_module, to_numpy
from OOP.parallel.equations import ParallelEulerEquation2D
from OOP.problems.riemann_config3 import (
    LocalHyperviscosity2D,
    NonPeriodicEulerShockSensor2D,
    NonPeriodicHybridEuler2DOperator,
    finite_difference,
    interface_mask_from_nodes,
    raw_density_pressure,
    schlieren_field,
    validate_raw_physical_state,
    vorticity_z,
)
from OOP.run_utils import make_run_id, write_json


@dataclass(frozen=True)
class ShockShearLayerConfig:
    """Two-dimensional shock--shear-layer interaction benchmark.

    The reference setup follows Section 3.2 of Kang and Lee (2026): a
    spatially evolving mixing layer in ``[0, 200] x [-20, 20]`` is disturbed
    at the inlet and impinged by an oblique shock generated from the upper
    boundary.

    The paper specifies ``Re=500`` and states that its solver uses
    Sutherland-law viscosity.  The exact nondimensional reference
    temperature used in that law is not stated.  Here the upper-stream
    temperature is the reference temperature, corresponding to 300 K by
    default.  ``viscosity_model='constant'`` is also available for the
    classical constant-viscosity benchmark interpretation.
    """

    scenario_name: ClassVar[str] = "shock_shear_layer"
    reference_name: ClassVar[str] = "Kang and Lee (2026), Section 3.2"

    nx: int = 500
    ny: int = 100
    tfinal: float = 120.0
    cfl: float = 0.4
    viscous_cfl: float = 0.5
    gamma: float = 1.4
    gas_constant: float = 1.0
    prandtl: float = 0.72
    reynolds: float = 500.0

    x_min: float = 0.0
    x_max: float = 200.0
    y_min: float = -20.0
    y_max: float = 20.0

    backend: str = "cupy"
    scheme: str = "hybrid"  # hybrid or weno

    # Mixing-layer inflow.
    velocity_center: float = 2.5
    velocity_half_jump: float = 0.5
    velocity_tanh_factor: float = 2.0
    pressure_inflow: float = 0.3327
    density_upper: float = 1.6374
    density_lower: float = 0.3626

    perturbation_amplitude_1: float = 0.05
    perturbation_amplitude_2: float = 0.05
    perturbation_phase_1: float = 0.0
    perturbation_phase_2: float = math.pi / 2.0
    perturbation_b: float = 10.0
    wavelength: float = 30.0
    convective_velocity: float = 2.68

    # Post-shock state imposed at the upper boundary.
    postshock_density: float = 2.1101
    postshock_u: float = 2.9709
    postshock_v: float = -0.1367
    postshock_pressure: float = 0.4754

    # Viscosity model.
    viscosity_model: str = "sutherland"  # sutherland or constant
    reference_temperature_kelvin: float = 300.0
    sutherland_constant_kelvin: float = 110.4

    # Hybrid compact/WENO controls.  The shear sensor is disabled by default
    # so that the physical mixing layer is not classified as a shock.
    sensor_width: int = 2
    jump_threshold: float = 0.04
    compression_threshold: float = 2.5
    shear_threshold: float | None = None
    boundary_guard: int = 4
    guard_cells: int = 4

    # Hyperviscosity is applied only on compact-FD nodes.  It is disabled
    # automatically for WENO-only runs.
    mn: float = 5.0e-4
    hyperviscosity_interval: int = 5
    hyperviscosity_density_weight: float = 1.0
    hyperviscosity_momentum_weight: float = 1.0
    hyperviscosity_energy_weight: float = 1.0

    progress_every: int = 100

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("nx and ny must be positive")
        if self.tfinal <= 0.0:
            raise ValueError("tfinal must be positive")
        if self.cfl <= 0.0 or self.viscous_cfl <= 0.0:
            raise ValueError("CFL values must be positive")
        if self.reynolds <= 0.0:
            raise ValueError("reynolds must be positive")
        if self.prandtl <= 0.0:
            raise ValueError("prandtl must be positive")
        if self.scheme not in {"hybrid", "weno"}:
            raise ValueError("scheme must be 'hybrid' or 'weno'")
        if self.viscosity_model not in {"constant", "sutherland"}:
            raise ValueError("viscosity_model must be 'constant' or 'sutherland'")
        if self.guard_cells < 1:
            raise ValueError("guard_cells must be at least one")
        if 2 * self.guard_cells >= min(self.nx, self.ny):
            raise ValueError("guard_cells is too large for the selected grid")

    @property
    def domain(self) -> Domain2D:
        return Domain2D(
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.nx,
            self.ny,
        )

    @property
    def equation(self) -> ParallelEulerEquation2D:
        return ParallelEulerEquation2D(gamma=self.gamma)

    @property
    def forcing_period(self) -> float:
        return self.wavelength / self.convective_velocity

    @property
    def upper_temperature(self) -> float:
        return self.pressure_inflow / (self.density_upper * self.gas_constant)

    @property
    def upper_inflow_mach(self) -> float:
        sound_speed = math.sqrt(
            self.gamma * self.pressure_inflow / self.density_upper
        )
        return (
            self.velocity_center + self.velocity_half_jump
        ) / sound_speed

    @property
    def lower_inflow_mach(self) -> float:
        sound_speed = math.sqrt(
            self.gamma * self.pressure_inflow / self.density_lower
        )
        return (
            self.velocity_center - self.velocity_half_jump
        ) / sound_speed

    def inlet_primitive(self, time: float, xp=None):
        xp = xp or select_array_module(self.backend)
        y = xp.asarray(self.domain.y)
        u = self.velocity_center + self.velocity_half_jump * xp.tanh(
            self.velocity_tanh_factor * y
        )
        rho = xp.where(y >= 0.0, self.density_upper, self.density_lower)
        pressure = xp.full_like(y, self.pressure_inflow)
        v = (
            self.perturbation_amplitude_1
            * xp.cos(
                2.0 * xp.pi * time / self.forcing_period
                + self.perturbation_phase_1
            )
            + self.perturbation_amplitude_2
            * xp.cos(
                4.0 * xp.pi * time / self.forcing_period
                + self.perturbation_phase_2
            )
        ) * xp.exp(-(y**2) / self.perturbation_b)
        return rho, u, v, pressure

    def initial_state(self):
        xp = select_array_module(self.backend)
        _x = xp.asarray(self.domain.x)
        y = xp.asarray(self.domain.y)
        _X, Y = xp.meshgrid(_x, y, indexing="xy")
        u = self.velocity_center + self.velocity_half_jump * xp.tanh(
            self.velocity_tanh_factor * Y
        )
        rho = xp.where(Y >= 0.0, self.density_upper, self.density_lower)
        v = xp.zeros_like(Y)
        pressure = xp.full_like(Y, self.pressure_inflow)
        return self.equation.conservative_from_primitive(rho, u, v, pressure)


@dataclass(frozen=True)
class ShockShearBoundary2D:
    config: ShockShearLayerConfig

    @property
    def equation(self) -> ParallelEulerEquation2D:
        return self.config.equation

    def apply(self, q, time: float):
        """Apply inlet, outlet, slip-wall, and post-shock boundaries.

        The array uses actual boundary guard rows/columns rather than an
        external ghost-cell allocation.  The upper and lower conditions are
        applied last, so the two corner states respect the wall/top boundary
        prescriptions.
        """

        xp = array_module(q)
        bounded = xp.array(q, copy=True)
        g = self.config.guard_cells

        # Non-reflecting approximation at the outlet: zero normal gradient.
        bounded[:, :, -g:] = bounded[:, :, -g - 1][:, :, None]

        # Time-dependent supersonic inlet.
        rho_i, u_i, v_i, p_i = self.config.inlet_primitive(time, xp=xp)
        inlet = self.equation.conservative_from_primitive(rho_i, u_i, v_i, p_i)
        bounded[:, :, :g] = inlet[:, :, None]

        # Lower slip wall.  Reflect the normal velocity and extrapolate the
        # remaining primitive variables evenly across the wall.
        source = bounded[:, g : 2 * g, :][:, ::-1, :]
        rho_b, u_b, v_b, p_b = self.equation.primitive_from_conservative(source)
        reflected = self.equation.conservative_from_primitive(rho_b, u_b, -v_b, p_b)
        bounded[:, :g, :] = reflected

        # Constant post-shock state at the complete upper boundary.  Together
        # with the pre-shock inlet this creates the incident oblique shock at
        # the upper-left corner.
        shape = (g, self.config.nx)
        rho_t = xp.full(shape, self.config.postshock_density, dtype=bounded.dtype)
        u_t = xp.full(shape, self.config.postshock_u, dtype=bounded.dtype)
        v_t = xp.full(shape, self.config.postshock_v, dtype=bounded.dtype)
        p_t = xp.full(shape, self.config.postshock_pressure, dtype=bounded.dtype)
        bounded[:, -g:, :] = self.equation.conservative_from_primitive(
            rho_t, u_t, v_t, p_t
        )
        return self.equation.enforce_physical_state(bounded)


@dataclass(frozen=True)
class NonPeriodicViscousRHS2D:
    config: ShockShearLayerConfig

    @property
    def equation(self) -> ParallelEulerEquation2D:
        return self.config.equation

    def dynamic_viscosity(self, temperature):
        xp = array_module(temperature)
        mu_ref = 1.0 / self.config.reynolds
        if self.config.viscosity_model == "constant":
            return xp.full_like(temperature, mu_ref)

        # Normalize local temperature with the upper-stream value.  This makes
        # mu_ref the upper-stream nondimensional viscosity and implements the
        # standard Sutherland ratio using T_ref=300 K by default.
        theta = xp.maximum(
            temperature / self.config.upper_temperature,
            1.0e-12,
        )
        s_ratio = (
            self.config.sutherland_constant_kelvin
            / self.config.reference_temperature_kelvin
        )
        return (
            mu_ref
            * theta**1.5
            * (1.0 + s_ratio)
            / (theta + s_ratio)
        )

    def transport_coefficients(self, q):
        rho, _u, _v, pressure = self.equation.primitive_from_conservative(q)
        temperature = pressure / (rho * self.config.gas_constant)
        mu = self.dynamic_viscosity(temperature)
        cp = self.config.gamma * self.config.gas_constant / (
            self.config.gamma - 1.0
        )
        conductivity = mu * cp / self.config.prandtl
        return mu, conductivity, cp

    def __call__(self, q):
        xp = array_module(q)
        rho, u, v, pressure = self.equation.primitive_from_conservative(q)
        temperature = pressure / (rho * self.config.gas_constant)
        mu, conductivity, _cp = self.transport_coefficients(q)
        domain = self.config.domain

        du_dx = finite_difference(u, domain.dx, axis=-1)
        du_dy = finite_difference(u, domain.dy, axis=-2)
        dv_dx = finite_difference(v, domain.dx, axis=-1)
        dv_dy = finite_difference(v, domain.dy, axis=-2)
        divergence = du_dx + dv_dy

        tau_xx = 2.0 * mu * du_dx - (2.0 / 3.0) * mu * divergence
        tau_yy = 2.0 * mu * dv_dy - (2.0 / 3.0) * mu * divergence
        tau_xy = mu * (du_dy + dv_dx)

        dT_dx = finite_difference(temperature, domain.dx, axis=-1)
        dT_dy = finite_difference(temperature, domain.dy, axis=-2)

        rhs = xp.zeros_like(q)
        rhs[1] = finite_difference(tau_xx, domain.dx, axis=-1)
        rhs[1] += finite_difference(tau_xy, domain.dy, axis=-2)
        rhs[2] = finite_difference(tau_xy, domain.dx, axis=-1)
        rhs[2] += finite_difference(tau_yy, domain.dy, axis=-2)

        energy_flux_x = u * tau_xx + v * tau_xy + conductivity * dT_dx
        energy_flux_y = u * tau_xy + v * tau_yy + conductivity * dT_dy
        rhs[3] = finite_difference(energy_flux_x, domain.dx, axis=-1)
        rhs[3] += finite_difference(energy_flux_y, domain.dy, axis=-2)
        return rhs


@dataclass(frozen=True)
class ShockShearLayerDiagnostics:
    backend: str
    scheme: str
    viscosity_model: str
    reynolds: float
    prandtl: float
    fixed_dt: float
    steps: int
    time: float
    hyperviscosity_enabled: bool
    hyperviscosity_applications: int
    final_sensor_fraction: float
    final_weno_fraction: float
    density_min: float
    density_max: float
    pressure_min: float
    pressure_max: float
    mach_min: float
    mach_max: float
    max_abs_vorticity: float
    centerline_density_min: float
    centerline_density_max: float
    has_nonfinite: bool
    density_positive: bool
    pressure_positive: bool


def hyperviscosity_enabled(config: ShockShearLayerConfig) -> bool:
    return (
        config.scheme == "hybrid"
        and config.mn > 0.0
        and config.hyperviscosity_interval > 0
    )


def _fixed_time_step(
    q,
    config: ShockShearLayerConfig,
    inviscid_operator: NonPeriodicHybridEuler2DOperator,
    viscous_operator: NonPeriodicViscousRHS2D,
) -> tuple[float, int, float, float]:
    """Return one constant time step constrained by convection and diffusion."""

    xp = array_module(q)
    rho, u, v, pressure = config.equation.primitive_from_conservative(q)
    sound_speed = xp.sqrt(config.gamma * pressure / rho)
    spectral_radius = (
        (xp.abs(u) + sound_speed) / config.domain.dx
        + (xp.abs(v) + sound_speed) / config.domain.dy
    )
    max_radius = float(xp.max(spectral_radius))
    dt_convective = config.cfl / max_radius

    mu, conductivity, cp = viscous_operator.transport_coefficients(q)
    nu = mu / rho
    thermal_diffusivity = conductivity / (rho * cp)
    max_diffusivity = float(xp.max(xp.maximum(nu, thermal_diffusivity)))
    if max_diffusivity > 0.0:
        dt_diffusive = config.viscous_cfl / (
            max_diffusivity
            * (1.0 / config.domain.dx**2 + 1.0 / config.domain.dy**2)
        )
    else:
        dt_diffusive = math.inf

    initial_dt = min(dt_convective, dt_diffusive)
    if not np.isfinite(initial_dt) or initial_dt <= 0.0:
        raise FloatingPointError(f"Invalid initial time step: {initial_dt}")
    n_steps = int(math.ceil(config.tfinal / initial_dt))
    return config.tfinal / n_steps, n_steps, dt_convective, dt_diffusive


def _time_dependent_ssprk3_step(q, time: float, dt: float, rhs, clean):
    """Third-order SSP Runge--Kutta step for time-dependent boundaries."""

    q0 = clean(q, time)
    q1 = clean(q0 + dt * rhs(q0, time), time + dt)
    q2 = clean(
        0.75 * q0 + 0.25 * (q1 + dt * rhs(q1, time + dt)),
        time + 0.5 * dt,
    )
    return clean(
        (1.0 / 3.0) * q0
        + (2.0 / 3.0) * (q2 + dt * rhs(q2, time + 0.5 * dt)),
        time + dt,
    )


def _weno_fraction(sensor_mask, scheme: str) -> float:
    xp = array_module(sensor_mask)
    if scheme == "weno":
        return 1.0
    weno_x = interface_mask_from_nodes(sensor_mask)
    weno_y = interface_mask_from_nodes(xp.moveaxis(sensor_mask, -2, -1))
    return (float(xp.sum(weno_x)) + float(xp.sum(weno_y))) / (
        weno_x.size + weno_y.size
    )


def interpolate_to_y_zero(values, y_coordinates):
    """Linearly interpolate a ``(..., ny, nx)`` field to the line y=0."""

    y = np.asarray(y_coordinates, dtype=float)
    if y.ndim != 1 or y.size != values.shape[-2]:
        raise ValueError("y_coordinates must match the field's penultimate axis")
    exact = np.flatnonzero(np.isclose(y, 0.0, atol=1.0e-14, rtol=0.0))
    if exact.size:
        return values[..., int(exact[0]), :]
    upper = int(np.searchsorted(y, 0.0))
    if upper <= 0 or upper >= y.size:
        raise ValueError("The domain does not bracket y=0")
    lower = upper - 1
    weight = (0.0 - y[lower]) / (y[upper] - y[lower])
    return (1.0 - weight) * values[..., lower, :] + weight * values[..., upper, :]


def _diagnostics(
    q,
    config: ShockShearLayerConfig,
    time: float,
    dt: float,
    steps: int,
    sensor,
    hyperviscosity_applications: int,
) -> ShockShearLayerDiagnostics:
    xp = array_module(q)
    rho, pressure = raw_density_pressure(q, config.equation)
    _rho_safe, u, v, _p_safe = config.equation.primitive_from_conservative(q)
    sound_speed = xp.sqrt(config.gamma * pressure / rho)
    mach = xp.sqrt(u**2 + v**2) / sound_speed
    omega = vorticity_z(q, config.domain, config.equation)
    sensor_mask = sensor.detect(q)
    centerline = interpolate_to_y_zero(rho, config.domain.y)
    has_nonfinite = bool(
        xp.any(~xp.isfinite(q))
        | xp.any(~xp.isfinite(rho))
        | xp.any(~xp.isfinite(pressure))
    )
    return ShockShearLayerDiagnostics(
        backend=config.backend,
        scheme=config.scheme,
        viscosity_model=config.viscosity_model,
        reynolds=config.reynolds,
        prandtl=config.prandtl,
        fixed_dt=dt,
        steps=steps,
        time=time,
        hyperviscosity_enabled=hyperviscosity_enabled(config),
        hyperviscosity_applications=hyperviscosity_applications,
        final_sensor_fraction=float(xp.mean(sensor_mask)),
        final_weno_fraction=_weno_fraction(sensor_mask, config.scheme),
        density_min=float(xp.min(rho)),
        density_max=float(xp.max(rho)),
        pressure_min=float(xp.min(pressure)),
        pressure_max=float(xp.max(pressure)),
        mach_min=float(xp.min(mach)),
        mach_max=float(xp.max(mach)),
        max_abs_vorticity=float(xp.max(xp.abs(omega))),
        centerline_density_min=float(xp.min(centerline)),
        centerline_density_max=float(xp.max(centerline)),
        has_nonfinite=has_nonfinite,
        density_positive=bool(xp.all(rho > 0.0)),
        pressure_positive=bool(xp.all(pressure > 0.0)),
    )


def print_diagnostic_summary(diagnostics: ShockShearLayerDiagnostics) -> None:
    print("Diagnostic summary:")
    print(f"  backend               : {diagnostics.backend}")
    print(f"  scheme                : {diagnostics.scheme}")
    print(f"  viscosity model       : {diagnostics.viscosity_model}")
    print(f"  Re / Pr               : {diagnostics.reynolds:g} / {diagnostics.prandtl:g}")
    print(f"  fixed dt / steps      : {diagnostics.fixed_dt:.8e} / {diagnostics.steps}")
    print(f"  density min/max       : {diagnostics.density_min:.8e} / {diagnostics.density_max:.8e}")
    print(f"  pressure min/max      : {diagnostics.pressure_min:.8e} / {diagnostics.pressure_max:.8e}")
    print(f"  Mach min/max          : {diagnostics.mach_min:.8e} / {diagnostics.mach_max:.8e}")
    print(f"  max |vorticity|       : {diagnostics.max_abs_vorticity:.8e}")
    print(f"  final sensor fraction : {diagnostics.final_sensor_fraction:.6f}")
    print(f"  final WENO fraction   : {diagnostics.final_weno_fraction:.6f}")
    print(
        "  hyperviscosity        : "
        f"{diagnostics.hyperviscosity_enabled} "
        f"({diagnostics.hyperviscosity_applications} applications)"
    )
    print(f"  density positive      : {diagnostics.density_positive}")
    print(f"  pressure positive     : {diagnostics.pressure_positive}")
    print(f"  NaN/Inf present       : {diagnostics.has_nonfinite}")


def run_shock_shear_layer(config: ShockShearLayerConfig):
    xp = select_array_module(config.backend)
    print(f"Using array backend: {xp.__name__}")
    print(
        "Reference inflow Mach numbers: "
        f"upper={config.upper_inflow_mach:.4f}, "
        f"lower={config.lower_inflow_mach:.4f}"
    )

    boundary = ShockShearBoundary2D(config)
    q = boundary.apply(config.initial_state(), 0.0)
    validate_raw_physical_state(q, config.equation, "initial state")

    sensor = NonPeriodicEulerShockSensor2D(
        config.domain,
        config.equation,
        width=config.sensor_width,
        compression_threshold=config.compression_threshold,
        jump_threshold=config.jump_threshold,
        shear_threshold=config.shear_threshold,
        boundary_guard=config.boundary_guard,
    )
    inviscid_operator = NonPeriodicHybridEuler2DOperator(
        config.domain,
        config.equation,
        sensor,
        scheme=config.scheme,
    )
    viscous_operator = NonPeriodicViscousRHS2D(config)
    dt, n_steps, dt_convective, dt_diffusive = _fixed_time_step(
        q,
        config,
        inviscid_operator,
        viscous_operator,
    )

    use_hyperviscosity = hyperviscosity_enabled(config)
    numerical_filter = (
        LocalHyperviscosity2D(
            config.domain,
            mn=config.mn,
            density_weight=config.hyperviscosity_density_weight,
            momentum_weight=config.hyperviscosity_momentum_weight,
            energy_weight=config.hyperviscosity_energy_weight,
        )
        if use_hyperviscosity
        else None
    )

    print(
        "Shock--shear layer: "
        f"nx={config.nx}, ny={config.ny}, tfinal={config.tfinal}, "
        f"fixed dt={dt:.8e}, steps={n_steps}, initial CFL={config.cfl}"
    )
    print(
        f"initial dt limits: convective={dt_convective:.8e}, "
        f"diffusive={dt_diffusive:.8e}"
    )
    if config.scheme == "weno":
        print("scheme=weno; numerical hyperviscosity disabled (no compact nodes)")
    else:
        print(
            f"scheme=hybrid; compact-node hyperviscosity="
            f"{use_hyperviscosity} (mn={config.mn})"
        )

    def clean(state, stage_time: float):
        return boundary.apply(config.equation.enforce_physical_state(state), stage_time)

    def rhs(state, stage_time: float):
        bounded = boundary.apply(state, stage_time)
        return inviscid_operator.rhs(bounded) + viscous_operator(bounded)

    time = 0.0
    hyperviscosity_applications = 0
    for step_index in range(n_steps):
        q = _time_dependent_ssprk3_step(q, time, dt, rhs, clean)
        step = step_index + 1
        time = step * dt

        if use_hyperviscosity and step % config.hyperviscosity_interval == 0:
            weno_mask = sensor.detect(q)
            q = numerical_filter.apply(q, ~weno_mask, config.equation)
            q = boundary.apply(q, time)
            hyperviscosity_applications += 1

        if config.progress_every > 0 and (
            step % config.progress_every == 0 or step == n_steps
        ):
            validate_raw_physical_state(q, config.equation, f"state at step {step}")
            print(f"step={step:6d}, time={time:.6f}, dt={dt:.8e}")

    validate_raw_physical_state(q, config.equation, "final state")
    diagnostics = _diagnostics(
        q,
        config,
        time,
        dt,
        n_steps,
        sensor,
        hyperviscosity_applications,
    )
    print_diagnostic_summary(diagnostics)
    return q, diagnostics


def shock_shear_config_dict(config: ShockShearLayerConfig) -> dict:
    values = asdict(config)
    values.update(
        {
            "problem": config.scenario_name,
            "reference": config.reference_name,
            "upper_inflow_mach_computed": config.upper_inflow_mach,
            "lower_inflow_mach_computed": config.lower_inflow_mach,
            "forcing_period": config.forcing_period,
            "spatial_discretization": "hybrid_compact_weno7"
            if config.scheme == "hybrid"
            else "weno7_only",
            "time_integration": "fixed_dt_ssp_rk3",
            "hyperviscosity_policy": "compact_nodes_only",
            "viscosity_normalization_note": (
                "mu_ref=1/Re; for Sutherland viscosity the upper-stream "
                "temperature is the nondimensional reference mapped to "
                f"{config.reference_temperature_kelvin:g} K"
            ),
        }
    )
    return values


def save_shock_shear_npz(
    path: Path,
    q,
    config: ShockShearLayerConfig,
    diagnostics: ShockShearLayerDiagnostics,
) -> Path:
    rho, u, v, pressure = config.equation.primitive_from_conservative(q)
    omega = vorticity_z(q, config.domain, config.equation)
    schlieren = schlieren_field(rho, config.domain, k=20.0)
    centerline_density = interpolate_to_y_zero(rho, config.domain.y)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=config.domain.x,
        y=config.domain.y,
        q=to_numpy(q),
        rho=to_numpy(rho),
        u=to_numpy(u),
        v=to_numpy(v),
        pressure=to_numpy(pressure),
        omega_z=to_numpy(omega),
        schlieren=to_numpy(schlieren),
        centerline_x=config.domain.x,
        centerline_density=to_numpy(centerline_density),
        centerline_y=np.asarray(0.0),
        time=np.asarray(diagnostics.time),
        steps=np.asarray(diagnostics.steps),
        config_json=np.asarray(json.dumps(shock_shear_config_dict(config), default=str)),
        diagnostics_json=np.asarray(json.dumps(asdict(diagnostics), default=str)),
    )
    return path


def plot_density_contours(
    rho,
    config: ShockShearLayerConfig,
    time: float,
    output_path: Path,
    num_contours: int = 31,
    density_limits: tuple[float, float] = (0.4, 2.60),
    show: bool = False,
) -> Path:
    rho = np.asarray(rho)
    levels = np.linspace(density_limits[0], density_limits[1], num_contours)
    fig, ax = plt.subplots(figsize=(12.0, 3.2), constrained_layout=True)
    contour = ax.contourf(
        config.domain.x,
        config.domain.y,
        rho,
        levels=levels,
        extend="both",
        cmap="viridis",
    )
    ax.contour(
        config.domain.x,
        config.domain.y,
        rho,
        levels=levels,
        colors="black",
        linewidths=0.18,
        alpha=0.35,
    )
    ax.set_title(f"Shock--shear layer density at t={time:.1f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(config.x_min, config.x_max)
    ax.set_ylim(config.y_min, config.y_max)
    ax.set_aspect("auto")
    fig.colorbar(contour, ax=ax, label="Density")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)
    return output_path


def plot_centerline_density(
    rho,
    config: ShockShearLayerConfig,
    time: float,
    output_path: Path,
    show: bool = False,
) -> Path:
    rho = np.asarray(rho)
    centerline = interpolate_to_y_zero(rho, config.domain.y)
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    ax.plot(config.domain.x, centerline, linewidth=1.25)
    ax.set_title(f"Density near the centerline at t={time:.1f}")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_xlim(config.x_min, config.x_max)
    ax.grid(True, alpha=0.25)
    ax.text(
        0.02,
        0.96,
        "linearly interpolated to y=0",
        transform=ax.transAxes,
        va="top",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)
    return output_path


def plot_fields(
    q,
    config: ShockShearLayerConfig,
    time: float,
    output_path: Path,
    show: bool = False,
    vorticity_limit: float | None = None,
) -> Path:
    rho, _u, _v, pressure = config.equation.primitive_from_conservative(q)
    omega = vorticity_z(q, config.domain, config.equation)
    schlieren = schlieren_field(rho, config.domain, k=20.0)
    rho = to_numpy(rho)
    pressure = to_numpy(pressure)
    omega = to_numpy(omega)
    schlieren = to_numpy(schlieren)

    if vorticity_limit is None:
        vorticity_limit = float(np.nanpercentile(np.abs(omega), 99.5))
        vorticity_limit = max(vorticity_limit, 1.0e-12)

    extent = [config.x_min, config.x_max, config.y_min, config.y_max]
    fields = [
        (rho, "Density", "viridis", None, None),
        (pressure, "Pressure", "viridis", None, None),
        (omega, "Vorticity", "coolwarm", -vorticity_limit, vorticity_limit),
        (schlieren, "Schlieren", "gray", 0.0, 1.0),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(12.0, 10.0), constrained_layout=True)
    for ax, (values, title, cmap, vmin, vmax) in zip(axes, fields):
        image = ax.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax)
    fig.suptitle(f"Shock--shear layer at t={time:.1f}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)
    return output_path


def save_run_outputs(
    q,
    config: ShockShearLayerConfig,
    diagnostics: ShockShearLayerDiagnostics,
    output_dir: Path,
    run_id: str | None = None,
    plot: bool = True,
    show: bool = False,
    num_contours: int = 31,
    density_limits: tuple[float, float] = (0.4, 2.60),
    vorticity_limit: float | None = None,
) -> dict[str, Path]:
    run_id = run_id or make_run_id(f"shock_shear_{config.scheme}")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "config.json", shock_shear_config_dict(config))
    write_json(run_dir / "diagnostics.json", asdict(diagnostics))
    paths: dict[str, Path] = {
        "config": run_dir / "config.json",
        "diagnostics": run_dir / "diagnostics.json",
    }
    npz_path = run_dir / f"{run_id}_final.npz"
    save_shock_shear_npz(npz_path, q, config, diagnostics)
    paths["npz"] = npz_path

    if plot:
        rho, _u, _v, _pressure = config.equation.primitive_from_conservative(q)
        rho_np = to_numpy(rho)
        paths["density_contours"] = plot_density_contours(
            rho_np,
            config,
            diagnostics.time,
            run_dir / f"{run_id}_density_contours.png",
            num_contours=num_contours,
            density_limits=density_limits,
            show=show,
        )
        paths["centerline_density"] = plot_centerline_density(
            rho_np,
            config,
            diagnostics.time,
            run_dir / f"{run_id}_centerline_density.png",
            show=show,
        )
        paths["fields"] = plot_fields(
            q,
            config,
            diagnostics.time,
            run_dir / f"{run_id}_fields.png",
            show=show,
            vorticity_limit=vorticity_limit,
        )
    return paths


def plot_from_npz(
    npz_path: Path,
    output_dir: Path | None = None,
    num_contours: int = 31,
    density_limits: tuple[float, float] = (0.4, 2.60),
    show: bool = False,
) -> dict[str, Path]:
    with np.load(npz_path, allow_pickle=False) as data:
        rho = data["rho"]
        q = data["q"]
        time = float(data["time"])
        config_data = json.loads(str(data["config_json"]))
    fields = ShockShearLayerConfig.__dataclass_fields__
    config = ShockShearLayerConfig(
        **{key: value for key, value in config_data.items() if key in fields}
    )
    output_dir = output_dir or npz_path.parent
    stem = npz_path.stem.replace("_final", "")
    paths = {
        "density_contours": plot_density_contours(
            rho,
            config,
            time,
            output_dir / f"{stem}_density_contours.png",
            num_contours=num_contours,
            density_limits=density_limits,
            show=show,
        ),
        "centerline_density": plot_centerline_density(
            rho,
            config,
            time,
            output_dir / f"{stem}_centerline_density.png",
            show=show,
        ),
    }
    # Recreate the four-field figure using a NumPy-backed configuration.
    numpy_config = ShockShearLayerConfig(
        **{
            **{key: value for key, value in config_data.items() if key in fields},
            "backend": "numpy",
        }
    )
    paths["fields"] = plot_fields(
        np.asarray(q),
        numpy_config,
        time,
        output_dir / f"{stem}_fields.png",
        show=show,
    )
    return paths
