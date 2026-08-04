from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np

from OOP.domain import Domain2D
from OOP.parallel.backend import array_module, select_array_module, to_numpy
from OOP.parallel.equations import ParallelEulerEquation2D
from OOP.time_operator import SSPRK3
from OOP.spatial_operator import weno7_flux
from OOP.run_utils import make_run_id, write_json


@dataclass(frozen=True)
class RiemannConfig3:
    """2D Euler Riemann problem, Lax-Liu/Kurganov-Tadmor Configuration 3."""

    configuration_number: ClassVar[int] = 3

    nx: int = 512
    ny: int = 512
    tfinal: float = 0.3
    cfl: float = 0.4
    gamma: float = 1.4
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    x_split: float = 0.5
    y_split: float = 0.5
    backend: str = "numpy"
    scheme: str = "hybrid"  # hybrid or weno
    sensor_width: int = 4
    jump_threshold: float = 0.025
    compression_threshold: float = 2.5
    shear_threshold: float | None = 0.0
    boundary_guard: int = 4
    mn: float = 0.001
    hyperviscosity_interval: int = 1
    hyperviscosity_density_weight: float = 1.0
    hyperviscosity_momentum_weight: float = 1.0
    hyperviscosity_energy_weight: float = 1.0
    progress_every: int = 500
    guard_cells: int = 2
    save_final_npz: bool = True

    @property
    def domain(self) -> Domain2D:
        return Domain2D(self.x_min, self.x_max, self.y_min, self.y_max, self.nx, self.ny)

    @property
    def equation(self) -> ParallelEulerEquation2D:
        return ParallelEulerEquation2D(gamma=self.gamma)

    def initial_state(self):
        xp = select_array_module(self.backend)
        domain = self.domain
        x = xp.asarray(domain.x)
        y = xp.asarray(domain.y)
        X, Y = xp.meshgrid(x, y, indexing="xy")

        rho = xp.empty((self.ny, self.nx), dtype=xp.float64)
        u = xp.empty_like(rho)
        v = xp.empty_like(rho)
        p = xp.empty_like(rho)

        top = Y >= self.y_split
        right = X >= self.x_split

        # Quadrant ordering used in Lax-Liu Configuration 3.
        rho[...] = 0.138
        u[...] = 1.206
        v[...] = 1.206
        p[...] = 0.029

        mask = top & right
        rho[mask] = 1.5
        u[mask] = 0.0
        v[mask] = 0.0
        p[mask] = 1.5

        mask = top & (~right)
        rho[mask] = 0.5323
        u[mask] = 1.206
        v[mask] = 0.0
        p[mask] = 0.3

        mask = (~top) & right
        rho[mask] = 0.5323
        u[mask] = 0.0
        v[mask] = 1.206
        p[mask] = 0.3

        return self.equation.conservative_from_primitive(rho, u, v, p)


@dataclass(frozen=True)
class RiemannDiagnostics:
    backend: str
    scheme: str
    hyperviscosity_enabled: bool
    hyperviscosity_applications: int
    final_sensor_fraction: float
    final_weno_fraction: float
    density_min: float
    density_max: float
    pressure_min: float
    pressure_max: float
    max_abs_vorticity: float
    has_nonfinite: bool
    density_positive: bool
    pressure_positive: bool
    time: float
    steps: int


def take_clipped(values, indices, axis: int = -1):
    xp = array_module(values)
    n = values.shape[axis]
    idx = xp.clip(xp.asarray(indices), 0, n - 1).astype(xp.int64)
    return xp.take(values, idx, axis=axis)


def relative_jump_axis(values, axis: int):
    xp = array_module(values)
    moved = xp.moveaxis(values, axis, -1)
    eps = 1e-14 + 1e-12 * xp.max(xp.abs(moved))
    pair_jump = xp.abs(moved[..., 1:] - moved[..., :-1])
    pair_jump = pair_jump / (xp.abs(moved[..., 1:]) + xp.abs(moved[..., :-1]) + eps)

    jumps = xp.zeros_like(moved)
    jumps[..., :-1] = xp.maximum(jumps[..., :-1], pair_jump)
    jumps[..., 1:] = xp.maximum(jumps[..., 1:], pair_jump)

    center = moved
    idx = xp.arange(moved.shape[-1])
    left = take_clipped(moved, idx - 1)
    right = take_clipped(moved, idx + 1)
    curvature = xp.abs(right - 2.0 * center + left)
    curvature = curvature / (xp.abs(right) + 2.0 * xp.abs(center) + xp.abs(left) + eps)
    jumps = xp.maximum(jumps, curvature)
    return xp.moveaxis(jumps, -1, axis)


def dilate_mask_2d(mask, width: int):
    xp = array_module(mask)
    expanded = xp.array(mask, copy=True)
    ny, nx = mask.shape
    for y_offset in range(-width, width + 1):
        for x_offset in range(-width, width + 1):
            if x_offset == 0 and y_offset == 0:
                continue
            y_src = slice(max(0, -y_offset), min(ny, ny - y_offset))
            y_dst = slice(max(0, y_offset), min(ny, ny + y_offset))
            x_src = slice(max(0, -x_offset), min(nx, nx - x_offset))
            x_dst = slice(max(0, x_offset), min(nx, nx + x_offset))
            expanded[y_dst, x_dst] = expanded[y_dst, x_dst] | mask[y_src, x_src]
    return expanded


def interface_mask_from_nodes(mask):
    n = mask.shape[-1]
    xp = array_module(mask)
    interface_index = xp.arange(n + 1)
    left = take_clipped(mask, interface_index - 1)
    right = take_clipped(mask, interface_index)
    return left | right


def smooth_compact_interfaces(point_flux):
    xp = array_module(point_flux)
    a1_c = 25 / 32
    b1_c = 1 / 20
    c1_c = -1 / 480
    n = point_flux.shape[-1]
    i_left = xp.arange(-1, n)
    return (
        c1_c * (take_clipped(point_flux, i_left + 3) + take_clipped(point_flux, i_left - 2))
        + (b1_c + c1_c) * (take_clipped(point_flux, i_left + 2) + take_clipped(point_flux, i_left - 1))
        + (a1_c + b1_c + c1_c) * (take_clipped(point_flux, i_left + 1) + take_clipped(point_flux, i_left))
    )


def weno7_interfaces(q, point_flux, alpha: float):
    xp = array_module(q)
    n = q.shape[-1]
    i_left = xp.arange(-1, n)
    f_plus = 0.5 * (point_flux + alpha * q)
    f_minus = 0.5 * (point_flux - alpha * q)
    plus_flux = weno7_flux(
        take_clipped(f_plus, i_left - 3),
        take_clipped(f_plus, i_left - 2),
        take_clipped(f_plus, i_left - 1),
        take_clipped(f_plus, i_left),
        take_clipped(f_plus, i_left + 1),
        take_clipped(f_plus, i_left + 2),
        take_clipped(f_plus, i_left + 3),
    )
    minus_flux = weno7_flux(
        take_clipped(f_minus, i_left + 4),
        take_clipped(f_minus, i_left + 3),
        take_clipped(f_minus, i_left + 2),
        take_clipped(f_minus, i_left + 1),
        take_clipped(f_minus, i_left),
        take_clipped(f_minus, i_left - 1),
        take_clipped(f_minus, i_left - 2),
    )
    return plus_flux + minus_flux


def compactify_weno_interfaces(weno_flux, alpha: float = 3.0 / 8.0):
    xp = array_module(weno_flux)
    interface_index = xp.arange(weno_flux.shape[-1])
    return (
        alpha * take_clipped(weno_flux, interface_index + 1)
        + weno_flux
        + alpha * take_clipped(weno_flux, interface_index - 1)
    )


@dataclass
class NonPeriodicLineCompactDerivative:
    """Non-periodic compact derivative solve along the last axis.

    The solve is implemented with a batched Thomas algorithm, so it works with
    NumPy arrays and CuPy arrays without SciPy sparse matrices.
    """

    n: int
    dx: float
    alpha: float = 3.0 / 8.0

    def from_interface_flux(self, interface_flux):
        xp = array_module(interface_flux)
        raw = (interface_flux[..., 1:] - interface_flux[..., :-1]) / self.dx
        leading_shape = raw.shape[:-1]
        rhs = raw.reshape((-1, self.n))
        m = rhs.shape[0]

        lower = xp.full(self.n, self.alpha, dtype=raw.dtype)
        diag = xp.ones(self.n, dtype=raw.dtype)
        upper = xp.full(self.n, self.alpha, dtype=raw.dtype)
        lower[0] = 0.0
        upper[0] = 0.0
        lower[-1] = 0.0
        upper[-1] = 0.0

        cprime = xp.zeros(self.n, dtype=raw.dtype)
        dprime = xp.zeros((m, self.n), dtype=raw.dtype)
        cprime[0] = upper[0] / diag[0]
        dprime[:, 0] = rhs[:, 0] / diag[0]
        for i in range(1, self.n):
            denom = diag[i] - lower[i] * cprime[i - 1]
            cprime[i] = upper[i] / denom if i < self.n - 1 else 0.0
            dprime[:, i] = (rhs[:, i] - lower[i] * dprime[:, i - 1]) / denom

        out = xp.zeros_like(dprime)
        out[:, -1] = dprime[:, -1]
        for i in range(self.n - 2, -1, -1):
            out[:, i] = dprime[:, i] - cprime[i] * out[:, i + 1]
        return out.reshape((*leading_shape, self.n))


def finite_difference(values, spacing: float, axis: int):
    xp = array_module(values)
    moved = xp.moveaxis(values, axis, -1)
    derivative = xp.empty_like(moved)
    derivative[..., 1:-1] = (moved[..., 2:] - moved[..., :-2]) / (2.0 * spacing)
    derivative[..., 0] = (moved[..., 1] - moved[..., 0]) / spacing
    derivative[..., -1] = (moved[..., -1] - moved[..., -2]) / spacing
    return xp.moveaxis(derivative, -1, axis)


@dataclass(frozen=True)
class NonPeriodicEulerShockSensor2D:
    domain: Domain2D
    equation: ParallelEulerEquation2D
    width: int = 4
    compression_threshold: float = 2.5
    jump_threshold: float = 0.04
    shear_threshold: float | None = 2.0
    boundary_guard: int = 4

    def detect(self, q):
        xp = array_module(q)
        rho, u, v, pressure = self.equation.primitive_from_conservative(q)
        du_dx = finite_difference(u, self.domain.dx, axis=-1)
        du_dy = finite_difference(u, self.domain.dy, axis=-2)
        dv_dx = finite_difference(v, self.domain.dx, axis=-1)
        dv_dy = finite_difference(v, self.domain.dy, axis=-2)
        divergence = du_dx + dv_dy
        div_rms = float(xp.sqrt(xp.mean(divergence**2)))

        compression = xp.zeros_like(divergence, dtype=bool)
        if div_rms > 1e-12:
            compression = divergence < -self.compression_threshold * div_rms

        shear = xp.zeros_like(divergence, dtype=bool)
        if self.shear_threshold is not None and self.shear_threshold > 0.0:
            vorticity = dv_dx - du_dy
            vort_rms = float(xp.sqrt(xp.mean(vorticity**2)))
            if vort_rms > 1e-12:
                shear = xp.abs(vorticity) > self.shear_threshold * vort_rms

        internal_energy = pressure / (rho * (self.equation.gamma - 1.0))
        density_jump = xp.maximum(relative_jump_axis(rho, -1), relative_jump_axis(rho, -2))
        pressure_jump = xp.maximum(relative_jump_axis(pressure, -1), relative_jump_axis(pressure, -2))
        energy_jump = xp.maximum(relative_jump_axis(internal_energy, -1), relative_jump_axis(internal_energy, -2))
        mask = (
            compression
            | shear
            | (density_jump > self.jump_threshold)
            | (pressure_jump > self.jump_threshold)
            | (energy_jump > self.jump_threshold)
        )
        if self.boundary_guard:
            g = self.boundary_guard
            mask[:g, :] = True
            mask[-g:, :] = True
            mask[:, :g] = True
            mask[:, -g:] = True
        return dilate_mask_2d(mask, self.width)


@dataclass
class NonPeriodicHybridEuler2DOperator:
    domain: Domain2D
    equation: ParallelEulerEquation2D
    sensor: NonPeriodicEulerShockSensor2D
    scheme: str = "hybrid"
    compact_x: NonPeriodicLineCompactDerivative = None
    compact_y: NonPeriodicLineCompactDerivative = None

    def __post_init__(self):
        if self.scheme not in {"hybrid", "weno"}:
            raise ValueError("scheme must be 'hybrid' or 'weno'")
        self.compact_x = NonPeriodicLineCompactDerivative(self.domain.nx, self.domain.dx)
        self.compact_y = NonPeriodicLineCompactDerivative(self.domain.ny, self.domain.dy)

    def _axis_derivative(self, q_axis, flux_axis, shock_axis, compact, normal_velocity_index: int):
        xp = array_module(q_axis)
        rho, u, v, pressure = self.equation.primitive_from_conservative(q_axis)
        normal_velocity = u if normal_velocity_index == 1 else v
        sound_speed = xp.sqrt(self.equation.gamma * pressure / rho)
        alpha = float(xp.max(xp.abs(normal_velocity) + sound_speed))
        smooth_flux = smooth_compact_interfaces(flux_axis)
        shock_flux = compactify_weno_interfaces(weno7_interfaces(q_axis, flux_axis, alpha), alpha=compact.alpha)
        if self.scheme == "weno":
            hybrid_flux = shock_flux
        else:
            if shock_axis is None:
                raise ValueError("shock_axis is required in hybrid mode")
            shock_interfaces = interface_mask_from_nodes(shock_axis)
            hybrid_flux = xp.where(shock_interfaces[None, ...], shock_flux, smooth_flux)
        return compact.from_interface_flux(hybrid_flux)

    def rhs(self, q):
        xp = array_module(q)
        q_safe = self.equation.enforce_physical_state(q)

        # The shock sensor is irrelevant in WENO-only mode: WENO is used at
        # every interface.  Skipping it avoids unnecessary work and makes the
        # numerical path independent of the sensor parameters.
        if self.scheme == "weno":
            derivative_x = self._axis_derivative(
                q_safe,
                self.equation.flux_x(q_safe),
                None,
                self.compact_x,
                1,
            )
            q_y = xp.moveaxis(q_safe, -2, -1)
            flux_y = xp.moveaxis(self.equation.flux_y(q_safe), -2, -1)
            derivative_y = self._axis_derivative(
                q_y,
                flux_y,
                None,
                self.compact_y,
                2,
            )
        else:
            shock_mask = self.sensor.detect(q_safe)
            derivative_x = self._axis_derivative(
                q_safe,
                self.equation.flux_x(q_safe),
                shock_mask,
                self.compact_x,
                1,
            )
            q_y = xp.moveaxis(q_safe, -2, -1)
            flux_y = xp.moveaxis(self.equation.flux_y(q_safe), -2, -1)
            shock_y = xp.moveaxis(shock_mask, -2, -1)
            derivative_y = self._axis_derivative(
                q_y,
                flux_y,
                shock_y,
                self.compact_y,
                2,
            )

        derivative_y = xp.moveaxis(derivative_y, -1, -2)
        return -derivative_x - derivative_y

    def fixed_time_step(self, q, cfl: float, t_end: float) -> tuple[float, int]:
        """Compute one constant time step from the initial solution state."""
        if cfl <= 0.0:
            raise ValueError("cfl must be positive")
        xp = array_module(q)
        rho, u, v, pressure = self.equation.primitive_from_conservative(q)
        c = xp.sqrt(self.equation.gamma * pressure / rho)
        spectral_radius = (
            (xp.abs(u) + c) / self.domain.dx
            + (xp.abs(v) + c) / self.domain.dy
        )
        maximum_radius = float(xp.max(spectral_radius))
        if not np.isfinite(maximum_radius) or maximum_radius <= 0.0:
            raise FloatingPointError(
                f"Invalid maximum spectral radius: {maximum_radius}"
            )
        initial_dt = cfl / maximum_radius
        n_steps = int(np.ceil(t_end / initial_dt))
        return t_end / n_steps, n_steps

@dataclass(frozen=True)
class LocalHyperviscosity2D:
    """Biharmonic filter applied only where the compact scheme is active."""
    domain: Domain2D
    mn: float = 0.001
    density_weight: float = 1.0
    momentum_weight: float = 1.0
    energy_weight: float = 1.0

    def _laplacian(self, values):
        xp = array_module(values)
        padded = xp.pad(values, ((0, 0), (1, 1), (1, 1)), mode="edge")
        center = padded[:, 1:-1, 1:-1]
        lap_x = (padded[:, 1:-1, 2:] - 2.0 * center + padded[:, 1:-1, :-2]) / self.domain.dx**2
        lap_y = (padded[:, 2:, 1:-1] - 2.0 * center + padded[:, :-2, 1:-1]) / self.domain.dy**2
        return lap_x + lap_y

    def apply(self, q, compact_mask, equation):
        xp = array_module(q)
        if compact_mask.shape != q.shape[-2:]:
            raise ValueError(
                f"compact_mask shape {compact_mask.shape} does not match grid {q.shape[-2:]}"
            )
        h = min(self.domain.dx, self.domain.dy)
        biharmonic = self._laplacian(self._laplacian(q))
        weights = xp.array(
            [
                self.density_weight,
                self.momentum_weight,
                self.momentum_weight,
                self.energy_weight,
            ],
            dtype=q.dtype,
        )[:, None, None]
        filtered = q - self.mn * weights * h**4 * compact_mask[None, :, :] * biharmonic
        return equation.enforce_physical_state(filtered)


def apply_outflow_guard(q, guard_cells: int = 2):
    xp = array_module(q)
    if guard_cells <= 0:
        return q
    guarded = xp.array(q, copy=True)
    g = guard_cells
    guarded[:, :, :g] = guarded[:, :, g][:, :, None]
    guarded[:, :, -g:] = guarded[:, :, -g - 1][:, :, None]
    guarded[:, :g, :] = guarded[:, g, :][:, None, :]
    guarded[:, -g:, :] = guarded[:, -g - 1, :][:, None, :]
    return guarded


def raw_density_pressure(q, equation):
    xp = array_module(q)
    rho = q[0]
    u = q[1] / rho
    v = q[2] / rho
    kinetic = 0.5 * rho * (u**2 + v**2)
    pressure = (equation.gamma - 1.0) * (q[3] - kinetic)
    return rho, pressure


def validate_raw_physical_state(q, equation, label: str = "state") -> None:
    xp = array_module(q)
    if bool(xp.any(~xp.isfinite(q))):
        raise FloatingPointError(f"{label} contains NaN or Inf values")
    rho, pressure = raw_density_pressure(q, equation)
    if bool(xp.any(~xp.isfinite(rho))) or bool(xp.any(~xp.isfinite(pressure))):
        raise FloatingPointError(f"{label} has non-finite density or pressure")
    if bool(xp.any(rho <= 0.0)):
        raise FloatingPointError(f"{label} contains non-positive density")
    if bool(xp.any(pressure <= 0.0)):
        raise FloatingPointError(f"{label} contains non-positive pressure")


def primitive_fields(q, equation):
    return equation.primitive_from_conservative(q)


def vorticity_z(q, domain: Domain2D, equation):
    rho, u, v, _p = equation.primitive_from_conservative(q)
    dv_dx = finite_difference(v, domain.dx, axis=-1)
    du_dy = finite_difference(u, domain.dy, axis=-2)
    return dv_dx - du_dy


def density_gradient_magnitude(rho, domain: Domain2D):
    grad_x = finite_difference(rho, domain.dx, axis=-1)
    grad_y = finite_difference(rho, domain.dy, axis=-2)
    xp = array_module(rho)
    return xp.sqrt(grad_x**2 + grad_y**2)


def schlieren_field(rho, domain: Domain2D, k: float = 20.0):
    xp = array_module(rho)
    grad_rho = density_gradient_magnitude(rho, domain)
    max_grad = float(xp.max(grad_rho))
    if max_grad <= 0.0 or not np.isfinite(max_grad):
        return xp.ones_like(rho)
    return xp.exp(-k * grad_rho / max_grad)


def _extent_from_domain(domain: Domain2D):
    return [domain.x_min, domain.x_max, domain.y_min, domain.y_max]


def riemann_config_dict(config: RiemannConfig3) -> dict:
    values = {"configuration_number": config.configuration_number, **asdict(config)}
    scenario_name = getattr(config, "scenario_name", None)
    if scenario_name is not None:
        values["scenario_name"] = scenario_name
    configuration_label = getattr(config, "configuration_label", None)
    if configuration_label is not None:
        values["configuration_label"] = configuration_label
    return values


def save_riemann_npz(path: Path, q, config: RiemannConfig3, diagnostics: RiemannDiagnostics) -> Path:
    domain = config.domain
    equation = config.equation
    rho, u, v, pressure = primitive_fields(q, equation)
    omega = vorticity_z(q, domain, equation)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=domain.x,
        y=domain.y,
        q=to_numpy(q),
        rho=to_numpy(rho),
        u=to_numpy(u),
        v=to_numpy(v),
        pressure=to_numpy(pressure),
        omega_z=to_numpy(omega),
        time=np.asarray(diagnostics.time),
        steps=np.asarray(diagnostics.steps),
        configuration_number=np.asarray(config.configuration_number),
        config_json=np.asarray(json.dumps(riemann_config_dict(config), default=str)),
        diagnostics_json=np.asarray(json.dumps(asdict(diagnostics), default=str)),
    )
    return path


def load_riemann_npz(path: Path):
    data = np.load(path, allow_pickle=False)
    config_data = json.loads(str(data["config_json"])) if "config_json" in data else {}
    fields = RiemannConfig3.__dataclass_fields__
    config = RiemannConfig3(
        **{
            key: value
            for key, value in config_data.items()
            if key != "configuration_number" and key in fields
        }
    )
    return data, config


def plot_riemann_solution(
    rho,
    pressure,
    omega,
    x,
    y,
    time: float,
    output_path: Path | None = None,
    show: bool = True,
    vorticity_limit: float | None = 100.0,
    vorticity_cmap: str = "coolwarm",
    density_contours: bool = False,
    num_contours: int = 40,
    schlieren_values=None,
    zoom_center: bool = False,
    zoom_window: float = 0.35,
    zoom_x: float = 0.375,
    zoom_y: float = 0.375,
    fixed_density_limits: tuple[float, float] | None = None,
    fixed_pressure_limits: tuple[float, float] | None = None,
    configuration_number: int = 3,
    configuration_label: str | None = None,
):
    rho = np.asarray(rho)
    pressure = np.asarray(pressure)
    omega = np.asarray(omega)
    x = np.asarray(x)
    y = np.asarray(y)
    dx = float(x[1] - x[0]) if x.size > 1 else 1.0
    dy = float(y[1] - y[0]) if y.size > 1 else 1.0
    extent = [float(x[0] - 0.5 * dx), float(x[-1] + 0.5 * dx), float(y[0] - 0.5 * dy), float(y[-1] + 0.5 * dy)]

    fields = [
        (rho, "Density", "rho", "viridis", None if fixed_density_limits is None else fixed_density_limits[0], None if fixed_density_limits is None else fixed_density_limits[1]),
        (pressure, "Pressure", "p", "viridis", None if fixed_pressure_limits is None else fixed_pressure_limits[0], None if fixed_pressure_limits is None else fixed_pressure_limits[1]),
        (omega, "Vorticity", "omega_z", vorticity_cmap, None if vorticity_limit is None else -vorticity_limit, vorticity_limit),
    ]
    if schlieren_values is not None:
        fields.append((np.asarray(schlieren_values), "Schlieren", "S", "gray", 0.0, 1.0))

    fig_width = 19.5 if len(fields) == 4 else 15.0
    fig, axes = plt.subplots(1, len(fields), figsize=(fig_width, 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    label = configuration_label or f"Configuration {configuration_number}"
    fig.suptitle(f"2-D Riemann problem, {label} — t={time:.3f}", fontsize=13, fontweight="bold")
    for index, (ax, (values, title, label, cmap, vmin, vmax)) in enumerate(zip(axes, fields)):
        image = ax.imshow(values, origin="lower", extent=extent, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if density_contours and index == 0:
            levels = np.linspace(float(np.nanmin(rho)), float(np.nanmax(rho)), max(2, num_contours))
            ax.contour(rho, levels=levels, origin="lower", extent=extent, colors="black", linewidths=0.35, alpha=0.55)
        if zoom_center:
            half_window = 0.5 * zoom_window
            ax.set_xlim(max(extent[0], zoom_x - half_window), min(extent[1], zoom_x + half_window))
            ax.set_ylim(max(extent[2], zoom_y - half_window), min(extent[3], zoom_y + half_window))
        fig.colorbar(image, ax=ax, label=label)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)
    return output_path


def plot_from_npz(npz_path: Path, output_path: Path | None = None, **kwargs):
    with np.load(npz_path, allow_pickle=False) as data:
        rho = data["rho"]
        pressure = data["pressure"]
        omega = data["omega_z"]
        x = data["x"]
        y = data["y"]
        time = float(data["time"])
        configuration_number = int(data["configuration_number"]) if "configuration_number" in data else 3
        if "config_json" in data:
            saved_config = json.loads(str(data["config_json"]))
            configuration_label = saved_config.get("configuration_label")
        else:
            configuration_label = None
    if kwargs.pop("schlieren", False):
        domain = Domain2D(float(x[0] - 0.5 * (x[1] - x[0])), float(x[-1] + 0.5 * (x[1] - x[0])), float(y[0] - 0.5 * (y[1] - y[0])), float(y[-1] + 0.5 * (y[1] - y[0])), len(x), len(y))
        schlieren_values = to_numpy(schlieren_field(np.asarray(rho), domain, kwargs.pop("schlieren_k", 20.0)))
    else:
        schlieren_values = None
        kwargs.pop("schlieren_k", None)
    kwargs.setdefault("configuration_number", configuration_number)
    kwargs.setdefault("configuration_label", configuration_label)
    return plot_riemann_solution(rho, pressure, omega, x, y, time, output_path=output_path, schlieren_values=schlieren_values, **kwargs)


def _diagnostics(q, config: RiemannConfig3, time: float, steps: int, hyperviscosity_applications: int, sensor):
    xp = array_module(q)
    rho, pressure = raw_density_pressure(q, config.equation)
    omega = vorticity_z(q, config.domain, config.equation)
    has_nonfinite = bool(xp.any(~xp.isfinite(q)) | xp.any(~xp.isfinite(rho)) | xp.any(~xp.isfinite(pressure)))
    sensor_mask = sensor.detect(q)
    sensor_fraction = float(xp.mean(sensor_mask))
    if config.scheme == "weno":
        # WENO is selected at every x- and y-interface.
        weno_fraction = 1.0
    else:
        # Report the actual fraction of directional interfaces reconstructed
        # with WENO, rather than merely repeating the node-sensor fraction.
        weno_x = interface_mask_from_nodes(sensor_mask)
        sensor_y = xp.moveaxis(sensor_mask, -2, -1)
        weno_y = interface_mask_from_nodes(sensor_y)
        weno_count = float(xp.sum(weno_x)) + float(xp.sum(weno_y))
        interface_count = weno_x.size + weno_y.size
        weno_fraction = weno_count / interface_count
    return RiemannDiagnostics(
        backend=config.backend,
        scheme=config.scheme,
        hyperviscosity_enabled=hyperviscosity_enabled_for_config(config),
        hyperviscosity_applications=hyperviscosity_applications,
        final_sensor_fraction=sensor_fraction,
        final_weno_fraction=weno_fraction,
        density_min=float(xp.min(rho)),
        density_max=float(xp.max(rho)),
        pressure_min=float(xp.min(pressure)),
        pressure_max=float(xp.max(pressure)),
        max_abs_vorticity=float(xp.max(xp.abs(omega))),
        has_nonfinite=has_nonfinite,
        density_positive=bool(xp.all(rho > 0.0)),
        pressure_positive=bool(xp.all(pressure > 0.0)),
        time=time,
        steps=steps,
    )


def print_diagnostic_summary(diagnostics: RiemannDiagnostics) -> None:
    print("Diagnostic summary:")
    print(f"  backend              : {diagnostics.backend}")
    print(f"  scheme               : {diagnostics.scheme}")
    print(f"  density min/max      : {diagnostics.density_min:.8e} / {diagnostics.density_max:.8e}")
    print(f"  pressure min/max     : {diagnostics.pressure_min:.8e} / {diagnostics.pressure_max:.8e}")
    print(f"  max |vorticity|      : {diagnostics.max_abs_vorticity:.8e}")
    print(f"  NaN/Inf present      : {diagnostics.has_nonfinite}")
    print(f"  density positive     : {diagnostics.density_positive}")
    print(f"  pressure positive    : {diagnostics.pressure_positive}")
    print(f"  final sensor fraction: {diagnostics.final_sensor_fraction:.6f}")
    print(f"  final WENO fraction  : {diagnostics.final_weno_fraction:.6f}")
    print(f"  hyperviscosity       : {diagnostics.hyperviscosity_enabled} ({diagnostics.hyperviscosity_applications} applications)")


def hyperviscosity_enabled_for_config(config: RiemannConfig3) -> bool:
    """Return whether compact-node hyperviscosity is active for a run.

    WENO-only simulations disable numerical hyperviscosity because they
    contain no compact-FD nodes on which it may be applied.
    """

    return (
        config.scheme == "hybrid"
        and config.mn > 0.0
        and config.hyperviscosity_interval > 0
    )


def run_riemann(config: RiemannConfig3):
    xp = select_array_module(config.backend)
    print(f"Using array backend: {xp.__name__}")
    q = config.initial_state()
    q = config.equation.enforce_physical_state(q)
    q = apply_outflow_guard(q, config.guard_cells)
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
    operator = NonPeriodicHybridEuler2DOperator(
        config.domain,
        config.equation,
        sensor,
        scheme=config.scheme,
    )
    dt, n_steps = operator.fixed_time_step(q, config.cfl, config.tfinal)
    hyperviscosity_enabled = hyperviscosity_enabled_for_config(config)
    hyperviscosity = LocalHyperviscosity2D(
        config.domain,
        mn=config.mn,
        density_weight=config.hyperviscosity_density_weight,
        momentum_weight=config.hyperviscosity_momentum_weight,
        energy_weight=config.hyperviscosity_energy_weight,
    ) if hyperviscosity_enabled else None
    integrator = SSPRK3()
    label = getattr(config, "configuration_label", f"Config {config.configuration_number}")
    print(
        f"Riemann {label}: nx={config.nx}, ny={config.ny}, "
        f"fixed dt={dt:.5e}, steps={n_steps}, tfinal={config.tfinal}, "
        f"initial CFL={config.cfl}"
    )
    hyperviscosity_note = (
        "enabled on compact-FD nodes only"
        if hyperviscosity_enabled
        else "disabled"
    )
    if config.scheme == "weno":
        hyperviscosity_note += " (WENO-only scheme)"
    print(
        f"scheme={config.scheme}; local hyperviscosity {hyperviscosity_note} "
        f"(mn={config.mn})"
    )
    time = 0.0
    hyperviscosity_applications = 0
    for step_index in range(n_steps):
        q = integrator.step(
            q,
            operator.rhs,
            dt,
            clean=lambda state: apply_outflow_guard(
                config.equation.enforce_physical_state(state),
                config.guard_cells,
            ),
        )
        step = step_index + 1

        if hyperviscosity_enabled and step % config.hyperviscosity_interval == 0:
            weno_mask = sensor.detect(q)
            compact_mask = ~weno_mask
            q = hyperviscosity.apply(q, compact_mask, config.equation)
            q = apply_outflow_guard(q, config.guard_cells)
            hyperviscosity_applications += 1

        time = step * dt
        if config.progress_every > 0 and (
            step % config.progress_every == 0 or step == n_steps
        ):
            validate_raw_physical_state(q, config.equation, f"state at step {step}")
            print(f"step={step:6d}, time={time:.6f}, dt={dt:.5e}")

    diagnostics = _diagnostics(q, config, time, step, hyperviscosity_applications, sensor)
    validate_raw_physical_state(q, config.equation, "final state")
    print_diagnostic_summary(diagnostics)
    return q, diagnostics


def run_riemann_config3(config: RiemannConfig3):
    return run_riemann(config)


def save_run_outputs(
    q,
    config: RiemannConfig3,
    diagnostics: RiemannDiagnostics,
    output_dir: Path,
    run_id: str | None = None,
    plot: bool = True,
    no_show: bool = True,
    density_contours: bool = False,
    num_contours: int = 40,
    schlieren: bool = False,
    schlieren_k: float = 20.0,
    zoom_center: bool = False,
    zoom_window: float = 0.35,
    vorticity_limit: float | None = 100.0,
    fixed_density_limits: tuple[float, float] | None = None,
    fixed_pressure_limits: tuple[float, float] | None = None,
) -> dict[str, Path]:
    run_id = run_id or make_run_id(f"riemann{config.configuration_number}")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", riemann_config_dict(config))
    write_json(run_dir / "diagnostics.json", asdict(diagnostics))
    paths: dict[str, Path] = {}
    npz_path = run_dir / f"{run_id}_final.npz"
    save_riemann_npz(npz_path, q, config, diagnostics)
    paths["npz"] = npz_path
    if plot:
        rho, _u, _v, pressure = primitive_fields(q, config.equation)
        omega = vorticity_z(q, config.domain, config.equation)
        schlieren_values = schlieren_field(rho, config.domain, schlieren_k) if schlieren else None
        plot_path = run_dir / f"{run_id}_fields.png"
        plot_riemann_solution(
            to_numpy(rho),
            to_numpy(pressure),
            to_numpy(omega),
            config.domain.x,
            config.domain.y,
            diagnostics.time,
            output_path=plot_path,
            show=not no_show,
            vorticity_limit=vorticity_limit,
            density_contours=density_contours,
            num_contours=num_contours,
            schlieren_values=to_numpy(schlieren_values) if schlieren_values is not None else None,
            zoom_center=zoom_center,
            zoom_window=zoom_window,
            fixed_density_limits=fixed_density_limits,
            fixed_pressure_limits=fixed_pressure_limits,
            configuration_number=config.configuration_number,
            configuration_label=getattr(config, "configuration_label", None),
        )
        paths["figure"] = plot_path
    return paths
