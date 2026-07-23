from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from Configuration3_reference import Riemann2DConfig3, primitive_fields, validate_state, vorticity_z
except ModuleNotFoundError:
    from Configuration3 import Riemann2DConfig3, primitive_fields, validate_state, vorticity_z
from OOP.domain import Domain2D
from OOP.equations import EulerEquation2D
from OOP.time_operator import SSPRK3


@dataclass(frozen=True)
class HybridRunDiagnostics:
    hyperviscosity_enabled: bool
    hyperviscosity_applications: int
    final_sensor_fraction: float


def weno7_flux(v1, v2, v3, v4, v5, v6, v7):
    eps = 1e-10
    q0 = -(1 / 4) * v1 + (13 / 12) * v2 - (23 / 12) * v3 + (25 / 12) * v4
    q1 = (1 / 12) * v2 - (5 / 12) * v3 + (13 / 12) * v4 + (1 / 4) * v5
    q2 = -(1 / 12) * v3 + (7 / 12) * v4 + (7 / 12) * v5 - (1 / 12) * v6
    q3 = (1 / 4) * v4 + (13 / 12) * v5 - (5 / 12) * v6 + (1 / 12) * v7

    is0 = (
        v1 * (544 * v1 - 3882 * v2 + 4642 * v3 - 1854 * v4)
        + v2 * (7043 * v2 - 17246 * v3 + 7042 * v4)
        + v3 * (11003 * v3 - 9402 * v4)
        + 2107 * v4**2
    )
    is1 = (
        v2 * (267 * v2 - 1642 * v3 + 1602 * v4 - 494 * v5)
        + v3 * (2843 * v3 - 5966 * v4 + 1922 * v5)
        + v4 * (3443 * v4 - 2522 * v5)
        + 547 * v5**2
    )
    is2 = (
        v3 * (547 * v3 - 2522 * v4 + 1922 * v5 - 494 * v6)
        + v4 * (3443 * v4 - 5966 * v5 + 1602 * v6)
        + v5 * (2843 * v5 - 1642 * v6)
        + 267 * v6**2
    )
    is3 = (
        v4 * (2107 * v4 - 9402 * v5 + 7042 * v6 - 1854 * v7)
        + v5 * (11003 * v5 - 17246 * v6 + 4642 * v7)
        + v6 * (7043 * v6 - 3882 * v7)
        + 547 * v7**2
    )

    a0 = (1 / 35) / (eps + is0) ** 2
    a1 = (12 / 35) / (eps + is1) ** 2
    a2 = (18 / 35) / (eps + is2) ** 2
    a3 = (4 / 35) / (eps + is3) ** 2
    asum = a0 + a1 + a2 + a3
    return (a0 * q0 + a1 * q1 + a2 * q2 + a3 * q3) / asum


def take_clipped(values: np.ndarray, indices: np.ndarray, axis: int = -1) -> np.ndarray:
    n = values.shape[axis]
    return np.take(values, np.clip(indices, 0, n - 1), axis=axis)


def relative_jump_axis(values: np.ndarray, axis: int) -> np.ndarray:
    values = np.moveaxis(values, axis, -1)
    eps = 1e-14 + 1e-12 * np.max(np.abs(values))

    pair_jump = np.abs(values[..., 1:] - values[..., :-1])
    pair_jump /= np.abs(values[..., 1:]) + np.abs(values[..., :-1]) + eps

    jumps = np.zeros_like(values)
    jumps[..., :-1] = np.maximum(jumps[..., :-1], pair_jump)
    jumps[..., 1:] = np.maximum(jumps[..., 1:], pair_jump)

    center = values
    left = take_clipped(values, np.arange(values.shape[-1]) - 1)
    right = take_clipped(values, np.arange(values.shape[-1]) + 1)
    curvature = np.abs(right - 2.0 * center + left)
    curvature /= np.abs(right) + 2.0 * np.abs(center) + np.abs(left) + eps
    jumps = np.maximum(jumps, curvature)

    return np.moveaxis(jumps, -1, axis)


def dilate_mask_2d(mask: np.ndarray, width: int) -> np.ndarray:
    expanded = np.array(mask, copy=True)
    ny, nx = mask.shape
    for y_offset in range(-width, width + 1):
        for x_offset in range(-width, width + 1):
            if x_offset == 0 and y_offset == 0:
                continue

            y_src = slice(max(0, -y_offset), min(ny, ny - y_offset))
            y_dst = slice(max(0, y_offset), min(ny, ny + y_offset))
            x_src = slice(max(0, -x_offset), min(nx, nx - x_offset))
            x_dst = slice(max(0, x_offset), min(nx, nx + x_offset))
            expanded[y_dst, x_dst] |= mask[y_src, x_src]
    return expanded


def interface_mask_from_nodes(mask: np.ndarray) -> np.ndarray:
    n = mask.shape[-1]
    interface_index = np.arange(n + 1)
    left = take_clipped(mask, interface_index - 1)
    right = take_clipped(mask, interface_index)
    return left | right


def smooth_compact_interfaces(point_flux: np.ndarray) -> np.ndarray:
    a1_c = 25 / 32
    b1_c = 1 / 20
    c1_c = -1 / 480

    n = point_flux.shape[-1]
    i_left = np.arange(-1, n)
    return (
        c1_c * (take_clipped(point_flux, i_left + 3) + take_clipped(point_flux, i_left - 2))
        + (b1_c + c1_c)
        * (take_clipped(point_flux, i_left + 2) + take_clipped(point_flux, i_left - 1))
        + (a1_c + b1_c + c1_c)
        * (take_clipped(point_flux, i_left + 1) + take_clipped(point_flux, i_left))
    )


def weno7_interfaces(q: np.ndarray, point_flux: np.ndarray, alpha: float) -> np.ndarray:
    n = q.shape[-1]
    i_left = np.arange(-1, n)
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


def compactify_weno_interfaces(weno_flux: np.ndarray, alpha: float = 3.0 / 8.0) -> np.ndarray:
    """Apply the compact interface operator to raw WENO fluxes.

    This mirrors the validated 1D hybrid implementation:
    F_weno_hat = alpha * F_{i+1} + F_i + alpha * F_{i-1}.
    The subsequent compact derivative solve then recovers the WENO derivative
    in shock regions instead of distorting it through the compact matrix.
    """

    interface_index = np.arange(weno_flux.shape[-1])
    return (
        alpha * take_clipped(weno_flux, interface_index + 1)
        + weno_flux
        + alpha * take_clipped(weno_flux, interface_index - 1)
    )


@dataclass
class LineCompactDerivative:
    """Compact derivative solve applied line-by-line along the last axis."""

    n: int
    dx: float
    alpha: float = 3.0 / 8.0

    def __post_init__(self) -> None:
        matrix = sp.diags(
            [self.alpha * np.ones(self.n - 1), np.ones(self.n), self.alpha * np.ones(self.n - 1)],
            [-1, 0, 1],
            shape=(self.n, self.n),
        ).tolil()
        matrix[0, 1] = 0.0
        matrix[-1, -2] = 0.0
        self.solve_matrix = spla.factorized(matrix.tocsc())

    def from_interface_flux(self, interface_flux: np.ndarray) -> np.ndarray:
        raw = (interface_flux[..., 1:] - interface_flux[..., :-1]) / self.dx
        leading_shape = raw.shape[:-1]
        rhs = raw.reshape(-1, self.n).T
        derivative = self.solve_matrix(rhs).T.reshape(*leading_shape, self.n)
        return derivative


@dataclass(frozen=True)
class EulerShockSensor2D:
    domain: Domain2D
    equation: EulerEquation2D
    width: int = 4
    compression_threshold: float = 2.5
    jump_threshold: float = 0.04
    shear_threshold: float | None = 2.0
    boundary_guard: int = 4

    def detect(self, q: np.ndarray) -> np.ndarray:
        rho, u, v, pressure = self.equation.primitive_from_conservative(q)
        du_dy, du_dx = np.gradient(u, self.domain.dy, self.domain.dx, edge_order=2)
        dv_dy, dv_dx = np.gradient(v, self.domain.dy, self.domain.dx, edge_order=2)
        divergence = du_dx + dv_dy
        div_rms = np.sqrt(np.mean(divergence**2))

        compression = np.zeros_like(divergence, dtype=bool)
        if div_rms > 1e-12:
            compression = divergence < -self.compression_threshold * div_rms

        shear = np.zeros_like(divergence, dtype=bool)
        if self.shear_threshold is not None:
            vorticity = dv_dx - du_dy
            vort_rms = np.sqrt(np.mean(vorticity**2))
            if vort_rms > 1e-12:
                shear = np.abs(vorticity) > self.shear_threshold * vort_rms

        internal_energy = pressure / (rho * (self.equation.gamma - 1.0))
        density_jump = np.maximum(relative_jump_axis(rho, -1), relative_jump_axis(rho, -2))
        pressure_jump = np.maximum(relative_jump_axis(pressure, -1), relative_jump_axis(pressure, -2))
        energy_jump = np.maximum(
            relative_jump_axis(internal_energy, -1),
            relative_jump_axis(internal_energy, -2),
        )

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
class HybridEuler2DOperator:
    domain: Domain2D
    equation: EulerEquation2D
    sensor: EulerShockSensor2D
    compact_x: LineCompactDerivative = field(init=False)
    compact_y: LineCompactDerivative = field(init=False)

    def __post_init__(self) -> None:
        self.compact_x = LineCompactDerivative(self.domain.nx, self.domain.dx)
        self.compact_y = LineCompactDerivative(self.domain.ny, self.domain.dy)

    def _axis_derivative(
        self,
        q_axis: np.ndarray,
        flux_axis: np.ndarray,
        shock_axis: np.ndarray,
        compact: LineCompactDerivative,
        normal_velocity_index: int,
    ) -> np.ndarray:
        rho, u, v, pressure = self.equation.primitive_from_conservative(q_axis)
        normal_velocity = u if normal_velocity_index == 1 else v
        sound_speed = np.sqrt(self.equation.gamma * pressure / rho)
        alpha = float(np.max(np.abs(normal_velocity) + sound_speed))

        smooth_flux = smooth_compact_interfaces(flux_axis)
        shock_flux = compactify_weno_interfaces(
            weno7_interfaces(q_axis, flux_axis, alpha),
            alpha=compact.alpha,
        )
        shock_interfaces = interface_mask_from_nodes(shock_axis)

        hybrid_flux = np.where(shock_interfaces[None, ...], shock_flux, smooth_flux)
        return compact.from_interface_flux(hybrid_flux)

    def rhs(self, q: np.ndarray) -> np.ndarray:
        q_safe = self.equation.enforce_physical_state(q)
        shock_mask = self.sensor.detect(q_safe)

        flux_x = self.equation.flux_x(q_safe)
        derivative_x = self._axis_derivative(
            q_safe,
            flux_x,
            shock_mask,
            self.compact_x,
            normal_velocity_index=1,
        )

        q_y = np.moveaxis(q_safe, -2, -1)
        flux_y = np.moveaxis(self.equation.flux_y(q_safe), -2, -1)
        shock_y = np.moveaxis(shock_mask, -2, -1)
        derivative_y = self._axis_derivative(
            q_y,
            flux_y,
            shock_y,
            self.compact_y,
            normal_velocity_index=2,
        )
        derivative_y = np.moveaxis(derivative_y, -1, -2)

        return -derivative_x - derivative_y

    def fixed_time_step(self, q: np.ndarray, cfl: float, t_end: float) -> tuple[float, int]:
        rho, u, v, pressure = self.equation.primitive_from_conservative(q)
        c = np.sqrt(self.equation.gamma * pressure / rho)
        spectral_radius = (np.abs(u) + c) / self.domain.dx
        spectral_radius += (np.abs(v) + c) / self.domain.dy
        dt = cfl / float(np.max(spectral_radius))
        n_steps = int(np.ceil(t_end / dt))
        return t_end / n_steps, n_steps


@dataclass(frozen=True)
class LocalHyperviscosity2D:
    """Nonperiodic fourth-order shock filter for the 2D outflow problem.

    The validated 1D Wang filter uses periodic compact matrices. Applying those
    matrices line-by-line to this Riemann problem couples opposite outflow
    boundaries, so this class keeps the same role but uses edge-padded local
    fourth-order dissipation near the shock mask.
    """

    domain: Domain2D
    mn: float = 0.01
    dilation: int = 2
    density_weight: float = 1.0
    momentum_weight: float = 1.0
    energy_weight: float = 1.0

    def _laplacian(self, values: np.ndarray) -> np.ndarray:
        padded = np.pad(values, ((0, 0), (1, 1), (1, 1)), mode="edge")
        center = padded[:, 1:-1, 1:-1]
        lap_x = (padded[:, 1:-1, 2:] - 2.0 * center + padded[:, 1:-1, :-2]) / self.domain.dx**2
        lap_y = (padded[:, 2:, 1:-1] - 2.0 * center + padded[:, :-2, 1:-1]) / self.domain.dy**2
        return lap_x + lap_y

    def apply(self, q: np.ndarray, shock_mask: np.ndarray, equation: EulerEquation2D) -> np.ndarray:
        active = dilate_mask_2d(shock_mask, self.dilation)
        h = min(self.domain.dx, self.domain.dy)
        biharmonic = self._laplacian(self._laplacian(q))
        weights = np.array(
            [
                self.density_weight,
                self.momentum_weight,
                self.momentum_weight,
                self.energy_weight,
            ],
            dtype=q.dtype,
        )[:, None, None]
        filtered = q - self.mn * weights * h**4 * active[None, :, :] * biharmonic
        return equation.enforce_physical_state(filtered)


def apply_outflow_guard(q: np.ndarray, guard_cells: int = 2) -> np.ndarray:
    if guard_cells <= 0:
        return q
    guarded = np.array(q, copy=True)
    g = guard_cells
    guarded[:, :, :g] = guarded[:, :, g][:, :, None]
    guarded[:, :, -g:] = guarded[:, :, -g - 1][:, :, None]
    guarded[:, :g, :] = guarded[:, g, :][:, None, :]
    guarded[:, -g:, :] = guarded[:, -g - 1, :][:, None, :]
    return guarded


def raw_density_pressure(q: np.ndarray, equation: EulerEquation2D) -> tuple[np.ndarray, np.ndarray]:
    rho = q[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = q[1] / rho
        v = q[2] / rho
        kinetic = 0.5 * rho * (u**2 + v**2)
        pressure = (equation.gamma - 1.0) * (q[3] - kinetic)
    return rho, pressure


def validate_raw_physical_state(q: np.ndarray, equation: EulerEquation2D, label: str = "state") -> None:
    if not np.all(np.isfinite(q)):
        raise FloatingPointError(f"{label} contains NaN or Inf values")
    rho, pressure = raw_density_pressure(q, equation)
    if np.any(~np.isfinite(rho)) or np.any(~np.isfinite(pressure)):
        raise FloatingPointError(f"{label} has non-finite density or pressure")
    if np.any(rho <= 0.0):
        raise FloatingPointError(f"{label} contains non-positive density")
    if np.any(pressure <= 0.0):
        raise FloatingPointError(f"{label} contains non-positive pressure")


def density_gradient_magnitude(rho: np.ndarray, domain: Domain2D) -> np.ndarray:
    grad_y, grad_x = np.gradient(rho, domain.dy, domain.dx, edge_order=2)
    return np.sqrt(grad_x**2 + grad_y**2)


def schlieren_field(rho: np.ndarray, domain: Domain2D, k: float = 20.0) -> np.ndarray:
    grad_rho = density_gradient_magnitude(rho, domain)
    max_grad = float(np.max(grad_rho))
    if max_grad <= 0.0 or not np.isfinite(max_grad):
        return np.ones_like(rho)
    return np.exp(-k * grad_rho / max_grad)


def suffixed_output_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}")


def print_diagnostic_summary(
    q: np.ndarray,
    config: Riemann2DConfig3,
    diagnostics: HybridRunDiagnostics,
) -> None:
    equation = config.equation
    rho, pressure = raw_density_pressure(q, equation)
    omega = vorticity_z(q, config.domain, equation)
    has_nonfinite = not (np.all(np.isfinite(q)) and np.all(np.isfinite(rho)) and np.all(np.isfinite(pressure)))
    density_positive = bool(np.all(rho > 0.0))
    pressure_positive = bool(np.all(pressure > 0.0))

    print("Diagnostic summary:")
    print(f"  density min/max       : {np.nanmin(rho):.8e} / {np.nanmax(rho):.8e}")
    print(f"  pressure min/max      : {np.nanmin(pressure):.8e} / {np.nanmax(pressure):.8e}")
    print(f"  max |vorticity|       : {np.nanmax(np.abs(omega)):.8e}")
    print(f"  NaN/Inf present       : {has_nonfinite}")
    print(f"  density positive      : {density_positive}")
    print(f"  pressure positive     : {pressure_positive}")
    print(f"  final sensor fraction : {diagnostics.final_sensor_fraction:.6f}")
    print(
        "  hyperviscosity applied: "
        f"{diagnostics.hyperviscosity_enabled} "
        f"({diagnostics.hyperviscosity_applications} applications)"
    )


def run_hybrid_case(
    config: Riemann2DConfig3,
    sensor_width: int = 4,
    jump_threshold: float = 0.04,
    compression_threshold: float = 2.5,
    shear_threshold: float | None = 2.0,
    hyperviscosity_mn: float = 0.01,
    hyperviscosity_interval: int = 1,
    hyperviscosity_density_weight: float = 1.0,
    hyperviscosity_momentum_weight: float = 1.0,
    hyperviscosity_energy_weight: float = 1.0,
    progress_every: int = 25,
) -> tuple[np.ndarray, float, int, HybridRunDiagnostics]:
    equation = config.equation
    sensor = EulerShockSensor2D(
        config.domain,
        equation,
        width=sensor_width,
        compression_threshold=compression_threshold,
        jump_threshold=jump_threshold,
        shear_threshold=shear_threshold,
    )
    operator = HybridEuler2DOperator(config.domain, equation, sensor)

    q = equation.enforce_physical_state(config.initial_state())
    q = apply_outflow_guard(q)
    validate_state(q, equation, "initial state")
    validate_raw_physical_state(q, equation, "initial state")

    dt, n_steps = operator.fixed_time_step(q, config.cfl, config.tfinal)
    hyperviscosity_enabled = hyperviscosity_mn > 0.0 and hyperviscosity_interval > 0
    hyperviscosity = None
    if hyperviscosity_enabled:
        hyperviscosity = LocalHyperviscosity2D(
            config.domain,
            mn=hyperviscosity_mn,
            density_weight=hyperviscosity_density_weight,
            momentum_weight=hyperviscosity_momentum_weight,
            energy_weight=hyperviscosity_energy_weight,
        )
    integrator = SSPRK3()

    print(
        f"Hybrid 2D Euler Config 3: nx={config.nx}, ny={config.ny}, "
        f"dt={dt:.5e}, steps={n_steps}, tfinal={config.tfinal}"
    )
    if hyperviscosity_enabled:
        print(
            f"Local hyperviscosity enabled: mn={hyperviscosity_mn:.6g}, "
            f"interval={hyperviscosity_interval}"
        )
    else:
        print(f"Local hyperviscosity disabled: mn={hyperviscosity_mn:.6g}")

    time = 0.0
    hyperviscosity_applications = 0
    for step in range(n_steps):
        q = integrator.step(
            q,
            operator.rhs,
            dt,
            clean=lambda state: apply_outflow_guard(equation.enforce_physical_state(state)),
        )

        if hyperviscosity_enabled and (step + 1) % hyperviscosity_interval == 0:
            shock_mask = sensor.detect(q)
            assert hyperviscosity is not None
            q = hyperviscosity.apply(q, shock_mask, equation)
            q = apply_outflow_guard(q)
            hyperviscosity_applications += 1

        time = (step + 1) * dt
        if (step + 1) % progress_every == 0 or step + 1 == n_steps:
            validate_raw_physical_state(q, equation, f"state at step {step + 1}")
            validate_state(q, equation, f"state at step {step + 1}")
            print(f"step={step + 1:6d}, time={time:.6f}")

    final_sensor_fraction = float(np.mean(sensor.detect(q)))
    diagnostics = HybridRunDiagnostics(
        hyperviscosity_enabled=hyperviscosity_enabled,
        hyperviscosity_applications=hyperviscosity_applications,
        final_sensor_fraction=final_sensor_fraction,
    )
    return q, time, n_steps, diagnostics


def plot_hybrid_solution(
    q: np.ndarray,
    config: Riemann2DConfig3,
    output_path: Path | None = None,
    show: bool = True,
    vorticity_limit: float | None = 60.0,
    vorticity_cmap: str = "coolwarm",
    density_contours: bool = False,
    num_contours: int = 40,
    schlieren: bool = False,
    schlieren_k: float = 20.0,
    zoom_center: bool = False,
    zoom_window: float = 0.35,
    fixed_density_limits: tuple[float, float] | None = None,
    fixed_pressure_limits: tuple[float, float] | None = None,
) -> None:
    domain = config.domain
    equation = config.equation
    rho, _u, _v, pressure = primitive_fields(q, equation)
    omega = vorticity_z(q, domain, equation)
    schlieren_values = schlieren_field(rho, domain, schlieren_k) if schlieren else None

    fields = [
        (
            rho,
            "Density",
            "rho",
            "viridis",
            fixed_density_limits[0] if fixed_density_limits is not None else None,
            fixed_density_limits[1] if fixed_density_limits is not None else None,
        ),
        (
            pressure,
            "Pressure",
            "p",
            "viridis",
            fixed_pressure_limits[0] if fixed_pressure_limits is not None else None,
            fixed_pressure_limits[1] if fixed_pressure_limits is not None else None,
        ),
        (
            omega,
            "Vorticity",
            "omega_z",
            vorticity_cmap,
            -vorticity_limit if vorticity_limit is not None else None,
            vorticity_limit,
        ),
    ]
    if schlieren_values is not None:
        fields.append((schlieren_values, "Schlieren", "S", "gray", 0.0, 1.0))
    extent = [domain.x_min, domain.x_max, domain.y_min, domain.y_max]

    fig_width = 19.5 if schlieren_values is not None else 15.0
    fig, axes = plt.subplots(1, len(fields), figsize=(fig_width, 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(
        f"Hybrid 2D Euler Riemann Problem - Configuration 3, t = {config.tfinal:.3f}",
        fontsize=13,
        fontweight="bold",
    )

    for index, (ax, (field_values, title, label, cmap, vmin, vmax)) in enumerate(zip(axes, fields)):
        image = ax.imshow(
            field_values,
            origin="lower",
            extent=extent,
            cmap=cmap,
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if density_contours and index == 0:
            levels = np.linspace(float(np.min(rho)), float(np.max(rho)), max(2, num_contours))
            ax.contour(
                rho,
                levels=levels,
                origin="lower",
                extent=extent,
                colors="black",
                linewidths=0.35,
                alpha=0.55,
            )
        if zoom_center:
            half_window = 0.5 * zoom_window
            center_x, center_y = 0.375, 0.375
            ax.set_xlim(max(domain.x_min, center_x - half_window), min(domain.x_max, center_x + half_window))
            ax.set_ylim(max(domain.y_min, center_y - half_window), min(domain.y_max, center_y + half_window))
        fig.colorbar(image, ax=ax, label=label)

    if output_path is not None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches="tight", dpi=200)
            print(f"Saved figure to {output_path}")
            if density_contours:
                contour_path = suffixed_output_path(output_path, "_contours")
                fig.savefig(contour_path, bbox_inches="tight", dpi=200)
                print(f"Saved contour figure to {contour_path}")
            if schlieren:
                schlieren_path = suffixed_output_path(output_path, "_schlieren")
                fig.savefig(schlieren_path, bbox_inches="tight", dpi=200)
                print(f"Saved schlieren figure to {schlieren_path}")
            if zoom_center:
                zoom_path = suffixed_output_path(output_path, "_zoom")
                fig.savefig(zoom_path, bbox_inches="tight", dpi=200)
                print(f"Saved zoom figure to {zoom_path}")
        except OSError as exc:
            print(f"Could not save figure to {output_path}: {exc}")

    if show:
        plt.show(block=True)
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hybrid 2D Euler Configuration 3 benchmark.")
    parser.add_argument("--nx", type=int, default=200, help="Number of cells in x.")
    parser.add_argument("--ny", type=int, default=200, help="Number of cells in y.")
    parser.add_argument("--tfinal", type=float, default=0.3, help="Final time.")
    parser.add_argument("--cfl", type=float, default=0.08, help="CFL number.")
    parser.add_argument("--sensor-width", type=int, default=8, help="Shock-sensor dilation width.")
    parser.add_argument("--jump-threshold", type=float, default=0.015, help="Relative jump threshold.")
    parser.add_argument("--compression-threshold", type=float, default=2.5, help="Compression threshold.")
    parser.add_argument(
        "--shear-threshold",
        type=float,
        default=2.0,
        help="RMS-based vorticity/shear threshold. Use 0 to disable.",
    )
    parser.add_argument("--mn", type=float, default=0.01, help="Local hyperviscosity strength.")
    parser.add_argument("--hyperviscosity-interval", type=int, default=1, help="Hyperviscosity application interval.")
    parser.add_argument("--hyperviscosity-density-weight", type=float, default=1.0)
    parser.add_argument("--hyperviscosity-momentum-weight", type=float, default=1.0)
    parser.add_argument("--hyperviscosity-energy-weight", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N steps.")
    parser.add_argument("--no-show", action="store_true", help="Do not open the Matplotlib window.")
    parser.add_argument(
        "--vorticity-limit",
        type=float,
        default=60.0,
        help="Symmetric vorticity color limit. Use 0 for automatic scaling.",
    )
    parser.add_argument("--vorticity-cmap", default="coolwarm", help="Colormap for vorticity.")
    parser.add_argument("--density-contours", action="store_true", help="Overlay density contours on the density panel.")
    parser.add_argument("--num-contours", type=int, default=40, help="Number of density contour levels.")
    parser.add_argument("--schlieren", action="store_true", help="Add a schlieren-like density-gradient panel.")
    parser.add_argument("--schlieren-k", type=float, default=20.0, help="Schlieren contrast factor.")
    parser.add_argument("--zoom-center", action="store_true", help="Zoom panels around the central interaction region.")
    parser.add_argument("--zoom-window", type=float, default=0.35, help="Width of the central zoom window.")
    parser.add_argument(
        "--fixed-density-limits",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Fixed density color limits.",
    )
    parser.add_argument(
        "--fixed-pressure-limits",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Fixed pressure color limits.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path(__file__).resolve().with_name("Riemann2D_Config3_hybrid.png"),
        help="Path for the output figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Riemann2DConfig3(nx=args.nx, ny=args.ny, tfinal=args.tfinal, cfl=args.cfl)
    q_final, time, steps, diagnostics = run_hybrid_case(
        config,
        sensor_width=args.sensor_width,
        jump_threshold=args.jump_threshold,
        compression_threshold=args.compression_threshold,
        shear_threshold=None if args.shear_threshold == 0.0 else args.shear_threshold,
        hyperviscosity_mn=args.mn,
        hyperviscosity_interval=args.hyperviscosity_interval,
        hyperviscosity_density_weight=args.hyperviscosity_density_weight,
        hyperviscosity_momentum_weight=args.hyperviscosity_momentum_weight,
        hyperviscosity_energy_weight=args.hyperviscosity_energy_weight,
        progress_every=args.progress_every,
    )
    print(f"Completed hybrid Configuration 3 at t={time:.6f} in {steps} steps.")
    validate_raw_physical_state(q_final, config.equation, "final state")
    print_diagnostic_summary(q_final, config, diagnostics)
    vorticity_limit = None if args.vorticity_limit == 0.0 else args.vorticity_limit
    plot_hybrid_solution(
        q_final,
        config,
        output_path=args.save,
        show=not args.no_show,
        vorticity_limit=vorticity_limit,
        vorticity_cmap=args.vorticity_cmap,
        density_contours=args.density_contours,
        num_contours=args.num_contours,
        schlieren=args.schlieren,
        schlieren_k=args.schlieren_k,
        zoom_center=args.zoom_center,
        zoom_window=args.zoom_window,
        fixed_density_limits=tuple(args.fixed_density_limits) if args.fixed_density_limits is not None else None,
        fixed_pressure_limits=tuple(args.fixed_pressure_limits) if args.fixed_pressure_limits is not None else None,
    )


if __name__ == "__main__":
    main()
