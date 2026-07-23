from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as scipy_ndimage
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cupy_ndimage
except ImportError:
    cp = None
    cupy_ndimage = None

# Array module selected by --backend. NumPy remains available for plotting/I/O.
xp = np
BACKEND = "numpy"


def configure_backend(backend: str, device: int = 0) -> None:
    global xp, BACKEND
    backend = backend.lower()
    if backend == "cupy":
        if cp is None:
            raise RuntimeError(
                "CuPy is not installed. Install a CUDA-compatible CuPy package "
                "or run with --backend numpy."
            )
        cp.cuda.Device(device).use()
        xp = cp
        BACKEND = "cupy"
    elif backend == "numpy":
        xp = np
        BACKEND = "numpy"
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def to_numpy(values):
    if cp is not None and isinstance(values, cp.ndarray):
        return cp.asnumpy(values)
    return np.asarray(values)


def scalar(value) -> float:
    if cp is not None and isinstance(value, cp.ndarray):
        return float(value.item())
    return float(value)


def primitive_backend(q, equation):
    rho = q[0]
    u = q[1] / rho
    v = q[2] / rho
    kinetic = 0.5 * rho * (u * u + v * v)
    pressure = (equation.gamma - 1.0) * (q[3] - kinetic)
    return rho, u, v, pressure


def enforce_physical_backend(q, equation):
    rho_floor = float(getattr(equation, "rho_floor", 1.0e-12))
    pressure_floor = float(getattr(equation, "pressure_floor", 1.0e-12))
    out = xp.array(q, copy=True)
    rho = xp.maximum(out[0], rho_floor)
    out[0] = rho
    u = out[1] / rho
    v = out[2] / rho
    kinetic = 0.5 * rho * (u * u + v * v)
    min_energy = kinetic + pressure_floor / (equation.gamma - 1.0)
    out[3] = xp.maximum(out[3], min_energy)
    return out


def flux_x_backend(q, equation):
    rho, u, v, pressure = primitive_backend(q, equation)
    return xp.stack(
        (rho * u, rho * u * u + pressure, rho * u * v, (q[3] + pressure) * u),
        axis=0,
    )


def flux_y_backend(q, equation):
    rho, u, v, pressure = primitive_backend(q, equation)
    return xp.stack(
        (rho * v, rho * u * v, rho * v * v + pressure, (q[3] + pressure) * v),
        axis=0,
    )


def validate_backend(q, equation, label: str) -> None:
    rho, _u, _v, pressure = primitive_backend(q, equation)
    finite = scalar(xp.all(xp.isfinite(q)))
    rho_min = scalar(xp.min(rho))
    pressure_min = scalar(xp.min(pressure))
    if not finite or rho_min <= 0.0 or pressure_min <= 0.0:
        raise FloatingPointError(
            f"{label}: nonphysical state (rho_min={rho_min:.6e}, "
            f"p_min={pressure_min:.6e})"
        )


def ssprk3_step(q, rhs, dt, clean):
    q1 = clean(q + dt * rhs(q))
    q2 = clean(0.75 * q + 0.25 * (q1 + dt * rhs(q1)))
    return clean((1.0 / 3.0) * q + (2.0 / 3.0) * (q2 + dt * rhs(q2)))


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


def take_clipped(values: xp.ndarray, indices: xp.ndarray, axis: int = -1) -> xp.ndarray:
    n = values.shape[axis]
    return xp.take(values, xp.clip(indices, 0, n - 1), axis=axis)


def relative_jump_axis(values: xp.ndarray, axis: int) -> xp.ndarray:
    values = xp.moveaxis(values, axis, -1)
    eps = 1e-14 + 1e-12 * xp.max(xp.abs(values))

    pair_jump = xp.abs(values[..., 1:] - values[..., :-1])
    pair_jump /= xp.abs(values[..., 1:]) + xp.abs(values[..., :-1]) + eps

    jumps = xp.zeros_like(values)
    jumps[..., :-1] = xp.maximum(jumps[..., :-1], pair_jump)
    jumps[..., 1:] = xp.maximum(jumps[..., 1:], pair_jump)

    center = values
    left = take_clipped(values, xp.arange(values.shape[-1]) - 1)
    right = take_clipped(values, xp.arange(values.shape[-1]) + 1)
    curvature = xp.abs(right - 2.0 * center + left)
    curvature /= xp.abs(right) + 2.0 * xp.abs(center) + xp.abs(left) + eps
    jumps = xp.maximum(jumps, curvature)

    return xp.moveaxis(jumps, -1, axis)


def dilate_mask_2d(mask, width: int):
    if width <= 0:
        return xp.array(mask, copy=True)
    size = 2 * width + 1
    if BACKEND == "cupy":
        return cupy_ndimage.maximum_filter(mask, size=(size, size), mode="constant", cval=0)
    return scipy_ndimage.maximum_filter(mask, size=(size, size), mode="constant", cval=0)


def interface_mask_from_nodes(mask: xp.ndarray) -> xp.ndarray:
    n = mask.shape[-1]
    interface_index = xp.arange(n + 1)
    left = take_clipped(mask, interface_index - 1)
    right = take_clipped(mask, interface_index)
    return left | right


def smooth_compact_interfaces(point_flux: xp.ndarray) -> xp.ndarray:
    a1_c = 25 / 32
    b1_c = 1 / 20
    c1_c = -1 / 480

    n = point_flux.shape[-1]
    i_left = xp.arange(-1, n)
    return (
        c1_c * (take_clipped(point_flux, i_left + 3) + take_clipped(point_flux, i_left - 2))
        + (b1_c + c1_c)
        * (take_clipped(point_flux, i_left + 2) + take_clipped(point_flux, i_left - 1))
        + (a1_c + b1_c + c1_c)
        * (take_clipped(point_flux, i_left + 1) + take_clipped(point_flux, i_left))
    )


def weno7_interfaces(q: xp.ndarray, point_flux: xp.ndarray, alpha: float) -> xp.ndarray:
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


def compactify_weno_interfaces(weno_flux: xp.ndarray, alpha: float = 3.0 / 8.0) -> xp.ndarray:
    """Apply the compact interface operator to raw WENO fluxes.

    This mirrors the validated 1D hybrid implementation:
    F_weno_hat = alpha * F_{i+1} + F_i + alpha * F_{i-1}.
    The subsequent compact derivative solve then recovers the WENO derivative
    in shock regions instead of distorting it through the compact matrix.
    """

    interface_index = xp.arange(weno_flux.shape[-1])
    return (
        alpha * take_clipped(weno_flux, interface_index + 1)
        + weno_flux
        + alpha * take_clipped(weno_flux, interface_index - 1)
    )


@dataclass
class LineCompactDerivative:
    """Compact derivative solve along the last axis.

    NumPy uses a pre-factorized SciPy tridiagonal matrix. CuPy uses a custom
    CUDA kernel that solves all independent grid lines concurrently. Each CUDA
    thread performs one Thomas solve, while the complete set of x- or y-lines
    is processed in parallel on the GPU.
    """

    n: int
    dx: float
    alpha: float = 3.0 / 8.0

    def __post_init__(self) -> None:
        if BACKEND == "numpy":
            matrix = sp.diags(
                [
                    self.alpha * np.ones(self.n - 1),
                    np.ones(self.n),
                    self.alpha * np.ones(self.n - 1),
                ],
                [-1, 0, 1],
                shape=(self.n, self.n),
            ).tolil()
            matrix[0, 1] = 0.0
            matrix[-1, -2] = 0.0
            self.solve_matrix = spla.factorized(matrix.tocsc())
            self.cuda_kernel = None
        else:
            self.solve_matrix = None
            # Modified upper diagonal of the Thomas factorization. It depends
            # only on n and alpha, so it is computed once and reused.
            cprime = np.zeros(self.n, dtype=np.float64)
            for i in range(1, self.n):
                lower = 0.0 if i == self.n - 1 else self.alpha
                upper = 0.0 if i == self.n - 1 else self.alpha
                denom = 1.0 - lower * cprime[i - 1]
                cprime[i] = upper / denom
            self.cuda_cprime = cp.asarray(cprime)
            self.cuda_kernel = cp.RawKernel(
                r"""
                extern "C" __global__
                void batched_compact_thomas(
                    const double* rhs,
                    const double* cprime,
                    double* out,
                    const int n,
                    const int batch,
                    const double alpha)
                {
                    const int line = blockDim.x * blockIdx.x + threadIdx.x;
                    if (line >= batch) return;

                    const long base = ((long) line) * n;

                    // The first boundary row is x_0 = rhs_0.
                    out[base] = rhs[base];

                    // Forward elimination. All independent lines are handled
                    // concurrently by separate CUDA threads.
                    for (int i = 1; i < n; ++i) {
                        const double lower = (i == n - 1) ? 0.0 : alpha;
                        const double denom = 1.0 - lower * cprime[i - 1];
                        out[base + i] =
                            (rhs[base + i] - lower * out[base + i - 1]) / denom;
                    }

                    for (int i = n - 2; i >= 0; --i) {
                        out[base + i] -= cprime[i] * out[base + i + 1];
                    }
                }
                """,
                "batched_compact_thomas",
            )

    def from_interface_flux(self, interface_flux):
        raw = (interface_flux[..., 1:] - interface_flux[..., :-1]) / self.dx
        leading_shape = raw.shape[:-1]

        if BACKEND == "numpy":
            rhs = raw.reshape(-1, self.n).T
            derivative = self.solve_matrix(rhs).T.reshape(*leading_shape, self.n)
            return derivative

        if self.n > 4096:
            raise ValueError("CUDA compact solver supports at most 4096 cells per line.")

        rhs = cp.ascontiguousarray(raw.reshape(-1, self.n), dtype=cp.float64)
        derivative = cp.empty_like(rhs)
        batch = rhs.shape[0]
        threads = 128
        blocks = (batch + threads - 1) // threads
        self.cuda_kernel(
            (blocks,),
            (threads,),
            (
                rhs,
                self.cuda_cprime,
                derivative,
                np.int32(self.n),
                np.int32(batch),
                np.float64(self.alpha),
            ),
        )
        return derivative.reshape(*leading_shape, self.n)


@dataclass(frozen=True)
class EulerShockSensor2D:
    domain: Domain2D
    equation: EulerEquation2D
    width: int = 4
    compression_threshold: float = 2.5
    jump_threshold: float = 0.04
    shear_threshold: float | None = 2.0
    boundary_guard: int = 4

    def detect(self, q: xp.ndarray) -> xp.ndarray:
        rho, u, v, pressure = primitive_backend(q, self.equation)
        du_dy, du_dx = xp.gradient(u, self.domain.dy, self.domain.dx, edge_order=2)
        dv_dy, dv_dx = xp.gradient(v, self.domain.dy, self.domain.dx, edge_order=2)
        divergence = du_dx + dv_dy
        div_rms = xp.sqrt(xp.mean(divergence**2))

        compression = xp.zeros_like(divergence, dtype=bool)
        if div_rms > 1e-12:
            compression = divergence < -self.compression_threshold * div_rms

        shear = xp.zeros_like(divergence, dtype=bool)
        if self.shear_threshold is not None:
            vorticity = dv_dx - du_dy
            vort_rms = xp.sqrt(xp.mean(vorticity**2))
            if vort_rms > 1e-12:
                shear = xp.abs(vorticity) > self.shear_threshold * vort_rms

        internal_energy = pressure / (rho * (self.equation.gamma - 1.0))
        density_jump = xp.maximum(relative_jump_axis(rho, -1), relative_jump_axis(rho, -2))
        pressure_jump = xp.maximum(relative_jump_axis(pressure, -1), relative_jump_axis(pressure, -2))
        energy_jump = xp.maximum(
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
        q_axis: xp.ndarray,
        flux_axis: xp.ndarray,
        shock_axis: xp.ndarray,
        compact: LineCompactDerivative,
        normal_velocity_index: int,
    ) -> xp.ndarray:
        rho, u, v, pressure = primitive_backend(q_axis, self.equation)
        normal_velocity = u if normal_velocity_index == 1 else v
        sound_speed = xp.sqrt(self.equation.gamma * pressure / rho)
        alpha = scalar(xp.max(xp.abs(normal_velocity) + sound_speed))

        smooth_flux = smooth_compact_interfaces(flux_axis)
        shock_flux = compactify_weno_interfaces(
            weno7_interfaces(q_axis, flux_axis, alpha),
            alpha=compact.alpha,
        )
        shock_interfaces = interface_mask_from_nodes(shock_axis)

        hybrid_flux = xp.where(shock_interfaces[None, ...], shock_flux, smooth_flux)
        return compact.from_interface_flux(hybrid_flux)

    def rhs(self, q: xp.ndarray) -> xp.ndarray:
        q_safe = enforce_physical_backend(q, self.equation)
        shock_mask = self.sensor.detect(q_safe)

        flux_x = flux_x_backend(q_safe, self.equation)
        derivative_x = self._axis_derivative(
            q_safe,
            flux_x,
            shock_mask,
            self.compact_x,
            normal_velocity_index=1,
        )

        q_y = xp.moveaxis(q_safe, -2, -1)
        flux_y = xp.moveaxis(flux_y_backend(q_safe, self.equation), -2, -1)
        shock_y = xp.moveaxis(shock_mask, -2, -1)
        derivative_y = self._axis_derivative(
            q_y,
            flux_y,
            shock_y,
            self.compact_y,
            normal_velocity_index=2,
        )
        derivative_y = xp.moveaxis(derivative_y, -1, -2)

        return -derivative_x - derivative_y

    def fixed_time_step(self, q: xp.ndarray, cfl: float, t_end: float) -> tuple[float, int]:
        rho, u, v, pressure = primitive_backend(q, self.equation)
        c = xp.sqrt(self.equation.gamma * pressure / rho)
        spectral_radius = (xp.abs(u) + c) / self.domain.dx
        spectral_radius += (xp.abs(v) + c) / self.domain.dy
        dt = cfl / scalar(xp.max(spectral_radius))
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

    def _laplacian(self, values: xp.ndarray) -> xp.ndarray:
        padded = xp.pad(values, ((0, 0), (1, 1), (1, 1)), mode="edge")
        center = padded[:, 1:-1, 1:-1]
        lap_x = (padded[:, 1:-1, 2:] - 2.0 * center + padded[:, 1:-1, :-2]) / self.domain.dx**2
        lap_y = (padded[:, 2:, 1:-1] - 2.0 * center + padded[:, :-2, 1:-1]) / self.domain.dy**2
        return lap_x + lap_y

    def apply(self, q: xp.ndarray, shock_mask: xp.ndarray, equation: EulerEquation2D) -> xp.ndarray:
        active = dilate_mask_2d(shock_mask, self.dilation)
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
        filtered = q - self.mn * weights * h**4 * active[None, :, :] * biharmonic
        return enforce_physical_backend(filtered, equation)


def apply_outflow_guard(q: xp.ndarray, guard_cells: int = 2) -> xp.ndarray:
    if guard_cells <= 0:
        return q
    guarded = xp.array(q, copy=True)
    g = guard_cells
    guarded[:, :, :g] = guarded[:, :, g][:, :, None]
    guarded[:, :, -g:] = guarded[:, :, -g - 1][:, :, None]
    guarded[:, :g, :] = guarded[:, g, :][:, None, :]
    guarded[:, -g:, :] = guarded[:, -g - 1, :][:, None, :]
    return guarded


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
) -> tuple[xp.ndarray, float, int]:
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

    q = enforce_physical_backend(xp.asarray(config.initial_state()), equation)
    q = apply_outflow_guard(q)
    validate_backend(q, equation, "initial state")

    dt, n_steps = operator.fixed_time_step(q, config.cfl, config.tfinal)
    hyperviscosity = LocalHyperviscosity2D(
        config.domain,
        mn=hyperviscosity_mn,
        density_weight=hyperviscosity_density_weight,
        momentum_weight=hyperviscosity_momentum_weight,
        energy_weight=hyperviscosity_energy_weight,
    )

    print(
        f"Hybrid 2D Euler Config 3: nx={config.nx}, ny={config.ny}, "
        f"dt={dt:.5e}, steps={n_steps}, tfinal={config.tfinal}"
    )

    q0 = xp.copy(q)
    time = 0.0
    for step in range(n_steps):
        q = ssprk3_step(
            q,
            operator.rhs,
            dt,
            clean=lambda state: apply_outflow_guard(enforce_physical_backend(state, equation)),
        )

        if (step + 1) % hyperviscosity_interval == 0:
            shock_mask = sensor.detect(q)
            q = hyperviscosity.apply(q, shock_mask, equation)
            q = apply_outflow_guard(q)

        time = (step + 1) * dt
        if (step + 1) % progress_every == 0 or step + 1 == n_steps:
            try:
                validate_backend(q, equation, f"state at step {step + 1}")
            except FloatingPointError:
                print(f"Instability detected at step {step + 1}; returning last checked state.")
                return q0, time, step + 1
            print(f"step={step + 1:6d}, time={time:.6f}")
            q0 = xp.copy(q)

    if BACKEND == "cupy":
        cp.cuda.Stream.null.synchronize()
    return q, time, n_steps


def plot_hybrid_solution(
    q: xp.ndarray,
    config: Riemann2DConfig3,
    output_path: Path | None = None,
    show: bool = True,
    vorticity_limit: float | None = 60.0,
    vorticity_cmap: str = "coolwarm",
) -> None:
    domain = config.domain
    equation = config.equation
    q = to_numpy(q)
    rho, _u, _v, pressure = primitive_fields(q, equation)
    omega = vorticity_z(q, domain, equation)

    fields = [
        (rho, "Density", "rho", "viridis", None, None),
        (pressure, "Pressure", "p", "viridis", None, None),
        (
            omega,
            "Vorticity",
            "omega_z",
            vorticity_cmap,
            -vorticity_limit if vorticity_limit is not None else None,
            vorticity_limit,
        ),
    ]
    extent = [domain.x_min, domain.x_max, domain.y_min, domain.y_max]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fig.suptitle(
        f"Hybrid 2D Euler Riemann Problem - Configuration 3, t = {config.tfinal:.3f}",
        fontsize=13,
        fontweight="bold",
    )

    for ax, (field_values, title, label, cmap, vmin, vmax) in zip(axes, fields):
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
        fig.colorbar(image, ax=ax, label=label)

    if output_path is not None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches="tight", dpi=200)
            print(f"Saved figure to {output_path}")
        except OSError as exc:
            print(f"Could not save figure to {output_path}: {exc}")

    if show:
        plt.show(block=True)
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hybrid 2D Euler Configuration 3 benchmark.")
    parser.add_argument(
        "--backend",
        choices=("numpy", "cupy"),
        default="cupy",
        help="Array backend. Use cupy for CUDA execution or numpy for CPU execution.",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
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
    parser.add_argument(
        "--save",
        type=Path,
        default=Path(__file__).resolve().with_name("Riemann2D_Config3_hybrid.png"),
        help="Path for the output figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_backend(args.backend, args.device)
    if BACKEND == "cupy":
        properties = cp.cuda.runtime.getDeviceProperties(args.device)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode()
        print(f"Backend: CuPy/CUDA device {args.device}: {name}")
    else:
        print("Backend: NumPy/SciPy CPU")
    config = Riemann2DConfig3(nx=args.nx, ny=args.ny, tfinal=args.tfinal, cfl=args.cfl)
    q_final, time, steps = run_hybrid_case(
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
    vorticity_limit = None if args.vorticity_limit == 0.0 else args.vorticity_limit
    plot_hybrid_solution(
        q_final,
        config,
        output_path=args.save,
        show=not args.no_show,
        vorticity_limit=vorticity_limit,
        vorticity_cmap=args.vorticity_cmap,
    )


if __name__ == "__main__":
    main()
