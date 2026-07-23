#!/usr/bin/env python3
"""GPU hybrid compact/WENO-7 solver for 2-D Riemann problem, configuration 3.

This is a self-contained validation driver for the 2-D compressible Euler
system.  It combines

* the conservative eighth-order compact central flux formulation of
  Wang et al. (2010),
* seventh-order WENO-JS reconstruction with LLF flux splitting in local Roe
  characteristic variables,
* the shock/jump sensor and compact-WENO interface blending used by the
  accompanying 1-D validation programs,
* the conservative Wang et al. compact hyperviscosity switch (their
  Eqs. 39-42, 55, and 63), applied by operator splitting, and
* SSP-RK3 time integration.

The compact matrices are circulant and are inverted with batched FFTs.  With
``--backend cupy`` (the default), the state, sensors, WENO reconstruction,
compact solves, RK stages, and hyperviscosity remain on the CUDA device.
Only diagnostics and final plotting are copied to the CPU.

Because the compact solver is periodic while the published Riemann problem is
not, the physical square [0, 1] x [0, 1] is embedded in a padded periodic
square.  The requested resolution refers to the physical square.  The final
plots are cropped back to [0, 1] x [0, 1].

References
----------
J. Wang et al., Journal of Computational Physics 229 (2010), 5257-5279.
A. Kurganov and E. Tadmor, Numerical Methods for Partial Differential
Equations 18 (2002), 584-608.

Example
-------
python riemann2d_config3_gpu.py \
    --backend cupy --device 0 --nx 512 --ny 512 \
    --tfinal 0.3 --cfl 0.05 --mn 0.001 \
    --hyperviscosity-interval 5 --sensor-width 6 \
    --jump-threshold 0.01 --output-dir results
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Configuration and backend
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverConfig:
    nx: int = 400
    ny: int = 400
    tfinal: float = 0.3
    cfl: float = 0.05
    gamma: float = 1.4
    padding: float = 0.75
    mn: float = 0.001
    hyperviscosity_interval: int = 5
    sensor_width: int = 6
    compression_threshold: float = 3.0
    jump_threshold: float = 0.01
    shear_threshold: float | None = None
    rho_floor: float = 1.0e-10
    pressure_floor: float = 1.0e-10
    weno_epsilon: float = 1.0e-10
    weno_mode: str = "characteristic"
    dtype: str = "float64"
    backend: str = "cupy"
    device: int = 0
    strict_positivity: bool = False
    progress_every: int = 100
    output_dir: str = "results"
    prefix: str = "Riemann2D_Config3_hybrid"
    density_contours: int = 40
    vorticity_percentile: float = 99.5
    vorticity_limit: float | None = None
    save_fields: bool = True


@dataclass(frozen=True)
class PaddedGrid:
    nx_physical: int
    ny_physical: int
    pad_x: int
    pad_y: int
    nx: int
    ny: int
    dx: float
    dy: float
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @classmethod
    def build(cls, nx: int, ny: int, padding: float) -> "PaddedGrid":
        if nx < 16 or ny < 16:
            raise ValueError("nx and ny must each be at least 16")
        if padding < 0.0:
            raise ValueError("padding must be non-negative")
        dx = 1.0 / nx
        dy = 1.0 / ny
        pad_x = int(math.ceil(padding / dx))
        pad_y = int(math.ceil(padding / dy))
        solve_nx = nx + 2 * pad_x
        solve_ny = ny + 2 * pad_y
        return cls(
            nx_physical=nx,
            ny_physical=ny,
            pad_x=pad_x,
            pad_y=pad_y,
            nx=solve_nx,
            ny=solve_ny,
            dx=dx,
            dy=dy,
            xmin=-pad_x * dx,
            xmax=1.0 + pad_x * dx,
            ymin=-pad_y * dy,
            ymax=1.0 + pad_y * dy,
        )

    @property
    def physical_slice(self) -> tuple[slice, slice]:
        return (
            slice(self.pad_y, self.pad_y + self.ny_physical),
            slice(self.pad_x, self.pad_x + self.nx_physical),
        )


class Backend:
    def __init__(self, name: str, device: int, dtype_name: str) -> None:
        self.name = name
        self.device = device
        if dtype_name not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")

        if name == "numpy":
            self.xp = np
            self.dtype = np.float32 if dtype_name == "float32" else np.float64
            self.is_gpu = False
            self.description = "NumPy CPU debug backend"
            return

        if name != "cupy":
            raise ValueError("backend must be 'cupy' or 'numpy'")

        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError(
                "CuPy is required for --backend cupy. Install a CuPy build "
                "compatible with the workstation CUDA driver/toolkit."
            ) from exc

        cp.cuda.Device(device).use()
        self.xp = cp
        self.dtype = cp.float32 if dtype_name == "float32" else cp.float64
        self.is_gpu = True
        properties = cp.cuda.runtime.getDeviceProperties(device)
        raw_name = properties["name"]
        if isinstance(raw_name, bytes):
            raw_name = raw_name.decode(errors="replace")
        total_gib = properties["totalGlobalMem"] / 1024**3
        self.description = f"CuPy/CUDA device {device}: {raw_name} ({total_gib:.1f} GiB)"

    def synchronize(self) -> None:
        if self.is_gpu:
            self.xp.cuda.Stream.null.synchronize()

    def to_numpy(self, values: Any) -> np.ndarray:
        if self.is_gpu:
            return self.xp.asnumpy(values)
        return np.asarray(values)

    def memory_string(self) -> str:
        if not self.is_gpu:
            return ""
        pool = self.xp.get_default_memory_pool()
        used = pool.used_bytes() / 1024**3
        total = pool.total_bytes() / 1024**3
        return f", GPU pool used/held={used:.2f}/{total:.2f} GiB"


# -----------------------------------------------------------------------------
# Euler helpers and configuration-3 initial state
# -----------------------------------------------------------------------------


class Euler2D:
    """Ideal-gas Euler equations, U = [rho, rho*u, rho*v, E]."""

    def __init__(
        self,
        xp: Any,
        gamma: float,
        rho_floor: float,
        pressure_floor: float,
    ) -> None:
        self.xp = xp
        self.gamma = gamma
        self.gm1 = gamma - 1.0
        self.rho_floor = rho_floor
        self.pressure_floor = pressure_floor

    def primitive(self, q: Any, safe: bool = True) -> tuple[Any, Any, Any, Any]:
        xp = self.xp
        rho = xp.maximum(q[0], self.rho_floor) if safe else q[0]
        u = q[1] / rho
        v = q[2] / rho
        kinetic = 0.5 * rho * (u * u + v * v)
        pressure = self.gm1 * (q[3] - kinetic)
        if safe:
            pressure = xp.maximum(pressure, self.pressure_floor)
        return rho, u, v, pressure

    def conservative(self, rho: Any, u: Any, v: Any, pressure: Any) -> Any:
        xp = self.xp
        q = xp.empty((4, *rho.shape), dtype=rho.dtype)
        q[0] = rho
        q[1] = rho * u
        q[2] = rho * v
        q[3] = pressure / self.gm1 + 0.5 * rho * (u * u + v * v)
        return q

    def clean(self, q: Any, strict: bool = False) -> Any:
        """Emergency positivity protection, matching the 1-D validation codes."""
        xp = self.xp
        rho_raw = q[0]
        rho = xp.maximum(rho_raw, self.rho_floor)
        u = q[1] / rho
        v = q[2] / rho
        kinetic = 0.5 * rho * (u * u + v * v)
        pressure_raw = self.gm1 * (q[3] - kinetic)

        if strict:
            bad = xp.any((rho_raw <= 0.0) | (pressure_raw <= 0.0))
            if bool(bad):
                min_rho = float(xp.min(rho_raw))
                min_pressure = float(xp.min(pressure_raw))
                raise FloatingPointError(
                    "Non-positive state produced: "
                    f"min(rho)={min_rho:.6e}, min(p)={min_pressure:.6e}"
                )

        fixed = xp.array(q, copy=True)
        fixed[0] = rho
        fixed[3] = xp.where(
            pressure_raw > self.pressure_floor,
            q[3],
            self.pressure_floor / self.gm1 + kinetic,
        )
        return fixed

    def normal_flux(self, q_normal: Any) -> Any:
        """Flux for state [rho, rho*u_n, rho*u_t, E]."""
        xp = self.xp
        rho = xp.maximum(q_normal[0], self.rho_floor)
        un = q_normal[1] / rho
        ut = q_normal[2] / rho
        kinetic = 0.5 * rho * (un * un + ut * ut)
        pressure = xp.maximum(
            self.gm1 * (q_normal[3] - kinetic), self.pressure_floor
        )
        flux = xp.empty_like(q_normal)
        flux[0] = q_normal[1]
        flux[1] = rho * un * un + pressure
        flux[2] = rho * un * ut
        flux[3] = (q_normal[3] + pressure) * un
        return flux


def initialize_configuration_3(
    backend: Backend,
    grid: PaddedGrid,
    equation: Euler2D,
) -> Any:
    """Kurganov-Tadmor configuration 3, extended through the padded domain."""
    xp = backend.xp
    dtype = backend.dtype
    x = grid.xmin + (xp.arange(grid.nx, dtype=dtype) + 0.5) * grid.dx
    y = grid.ymin + (xp.arange(grid.ny, dtype=dtype) + 0.5) * grid.dy
    xg, yg = xp.meshgrid(x, y, indexing="xy")

    ne = (xg >= 0.5) & (yg >= 0.5)  # state 1
    nw = (xg < 0.5) & (yg >= 0.5)   # state 2
    sw = (xg < 0.5) & (yg < 0.5)    # state 3
    # Remaining quadrant is state 4 (SE).

    rho = xp.full((grid.ny, grid.nx), 0.5323, dtype=dtype)
    u = xp.zeros_like(rho)
    v = xp.full_like(rho, 1.206)
    pressure = xp.full_like(rho, 0.3)

    rho = xp.where(ne, 1.5, rho)
    u = xp.where(ne, 0.0, u)
    v = xp.where(ne, 0.0, v)
    pressure = xp.where(ne, 1.5, pressure)

    rho = xp.where(nw, 0.5323, rho)
    u = xp.where(nw, 1.206, u)
    v = xp.where(nw, 0.0, v)
    pressure = xp.where(nw, 0.3, pressure)

    rho = xp.where(sw, 0.138, rho)
    u = xp.where(sw, 1.206, u)
    v = xp.where(sw, 1.206, v)
    pressure = xp.where(sw, 0.029, pressure)

    return equation.conservative(rho, u, v, pressure)


# -----------------------------------------------------------------------------
# Sensor and compact/WENO spatial operator
# -----------------------------------------------------------------------------


def relative_jump_axis(values: Any, axis: int, xp: Any) -> Any:
    eps = 1.0e-14 + 1.0e-12 * xp.max(xp.abs(values))
    right = xp.roll(values, -1, axis=axis)
    left = xp.roll(values, 1, axis=axis)
    jump_r = xp.abs(right - values) / (xp.abs(right) + xp.abs(values) + eps)
    jump_l = xp.abs(values - left) / (xp.abs(values) + xp.abs(left) + eps)
    curvature = xp.abs(right - 2.0 * values + left)
    curvature /= xp.abs(right) + 2.0 * xp.abs(values) + xp.abs(left) + eps
    return xp.maximum(xp.maximum(jump_l, jump_r), curvature)


def dilate_square_periodic(mask: Any, width: int, xp: Any) -> Any:
    """Chebyshev-width dilation using O(width), rather than O(width^2), rolls."""
    if width <= 0:
        return mask
    expanded_x = xp.array(mask, copy=True)
    for offset in range(1, width + 1):
        expanded_x |= xp.roll(mask, offset, axis=-1)
        expanded_x |= xp.roll(mask, -offset, axis=-1)
    expanded = xp.array(expanded_x, copy=True)
    for offset in range(1, width + 1):
        expanded |= xp.roll(expanded_x, offset, axis=-2)
        expanded |= xp.roll(expanded_x, -offset, axis=-2)
    return expanded


class ShockSensor2D:
    def __init__(
        self,
        xp: Any,
        grid: PaddedGrid,
        equation: Euler2D,
        width: int,
        compression_threshold: float,
        jump_threshold: float,
        shear_threshold: float | None,
    ) -> None:
        self.xp = xp
        self.grid = grid
        self.equation = equation
        self.width = width
        self.compression_threshold = compression_threshold
        self.jump_threshold = jump_threshold
        self.shear_threshold = shear_threshold

    def detect(self, q: Any) -> Any:
        xp = self.xp
        rho, u, v, pressure = self.equation.primitive(q, safe=True)
        du_dx = (xp.roll(u, -1, axis=-1) - xp.roll(u, 1, axis=-1)) / (
            2.0 * self.grid.dx
        )
        du_dy = (xp.roll(u, -1, axis=-2) - xp.roll(u, 1, axis=-2)) / (
            2.0 * self.grid.dy
        )
        dv_dx = (xp.roll(v, -1, axis=-1) - xp.roll(v, 1, axis=-1)) / (
            2.0 * self.grid.dx
        )
        dv_dy = (xp.roll(v, -1, axis=-2) - xp.roll(v, 1, axis=-2)) / (
            2.0 * self.grid.dy
        )

        divergence = du_dx + dv_dy
        div_rms = xp.sqrt(xp.mean(divergence * divergence)) + 1.0e-30
        compression = divergence < -self.compression_threshold * div_rms

        internal_energy = pressure / (rho * self.equation.gm1)
        rho_jump = xp.maximum(
            relative_jump_axis(rho, -1, xp), relative_jump_axis(rho, -2, xp)
        )
        pressure_jump = xp.maximum(
            relative_jump_axis(pressure, -1, xp),
            relative_jump_axis(pressure, -2, xp),
        )
        energy_jump = xp.maximum(
            relative_jump_axis(internal_energy, -1, xp),
            relative_jump_axis(internal_energy, -2, xp),
        )

        mask = (
            compression
            | (rho_jump > self.jump_threshold)
            | (pressure_jump > self.jump_threshold)
            | (energy_jump > self.jump_threshold)
        )

        if self.shear_threshold is not None:
            vorticity = dv_dx - du_dy
            vort_rms = xp.sqrt(xp.mean(vorticity * vorticity)) + 1.0e-30
            mask |= xp.abs(vorticity) > self.shear_threshold * vort_rms

        return dilate_square_periodic(mask, self.width, xp)


def weno7_left(
    v1: Any,
    v2: Any,
    v3: Any,
    v4: Any,
    v5: Any,
    v6: Any,
    v7: Any,
    eps: float,
    xp: Any,
) -> Any:
    """Seventh-order Jiang-Shu WENO left-biased interface reconstruction."""
    q0 = -0.25 * v1 + (13.0 / 12.0) * v2 - (23.0 / 12.0) * v3 + (25.0 / 12.0) * v4
    q1 = (1.0 / 12.0) * v2 - (5.0 / 12.0) * v3 + (13.0 / 12.0) * v4 + 0.25 * v5
    q2 = -(1.0 / 12.0) * v3 + (7.0 / 12.0) * v4 + (7.0 / 12.0) * v5 - (1.0 / 12.0) * v6
    q3 = 0.25 * v4 + (13.0 / 12.0) * v5 - (5.0 / 12.0) * v6 + (1.0 / 12.0) * v7

    beta0 = (
        v1 * (544.0 * v1 - 3882.0 * v2 + 4642.0 * v3 - 1854.0 * v4)
        + v2 * (7043.0 * v2 - 17246.0 * v3 + 7042.0 * v4)
        + v3 * (11003.0 * v3 - 9402.0 * v4)
        + 2107.0 * v4 * v4
    )
    beta1 = (
        v2 * (267.0 * v2 - 1642.0 * v3 + 1602.0 * v4 - 494.0 * v5)
        + v3 * (2843.0 * v3 - 5966.0 * v4 + 1922.0 * v5)
        + v4 * (3443.0 * v4 - 2522.0 * v5)
        + 547.0 * v5 * v5
    )
    beta2 = (
        v3 * (547.0 * v3 - 2522.0 * v4 + 1922.0 * v5 - 494.0 * v6)
        + v4 * (3443.0 * v4 - 5966.0 * v5 + 1602.0 * v6)
        + v5 * (2843.0 * v5 - 1642.0 * v6)
        + 267.0 * v6 * v6
    )
    beta3 = (
        v4 * (2107.0 * v4 - 9402.0 * v5 + 7042.0 * v6 - 1854.0 * v7)
        + v5 * (11003.0 * v5 - 17246.0 * v6 + 4642.0 * v7)
        + v6 * (7043.0 * v6 - 3882.0 * v7)
        + 547.0 * v7 * v7
    )

    # The quadratic forms are non-negative analytically; clipping only removes
    # tiny negative round-off values on GPUs.
    beta0 = xp.maximum(beta0, 0.0)
    beta1 = xp.maximum(beta1, 0.0)
    beta2 = xp.maximum(beta2, 0.0)
    beta3 = xp.maximum(beta3, 0.0)

    a0 = (1.0 / 35.0) / (eps + beta0) ** 2
    a1 = (12.0 / 35.0) / (eps + beta1) ** 2
    a2 = (18.0 / 35.0) / (eps + beta2) ** 2
    a3 = (4.0 / 35.0) / (eps + beta3) ** 2
    asum = a0 + a1 + a2 + a3
    return (a0 * q0 + a1 * q1 + a2 * q2 + a3 * q3) / asum


class PeriodicCompactDerivative:
    """Batched cyclic compact solve along the last array axis."""

    def __init__(self, xp: Any, n: int, dx: float, dtype: Any, alpha: float = 3.0 / 8.0) -> None:
        self.xp = xp
        self.n = n
        self.dx = dx
        self.alpha = alpha
        theta = 2.0 * np.pi * np.fft.fftfreq(n)
        eigenvalues = 1.0 + 2.0 * alpha * np.cos(theta)
        self.eigenvalues = xp.asarray(eigenvalues, dtype=dtype)

    def from_interface_flux(self, interface_flux: Any) -> Any:
        xp = self.xp
        rhs = (interface_flux - xp.roll(interface_flux, 1, axis=-1)) / self.dx
        derivative_hat = xp.fft.fft(rhs, axis=-1) / self.eigenvalues
        return xp.fft.ifft(derivative_hat, axis=-1).real

    def from_point_values(self, values: Any) -> Any:
        """Eighth-order compact first derivative, Wang Eq. (22)."""
        xp = self.xp
        a1 = 25.0 / 32.0
        b1 = 1.0 / 20.0
        c1 = -1.0 / 480.0
        rhs = (
            a1 * (xp.roll(values, -1, axis=-1) - xp.roll(values, 1, axis=-1))
            + b1 * (xp.roll(values, -2, axis=-1) - xp.roll(values, 2, axis=-1))
            + c1 * (xp.roll(values, -3, axis=-1) - xp.roll(values, 3, axis=-1))
        ) / self.dx
        derivative_hat = xp.fft.fft(rhs, axis=-1) / self.eigenvalues
        return xp.fft.ifft(derivative_hat, axis=-1).real


def smooth_compact_interface_flux(point_flux: Any, xp: Any) -> Any:
    """Conservative interface flux corresponding to the compact derivative."""
    a1 = 25.0 / 32.0
    b1 = 1.0 / 20.0
    c1 = -1.0 / 480.0
    return (
        c1 * (xp.roll(point_flux, -3, axis=-1) + xp.roll(point_flux, 2, axis=-1))
        + (b1 + c1)
        * (xp.roll(point_flux, -2, axis=-1) + xp.roll(point_flux, 1, axis=-1))
        + (a1 + b1 + c1)
        * (xp.roll(point_flux, -1, axis=-1) + point_flux)
    )


def roe_characteristic_matrices(
    q_left: Any,
    q_right: Any,
    equation: Euler2D,
) -> tuple[Any, Any]:
    """Analytical left/right Roe eigenvectors for a normal-direction state."""
    xp = equation.xp
    rho_l = xp.maximum(q_left[0], equation.rho_floor)
    rho_r = xp.maximum(q_right[0], equation.rho_floor)
    un_l = q_left[1] / rho_l
    un_r = q_right[1] / rho_r
    ut_l = q_left[2] / rho_l
    ut_r = q_right[2] / rho_r
    p_l = xp.maximum(
        equation.gm1
        * (q_left[3] - 0.5 * rho_l * (un_l * un_l + ut_l * ut_l)),
        equation.pressure_floor,
    )
    p_r = xp.maximum(
        equation.gm1
        * (q_right[3] - 0.5 * rho_r * (un_r * un_r + ut_r * ut_r)),
        equation.pressure_floor,
    )
    h_l = (q_left[3] + p_l) / rho_l
    h_r = (q_right[3] + p_r) / rho_r

    sr_l = xp.sqrt(rho_l)
    sr_r = xp.sqrt(rho_r)
    denom = sr_l + sr_r
    un = (sr_l * un_l + sr_r * un_r) / denom
    ut = (sr_l * ut_l + sr_r * ut_r) / denom
    enthalpy = (sr_l * h_l + sr_r * h_r) / denom
    q2 = un * un + ut * ut
    c2 = xp.maximum(equation.gm1 * (enthalpy - 0.5 * q2), 1.0e-14)
    c = xp.sqrt(c2)
    beta = equation.gm1 / c2

    n = q_left.shape[1]
    left = xp.empty((4, 4, n), dtype=q_left.dtype)
    right = xp.empty((4, 4, n), dtype=q_left.dtype)

    # Right eigenvectors: acoustic-, shear, entropy, acoustic+.
    right[0, 0] = 1.0
    right[1, 0] = un - c
    right[2, 0] = ut
    right[3, 0] = enthalpy - un * c

    right[0, 1] = 0.0
    right[1, 1] = 0.0
    right[2, 1] = 1.0
    right[3, 1] = ut

    right[0, 2] = 1.0
    right[1, 2] = un
    right[2, 2] = ut
    right[3, 2] = 0.5 * q2

    right[0, 3] = 1.0
    right[1, 3] = un + c
    right[2, 3] = ut
    right[3, 3] = enthalpy + un * c

    left[0, 0] = 0.5 * (0.5 * beta * q2 + un / c)
    left[0, 1] = -0.5 * (beta * un + 1.0 / c)
    left[0, 2] = -0.5 * beta * ut
    left[0, 3] = 0.5 * beta

    left[1, 0] = -ut
    left[1, 1] = 0.0
    left[1, 2] = 1.0
    left[1, 3] = 0.0

    left[2, 0] = 1.0 - 0.5 * beta * q2
    left[2, 1] = beta * un
    left[2, 2] = beta * ut
    left[2, 3] = -beta

    left[3, 0] = 0.5 * (0.5 * beta * q2 - un / c)
    left[3, 1] = -0.5 * (beta * un - 1.0 / c)
    left[3, 2] = -0.5 * beta * ut
    left[3, 3] = 0.5 * beta
    return left, right


class HybridEulerOperator2D:
    def __init__(
        self,
        backend: Backend,
        grid: PaddedGrid,
        equation: Euler2D,
        sensor: ShockSensor2D,
        weno_epsilon: float,
        weno_mode: str,
    ) -> None:
        self.backend = backend
        self.xp = backend.xp
        self.grid = grid
        self.equation = equation
        self.sensor = sensor
        self.weno_epsilon = weno_epsilon
        self.weno_mode = weno_mode
        self.compact_x = PeriodicCompactDerivative(
            self.xp, grid.nx, grid.dx, backend.dtype
        )
        self.compact_y = PeriodicCompactDerivative(
            self.xp, grid.ny, grid.dy, backend.dtype
        )

    def _weno_raw_flux_masked(
        self,
        q: Any,
        point_flux: Any,
        alpha: float,
        needed: Any,
    ) -> Any:
        """Compute raw WENO flux only at requested i+1/2 interfaces."""
        xp = self.xp
        raw = xp.zeros_like(q)
        rows, cols = xp.nonzero(needed)
        if rows.size == 0:
            return raw

        nline = q.shape[-1]
        q_left = q[:, rows, cols]
        q_right = q[:, rows, (cols + 1) % nline]

        if self.weno_mode == "characteristic":
            left_matrix, right_matrix = roe_characteristic_matrices(
                q_left, q_right, self.equation
            )

            def projected(offset: int, sign: float) -> Any:
                col = (cols + offset) % nline
                split = 0.5 * (
                    point_flux[:, rows, col] + sign * alpha * q[:, rows, col]
                )
                return xp.einsum("abk,bk->ak", left_matrix, split)

            plus = [projected(offset, +1.0) for offset in (-3, -2, -1, 0, 1, 2, 3)]
            minus = [projected(offset, -1.0) for offset in (4, 3, 2, 1, 0, -1, -2)]
            char_flux = weno7_left(
                *plus, eps=self.weno_epsilon, xp=xp
            ) + weno7_left(*minus, eps=self.weno_epsilon, xp=xp)
            interface_flux = xp.einsum("abk,bk->ak", right_matrix, char_flux)
        else:
            def split_component(offset: int, sign: float) -> Any:
                col = (cols + offset) % nline
                return 0.5 * (
                    point_flux[:, rows, col] + sign * alpha * q[:, rows, col]
                )

            plus = [
                split_component(offset, +1.0)
                for offset in (-3, -2, -1, 0, 1, 2, 3)
            ]
            minus = [
                split_component(offset, -1.0)
                for offset in (4, 3, 2, 1, 0, -1, -2)
            ]
            interface_flux = weno7_left(
                *plus, eps=self.weno_epsilon, xp=xp
            ) + weno7_left(*minus, eps=self.weno_epsilon, xp=xp)

        raw[:, rows, cols] = interface_flux
        return raw

    def _axis_derivative(
        self,
        q_normal: Any,
        shock_mask: Any,
        compact: PeriodicCompactDerivative,
    ) -> Any:
        xp = self.xp
        point_flux = self.equation.normal_flux(q_normal)
        rho = xp.maximum(q_normal[0], self.equation.rho_floor)
        un = q_normal[1] / rho
        ut = q_normal[2] / rho
        pressure = xp.maximum(
            self.equation.gm1
            * (q_normal[3] - 0.5 * rho * (un * un + ut * ut)),
            self.equation.pressure_floor,
        )
        sound_speed = xp.sqrt(self.equation.gamma * pressure / rho)
        # Keep the LLF speed as a device scalar; only the CFL reduction is
        # transferred to the host once per full time step.
        alpha = xp.max(xp.abs(un) + sound_speed)

        smooth_flux = smooth_compact_interface_flux(point_flux, xp)
        mask_right = xp.roll(shock_mask, -1, axis=-1)
        shock_edge = shock_mask & mask_right
        smooth_edge = (~shock_mask) & (~mask_right)
        joint_edge = ~(shock_edge | smooth_edge)
        required = shock_edge | joint_edge

        # Eq. (28) needs raw WENO fluxes at i-1/2, i+1/2, and i+3/2.
        raw_needed = required | xp.roll(required, 1, axis=-1) | xp.roll(
            required, -1, axis=-1
        )
        weno_raw = self._weno_raw_flux_masked(q_normal, point_flux, alpha, raw_needed)
        weno_consistent = (
            compact.alpha * xp.roll(weno_raw, -1, axis=-1)
            + weno_raw
            + compact.alpha * xp.roll(weno_raw, 1, axis=-1)
        )

        hybrid_flux = xp.where(
            smooth_edge[None, ...],
            smooth_flux,
            xp.where(
                shock_edge[None, ...],
                weno_consistent,
                0.5 * (smooth_flux + weno_consistent),
            ),
        )
        return compact.from_interface_flux(hybrid_flux)

    def rhs(self, q: Any) -> Any:
        xp = self.xp
        q_safe = self.equation.clean(q, strict=False)
        shock_mask = self.sensor.detect(q_safe)

        derivative_x = self._axis_derivative(q_safe, shock_mask, self.compact_x)

        # Rotate y into the normal direction: [rho, rho*v, rho*u, E],
        # and put y along the last array axis for the batched compact solve.
        q_y = xp.swapaxes(q_safe[[0, 2, 1, 3]], -1, -2)
        mask_y = xp.swapaxes(shock_mask, -1, -2)
        derivative_y_rot = self._axis_derivative(q_y, mask_y, self.compact_y)
        derivative_y = xp.swapaxes(derivative_y_rot, -1, -2)[[0, 2, 1, 3]]
        return -derivative_x - derivative_y

    def stable_time_step(self, q: Any, cfl: float, remaining: float) -> float:
        xp = self.xp
        rho, u, v, pressure = self.equation.primitive(q, safe=True)
        sound_speed = xp.sqrt(self.equation.gamma * pressure / rho)
        rate = (xp.abs(u) + sound_speed) / self.grid.dx
        rate += (xp.abs(v) + sound_speed) / self.grid.dy
        dt = cfl / float(xp.max(rate))
        return min(dt, remaining)


# -----------------------------------------------------------------------------
# Wang compact hyperviscosity, conservative switched form
# -----------------------------------------------------------------------------


class WangHyperviscosityAxis:
    """One directional application of Wang et al. Eqs. (39)-(42), (63)."""

    def __init__(self, xp: Any, n: int, dx: float, dtype: Any, mn: float) -> None:
        self.xp = xp
        self.n = n
        self.dx = dx
        self.mn = mn

        self.a2_h = 4.0 / 9.0
        self.b2_h = 1.0 / 36.0
        self.a2_t = 20.0 / 27.0
        self.b2_t = 25.0 / 216.0

        self.a3_h = 344.0 / 1179.0
        self.b3_h = (38.0 * self.a3_h - 9.0) / 214.0
        self.a3_t = (696.0 - 1191.0 * self.a3_h) / 428.0
        self.b3_t = (1227.0 * self.a3_h - 147.0) / 1070.0

        theta = 2.0 * np.pi * np.fft.fftfreq(n)
        a_eig = 1.0 + 2.0 * self.a2_h * np.cos(theta) + 2.0 * self.b2_h * np.cos(2.0 * theta)
        c_eig = 1.0 + 2.0 * self.a3_h * np.cos(theta) + 2.0 * self.b3_h * np.cos(2.0 * theta)
        d_eig = (
            2.0 * self.a3_t * (np.cos(theta) - 1.0)
            + 2.0 * self.b3_t * (np.cos(2.0 * theta) - 1.0)
        ) / dx**2
        self.a_eig = xp.asarray(a_eig, dtype=dtype)
        self.c_eig = xp.asarray(c_eig, dtype=dtype)
        self.d_eig = xp.asarray(d_eig, dtype=dtype)

    def _solve(self, rhs: Any, eigenvalues: Any) -> Any:
        xp = self.xp
        return xp.fft.ifft(xp.fft.fft(rhs, axis=-1) / eigenvalues, axis=-1).real

    def _apply_a(self, values: Any) -> Any:
        xp = self.xp
        return (
            values
            + self.a2_h
            * (xp.roll(values, 1, axis=-1) + xp.roll(values, -1, axis=-1))
            + self.b2_h
            * (xp.roll(values, 2, axis=-1) + xp.roll(values, -2, axis=-1))
        )

    def apply(self, values: Any, shock_mask: Any, dt_filter: float) -> Any:
        if self.mn <= 0.0 or dt_filter <= 0.0:
            return values
        xp = self.xp

        b_values = (
            -self.b2_t * xp.roll(values, 2, axis=-1)
            - self.a2_t * xp.roll(values, 1, axis=-1)
            + self.a2_t * xp.roll(values, -1, axis=-1)
            + self.b2_t * xp.roll(values, -2, axis=-1)
        ) / self.dx
        first_derivative = self._solve(b_values, self.a_eig)

        g_half = (
            self.b2_t * xp.roll(first_derivative, 1, axis=-1)
            + (self.a2_t + self.b2_t) * first_derivative
            + (self.a2_t + self.b2_t)
            * xp.roll(first_derivative, -1, axis=-1)
            + self.b2_t * xp.roll(first_derivative, -2, axis=-1)
        ) / self.dx

        d_half = (
            -self.b3_t * xp.roll(values, 1, axis=-1)
            - (self.a3_t + self.b3_t) * values
            + (self.a3_t + self.b3_t) * xp.roll(values, -1, axis=-1)
            + self.b3_t * xp.roll(values, -2, axis=-1)
        ) / self.dx**2
        h_half = self._apply_a(self._solve(d_half, self.c_eig))

        mask_right = xp.roll(shock_mask, -1, axis=-1)
        shock_edge = shock_mask & mask_right
        smooth_edge = (~shock_mask) & (~mask_right)
        g_modified = xp.where(
            smooth_edge[None, ...],
            g_half,
            xp.where(
                shock_edge[None, ...], h_half, 0.5 * (g_half + h_half)
            ),
        )

        repeated_first = self._solve(
            g_modified - xp.roll(g_modified, 1, axis=-1), self.a_eig
        )
        explicit_state = values - dt_filter * self.mn * repeated_first

        # Semi-implicit direct second derivative:
        # (C - dt*mn*D) f_new = C (f_old - dt*mn*(f'_n)'_n).
        numerator = self.c_eig * xp.fft.fft(explicit_state, axis=-1)
        denominator = self.c_eig - dt_filter * self.mn * self.d_eig
        return xp.fft.ifft(numerator / denominator, axis=-1).real


class WangHyperviscosity2D:
    def __init__(
        self,
        backend: Backend,
        grid: PaddedGrid,
        equation: Euler2D,
        mn: float,
    ) -> None:
        self.xp = backend.xp
        self.equation = equation
        self.mn = mn
        self.x_axis = WangHyperviscosityAxis(
            self.xp, grid.nx, grid.dx, backend.dtype, mn
        )
        self.y_axis = WangHyperviscosityAxis(
            self.xp, grid.ny, grid.dy, backend.dtype, mn
        )

    def apply(self, q: Any, shock_mask: Any, dt_filter: float, strict: bool) -> Any:
        if self.mn <= 0.0:
            return q
        xp = self.xp
        filtered = self.x_axis.apply(q, shock_mask, dt_filter)
        filtered_y = xp.swapaxes(filtered, -1, -2)
        mask_y = xp.swapaxes(shock_mask, -1, -2)
        filtered_y = self.y_axis.apply(filtered_y, mask_y, dt_filter)
        filtered = xp.swapaxes(filtered_y, -1, -2)
        return self.equation.clean(filtered, strict=strict)


# -----------------------------------------------------------------------------
# Time integration, diagnostics, and plotting
# -----------------------------------------------------------------------------


def ssprk3_step(
    q: Any,
    dt: float,
    operator: HybridEulerOperator2D,
    equation: Euler2D,
    strict: bool,
) -> Any:
    q0 = q
    q1 = equation.clean(q0 + dt * operator.rhs(q0), strict=strict)
    q2 = equation.clean(
        0.75 * q0 + 0.25 * (q1 + dt * operator.rhs(q1)), strict=strict
    )
    return equation.clean(
        (1.0 / 3.0) * q0
        + (2.0 / 3.0) * (q2 + dt * operator.rhs(q2)),
        strict=strict,
    )


def compact_vorticity(
    q: Any,
    equation: Euler2D,
    compact_x: PeriodicCompactDerivative,
    compact_y: PeriodicCompactDerivative,
    xp: Any,
) -> Any:
    _rho, u, v, _pressure = equation.primitive(q, safe=True)
    dv_dx = compact_x.from_point_values(v)
    du_dy = xp.swapaxes(
        compact_y.from_point_values(xp.swapaxes(u, -1, -2)), -1, -2
    )
    return dv_dx - du_dy


def make_plots_and_save(
    backend: Backend,
    grid: PaddedGrid,
    equation: Euler2D,
    operator: HybridEulerOperator2D,
    q: Any,
    config: SolverConfig,
    steps: int,
    elapsed: float,
) -> dict[str, Path]:
    xp = backend.xp
    ys, xs = grid.physical_slice
    rho, u, v, pressure = equation.primitive(q, safe=True)
    omega = compact_vorticity(q, equation, operator.compact_x, operator.compact_y, xp)

    rho_np = backend.to_numpy(rho[ys, xs])
    u_np = backend.to_numpy(u[ys, xs])
    v_np = backend.to_numpy(v[ys, xs])
    p_np = backend.to_numpy(pressure[ys, xs])
    omega_np = backend.to_numpy(omega[ys, xs])
    x_np = (np.arange(grid.nx_physical) + 0.5) / grid.nx_physical
    y_np = (np.arange(grid.ny_physical) + 0.5) / grid.ny_physical

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    time_tag = f"{config.tfinal:.3f}".replace(".", "p")
    base = f"{config.prefix}_t{time_tag}_n{grid.nx_physical}x{grid.ny_physical}"
    density_path = output_dir / f"{base}_density.png"
    vorticity_path = output_dir / f"{base}_vorticity.png"
    fields_path = output_dir / f"{base}_fields.npz"
    metadata_path = output_dir / f"{base}_metadata.json"

    fig, ax = plt.subplots(figsize=(8.0, 7.0), constrained_layout=True)
    levels = max(8, config.density_contours)
    x_edges = np.linspace(0.0, 1.0, grid.nx_physical + 1)
    y_edges = np.linspace(0.0, 1.0, grid.ny_physical + 1)
    filled = ax.pcolormesh(
        x_edges, y_edges, rho_np, shading="flat", cmap="viridis"
    )
    ax.contour(
        x_np,
        y_np,
        rho_np,
        levels=levels,
        colors="black",
        linewidths=0.30,
        alpha=0.55,
    )
    fig.colorbar(filled, ax=ax, label=r"Density, $\rho$")
    ax.set_title(
        f"2-D Riemann problem, configuration 3: density at t = {config.tfinal:g}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.savefig(density_path, dpi=220)
    plt.close(fig)

    if config.vorticity_limit is not None:
        vort_limit = abs(config.vorticity_limit)
    else:
        vort_limit = float(
            np.percentile(np.abs(omega_np), config.vorticity_percentile)
        )
    if not np.isfinite(vort_limit) or vort_limit <= np.finfo(float).eps:
        vort_limit = max(float(np.max(np.abs(omega_np))), 1.0)

    fig, ax = plt.subplots(figsize=(8.0, 7.0), constrained_layout=True)
    image = ax.imshow(
        omega_np,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-vort_limit,
        vmax=vort_limit,
    )
    fig.colorbar(image, ax=ax, label=r"Vorticity, $\omega_z=\partial_xv-\partial_yu$")
    ax.set_title(
        f"2-D Riemann problem, configuration 3: vorticity at t = {config.tfinal:g}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.savefig(vorticity_path, dpi=220)
    plt.close(fig)

    metadata = {
        "configuration": asdict(config),
        "grid": asdict(grid),
        "steps": steps,
        "elapsed_seconds": elapsed,
        "density_min": float(np.min(rho_np)),
        "density_max": float(np.max(rho_np)),
        "pressure_min": float(np.min(p_np)),
        "pressure_max": float(np.max(p_np)),
        "vorticity_min": float(np.min(omega_np)),
        "vorticity_max": float(np.max(omega_np)),
        "vorticity_plot_limit": vort_limit,
        "backend": backend.description,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    paths = {
        "density": density_path,
        "vorticity": vorticity_path,
        "metadata": metadata_path,
    }
    if config.save_fields:
        np.savez_compressed(
            fields_path,
            x=x_np,
            y=y_np,
            rho=rho_np,
            u=u_np,
            v=v_np,
            pressure=p_np,
            vorticity=omega_np,
            time=np.array(config.tfinal),
        )
        paths["fields"] = fields_path
    return paths


def run(config: SolverConfig) -> dict[str, Path]:
    backend = Backend(config.backend, config.device, config.dtype)
    xp = backend.xp
    grid = PaddedGrid.build(config.nx, config.ny, config.padding)
    equation = Euler2D(
        xp,
        gamma=config.gamma,
        rho_floor=config.rho_floor,
        pressure_floor=config.pressure_floor,
    )
    sensor = ShockSensor2D(
        xp,
        grid,
        equation,
        width=config.sensor_width,
        compression_threshold=config.compression_threshold,
        jump_threshold=config.jump_threshold,
        shear_threshold=config.shear_threshold,
    )
    operator = HybridEulerOperator2D(
        backend,
        grid,
        equation,
        sensor,
        weno_epsilon=config.weno_epsilon,
        weno_mode=config.weno_mode,
    )
    hyperviscosity = WangHyperviscosity2D(
        backend, grid, equation, mn=config.mn
    )

    q = initialize_configuration_3(backend, grid, equation)
    q = equation.clean(q, strict=True)
    backend.synchronize()

    bytes_per_state = 4 * grid.nx * grid.ny * np.dtype(config.dtype).itemsize
    print(backend.description)
    print(
        f"Physical grid: {config.nx} x {config.ny}; "
        f"padded solve grid: {grid.nx} x {grid.ny}; "
        f"dx={grid.dx:.6e}, dy={grid.dy:.6e}"
    )
    print(
        f"One conservative state uses {bytes_per_state / 1024**2:.1f} MiB; "
        f"WENO={config.weno_mode}, mn={config.mn:g}, "
        f"filter interval={config.hyperviscosity_interval}"
    )
    initial_speed_bound = 2.10
    if config.padding < initial_speed_bound * config.tfinal:
        print(
            "WARNING: the padding is smaller than the initial wave-speed travel "
            "estimate. Increase --padding if boundary-generated waves appear in "
            "the physical crop."
        )

    current_time = 0.0
    step = 0
    filter_elapsed = 0.0
    backend.synchronize()
    start = time.perf_counter()
    last_report = start

    while current_time < config.tfinal - 10.0 * np.finfo(float).eps:
        dt = operator.stable_time_step(
            q, config.cfl, config.tfinal - current_time
        )
        q = ssprk3_step(
            q,
            dt,
            operator,
            equation,
            strict=config.strict_positivity,
        )
        current_time += dt
        filter_elapsed += dt
        step += 1

        if (
            config.mn > 0.0
            and config.hyperviscosity_interval > 0
            and step % config.hyperviscosity_interval == 0
        ):
            shock_mask = sensor.detect(q)
            q = hyperviscosity.apply(
                q,
                shock_mask,
                filter_elapsed,
                strict=config.strict_positivity,
            )
            filter_elapsed = 0.0

        if config.progress_every > 0 and (
            step % config.progress_every == 0 or current_time >= config.tfinal
        ):
            backend.synchronize()
            now = time.perf_counter()
            rho, _u, _v, pressure = equation.primitive(q, safe=False)
            min_rho = float(xp.min(rho))
            min_pressure = float(xp.min(pressure))
            shock_fraction = float(xp.mean(sensor.detect(q)))
            chunk = now - last_report
            print(
                f"step={step:7d}, t={current_time:.8f}, dt={dt:.3e}, "
                f"chunk={chunk:.2f}s, WENO nodes={100.0 * shock_fraction:.2f}%, "
                f"min(rho)={min_rho:.4e}, min(p)={min_pressure:.4e}"
                f"{backend.memory_string()}"
            )
            last_report = now

    backend.synchronize()
    elapsed = time.perf_counter() - start
    print(
        f"Completed {step} steps to t={current_time:.8f} in {elapsed:.2f} s "
        f"({1000.0 * elapsed / max(step, 1):.3f} ms/step)."
    )

    return make_plots_and_save(
        backend, grid, equation, operator, q, config, step, elapsed
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> SolverConfig:
    parser = argparse.ArgumentParser(
        description=(
            "GPU hybrid compact/WENO-7 solution of Kurganov-Tadmor 2-D "
            "Riemann problem configuration 3."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nx", type=int, default=400, help="physical x resolution")
    parser.add_argument("--ny", type=int, default=400, help="physical y resolution")
    parser.add_argument("--tfinal", type=float, default=0.3)
    parser.add_argument("--cfl", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument(
        "--padding",
        type=float,
        default=0.75,
        help="padding added on each side of the physical unit square",
    )
    parser.add_argument("--mn", type=float, default=0.001)
    parser.add_argument("--hyperviscosity-interval", type=int, default=5)
    parser.add_argument("--sensor-width", type=int, default=6)
    parser.add_argument("--compression-threshold", type=float, default=3.0)
    parser.add_argument("--jump-threshold", type=float, default=0.01)
    parser.add_argument(
        "--shear-threshold",
        type=float,
        default=None,
        help="optional |omega|/omega_rms sensor threshold; disabled by default",
    )
    parser.add_argument("--rho-floor", type=float, default=1.0e-10)
    parser.add_argument("--pressure-floor", type=float, default=1.0e-10)
    parser.add_argument("--weno-epsilon", type=float, default=1.0e-10)
    parser.add_argument(
        "--weno-mode",
        choices=("characteristic", "component"),
        default="characteristic",
        help="characteristic is recommended; component is a faster diagnostic",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--backend", choices=("cupy", "numpy"), default="cupy")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--strict-positivity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="raise instead of silently applying emergency density/pressure floors",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--prefix", default="Riemann2D_Config3_hybrid")
    parser.add_argument("--density-contours", type=int, default=40)
    parser.add_argument("--vorticity-percentile", type=float, default=99.5)
    parser.add_argument("--vorticity-limit", type=float, default=None)
    parser.add_argument(
        "--save-fields",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.tfinal <= 0.0:
        parser.error("--tfinal must be positive")
    if not (0.0 < args.cfl <= 1.0):
        parser.error("--cfl must lie in (0, 1]")
    if args.mn < 0.0:
        parser.error("--mn cannot be negative")
    if args.hyperviscosity_interval < 0:
        parser.error("--hyperviscosity-interval cannot be negative")
    if args.sensor_width < 0:
        parser.error("--sensor-width cannot be negative")
    if not (0.0 < args.vorticity_percentile <= 100.0):
        parser.error("--vorticity-percentile must lie in (0, 100]")

    return SolverConfig(**vars(args))


def main() -> None:
    config = parse_args()
    paths = run(config)
    print("Generated outputs:")
    for label, path in paths.items():
        print(f"  {label:10s}: {path}")


if __name__ == "__main__":
    main()
