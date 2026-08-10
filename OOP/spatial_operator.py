from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from OOP.domain import Domain1D, Domain2D
from OOP.equations import BurgersEquation, EulerEquation, EulerEquation2D


def relative_jump_sensor(values: np.ndarray) -> np.ndarray:
    eps = 1e-14 + 1e-12 * np.max(np.abs(values))
    jump_r = np.abs(np.roll(values, -1) - values)
    jump_r /= np.abs(np.roll(values, -1)) + np.abs(values) + eps

    jump_l = np.abs(values - np.roll(values, 1))
    jump_l /= np.abs(values) + np.abs(np.roll(values, 1)) + eps

    curvature = np.abs(np.roll(values, -1) - 2.0 * values + np.roll(values, 1))
    curvature /= (
        np.abs(np.roll(values, -1))
        + 2.0 * np.abs(values)
        + np.abs(np.roll(values, 1))
        + eps
    )
    return np.maximum.reduce([jump_l, jump_r, curvature])


def dilate_periodic_mask(mask: np.ndarray, width: int) -> np.ndarray:
    expanded = np.array(mask, copy=True)
    for offset in range(1, width + 1):
        expanded |= np.roll(mask, offset)
        expanded |= np.roll(mask, -offset)
    return expanded


def interface_masks(node_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weno_edge = node_mask | np.roll(node_mask, -1)
    smooth_edge = ~weno_edge
    return weno_edge, smooth_edge


@dataclass(frozen=True)
class EulerShockSensor:
    equation: EulerEquation
    dx: float
    width: int = 4
    compression_threshold: float = 2.5
    jump_threshold: float = 0.04
    boundary_guard: int = 0

    def detect(self, q: np.ndarray) -> np.ndarray:
        rho, u, pressure = self.equation.primitive_from_conservative(q)
        theta = (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * self.dx)
        theta_rms = np.sqrt(np.mean(theta**2))

        compression = np.zeros_like(theta, dtype=bool)
        if theta_rms > 1e-12:
            compression = theta < -self.compression_threshold * theta_rms

        density_jump = relative_jump_sensor(rho) > self.jump_threshold
        pressure_jump = relative_jump_sensor(pressure) > self.jump_threshold
        energy_jump = relative_jump_sensor(self.equation.internal_energy(rho, pressure))
        energy_jump = energy_jump > self.jump_threshold

        mask = compression | density_jump | pressure_jump | energy_jump
        if self.boundary_guard:
            mask[: self.boundary_guard] = False
            mask[-self.boundary_guard :] = False
        return dilate_periodic_mask(mask, self.width)


def periodic_diags(values: list[float], offsets: list[int], size: int) -> sp.csc_matrix:
    matrix = sp.diags(values, offsets, shape=(size, size)).tolil()
    for value, offset in zip(values, offsets):
        if offset > 0:
            for row in range(offset):
                matrix[size - offset + row, row] = value
        elif offset < 0:
            for row in range(-offset):
                matrix[row, size + offset + row] = value
    return matrix.tocsc()


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


def smooth_compact_flux(point_flux: np.ndarray) -> np.ndarray:
    a1_c = 25 / 32
    b1_c = 1 / 20
    c1_c = -1 / 480
    return (
        c1_c * (np.roll(point_flux, -3) + np.roll(point_flux, 2))
        + (b1_c + c1_c) * (np.roll(point_flux, -2) + np.roll(point_flux, 1))
        + (a1_c + b1_c + c1_c) * (np.roll(point_flux, -1) + point_flux)
    )


def second_derivative_6th(values: np.ndarray, dx: float) -> np.ndarray:
    return (
        (1 / 90) * (np.roll(values, -3) + np.roll(values, 3))
        - (3 / 20) * (np.roll(values, -2) + np.roll(values, 2))
        + (3 / 2) * (np.roll(values, -1) + np.roll(values, 1))
        - (49 / 18) * values
    ) / dx**2


def relative_jump_axis_periodic(values: np.ndarray, axis: int) -> np.ndarray:
    eps = 1e-14 + 1e-12 * np.max(np.abs(values))
    jump_r = np.abs(np.roll(values, -1, axis=axis) - values)
    jump_r /= np.abs(np.roll(values, -1, axis=axis)) + np.abs(values) + eps

    jump_l = np.abs(values - np.roll(values, 1, axis=axis))
    jump_l /= np.abs(values) + np.abs(np.roll(values, 1, axis=axis)) + eps

    curvature = np.abs(
        np.roll(values, -1, axis=axis)
        - 2.0 * values
        + np.roll(values, 1, axis=axis)
    )
    curvature /= (
        np.abs(np.roll(values, -1, axis=axis))
        + 2.0 * np.abs(values)
        + np.abs(np.roll(values, 1, axis=axis))
        + eps
    )
    return np.maximum(np.maximum(jump_l, jump_r), curvature)


def dilate_periodic_mask_2d(mask: np.ndarray, width: int) -> np.ndarray:
    expanded = np.array(mask, copy=True)
    for y_offset in range(-width, width + 1):
        for x_offset in range(-width, width + 1):
            if x_offset == 0 and y_offset == 0:
                continue
            expanded |= np.roll(np.roll(mask, y_offset, axis=-2), x_offset, axis=-1)
    return expanded


def smooth_compact_flux_axis(point_flux: np.ndarray) -> np.ndarray:
    a1_c = 25 / 32
    b1_c = 1 / 20
    c1_c = -1 / 480
    return (
        c1_c * (np.roll(point_flux, -3, axis=-1) + np.roll(point_flux, 2, axis=-1))
        + (b1_c + c1_c)
        * (np.roll(point_flux, -2, axis=-1) + np.roll(point_flux, 1, axis=-1))
        + (a1_c + b1_c + c1_c)
        * (np.roll(point_flux, -1, axis=-1) + point_flux)
    )


def weno7_flux_axis(q: np.ndarray, point_flux: np.ndarray, alpha: float) -> np.ndarray:
    f_plus = 0.5 * (point_flux + alpha * q)
    f_minus = 0.5 * (point_flux - alpha * q)
    return weno7_flux(
        np.roll(f_plus, 3, axis=-1),
        np.roll(f_plus, 2, axis=-1),
        np.roll(f_plus, 1, axis=-1),
        f_plus,
        np.roll(f_plus, -1, axis=-1),
        np.roll(f_plus, -2, axis=-1),
        np.roll(f_plus, -3, axis=-1),
    ) + weno7_flux(
        np.roll(f_minus, -4, axis=-1),
        np.roll(f_minus, -3, axis=-1),
        np.roll(f_minus, -2, axis=-1),
        np.roll(f_minus, -1, axis=-1),
        f_minus,
        np.roll(f_minus, 1, axis=-1),
        np.roll(f_minus, 2, axis=-1),
    )


@dataclass
class CompactDerivative:
    domain: Domain1D
    alpha: float = 3.0 / 8.0

    def __post_init__(self) -> None:
        matrix = periodic_diags([self.alpha, 1.0, self.alpha], [-1, 0, 1], self.domain.nx)
        self.solve_matrix = spla.factorized(matrix)

    def from_interface_flux(self, interface_flux: np.ndarray) -> np.ndarray:
        return self.solve_matrix((interface_flux - np.roll(interface_flux, 1)) / self.domain.dx)


@dataclass
class PeriodicLineCompactDerivative:
    """Periodic compact derivative solve applied along the last axis.

    The compact matrix is circulant for periodic boundaries:
    alpha*d_{i-1} + d_i + alpha*d_{i+1} = rhs_i.
    A circulant matrix is diagonal in Fourier space, so the solve can be done
    for all rows/components at once with FFTs. This avoids Python line loops and
    keeps the operation compatible with NumPy or CuPy arrays.
    """

    n: int
    dx: float
    alpha: float = 3.0 / 8.0

    def __post_init__(self) -> None:
        theta = 2.0 * np.pi * np.fft.fftfreq(self.n)
        self._eigenvalues_np = 1.0 + 2.0 * self.alpha * np.cos(theta)

    def from_interface_flux(self, interface_flux: np.ndarray) -> np.ndarray:
        raw = (interface_flux - np.roll(interface_flux, 1, axis=-1)) / self.dx
        derivative_hat = np.fft.fft(raw, axis=-1) / self._eigenvalues_np
        return np.fft.ifft(derivative_hat, axis=-1).real


@dataclass(frozen=True)
class PeriodicEulerShockSensor2D:
    """Periodic 2D Euler shock sensor for hybrid compact/WENO switching."""

    domain: Domain2D
    equation: EulerEquation2D
    width: int = 4
    compression_threshold: float = 2.5
    jump_threshold: float = 0.04
    shear_threshold: float | None = None

    def detect(self, q: np.ndarray) -> np.ndarray:
        rho, u, v, pressure = self.equation.primitive_from_conservative(q)
        du_dx = (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1)) / (2.0 * self.domain.dx)
        du_dy = (np.roll(u, -1, axis=-2) - np.roll(u, 1, axis=-2)) / (2.0 * self.domain.dy)
        dv_dx = (np.roll(v, -1, axis=-1) - np.roll(v, 1, axis=-1)) / (2.0 * self.domain.dx)
        dv_dy = (np.roll(v, -1, axis=-2) - np.roll(v, 1, axis=-2)) / (2.0 * self.domain.dy)

        divergence = du_dx + dv_dy
        div_rms = float(np.sqrt(np.mean(divergence**2)))
        compression = np.zeros_like(divergence, dtype=bool)
        if div_rms > 1e-12:
            compression = divergence < -self.compression_threshold * div_rms

        shear = np.zeros_like(divergence, dtype=bool)
        if self.shear_threshold is not None:
            vorticity = dv_dx - du_dy
            vort_rms = float(np.sqrt(np.mean(vorticity**2)))
            if vort_rms > 1e-12:
                shear = np.abs(vorticity) > self.shear_threshold * vort_rms

        internal_energy = pressure / (rho * (self.equation.gamma - 1.0))
        density_jump = np.maximum(
            relative_jump_axis_periodic(rho, -1),
            relative_jump_axis_periodic(rho, -2),
        )
        pressure_jump = np.maximum(
            relative_jump_axis_periodic(pressure, -1),
            relative_jump_axis_periodic(pressure, -2),
        )
        energy_jump = np.maximum(
            relative_jump_axis_periodic(internal_energy, -1),
            relative_jump_axis_periodic(internal_energy, -2),
        )

        mask = (
            compression
            | shear
            | (density_jump > self.jump_threshold)
            | (pressure_jump > self.jump_threshold)
            | (energy_jump > self.jump_threshold)
        )
        return dilate_periodic_mask_2d(mask, self.width)


@dataclass
class PeriodicHybridEuler2DOperator:
    """Periodic 2D hybrid compact/WENO Euler RHS with the repo state layout."""

    domain: Domain2D
    equation: EulerEquation2D
    sensor: PeriodicEulerShockSensor2D
    compact_x: PeriodicLineCompactDerivative = field(init=False)
    compact_y: PeriodicLineCompactDerivative = field(init=False)

    def __post_init__(self) -> None:
        self.compact_x = PeriodicLineCompactDerivative(self.domain.nx, self.domain.dx)
        self.compact_y = PeriodicLineCompactDerivative(self.domain.ny, self.domain.dy)

    def _axis_derivative(
        self,
        q_axis: np.ndarray,
        flux_axis: np.ndarray,
        shock_axis: np.ndarray,
        compact: PeriodicLineCompactDerivative,
        normal_velocity_index: int,
    ) -> np.ndarray:
        rho, u, v, pressure = self.equation.primitive_from_conservative(q_axis)
        normal_velocity = u if normal_velocity_index == 1 else v
        sound_speed = np.sqrt(self.equation.gamma * pressure / rho)
        alpha = float(np.max(np.abs(normal_velocity) + sound_speed))

        smooth_flux = smooth_compact_flux_axis(flux_axis)
        weno_raw = weno7_flux_axis(q_axis, flux_axis, alpha)
        weno_flux = compact.alpha * np.roll(weno_raw, -1, axis=-1)
        weno_flux += weno_raw
        weno_flux += compact.alpha * np.roll(weno_raw, 1, axis=-1)

        shock_interfaces = shock_axis | np.roll(shock_axis, -1, axis=-1)
        hybrid_flux = np.where(shock_interfaces[None, ...], weno_flux, smooth_flux)
        return compact.from_interface_flux(hybrid_flux)

    def rhs(self, q: np.ndarray) -> np.ndarray:
        q_safe = self.equation.enforce_physical_state(q)
        shock_mask = self.sensor.detect(q_safe)

        derivative_x = self._axis_derivative(
            q_safe,
            self.equation.flux_x(q_safe),
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

    def stable_time_step(self, q: np.ndarray, cfl: float, remaining_time: float | None = None) -> float:
        rho, u, v, pressure = self.equation.primitive_from_conservative(q)
        sound_speed = np.sqrt(self.equation.gamma * pressure / rho)
        spectral_radius = (np.abs(u) + sound_speed) / self.domain.dx
        spectral_radius += (np.abs(v) + sound_speed) / self.domain.dy
        dt = cfl / float(np.max(spectral_radius))
        if remaining_time is not None:
            dt = min(dt, remaining_time)
        return dt


@dataclass(frozen=True)
class PeriodicHyperviscosity2D:
    """Periodic compact-region hyperviscosity.

    Two formulations are available:

    ``conservative_flux`` (default)
        Applies the biharmonic correction as the divergence of face fluxes.
        A face is active only when the two nodes adjacent to that face are
        compact-FD nodes. The periodic discrete flux divergence therefore
        telescopes exactly, so each conservative variable is preserved to
        roundoff provided the physical-state limiter does not activate.

    ``legacy_node``
        Retains the previous pointwise masked biharmonic correction for
        reproducibility of earlier runs. It is not globally conservative
        when the compact/WENO mask varies in space.

    The correction remains a discrete filter applied every configured number
    of complete RK steps; it is intentionally *not* multiplied by ``dt``.
    Consequently its effective strength per unit physical time depends on CFL.
    """

    domain: Domain2D
    mn: float = 0.002
    density_weight: float = 0.25
    momentum_weight: float = 1.0
    energy_weight: float = 0.25
    mode: str = "conservative_flux"

    def _laplacian(self, values: np.ndarray) -> np.ndarray:
        lap_x = (
            np.roll(values, -1, axis=-1)
            - 2.0 * values
            + np.roll(values, 1, axis=-1)
        ) / self.domain.dx**2
        lap_y = (
            np.roll(values, -1, axis=-2)
            - 2.0 * values
            + np.roll(values, 1, axis=-2)
        ) / self.domain.dy**2
        return lap_x + lap_y

    def _conservative_biharmonic(
        self,
        values: np.ndarray,
        active_mask: np.ndarray,
    ) -> np.ndarray:
        """Return a masked biharmonic written as a periodic flux divergence."""

        lap = self._laplacian(values)

        # Face i+1/2 is eligible only when both adjacent nodes are compact.
        face_x = active_mask & np.roll(active_mask, -1, axis=-1)
        face_y = active_mask & np.roll(active_mask, -1, axis=-2)

        grad_lap_x = (np.roll(lap, -1, axis=-1) - lap) / self.domain.dx
        grad_lap_y = (np.roll(lap, -1, axis=-2) - lap) / self.domain.dy
        flux_x = face_x[None, :, :] * grad_lap_x
        flux_y = face_y[None, :, :] * grad_lap_y

        div_x = (flux_x - np.roll(flux_x, 1, axis=-1)) / self.domain.dx
        div_y = (flux_y - np.roll(flux_y, 1, axis=-2)) / self.domain.dy
        return div_x + div_y

    def apply(
        self,
        q: np.ndarray,
        equation: EulerEquation2D,
        active_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if active_mask is None:
            raise ValueError(
                "active_mask is required: hyperviscosity may only be "
                "applied on compact-FD nodes"
            )
        if active_mask.shape != q.shape[-2:]:
            raise ValueError(
                f"active_mask shape {active_mask.shape} does not match grid {q.shape[-2:]}"
            )
        if self.mode not in {"conservative_flux", "legacy_node"}:
            raise ValueError(
                "hyperviscosity mode must be 'conservative_flux' or 'legacy_node'"
            )

        h = min(self.domain.dx, self.domain.dy)
        if self.mode == "conservative_flux":
            biharmonic = self._conservative_biharmonic(q, active_mask)
        else:
            active = active_mask[None, :, :]
            biharmonic = active * self._laplacian(self._laplacian(q))

        weights = np.array(
            [
                self.density_weight,
                self.momentum_weight,
                self.momentum_weight,
                self.energy_weight,
            ],
            dtype=q.dtype,
        )[:, None, None]
        filtered = q - self.mn * weights * h**4 * biharmonic
        return equation.enforce_physical_state(filtered)


@dataclass
class BurgersHybridSpatialOperator:
    domain: Domain1D
    equation: BurgersEquation
    sensor_width: int = 4
    compression_threshold: float = 3.0
    jump_threshold: float | None = None

    def __post_init__(self) -> None:
        self.compact = CompactDerivative(self.domain)

    def shock_mask(self, u: np.ndarray) -> np.ndarray:
        theta = (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * self.domain.dx)
        theta_rms = np.sqrt(np.mean(theta**2))

        compression = np.zeros_like(u, dtype=bool)
        if theta_rms > 1e-12:
            compression = theta < -self.compression_threshold * theta_rms

        if self.jump_threshold is not None:
            compression |= relative_jump_sensor(u) > self.jump_threshold

        return dilate_periodic_mask(compression, self.sensor_width)

    def weno_flux(self, u: np.ndarray) -> np.ndarray:
        f = self.equation.flux(u)
        alpha = self.equation.max_wave_speed(u)
        f_plus = 0.5 * (f + alpha * u)
        f_minus = 0.5 * (f - alpha * u)
        return weno7_flux(
            np.roll(f_plus, 3),
            np.roll(f_plus, 2),
            np.roll(f_plus, 1),
            f_plus,
            np.roll(f_plus, -1),
            np.roll(f_plus, -2),
            np.roll(f_plus, -3),
        ) + weno7_flux(
            np.roll(f_minus, -4),
            np.roll(f_minus, -3),
            np.roll(f_minus, -2),
            np.roll(f_minus, -1),
            f_minus,
            np.roll(f_minus, 1),
            np.roll(f_minus, 2),
        )

    def rhs(self, u: np.ndarray) -> np.ndarray:
        f = self.equation.flux(u)
        compact_flux = smooth_compact_flux(f)
        weno_raw = self.weno_flux(u)
        weno_flux_hat = self.compact.alpha * np.roll(weno_raw, -1) + weno_raw
        weno_flux_hat += self.compact.alpha * np.roll(weno_raw, 1)

        weno_edge, smooth_edge = interface_masks(self.shock_mask(u))
        hybrid_flux = np.empty_like(u)
        hybrid_flux[smooth_edge] = compact_flux[smooth_edge]
        hybrid_flux[weno_edge] = weno_flux_hat[weno_edge]

        advection = self.compact.from_interface_flux(hybrid_flux)
        diffusion = self.equation.viscosity * second_derivative_6th(u, self.domain.dx)
        return -advection + diffusion


@dataclass
class EulerHybridSpatialOperator:
    domain: Domain1D
    equation: EulerEquation
    sensor: EulerShockSensor

    def __post_init__(self) -> None:
        self.compact = CompactDerivative(self.domain)

    def characteristic_weno_flux(
        self,
        q: np.ndarray,
        flux: np.ndarray,
        alpha: float,
        required_edges: np.ndarray,
    ) -> np.ndarray:
        f_half = np.zeros_like(q)
        f_plus = 0.5 * (flux + alpha * q)
        f_minus = 0.5 * (flux - alpha * q)
        indices = np.flatnonzero(dilate_periodic_mask(required_edges, 1))

        for i in indices:
            ip1 = (i + 1) % self.domain.nx
            left_matrix, right_matrix = self.equation.roe_eigenvectors(q[:, i], q[:, ip1])
            plus_stencil = np.array(
                [left_matrix @ f_plus[:, (i + offset) % self.domain.nx] for offset in range(-3, 4)]
            )
            minus_stencil = np.array(
                [left_matrix @ f_minus[:, (i + offset) % self.domain.nx] for offset in range(4, -3, -1)]
            )

            char_flux = np.empty(3)
            for component in range(3):
                char_flux[component] = weno7_flux(*plus_stencil[:, component])
                char_flux[component] += weno7_flux(*minus_stencil[:, component])
            f_half[:, i] = right_matrix @ char_flux

        return f_half

    def rhs(self, q: np.ndarray) -> np.ndarray:
        q_safe = self.equation.enforce_physical_state(q)
        flux = self.equation.flux(q_safe)
        alpha = self.equation.max_wave_speed(q_safe)
        weno_edge, smooth_edge = interface_masks(self.sensor.detect(q_safe))
        weno_raw = self.characteristic_weno_flux(q_safe, flux, alpha, weno_edge)

        rhs = np.zeros_like(q_safe)
        for component in range(3):
            compact_flux = smooth_compact_flux(flux[component])
            weno_flux_hat = self.compact.alpha * np.roll(weno_raw[component], -1)
            weno_flux_hat += weno_raw[component]
            weno_flux_hat += self.compact.alpha * np.roll(weno_raw[component], 1)

            hybrid_flux = np.empty_like(flux[component])
            hybrid_flux[smooth_edge] = compact_flux[smooth_edge]
            hybrid_flux[weno_edge] = weno_flux_hat[weno_edge]
            rhs[component] = -self.compact.from_interface_flux(hybrid_flux)

        return rhs


@dataclass
class WangHyperviscosity:
    """Semi-implicit hyperviscosity used in the current hybrid 1D scripts."""

    domain: Domain1D
    dt: float
    mn: float = 0.02
    interval: int = 5

    def __post_init__(self) -> None:
        dx = self.domain.dx
        nx = self.domain.nx
        self.dt_hyp = self.interval * self.dt

        self.a2_t = 20 / 27
        self.b2_t = 25 / 216
        self.a3_h = 344 / 1179
        self.b3_h = (38 * self.a3_h - 9) / 214
        self.a3_t = (696 - 1191 * self.a3_h) / 428
        self.b3_t = (1227 * self.a3_h - 147) / 1070

        self.a_matrix = periodic_diags([1 / 36, 4 / 9, 1.0, 4 / 9, 1 / 36], [-2, -1, 0, 1, 2], nx)
        self.solve_a = spla.factorized(self.a_matrix)

        self.b_matrix = periodic_diags(
            [-self.b2_t / dx, -self.a2_t / dx, self.a2_t / dx, self.b2_t / dx],
            [-2, -1, 1, 2],
            nx,
        )

        self.c_matrix = periodic_diags(
            [self.b3_h, self.a3_h, 1.0, self.a3_h, self.b3_h],
            [-2, -1, 0, 1, 2],
            nx,
        )
        self.solve_c = spla.factorized(self.c_matrix)

        self.d_matrix = periodic_diags(
            [
                self.b3_t / dx**2,
                self.a3_t / dx**2,
                -2 * (self.a3_t + self.b3_t) / dx**2,
                self.a3_t / dx**2,
                self.b3_t / dx**2,
            ],
            [-2, -1, 0, 1, 2],
            nx,
        )
        self.solve_implicit = spla.factorized(self.c_matrix - self.dt_hyp * self.mn * self.d_matrix)

    def apply_scalar(self, values: np.ndarray, shock_mask: np.ndarray) -> np.ndarray:
        dx = self.domain.dx
        values_x = self.solve_a(self.b_matrix @ values)
        g_half = (
            self.b2_t * np.roll(values_x, 1)
            + (self.a2_t + self.b2_t) * values_x
            + (self.a2_t + self.b2_t) * np.roll(values_x, -1)
            + self.b2_t * np.roll(values_x, -2)
        ) / dx

        d_half = (
            -self.b3_t * np.roll(values, 1)
            - (self.a3_t + self.b3_t) * values
            + (self.a3_t + self.b3_t) * np.roll(values, -1)
            + self.b3_t * np.roll(values, -2)
        ) / dx**2
        h_half = self.a_matrix @ self.solve_c(d_half)

        shock_edge, smooth_edge = interface_masks(shock_mask)
        blended = np.empty_like(values)
        blended[smooth_edge] = g_half[smooth_edge]
        blended[shock_edge] = h_half[shock_edge]

        explicit = self.solve_a(blended - np.roll(blended, 1))
        rhs = values - self.dt_hyp * self.mn * explicit
        return self.solve_implicit(self.c_matrix @ rhs)

    def apply(self, state: np.ndarray, shock_mask: np.ndarray) -> np.ndarray:
        if state.ndim == 1:
            return self.apply_scalar(state, shock_mask)

        updated = np.empty_like(state)
        for component in range(state.shape[0]):
            updated[component] = self.apply_scalar(state[component], shock_mask)
        return updated
