from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from OOP.domain import Domain2D
from OOP.parallel.backend import array_module
from OOP.spatial_operator import weno7_flux


def relative_jump_axis_periodic(values, axis: int):
    xp = array_module(values)
    eps = 1e-14 + 1e-12 * xp.max(xp.abs(values))
    jump_r = xp.abs(xp.roll(values, -1, axis=axis) - values)
    jump_r /= xp.abs(xp.roll(values, -1, axis=axis)) + xp.abs(values) + eps

    jump_l = xp.abs(values - xp.roll(values, 1, axis=axis))
    jump_l /= xp.abs(values) + xp.abs(xp.roll(values, 1, axis=axis)) + eps

    curvature = xp.abs(
        xp.roll(values, -1, axis=axis)
        - 2.0 * values
        + xp.roll(values, 1, axis=axis)
    )
    curvature /= (
        xp.abs(xp.roll(values, -1, axis=axis))
        + 2.0 * xp.abs(values)
        + xp.abs(xp.roll(values, 1, axis=axis))
        + eps
    )
    return xp.maximum(xp.maximum(jump_l, jump_r), curvature)


def dilate_periodic_mask_2d(mask, width: int):
    xp = array_module(mask)
    expanded = xp.array(mask, copy=True)
    for y_offset in range(-width, width + 1):
        for x_offset in range(-width, width + 1):
            if x_offset == 0 and y_offset == 0:
                continue
            expanded |= xp.roll(xp.roll(mask, y_offset, axis=-2), x_offset, axis=-1)
    return expanded


def dilate_periodic_mask_axis(mask, width: int, axis: int):
    xp = array_module(mask)
    expanded = xp.array(mask, copy=True)
    for offset in range(1, width + 1):
        expanded |= xp.roll(mask, offset, axis=axis)
        expanded |= xp.roll(mask, -offset, axis=axis)
    return expanded


def smooth_compact_flux_axis(point_flux):
    xp = array_module(point_flux)
    a1_c = 25 / 32
    b1_c = 1 / 20
    c1_c = -1 / 480
    return (
        c1_c * (xp.roll(point_flux, -3, axis=-1) + xp.roll(point_flux, 2, axis=-1))
        + (b1_c + c1_c)
        * (xp.roll(point_flux, -2, axis=-1) + xp.roll(point_flux, 1, axis=-1))
        + (a1_c + b1_c + c1_c)
        * (xp.roll(point_flux, -1, axis=-1) + point_flux)
    )


def weno7_flux_axis(q, point_flux, alpha: float):
    xp = array_module(q)
    f_plus = 0.5 * (point_flux + alpha * q)
    f_minus = 0.5 * (point_flux - alpha * q)
    return weno7_flux(
        xp.roll(f_plus, 3, axis=-1),
        xp.roll(f_plus, 2, axis=-1),
        xp.roll(f_plus, 1, axis=-1),
        f_plus,
        xp.roll(f_plus, -1, axis=-1),
        xp.roll(f_plus, -2, axis=-1),
        xp.roll(f_plus, -3, axis=-1),
    ) + weno7_flux(
        xp.roll(f_minus, -4, axis=-1),
        xp.roll(f_minus, -3, axis=-1),
        xp.roll(f_minus, -2, axis=-1),
        xp.roll(f_minus, -1, axis=-1),
        f_minus,
        xp.roll(f_minus, 1, axis=-1),
        xp.roll(f_minus, 2, axis=-1),
    )


def weno7_flux_axis_local_lf(q, point_flux, wave_speed):
    """Backend-aware WENO7 with one local LF alpha per interface stencil."""

    xp = array_module(q)
    alpha = xp.array(wave_speed, copy=True)
    for shift in (3, 2, 1, 0, -1, -2, -3, -4):
        alpha = xp.maximum(alpha, xp.roll(wave_speed, shift, axis=-1))

    def split(shift: int, sign: float):
        q_s = xp.roll(q, shift, axis=-1)
        f_s = xp.roll(point_flux, shift, axis=-1)
        return 0.5 * (f_s + sign * alpha[None, ...] * q_s)

    plus = [split(shift, +1.0) for shift in (3, 2, 1, 0, -1, -2, -3)]
    minus = [split(shift, -1.0) for shift in (-4, -3, -2, -1, 0, 1, 2)]
    return weno7_flux(*plus) + weno7_flux(*minus)


@dataclass
class ParallelPeriodicLineCompactDerivative:
    """Periodic compact derivative solve using an FFT/circulant solve."""

    n: int
    dx: float
    alpha: float = 3.0 / 8.0

    def __post_init__(self) -> None:
        theta = 2.0 * np.pi * np.fft.fftfreq(self.n)
        self._eigenvalues_np = 1.0 + 2.0 * self.alpha * np.cos(theta)
        self._eigenvalues_cache = {}

    def _eigenvalues(self, values):
        xp = array_module(values)
        if xp is np:
            return self._eigenvalues_np

        key = xp.__name__
        if key not in self._eigenvalues_cache:
            self._eigenvalues_cache[key] = xp.asarray(self._eigenvalues_np)
        return self._eigenvalues_cache[key]

    def from_interface_flux(self, interface_flux):
        xp = array_module(interface_flux)
        raw = (interface_flux - xp.roll(interface_flux, 1, axis=-1)) / self.dx
        derivative_hat = xp.fft.fft(raw, axis=-1) / self._eigenvalues(raw)
        return xp.fft.ifft(derivative_hat, axis=-1).real


@dataclass(frozen=True)
class ParallelPeriodicEulerShockSensor2D:
    domain: Domain2D
    equation: object
    width: int = 4
    compression_threshold: float = 2.5
    jump_threshold: float = 0.04
    shear_threshold: float | None = None
    mode: str = "legacy"
    ducros_threshold: float = 0.5

    def _raw_fields(self, q):
        xp = array_module(q)
        rho, u, v, pressure = self.equation.primitive_from_conservative(q)
        du_dx = (xp.roll(u, -1, axis=-1) - xp.roll(u, 1, axis=-1)) / (2.0 * self.domain.dx)
        du_dy = (xp.roll(u, -1, axis=-2) - xp.roll(u, 1, axis=-2)) / (2.0 * self.domain.dy)
        dv_dx = (xp.roll(v, -1, axis=-1) - xp.roll(v, 1, axis=-1)) / (2.0 * self.domain.dx)
        dv_dy = (xp.roll(v, -1, axis=-2) - xp.roll(v, 1, axis=-2)) / (2.0 * self.domain.dy)
        divergence = du_dx + dv_dy
        vorticity = dv_dx - du_dy
        internal_energy = pressure / (rho * (self.equation.gamma - 1.0))
        jumps_x = xp.maximum(
            xp.maximum(relative_jump_axis_periodic(rho, -1), relative_jump_axis_periodic(pressure, -1)),
            relative_jump_axis_periodic(internal_energy, -1),
        )
        jumps_y = xp.maximum(
            xp.maximum(relative_jump_axis_periodic(rho, -2), relative_jump_axis_periodic(pressure, -2)),
            relative_jump_axis_periodic(internal_energy, -2),
        )
        scale = float(xp.mean(divergence**2 + vorticity**2))
        eps = max(1e-30, 1e-12 * scale)
        ducros = divergence**2 / (divergence**2 + vorticity**2 + eps)
        return du_dx, dv_dy, divergence, vorticity, jumps_x, jumps_y, ducros

    def detect_directional(self, q):
        xp = array_module(q)
        if self.mode not in {"legacy", "compression_gated", "directional"}:
            raise ValueError(
                "sensor mode must be 'legacy', 'compression_gated', or 'directional'"
            )
        du_dx, dv_dy, divergence, vorticity, jumps_x, jumps_y, ducros = self._raw_fields(q)

        if self.mode == "legacy":
            div_rms = float(xp.sqrt(xp.mean(divergence**2)))
            compression = xp.zeros_like(divergence, dtype=bool)
            if div_rms > 1e-12:
                compression = divergence < -self.compression_threshold * div_rms
            shear = xp.zeros_like(divergence, dtype=bool)
            if self.shear_threshold is not None:
                vort_rms = float(xp.sqrt(xp.mean(vorticity**2)))
                if vort_rms > 1e-12:
                    shear = xp.abs(vorticity) > self.shear_threshold * vort_rms
            raw = compression | shear | (jumps_x > self.jump_threshold) | (jumps_y > self.jump_threshold)
            mask = dilate_periodic_mask_2d(raw, self.width)
            return mask, mask

        compression_dominated = ducros >= self.ducros_threshold
        thermo_x = jumps_x > self.jump_threshold
        thermo_y = jumps_y > self.jump_threshold

        if self.mode == "compression_gated":
            div_rms = float(xp.sqrt(xp.mean(divergence**2)))
            strong_compression = xp.zeros_like(divergence, dtype=bool)
            if div_rms > 1e-12:
                strong_compression = divergence < -self.compression_threshold * div_rms
            raw = strong_compression & compression_dominated & (thermo_x | thermo_y)
            mask = dilate_periodic_mask_2d(raw, self.width)
            return mask, mask

        du_rms = float(xp.sqrt(xp.mean(du_dx**2)))
        dv_rms = float(xp.sqrt(xp.mean(dv_dy**2)))
        comp_x = xp.zeros_like(divergence, dtype=bool)
        comp_y = xp.zeros_like(divergence, dtype=bool)
        if du_rms > 1e-12:
            comp_x = du_dx < -self.compression_threshold * du_rms
        if dv_rms > 1e-12:
            comp_y = dv_dy < -self.compression_threshold * dv_rms
        common = (divergence < 0.0) & compression_dominated
        mask_x = dilate_periodic_mask_axis(common & comp_x & thermo_x, self.width, axis=-1)
        mask_y = dilate_periodic_mask_axis(common & comp_y & thermo_y, self.width, axis=-2)
        return mask_x, mask_y

    def detect(self, q):
        mask_x, mask_y = self.detect_directional(q)
        return mask_x | mask_y


@dataclass
class ParallelPeriodicHybridEuler2DOperator:
    domain: Domain2D
    equation: object
    sensor: ParallelPeriodicEulerShockSensor2D
    flux_splitting: str = "global"
    compact_x: ParallelPeriodicLineCompactDerivative = field(init=False)
    compact_y: ParallelPeriodicLineCompactDerivative = field(init=False)

    def __post_init__(self) -> None:
        if self.flux_splitting not in {"global", "local"}:
            raise ValueError("flux_splitting must be 'global' or 'local'")
        self.compact_x = ParallelPeriodicLineCompactDerivative(self.domain.nx, self.domain.dx)
        self.compact_y = ParallelPeriodicLineCompactDerivative(self.domain.ny, self.domain.dy)

    def _axis_derivative(self, q_axis, flux_axis, shock_axis, compact, normal_velocity_index: int):
        rho, u, v, pressure = self.equation.primitive_from_conservative(q_axis)
        normal_velocity = u if normal_velocity_index == 1 else v
        xp = array_module(q_axis)
        sound_speed = xp.sqrt(self.equation.gamma * pressure / rho)
        wave_speed = xp.abs(normal_velocity) + sound_speed

        smooth_flux = smooth_compact_flux_axis(flux_axis)
        if self.flux_splitting == "global":
            weno_raw = weno7_flux_axis(q_axis, flux_axis, float(xp.max(wave_speed)))
        else:
            weno_raw = weno7_flux_axis_local_lf(q_axis, flux_axis, wave_speed)
        weno_flux = compact.alpha * xp.roll(weno_raw, -1, axis=-1)
        weno_flux += weno_raw
        weno_flux += compact.alpha * xp.roll(weno_raw, 1, axis=-1)

        shock_interfaces = shock_axis | xp.roll(shock_axis, -1, axis=-1)
        hybrid_flux = xp.where(shock_interfaces[None, ...], weno_flux, smooth_flux)
        return compact.from_interface_flux(hybrid_flux)

    def rhs(self, q):
        q_safe = self.equation.enforce_physical_state(q)
        shock_x, shock_y = self.sensor.detect_directional(q_safe)

        derivative_x = self._axis_derivative(
            q_safe, self.equation.flux_x(q_safe), shock_x, self.compact_x, normal_velocity_index=1
        )

        xp = array_module(q_safe)
        q_y = xp.moveaxis(q_safe, -2, -1)
        flux_y = xp.moveaxis(self.equation.flux_y(q_safe), -2, -1)
        shock_y_axis = xp.moveaxis(shock_y, -2, -1)
        derivative_y = self._axis_derivative(
            q_y, flux_y, shock_y_axis, self.compact_y, normal_velocity_index=2
        )
        derivative_y = xp.moveaxis(derivative_y, -1, -2)
        return -derivative_x - derivative_y

    def stable_time_step(self, q, cfl: float, remaining_time: float | None = None) -> float:
        rho, u, v, pressure = self.equation.primitive_from_conservative(q)
        xp = array_module(q)
        sound_speed = xp.sqrt(self.equation.gamma * pressure / rho)
        spectral_radius = (xp.abs(u) + sound_speed) / self.domain.dx
        spectral_radius += (xp.abs(v) + sound_speed) / self.domain.dy
        dt = cfl / float(xp.max(spectral_radius))
        if remaining_time is not None:
            dt = min(dt, remaining_time)
        return dt


@dataclass(frozen=True)
class ParallelPeriodicHyperviscosity2D:
    """Backend-aware compact-region hyperviscosity.

    ``conservative_flux`` uses a periodic face-flux divergence and therefore
    conserves each filtered conservative variable to roundoff when no physical
    clipping is required. ``legacy_node`` reproduces the previous pointwise
    masked biharmonic filter. The operation remains a per-application filter
    and is not multiplied by ``dt``.
    """
    domain: Domain2D
    mn: float = 0.002
    density_weight: float = 0.25
    momentum_weight: float = 1.0
    energy_weight: float = 0.25
    mode: str = "conservative_flux"

    def _laplacian(self, values):
        xp = array_module(values)
        lap_x = (
            xp.roll(values, -1, axis=-1)
            - 2.0 * values
            + xp.roll(values, 1, axis=-1)
        ) / self.domain.dx**2
        lap_y = (
            xp.roll(values, -1, axis=-2)
            - 2.0 * values
            + xp.roll(values, 1, axis=-2)
        ) / self.domain.dy**2
        return lap_x + lap_y

    def _conservative_biharmonic(self, values, active_mask):
        xp = array_module(values)
        lap = self._laplacian(values)
        face_x = active_mask & xp.roll(active_mask, -1, axis=-1)
        face_y = active_mask & xp.roll(active_mask, -1, axis=-2)
        grad_lap_x = (xp.roll(lap, -1, axis=-1) - lap) / self.domain.dx
        grad_lap_y = (xp.roll(lap, -1, axis=-2) - lap) / self.domain.dy
        flux_x = face_x[None, :, :] * grad_lap_x
        flux_y = face_y[None, :, :] * grad_lap_y
        div_x = (flux_x - xp.roll(flux_x, 1, axis=-1)) / self.domain.dx
        div_y = (flux_y - xp.roll(flux_y, 1, axis=-2)) / self.domain.dy
        return div_x + div_y

    def apply(self, q, equation, active_mask=None):
        xp = array_module(q)
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
        weights = xp.array(
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
