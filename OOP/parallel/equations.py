from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from OOP.parallel.backend import array_module


@dataclass(frozen=True)
class ParallelEulerEquation2D:
    """2D ideal-gas Euler equation using NumPy-like array backends."""

    gamma: float = 1.4
    rho_floor: float = 1e-10
    pressure_floor: float = 1e-10

    def primitive_from_conservative(self, q):
        xp = array_module(q)
        rho = xp.maximum(q[0], self.rho_floor)
        u = q[1] / rho
        v = q[2] / rho
        kinetic = 0.5 * rho * (u**2 + v**2)
        pressure = (self.gamma - 1.0) * (q[3] - kinetic)
        return rho, u, v, xp.maximum(pressure, self.pressure_floor)

    def conservative_from_primitive(self, rho, u, v, pressure):
        xp = array_module(rho)
        rho, u, v, pressure = xp.broadcast_arrays(rho, u, v, pressure)
        rho = xp.maximum(rho, self.rho_floor)
        pressure = xp.maximum(pressure, self.pressure_floor)

        q = xp.zeros((4, *rho.shape), dtype=xp.result_type(rho, u, v, pressure))
        q[0] = rho
        q[1] = rho * u
        q[2] = rho * v
        q[3] = pressure / (self.gamma - 1.0) + 0.5 * rho * (u**2 + v**2)
        return q

    def enforce_physical_state(self, q):
        xp = array_module(q)
        q_fixed = xp.array(q, copy=True)
        rho = xp.maximum(q_fixed[0], self.rho_floor)
        u = q_fixed[1] / rho
        v = q_fixed[2] / rho
        kinetic = 0.5 * rho * (u**2 + v**2)
        pressure = (self.gamma - 1.0) * (q_fixed[3] - kinetic)

        q_fixed[0] = rho
        q_fixed[1] = rho * u
        q_fixed[2] = rho * v
        q_fixed[3] = xp.where(
            pressure > self.pressure_floor,
            q_fixed[3],
            self.pressure_floor / (self.gamma - 1.0) + kinetic,
        )
        return q_fixed

    def flux_x(self, q):
        rho, u, v, pressure = self.primitive_from_conservative(q)
        xp = array_module(q)
        flux = xp.zeros_like(q)
        flux[0] = q[1]
        flux[1] = rho * u**2 + pressure
        flux[2] = rho * u * v
        flux[3] = (q[3] + pressure) * u
        return flux

    def flux_y(self, q):
        rho, u, v, pressure = self.primitive_from_conservative(q)
        xp = array_module(q)
        flux = xp.zeros_like(q)
        flux[0] = q[2]
        flux[1] = rho * u * v
        flux[2] = rho * v**2 + pressure
        flux[3] = (q[3] + pressure) * v
        return flux

    def sound_speed(self, q):
        rho, _u, _v, pressure = self.primitive_from_conservative(q)
        xp = array_module(q)
        return xp.sqrt(self.gamma * pressure / rho)

    def max_wave_speed(self, q) -> float:
        rho, u, v, pressure = self.primitive_from_conservative(q)
        xp = array_module(q)
        c = xp.sqrt(self.gamma * pressure / rho)
        return float(xp.max(xp.maximum(xp.abs(u) + c, xp.abs(v) + c)))


@dataclass(frozen=True)
class ParallelCompressibleNavierStokes2D(ParallelEulerEquation2D):
    """2D ideal-gas Navier-Stokes helper using NumPy-like array backends."""

    viscosity: float = 0.0
    prandtl: float = 0.72
    gas_constant: float = 1.0

    def temperature(self, rho, pressure):
        return pressure / (rho * self.gas_constant)

    @staticmethod
    def _periodic_derivative(values, spacing: float, axis: int):
        xp = array_module(values)
        return (xp.roll(values, -1, axis=axis) - xp.roll(values, 1, axis=axis)) / (2.0 * spacing)

    def viscous_rhs(self, q, dx: float, dy: float):
        xp = array_module(q)
        if self.viscosity == 0.0:
            return xp.zeros_like(q)

        rho, u, v, pressure = self.primitive_from_conservative(q)
        temperature = self.temperature(rho, pressure)
        mu = self.viscosity
        cp = self.gamma * self.gas_constant / (self.gamma - 1.0)
        conductivity = mu * cp / self.prandtl

        du_dx = self._periodic_derivative(u, dx, axis=-1)
        du_dy = self._periodic_derivative(u, dy, axis=-2)
        dv_dx = self._periodic_derivative(v, dx, axis=-1)
        dv_dy = self._periodic_derivative(v, dy, axis=-2)
        div_u = du_dx + dv_dy

        tau_xx = 2.0 * mu * du_dx - (2.0 / 3.0) * mu * div_u
        tau_yy = 2.0 * mu * dv_dy - (2.0 / 3.0) * mu * div_u
        tau_xy = mu * (du_dy + dv_dx)

        dT_dx = self._periodic_derivative(temperature, dx, axis=-1)
        dT_dy = self._periodic_derivative(temperature, dy, axis=-2)

        rhs = xp.zeros_like(q)
        rhs[1] = self._periodic_derivative(tau_xx, dx, axis=-1)
        rhs[1] += self._periodic_derivative(tau_xy, dy, axis=-2)
        rhs[2] = self._periodic_derivative(tau_xy, dx, axis=-1)
        rhs[2] += self._periodic_derivative(tau_yy, dy, axis=-2)

        energy_flux_x = u * tau_xx + v * tau_xy + conductivity * dT_dx
        energy_flux_y = u * tau_xy + v * tau_yy + conductivity * dT_dy
        rhs[3] = self._periodic_derivative(energy_flux_x, dx, axis=-1)
        rhs[3] += self._periodic_derivative(energy_flux_y, dy, axis=-2)
        return rhs
