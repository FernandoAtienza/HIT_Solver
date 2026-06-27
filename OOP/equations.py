from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BurgersEquation:
    """1D viscous Burgers equation: u_t + (u^2 / 2)_x = nu u_xx."""

    viscosity: float

    def flux(self, u: np.ndarray) -> np.ndarray:
        return 0.5 * u**2

    def max_wave_speed(self, u: np.ndarray) -> float:
        return float(np.max(np.abs(u)))


@dataclass(frozen=True)
class EulerEquation:
    """1D ideal-gas Euler equation in conservative variables."""

    gamma: float = 1.4
    gas_constant: float = 1.0
    rho_floor: float = 1e-10
    pressure_floor: float = 1e-10

    def primitive_from_conservative(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho = np.maximum(q[0], self.rho_floor)
        u = q[1] / rho
        pressure = (self.gamma - 1.0) * (q[2] - 0.5 * rho * u**2)
        return rho, u, np.maximum(pressure, self.pressure_floor)

    def conservative_from_primitive(
        self,
        rho: np.ndarray,
        u: np.ndarray,
        pressure: np.ndarray,
    ) -> np.ndarray:
        rho = np.maximum(rho, self.rho_floor)
        pressure = np.maximum(pressure, self.pressure_floor)
        q = np.zeros((3, rho.size), dtype=np.result_type(rho, u, pressure))
        q[0] = rho
        q[1] = rho * u
        q[2] = pressure / (self.gamma - 1.0) + 0.5 * rho * u**2
        return q

    def enforce_physical_state(self, q: np.ndarray) -> np.ndarray:
        q_fixed = np.array(q, copy=True)
        rho = np.maximum(q_fixed[0], self.rho_floor)
        u = q_fixed[1] / rho
        kinetic = 0.5 * rho * u**2
        pressure = (self.gamma - 1.0) * (q_fixed[2] - kinetic)

        q_fixed[0] = rho
        q_fixed[1] = rho * u
        q_fixed[2] = np.where(
            pressure > self.pressure_floor,
            q_fixed[2],
            self.pressure_floor / (self.gamma - 1.0) + kinetic,
        )
        return q_fixed

    def flux(self, q: np.ndarray) -> np.ndarray:
        rho, u, pressure = self.primitive_from_conservative(q)
        flux = np.zeros_like(q)
        flux[0] = q[1]
        flux[1] = rho * u**2 + pressure
        flux[2] = (q[2] + pressure) * u
        return flux

    def max_wave_speed(self, q: np.ndarray) -> float:
        rho, u, pressure = self.primitive_from_conservative(q)
        sound_speed = np.sqrt(self.gamma * pressure / rho)
        return float(np.max(np.abs(u) + sound_speed))

    def internal_energy(self, rho: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return pressure / (rho * (self.gamma - 1.0))

    def roe_eigenvectors(self, q_left: np.ndarray, q_right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rho_l, u_l, p_l = self.primitive_from_conservative(q_left[:, None])
        rho_r, u_r, p_r = self.primitive_from_conservative(q_right[:, None])
        rho_l, u_l, p_l = rho_l[0], u_l[0], p_l[0]
        rho_r, u_r, p_r = rho_r[0], u_r[0], p_r[0]

        h_l = (q_left[2] + p_l) / rho_l
        h_r = (q_right[2] + p_r) / rho_r
        sqrt_l = np.sqrt(rho_l)
        sqrt_r = np.sqrt(rho_r)
        u_roe = (sqrt_l * u_l + sqrt_r * u_r) / (sqrt_l + sqrt_r)
        h_roe = (sqrt_l * h_l + sqrt_r * h_r) / (sqrt_l + sqrt_r)

        kinetic = 0.5 * u_roe**2
        c2 = max((self.gamma - 1.0) * (h_roe - kinetic), self.pressure_floor)
        c = np.sqrt(c2)
        r_matrix = np.array(
            [
                [1.0, 1.0, 1.0],
                [u_roe - c, u_roe, u_roe + c],
                [h_roe - u_roe * c, kinetic, h_roe + u_roe * c],
            ]
        )

        gm1 = self.gamma - 1.0
        l_matrix = np.array(
            [
                [
                    (gm1 * kinetic + u_roe * c) / (2.0 * c2),
                    -(gm1 * u_roe + c) / (2.0 * c2),
                    gm1 / (2.0 * c2),
                ],
                [1.0 - gm1 * kinetic / c2, gm1 * u_roe / c2, -gm1 / c2],
                [
                    (gm1 * kinetic - u_roe * c) / (2.0 * c2),
                    -(gm1 * u_roe - c) / (2.0 * c2),
                    gm1 / (2.0 * c2),
                ],
            ]
        )
        return l_matrix, r_matrix


@dataclass(frozen=True)
class EulerEquation2D:
    """2D ideal-gas Euler equation in conservative variables.

    Conservative state ordering is [rho, rho*u, rho*v, E].
    """

    gamma: float = 1.4
    rho_floor: float = 1e-10
    pressure_floor: float = 1e-10

    def primitive_from_conservative(
        self,
        q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rho = np.maximum(q[0], self.rho_floor)
        u = q[1] / rho
        v = q[2] / rho
        kinetic = 0.5 * rho * (u**2 + v**2)
        pressure = (self.gamma - 1.0) * (q[3] - kinetic)
        return rho, u, v, np.maximum(pressure, self.pressure_floor)

    def conservative_from_primitive(
        self,
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        pressure: np.ndarray,
    ) -> np.ndarray:
        rho, u, v, pressure = np.broadcast_arrays(rho, u, v, pressure)
        rho = np.maximum(rho, self.rho_floor)
        pressure = np.maximum(pressure, self.pressure_floor)

        q = np.zeros((4, *rho.shape), dtype=np.result_type(rho, u, v, pressure))
        q[0] = rho
        q[1] = rho * u
        q[2] = rho * v
        q[3] = pressure / (self.gamma - 1.0) + 0.5 * rho * (u**2 + v**2)
        return q

    def enforce_physical_state(self, q: np.ndarray) -> np.ndarray:
        q_fixed = np.array(q, copy=True)
        rho = np.maximum(q_fixed[0], self.rho_floor)
        u = q_fixed[1] / rho
        v = q_fixed[2] / rho
        kinetic = 0.5 * rho * (u**2 + v**2)
        pressure = (self.gamma - 1.0) * (q_fixed[3] - kinetic)

        q_fixed[0] = rho
        q_fixed[1] = rho * u
        q_fixed[2] = rho * v
        q_fixed[3] = np.where(
            pressure > self.pressure_floor,
            q_fixed[3],
            self.pressure_floor / (self.gamma - 1.0) + kinetic,
        )
        return q_fixed

    def flux_x(self, q: np.ndarray) -> np.ndarray:
        rho, u, v, pressure = self.primitive_from_conservative(q)
        flux = np.zeros_like(q)
        flux[0] = q[1]
        flux[1] = rho * u**2 + pressure
        flux[2] = rho * u * v
        flux[3] = (q[3] + pressure) * u
        return flux

    def flux_y(self, q: np.ndarray) -> np.ndarray:
        rho, u, v, pressure = self.primitive_from_conservative(q)
        flux = np.zeros_like(q)
        flux[0] = q[2]
        flux[1] = rho * u * v
        flux[2] = rho * v**2 + pressure
        flux[3] = (q[3] + pressure) * v
        return flux

    def sound_speed(self, q: np.ndarray) -> np.ndarray:
        rho, _u, _v, pressure = self.primitive_from_conservative(q)
        return np.sqrt(self.gamma * pressure / rho)

    def max_wave_speed(self, q: np.ndarray) -> float:
        rho, u, v, pressure = self.primitive_from_conservative(q)
        c = np.sqrt(self.gamma * pressure / rho)
        return float(np.max(np.maximum(np.abs(u) + c, np.abs(v) + c)))


@dataclass(frozen=True)
class CompressibleNavierStokes2D(EulerEquation2D):
    """2D ideal-gas Navier-Stokes helper with periodic viscous terms.

    Conservative state ordering remains [rho, rho*u, rho*v, E]. The inviscid
    fluxes are inherited from EulerEquation2D; viscous stresses and heat
    conduction can be added as a source/RHS contribution by preliminary DNS
    drivers.
    """

    viscosity: float = 0.0
    prandtl: float = 0.72
    gas_constant: float = 1.0

    def temperature(self, rho: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return pressure / (rho * self.gas_constant)

    @staticmethod
    def _periodic_derivative(values: np.ndarray, spacing: float, axis: int) -> np.ndarray:
        return (np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)) / (2.0 * spacing)

    def viscous_rhs(self, q: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """Return the conservative viscous/thermal RHS on a periodic grid."""

        if self.viscosity == 0.0:
            return np.zeros_like(q)

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

        rhs = np.zeros_like(q)
        rhs[1] = self._periodic_derivative(tau_xx, dx, axis=-1)
        rhs[1] += self._periodic_derivative(tau_xy, dy, axis=-2)
        rhs[2] = self._periodic_derivative(tau_xy, dx, axis=-1)
        rhs[2] += self._periodic_derivative(tau_yy, dy, axis=-2)

        energy_flux_x = u * tau_xx + v * tau_xy + conductivity * dT_dx
        energy_flux_y = u * tau_xy + v * tau_yy + conductivity * dT_dy
        rhs[3] = self._periodic_derivative(energy_flux_x, dx, axis=-1)
        rhs[3] += self._periodic_derivative(energy_flux_y, dy, axis=-2)
        return rhs
