from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OOP.domain import Domain2D
from OOP.equations import EulerEquation2D
from OOP.time_operator import SSPRK3


@dataclass(frozen=True)
class Riemann2DConfig3:
    """Lax-Liu / Kurganov-Tadmor 2D Riemann problem, Configuration 3."""

    nx: int = 400
    ny: int = 400
    tfinal: float = 0.3
    cfl: float = 0.25
    gamma: float = 1.4
    x0: float = 0.5
    y0: float = 0.5

    @property
    def domain(self) -> Domain2D:
        return Domain2D(0.0, 1.0, 0.0, 1.0, self.nx, self.ny)

    @property
    def equation(self) -> EulerEquation2D:
        return EulerEquation2D(gamma=self.gamma)

    def initial_state(self) -> np.ndarray:
        """Return U with ordering [rho, rho*u, rho*v, E] and shape (4, ny, nx)."""

        x_grid, y_grid = self.domain.mesh()
        pressure = np.empty_like(x_grid)
        rho = np.empty_like(x_grid)
        u = np.empty_like(x_grid)
        v = np.empty_like(x_grid)

        right = x_grid > self.x0
        top = y_grid > self.y0

        state_1 = right & top
        state_2 = ~right & top
        state_3 = ~right & ~top
        state_4 = right & ~top

        pressure[state_1], rho[state_1], u[state_1], v[state_1] = 1.5, 1.5, 0.0, 0.0
        pressure[state_2], rho[state_2], u[state_2], v[state_2] = 0.3, 0.5323, 1.206, 0.0
        pressure[state_3], rho[state_3], u[state_3], v[state_3] = 0.029, 0.138, 1.206, 1.206
        pressure[state_4], rho[state_4], u[state_4], v[state_4] = 0.3, 0.5323, 0.0, 1.206

        return self.equation.conservative_from_primitive(rho, u, v, pressure)


@dataclass(frozen=True)
class RusanovEuler2DOperator:
    """Dimensionally conservative 2D Euler RHS with outflow edge padding."""

    domain: Domain2D
    equation: EulerEquation2D

    def _pad_outflow(self, q: np.ndarray) -> np.ndarray:
        return np.pad(q, ((0, 0), (1, 1), (1, 1)), mode="edge")

    def _interface_flux_x(self, q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
        flux_left = self.equation.flux_x(q_left)
        flux_right = self.equation.flux_x(q_right)
        _rho_l, u_l, _v_l, _p_l = self.equation.primitive_from_conservative(q_left)
        _rho_r, u_r, _v_r, _p_r = self.equation.primitive_from_conservative(q_right)
        speed = np.maximum(np.abs(u_l) + self.equation.sound_speed(q_left), np.abs(u_r) + self.equation.sound_speed(q_right))
        return 0.5 * (flux_left + flux_right) - 0.5 * speed[None, ...] * (q_right - q_left)

    def _interface_flux_y(self, q_bottom: np.ndarray, q_top: np.ndarray) -> np.ndarray:
        flux_bottom = self.equation.flux_y(q_bottom)
        flux_top = self.equation.flux_y(q_top)
        _rho_b, _u_b, v_b, _p_b = self.equation.primitive_from_conservative(q_bottom)
        _rho_t, _u_t, v_t, _p_t = self.equation.primitive_from_conservative(q_top)
        speed = np.maximum(np.abs(v_b) + self.equation.sound_speed(q_bottom), np.abs(v_t) + self.equation.sound_speed(q_top))
        return 0.5 * (flux_bottom + flux_top) - 0.5 * speed[None, ...] * (q_top - q_bottom)

    def rhs(self, q: np.ndarray) -> np.ndarray:
        q_safe = self.equation.enforce_physical_state(q)
        q_pad = self._pad_outflow(q_safe)

        flux_x = self._interface_flux_x(q_pad[:, 1:-1, :-1], q_pad[:, 1:-1, 1:])
        flux_y = self._interface_flux_y(q_pad[:, :-1, 1:-1], q_pad[:, 1:, 1:-1])

        rhs = -(flux_x[:, :, 1:] - flux_x[:, :, :-1]) / self.domain.dx
        rhs -= (flux_y[:, 1:, :] - flux_y[:, :-1, :]) / self.domain.dy
        return rhs

    def stable_time_step(self, q: np.ndarray, cfl: float, remaining_time: float) -> float:
        rho, u, v, pressure = self.equation.primitive_from_conservative(q)
        sound_speed = np.sqrt(self.equation.gamma * pressure / rho)
        spectral_radius = (np.abs(u) + sound_speed) / self.domain.dx
        spectral_radius += (np.abs(v) + sound_speed) / self.domain.dy
        dt = cfl / float(np.max(spectral_radius))
        return min(dt, remaining_time)


def validate_state(q: np.ndarray, equation: EulerEquation2D, label: str = "state") -> None:
    rho, _u, _v, pressure = equation.primitive_from_conservative(q)
    if not np.all(np.isfinite(q)):
        raise FloatingPointError(f"{label} contains NaN or Inf values")
    if np.any(rho <= 0.0):
        raise FloatingPointError(f"{label} contains non-positive density")
    if np.any(pressure <= 0.0):
        raise FloatingPointError(f"{label} contains non-positive pressure")


def vorticity_z(q: np.ndarray, domain: Domain2D, equation: EulerEquation2D) -> np.ndarray:
    rho, u, v, _pressure = equation.primitive_from_conservative(q)
    _dv_dy, dv_dx = np.gradient(v, domain.dy, domain.dx, edge_order=2)
    du_dy, _du_dx = np.gradient(u, domain.dy, domain.dx, edge_order=2)
    return dv_dx - du_dy


def primitive_fields(
    q: np.ndarray,
    equation: EulerEquation2D,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return equation.primitive_from_conservative(q)


def run_case(config: Riemann2DConfig3, progress_every: int = 50) -> tuple[np.ndarray, float, int]:
    equation = config.equation
    operator = RusanovEuler2DOperator(config.domain, equation)
    integrator = SSPRK3()

    q = equation.enforce_physical_state(config.initial_state())
    validate_state(q, equation, "initial state")

    time = 0.0
    step = 0
    while time < config.tfinal:
        dt = operator.stable_time_step(q, config.cfl, config.tfinal - time)
        q = integrator.step(q, operator.rhs, dt, clean=equation.enforce_physical_state)
        time += dt
        step += 1

        if step % progress_every == 0 or time >= config.tfinal:
            validate_state(q, equation, f"state at step {step}")
            print(f"step={step:5d}, time={time:.6f}, dt={dt:.3e}")

    return q, time, step


def plot_solution(
    q: np.ndarray,
    config: Riemann2DConfig3,
    output_path: Path | None = None,
    show: bool = True,
    vorticity_limit: float | None = 60.0,
    vorticity_cmap: str = "coolwarm",
) -> None:
    domain = config.domain
    equation = config.equation
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
        f"2D Euler Riemann Problem - Configuration 3, t = {config.tfinal:.3f}",
        fontsize=13,
        fontweight="bold",
    )

    for ax, (field, title, label, cmap, vmin, vmax) in zip(axes, fields):
        image = ax.imshow(
            field,
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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(output_path, bbox_inches="tight", dpi=200)
            print(f"Saved figure to {output_path}")
        except OSError as exc:
            print(f"Could not save figure to {output_path}: {exc}")

    if show:
        plt.show(block=True)
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 2D Euler Riemann problem Configuration 3.")
    parser.add_argument("--nx", type=int, default=400, help="Number of cells in x.")
    parser.add_argument("--ny", type=int, default=400, help="Number of cells in y.")
    parser.add_argument("--tfinal", type=float, default=0.3, help="Final time.")
    parser.add_argument("--cfl", type=float, default=0.25, help="CFL number.")
    parser.add_argument("--no-show", action="store_true", help="Save/compute without opening a plot window.")
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
        default=Path(__file__).resolve().with_name("Riemann2D_Config3.png"),
        help="Path for the output figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Riemann2DConfig3(nx=args.nx, ny=args.ny, tfinal=args.tfinal, cfl=args.cfl)
    q_final, time, steps = run_case(config)
    print(f"Completed Configuration 3 at t={time:.6f} in {steps} steps.")
    vorticity_limit = None if args.vorticity_limit == 0.0 else args.vorticity_limit
    plot_solution(
        q_final,
        config,
        output_path=args.save,
        show=not args.no_show,
        vorticity_limit=vorticity_limit,
        vorticity_cmap=args.vorticity_cmap,
    )


if __name__ == "__main__":
    main()
