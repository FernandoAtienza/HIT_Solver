from __future__ import annotations

from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

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


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from Configuration3_reference import Riemann2DConfig3
except ModuleNotFoundError:
    from Configuration3 import Riemann2DConfig3


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
    minimum_energy = kinetic + pressure_floor / (equation.gamma - 1.0)
    out[3] = xp.maximum(out[3], minimum_energy)
    return out


def flux_x_backend(q, equation):
    rho, u, v, pressure = primitive_backend(q, equation)
    return xp.stack(
        (
            rho * u,
            rho * u * u + pressure,
            rho * u * v,
            (q[3] + pressure) * u,
        ),
        axis=0,
    )


def flux_y_backend(q, equation):
    rho, u, v, pressure = primitive_backend(q, equation)
    return xp.stack(
        (
            rho * v,
            rho * u * v,
            rho * v * v + pressure,
            (q[3] + pressure) * v,
        ),
        axis=0,
    )


def validate_backend(q, equation, label: str) -> None:
    rho, _u, _v, pressure = primitive_backend(q, equation)
    if not bool(scalar(xp.all(xp.isfinite(q)))):
        raise FloatingPointError(f"{label}: non-finite conservative state")

    rho_min = scalar(xp.min(rho))
    pressure_min = scalar(xp.min(pressure))
    if rho_min <= 0.0 or pressure_min <= 0.0:
        raise FloatingPointError(
            f"{label}: nonphysical state "
            f"(rho_min={rho_min:.6e}, p_min={pressure_min:.6e})"
        )


def apply_outflow_guard(q, guard_cells: int = 4):
    if guard_cells <= 0:
        return q

    guarded = xp.array(q, copy=True)
    g = guard_cells
    guarded[:, :, :g] = guarded[:, :, g][:, :, None]
    guarded[:, :, -g:] = guarded[:, :, -g - 1][:, :, None]
    guarded[:, :g, :] = guarded[:, g, :][:, None, :]
    guarded[:, -g:, :] = guarded[:, -g - 1, :][:, None, :]
    return guarded


def take_clipped(values, indices, axis: int = -1):
    n = values.shape[axis]
    return xp.take(values, xp.clip(indices, 0, n - 1), axis=axis)


def weno7_flux(v1, v2, v3, v4, v5, v6, v7, epsilon: float = 1.0e-10):
    q0 = -(1.0 / 4.0) * v1 + (13.0 / 12.0) * v2 - (23.0 / 12.0) * v3 + (25.0 / 12.0) * v4
    q1 = (1.0 / 12.0) * v2 - (5.0 / 12.0) * v3 + (13.0 / 12.0) * v4 + (1.0 / 4.0) * v5
    q2 = -(1.0 / 12.0) * v3 + (7.0 / 12.0) * v4 + (7.0 / 12.0) * v5 - (1.0 / 12.0) * v6
    q3 = (1.0 / 4.0) * v4 + (13.0 / 12.0) * v5 - (5.0 / 12.0) * v6 + (1.0 / 12.0) * v7

    is0 = (
        v1 * (544.0 * v1 - 3882.0 * v2 + 4642.0 * v3 - 1854.0 * v4)
        + v2 * (7043.0 * v2 - 17246.0 * v3 + 7042.0 * v4)
        + v3 * (11003.0 * v3 - 9402.0 * v4)
        + 2107.0 * v4**2
    )
    is1 = (
        v2 * (267.0 * v2 - 1642.0 * v3 + 1602.0 * v4 - 494.0 * v5)
        + v3 * (2843.0 * v3 - 5966.0 * v4 + 1922.0 * v5)
        + v4 * (3443.0 * v4 - 2522.0 * v5)
        + 547.0 * v5**2
    )
    is2 = (
        v3 * (547.0 * v3 - 2522.0 * v4 + 1922.0 * v5 - 494.0 * v6)
        + v4 * (3443.0 * v4 - 5966.0 * v5 + 1602.0 * v6)
        + v5 * (2843.0 * v5 - 1642.0 * v6)
        + 267.0 * v6**2
    )
    is3 = (
        v4 * (2107.0 * v4 - 9402.0 * v5 + 7042.0 * v6 - 1854.0 * v7)
        + v5 * (11003.0 * v5 - 17246.0 * v6 + 4642.0 * v7)
        + v6 * (7043.0 * v6 - 3882.0 * v7)
        + 547.0 * v7**2
    )

    a0 = (1.0 / 35.0) / (epsilon + is0) ** 2
    a1 = (12.0 / 35.0) / (epsilon + is1) ** 2
    a2 = (18.0 / 35.0) / (epsilon + is2) ** 2
    a3 = (4.0 / 35.0) / (epsilon + is3) ** 2
    weight_sum = a0 + a1 + a2 + a3
    return (a0 * q0 + a1 * q1 + a2 * q2 + a3 * q3) / weight_sum


def weno7_interfaces(q, point_flux, alpha: float):
    """Componentwise WENO-7 with global local-Lax--Friedrichs splitting."""
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


class WENO7Euler2DOperator:
    """Seventh-order WENO operator applied at every x- and y-interface."""

    def __init__(self, domain, equation):
        self.domain = domain
        self.equation = equation

    def _axis_derivative(self, q_axis, flux_axis, spacing: float, normal_velocity_index: int):
        rho, u, v, pressure = primitive_backend(q_axis, self.equation)
        normal_velocity = u if normal_velocity_index == 1 else v
        sound_speed = xp.sqrt(self.equation.gamma * pressure / rho)
        alpha = scalar(xp.max(xp.abs(normal_velocity) + sound_speed))

        interface_flux = weno7_interfaces(q_axis, flux_axis, alpha)
        return (interface_flux[..., 1:] - interface_flux[..., :-1]) / spacing

    def rhs(self, q):
        q_safe = enforce_physical_backend(q, self.equation)

        derivative_x = self._axis_derivative(
            q_safe,
            flux_x_backend(q_safe, self.equation),
            self.domain.dx,
            normal_velocity_index=1,
        )

        q_y = xp.moveaxis(q_safe, -2, -1)
        flux_y = xp.moveaxis(flux_y_backend(q_safe, self.equation), -2, -1)
        derivative_y = self._axis_derivative(
            q_y,
            flux_y,
            self.domain.dy,
            normal_velocity_index=2,
        )
        derivative_y = xp.moveaxis(derivative_y, -1, -2)

        return -derivative_x - derivative_y

    def fixed_time_step(self, q, cfl: float, t_end: float):
        rho, u, v, pressure = primitive_backend(q, self.equation)
        sound_speed = xp.sqrt(self.equation.gamma * pressure / rho)
        spectral_radius = (xp.abs(u) + sound_speed) / self.domain.dx
        spectral_radius += (xp.abs(v) + sound_speed) / self.domain.dy
        dt = cfl / scalar(xp.max(spectral_radius))
        n_steps = int(np.ceil(t_end / dt))
        return t_end / n_steps, n_steps


def ssprk3_step(q, rhs, dt, clean):
    q1 = clean(q + dt * rhs(q))
    q2 = clean(0.75 * q + 0.25 * (q1 + dt * rhs(q1)))
    return clean((1.0 / 3.0) * q + (2.0 / 3.0) * (q2 + dt * rhs(q2)))


def run_weno_case(config, progress_every: int = 100, guard_cells: int = 4):
    equation = config.equation
    operator = WENO7Euler2DOperator(config.domain, equation)

    q = xp.asarray(config.initial_state(), dtype=xp.float64)
    q = apply_outflow_guard(enforce_physical_backend(q, equation), guard_cells)
    validate_backend(q, equation, "initial state")

    dt, n_steps = operator.fixed_time_step(q, config.cfl, config.tfinal)
    print(
        f"WENO-only 2D Euler Config 3: backend={BACKEND}, "
        f"nx={config.nx}, ny={config.ny}, dt={dt:.5e}, "
        f"steps={n_steps}, tfinal={config.tfinal}"
    )

    last_valid = xp.array(q, copy=True)
    time = 0.0

    def clean(state):
        return apply_outflow_guard(enforce_physical_backend(state, equation), guard_cells)

    for step in range(n_steps):
        q = ssprk3_step(q, operator.rhs, dt, clean)
        time = (step + 1) * dt

        if (step + 1) % progress_every == 0 or step + 1 == n_steps:
            try:
                validate_backend(q, equation, f"state at step {step + 1}")
            except FloatingPointError:
                print(f"Instability detected at step {step + 1}; returning last valid state.")
                return last_valid, time, step + 1

            rho, _u, _v, pressure = primitive_backend(q, equation)
            print(
                f"step={step + 1:6d}, time={time:.6f}, "
                f"rho=[{scalar(xp.min(rho)):.6e}, {scalar(xp.max(rho)):.6e}], "
                f"p=[{scalar(xp.min(pressure)):.6e}, {scalar(xp.max(pressure)):.6e}]"
            )
            last_valid = xp.array(q, copy=True)

    if BACKEND == "cupy":
        cp.cuda.Stream.null.synchronize()
    return q, time, n_steps


def vorticity_numpy(q_numpy, domain, equation):
    rho = q_numpy[0]
    u = q_numpy[1] / rho
    v = q_numpy[2] / rho
    dv_dy, dv_dx = np.gradient(v, domain.dy, domain.dx, edge_order=2)
    du_dy, du_dx = np.gradient(u, domain.dy, domain.dx, edge_order=2)
    return dv_dx - du_dy


def plot_weno_solution(
    q,
    config,
    output_path: Path | None = None,
    show: bool = True,
    vorticity_limit: float | None = 60.0,
    vorticity_cmap: str = "coolwarm",
):
    q_numpy = to_numpy(q)
    equation = config.equation
    domain = config.domain

    rho = q_numpy[0]
    u = q_numpy[1] / rho
    v = q_numpy[2] / rho
    pressure = (equation.gamma - 1.0) * (
        q_numpy[3] - 0.5 * rho * (u * u + v * v)
    )
    omega = vorticity_numpy(q_numpy, domain, equation)

    fields = [
        (rho, "Density", r"$\rho$", "viridis", None, None),
        (pressure, "Pressure", r"$p$", "viridis", None, None),
        (
            omega,
            "Vorticity",
            r"$\omega_z$",
            vorticity_cmap,
            -vorticity_limit if vorticity_limit is not None else None,
            vorticity_limit,
        ),
    ]
    extent = [domain.x_min, domain.x_max, domain.y_min, domain.y_max]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fig.suptitle(
        f"WENO-7 2D Euler Riemann Problem -- Configuration 3, "
        f"t = {config.tfinal:.3f}",
        fontsize=13,
        fontweight="bold",
    )

    for axis, (values, title, label, cmap, vmin, vmax) in zip(axes, fields):
        image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            cmap=cmap,
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(image, ax=axis, label=label)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        print(f"Saved figure to {output_path}")

    if show:
        plt.show(block=True)
    else:
        plt.close(fig)


def save_fields(q, config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    q_numpy = to_numpy(q)
    np.savez_compressed(
        path,
        q=q_numpy,
        x=np.linspace(config.domain.x_min, config.domain.x_max, config.domain.nx),
        y=np.linspace(config.domain.y_min, config.domain.y_max, config.domain.ny),
        tfinal=config.tfinal,
        nx=config.nx,
        ny=config.ny,
        backend=BACKEND,
    )
    print(f"Saved fields to {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the WENO-only 2D Euler Configuration 3 benchmark."
    )
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="cupy")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--ny", type=int, default=512)
    parser.add_argument("--tfinal", type=float, default=0.3)
    parser.add_argument("--cfl", type=float, default=0.05)
    parser.add_argument("--guard-cells", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--vorticity-limit", type=float, default=60.0)
    parser.add_argument("--vorticity-cmap", default="coolwarm")
    parser.add_argument(
        "--save",
        type=Path,
        default=Path(__file__).resolve().with_name("Riemann2D_Config3_WENO_only.png"),
    )
    parser.add_argument(
        "--save-fields",
        type=Path,
        default=None,
        help="Optional .npz path for the final conservative fields.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_backend(args.backend, args.device)

    config = Riemann2DConfig3(
        nx=args.nx,
        ny=args.ny,
        tfinal=args.tfinal,
        cfl=args.cfl,
    )

    q_final, time, steps = run_weno_case(
        config,
        progress_every=args.progress_every,
        guard_cells=args.guard_cells,
    )
    print(f"Completed WENO-only Configuration 3 at t={time:.6f} in {steps} steps.")

    if args.save_fields is not None:
        save_fields(q_final, config, args.save_fields)

    vorticity_limit = None if args.vorticity_limit == 0.0 else args.vorticity_limit
    plot_weno_solution(
        q_final,
        config,
        output_path=args.save,
        show=not args.no_show,
        vorticity_limit=vorticity_limit,
        vorticity_cmap=args.vorticity_cmap,
    )


if __name__ == "__main__":
    main()
