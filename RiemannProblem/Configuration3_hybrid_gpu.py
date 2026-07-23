"""CUDA/CuPy version of Configuration3_hybrid.py.

The conservative state, shock sensor, WENO fluxes, hyperviscosity, and compact
line solves remain on the GPU.  Only the final density field is copied to the
CPU for Matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    import cupy as cp
except ImportError as exc:
    raise SystemExit(
        "This program requires CuPy. Install the wheel matching the workstation's "
        "CUDA version, for example: pip install cupy-cuda12x"
    ) from exc


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import Configuration3_hybrid as hybrid
import OOP.equations as equations_module
from Configuration3_reference import Riemann2DConfig3
from OOP.time_operator import SSPRK3


# The original algorithm is array-module clean except for its SciPy compact
# solve. Redirect its NumPy globals to CuPy and replace that solve below.
hybrid.np = cp
equations_module.np = cp


@dataclass
class GPULineCompactDerivative:
    """Batched Thomas solve for the compact tridiagonal system on CUDA."""

    n: int
    dx: float
    alpha: float = 3.0 / 8.0

    def __post_init__(self) -> None:
        # Boundary rows in the CPU implementation have no off-diagonal term.
        lower = cp.full(self.n, self.alpha, dtype=cp.float64)
        upper = cp.full(self.n, self.alpha, dtype=cp.float64)
        lower[0] = 0.0
        lower[-1] = 0.0
        upper[0] = 0.0
        upper[-1] = 0.0

        self.c_prime = cp.zeros(self.n, dtype=cp.float64)
        self.inv_pivot = cp.empty(self.n, dtype=cp.float64)
        self.inv_pivot[0] = 1.0
        if self.n > 1:
            self.c_prime[0] = upper[0]
            for i in range(1, self.n):
                self.inv_pivot[i] = 1.0 / (1.0 - lower[i] * self.c_prime[i - 1])
                if i < self.n - 1:
                    self.c_prime[i] = upper[i] * self.inv_pivot[i]
        self.lower = lower

    def from_interface_flux(self, interface_flux: cp.ndarray) -> cp.ndarray:
        raw = (interface_flux[..., 1:] - interface_flux[..., :-1]) / self.dx
        leading_shape = raw.shape[:-1]
        rhs = raw.reshape(-1, self.n)
        forward = cp.empty_like(rhs)
        forward[:, 0] = rhs[:, 0]
        for i in range(1, self.n):
            forward[:, i] = (rhs[:, i] - self.lower[i] * forward[:, i - 1]) * self.inv_pivot[i]

        solution = cp.empty_like(rhs)
        solution[:, -1] = forward[:, -1]
        for i in range(self.n - 2, -1, -1):
            solution[:, i] = forward[:, i] - self.c_prime[i] * solution[:, i + 1]
        return solution.reshape(*leading_shape, self.n)


class GPUHybridEuler2DOperator(hybrid.HybridEuler2DOperator):
    def __post_init__(self) -> None:
        self.compact_x = GPULineCompactDerivative(self.domain.nx, self.domain.dx)
        self.compact_y = GPULineCompactDerivative(self.domain.ny, self.domain.dy)


def validate_gpu_state(q: cp.ndarray, equation, label: str) -> None:
    # Compute the raw quantities here because CuPy deliberately has no
    # ``numpy.errstate`` context manager. Invalid values are detected by the
    # finite/positivity checks immediately below.
    rho = q[0]
    u = q[1] / rho
    v = q[2] / rho
    kinetic = 0.5 * rho * (u**2 + v**2)
    pressure = (equation.gamma - 1.0) * (q[3] - kinetic)
    valid = cp.all(cp.isfinite(q)) & cp.all(rho > 0.0) & cp.all(pressure > 0.0)
    if not bool(valid.item()):
        raise FloatingPointError(f"{label} contains non-finite or non-physical values")


def initial_state_gpu(config: Riemann2DConfig3) -> cp.ndarray:
    """Build Configuration 3 directly on the GPU.

    ``Riemann2DConfig3.initial_state`` creates NumPy arrays.  The equation
    module is redirected to CuPy in this program, so passing those host arrays
    to its CuPy ufuncs would mix the two array libraries.
    """

    domain = config.domain
    x = domain.x_min + domain.dx * (cp.arange(config.nx) + 0.5)
    y = domain.y_min + domain.dy * (cp.arange(config.ny) + 0.5)
    x_grid, y_grid = cp.meshgrid(x, y, indexing="xy")

    right = x_grid > config.x0
    top = y_grid > config.y0
    state_1 = right & top
    state_2 = ~right & top
    state_3 = ~right & ~top
    state_4 = right & ~top

    rho = cp.empty_like(x_grid)
    u = cp.empty_like(x_grid)
    v = cp.empty_like(x_grid)
    pressure = cp.empty_like(x_grid)

    pressure[state_1], rho[state_1], u[state_1], v[state_1] = 1.5, 1.5, 0.0, 0.0
    pressure[state_2], rho[state_2], u[state_2], v[state_2] = 0.3, 0.5323, 1.206, 0.0
    pressure[state_3], rho[state_3], u[state_3], v[state_3] = 0.029, 0.138, 1.206, 1.206
    pressure[state_4], rho[state_4], u[state_4], v[state_4] = 0.3, 0.5323, 0.0, 1.206

    return config.equation.conservative_from_primitive(rho, u, v, pressure)


def run_gpu_case(config: Riemann2DConfig3, args: argparse.Namespace):
    equation = config.equation
    sensor = hybrid.EulerShockSensor2D(
        config.domain,
        equation,
        width=args.sensor_width,
        compression_threshold=args.compression_threshold,
        jump_threshold=args.jump_threshold,
        shear_threshold=None if args.shear_threshold == 0.0 else args.shear_threshold,
    )
    operator = GPUHybridEuler2DOperator(config.domain, equation, sensor)

    q = initial_state_gpu(config)
    q = hybrid.apply_outflow_guard(equation.enforce_physical_state(q))
    validate_gpu_state(q, equation, "initial state")
    dt, n_steps = operator.fixed_time_step(q, config.cfl, config.tfinal)

    use_filter = args.mn > 0.0 and args.hyperviscosity_interval > 0
    filter_ = hybrid.LocalHyperviscosity2D(
        config.domain,
        mn=args.mn,
        density_weight=args.hyperviscosity_density_weight,
        momentum_weight=args.hyperviscosity_momentum_weight,
        energy_weight=args.hyperviscosity_energy_weight,
    )
    integrator = SSPRK3()

    print(
        f"GPU hybrid 2D Euler Config 3: nx={config.nx}, ny={config.ny}, "
        f"dt={dt:.5e}, steps={n_steps}, device={cp.cuda.runtime.getDevice()}"
    )
    applications = 0
    for step in range(n_steps):
        q = integrator.step(
            q,
            operator.rhs,
            dt,
            clean=lambda state: hybrid.apply_outflow_guard(equation.enforce_physical_state(state)),
        )
        if use_filter and (step + 1) % args.hyperviscosity_interval == 0:
            q = filter_.apply(q, sensor.detect(q), equation)
            q = hybrid.apply_outflow_guard(q)
            applications += 1
        if (step + 1) % args.progress_every == 0 or step + 1 == n_steps:
            cp.cuda.Stream.null.synchronize()
            validate_gpu_state(q, equation, f"state at step {step + 1}")
            print(f"step={step + 1:6d}, time={(step + 1) * dt:.6f}")

    diagnostics = hybrid.HybridRunDiagnostics(
        hyperviscosity_enabled=use_filter,
        hyperviscosity_applications=applications,
        final_sensor_fraction=float(cp.mean(sensor.detect(q)).item()),
    )
    return q, n_steps * dt, n_steps, diagnostics


def plot_density_panels(
    q: cp.ndarray,
    config: Riemann2DConfig3,
    output_path: Path | None,
    show: bool,
    zoom_center: tuple[float, float],
    zoom_window: float,
    fixed_limits: tuple[float, float] | None,
) -> None:
    rho = cp.asnumpy(q[0])
    domain = config.domain
    extent = [domain.x_min, domain.x_max, domain.y_min, domain.y_max]
    vmin, vmax = fixed_limits if fixed_limits is not None else (None, None)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    fig.suptitle(
        f"Hybrid 2D Euler Configuration 3 - Density, t = {config.tfinal:.3f}",
        fontsize=13,
        fontweight="bold",
    )
    for ax, title in zip(axes, ("Whole domain", "Interaction-region zoom")):
        image = ax.imshow(
            rho, origin="lower", extent=extent, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, label="rho")

    half = 0.5 * zoom_window
    axes[1].set_xlim(max(domain.x_min, zoom_center[0] - half), min(domain.x_max, zoom_center[0] + half))
    axes[1].set_ylim(max(domain.y_min, zoom_center[1] - half), min(domain.y_max, zoom_center[1] + half))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        print(f"Saved density figure to {output_path}")
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid 2D Euler Configuration 3 on a CUDA GPU.")
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--ny", type=int, default=200)
    parser.add_argument("--tfinal", type=float, default=0.3)
    parser.add_argument("--cfl", type=float, default=0.08)
    parser.add_argument("--sensor-width", type=int, default=8)
    parser.add_argument("--jump-threshold", type=float, default=0.015)
    parser.add_argument("--compression-threshold", type=float, default=2.5)
    parser.add_argument("--shear-threshold", type=float, default=2.0)
    parser.add_argument("--mn", type=float, default=0.01)
    parser.add_argument("--hyperviscosity-interval", type=int, default=1)
    parser.add_argument("--hyperviscosity-density-weight", type=float, default=1.0)
    parser.add_argument("--hyperviscosity-momentum-weight", type=float, default=1.0)
    parser.add_argument("--hyperviscosity-energy-weight", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--zoom-center", type=float, nargs=2, default=(0.375, 0.375), metavar=("X", "Y"))
    parser.add_argument("--zoom-window", type=float, default=0.35)
    parser.add_argument("--fixed-density-limits", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--save",
        type=Path,
        default=Path(__file__).resolve().with_name("Riemann2D_Config3_hybrid_gpu_density.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        device = cp.cuda.Device()
        device.use()
        device_name = cp.cuda.runtime.getDeviceProperties(device.id)["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode()
        print(f"Using CUDA device: {device_name}")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        raise SystemExit(f"No usable CUDA GPU was found: {exc}") from exc

    config = Riemann2DConfig3(nx=args.nx, ny=args.ny, tfinal=args.tfinal, cfl=args.cfl)
    q_final, time, steps, diagnostics = run_gpu_case(config, args)
    print(f"Completed GPU hybrid Configuration 3 at t={time:.6f} in {steps} steps.")
    print(f"Final sensor fraction: {diagnostics.final_sensor_fraction:.6f}")
    plot_density_panels(
        q_final,
        config,
        args.save,
        not args.no_show,
        tuple(args.zoom_center),
        args.zoom_window,
        tuple(args.fixed_density_limits) if args.fixed_density_limits is not None else None,
    )


if __name__ == "__main__":
    main()
