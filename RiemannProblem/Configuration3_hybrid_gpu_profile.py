"""Profiled CUDA/CuPy validation run for Configuration 3.

The conservative state, shock sensor, WENO fluxes, hyperviscosity, and compact
line solves remain on the GPU.  Only the final density field is copied to the
CPU for Matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import sys
import time as walltime

import matplotlib.pyplot as plt
import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.ndimage import maximum_filter
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


_BATCHED_THOMAS_KERNEL = cp.RawKernel(
    r"""
    extern "C" __global__
    void batched_thomas(const double* rhs, double* solution,
                        const double* c_prime, const double* inv_pivot,
                        const int n, const int n_lines, const double alpha)
    {
        const int line = blockDim.x * blockIdx.x + threadIdx.x;
        if (line >= n_lines) return;
        const long long base = (long long)line * n;
        solution[base] = rhs[base];
        for (int i = 1; i < n; ++i) {
            const double lower = (i == n - 1) ? 0.0 : alpha;
            solution[base + i] =
                (rhs[base + i] - lower * solution[base + i - 1]) * inv_pivot[i];
        }
        for (int i = n - 2; i >= 0; --i)
            solution[base + i] -= c_prime[i] * solution[base + i + 1];
    }
    """,
    "batched_thomas",
)


def dilate_mask_gpu(mask: cp.ndarray, width: int) -> cp.ndarray:
    if width <= 0:
        return mask.copy()
    size = 2 * width + 1
    return maximum_filter(mask, size=(size, size), mode="constant", cval=False)


hybrid.dilate_mask_2d = dilate_mask_gpu


@dataclass
class GPULineCompactDerivative:
    """Batched Thomas solve for the compact tridiagonal system on CUDA."""

    n: int
    dx: float
    alpha: float = 3.0 / 8.0

    def __post_init__(self) -> None:
        c_prime = np.zeros(self.n, dtype=np.float64)
        inv_pivot = np.empty(self.n, dtype=np.float64)
        inv_pivot[0] = 1.0
        if self.n > 1:
            for i in range(1, self.n):
                lower = 0.0 if i == self.n - 1 else self.alpha
                inv_pivot[i] = 1.0 / (1.0 - lower * c_prime[i - 1])
                if i < self.n - 1:
                    c_prime[i] = self.alpha * inv_pivot[i]
        self.c_prime = cp.asarray(c_prime)
        self.inv_pivot = cp.asarray(inv_pivot)

    def from_interface_flux(self, interface_flux: cp.ndarray) -> cp.ndarray:
        raw = (interface_flux[..., 1:] - interface_flux[..., :-1]) / self.dx
        leading_shape = raw.shape[:-1]
        rhs = cp.ascontiguousarray(raw.reshape(-1, self.n))
        n_lines = rhs.shape[0]
        solution = cp.empty_like(rhs)
        threads = 128
        blocks = (n_lines + threads - 1) // threads
        _BATCHED_THOMAS_KERNEL(
            (blocks,), (threads,),
            (rhs, solution, self.c_prime, self.inv_pivot,
             np.int32(self.n), np.int32(n_lines), np.float64(self.alpha)),
        )
        return solution.reshape(*leading_shape, self.n)


class GPUHybridEuler2DOperator(hybrid.HybridEuler2DOperator):
    def __post_init__(self) -> None:
        self.compact_x = GPULineCompactDerivative(self.domain.nx, self.domain.dx)
        self.compact_y = GPULineCompactDerivative(self.domain.ny, self.domain.dy)

    def stable_time_step(self, q: cp.ndarray, cfl: float, remaining_time: float) -> float:
        rho, u, v, pressure = self.equation.primitive_from_conservative(q)
        sound_speed = cp.sqrt(self.equation.gamma * pressure / rho)
        spectral_radius = (cp.abs(u) + sound_speed) / self.domain.dx
        spectral_radius += (cp.abs(v) + sound_speed) / self.domain.dy
        dt = cfl / float(cp.max(spectral_radius).item())
        return min(dt, remaining_time)


def validate_gpu_state(q: cp.ndarray, equation, label: str) -> None:
    rho = q[0]
    u = q[1] / rho
    v = q[2] / rho
    pressure = (equation.gamma - 1.0) * (q[3] - 0.5 * rho * (u**2 + v**2))
    valid = cp.all(cp.isfinite(q)) & cp.all(rho > 0.0) & cp.all(pressure > 0.0)
    if not bool(valid.item()):
        raise FloatingPointError(f"{label} contains non-finite or non-physical values")


def initial_state_gpu(config: Riemann2DConfig3) -> cp.ndarray:
    domain = config.domain
    x = domain.x_min + domain.dx * (cp.arange(config.nx) + 0.5)
    y = domain.y_min + domain.dy * (cp.arange(config.ny) + 0.5)
    x_grid, y_grid = cp.meshgrid(x, y, indexing="xy")
    right, top = x_grid > config.x0, y_grid > config.y0

    rho = cp.empty_like(x_grid)
    u = cp.empty_like(x_grid)
    v = cp.empty_like(x_grid)
    pressure = cp.empty_like(x_grid)
    states = (
        (right & top, 1.5, 1.5, 0.0, 0.0),
        (~right & top, 0.3, 0.5323, 1.206, 0.0),
        (~right & ~top, 0.029, 0.138, 1.206, 1.206),
        (right & ~top, 0.3, 0.5323, 0.0, 1.206),
    )
    for mask, p_value, rho_value, u_value, v_value in states:
        pressure[mask] = p_value
        rho[mask] = rho_value
        u[mask] = u_value
        v[mask] = v_value
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
        f"adaptive CFL={config.cfl:.5g}, tfinal={config.tfinal}, "
        f"device={cp.cuda.runtime.getDevice()}"
    )
    applications = 0
    time = 0.0
    step = 0
    cp.cuda.Stream.null.synchronize()
    run_start = walltime.perf_counter()
    report_start = run_start
    report_step = 0
    while time < config.tfinal:
        dt = operator.stable_time_step(q, config.cfl, config.tfinal - time)
        q = integrator.step(
            q,
            operator.rhs,
            dt,
            clean=lambda state: hybrid.apply_outflow_guard(equation.enforce_physical_state(state)),
        )
        step += 1
        time += dt
        if use_filter and step % args.hyperviscosity_interval == 0:
            q = filter_.apply(q, sensor.detect(q), equation)
            q = hybrid.apply_outflow_guard(q)
            applications += 1
        if step % args.progress_every == 0 or time >= config.tfinal:
            cp.cuda.Stream.null.synchronize()
            validate_gpu_state(q, equation, f"state at step {step}")
            now = walltime.perf_counter()
            chunk_steps = step - report_step
            chunk_ms = 1.0e3 * (now - report_start) / max(chunk_steps, 1)
            total_ms = 1.0e3 * (now - run_start) / max(step, 1)
            used_bytes = cp.get_default_memory_pool().used_bytes()
            total_bytes = cp.get_default_memory_pool().total_bytes()
            print(
                f"step={step:6d}, time={time:.6f}, dt={dt:.5e}, "
                f"chunk={chunk_ms:.2f} ms/step, avg={total_ms:.2f} ms/step, "
                f"pool={used_bytes / 2**20:.1f}/{total_bytes / 2**20:.1f} MiB"
            )
            report_start = now
            report_step = step

    cp.cuda.Stream.null.synchronize()
    elapsed = walltime.perf_counter() - run_start
    print(f"GPU solve wall time: {elapsed:.3f} s ({1.0e3 * elapsed / max(step, 1):.3f} ms/step)")

    diagnostics = hybrid.HybridRunDiagnostics(
        hyperviscosity_enabled=use_filter,
        hyperviscosity_applications=applications,
        final_sensor_fraction=float(cp.mean(sensor.detect(q)).item()),
    )
    print(
        f"Final density range: {float(cp.min(q[0]).item()):.8e} / "
        f"{float(cp.max(q[0]).item()):.8e}"
    )
    return q, time, step, diagnostics, elapsed


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
        f"GPU Hybrid 2D Euler Configuration 3 - Density, t = {config.tfinal:.3f}",
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
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--ny", type=int, default=1024)
    parser.add_argument(
        "--tfinal", type=float, default=0.3,
        help="First validation run at the paper comparison time. Use 0.8 or 0.85 only after validation.",
    )
    parser.add_argument("--cfl", type=float, default=0.05)
    parser.add_argument("--sensor-width", type=int, default=8)
    parser.add_argument("--jump-threshold", type=float, default=0.015)
    parser.add_argument("--compression-threshold", type=float, default=2.5)
    parser.add_argument("--shear-threshold", type=float, default=2.0)
    parser.add_argument(
        "--mn", type=float, default=0.001,
        help="Numerical hyperviscosity coefficient selected from the CPU tuning study.",
    )
    parser.add_argument("--hyperviscosity-interval", type=int, default=1)
    parser.add_argument("--hyperviscosity-density-weight", type=float, default=1.0)
    parser.add_argument("--hyperviscosity-momentum-weight", type=float, default=1.0)
    parser.add_argument("--hyperviscosity-energy-weight", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--zoom-center", type=float, nargs=2, default=(0.685, 0.685), metavar=("X", "Y"))
    parser.add_argument("--zoom-window", type=float, default=0.37)
    parser.add_argument("--fixed-density-limits", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--no-show", action="store_true")

    parser.add_argument(
        "--save-state",
        type=Path,
        default=Path(__file__).resolve().with_name("Riemann2D_Config3_gpu_state.npz"),
        help="Save the final conservative state and run metadata for CPU/GPU comparison.",
    )
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
    q_final, time, steps, diagnostics, elapsed = run_gpu_case(config, args)
    print(f"Completed GPU hybrid Configuration 3 at t={time:.6f} in {steps} steps.")
    print(f"Final sensor fraction: {diagnostics.final_sensor_fraction:.6f}")

    if args.save_state is not None:
        args.save_state.parent.mkdir(parents=True, exist_ok=True)
        q_cpu = cp.asnumpy(q_final)
        metadata = {
            "nx": config.nx,
            "ny": config.ny,
            "tfinal": float(time),
            "steps": int(steps),
            "cfl": float(config.cfl),
            "mn": float(args.mn),
            "hyperviscosity_interval": int(args.hyperviscosity_interval),
            "sensor_width": int(args.sensor_width),
            "jump_threshold": float(args.jump_threshold),
            "compression_threshold": float(args.compression_threshold),
            "shear_threshold": float(args.shear_threshold),
            "sensor_fraction": float(diagnostics.final_sensor_fraction),
            "elapsed_seconds": float(elapsed),
            "milliseconds_per_step": float(1.0e3 * elapsed / max(steps, 1)),
            "cupy_version": cp.__version__,
            "device": str(device_name),
        }
        np.savez_compressed(args.save_state, q=q_cpu, metadata=json.dumps(metadata, indent=2))
        print(f"Saved final state and metadata to {args.save_state}")
        print(json.dumps(metadata, indent=2))

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
