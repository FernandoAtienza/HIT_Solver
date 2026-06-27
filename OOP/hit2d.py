from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import argparse
import sys

import numpy as np


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OOP.domain import Domain2D
from OOP.equations import CompressibleNavierStokes2D, EulerEquation2D
from OOP.forcing import IsotropicShellOUForcing2D
from OOP.parallel.backend import array_module, select_array_module, to_numpy
from OOP.parallel.equations import ParallelCompressibleNavierStokes2D, ParallelEulerEquation2D
from OOP.parallel.spatial_operator import (
    ParallelPeriodicEulerShockSensor2D,
    ParallelPeriodicHybridEuler2DOperator,
    ParallelPeriodicHyperviscosity2D,
)
from OOP.spatial_operator import (
    PeriodicEulerShockSensor2D,
    PeriodicHybridEuler2DOperator,
    PeriodicHyperviscosity2D,
)
from OOP.time_operator import SSPRK3


DEFAULT_SNAPSHOT_DIR = Path(__file__).resolve().parents[1] / "2D" / "hit2d_snapshots"


@dataclass(frozen=True)
class HIT2DConfig:
    nx: int = 128
    ny: int = 128
    length: float = 2.0 * np.pi
    gamma: float = 1.4
    cfl: float = 0.05
    tfinal: float = 1.0
    target_mach: float = 0.1
    initial_kmin: int = 1
    initial_kmax: int = 3
    forcing_kmin: float | None = None
    forcing_kmax: float = 3.0
    forcing_rms: float = 1.0
    forcing_correlation_time: float = 1.0
    forcing_alpha_memory: float = 0.2
    target_energy_injection: float | None = 1.0e-3
    min_forcing_power: float = 1.0e-6
    max_forcing_rescale: float = 20.0
    forcing_seed: int = 1234
    viscosity: float = 1.0e-3
    prandtl: float = 0.72
    sensor_width: int = 4
    jump_threshold: float = 0.04
    compression_threshold: float = 2.5
    shear_threshold: float | None = None
    hyperviscosity_mn: float = 0.002
    hyperviscosity_interval: int = 5
    hyperviscosity_on_shocks_only: bool = False
    diagnostics_every: int = 25
    snapshot_every: int = 100
    forcing_anisotropy_warning: float = 0.2
    flow_anisotropy_warning: float = 0.4
    weno_fraction_warning: float = 0.5
    output_dir: Path = field(default_factory=lambda: DEFAULT_SNAPSHOT_DIR)
    backend: str = "numpy"

    @classmethod
    def isotropic_128(
        cls,
        output_dir: Path = DEFAULT_SNAPSHOT_DIR,
        backend: str = "cupy",
    ) -> "HIT2DConfig":
        """Recommended 128x128 annular-shell OU-forced HIT configuration."""

        return cls(
            nx=128,
            ny=128,
            tfinal=12.0,
            target_mach=0.5,
            forcing_kmin=3.0,
            forcing_kmax=5.0,
            forcing_correlation_time=1.0,
            forcing_alpha_memory=0.2,
            target_energy_injection=1.0e-3,
            viscosity=7.5e-4,
            hyperviscosity_mn=0.002,
            snapshot_every=75,
            diagnostics_every=25,
            output_dir=output_dir,
            backend=backend,
        )


def create_grid(config: HIT2DConfig) -> tuple[Domain2D, np.ndarray, np.ndarray]:
    """Create the square periodic 2D domain and mesh."""

    domain = Domain2D(0.0, config.length, 0.0, config.length, config.nx, config.ny)
    x_grid, y_grid = domain.mesh()
    return domain, x_grid, y_grid


def primitive_to_conservative(
    rho: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    pressure: np.ndarray,
    equation: EulerEquation2D,
) -> np.ndarray:
    return equation.conservative_from_primitive(rho, u, v, pressure)


def conservative_to_primitive(
    q: np.ndarray,
    equation: EulerEquation2D,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return equation.primitive_from_conservative(q)


def _spectral_wavenumbers(domain: Domain2D, xp=np) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kx_1d = xp.asarray(2.0 * np.pi * np.fft.fftfreq(domain.nx, d=domain.dx))
    ky_1d = xp.asarray(2.0 * np.pi * np.fft.fftfreq(domain.ny, d=domain.dy))
    kx, ky = xp.meshgrid(kx_1d, ky_1d, indexing="xy")
    return kx, ky, kx**2 + ky**2


def _spectral_derivatives(
    values: np.ndarray,
    domain: Domain2D,
) -> tuple[np.ndarray, np.ndarray]:
    xp = array_module(values)
    kx, ky, _k2 = _spectral_wavenumbers(domain, xp=xp)
    values_hat = xp.fft.fft2(values)
    d_dx = xp.fft.ifft2(1j * kx * values_hat).real
    d_dy = xp.fft.ifft2(1j * ky * values_hat).real
    return d_dx, d_dy


def initialize_hit_2d(
    config: HIT2DConfig,
    domain: Domain2D,
    equation: EulerEquation2D,
    rng: np.random.Generator,
    xp=np,
) -> np.ndarray:
    """Initialize weakly compressible HIT from a low-k streamfunction.

    A streamfunction gives u = dpsi/dy and v = -dpsi/dx, so the initial velocity
    is divergence-free to roundoff. The velocity is then scaled to the requested
    turbulent Mach number based on the mean sound speed.
    """

    x_grid_np, y_grid_np = domain.mesh()
    x_grid = xp.asarray(x_grid_np)
    y_grid = xp.asarray(y_grid_np)
    psi = xp.zeros_like(x_grid)
    for kx in range(-config.initial_kmax, config.initial_kmax + 1):
        for ky in range(-config.initial_kmax, config.initial_kmax + 1):
            # Keep one representative of each +/- pair while retaining all
            # rotations and reflections of each wavevector shell.
            if ky < 0 or (ky == 0 and kx <= 0):
                continue
            magnitude = np.sqrt(kx**2 + ky**2)
            if magnitude < config.initial_kmin or magnitude > config.initial_kmax:
                continue
            amplitude = 1.0 / max(kx**2 + ky**2, 1)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            psi += amplitude * xp.cos(kx * x_grid + ky * y_grid + phase)

    dpsi_dx, dpsi_dy = _spectral_derivatives(psi, domain)
    u = dpsi_dy
    v = -dpsi_dx
    u -= xp.mean(u)
    v -= xp.mean(v)

    rho = xp.ones_like(u)
    pressure = xp.full_like(u, 1.0 / config.gamma)
    sound_speed = xp.sqrt(equation.gamma * pressure / rho)
    velocity_rms = xp.sqrt(xp.mean(u**2 + v**2))
    if float(velocity_rms) <= np.finfo(float).eps:
        raise ValueError("initial streamfunction produced zero velocity")
    velocity_scale = config.target_mach * float(xp.mean(sound_speed)) / velocity_rms
    u *= velocity_scale
    v *= velocity_scale
    return primitive_to_conservative(rho, u, v, pressure, equation)


def make_solenoidal_forcing_2d(
    domain: Domain2D,
    rng: np.random.Generator,
    kf: int = 3,
    force_rms: float = 1.0,
    rho: np.ndarray | None = None,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
    target_energy_injection: float | None = None,
    min_power: float = 1.0e-6,
    max_rescale: float | None = 20.0,
    eps: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Generate low-wavenumber solenoidal acceleration forcing.

    Random Fourier coefficients are truncated to |k| <= kf and projected
    perpendicular to k, matching the large-scale solenoidal forcing philosophy
    used for stationary compressible HIT. The returned field is an acceleration
    f; the conservative source uses rho*f in momentum.
    """

    xp = array_module(rho) if rho is not None else np
    fx_hat = xp.fft.fft2(xp.asarray(rng.standard_normal((domain.ny, domain.nx))))
    fy_hat = xp.fft.fft2(xp.asarray(rng.standard_normal((domain.ny, domain.nx))))
    kx, ky, k2 = _spectral_wavenumbers(domain, xp=xp)

    mask = (k2 > 0.0) & (xp.sqrt(k2) <= float(kf))
    k2_safe = xp.where(mask, k2, 1.0)
    k_dot_f = kx * fx_hat + ky * fy_hat
    fx_hat = xp.where(mask, fx_hat - kx * k_dot_f / k2_safe, 0.0)
    fy_hat = xp.where(mask, fy_hat - ky * k_dot_f / k2_safe, 0.0)

    fx = xp.fft.ifft2(fx_hat).real
    fy = xp.fft.ifft2(fy_hat).real
    current_force_rms = float(xp.sqrt(xp.mean(fx**2 + fy**2)))
    if current_force_rms > eps and force_rms > 0.0:
        fx *= force_rms / current_force_rms
        fy *= force_rms / current_force_rms

    info = {"alpha": 1.0, "power_before_rescale": 0.0}
    if target_energy_injection is not None and rho is not None and u is not None and v is not None:
        power = float(xp.mean(rho * (fx * u + fy * v)))
        info["power_before_rescale"] = power
        if abs(power) > max(min_power, eps):
            alpha = target_energy_injection / power
            if max_rescale is not None:
                alpha = float(np.clip(alpha, -max_rescale, max_rescale))
            fx *= alpha
            fy *= alpha
            info["alpha"] = float(alpha)
        else:
            fx.fill(0.0)
            fy.fill(0.0)
            info["alpha"] = 0.0
    return fx, fy, info


def forcing_source(
    q: np.ndarray,
    equation: EulerEquation2D,
    fx: np.ndarray,
    fy: np.ndarray,
) -> np.ndarray:
    rho, u, v, _pressure = conservative_to_primitive(q, equation)
    xp = array_module(q)
    source = xp.zeros_like(q)
    source[1] = rho * fx
    source[2] = rho * fy
    source[3] = rho * (u * fx + v * fy)
    return source


def divergence_and_vorticity(
    u: np.ndarray,
    v: np.ndarray,
    domain: Domain2D,
) -> tuple[np.ndarray, np.ndarray]:
    du_dx, du_dy = _spectral_derivatives(u, domain)
    dv_dx, dv_dy = _spectral_derivatives(v, domain)
    divergence = du_dx + dv_dy
    vorticity = dv_dx - du_dy
    return divergence, vorticity


def compute_diagnostics(
    q: np.ndarray,
    equation: EulerEquation2D,
    domain: Domain2D,
    initial_mass: float | None = None,
    forcing_info: dict[str, float] | None = None,
    weno_fraction: float = 0.0,
) -> dict[str, float]:
    """Compute compact scalar diagnostics for the periodic turbulence state."""

    rho, u, v, pressure = conservative_to_primitive(q, equation)
    divergence, vorticity = divergence_and_vorticity(u, v, domain)
    xp = array_module(q)
    sound_speed = xp.sqrt(equation.gamma * pressure / rho)
    mass = float(xp.mean(rho) * (domain.x_max - domain.x_min) * (domain.y_max - domain.y_min))
    u_fluct = u - xp.mean(u)
    v_fluct = v - xp.mean(v)
    uu = float(xp.mean(rho * u_fluct**2))
    vv = float(xp.mean(rho * v_fluct**2))
    uv = float(xp.mean(rho * u_fluct * v_fluct))
    component_sum = uu + vv
    covariance_scale = np.sqrt(max(uu * vv, 0.0))
    diagnostics = {
        "kinetic_energy": 0.5 * component_sum,
        "Kx": 0.5 * uu,
        "Ky": 0.5 * vv,
        "A_K": abs(uu - vv) / component_sum if component_sum > 0.0 else 0.0,
        "C_uv": uv / covariance_scale if covariance_scale > 0.0 else 0.0,
        "rms_velocity": float(xp.sqrt(xp.mean(u**2 + v**2))),
        "turbulent_mach": float(xp.sqrt(xp.mean(u**2 + v**2)) / xp.mean(sound_speed)),
        "mean_density": float(xp.mean(rho)),
        "mean_pressure": float(xp.mean(pressure)),
        "divergence_rms": float(xp.sqrt(xp.mean(divergence**2))),
        "vorticity_rms": float(xp.sqrt(xp.mean(vorticity**2))),
        "mass": mass,
        "mass_error": 0.0 if initial_mass is None else mass - initial_mass,
        "weno_fraction": weno_fraction,
    }
    diagnostics.update(
        {
            "P_in": 0.0,
            "Fxx": 0.0,
            "Fyy": 0.0,
            "Fxy": 0.0,
            "A_F": 0.0,
            "forcing_alpha": 0.0,
        }
    )
    if forcing_info is not None:
        diagnostics.update(
            {
                "P_in": forcing_info["injected_power"],
                "Fxx": forcing_info["Fxx"],
                "Fyy": forcing_info["Fyy"],
                "Fxy": forcing_info["Fxy"],
                "A_F": forcing_info["A_F"],
                "forcing_alpha": forcing_info["alpha"],
            }
        )
    return diagnostics


def save_snapshot(
    output_dir: Path,
    step: int,
    time: float,
    q: np.ndarray,
    equation: EulerEquation2D,
    domain: Domain2D,
) -> Path:
    rho, u, v, pressure = conservative_to_primitive(q, equation)
    divergence, vorticity = divergence_and_vorticity(u, v, domain)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"hit2d_step{step:07d}.npz"
    np.savez_compressed(
        path,
        step=step,
        time=time,
        x=domain.x,
        y=domain.y,
        rho=to_numpy(rho),
        u=to_numpy(u),
        v=to_numpy(v),
        pressure=to_numpy(pressure),
        vorticity=to_numpy(vorticity),
        divergence=to_numpy(divergence),
    )
    return path


def _make_equation(config: HIT2DConfig, xp=np):
    if xp is not np:
        if config.viscosity > 0.0:
            return ParallelCompressibleNavierStokes2D(
                gamma=config.gamma,
                viscosity=config.viscosity,
                prandtl=config.prandtl,
            )
        return ParallelEulerEquation2D(gamma=config.gamma)

    if config.viscosity > 0.0:
        return CompressibleNavierStokes2D(
            gamma=config.gamma,
            viscosity=config.viscosity,
            prandtl=config.prandtl,
        )
    return EulerEquation2D(gamma=config.gamma)


def _print_diagnostics(step: int, time: float, dt: float, diagnostics: dict[str, float]) -> None:
    print(
        f"step={step:6d}, t={time:.6f}, dt={dt:.3e}, "
        f"KE={diagnostics['kinetic_energy']:.6e}, "
        f"Mt={diagnostics['turbulent_mach']:.4f}, "
        f"A_K={diagnostics['A_K']:.3f}, "
        f"Cuv={diagnostics['C_uv']:.3f}, "
        f"P_in={diagnostics['P_in']:.3e}, "
        f"A_F={diagnostics['A_F']:.3f}, "
        f"WENO={diagnostics['weno_fraction']:.3f}, "
        f"rho_mean={diagnostics['mean_density']:.6f}, "
        f"p_mean={diagnostics['mean_pressure']:.6f}, "
        f"div_rms={diagnostics['divergence_rms']:.3e}, "
        f"vort_rms={diagnostics['vorticity_rms']:.3e}, "
        f"mass_err={diagnostics['mass_error']:.3e}"
    )


def save_diagnostic_history(
    output_dir: Path,
    history: list[dict[str, float]],
    config: HIT2DConfig,
) -> Path:
    """Persist scalar flow, forcing, and shock-sensor histories."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "diagnostic_history.npz"
    keys = tuple(history[0])
    data = {key: np.asarray([record[key] for record in history]) for key in keys}
    data.update(
        {
            "forcing_kmin": np.asarray(
                0.0 if config.forcing_kmin is None else config.forcing_kmin
            ),
            "forcing_kmax": np.asarray(config.forcing_kmax),
            "forcing_correlation_time": np.asarray(config.forcing_correlation_time),
            "target_energy_injection": np.asarray(
                np.nan
                if config.target_energy_injection is None
                else config.target_energy_injection
            ),
        }
    )
    np.savez_compressed(path, **data)
    return path


def _diagnostic_record(
    step: int,
    time: float,
    dt: float,
    diagnostics: dict[str, float],
) -> dict[str, float]:
    return {
        "step": float(step),
        "time": time,
        "dt": dt,
        **diagnostics,
    }


def _warn_diagnostic_state(
    diagnostics: dict[str, float],
    history: list[dict[str, float]],
    config: HIT2DConfig,
) -> None:
    if not all(np.isfinite(value) for value in diagnostics.values()):
        raise FloatingPointError("NaN or infinite value detected in HIT2D diagnostics")

    if history:
        mean_fxx = float(np.mean([item["Fxx"] for item in history]))
        mean_fyy = float(np.mean([item["Fyy"] for item in history]))
        mean_fxy = float(np.mean([item["Fxy"] for item in history]))
        force_sum = mean_fxx + mean_fyy
        cumulative_af = (
            abs(mean_fxx - mean_fyy) / force_sum if force_sum > 0.0 else 0.0
        )
        normalized_fxy = abs(mean_fxy) / force_sum if force_sum > 0.0 else 0.0
        if cumulative_af > config.forcing_anisotropy_warning:
            print(
                "warning: cumulative forcing anisotropy is high "
                f"(A_F={cumulative_af:.3f})"
            )
        if normalized_fxy > 0.1:
            print(
                "warning: cumulative forcing cross-correlation is high "
                f"(|Fxy|/(Fxx+Fyy)={normalized_fxy:.3f})"
            )

    if diagnostics["A_K"] > config.flow_anisotropy_warning:
        print(f"warning: flow component anisotropy is high (A_K={diagnostics['A_K']:.3f})")
    if len(history) >= 3:
        recent_anisotropy = [item["A_K"] for item in history[-3:]]
        if (
            recent_anisotropy[0] < recent_anisotropy[1] < recent_anisotropy[2]
            and recent_anisotropy[-1] > 0.2
        ):
            print(
                "warning: flow component anisotropy has increased over the "
                "last three diagnostic samples"
            )
    if diagnostics["weno_fraction"] > config.weno_fraction_warning:
        print(
            "warning: WENO is active over a large domain fraction "
            f"({diagnostics['weno_fraction']:.3f})"
        )


def run_simulation(config: HIT2DConfig) -> tuple[np.ndarray, float, int]:
    xp = select_array_module(config.backend)
    print(f"Using array backend: {xp.__name__}")
    domain, _x_grid, _y_grid = create_grid(config)
    equation = _make_equation(config, xp=xp)
    sensor_class = ParallelPeriodicEulerShockSensor2D if xp is not np else PeriodicEulerShockSensor2D
    operator_class = ParallelPeriodicHybridEuler2DOperator if xp is not np else PeriodicHybridEuler2DOperator
    hyperviscosity_class = ParallelPeriodicHyperviscosity2D if xp is not np else PeriodicHyperviscosity2D

    sensor = sensor_class(
        domain,
        equation,
        width=config.sensor_width,
        compression_threshold=config.compression_threshold,
        jump_threshold=config.jump_threshold,
        shear_threshold=config.shear_threshold,
    )
    operator = operator_class(domain, equation, sensor)
    hyperviscosity = hyperviscosity_class(domain, mn=config.hyperviscosity_mn)
    integrator = SSPRK3()
    rng = np.random.default_rng(config.forcing_seed)
    forcing = IsotropicShellOUForcing2D(
        domain=domain,
        k_min=0.0 if config.forcing_kmin is None else config.forcing_kmin,
        k_max=config.forcing_kmax,
        correlation_time=config.forcing_correlation_time,
        force_rms=config.forcing_rms,
        target_power=config.target_energy_injection,
        min_power=config.min_forcing_power,
        max_rescale=config.max_forcing_rescale,
        alpha_memory=config.forcing_alpha_memory,
        seed=config.forcing_seed + 1,
        xp=xp,
    )

    q = equation.enforce_physical_state(initialize_hit_2d(config, domain, equation, rng, xp=xp))
    initial_weno_fraction = float(xp.mean(sensor.detect(q)))
    initial_diagnostics = compute_diagnostics(
        q,
        equation,
        domain,
        weno_fraction=initial_weno_fraction,
    )
    initial_mass = initial_diagnostics["mass"]
    diagnostic_history = [_diagnostic_record(0, 0.0, 0.0, initial_diagnostics)]
    save_snapshot(config.output_dir, 0, 0.0, q, equation, domain)
    save_diagnostic_history(config.output_dir, diagnostic_history, config)
    _print_diagnostics(0, 0.0, 0.0, initial_diagnostics)

    time = 0.0
    step = 0
    forcing_info: dict[str, float] | None = None
    while time < config.tfinal:
        dt = operator.stable_time_step(q, config.cfl, config.tfinal - time)
        if config.viscosity > 0.0:
            h = min(domain.dx, domain.dy)
            dt = min(dt, 0.25 * h**2 / config.viscosity, config.tfinal - time)

        rho, u, v, _pressure = conservative_to_primitive(q, equation)
        fx, fy, forcing_info = forcing.update(dt, rho, u, v)

        def rhs_with_forcing(state: np.ndarray) -> np.ndarray:
            rhs = operator.rhs(state)
            if hasattr(equation, "viscous_rhs"):
                rhs = rhs + equation.viscous_rhs(state, domain.dx, domain.dy)
            return rhs + forcing_source(state, equation, fx, fy)

        q = integrator.step(q, rhs_with_forcing, dt, clean=equation.enforce_physical_state)
        step += 1
        time += dt

        if config.hyperviscosity_mn > 0.0 and step % config.hyperviscosity_interval == 0:
            active_mask = sensor.detect(q) if config.hyperviscosity_on_shocks_only else None
            q = hyperviscosity.apply(q, equation, active_mask=active_mask)

        if step % config.diagnostics_every == 0 or time >= config.tfinal:
            weno_fraction = float(xp.mean(sensor.detect(q)))
            diagnostics = compute_diagnostics(
                q,
                equation,
                domain,
                initial_mass=initial_mass,
                forcing_info=forcing_info,
                weno_fraction=weno_fraction,
            )
            diagnostic_history.append(_diagnostic_record(step, time, dt, diagnostics))
            _warn_diagnostic_state(diagnostics, diagnostic_history[1:], config)
            save_diagnostic_history(config.output_dir, diagnostic_history, config)
            _print_diagnostics(step, time, dt, diagnostics)

        if step % config.snapshot_every == 0 or time >= config.tfinal:
            path = save_snapshot(config.output_dir, step, time, q, equation, domain)
            print(f"saved snapshot: {path}")

    return q, time, step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preliminary 2D forced compressible HIT setup.")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--tfinal", type=float, default=1.0)
    parser.add_argument("--cfl", type=float, default=0.05)
    parser.add_argument("--mach", type=float, default=0.1, help="Initial turbulent Mach number.")
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--viscosity", type=float, default=1.0e-3)
    parser.add_argument(
        "--kf",
        type=float,
        default=3.0,
        help="Legacy forcing cutoff. Used as kf-max when --kf-max is omitted.",
    )
    parser.add_argument("--kf-min", type=float, default=None, help="Minimum forced shell wavenumber.")
    parser.add_argument("--kf-max", type=float, default=None, help="Maximum forced shell wavenumber.")
    parser.add_argument(
        "--p-target",
        "--pget",
        dest="p_target",
        type=float,
        default=1.0e-3,
        help="Target mean kinetic-energy injection.",
    )
    parser.add_argument("--force-rms", type=float, default=1.0)
    parser.add_argument("--forcing-correlation-time", type=float, default=1.0)
    parser.add_argument(
        "--forcing-alpha-memory",
        type=float,
        default=0.2,
        help="Memory used to smooth the scalar power-rescaling coefficient.",
    )
    parser.add_argument(
        "--min-forcing-power",
        type=float,
        default=1.0e-6,
        help="Skip exact forcing rescale when |P_current| is below this value.",
    )
    parser.add_argument(
        "--max-forcing-rescale",
        type=float,
        default=20.0,
        help="Cap |P_target / P_current| to avoid impulsive random forcing. Use 0 to disable the cap.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mn", type=float, default=0.002, help="Periodic hyperviscosity strength.")
    parser.add_argument("--diagnostics-every", type=int, default=25)
    parser.add_argument("--snapshot-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument(
        "--backend",
        choices=("auto", "numpy", "cupy"),
        default="numpy",
        help="Array backend for grid-wide HIT operations. Use 'cupy' explicitly for GPU.",
    )
    parser.add_argument("--no-forcing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = HIT2DConfig(
        nx=args.nx,
        ny=args.ny,
        tfinal=args.tfinal,
        cfl=args.cfl,
        target_mach=args.mach,
        gamma=args.gamma,
        viscosity=args.viscosity,
        forcing_kmin=args.kf_min,
        forcing_kmax=args.kf if args.kf_max is None else args.kf_max,
        forcing_rms=0.0 if args.no_forcing else args.force_rms,
        forcing_correlation_time=args.forcing_correlation_time,
        forcing_alpha_memory=args.forcing_alpha_memory,
        target_energy_injection=None if args.no_forcing else args.p_target,
        min_forcing_power=args.min_forcing_power,
        max_forcing_rescale=None if args.max_forcing_rescale == 0.0 else args.max_forcing_rescale,
        forcing_seed=args.seed,
        hyperviscosity_mn=args.mn,
        diagnostics_every=args.diagnostics_every,
        snapshot_every=args.snapshot_every,
        output_dir=args.output_dir,
        backend=args.backend,
    )
    _q_final, time, steps = run_simulation(config)
    print(f"completed 2D HIT run at t={time:.6f} in {steps} steps")


if __name__ == "__main__":
    main()
