from __future__ import annotations

from pathlib import Path
import argparse
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OOP.hit2d import HIT2DConfig, run_simulation
from OOP.postprocess.turnover import cumulative_turnover, infer_turnover_length


DEFAULT_SNAPSHOT_DIR = Path(__file__).resolve().parent / "hit2d_snapshots"


def run_id_from_snapshot_dir(snapshot_dir: Path) -> str | None:
    name = snapshot_dir.name
    if not name.startswith("run_"):
        return None
    return name.removeprefix("run_")


def create_timestamped_run_dir(base_dir: Path) -> Path:
    """Create a fresh output folder for one simulation run."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base_dir / f"run_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = base_dir / f"run_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def default_postprocess_dir(run_dir: Path) -> Path:
    """Default location for figures and animations belonging to one run."""

    return run_dir / "postprocess"


def find_snapshots(snapshot_dir: Path, stride: int = 1) -> list[Path]:
    snapshots = sorted(snapshot_dir.glob("hit2d_step*.npz"))
    if not snapshots:
        raise FileNotFoundError(f"No HIT snapshots found in {snapshot_dir}")
    return snapshots[:: max(stride, 1)]


def resolve_snapshot_dir(snapshot_dir: Path) -> Path:
    """Use snapshot_dir, or its newest child run folder when snapshots live there."""

    if not snapshot_dir.exists():
        return snapshot_dir
    if list(snapshot_dir.glob("hit2d_step*.npz")):
        return snapshot_dir

    child_dirs = sorted(
        (path for path in snapshot_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for child_dir in child_dirs:
        if list(child_dir.glob("hit2d_step*.npz")):
            return child_dir
    return snapshot_dir


def find_latest_snapshot(snapshot_dir: Path) -> Path:
    return find_snapshots(snapshot_dir)[-1]


def _extent_from_centers(x: np.ndarray, y: np.ndarray) -> list[float]:
    dx = float(x[1] - x[0]) if x.size > 1 else 1.0
    dy = float(y[1] - y[0]) if y.size > 1 else 1.0
    return [
        float(x[0] - 0.5 * dx),
        float(x[-1] + 0.5 * dx),
        float(y[0] - 0.5 * dy),
        float(y[-1] + 0.5 * dy),
    ]


def plot_hit2d_fields(
    snapshot_path: Path,
    output_path: Path,
    show: bool = False,
    cmap: str = "viridis",
    velocity_cmap: str = "coolwarm",
) -> Path:
    """Plot density, pressure, x-velocity, and y-velocity from one HIT snapshot."""

    with np.load(snapshot_path) as data:
        x = data["x"]
        y = data["y"]
        rho = data["rho"]
        pressure = data["pressure"]
        u = data["u"]
        v = data["v"]
        time = float(data["time"])
        step = int(data["step"])

    fields = [
        (rho, "Density", "rho", cmap, None, None),
        (pressure, "Pressure", "p", cmap, None, None),
        (u, "X-velocity", "u", velocity_cmap, -np.max(np.abs(u)), np.max(np.abs(u))),
        (v, "Y-velocity", "v", velocity_cmap, -np.max(np.abs(v)), np.max(np.abs(v))),
    ]
    extent = _extent_from_centers(x, y)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0), constrained_layout=True)
    fig.suptitle(f"2D Forced Compressible HIT, step={step}, t={time:.6f}", fontsize=13)

    for ax, (values, title, label, field_cmap, vmin, vmax) in zip(axes.ravel(), fields):
        image = ax.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=field_cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, label=label)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
    except PermissionError:
        fallback_path = snapshot_path.with_name(output_path.name)
        fig.savefig(fallback_path, bbox_inches="tight", dpi=200)
        print(f"could not write to {output_path}; saved next to snapshot instead")
        output_path = fallback_path
    print(f"saved HIT field plot: {output_path}")
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)
    return output_path


def _symmetric_limits(values: np.ndarray) -> tuple[float, float]:
    max_abs = float(np.nanmax(np.abs(values)))
    if max_abs <= 0.0 or not np.isfinite(max_abs):
        max_abs = 1.0
    return -max_abs, max_abs


def _positive_limits(values: np.ndarray) -> tuple[float, float]:
    max_value = float(np.nanmax(values))
    if max_value <= 0.0 or not np.isfinite(max_value):
        max_value = 1.0
    return 0.0, max_value


def plot_hit2d_physics_fields(
    snapshot_path: Path,
    output_path: Path,
    gamma: float = 1.4,
    show: bool = False,
) -> Path:
    """Plot derived flow diagnostics from one HIT snapshot.

    Vorticity highlights rotational turbulent structures. Divergence shows
    compressive/dilatational regions. Local Mach number shows where the flow is
    locally closer to acoustic or shocklet-like behavior. Density fluctuation
    rho' removes the mean density so small compressible fluctuations are easier
    to see.
    """

    with np.load(snapshot_path) as data:
        x = data["x"]
        y = data["y"]
        rho = data["rho"]
        pressure = data["pressure"]
        u = data["u"]
        v = data["v"]
        vorticity = data["vorticity"]
        divergence = data["divergence"]
        time = float(data["time"])
        step = int(data["step"])

    derived_fields = _physics_fields_from_arrays(
        rho,
        pressure,
        u,
        v,
        vorticity,
        divergence,
        gamma,
    )

    fields = [
        (derived_fields["vorticity"], "Vorticity", "omega_z", "coolwarm", *_symmetric_limits(derived_fields["vorticity"])),
        (derived_fields["divergence"], "Divergence", "nabla . u", "coolwarm", *_symmetric_limits(derived_fields["divergence"])),
        (derived_fields["local_mach"], "Local Mach", "M(x,y)", "magma", *_positive_limits(derived_fields["local_mach"])),
        (derived_fields["rho_prime"], "Density fluctuation", "rho'", "coolwarm", *_symmetric_limits(derived_fields["rho_prime"])),
    ]
    extent = _extent_from_centers(x, y)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0), constrained_layout=True)
    fig.suptitle(f"2D Forced Compressible HIT diagnostics, step={step}, t={time:.6f}", fontsize=13)

    for ax, (values, title, label, field_cmap, vmin, vmax) in zip(axes.ravel(), fields):
        image = ax.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=field_cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, label=label)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
    except PermissionError:
        fallback_path = snapshot_path.with_name(output_path.name)
        fig.savefig(fallback_path, bbox_inches="tight", dpi=200)
        print(f"could not write to {output_path}; saved next to snapshot instead")
        output_path = fallback_path
    print(f"saved HIT diagnostic field plot: {output_path}")
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)
    return output_path


def _physics_fields_from_arrays(
    rho: np.ndarray,
    pressure: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    vorticity: np.ndarray,
    divergence: np.ndarray,
    gamma: float,
) -> dict[str, np.ndarray]:
    sound_speed = np.sqrt(np.maximum(gamma * pressure / rho, np.finfo(float).tiny))
    return {
        "vorticity": vorticity,
        "divergence": divergence,
        "local_mach": np.sqrt(u**2 + v**2) / sound_speed,
        "rho_prime": rho - float(np.mean(rho)),
    }


def _snapshot_cell_area(x: np.ndarray, y: np.ndarray) -> float:
    dx = float(x[1] - x[0]) if x.size > 1 else 1.0
    dy = float(y[1] - y[0]) if y.size > 1 else 1.0
    return dx * dy


def compute_hit2d_history(
    snapshot_dir: Path,
    gamma: float = 1.4,
    turnover_length: float | None = None,
) -> dict[str, np.ndarray]:
    """Compute scalar histories, preferring the dense diagnostic-history file."""

    diagnostic_path = snapshot_dir / "diagnostic_history.npz"
    if diagnostic_path.exists():
        with np.load(diagnostic_path) as data:
            history = {
                key: np.asarray(data[key], dtype=float)
                for key in data.files
                if np.asarray(data[key]).ndim == 1
                and np.asarray(data[key]).size == np.asarray(data["time"]).size
            }
        required = {
            "time",
            "kinetic_energy",
            "turbulent_mach",
            "divergence_rms",
            "vorticity_rms",
            "mean_pressure",
            "mass_error",
            "rms_velocity",
        }
        if required.issubset(history):
            history["u_rms"] = history["rms_velocity"]
            length_ref = (
                infer_turnover_length(snapshot_dir)
                if turnover_length is None
                else turnover_length
            )
            history["turnover"] = cumulative_turnover(
                history["time"], history["u_rms"], length_ref
            )
            history["turnover_length"] = np.asarray(length_ref)
            return history

    snapshots = find_snapshots(snapshot_dir)
    history_lists: dict[str, list[float]] = {
        "time": [],
        "kinetic_energy": [],
        "turbulent_mach": [],
        "mach_control_filtered_mach": [],
        "mach_control_target": [],
        "divergence_rms": [],
        "vorticity_rms": [],
        "mean_pressure": [],
        "mass_error": [],
        "u_rms": [],
    }
    initial_mass: float | None = None

    for snapshot in snapshots:
        with np.load(snapshot) as data:
            x = data["x"]
            y = data["y"]
            rho = data["rho"]
            pressure = data["pressure"]
            u = data["u"]
            v = data["v"]
            divergence = data["divergence"]
            vorticity = data["vorticity"]
            time = float(data["time"])

        speed_sq = u**2 + v**2
        sound_speed = np.sqrt(np.maximum(gamma * pressure / rho, np.finfo(float).tiny))
        cell_area = _snapshot_cell_area(x, y)
        mass = float(np.sum(rho) * cell_area)
        if initial_mass is None:
            initial_mass = mass

        u_rms = float(np.sqrt(np.mean(speed_sq)))
        mt = float(u_rms / np.mean(sound_speed))
        history_lists["time"].append(time)
        history_lists["kinetic_energy"].append(float(0.5 * np.mean(rho * speed_sq)))
        history_lists["u_rms"].append(u_rms)
        history_lists["turbulent_mach"].append(mt)
        history_lists["mach_control_filtered_mach"].append(mt)
        history_lists["mach_control_target"].append(0.0)
        history_lists["divergence_rms"].append(float(np.sqrt(np.mean(divergence**2))))
        history_lists["vorticity_rms"].append(float(np.sqrt(np.mean(vorticity**2))))
        history_lists["mean_pressure"].append(float(np.mean(pressure)))
        history_lists["mass_error"].append(float(mass - initial_mass))

    history = {name: np.asarray(values) for name, values in history_lists.items()}
    length_ref = infer_turnover_length(snapshot_dir) if turnover_length is None else turnover_length
    history["turnover"] = cumulative_turnover(history["time"], history["u_rms"], length_ref)
    history["turnover_length"] = np.asarray(length_ref)
    return history


def plot_hit2d_history(
    snapshot_dir: Path,
    output_path: Path,
    gamma: float = 1.4,
    x_axis: str = "turnover",
    turnover_length: float | None = None,
    show: bool = False,
) -> Path:
    """Plot the main scalar HIT diagnostics as time histories."""

    history = compute_hit2d_history(
        snapshot_dir,
        gamma=gamma,
        turnover_length=turnover_length,
    )
    if x_axis == "turnover":
        x_values = history["turnover"]
        x_label = r"$N_{eddy}=\int u_{rms}/L_{ref}\,dt$"
    elif x_axis == "time":
        x_values = history["time"]
        x_label = "t"
    else:
        raise ValueError("x_axis must be 'turnover' or 'time'")
    panels = [
        ("kinetic_energy", "Kinetic energy", "K"),
        ("turbulent_mach", "Turbulent Mach", "Mt"),
        ("divergence_rms", "Divergence RMS", "div_rms"),
        ("vorticity_rms", "Vorticity RMS", "vort_rms"),
        ("mean_pressure", "Mean pressure", "<p>"),
        ("mass_error", "Mass error", "M - M0"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12.0, 10.5), constrained_layout=True)
    fig.suptitle("2D Forced Compressible HIT time histories", fontsize=13)

    for ax, (name, title, ylabel) in zip(axes.ravel(), panels):
        ax.plot(x_values, history[name], linewidth=1.5)
        if name == "turbulent_mach":
            filtered = history.get("mach_control_filtered_mach")
            target = history.get("mach_control_target")
            if filtered is not None and filtered.size == x_values.size:
                ax.plot(x_values, filtered, linewidth=2.0, label="controller-filtered")
            if target is not None and target.size == x_values.size and np.any(target > 0.0):
                ax.plot(x_values, target, linestyle="--", linewidth=1.4, label="target")
            ax.legend()
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
    except PermissionError:
        fallback_path = snapshot_dir / output_path.name
        fig.savefig(fallback_path, bbox_inches="tight", dpi=200)
        print(f"could not write to {output_path}; saved in snapshot directory instead")
        output_path = fallback_path
    print(f"saved HIT history plot: {output_path}")
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)
    return output_path



def plot_hit2d_forcing_history(
    snapshot_dir: Path,
    output_path: Path,
    x_axis: str = "turnover",
    turnover_length: float | None = None,
    show: bool = False,
) -> Path | None:
    """Plot the smooth forcing and turbulent-Mach controller diagnostics."""

    history = compute_hit2d_history(snapshot_dir, turnover_length=turnover_length)
    required = {
        "turbulent_mach",
        "mach_control_filtered_mach",
        "mach_control_target",
        "P_in",
        "forcing_target_power",
        "forcing_alpha",
        "forcing_power_before_filtered",
        "mach_control_log_power_rate",
    }
    if not required.issubset(history):
        return None
    x_values = history["turnover"] if x_axis == "turnover" else history["time"]
    x_label = r"$N_{eddy}$" if x_axis == "turnover" else "t"

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    axes[0, 0].plot(x_values, history["turbulent_mach"])
    axes[0, 0].plot(
        x_values,
        history["mach_control_filtered_mach"],
        linewidth=2.0,
        label="controller-filtered",
    )
    if np.any(history["mach_control_target"] > 0.0):
        axes[0, 0].plot(
            x_values,
            history["mach_control_target"],
            linestyle="--",
            label="target",
        )
    axes[0, 0].set_title("Turbulent Mach control")
    axes[0, 0].set_ylabel(r"$M_t$")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(x_values, history["P_in"], label="instantaneous injected power")
    axes[0, 1].plot(
        x_values,
        history["forcing_target_power"],
        linestyle="--",
        label="requested mean power",
    )
    axes[0, 1].set_title("Energy injection")
    axes[0, 1].set_ylabel("power per unit volume")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(x_values, history["forcing_alpha"], label=r"$\alpha$")
    if "forcing_alpha_target" in history:
        axes[1, 0].plot(
            x_values,
            history["forcing_alpha_target"],
            linestyle="--",
            label=r"$\alpha_{target}$",
        )
    axes[1, 0].set_title("Forcing amplitude controller")
    axes[1, 0].set_ylabel("rescaling coefficient")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        x_values,
        history["forcing_power_before_filtered"],
        label="filtered raw correlation",
    )
    axes[1, 1].plot(
        x_values,
        history["mach_control_log_power_rate"],
        label=r"$d(\log P_{target})/dt$",
    )
    axes[1, 1].set_title("Controller response signals")
    axes[1, 1].legend(fontsize=8)

    for ax in axes.ravel():
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)
    print(f"saved HIT forcing-controller history: {output_path}")
    return output_path

def plot_hit2d_dns_history(
    snapshot_dir: Path,
    output_path: Path,
    x_axis: str = "turnover",
    turnover_length: float | None = None,
    show: bool = False,
) -> Path | None:
    """Plot resolution and dissipation diagnostics used to assess DNS quality."""

    history = compute_hit2d_history(snapshot_dir, turnover_length=turnover_length)
    required = {
        "re_lambda_2d",
        "eta_over_dx",
        "kmax_eta",
        "epsilon_physical",
        "dilatational_energy_fraction",
        "weno_fraction",
        "hyperviscosity_power",
    }
    if not required.issubset(history):
        return None
    x_values = history["turnover"] if x_axis == "turnover" else history["time"]
    x_label = r"$N_{eddy}$" if x_axis == "turnover" else "t"

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    axes[0, 0].plot(x_values, history["re_lambda_2d"])
    axes[0, 0].set_ylabel(r"$Re_{\lambda,2D}$")
    axes[0, 0].set_title("Taylor-microscale Reynolds number")

    axes[0, 1].plot(x_values, history["eta_over_dx"], label=r"$\eta_K/\Delta$")
    if "kraichnan_over_dx" in history:
        axes[0, 1].plot(
            x_values,
            history["kraichnan_over_dx"],
            label=r"$\eta_\Omega/\Delta$ (2-D)",
        )
    axes[0, 1].axhline(0.5, linestyle="--", linewidth=1.0, label=r"reference $0.5$")
    axes[0, 1].set_title("Nominal small-scale resolution")
    axes[0, 1].set_ylabel("microscale / grid spacing")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(x_values, history["epsilon_physical"], label="physical viscous")
    axes[1, 0].plot(
        x_values,
        -history["hyperviscosity_power"],
        label="numerical hyperviscosity drain",
    )
    axes[1, 0].set_title("Dissipation-rate indicators")
    axes[1, 0].set_ylabel("power per unit volume")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        x_values,
        history["dilatational_energy_fraction"],
        label=r"$\chi_d=K_d/(K_s+K_d)$",
    )
    axes[1, 1].plot(x_values, history["weno_fraction"], label="WENO fraction")
    axes[1, 1].set_title("Compressibility and shock-capturing activity")
    axes[1, 1].legend(fontsize=8)

    for ax in axes.ravel():
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)
    print(f"saved HIT DNS-resolution history: {output_path}")
    return output_path


def _field_limits(
    snapshots: list[Path],
    mode: str = "robust",
    percentile: float = 99.0,
    until_time: float | None = None,
) -> dict[str, tuple[float, float]]:
    limit_snapshots = snapshots
    if until_time is not None:
        selected = []
        for snapshot in snapshots:
            with np.load(snapshot) as data:
                if float(data["time"]) <= until_time:
                    selected.append(snapshot)
        if selected:
            limit_snapshots = selected

    if mode == "robust":
        rho_values = []
        pressure_values = []
        u_abs_values = []
        v_abs_values = []
        for snapshot in limit_snapshots:
            with np.load(snapshot) as data:
                rho_values.append(data["rho"].ravel())
                pressure_values.append(data["pressure"].ravel())
                u_abs_values.append(np.abs(data["u"]).ravel())
                v_abs_values.append(np.abs(data["v"]).ravel())

        lower = 100.0 - percentile
        rho_all = np.concatenate(rho_values)
        pressure_all = np.concatenate(pressure_values)
        u_abs_all = np.concatenate(u_abs_values)
        v_abs_all = np.concatenate(v_abs_values)
        return {
            "rho": (
                float(np.percentile(rho_all, lower)),
                float(np.percentile(rho_all, percentile)),
            ),
            "pressure": (
                float(np.percentile(pressure_all, lower)),
                float(np.percentile(pressure_all, percentile)),
            ),
            "u": (
                -float(np.percentile(u_abs_all, percentile)),
                float(np.percentile(u_abs_all, percentile)),
            ),
            "v": (
                -float(np.percentile(v_abs_all, percentile)),
                float(np.percentile(v_abs_all, percentile)),
            ),
        }

    limits = {
        "rho": [np.inf, -np.inf],
        "pressure": [np.inf, -np.inf],
        "u_abs": [0.0, 0.0],
        "v_abs": [0.0, 0.0],
    }
    for snapshot in limit_snapshots:
        with np.load(snapshot) as data:
            rho = data["rho"]
            pressure = data["pressure"]
            u = data["u"]
            v = data["v"]
        limits["rho"][0] = min(limits["rho"][0], float(np.min(rho)))
        limits["rho"][1] = max(limits["rho"][1], float(np.max(rho)))
        limits["pressure"][0] = min(limits["pressure"][0], float(np.min(pressure)))
        limits["pressure"][1] = max(limits["pressure"][1], float(np.max(pressure)))
        limits["u_abs"][1] = max(limits["u_abs"][1], float(np.max(np.abs(u))))
        limits["v_abs"][1] = max(limits["v_abs"][1], float(np.max(np.abs(v))))

    return {
        "rho": (limits["rho"][0], limits["rho"][1]),
        "pressure": (limits["pressure"][0], limits["pressure"][1]),
        "u": (-limits["u_abs"][1], limits["u_abs"][1]),
        "v": (-limits["v_abs"][1], limits["v_abs"][1]),
    }


def _physics_field_limits(
    snapshots: list[Path],
    mode: str = "robust",
    percentile: float = 99.0,
    until_time: float | None = None,
    gamma: float = 1.4,
) -> dict[str, tuple[float, float]]:
    limit_snapshots = snapshots
    if until_time is not None:
        selected = []
        for snapshot in snapshots:
            with np.load(snapshot) as data:
                if float(data["time"]) <= until_time:
                    selected.append(snapshot)
        if selected:
            limit_snapshots = selected

    signed_fields = ("vorticity", "divergence", "rho_prime")
    if mode == "robust":
        signed_values = {name: [] for name in signed_fields}
        mach_values = []
        for snapshot in limit_snapshots:
            with np.load(snapshot) as data:
                fields = _physics_fields_from_arrays(
                    data["rho"],
                    data["pressure"],
                    data["u"],
                    data["v"],
                    data["vorticity"],
                    data["divergence"],
                    gamma,
                )
            for name in signed_fields:
                signed_values[name].append(np.abs(fields[name]).ravel())
            mach_values.append(fields["local_mach"].ravel())

        limits = {}
        for name in signed_fields:
            max_abs = float(np.percentile(np.concatenate(signed_values[name]), percentile))
            if max_abs <= 0.0 or not np.isfinite(max_abs):
                max_abs = 1.0
            limits[name] = (-max_abs, max_abs)

        mach_max = float(np.percentile(np.concatenate(mach_values), percentile))
        if mach_max <= 0.0 or not np.isfinite(mach_max):
            mach_max = 1.0
        limits["local_mach"] = (0.0, mach_max)
        return limits

    limits = {name: [0.0, 0.0] for name in signed_fields}
    mach_max = 0.0
    for snapshot in limit_snapshots:
        with np.load(snapshot) as data:
            fields = _physics_fields_from_arrays(
                data["rho"],
                data["pressure"],
                data["u"],
                data["v"],
                data["vorticity"],
                data["divergence"],
                gamma,
            )
        for name in signed_fields:
            limits[name][1] = max(limits[name][1], float(np.max(np.abs(fields[name]))))
        mach_max = max(mach_max, float(np.max(fields["local_mach"])))

    output_limits = {}
    for name in signed_fields:
        max_abs = limits[name][1]
        if max_abs <= 0.0 or not np.isfinite(max_abs):
            max_abs = 1.0
        output_limits[name] = (-max_abs, max_abs)
    if mach_max <= 0.0 or not np.isfinite(mach_max):
        mach_max = 1.0
    output_limits["local_mach"] = (0.0, mach_max)
    return output_limits


def _render_current_figure(fig: plt.Figure) -> Image.Image:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(rgba[:, :, :3].copy())


def _make_palette_image(frames: list[Image.Image]) -> Image.Image:
    """Build one GIF palette from thumbnail samples of all rendered frames."""

    resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    adaptive = Image.Palette.ADAPTIVE if hasattr(Image, "Palette") else Image.ADAPTIVE
    cols = min(4, len(frames))
    rows = int(np.ceil(len(frames) / cols))
    thumbnails: list[Image.Image] = []

    for frame in frames:
        thumbnail = frame.copy()
        thumbnail.thumbnail((360, 300), resampling)
        thumbnails.append(thumbnail)

    thumb_width = max(thumbnail.width for thumbnail in thumbnails)
    thumb_height = max(thumbnail.height for thumbnail in thumbnails)
    palette_source = Image.new("RGB", (cols * thumb_width, rows * thumb_height), "white")
    for index, thumbnail in enumerate(thumbnails):
        x_offset = (index % cols) * thumb_width
        y_offset = (index // cols) * thumb_height
        palette_source.paste(thumbnail, (x_offset, y_offset))

    return palette_source.convert("P", palette=adaptive, colors=256)


def _save_fixed_palette_gif(frames: list[Image.Image], output_path: Path, fps: int) -> None:
    """Save a GIF whose frames all share the same 256-color palette."""

    dither_none = Image.Dither.NONE if hasattr(Image, "Dither") else Image.NONE
    palette_image = _make_palette_image(frames)
    quantized_frames = [
        frame.quantize(palette=palette_image, dither=dither_none)
        for frame in frames
    ]
    duration_ms = max(1, int(round(1000 / fps)))
    quantized_frames[0].save(
        output_path,
        save_all=True,
        append_images=quantized_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def animate_hit2d_snapshots(
    snapshot_dir: Path,
    output_path: Path,
    fps: int = 6,
    stride: int = 1,
    limit_mode: str = "robust",
    robust_percentile: float = 99.0,
    limit_until_time: float | None = None,
    cmap: str = "viridis",
    velocity_cmap: str = "coolwarm",
) -> Path:
    """Animate all saved HIT snapshots as density/pressure/u/v panels."""

    snapshots = find_snapshots(snapshot_dir, stride=stride)
    limits = _field_limits(
        snapshots,
        mode=limit_mode,
        percentile=robust_percentile,
        until_time=limit_until_time,
    )
    with np.load(snapshots[0]) as data:
        x = data["x"]
        y = data["y"]
        initial_fields = {
            "rho": data["rho"],
            "pressure": data["pressure"],
            "u": data["u"],
            "v": data["v"],
        }
        initial_step = int(data["step"])
        initial_time = float(data["time"])

    extent = _extent_from_centers(x, y)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0), constrained_layout=True)
    title = fig.suptitle(
        f"2D Forced Compressible HIT, step={initial_step}, t={initial_time:.6f}",
        fontsize=13,
    )

    image_specs = [
        ("rho", "Density", "rho", cmap),
        ("pressure", "Pressure", "p", cmap),
        ("u", "X-velocity", "u", velocity_cmap),
        ("v", "Y-velocity", "v", velocity_cmap),
    ]
    images = {}
    for ax, (name, panel_title, label, field_cmap) in zip(axes.ravel(), image_specs):
        image = ax.imshow(
            initial_fields[name],
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=field_cmap,
            vmin=limits[name][0],
            vmax=limits[name][1],
        )
        ax.set_title(panel_title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, label=label)
        images[name] = image

    def update(frame_index: int) -> None:
        snapshot = snapshots[frame_index]
        with np.load(snapshot) as data:
            images["rho"].set_data(data["rho"])
            images["pressure"].set_data(data["pressure"])
            images["u"].set_data(data["u"])
            images["v"].set_data(data["v"])
            step = int(data["step"])
            time = float(data["time"])
        title.set_text(f"2D Forced Compressible HIT, step={step}, t={time:.6f}")

    rendered_frames = []
    for frame_index in range(len(snapshots)):
        update(frame_index)
        rendered_frames.append(_render_current_figure(fig))

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _save_fixed_palette_gif(rendered_frames, output_path, fps=fps)
    except PermissionError:
        fallback_path = snapshot_dir / output_path.name
        _save_fixed_palette_gif(rendered_frames, fallback_path, fps=fps)
        print(f"could not write to {output_path}; saved in snapshot directory instead")
        output_path = fallback_path
    plt.close(fig)
    print(f"saved HIT animation: {output_path}")
    return output_path


def animate_hit2d_physics_snapshots(
    snapshot_dir: Path,
    output_path: Path,
    fps: int = 6,
    stride: int = 1,
    limit_mode: str = "robust",
    robust_percentile: float = 99.0,
    limit_until_time: float | None = None,
    gamma: float = 1.4,
) -> Path:
    """Animate vorticity, divergence, local Mach number, and rho' panels."""

    snapshots = find_snapshots(snapshot_dir, stride=stride)
    limits = _physics_field_limits(
        snapshots,
        mode=limit_mode,
        percentile=robust_percentile,
        until_time=limit_until_time,
        gamma=gamma,
    )
    with np.load(snapshots[0]) as data:
        x = data["x"]
        y = data["y"]
        initial_fields = _physics_fields_from_arrays(
            data["rho"],
            data["pressure"],
            data["u"],
            data["v"],
            data["vorticity"],
            data["divergence"],
            gamma,
        )
        initial_step = int(data["step"])
        initial_time = float(data["time"])

    extent = _extent_from_centers(x, y)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0), constrained_layout=True)
    title = fig.suptitle(
        f"2D Forced Compressible HIT diagnostics, step={initial_step}, t={initial_time:.6f}",
        fontsize=13,
    )

    image_specs = [
        ("vorticity", "Vorticity", "omega_z", "coolwarm"),
        ("divergence", "Divergence", "nabla . u", "coolwarm"),
        ("local_mach", "Local Mach", "M(x,y)", "magma"),
        ("rho_prime", "Density fluctuation", "rho'", "coolwarm"),
    ]
    images = {}
    for ax, (name, panel_title, label, field_cmap) in zip(axes.ravel(), image_specs):
        image = ax.imshow(
            initial_fields[name],
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=field_cmap,
            vmin=limits[name][0],
            vmax=limits[name][1],
        )
        ax.set_title(panel_title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, label=label)
        images[name] = image

    def update(frame_index: int) -> None:
        snapshot = snapshots[frame_index]
        with np.load(snapshot) as data:
            fields = _physics_fields_from_arrays(
                data["rho"],
                data["pressure"],
                data["u"],
                data["v"],
                data["vorticity"],
                data["divergence"],
                gamma,
            )
            step = int(data["step"])
            time = float(data["time"])
        for name in images:
            images[name].set_data(fields[name])
        title.set_text(
            f"2D Forced Compressible HIT diagnostics, step={step}, t={time:.6f}"
        )

    rendered_frames = []
    for frame_index in range(len(snapshots)):
        update(frame_index)
        rendered_frames.append(_render_current_figure(fig))

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _save_fixed_palette_gif(rendered_frames, output_path, fps=fps)
    except PermissionError:
        fallback_path = snapshot_dir / output_path.name
        _save_fixed_palette_gif(rendered_frames, fallback_path, fps=fps)
        print(f"could not write to {output_path}; saved in snapshot directory instead")
        output_path = fallback_path
    plt.close(fig)
    print(f"saved HIT physics animation: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or view the preliminary 2D HIT scenario and plot primitive fields."
    )
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--snapshot", type=Path, default=None, help="Specific .npz snapshot to plot.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path. Defaults to <run-dir>/postprocess/hit2d_fields.png, or hit2d_final_fields.png with --animate.",
    )
    parser.add_argument(
        "--physics-plots",
        action="store_true",
        help="Save derived-field and scalar time-history diagnostic PNGs.",
    )
    parser.add_argument(
        "--physics-output",
        type=Path,
        default=None,
        help="PNG output path for vorticity/divergence/local-Mach/rho' panels. Defaults to <run-dir>/postprocess.",
    )
    parser.add_argument(
        "--history-output",
        type=Path,
        default=None,
        help="PNG output path for K, Mt, divergence, vorticity, pressure, and mass histories. Defaults to <run-dir>/postprocess.",
    )
    parser.add_argument(
        "--history-x-axis",
        choices=("turnover", "time"),
        default="turnover",
        help="X-axis for scalar history plots.",
    )
    parser.add_argument(
        "--turnover-length",
        type=float,
        default=None,
        help="Reference length for turnover time. Defaults to 2*pi/k_shell_center.",
    )
    parser.add_argument("--show", action="store_true", help="Open an interactive Matplotlib window.")
    parser.add_argument("--animate", action="store_true", help="Animate all snapshots instead of plotting one.")
    parser.add_argument(
        "--physics-animate",
        action="store_true",
        help="Animate vorticity/divergence/local-Mach/rho' panels.",
    )
    parser.add_argument(
        "--animation-output",
        type=Path,
        default=None,
        help="GIF output path. Defaults to <run-dir>/postprocess/hit2d_simulation.gif.",
    )
    parser.add_argument(
        "--physics-animation-output",
        type=Path,
        default=None,
        help="GIF output path for the derived physics-field animation. Defaults to <run-dir>/postprocess.",
    )
    parser.add_argument("--fps", type=int, default=6, help="Frames per second for GIF output.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Use every Nth snapshot in the GIF.")
    parser.add_argument(
        "--limit-mode",
        choices=("robust", "global"),
        default="robust",
        help="Color limits for animations. 'robust' clips extreme tails; 'global' uses exact min/max.",
    )
    parser.add_argument(
        "--robust-percentile",
        type=float,
        default=99.0,
        help="Upper percentile for robust fixed color limits; lower limit uses 100 - this value.",
    )
    parser.add_argument(
        "--limit-until-time",
        type=float,
        default=None,
        help="Compute fixed animation color limits using only snapshots with time <= this value.",
    )
    parser.add_argument("--run", action="store_true", help="Run the HIT scenario before plotting.")
    parser.add_argument(
        "--no-timestamp-dir",
        action="store_true",
        help="When using --run, write directly into --snapshot-dir instead of a fresh timestamped child folder.",
    )
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--tfinal", type=float, default=1.0)
    parser.add_argument("--cfl", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--mach", type=float, default=0.1)
    parser.add_argument("--initial-kmin", type=int, default=1)
    parser.add_argument("--initial-kmax", type=int, default=3)
    parser.add_argument("--viscosity", type=float, default=1.0e-3)
    parser.add_argument("--initial-re-lambda", type=float, default=None)
    parser.add_argument(
        "--kf",
        type=float,
        default=3.0,
        help="Legacy forcing cutoff. Used as kf-max when --kf-max is omitted.",
    )
    parser.add_argument("--kf-min", type=float, default=None)
    parser.add_argument("--kf-max", type=float, default=None)
    parser.add_argument(
        "--p-target",
        "--pget",
        dest="p_target",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument("--forcing-correlation-time", type=float, default=1.0)
    parser.add_argument("--forcing-alpha-memory", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-forcing-power", type=float, default=1.0e-6)
    parser.add_argument("--max-forcing-rescale", type=float, default=20.0)
    parser.add_argument(
        "--mach-control",
        action="store_true",
        help="Slowly adapt --p-target so the turbulent Mach number stays near the requested target.",
    )
    parser.add_argument(
        "--mach-control-target",
        type=float,
        default=None,
        help="Target turbulent Mach for feedback control. Defaults to --mach.",
    )
    parser.add_argument(
        "--mach-control-memory",
        type=float,
        default=0.995,
        help="Memory for the adaptive power target. Larger values change the power more slowly.",
    )
    parser.add_argument(
        "--mach-control-exponent",
        type=float,
        default=2.0,
        help="Exponent in the power correction (Mt_target/Mt)^exponent.",
    )
    parser.add_argument(
        "--mach-control-min-power",
        type=float,
        default=0.0,
        help="Lower bound for adaptive target power. Use 0 for no explicit lower bound.",
    )
    parser.add_argument(
        "--mach-control-max-power",
        type=float,
        default=0.0,
        help="Upper bound for adaptive target power. Use 0 for no explicit upper bound.",
    )
    parser.add_argument("--mn", type=float, default=0.002)
    parser.add_argument("--hyperviscosity-interval", type=int, default=5)
    parser.add_argument(
        "--large-scale-drag",
        type=float,
        default=0.0,
        help="Linear spectral drag coefficient applied only to low-k momentum modes.",
    )
    parser.add_argument(
        "--drag-kmax",
        type=float,
        default=2.0,
        help="Largest wavenumber damped by --large-scale-drag.",
    )
    parser.add_argument(
        "--cooling-time",
        type=float,
        default=0.0,
        help="Mean-pressure relaxation time. Use 0 to disable homogeneous cooling.",
    )
    parser.add_argument(
        "--cooling-target-pressure",
        type=float,
        default=None,
        help="Target mean pressure for homogeneous cooling. Defaults to 1/gamma.",
    )
    parser.add_argument("--diagnostics-every", type=int, default=5000)
    parser.add_argument("--snapshot-every", type=int, default=10000)
    parser.add_argument(
        "--backend",
        choices=("auto", "numpy", "cupy"),
        default="numpy",
        help="Array backend for the HIT run. Use 'cupy' explicitly for GPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_dir = args.snapshot_dir
    run_id = run_id_from_snapshot_dir(snapshot_dir)
    if args.run:
        if not args.no_timestamp_dir:
            snapshot_dir = create_timestamped_run_dir(args.snapshot_dir)
            run_id = run_id_from_snapshot_dir(snapshot_dir)
            print(f"created timestamped run folder: {snapshot_dir}")

        config = HIT2DConfig(
            nx=args.nx,
            ny=args.ny,
            tfinal=args.tfinal,
            cfl=args.cfl,
            gamma=args.gamma,
            target_mach=args.mach,
            initial_kmin=args.initial_kmin,
            initial_kmax=args.initial_kmax,
            viscosity=args.viscosity,
            initial_re_lambda=args.initial_re_lambda,
            forcing_kmin=args.kf_min,
            forcing_kmax=args.kf if args.kf_max is None else args.kf_max,
            forcing_correlation_time=args.forcing_correlation_time,
            forcing_alpha_memory=args.forcing_alpha_memory,
            target_energy_injection=args.p_target,
            min_forcing_power=args.min_forcing_power,
            max_forcing_rescale=None if args.max_forcing_rescale == 0.0 else args.max_forcing_rescale,
            mach_control=args.mach_control,
            mach_control_target=args.mach_control_target,
            mach_control_memory=args.mach_control_memory,
            mach_control_exponent=args.mach_control_exponent,
            mach_control_min_power=(
                None if args.mach_control_min_power == 0.0 else args.mach_control_min_power
            ),
            mach_control_max_power=(
                None if args.mach_control_max_power == 0.0 else args.mach_control_max_power
            ),
            forcing_seed=args.seed,
            hyperviscosity_mn=args.mn,
            hyperviscosity_interval=args.hyperviscosity_interval,
            large_scale_drag=args.large_scale_drag,
            large_scale_drag_kmax=args.drag_kmax,
            cooling_time=None if args.cooling_time <= 0.0 else args.cooling_time,
            cooling_target_pressure=args.cooling_target_pressure,
            diagnostics_every=args.diagnostics_every,
            snapshot_every=args.snapshot_every,
            output_dir=snapshot_dir,
            backend=args.backend,
        )
        run_simulation(config)
    elif args.snapshot is None:
        resolved_dir = resolve_snapshot_dir(snapshot_dir)
        if resolved_dir != snapshot_dir:
            print(f"using latest snapshot run folder: {resolved_dir}")
            snapshot_dir = resolved_dir
        run_id = run_id_from_snapshot_dir(snapshot_dir)

    diagnostic_snapshot: Path | None = None
    diagnostic_dir = snapshot_dir

    if args.animate:
        postprocess_dir = default_postprocess_dir(snapshot_dir)
        animation_name = f"hit2d_simulation_{run_id}.gif" if run_id else "hit2d_simulation.gif"
        animation_output = args.animation_output or (postprocess_dir / animation_name)
        animate_hit2d_snapshots(
            snapshot_dir,
            animation_output,
            fps=args.fps,
            stride=args.frame_stride,
            limit_mode=args.limit_mode,
            robust_percentile=args.robust_percentile,
            limit_until_time=args.limit_until_time,
        )
        final_snapshot = find_latest_snapshot(snapshot_dir)
        final_name = f"hit2d_final_fields_{run_id}.png" if run_id else "hit2d_final_fields.png"
        final_output = args.output or (postprocess_dir / final_name)
        plot_hit2d_fields(final_snapshot, final_output, show=args.show)
        diagnostic_snapshot = final_snapshot
    else:
        snapshot_path = args.snapshot if args.snapshot is not None else find_latest_snapshot(snapshot_dir)
        diagnostic_snapshot = snapshot_path
        diagnostic_dir = snapshot_path.parent
        if not args.physics_plots and not args.physics_animate:
            output_path = args.output or (default_postprocess_dir(snapshot_path.parent) / "hit2d_fields.png")
            plot_hit2d_fields(snapshot_path, output_path, show=args.show)

    if args.physics_animate:
        animation_dir = diagnostic_dir
        postprocess_dir = default_postprocess_dir(animation_dir)
        diagnostic_run_id = run_id_from_snapshot_dir(animation_dir)
        physics_animation_name = (
            f"hit2d_physics_simulation_{diagnostic_run_id}.gif"
            if diagnostic_run_id
            else "hit2d_physics_simulation.gif"
        )
        physics_animation_output = (
            args.physics_animation_output or (postprocess_dir / physics_animation_name)
        )
        animate_hit2d_physics_snapshots(
            animation_dir,
            physics_animation_output,
            fps=args.fps,
            stride=args.frame_stride,
            limit_mode=args.limit_mode,
            robust_percentile=args.robust_percentile,
            limit_until_time=args.limit_until_time,
            gamma=args.gamma,
        )

    if args.physics_plots:
        if diagnostic_snapshot is None:
            diagnostic_snapshot = find_latest_snapshot(snapshot_dir)
        diagnostic_dir = diagnostic_snapshot.parent
        postprocess_dir = default_postprocess_dir(diagnostic_dir)
        diagnostic_run_id = run_id_from_snapshot_dir(diagnostic_dir)
        physics_name = (
            f"hit2d_physics_fields_{diagnostic_run_id}.png"
            if diagnostic_run_id
            else "hit2d_physics_fields.png"
        )
        history_name = (
            f"hit2d_history_{diagnostic_run_id}.png"
            if diagnostic_run_id
            else "hit2d_history.png"
        )
        physics_output = args.physics_output or (postprocess_dir / physics_name)
        history_output = args.history_output or (postprocess_dir / history_name)
        plot_hit2d_physics_fields(
            diagnostic_snapshot,
            physics_output,
            gamma=args.gamma,
            show=args.show and not args.animate,
        )
        plot_hit2d_history(
            diagnostic_dir,
            history_output,
            gamma=args.gamma,
            x_axis=args.history_x_axis,
            turnover_length=args.turnover_length,
            show=args.show and not args.animate,
        )
        dns_name = (
            f"hit2d_dns_resolution_{diagnostic_run_id}.png"
            if diagnostic_run_id
            else "hit2d_dns_resolution.png"
        )
        plot_hit2d_dns_history(
            diagnostic_dir,
            postprocess_dir / dns_name,
            x_axis=args.history_x_axis,
            turnover_length=args.turnover_length,
            show=args.show and not args.animate,
        )
        forcing_name = (
            f"hit2d_forcing_control_{diagnostic_run_id}.png"
            if diagnostic_run_id
            else "hit2d_forcing_control.png"
        )
        plot_hit2d_forcing_history(
            diagnostic_dir,
            postprocess_dir / forcing_name,
            x_axis=args.history_x_axis,
            turnover_length=args.turnover_length,
            show=args.show and not args.animate,
        )


if __name__ == "__main__":
    main()
