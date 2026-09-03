#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_scalar(data, key: str, default=np.nan) -> float:
    if key not in data:
        return float(default)
    value = np.asarray(data[key])
    if value.size == 0:
        return float(default)
    return float(value.reshape(-1)[-1])


def _mean_over_time(history, key: str, t0: float, t1: float) -> float:
    if key not in history or "time" not in history:
        return float("nan")
    time = np.asarray(history["time"], dtype=float)
    values = np.asarray(history[key], dtype=float)
    if values.ndim == 0 or values.shape[0] != time.shape[0]:
        return float(values.reshape(-1)[-1]) if values.size else float("nan")
    mask = (time >= t0) & (time <= t1) & np.isfinite(values)
    return float(np.mean(values[mask])) if np.any(mask) else float("nan")


def _std_over_time(history, key: str, t0: float, t1: float) -> float:
    if key not in history or "time" not in history:
        return float("nan")
    time = np.asarray(history["time"], dtype=float)
    values = np.asarray(history[key], dtype=float)
    if values.ndim == 0 or values.shape[0] != time.shape[0]:
        return float("nan")
    mask = (time >= t0) & (time <= t1) & np.isfinite(values)
    return float(np.std(values[mask])) if np.any(mask) else float("nan")


def _case_record(case_dir: Path) -> dict | None:
    required = [
        case_dir / "config.json",
        case_dir / "diagnostic_history.npz",
        case_dir / "isotropy_diagnostics.npz",
        case_dir / "spectra_diagnostics.npz",
    ]
    if not all(path.is_file() for path in required):
        return None

    config = _load_json(required[0])
    with np.load(required[1], allow_pickle=False) as history, np.load(
        required[2], allow_pickle=False
    ) as iso, np.load(required[3], allow_pickle=False) as spectra:
        interval = np.asarray(iso["selected_time_interval"], dtype=float)
        t0, t1 = float(interval[0]), float(interval[1])
        selected = np.asarray(iso["selected_mask"], dtype=bool)
        K = np.asarray(iso["K"], dtype=float)
        Mt = np.asarray(iso["Mt"], dtype=float)
        AK = np.asarray(iso["A_K"], dtype=float)
        Cuv = np.asarray(iso["C_uv"], dtype=float)

        record = {
            "case": case_dir.name,
            "nx": int(config["nx"]),
            "ny": int(config["ny"]),
            "target_Mt": float(config["target_mach"]),
            "seed": int(config["forcing_seed"]),
            "cfl": float(config["cfl"]),
            "initial_Re_lambda_2d": float(config.get("resolved_initial_re_lambda_2d", np.nan)),
            "viscosity": float(config.get("resolved_dynamic_viscosity", config.get("viscosity", np.nan))),
            "sensor_mode": str(config.get("sensor_mode", "")),
            "weno_flux_splitting": str(config.get("weno_flux_splitting", "")),
            "mn": float(config.get("hyperviscosity_mn", np.nan)),
            "hyperviscosity_interval": int(config.get("hyperviscosity_interval", 0)),
            "selected_t0": t0,
            "selected_t1": t1,
            "selected_turnover0": float(np.asarray(iso["selected_turnover_interval"])[0]),
            "selected_turnover1": float(np.asarray(iso["selected_turnover_interval"])[1]),
            "snapshots": int(np.asarray(iso["number_of_snapshots"])),
            "mean_K": float(np.mean(K[selected])),
            "std_K": float(np.std(K[selected])),
            "mean_Mt": float(np.mean(Mt[selected])),
            "std_Mt": float(np.std(Mt[selected])),
            "mean_A_K": float(np.mean(AK[selected])),
            "mean_C_uv": float(np.mean(Cuv[selected])),
            "E_LL_normalized": _safe_scalar(iso, "E_LL_normalized"),
            "E_NN_normalized": _safe_scalar(iso, "E_NN_normalized"),
            "mean_Re_lambda_2d": _mean_over_time(history, "re_lambda_2d", t0, t1),
            "std_Re_lambda_2d": _std_over_time(history, "re_lambda_2d", t0, t1),
            "mean_WENO": _mean_over_time(history, "weno_fraction", t0, t1),
            "mean_WENO_x": _mean_over_time(history, "weno_fraction_x", t0, t1),
            "mean_WENO_y": _mean_over_time(history, "weno_fraction_y", t0, t1),
            "physical_dissipation": _mean_over_time(history, "physical_viscous_dissipation", t0, t1),
            "HV_drain": _mean_over_time(history, "hyperviscosity_drain_power", t0, t1),
            "eta_over_dx": _mean_over_time(history, "eta_over_dx", t0, t1),
            "kmax_eta": _mean_over_time(history, "kmax_eta", t0, t1),
            "kraichnan_over_dx": _mean_over_time(history, "kraichnan_over_dx", t0, t1),
            "kmax_kraichnan": _mean_over_time(history, "kmax_kraichnan", t0, t1),
            "mass_error_final": _safe_scalar(history, "mass_error"),
            "mean_chi_d": _safe_scalar(spectra, "mean_dilatational_energy_fraction"),
            "high_k_energy_ratio": _safe_scalar(spectra, "high_k_energy_ratio"),
            "high_k_enstrophy_ratio": _safe_scalar(spectra, "high_k_enstrophy_ratio"),
        }
        physical = record["physical_dissipation"]
        record["HV_to_physical"] = (
            record["HV_drain"] / physical
            if np.isfinite(physical) and abs(physical) > np.finfo(float).tiny
            else float("nan")
        )
    return record


def _available_cases(root: Path) -> tuple[list[dict], dict[str, Path]]:
    records = []
    paths = {}
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        record = _case_record(case_dir)
        if record is not None:
            records.append(record)
            paths[case_dir.name] = case_dir
    return records, paths


def _write_csv(root: Path, records: list[dict]) -> Path:
    path = root / "thesis_campaign_summary.csv"
    if not records:
        return path
    fields = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return path


def _fmt(value, digits=5):
    if isinstance(value, str):
        return value
    try:
        value = float(value)
    except Exception:
        return str(value)
    return "nan" if not np.isfinite(value) else f"{value:.{digits}g}"


def _write_markdown(root: Path, records: list[dict]) -> Path:
    path = root / "thesis_campaign_summary.md"
    columns = [
        ("case", "Case"),
        ("nx", "N"),
        ("target_Mt", "Mt target"),
        ("mean_Mt", "Mt mean"),
        ("mean_Re_lambda_2d", "Reλ mean"),
        ("mean_A_K", "A_K"),
        ("mean_C_uv", "C_uv"),
        ("E_LL_normalized", "E_LL,n"),
        ("E_NN_normalized", "E_NN,n"),
        ("mean_WENO", "WENO"),
        ("mean_chi_d", "χ_d"),
        ("HV_to_physical", "HV/physical"),
        ("kmax_eta", "kmax η"),
        ("kmax_kraichnan", "kmax ηΩ"),
    ]
    lines = [
        "# Thesis HIT campaign summary",
        "",
        "All statistics use the common selected turnover-time interval stored in each case.",
        "",
        "| " + " | ".join(title for _, title in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for record in records:
        vals = []
        for key, _ in columns:
            value = record[key]
            if key == "nx":
                vals.append(str(int(value)))
            else:
                vals.append(_fmt(value))
        lines.append("| " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Numerical baseline",
        "",
        "- sensor: legacy shock sensor",
        "- WENO splitting: local stencil Lax–Friedrichs",
        "- conservative compact-region hyperviscosity: mn=0.005",
        "- hyperviscosity interval: every 5 complete RK steps",
        "- CFL: 0.10",
        "- initial 2-D Taylor Reynolds number: 120",
        "- forcing shell: 3 <= k <= 5",
        "- statistics window: 4 <= N_eddy <= 16",
        "",
        "## Interpretation checklist",
        "",
        "- Grid convergence is supported when N=256 and N=512 overlap over their common resolved spectral range and their stationary bulk statistics agree.",
        "- Cross-Mach comparisons are interpretable only if the stationary Re_lambda values remain reasonably close; otherwise viscosity should be recalibrated before the final 512² Mach campaign.",
        "- A rising dilatational fraction with Mt is a compressibility result only after grid convergence and comparable Re_lambda are established.",
        "- WENO activity should increase only when compressive structures require it; excessive domain fractions at higher Mt should trigger a sensor review.",
        "- The independent-seed Mt=0.25 case estimates stochastic sensitivity of isotropy and spectral statistics.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _load_spectrum(case_dir: Path):
    with np.load(case_dir / "spectra_diagnostics.npz", allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _shade_forcing(ax, spectrum):
    shell = np.asarray(spectrum.get("forcing_shell", [np.nan, np.nan]), dtype=float)
    if shell.size >= 2 and np.all(np.isfinite(shell[:2])):
        ax.axvspan(shell[0], shell[1], alpha=0.08, label="forced shell")


def _plot_grid(root: Path, paths: dict[str, Path]):
    names = [
        "grid_Mt025_N128_seed1234",
        "grid_Mt025_N256_seed1234",
        "grid_Mt025_N512_seed1234",
    ]
    available = [name for name in names if name in paths]
    if len(available) < 2:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)
    first = None
    for name in available:
        spec = _load_spectrum(paths[name])
        if first is None:
            first = spec
        k = np.asarray(spec["k"], dtype=float)
        E = np.asarray(spec["density_weighted_energy_mean"], dtype=float)
        Z = np.asarray(spec["enstrophy_mean"], dtype=float)
        n = _load_json(paths[name] / "config.json")["nx"]
        valid_E = (k > 0) & (E > 0)
        valid_Z = (k > 0) & (Z > 0)
        axes[0].loglog(k[valid_E], E[valid_E], label=f"N={n}")
        axes[1].loglog(k[valid_Z], Z[valid_Z], label=f"N={n}")
    if first is not None:
        _shade_forcing(axes[0], first)
        _shade_forcing(axes[1], first)
    # Figure 5.1: keep the panels visually dominant in the printed thesis.
    # The caption already explains that this is the grid-convergence figure, so
    # subplot titles are intentionally omitted and the ordinate symbols carry
    # the physical meaning directly.
    for ax in axes:
        ax.set_title("")
        ax.set_xlabel(r"$k$", fontsize=16)
        ax.tick_params(axis="both", which="both", labelsize=14)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=13)
    axes[0].set_ylabel(r"$E_{\sqrt{\rho}u}(k)$", fontsize=16)
    axes[1].set_ylabel(r"$Z(k)$", fontsize=16)
    out = root / "thesis_grid_convergence_spectra.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _mach_paths(paths: dict[str, Path]):
    ordered = [
        (0.10, "mach_Mt010_N256_seed1234"),
        (0.25, "grid_Mt025_N256_seed1234"),
        (0.50, "mach_Mt050_N256_seed1234"),
        (0.60, "mach_Mt060_N256_seed1234"),
    ]
    return [(mt, paths[name]) for mt, name in ordered if name in paths]


def _plot_mach_spectra(root: Path, paths: dict[str, Path]):
    cases = _mach_paths(paths)
    if len(cases) < 2:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.1), constrained_layout=True)
    first = None
    for mt, case_dir in cases:
        spec = _load_spectrum(case_dir)
        if first is None:
            first = spec
        k = np.asarray(spec["k"], dtype=float)
        E = np.asarray(spec["density_weighted_energy_mean"], dtype=float)
        Es = np.asarray(spec["solenoidal_energy_mean"], dtype=float)
        Ed = np.asarray(spec["dilatational_energy_mean"], dtype=float)
        valid = (k > 0) & (E > 0)
        axes[0].loglog(k[valid], E[valid], label=fr"$M_t={mt:.2f}$")
        total = float(np.sum(E[valid]))
        En = E / max(total, np.finfo(float).tiny)
        axes[1].loglog(k[valid], En[valid], label=fr"$M_t={mt:.2f}$")
        denom = Es + Ed
        chi = np.divide(Ed, denom, out=np.full_like(Ed, np.nan), where=denom > 0)
        valid_chi = (k > 0) & np.isfinite(chi) & (chi >= 0)
        axes[2].semilogx(k[valid_chi], chi[valid_chi], label=fr"$M_t={mt:.2f}$")
    if first is not None:
        for ax in axes:
            _shade_forcing(ax, first)
    axes[0].set_title("Absolute density-weighted spectrum")
    axes[1].set_title("Normalized spectral shape")
    axes[2].set_title("Scale-dependent dilatational fraction")
    axes[0].set_ylabel(r"$E_{\sqrt{\rho}u}(k)$")
    axes[1].set_ylabel(r"$E_{\sqrt{\rho}u}(k)/\sum_k E_{\sqrt{\rho}u}(k)$")
    axes[2].set_ylabel(r"$\chi_d(k)=E_d/(E_s+E_d)$")
    for ax in axes:
        ax.set_xlabel("k")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    out = root / "thesis_mach_spectra_comparison.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_mach_statistics(root: Path, records: list[dict]):
    wanted = {
        "mach_Mt010_N256_seed1234",
        "grid_Mt025_N256_seed1234",
        "mach_Mt050_N256_seed1234",
        "mach_Mt060_N256_seed1234",
    }
    subset = sorted((r for r in records if r["case"] in wanted), key=lambda r: r["target_Mt"])
    if len(subset) < 2:
        return None
    mt = np.asarray([r["target_Mt"] for r in subset], dtype=float)
    achieved = np.asarray([r["mean_Mt"] for r in subset], dtype=float)
    rel = np.asarray([r["mean_Re_lambda_2d"] for r in subset], dtype=float)
    chi = np.asarray([r["mean_chi_d"] for r in subset], dtype=float)
    weno = np.asarray([r["mean_WENO"] for r in subset], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)
    axes[0, 0].plot(mt, achieved, marker="o")
    axes[0, 0].plot(mt, mt, linestyle="--", linewidth=1.0, label="target")
    axes[0, 0].set_ylabel(r"stationary $M_t$")
    axes[0, 0].legend()
    axes[0, 1].plot(mt, rel, marker="o")
    axes[0, 1].set_ylabel(r"stationary $Re_{\lambda,2D}$")
    axes[1, 0].plot(mt, chi, marker="o")
    axes[1, 0].set_ylabel(r"integrated $\chi_d$")
    axes[1, 1].plot(mt, weno, marker="o")
    axes[1, 1].set_ylabel("mean WENO fraction")
    for ax in axes.flat:
        ax.set_xlabel(r"target $M_t$")
        ax.grid(True, alpha=0.3)
    out = root / "thesis_mach_stationary_statistics.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_repeatability(root: Path, paths: dict[str, Path]):
    names = ["grid_Mt025_N256_seed1234", "repeat_Mt025_N256_seed5678"]
    if not all(name in paths for name in names):
        return None
    fig, ax = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    first = None
    for name in names:
        spec = _load_spectrum(paths[name])
        if first is None:
            first = spec
        k = np.asarray(spec["k"], dtype=float)
        E = np.asarray(spec["density_weighted_energy_mean"], dtype=float)
        valid = (k > 0) & (E > 0)
        seed = _load_json(paths[name] / "config.json")["forcing_seed"]
        ax.loglog(k[valid], E[valid], label=f"seed={seed}")
    if first is not None:
        _shade_forcing(ax, first)
    # Figure 5.2: retain the existing title treatment while increasing the
    # axes and legend slightly for paper-size readability.
    ax.set_title(r"Independent-realization check: $M_t=0.25$, $N=256$")
    ax.set_xlabel(r"$k$", fontsize=15)
    ax.set_ylabel(r"$E_{\sqrt{\rho}u}(k)$", fontsize=15)
    ax.tick_params(axis="both", which="both", labelsize=14)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=13)
    out = root / "thesis_repeatability_spectra.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize thesis-oriented HIT grid/Mach campaign.")
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--start-turnover", type=float, default=4.0)
    parser.add_argument("--end-turnover", type=float, default=16.0)
    args = parser.parse_args()
    root = args.campaign.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    records, paths = _available_cases(root)
    if not records:
        raise RuntimeError(f"No fully post-processed cases found under {root}")

    csv_path = _write_csv(root, records)
    md_path = _write_markdown(root, records)
    outputs = [csv_path, md_path]
    for func in (_plot_grid, _plot_mach_spectra, _plot_repeatability):
        out = func(root, paths)
        if out is not None:
            outputs.append(out)
    out = _plot_mach_statistics(root, records)
    if out is not None:
        outputs.append(out)

    print("Saved thesis campaign products:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
