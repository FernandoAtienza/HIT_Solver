#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OOP.postprocess.turnover import turnover_from_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the HIT sensor/LF audit.")
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--start-turnover", type=float, default=4.0)
    parser.add_argument("--end-turnover", type=float, default=10.0)
    return parser.parse_args()


def read_config(case: Path) -> dict[str, object]:
    path = case / "config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def time_mean(values: np.ndarray, time: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    selected = values[mask]
    selected_time = time[mask]
    finite = np.isfinite(selected) & np.isfinite(selected_time)
    selected = selected[finite]
    selected_time = selected_time[finite]
    if selected.size == 0:
        return float("nan")
    if selected.size == 1 or selected_time[-1] <= selected_time[0]:
        return float(np.mean(selected))
    return float(np.trapezoid(selected, selected_time) / (selected_time[-1] - selected_time[0]))


def hist_mean(history, key: str, time: np.ndarray, mask: np.ndarray) -> float:
    if key not in history.files:
        return float("nan")
    values = np.asarray(history[key], dtype=float)
    if values.shape != mask.shape:
        return float("nan")
    return time_mean(values, time, mask)


def scalar(npz, key: str) -> float:
    if key not in npz.files:
        return float("nan")
    return float(np.asarray(npz[key]))


def summarize_case(case: Path, start: float, end: float) -> dict[str, object]:
    time, turnover, _ = turnover_from_history(case)
    mask = (turnover >= start) & (turnover <= end)
    if not np.any(mask):
        raise ValueError(f"{case.name}: no data in {start} <= Neddy <= {end}")
    cfg = read_config(case)
    with np.load(case / "diagnostic_history.npz") as h:
        physical = hist_mean(h, "physical_viscous_dissipation", time, mask)
        hv = hist_mean(h, "hyperviscosity_drain_power", time, mask)
        row = {
            "case": case.name,
            "sensor_mode": str(cfg.get("sensor_mode", "unknown")),
            "weno_flux_splitting": str(cfg.get("weno_flux_splitting", "unknown")),
            "ducros_threshold": float(cfg.get("ducros_threshold", float("nan"))),
            "mean_K": hist_mean(h, "kinetic_energy", time, mask),
            "mean_Mt": hist_mean(h, "turbulent_mach", time, mask),
            "mean_Re_lambda_2d": hist_mean(h, "re_lambda_2d", time, mask),
            "mean_A_K": hist_mean(h, "A_K", time, mask),
            "mean_C_uv": hist_mean(h, "C_uv", time, mask),
            "mean_WENO": hist_mean(h, "weno_fraction", time, mask),
            "mean_WENO_x": hist_mean(h, "weno_fraction_x", time, mask),
            "mean_WENO_y": hist_mean(h, "weno_fraction_y", time, mask),
            "physical_dissipation": physical,
            "HV_drain": hv,
            "HV_to_physical": hv / physical if np.isfinite(physical) and physical > 0 else float("nan"),
            "mean_chi_d": hist_mean(h, "dilatational_energy_fraction", time, mask),
            "mass_error_final": float(np.asarray(h["mass_error"])[-1]),
            "max_turnover": float(turnover[-1]),
        }
    sp = case / "spectra_diagnostics.npz"
    if sp.exists():
        with np.load(sp) as s:
            row["high_k_energy_ratio"] = scalar(s, "high_k_energy_ratio")
            row["high_k_enstrophy_ratio"] = scalar(s, "high_k_enstrophy_ratio")
    else:
        row["high_k_energy_ratio"] = float("nan")
        row["high_k_enstrophy_ratio"] = float("nan")
    return row


def write_outputs(rows: list[dict[str, object]], campaign: Path) -> None:
    csv_path = campaign / "sensor_lf_audit_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cols = [
        "case", "sensor_mode", "weno_flux_splitting", "mean_K", "mean_Mt",
        "mean_Re_lambda_2d", "mean_A_K", "mean_C_uv", "mean_WENO",
        "mean_WENO_x", "mean_WENO_y", "physical_dissipation", "HV_drain",
        "HV_to_physical", "high_k_energy_ratio", "high_k_enstrophy_ratio",
    ]
    labels = {
        "case": "Case", "sensor_mode": "Sensor", "weno_flux_splitting": "LF",
        "mean_K": "Mean K", "mean_Mt": "Mean Mt", "mean_Re_lambda_2d": "Mean Reλ,2D",
        "mean_A_K": "Mean A_K", "mean_C_uv": "Mean C_uv", "mean_WENO": "WENO",
        "mean_WENO_x": "WENO x", "mean_WENO_y": "WENO y",
        "physical_dissipation": "Physical diss.", "HV_drain": "HV drain",
        "HV_to_physical": "HV/physical", "high_k_energy_ratio": "High-k E",
        "high_k_enstrophy_ratio": "High-k Z",
    }
    lines = ["# HIT sensor / LF audit", "", "| " + " | ".join(labels[c] for c in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for row in rows:
        def fmt(v):
            if isinstance(v, float):
                return "—" if not np.isfinite(v) else f"{v:.6g}"
            return str(v)
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    lines += [
        "",
        "## Selection logic",
        "",
        "Prefer the least dissipative configuration that remains stable, preserves positive density/pressure, ",
        "keeps high-k energy/enstrophy free of pile-up, reduces false WENO activation, and does not degrade isotropy.",
        "Local LF should be judged from changes at resolved/intermediate k, not only the final cutoff.",
    ]
    (campaign / "sensor_lf_audit_summary.md").write_text("\n".join(lines) + "\n")


def plot_comparison(cases: list[Path], campaign: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    for case in cases:
        cfg = read_config(case)
        label = f"{cfg.get('sensor_mode','?')} / {cfg.get('weno_flux_splitting','?')}"
        sp = case / "spectra_diagnostics.npz"
        if sp.exists():
            with np.load(sp) as s:
                k = np.asarray(s["k"], dtype=float)
                E = np.asarray(s["energy_mean"], dtype=float)
                Z = np.asarray(s["enstrophy_mean"], dtype=float)
                complete = int(np.asarray(s["complete_shell_max"]))
            mE = (k > 0) & (np.arange(k.size) <= complete) & (E > 0)
            mZ = (k > 0) & (np.arange(k.size) <= complete) & (Z > 0)
            axes[0, 0].loglog(k[mE], E[mE], label=label)
            axes[0, 1].loglog(k[mZ], Z[mZ], label=label)
        try:
            _time, turnover, _ = turnover_from_history(case)
            with np.load(case / "diagnostic_history.npz") as h:
                axes[1, 0].plot(turnover, np.asarray(h["weno_fraction"]), label=label)
                axes[1, 1].plot(turnover, np.asarray(h["re_lambda_2d"]), label=label)
        except Exception:
            pass

    axes[0, 0].set_title("Kinetic-energy spectrum")
    axes[0, 0].set_xlabel("k"); axes[0, 0].set_ylabel("E(k)")
    axes[0, 1].set_title("Enstrophy spectrum")
    axes[0, 1].set_xlabel("k"); axes[0, 1].set_ylabel("Z(k)")
    axes[1, 0].set_title("WENO node fraction")
    axes[1, 0].set_xlabel(r"$N_{eddy}$"); axes[1, 0].set_ylabel("fraction")
    axes[1, 1].set_title(r"Taylor-microscale Reynolds number")
    axes[1, 1].set_xlabel(r"$N_{eddy}$"); axes[1, 1].set_ylabel(r"$Re_{\lambda,2D}$")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(campaign / "sensor_lf_audit_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cases = sorted(
        p for p in args.campaign_dir.iterdir()
        if p.is_dir() and (p / "diagnostic_history.npz").exists()
    )
    if not cases:
        raise SystemExit("No completed cases found")
    rows = [summarize_case(c, args.start_turnover, args.end_turnover) for c in cases]
    write_outputs(rows, args.campaign_dir)
    plot_comparison(cases, args.campaign_dir)
    print(f"saved: {args.campaign_dir / 'sensor_lf_audit_summary.md'}")
    print(f"saved: {args.campaign_dir / 'sensor_lf_audit_comparison.png'}")


if __name__ == "__main__":
    main()
