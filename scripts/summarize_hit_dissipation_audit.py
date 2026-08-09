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
    parser = argparse.ArgumentParser(
        description=(
            "Summarize a sequential HIT hyperviscosity audit and create "
            "cross-case spectra plots."
        )
    )
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--start-turnover", type=float, default=4.0)
    parser.add_argument("--end-turnover", type=float, default=10.0)
    return parser.parse_args()


def _scalar(data: np.lib.npyio.NpzFile, key: str, default: float = float("nan")) -> float:
    if key not in data.files:
        return default
    return float(np.asarray(data[key]))


def _time_mean(
    data: np.lib.npyio.NpzFile,
    key: str,
    mask: np.ndarray,
    time: np.ndarray,
) -> float:
    if key not in data.files:
        return float("nan")
    values = np.asarray(data[key], dtype=float)
    if values.shape != mask.shape:
        return float("nan")
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


def _read_config(case_dir: Path) -> dict[str, object]:
    path = case_dir / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def summarize_case(
    case_dir: Path,
    start_turnover: float,
    end_turnover: float,
) -> dict[str, object]:
    history_path = case_dir / "diagnostic_history.npz"
    if not history_path.exists():
        raise FileNotFoundError(history_path)

    time, turnover, _length = turnover_from_history(case_dir)
    mask = (turnover >= start_turnover) & (turnover <= end_turnover)
    if not np.any(mask):
        raise ValueError(
            f"{case_dir.name}: no diagnostic samples in "
            f"{start_turnover} <= Neddy <= {end_turnover}; "
            f"available maximum is {float(turnover[-1]):.3f}"
        )

    config = _read_config(case_dir)
    with np.load(history_path) as history:
        physical = _time_mean(
            history,
            "physical_viscous_dissipation",
            mask,
            time,
        )
        if "hyperviscosity_energy_removed_cumulative" in history.files:
            cumulative = np.asarray(
                history["hyperviscosity_energy_removed_cumulative"],
                dtype=float,
            )[mask]
            selected_time = time[mask]
            duration = float(selected_time[-1] - selected_time[0])
            numerical = (
                float(cumulative[-1] - cumulative[0]) / duration
                if cumulative.size > 1 and duration > 0.0
                else 0.0
            )
        else:
            numerical = _time_mean(
                history,
                "hyperviscosity_drain_power",
                mask,
                time,
            )
        ratio = numerical / physical if np.isfinite(physical) and physical > 0.0 else float("nan")
        row: dict[str, object] = {
            "case": case_dir.name,
            "mn": float(config.get("hyperviscosity_mn", _scalar(history, "hyperviscosity_mn"))),
            "hyperviscosity_interval": int(
                config.get(
                    "hyperviscosity_interval",
                    round(_scalar(history, "hyperviscosity_interval", 0.0)),
                )
            ),
            "dynamic_viscosity": float(
                config.get("viscosity", _scalar(history, "dynamic_viscosity"))
            ),
            "requested_initial_re_lambda_2d": float(
                config.get("requested_initial_re_lambda_2d", float("nan"))
            ),
            "mean_re_lambda_2d": _time_mean(history, "re_lambda_2d", mask, time),
            "mean_kinetic_energy": _time_mean(history, "kinetic_energy", mask, time),
            "mean_turbulent_mach": _time_mean(history, "turbulent_mach", mask, time),
            "mean_weno_fraction": _time_mean(history, "weno_fraction", mask, time),
            "mean_physical_viscous_dissipation": physical,
            "mean_hyperviscosity_drain_power": numerical,
            "hyperviscosity_to_physical_ratio": ratio,
            "mean_dilatational_energy_fraction": _time_mean(
                history,
                "dilatational_energy_fraction",
                mask,
                time,
            ),
            "stationary_samples": int(np.count_nonzero(mask)),
            "max_turnover": float(turnover[-1]),
        }

    spectra_path = case_dir / "spectra_diagnostics.npz"
    if spectra_path.exists():
        with np.load(spectra_path) as spectra:
            row["high_k_energy_ratio"] = _scalar(spectra, "high_k_energy_ratio")
            row["high_k_enstrophy_ratio"] = _scalar(spectra, "high_k_enstrophy_ratio")
            row["spectral_dilatational_fraction"] = _scalar(
                spectra, "mean_dilatational_energy_fraction"
            )
    else:
        row["high_k_energy_ratio"] = float("nan")
        row["high_k_enstrophy_ratio"] = float("nan")
        row["spectral_dilatational_fraction"] = float("nan")
    return row


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_value(value: object) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return "—"
        return f"{value:.6g}"
    return str(value)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    columns = [
        "case",
        "mn",
        "mean_re_lambda_2d",
        "mean_kinetic_energy",
        "mean_turbulent_mach",
        "mean_weno_fraction",
        "mean_physical_viscous_dissipation",
        "mean_hyperviscosity_drain_power",
        "hyperviscosity_to_physical_ratio",
        "mean_dilatational_energy_fraction",
    ]
    labels = {
        "case": "Case",
        "mn": "mn",
        "mean_re_lambda_2d": "Mean Re_lambda,2D",
        "mean_kinetic_energy": "Mean K",
        "mean_turbulent_mach": "Mean Mt",
        "mean_weno_fraction": "Mean WENO fraction",
        "mean_physical_viscous_dissipation": "Physical dissipation",
        "mean_hyperviscosity_drain_power": "HV drain",
        "hyperviscosity_to_physical_ratio": "HV / physical",
        "mean_dilatational_energy_fraction": "Mean chi_d",
    }
    lines = [
        "# HIT dissipation-audit summary",
        "",
        "| " + " | ".join(labels[column] for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row[column]) for column in columns) + " |")
    lines.extend(
        [
            "",
            "The hyperviscosity-to-physical ratio compares the measured kinetic-energy "
            "change caused by the discrete hyperviscosity filter with the physical "
            "viscous dissipation over the selected turnover interval.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def plot_spectra(case_dirs: list[Path], output_path: Path) -> None:
    available = [case for case in case_dirs if (case / "spectra_diagnostics.npz").exists()]
    if not available:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    for case_dir in available:
        config = _read_config(case_dir)
        label = f"mn={float(config.get('hyperviscosity_mn', float('nan'))):g}"
        with np.load(case_dir / "spectra_diagnostics.npz") as data:
            k = np.asarray(data["k"], dtype=float)
            energy = np.asarray(data["energy_mean"], dtype=float)
            enstrophy = np.asarray(data["enstrophy_mean"], dtype=float)
            complete = int(np.asarray(data["complete_shell_max"]))
        valid_energy = (k > 0.0) & (np.arange(k.size) <= complete) & (energy > 0.0)
        valid_enstrophy = (k > 0.0) & (np.arange(k.size) <= complete) & (enstrophy > 0.0)
        axes[0].loglog(k[valid_energy], energy[valid_energy], label=label)
        axes[1].loglog(k[valid_enstrophy], enstrophy[valid_enstrophy], label=label)

    axes[0].set_title("Kinetic-energy spectra")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("E(k)")
    axes[1].set_title("Enstrophy spectra")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Z(k)")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    campaign_dir = args.campaign_dir.expanduser().resolve()
    case_dirs = sorted(
        path
        for path in campaign_dir.iterdir()
        if path.is_dir() and (path / "diagnostic_history.npz").exists()
    )
    if not case_dirs:
        raise FileNotFoundError(f"No completed HIT cases found in {campaign_dir}")

    rows = [
        summarize_case(case, args.start_turnover, args.end_turnover)
        for case in case_dirs
    ]
    write_csv(rows, campaign_dir / "dissipation_audit_summary.csv")
    write_markdown(rows, campaign_dir / "dissipation_audit_summary.md")
    plot_spectra(case_dirs, campaign_dir / "dissipation_audit_spectra.png")

    print(f"saved: {campaign_dir / 'dissipation_audit_summary.csv'}")
    print(f"saved: {campaign_dir / 'dissipation_audit_summary.md'}")
    if (campaign_dir / "dissipation_audit_spectra.png").exists():
        print(f"saved: {campaign_dir / 'dissipation_audit_spectra.png'}")


if __name__ == "__main__":
    main()
