#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

# Allow imports from the repository root when executed as scripts/foo.py.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from OOP.postprocess.turnover import turnover_from_history
except Exception:
    turnover_from_history = None


MT_RE = re.compile(r"Mt0?(\d+)")
ISO_RE = re.compile(
    r"isotropy mismatch:\s*E_LL=(?P<ELL>[0-9.eE+-]+)\s*"
    r"\(normalized=(?P<ELLN>[0-9.eE+-]+)\),\s*"
    r"E_NN=(?P<ENN>[0-9.eE+-]+)\s*"
    r"\(normalized=(?P<ENNN>[0-9.eE+-]+)\)"
)
PDF_RE = re.compile(
    r"PDF moments:\s*S_theta=(?P<Sth>[0-9.eE+-]+),\s*"
    r"F_theta=(?P<Fth>[0-9.eE+-]+),\s*"
    r"S_omega=(?P<Som>[0-9.eE+-]+),\s*"
    r"F_omega=(?P<Fom>[0-9.eE+-]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create final cross-Mach thesis figures from the N=512 HIT campaign."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/hit2d/final_thesis_Mach_ReL130_N512"),
        help="Final campaign root directory.",
    )
    parser.add_argument(
        "--production-dir",
        type=Path,
        default=None,
        help="Override production directory. Defaults to ROOT/production_N512.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to ROOT/thesis_postprocess.",
    )
    parser.add_argument("--start-turnover", type=float, default=4.0)
    parser.add_argument("--end-turnover", type=float, default=16.0)
    parser.add_argument("--target-re", type=float, default=130.0)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument(
        "--chi-relative-floor",
        type=float,
        default=1.0e-6,
        help="Plot chi_d(k) only where E_s+E_d exceeds this fraction of its peak.",
    )
    parser.add_argument(
        "--field-machs",
        type=float,
        nargs=2,
        default=(0.10, 0.60),
        metavar=("LOW_MT", "HIGH_MT"),
        help="Mach numbers used in the optional low/high flow-field comparison.",
    )
    parser.add_argument(
        "--no-fields",
        action="store_true",
        help="Skip the low/high final-field comparison.",
    )
    return parser.parse_args()


def _scalar(data, key: str, default: float = float("nan")) -> float:
    if key not in data.files:
        return default
    try:
        return float(np.asarray(data[key]))
    except Exception:
        return default


def _case_mt(case: Path) -> float:
    config_path = case / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            for key in ("target_mach", "mach_control_target", "mach"):
                if key in cfg and cfg[key] is not None:
                    return float(cfg[key])
        except Exception:
            pass
    match = MT_RE.search(case.name)
    if not match:
        return float("nan")
    digits = match.group(1)
    if len(digits) == 2:
        return int(digits) / 100.0
    if len(digits) == 3:
        return int(digits) / 1000.0
    return float("nan")


def discover_cases(production_dir: Path) -> list[tuple[float, Path]]:
    cases: list[tuple[float, Path]] = []
    if not production_dir.exists():
        raise FileNotFoundError(f"Production directory not found: {production_dir}")
    for case in production_dir.iterdir():
        if not case.is_dir():
            continue
        mt = _case_mt(case)
        if np.isfinite(mt):
            cases.append((mt, case))
    cases.sort(key=lambda item: item[0])
    if not cases:
        raise RuntimeError(f"No Mach-number cases found in {production_dir}")
    return cases


def time_weighted_mean(values: np.ndarray, time: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    time = np.asarray(time, dtype=float)
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


def time_weighted_std(values: np.ndarray, time: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    time = np.asarray(time, dtype=float)
    selected = values[mask]
    selected_time = time[mask]
    finite = np.isfinite(selected) & np.isfinite(selected_time)
    selected = selected[finite]
    selected_time = selected_time[finite]
    if selected.size <= 1:
        return 0.0 if selected.size == 1 else float("nan")
    duration = selected_time[-1] - selected_time[0]
    if duration <= 0.0:
        return float(np.std(selected))
    mu = float(np.trapezoid(selected, selected_time) / duration)
    var = float(np.trapezoid((selected - mu) ** 2, selected_time) / duration)
    return float(np.sqrt(max(var, 0.0)))


def _fallback_summary(root: Path) -> dict[float, dict[str, float]]:
    path = root / "final_thesis_summary.csv"
    result: dict[float, dict[str, float]] = {}
    if not path.exists():
        return result
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                mt = float(row["Mt_target"])
            except Exception:
                continue
            mapped = {}
            for key, value in row.items():
                try:
                    mapped[key] = float(value)
                except Exception:
                    pass
            result[mt] = mapped
    return result


def _history_stats(case: Path, mt: float, start: float, end: float) -> dict[str, float]:
    history_path = case / "diagnostic_history.npz"
    if not history_path.exists() or turnover_from_history is None:
        return {}
    try:
        time, turnover, _ = turnover_from_history(case)
    except Exception:
        return {}
    mask = (turnover >= start) & (turnover <= end)
    if not np.any(mask):
        return {}
    out: dict[str, float] = {
        "selected_turnover_start": float(turnover[mask][0]),
        "selected_turnover_end": float(turnover[mask][-1]),
    }
    with np.load(history_path) as h:
        def add(name: str, source: str, std: bool = False):
            if source not in h.files:
                return
            values = np.asarray(h[source], dtype=float)
            if values.shape != time.shape:
                return
            out[name] = time_weighted_mean(values, time, mask)
            if std:
                out[name + "_std"] = time_weighted_std(values, time, mask)

        add("K_mean", "kinetic_energy", True)
        add("Mt_mean", "turbulent_mach", True)
        add("Re_lambda_mean", "re_lambda_2d", True)
        add("WENO_mean", "weno_fraction", True)
        add("A_K_mean", "A_K", True)
        add("C_uv_mean", "C_uv", True)
        add("eps_nu_mean", "physical_viscous_dissipation", False)
        add("P_hv_mean", "hyperviscosity_drain_power", False)
        if "mass_error" in h.files:
            out["mass_error_final"] = float(np.asarray(h["mass_error"], dtype=float)[-1])
    return out


def _log_stats(case: Path) -> dict[str, float]:
    path = case / "isotropy_postprocess.log"
    if not path.exists():
        return {}
    text = path.read_text(errors="ignore")
    out: dict[str, float] = {}
    match = ISO_RE.search(text)
    if match:
        out.update({
            "E_LL": float(match.group("ELL")),
            "E_LL_normalized": float(match.group("ELLN")),
            "E_NN": float(match.group("ENN")),
            "E_NN_normalized": float(match.group("ENNN")),
        })
    match = PDF_RE.search(text)
    if match:
        out.update({
            "dilatation_skewness": float(match.group("Sth")),
            "dilatation_flatness": float(match.group("Fth")),
            "vorticity_skewness": float(match.group("Som")),
            "vorticity_flatness": float(match.group("Fom")),
        })
    return out


def _spectrum_stats(case: Path) -> dict[str, float]:
    path = case / "spectra_diagnostics.npz"
    if not path.exists():
        return {}
    with np.load(path) as s:
        return {
            "chi_d": _scalar(s, "mean_dilatational_energy_fraction"),
            "energy_cutoff_ratio": _scalar(s, "high_k_energy_ratio"),
            "enstrophy_cutoff_ratio": _scalar(s, "high_k_enstrophy_ratio"),
            "parseval_error": _scalar(s, "parseval_energy_error"),
            "helmholtz_error": _scalar(s, "helmholtz_closure_error"),
            "density_weighted_parseval_error": _scalar(s, "density_weighted_parseval_error"),
        }


def _pdf_stats(case: Path) -> dict[str, float]:
    path = case / "pdf_diagnostics.npz"
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    with np.load(path) as p:
        for variable in ("dilatation", "vorticity", "pressure", "density"):
            for moment in ("rms", "skewness", "flatness"):
                key = f"{variable}_{moment}"
                if key in p.files:
                    out[key] = float(np.asarray(p[key]))
    return out


def collect_rows(
    root: Path,
    cases: list[tuple[float, Path]],
    start: float,
    end: float,
) -> list[dict[str, float | str]]:
    fallback = _fallback_summary(root)
    rows: list[dict[str, float | str]] = []
    for mt, case in cases:
        row: dict[str, float | str] = {"case": case.name, "Mt_target": mt}
        row.update(_history_stats(case, mt, start, end))
        row.update(_log_stats(case))
        row.update(_spectrum_stats(case))
        row.update(_pdf_stats(case))

        # Fill values from the campaign summary only when the dense data are absent.
        fb = fallback.get(mt, {})
        fallback_map = {
            "Mt_mean": "Mt_mean",
            "K_mean": "K_mean",
            "Re_lambda_mean": "Re_lambda_mean",
            "Re_lambda_mean_std": "Re_lambda_std",
            "WENO_mean": "WENO_mean",
            "eps_nu_mean": "eps_nu_mean",
            "P_hv_mean": "P_hv_mean",
            "A_K_mean": "A_K",
            "C_uv_mean": "C_uv",
            "E_LL_normalized": "E_LL_normalized",
            "E_NN_normalized": "E_NN_normalized",
            "chi_d": "chi_d",
            "energy_cutoff_ratio": "energy_cutoff_ratio",
            "enstrophy_cutoff_ratio": "enstrophy_cutoff_ratio",
        }
        for destination, source in fallback_map.items():
            if destination not in row and source in fb:
                row[destination] = fb[source]

        eps = float(row.get("eps_nu_mean", float("nan")))
        phv = float(row.get("P_hv_mean", float("nan")))
        if np.isfinite(eps) and eps > 0.0 and np.isfinite(phv):
            row["HV_to_viscous"] = phv / eps
        else:
            row["HV_to_viscous"] = float("nan")
        rows.append(row)
    return rows


def write_summary(rows: list[dict[str, float | str]], output_dir: Path, target_re: float) -> None:
    keys = [
        "case", "Mt_target", "Mt_mean", "Mt_mean_std", "K_mean", "K_mean_std",
        "Re_lambda_mean", "Re_lambda_mean_std", "WENO_mean", "WENO_mean_std",
        "chi_d", "A_K_mean", "C_uv_mean", "E_LL_normalized", "E_NN_normalized",
        "eps_nu_mean", "P_hv_mean", "HV_to_viscous", "mass_error_final",
        "energy_cutoff_ratio", "enstrophy_cutoff_ratio",
        "dilatation_skewness", "dilatation_flatness",
        "vorticity_skewness", "vorticity_flatness",
        "pressure_skewness", "pressure_flatness",
        "density_skewness", "density_flatness",
        "parseval_error", "helmholtz_error", "density_weighted_parseval_error",
    ]
    csv_path = output_dir / "final_cross_mach_statistics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    def fmt(row, key, spec=".4g"):
        value = row.get(key, float("nan"))
        try:
            value = float(value)
            return f"{value:{spec}}" if np.isfinite(value) else "—"
        except Exception:
            return "—"

    md = [
        "# Final cross-Mach HIT statistics",
        "",
        f"Target stationary $Re_{{\\lambda,2D}}={target_re:g}$.",
        "",
        "| $M_t$ | mean $M_t$ | $K$ | $Re_\\lambda$ | $\\chi_d$ | WENO | HV/visc. | $A_K$ | $E_{LL,n}$ | $E_{NN,n}$ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        re_mean = fmt(row, "Re_lambda_mean", ".2f")
        re_std = fmt(row, "Re_lambda_mean_std", ".2f")
        if re_std != "—":
            re_cell = f"{re_mean} ± {re_std}"
        else:
            re_cell = re_mean
        chi = float(row.get("chi_d", float("nan")))
        weno = float(row.get("WENO_mean", float("nan")))
        hv = float(row.get("HV_to_viscous", float("nan")))
        md.append(
            "| "
            + " | ".join([
                fmt(row, "Mt_target", ".2f"), fmt(row, "Mt_mean", ".4f"),
                fmt(row, "K_mean", ".5f"), re_cell,
                f"{100*chi:.2f}%" if np.isfinite(chi) else "—",
                f"{100*weno:.2f}%" if np.isfinite(weno) else "—",
                f"{100*hv:.2f}%" if np.isfinite(hv) else "—",
                fmt(row, "A_K_mean", ".3f"),
                fmt(row, "E_LL_normalized", ".3f"),
                fmt(row, "E_NN_normalized", ".3f"),
            ])
            + " |"
        )
    (output_dir / "final_cross_mach_statistics.md").write_text("\n".join(md) + "\n")

    # Compact LaTeX table ready to adapt in the thesis.
    tex = [
        r"\begin{tabular}{ccccccc}",
        r"\hline",
        r"$M_t$ & $\overline{M_t}$ & $\overline{K}$ & $\overline{Re}_{\lambda,2D}$ & $\chi_d$ [\%] & WENO [\%] & $\epsilon_{HV}/\epsilon_{\nu}$ [\%] \\",
        r"\hline",
    ]
    for row in rows:
        vals = {
            "mt": fmt(row, "Mt_target", ".2f"),
            "mtm": fmt(row, "Mt_mean", ".4f"),
            "K": fmt(row, "K_mean", ".5f"),
            "Re": fmt(row, "Re_lambda_mean", ".1f"),
        }
        chi = float(row.get("chi_d", float("nan")))
        weno = float(row.get("WENO_mean", float("nan")))
        hv = float(row.get("HV_to_viscous", float("nan")))
        tex.append(
            f"{vals['mt']} & {vals['mtm']} & {vals['K']} & {vals['Re']} & "
            f"{100*chi:.2f} & {100*weno:.2f} & {100*hv:.2f} \\\\" 
        )
    tex += [r"\hline", r"\end{tabular}"]
    (output_dir / "final_cross_mach_table.tex").write_text("\n".join(tex) + "\n")


def _shade_forcing(ax, forcing_shell: tuple[float, float] | None) -> None:
    if forcing_shell is None:
        return
    kmin, kmax = forcing_shell
    if np.isfinite(kmin) and np.isfinite(kmax) and kmax > kmin:
        ax.axvspan(kmin, kmax, alpha=0.08, label="forced shell")


def plot_cross_mach_spectra(cases: list[tuple[float, Path]], output_dir: Path, dpi: int, chi_relative_floor: float) -> Path | None:
    loaded = []
    forcing_shell = None
    for mt, case in cases:
        path = case / "spectra_diagnostics.npz"
        if not path.exists():
            continue
        with np.load(path) as s:
            required = {
                "k", "density_weighted_energy_mean", "solenoidal_energy_mean",
                "dilatational_energy_mean", "complete_shell_max",
            }
            if not required.issubset(s.files):
                continue
            record = {
                "mt": mt,
                "k": np.asarray(s["k"], dtype=float),
                "Edw": np.asarray(s["density_weighted_energy_mean"], dtype=float),
                "Es": np.asarray(s["solenoidal_energy_mean"], dtype=float),
                "Ed": np.asarray(s["dilatational_energy_mean"], dtype=float),
                "complete": int(np.asarray(s["complete_shell_max"])),
            }
            if "forcing_shell" in s.files:
                fs = np.asarray(s["forcing_shell"], dtype=float).ravel()
                if fs.size >= 2 and np.all(np.isfinite(fs[:2])):
                    forcing_shell = (float(fs[0]), float(fs[1]))
            loaded.append(record)
    if len(loaded) < 2:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.9), constrained_layout=True)

    for rec in loaded:
        k = rec["k"]
        Edw = rec["Edw"]
        Es = rec["Es"]
        Ed = rec["Ed"]
        complete = rec["complete"]
        shell = np.arange(k.size)
        valid = (k > 0.0) & (shell <= complete) & np.isfinite(Edw) & (Edw > 0.0)
        if not np.any(valid):
            continue
        label = rf"$M_t={rec['mt']:.2f}$"
        axes[0].loglog(k[valid], Edw[valid], linewidth=1.8, label=label)

        total_energy = float(np.sum(np.maximum(Edw, 0.0)))
        if total_energy > 0.0:
            axes[1].loglog(k[valid], Edw[valid] / total_energy, linewidth=1.8, label=label)

        helm = Es + Ed
        threshold = max(float(np.nanmax(helm)), np.finfo(float).tiny) * max(chi_relative_floor, 0.0)
        chi_valid = (
            (k > 0.0)
            & (shell <= complete)
            & np.isfinite(helm)
            & np.isfinite(Ed)
            & (helm > threshold)
        )
        chi = np.zeros_like(helm)
        np.divide(Ed, helm, out=chi, where=helm > 0.0)
        axes[2].semilogx(k[chi_valid], 100.0 * chi[chi_valid], linewidth=1.8, label=label)

    for ax in axes:
        _shade_forcing(ax, forcing_shell)
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_title("Density-weighted kinetic-energy spectrum")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel(r"$E_{\sqrt{\rho}u}(k)$")

    axes[1].set_title("Normalized spectral shape")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel(r"$E_{\sqrt{\rho}u}(k)/\sum_k E_{\sqrt{\rho}u}(k)$")

    axes[2].set_title("Scale-dependent dilatational fraction")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel(r"$\chi_d(k)$ [\%]")
    axes[2].set_ylim(bottom=0.0)

    # Keep a single clear legend in each panel because the same lines have different meaning.
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        # Deduplicate forcing-shell entries.
        unique = {}
        for h, label in zip(handles, labels):
            if label not in unique:
                unique[label] = h
        ax.legend(unique.values(), unique.keys(), fontsize=8)

    path = output_dir / "final_mach_spectra.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_mach_trends(rows: list[dict[str, float | str]], output_dir: Path, target_re: float, dpi: int) -> Path:
    sorted_rows = sorted(rows, key=lambda r: float(r["Mt_target"]))
    mt = np.asarray([float(r["Mt_target"]) for r in sorted_rows])
    chi = np.asarray([float(r.get("chi_d", np.nan)) for r in sorted_rows])
    weno = np.asarray([float(r.get("WENO_mean", np.nan)) for r in sorted_rows])
    weno_std = np.asarray([float(r.get("WENO_mean_std", np.nan)) for r in sorted_rows])
    re_l = np.asarray([float(r.get("Re_lambda_mean", np.nan)) for r in sorted_rows])
    re_std = np.asarray([float(r.get("Re_lambda_mean_std", np.nan)) for r in sorted_rows])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)

    ax = axes[0]
    valid = np.isfinite(mt) & np.isfinite(chi)
    ax.plot(mt[valid], 100.0 * chi[valid], marker="o", linewidth=1.8, label="campaign")
    if np.count_nonzero(valid) >= 3:
        x2 = mt[valid] ** 2
        a = float(np.dot(x2, chi[valid]) / np.dot(x2, x2))
        prediction = a * x2
        ss_res = float(np.sum((chi[valid] - prediction) ** 2))
        ss_tot = float(np.sum((chi[valid] - np.mean(chi[valid])) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
        xfit = np.linspace(0.0, max(mt[valid]) * 1.03, 200)
        ax.plot(
            xfit,
            100.0 * a * xfit**2,
            linestyle="--",
            linewidth=1.4,
            label=rf"$\chi_d={a:.3f}M_t^2$, $R^2={r2:.3f}$",
        )
    ax.set_title("Integrated dilatational energy")
    ax.set_xlabel(r"$M_t$")
    ax.set_ylabel(r"$\chi_d$ [\%]")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    valid = np.isfinite(mt) & np.isfinite(weno)
    if np.any(valid):
        yerr = np.where(np.isfinite(weno_std[valid]), 100.0 * weno_std[valid], 0.0)
        ax.errorbar(mt[valid], 100.0 * weno[valid], yerr=yerr, marker="o", capsize=3, linewidth=1.5)
    ax.set_title("Shock-capturing activity")
    ax.set_xlabel(r"$M_t$")
    ax.set_ylabel("WENO node fraction [%]")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    valid = np.isfinite(mt) & np.isfinite(re_l)
    if np.any(valid):
        yerr = np.where(np.isfinite(re_std[valid]), re_std[valid], 0.0)
        ax.errorbar(mt[valid], re_l[valid], yerr=yerr, marker="o", capsize=3, linewidth=1.5, label="stationary mean")
    ax.axhline(target_re, linestyle="--", linewidth=1.2, label=rf"target $Re_\lambda={target_re:g}$")
    ax.set_title("Taylor-scale Reynolds number")
    ax.set_xlabel(r"$M_t$")
    ax.set_ylabel(r"$Re_{\lambda,2D}$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    path = output_dir / "final_mach_trends.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pdf_comparison(cases: list[tuple[float, Path]], output_dir: Path, dpi: int) -> tuple[Path | None, Path | None]:
    loaded = []
    for mt, case in cases:
        path = case / "pdf_diagnostics.npz"
        if not path.exists():
            continue
        with np.load(path) as p:
            record = {"mt": mt}
            okay = True
            for variable in ("dilatation", "vorticity", "pressure", "density"):
                ck = f"{variable}_bin_centers"
                pk = f"{variable}_pdf"
                if ck not in p.files or pk not in p.files:
                    okay = False
                    break
                record[ck] = np.asarray(p[ck], dtype=float)
                record[pk] = np.asarray(p[pk], dtype=float)
                for moment in ("skewness", "flatness"):
                    mk = f"{variable}_{moment}"
                    record[mk] = _scalar(p, mk)
            if okay:
                loaded.append(record)
    if len(loaded) < 2:
        return None, None

    variables = [
        ("dilatation", r"$\theta/\theta_{rms}$", "Dilatation"),
        ("vorticity", r"$\omega_z/\omega_{rms}$", "Vorticity"),
        ("pressure", r"$p'/p_{rms}$", "Pressure fluctuation"),
        ("density", r"$\rho'/\rho_{rms}$", "Density fluctuation"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.6), constrained_layout=True)
    for ax, (variable, xlabel, title) in zip(axes.flat, variables):
        for rec in loaded:
            x = rec[f"{variable}_bin_centers"]
            y = rec[f"{variable}_pdf"]
            visible = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
            ax.semilogy(x[visible], y[visible], linewidth=1.5, label=rf"$M_t={rec['mt']:.2f}$")
        # All PDFs are standardized to zero mean / unit RMS, so one N(0,1) reference applies.
        xg = np.linspace(-6.0, 6.0, 500)
        yg = np.exp(-0.5 * xg**2) / np.sqrt(2.0 * np.pi)
        ax.semilogy(xg, yg, linestyle="--", linewidth=1.1, label="Gaussian")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("p.d.f.")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    pdf_path = output_dir / "final_mach_pdfs.png"
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    mt = np.asarray([rec["mt"] for rec in loaded], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for variable, _, title in variables:
        skew = np.asarray([rec[f"{variable}_skewness"] for rec in loaded], dtype=float)
        flat = np.asarray([rec[f"{variable}_flatness"] for rec in loaded], dtype=float)
        axes[0].plot(mt, skew, marker="o", linewidth=1.4, label=title)
        axes[1].plot(mt, flat, marker="o", linewidth=1.4, label=title)
    axes[0].axhline(0.0, linewidth=1.0)
    axes[0].set_title("PDF skewness")
    axes[0].set_xlabel(r"$M_t$")
    axes[0].set_ylabel("skewness")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].axhline(3.0, linestyle="--", linewidth=1.0, label="Gaussian flatness")
    axes[1].set_title("PDF flatness")
    axes[1].set_xlabel(r"$M_t$")
    axes[1].set_ylabel("flatness")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.3)
    handles, labels = axes[1].get_legend_handles_labels()
    unique = {}
    for h, label in zip(handles, labels):
        if label not in unique:
            unique[label] = h
    axes[1].legend(unique.values(), unique.keys(), fontsize=8)
    moments_path = output_dir / "final_pdf_moments_vs_mach.png"
    fig.savefig(moments_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, moments_path


def _nearest_case(cases: list[tuple[float, Path]], target: float) -> tuple[float, Path]:
    return min(cases, key=lambda item: abs(item[0] - target))


def _final_snapshot(case: Path) -> Path | None:
    paths = list(case.glob("hit2d_step*.npz"))
    if not paths:
        return None
    def step(path: Path) -> int:
        match = re.search(r"step(\d+)", path.stem)
        return int(match.group(1)) if match else -1
    return max(paths, key=step)


def plot_low_high_fields(
    cases: list[tuple[float, Path]],
    targets: tuple[float, float],
    output_dir: Path,
    dpi: int,
) -> Path | None:
    selected = [_nearest_case(cases, target) for target in targets]
    records = []
    for mt, case in selected:
        snap = _final_snapshot(case)
        if snap is None:
            return None
        with np.load(snap) as d:
            required = {"x", "y", "rho", "u", "v", "pressure", "vorticity", "divergence"}
            if not required.issubset(d.files):
                return None
            x = np.asarray(d["x"], dtype=float)
            y = np.asarray(d["y"], dtype=float)
            rho = np.asarray(d["rho"], dtype=float)
            u = np.asarray(d["u"], dtype=float)
            v = np.asarray(d["v"], dtype=float)
            p = np.asarray(d["pressure"], dtype=float)
            vort = np.asarray(d["vorticity"], dtype=float)
            div = np.asarray(d["divergence"], dtype=float)
        cfg = json.loads((case / "config.json").read_text()) if (case / "config.json").exists() else {}
        gamma = float(cfg.get("gamma", 1.4))
        sound = np.sqrt(np.maximum(gamma * p / np.maximum(rho, np.finfo(float).tiny), 0.0))
        local_mach = np.sqrt(u**2 + v**2) / np.maximum(sound, np.finfo(float).tiny)
        records.append({
            "mt": mt, "x": x, "y": y, "vorticity": vort,
            "divergence": div, "mach": local_mach, "rho_fluct": rho - np.mean(rho),
        })

    def symmetric_limit(key: str) -> float:
        values = np.concatenate([np.abs(np.asarray(r[key])).ravel() for r in records])
        return float(np.percentile(values, 99.5))

    limits = {
        "vorticity": (-symmetric_limit("vorticity"), symmetric_limit("vorticity")),
        "divergence": (-symmetric_limit("divergence"), symmetric_limit("divergence")),
        "rho_fluct": (-symmetric_limit("rho_fluct"), symmetric_limit("rho_fluct")),
    }
    mach_max = float(np.percentile(np.concatenate([r["mach"].ravel() for r in records]), 99.5))

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.3), constrained_layout=True)
    definitions = [
        ("vorticity", "Vorticity", "coolwarm"),
        ("divergence", "Dilatation", "coolwarm"),
        ("mach", "Local Mach number", "magma"),
        ("rho_fluct", "Density fluctuation", "coolwarm"),
    ]
    for row_index, rec in enumerate(records):
        extent = [float(rec["x"][0]), float(rec["x"][-1]), float(rec["y"][0]), float(rec["y"][-1])]
        for col_index, (key, title, cmap) in enumerate(definitions):
            ax = axes[row_index, col_index]
            if key == "mach":
                vmin, vmax = 0.0, mach_max
            else:
                vmin, vmax = limits[key]
            im = ax.imshow(
                rec[key], origin="lower", extent=extent, aspect="equal",
                cmap=cmap, vmin=vmin, vmax=vmax,
            )
            if row_index == 0:
                ax.set_title(title)
            if col_index == 0:
                ax.set_ylabel(rf"$M_t={rec['mt']:.2f}$\n$y$")
            else:
                ax.set_ylabel("y")
            ax.set_xlabel("x")
            fig.colorbar(im, ax=ax, shrink=0.82)
    path = output_dir / "final_low_high_fields.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    production = (args.production_dir or (root / "production_N512")).resolve()
    output_dir = (args.output_dir or (root / "thesis_postprocess")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(production)
    rows = collect_rows(root, cases, args.start_turnover, args.end_turnover)
    write_summary(rows, output_dir, args.target_re)

    outputs: list[Path] = []
    for path in (
        plot_cross_mach_spectra(cases, output_dir, args.dpi, args.chi_relative_floor),
        plot_mach_trends(rows, output_dir, args.target_re, args.dpi),
    ):
        if path is not None:
            outputs.append(path)

    pdf_paths = plot_pdf_comparison(cases, output_dir, args.dpi)
    outputs.extend(path for path in pdf_paths if path is not None)

    if not args.no_fields:
        field_path = plot_low_high_fields(
            cases,
            (float(args.field_machs[0]), float(args.field_machs[1])),
            output_dir,
            args.dpi,
        )
        if field_path is not None:
            outputs.append(field_path)

    print(f"Saved summary: {output_dir / 'final_cross_mach_statistics.csv'}")
    print(f"Saved summary: {output_dir / 'final_cross_mach_statistics.md'}")
    print(f"Saved table:   {output_dir / 'final_cross_mach_table.tex'}")
    if outputs:
        print("Saved figures:")
        for path in outputs:
            print(f"  {path}")
    missing_spectra = [case.name for _, case in cases if not (case / "spectra_diagnostics.npz").exists()]
    missing_pdfs = [case.name for _, case in cases if not (case / "pdf_diagnostics.npz").exists()]
    if missing_spectra:
        print("WARNING: missing spectra_diagnostics.npz for: " + ", ".join(missing_spectra))
    if missing_pdfs:
        print("WARNING: missing pdf_diagnostics.npz for: " + ", ".join(missing_pdfs))


if __name__ == "__main__":
    main()
