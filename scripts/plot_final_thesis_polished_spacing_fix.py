#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OOP.postprocess.turnover import turnover_from_history

MT_RE = re.compile(r"Mt0?(\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final polished thesis figures with temporal variability.")
    p.add_argument(
        "--root",
        type=Path,
        default=Path("results/hit2d/final_thesis_Mach_ReL130_N512"),
    )
    p.add_argument("--production-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--start-turnover", type=float, default=4.0)
    p.add_argument("--end-turnover", type=float, default=16.0)
    p.add_argument("--target-re", type=float, default=130.0)
    p.add_argument("--chi-relative-floor", type=float, default=1.0e-4)
    p.add_argument(
        "--absolute-spectrum-relative-floor",
        type=float,
        default=1.0e-16,
        help=(
            "Lower display floor for the absolute density-weighted spectrum, "
            "relative to the largest plotted spectral value. This only controls "
            "plot limits/uncertainty shading; it does not alter the spectrum."
        ),
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--field-machs", nargs=2, type=float, default=(0.10, 0.60))
    p.add_argument("--stationarity-machs", nargs=2, type=float, default=(0.25, 0.60))
    return p.parse_args()


def case_mt(case: Path) -> float:
    cfg = case / "config.json"
    if cfg.exists():
        try:
            payload = json.loads(cfg.read_text())
            for key in ("target_mach", "mach_control_target", "mach"):
                if payload.get(key) is not None:
                    return float(payload[key])
        except Exception:
            pass
    m = MT_RE.search(case.name)
    if not m:
        return float("nan")
    digits = m.group(1)
    return int(digits) / (100.0 if len(digits) == 2 else 1000.0)


def discover_cases(prod: Path) -> list[tuple[float, Path]]:
    cases = [(case_mt(p), p) for p in prod.iterdir() if p.is_dir()]
    cases = [(mt, p) for mt, p in cases if np.isfinite(mt)]
    cases.sort(key=lambda item: item[0])
    if not cases:
        raise RuntimeError(f"No Mach cases found under {prod}")
    return cases


def scalar(data, key: str, default=float("nan")) -> float:
    if key not in data.files:
        return default
    return float(np.asarray(data[key]))


def weighted_mean_std(values: np.ndarray, time: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)[mask]
    t = np.asarray(time, dtype=float)[mask]
    finite = np.isfinite(values) & np.isfinite(t)
    values, t = values[finite], t[finite]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1 or t[-1] <= t[0]:
        return float(np.mean(values)), 0.0
    duration = float(t[-1] - t[0])
    mu = float(np.trapezoid(values, t) / duration)
    var = float(np.trapezoid((values - mu) ** 2, t) / duration)
    return mu, math.sqrt(max(var, 0.0))


def history_stats(case: Path, start: float, end: float) -> dict[str, float]:
    hist_path = case / "diagnostic_history.npz"
    if not hist_path.exists():
        return {}
    time, turnover, _ = turnover_from_history(case)
    mask = (turnover >= start) & (turnover <= end)
    if not np.any(mask):
        return {}
    out: dict[str, float] = {}
    with np.load(hist_path) as h:
        for dest, src in (
            ("Mt", "turbulent_mach"),
            ("K", "kinetic_energy"),
            ("Re_lambda", "re_lambda_2d"),
            ("WENO", "weno_fraction"),
        ):
            if src in h.files and np.asarray(h[src]).shape == time.shape:
                mu, sigma = weighted_mean_std(np.asarray(h[src]), time, mask)
                out[f"{dest}_mean"] = mu
                out[f"{dest}_std"] = sigma
    return out


def uncertainty_stats(case: Path) -> dict[str, float]:
    path = case / "thesis_uncertainty_diagnostics.npz"
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    with np.load(path) as d:
        out["snapshot_count"] = scalar(d, "number_of_snapshots")
        for key in (
            "chi_d",
            "dilatation_rms", "dilatation_skewness", "dilatation_flatness",
            "vorticity_rms", "vorticity_skewness", "vorticity_flatness",
            "pressure_rms", "pressure_skewness", "pressure_flatness",
            "density_rms", "density_skewness", "density_flatness",
        ):
            out[f"{key}_temporal_mean"] = scalar(d, f"{key}_mean")
            out[f"{key}_temporal_std"] = scalar(d, f"{key}_std")
    return out


def spectrum_stats(case: Path) -> dict[str, float]:
    path = case / "spectra_diagnostics.npz"
    if not path.exists():
        return {}
    with np.load(path) as s:
        return {
            "chi_d": scalar(s, "mean_dilatational_energy_fraction"),
            "parseval_error": scalar(s, "parseval_energy_error"),
            "helmholtz_error": scalar(s, "helmholtz_closure_error"),
        }


def collect_rows(cases, start, end):
    rows = []
    for mt, case in cases:
        row: dict[str, float | str] = {"case": case.name, "Mt_target": mt}
        row.update(history_stats(case, start, end))
        row.update(uncertainty_stats(case))
        row.update(spectrum_stats(case))
        rows.append(row)
    return rows


def shade_forcing(ax, shell):
    if shell is None:
        return
    kmin, kmax = shell
    if np.isfinite(kmin) and np.isfinite(kmax) and kmax > kmin:
        ax.axvspan(kmin, kmax, alpha=0.07, label="forced shell")


def plot_spectra(cases, output_dir, dpi, chi_floor, absolute_floor):
    loaded = []
    forcing_shell = None
    for mt, case in cases:
        path = case / "spectra_diagnostics.npz"
        if not path.exists():
            continue
        with np.load(path) as s:
            required = {"k", "density_weighted_energy_mean", "solenoidal_energy_mean", "dilatational_energy_mean", "complete_shell_max"}
            if not required.issubset(s.files):
                continue
            rec = {
                "mt": mt, "k": np.asarray(s["k"], dtype=float),
                "Edw": np.asarray(s["density_weighted_energy_mean"], dtype=float),
                "Edw_std": np.asarray(s["density_weighted_energy_std"], dtype=float) if "density_weighted_energy_std" in s.files else None,
                "Es": np.asarray(s["solenoidal_energy_mean"], dtype=float),
                "Ed": np.asarray(s["dilatational_energy_mean"], dtype=float),
                "complete": int(np.asarray(s["complete_shell_max"])),
            }
            if "forcing_shell" in s.files:
                fs = np.asarray(s["forcing_shell"], dtype=float).ravel()
                if fs.size >= 2 and np.all(np.isfinite(fs[:2])):
                    forcing_shell = (float(fs[0]), float(fs[1]))
            loaded.append(rec)
    if len(loaded) < 2:
        return None

    plotted = []
    for rec in loaded:
        k, E = rec["k"], rec["Edw"]; sid=np.arange(k.size)
        valid=(k>0)&(sid<=rec["complete"])&np.isfinite(E)&(E>0)
        if np.any(valid): plotted.append(E[valid])
    vals=np.concatenate(plotted); peak=float(np.nanmax(vals)); pmin=float(np.nanmin(vals))
    ymin=max(0.5*pmin, max(float(absolute_floor),0.0)*peak); ymax=2.0*peak

    fig=plt.figure(figsize=(14.2,11.2), constrained_layout=True)
    gs=fig.add_gridspec(2,2)
    axes=[fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,:])]
    for rec in loaded:
        k,E,Es,Ed=rec["k"],rec["Edw"],rec["Es"],rec["Ed"]; sid=np.arange(k.size)
        valid=(k>0)&(sid<=rec["complete"])&np.isfinite(E)&(E>0); label=rf"$M_t={rec['mt']:.2f}$"
        line=axes[0].loglog(k[valid],E[valid],lw=2.0,label=label)[0]
        if rec["Edw_std"] is not None:
            lo=E-rec["Edw_std"]; hi=E+rec["Edw_std"]; sh=valid&np.isfinite(lo)&np.isfinite(hi)&(lo>ymin)&(hi>lo)
            if np.any(sh): axes[0].fill_between(k[sh],lo[sh],hi[sh],alpha=.05,color=line.get_color())
        total=float(np.sum(np.maximum(E,0.0)))
        if total>0: axes[1].loglog(k[valid],E[valid]/total,lw=2.0,label=label)
        helm=Es+Ed; threshold=max(float(np.nanmax(helm)),np.finfo(float).tiny)*chi_floor
        cvalid=(k>0)&(sid<=rec["complete"])&np.isfinite(helm)&np.isfinite(Ed)&(helm>threshold)
        chi=np.zeros_like(helm); np.divide(Ed,helm,out=chi,where=helm>0)
        axes[2].semilogx(k[cvalid],100*chi[cvalid],lw=2.0,label=label)
    for ax in axes:
        shade_forcing(ax,forcing_shell); ax.grid(True,which="both",alpha=.25); ax.tick_params(labelsize=13)
        h,l=ax.get_legend_handles_labels(); u={}
        for hh,ll in zip(h,l): u.setdefault(ll,hh)
        ax.legend(u.values(),u.keys(),fontsize=9.0,ncol=2 if ax is axes[2] else 1)
    axes[0].set_title("Density-weighted kinetic-energy spectrum",fontsize=17); axes[0].set_xlabel("k",fontsize=15); axes[0].set_ylabel(r"$E_{\sqrt{\rho}u}(k)$",fontsize=15); axes[0].set_ylim(ymin,ymax)
    axes[1].set_title("Normalized spectral shape",fontsize=17); axes[1].set_xlabel("k",fontsize=15); axes[1].set_ylabel(r"$E_{\sqrt{\rho}u}(k)/\sum_k E_{\sqrt{\rho}u}(k)$",fontsize=15)
    axes[2].set_title("Scale-dependent dilatational fraction",fontsize=17); axes[2].set_xlabel("k",fontsize=15); axes[2].set_ylabel(r"$\chi_d(k)$ [\%]",fontsize=15); axes[2].set_ylim(bottom=0)
    out=output_dir/"final_mach_spectra_polished.png"; fig.savefig(out,dpi=dpi,bbox_inches="tight"); plt.close(fig); return out

def fit_quadratic_through_origin(mt, chi):
    valid = np.isfinite(mt) & np.isfinite(chi)
    if np.count_nonzero(valid) < 3:
        return float("nan"), float("nan")
    x2 = mt[valid]**2
    a = float(np.dot(x2, chi[valid]) / np.dot(x2, x2))
    pred = a*x2
    ss_res = float(np.sum((chi[valid]-pred)**2))
    ss_tot = float(np.sum((chi[valid]-np.mean(chi[valid]))**2))
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    return a, r2


def plot_trends(rows, output_dir, target_re, dpi):
    rows = sorted(rows, key=lambda r: float(r["Mt_target"]))
    mt = np.array([float(r["Mt_target"]) for r in rows])
    chi = np.array([float(r.get("chi_d", np.nan)) for r in rows])
    chi_std = np.array([float(r.get("chi_d_temporal_std", np.nan)) for r in rows])
    weno = np.array([float(r.get("WENO_mean", np.nan)) for r in rows])
    weno_std = np.array([float(r.get("WENO_std", np.nan)) for r in rows])
    rel = np.array([float(r.get("Re_lambda_mean", np.nan)) for r in rows])
    rel_std = np.array([float(r.get("Re_lambda_std", np.nan)) for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.5), constrained_layout=True)

    valid = np.isfinite(mt) & np.isfinite(chi)
    yerr = np.where(np.isfinite(chi_std[valid]), 100*chi_std[valid], 0.0)
    axes[0].errorbar(mt[valid], 100*chi[valid], yerr=yerr, marker="o", capsize=3, lw=1.5, label="mean ± temporal 1σ")
    a, r2 = fit_quadratic_through_origin(mt, chi)
    if np.isfinite(a):
        xfit = np.linspace(0, max(mt)*1.03, 200)
        axes[0].plot(xfit, 100*a*xfit**2, "--", lw=1.3, label=rf"$\chi_d={a:.3f}M_t^2$, $R^2={r2:.3f}$")
    axes[0].set_title("Integrated dilatational energy", fontsize=22)
    axes[0].set_xlabel(r"$M_t$", fontsize=20)
    axes[0].set_ylabel(r"$\chi_d$ [\%]", fontsize=20)
    axes[0].tick_params(axis="both", which="both", labelsize=17)
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    valid = np.isfinite(mt) & np.isfinite(weno)
    axes[1].errorbar(
        mt[valid], 100*weno[valid],
        yerr=np.where(np.isfinite(weno_std[valid]), 100*weno_std[valid], 0.0),
        marker="o", capsize=3, lw=1.5,
    )
    axes[1].set_title("Shock-capturing activity", fontsize=22)
    axes[1].set_xlabel(r"$M_t$", fontsize=20)
    axes[1].set_ylabel("WENO node fraction [%]", fontsize=20)
    axes[1].tick_params(axis="both", which="both", labelsize=17)
    axes[1].set_ylim(bottom=0)
    axes[1].grid(True, alpha=0.3)

    valid = np.isfinite(mt) & np.isfinite(rel)
    axes[2].errorbar(
        mt[valid], rel[valid],
        yerr=np.where(np.isfinite(rel_std[valid]), rel_std[valid], 0.0),
        marker="o", capsize=3, lw=1.5, label="mean ± temporal 1σ",
    )
    axes[2].axhline(target_re, ls="--", lw=1.2, label=rf"target $Re_\lambda={target_re:g}$")
    axes[2].set_title("Taylor-scale Reynolds number", fontsize=22)
    axes[2].set_xlabel(r"$M_t$", fontsize=20)
    axes[2].set_ylabel(r"$Re_{\lambda,2D}$", fontsize=20)
    axes[2].tick_params(axis="both", which="both", labelsize=17)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=10)

    out = output_dir / "final_mach_trends_uncertainty.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pdf_moments(rows, output_dir, dpi):
    rows=sorted(rows,key=lambda r:float(r["Mt_target"])); mt=np.array([float(r["Mt_target"]) for r in rows])
    variables=[("dilatation","Dilatation"),("vorticity","Vorticity"),("pressure","Pressure fluctuation"),("density","Density fluctuation")]
    fig,axes=plt.subplots(1,2,figsize=(13.6,5.8),constrained_layout=True)
    for var,label in variables:
        skew=np.array([float(r.get(f"{var}_skewness_temporal_mean",np.nan)) for r in rows]); ss=np.array([float(r.get(f"{var}_skewness_temporal_std",np.nan)) for r in rows])
        flat=np.array([float(r.get(f"{var}_flatness_temporal_mean",np.nan)) for r in rows]); fs=np.array([float(r.get(f"{var}_flatness_temporal_std",np.nan)) for r in rows])
        g=np.isfinite(skew); axes[0].errorbar(mt[g],skew[g],yerr=np.where(np.isfinite(ss[g]),ss[g],0),marker="o",capsize=2.5,lw=1.3,label=label)
        g=np.isfinite(flat)&(flat>0); axes[1].errorbar(mt[g],flat[g],yerr=np.where(np.isfinite(fs[g]),fs[g],0),marker="o",capsize=2.5,lw=1.3,label=label)
    axes[0].axhline(0,lw=1); axes[0].set_title("Snapshot PDF skewness"); axes[0].set_xlabel(r"$M_t$"); axes[0].set_ylabel("skewness"); axes[0].grid(True,alpha=.3); axes[0].legend(fontsize=7.0,loc="best")
    axes[1].axhline(3,ls="--",lw=1,label="Gaussian flatness"); axes[1].set_title("Snapshot PDF flatness"); axes[1].set_xlabel(r"$M_t$"); axes[1].set_ylabel("flatness"); axes[1].set_yscale("log"); axes[1].grid(True,which="both",alpha=.3)
    h,l=axes[1].get_legend_handles_labels(); u={}
    for hh,ll in zip(h,l): u.setdefault(ll,hh)
    axes[1].legend(u.values(),u.keys(),fontsize=7.0,loc="upper left")
    fig.text(.5,-.01,"Error bars show snapshot-to-snapshot temporal standard deviation over the stationary interval.",ha="center",fontsize=8)
    out=output_dir/"final_pdf_moments_uncertainty.png"; fig.savefig(out,dpi=dpi,bbox_inches="tight"); plt.close(fig); return out

def nearest_case(cases, target):
    return min(cases, key=lambda item: abs(item[0]-target))


def snapshot_step(path):
    m = re.search(r"step(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def final_snapshot(case):
    paths = sorted(case.glob("hit2d_step*.npz"), key=snapshot_step)
    return paths[-1] if paths else None


def centered_rms(values):
    values = np.asarray(values, dtype=float)
    centered = values - float(np.mean(values))
    return centered, float(np.sqrt(np.mean(centered**2)))


def plot_normalized_fields(cases, targets, output_dir, dpi):
    selected = [nearest_case(cases, t) for t in targets]
    records = []
    for mt, case in selected:
        snap = final_snapshot(case)
        if snap is None:
            return None, []
        with np.load(snap) as d:
            x = np.asarray(d["x"], float)
            y = np.asarray(d["y"], float)
            rho = np.asarray(d["rho"], float)
            u = np.asarray(d["u"], float)
            v = np.asarray(d["v"], float)
            p = np.asarray(d["pressure"], float)
            vort = np.asarray(d["vorticity"], float)
            div = np.asarray(d["divergence"], float)
        cfg = json.loads((case / "config.json").read_text()) if (case / "config.json").exists() else {}
        gamma = float(cfg.get("gamma", 1.4))
        sound = np.sqrt(np.maximum(gamma * p / np.maximum(rho, np.finfo(float).tiny), 0))
        mach = np.sqrt(u**2 + v**2) / np.maximum(sound, np.finfo(float).tiny)
        vc, vr = centered_rms(vort)
        dc, dr = centered_rms(div)
        rc, rr = centered_rms(rho)
        records.append({
            "mt": mt, "x": x, "y": y,
            "vort": vc / max(vr, np.finfo(float).tiny),
            "div": dc / max(dr, np.finfo(float).tiny),
            "mach": mach / max(mt, np.finfo(float).tiny),
            "rho": rc / max(rr, np.finfo(float).tiny),
            "vort_rms": vr, "div_rms": dr, "rho_rms": rr,
            "mach_mean": float(np.mean(mach)), "mach_max": float(np.max(mach)),
        })

    def sl(key):
        return float(np.percentile(np.concatenate([np.abs(r[key]).ravel() for r in records]), 99.5))

    lim = {k: sl(k) for k in ("vort", "div", "rho")}
    mm = float(np.percentile(np.concatenate([r["mach"].ravel() for r in records]), 99.5))
    defs = [
        ("vort", r"$\omega_z/\omega_{rms}$", "coolwarm", True),
        ("div", r"$\theta/\theta_{rms}$", "coolwarm", True),
        ("mach", r"$M(x,y)/M_t$", "magma", False),
        ("rho", r"$\rho'/\rho'_{rms}$", "coolwarm", True),
    ]

    # Figure 5.10: full-page portrait layout.  The figure is intentionally
    # tall enough to use almost one thesis page while still leaving room below
    # it for the LaTeX caption.  Compact titles and reduced inter-panel spacing
    # maximize the physical size of the eight flow-field panels.
    #
    # A dedicated narrow colorbar column is used for each row.  Anchoring the
    # left and right image axes toward the center avoids the large empty gap
    # that equal-aspect images can otherwise create inside subplot cells.
    fig = plt.figure(figsize=(7.25, 10.85))
    gs = fig.add_gridspec(
        4, 3,
        width_ratios=(1.0, 1.0, 0.045),
        left=0.075, right=0.945, bottom=0.055, top=0.975,
        wspace=0.20, hspace=0.62,
    )
    axes = np.empty((4, 2), dtype=object)

    for row, (key, label, cmap, sym) in enumerate(defs):
        row_image = None
        for col, r in enumerate(records):
            ax = fig.add_subplot(gs[row, col])
            axes[row, col] = ax
            ext = [float(r["x"][0]), float(r["x"][-1]), float(r["y"][0]), float(r["y"][-1])]
            vmin, vmax = (-lim[key], lim[key]) if sym else (0, mm)
            row_image = ax.imshow(
                r[key], origin="lower", extent=ext, aspect="equal",
                cmap=cmap, vmin=vmin, vmax=vmax,
            )

            # Compact typography prevents collisions while preserving large
            # physical panels at printed-page size.
            ax.set_title(
                rf"{label}, $M_t={r['mt']:.2f}$",
                fontsize=8.6,
                pad=4.0,
                y=1.01,
            )
            ax.tick_params(
                axis="both",
                which="both",
                labelsize=8.0,
                pad=0.8,
            )

            # Avoid repeating axis labels on every panel.  The y label is shown
            # only in the left column and the x label only in the bottom row.
            if col == 0:
                ax.set_ylabel(r"$y$", fontsize=8.8, labelpad=0.8)
            else:
                ax.set_ylabel("")

            if row == 3:
                ax.set_xlabel(r"$x$", fontsize=8.8, labelpad=0.8)
            else:
                ax.set_xlabel("")

            # Bring the two square panels in each row toward the center while
            # retaining enough horizontal separation for the subplot titles.
            ax.set_anchor("E" if col == 0 else "W")

        if row_image is not None:
            cax = fig.add_subplot(gs[row, 2])
            cbar = fig.colorbar(row_image, cax=cax)
            cbar.ax.tick_params(labelsize=8.0, pad=1.5)

    # No overall title: the thesis caption supplies the context.  Keeping a
    # modest bottom margin leaves room for that caption once the figure is
    # inserted at approximately text width in LaTeX.
    out = output_dir / "final_low_high_fields_normalized.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out, records


def plot_stationarity(cases, targets, output_dir, dpi):
    """Generate the two half-page stationarity panels used in Figure 5.9."""
    outputs = []
    for target in targets:
        mt, case = nearest_case(cases, target)
        hist_path = case / "diagnostic_history.npz"
        if not hist_path.exists():
            continue
        time, turnover, _ = turnover_from_history(case)
        with np.load(hist_path) as h:
            if "kinetic_energy" not in h.files or "turbulent_mach" not in h.files:
                continue
            K = np.asarray(h["kinetic_energy"], dtype=float)
            Mt = np.asarray(h["turbulent_mach"], dtype=float)
        if K.shape != turnover.shape or Mt.shape != turnover.shape:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True, constrained_layout=True)
        axes[0].plot(turnover, K, linewidth=1.6)
        axes[0].set_title("Turbulent kinetic energy", fontsize=17)
        axes[0].set_ylabel(r"$K$", fontsize=17)

        axes[1].plot(turnover, Mt, linewidth=1.6)
        axes[1].set_title("Turbulent Mach number", fontsize=17)
        axes[1].set_ylabel(r"$M_t$", fontsize=17)
        axes[1].set_xlabel(r"$N_{\mathrm{eddy}}$", fontsize=17)

        for ax in axes:
            ax.tick_params(axis="both", which="both", labelsize=14)
            ax.grid(True, alpha=0.25)
        try:
            fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.035, hspace=0.10, wspace=0.05)
        except Exception:
            pass

        mt_tag = int(round(100 * mt))
        path = output_dir / f"stationarity_Mt{mt_tag:03d}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)
    return outputs

def write_uncertainty_tables(rows, field_records, output_dir):
    keys = [
        "case", "Mt_target", "snapshot_count", "chi_d", "chi_d_temporal_mean", "chi_d_temporal_std",
        "Mt_mean", "Mt_std", "Re_lambda_mean", "Re_lambda_std", "WENO_mean", "WENO_std",
        "dilatation_skewness_temporal_mean", "dilatation_skewness_temporal_std",
        "dilatation_flatness_temporal_mean", "dilatation_flatness_temporal_std",
        "vorticity_skewness_temporal_mean", "vorticity_skewness_temporal_std",
        "vorticity_flatness_temporal_mean", "vorticity_flatness_temporal_std",
        "pressure_skewness_temporal_mean", "pressure_skewness_temporal_std",
        "pressure_flatness_temporal_mean", "pressure_flatness_temporal_std",
        "density_skewness_temporal_mean", "density_skewness_temporal_std",
        "density_flatness_temporal_mean", "density_flatness_temporal_std",
    ]
    with (output_dir/"final_uncertainty_statistics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    md = [
        "# Final temporal-variability statistics",
        "",
        "> The ± values below are **snapshot-to-snapshot temporal standard deviations** over the stationary interval. They are not confidence intervals or numerical error bars.",
        "",
        "| $M_t$ | snapshots | $\\chi_d$ [\\%] | $Re_{\\lambda,2D}$ | WENO [\\%] | $S_\\theta$ | $F_\\theta$ |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: float(x["Mt_target"])):
        def val(k): return float(r.get(k, np.nan))
        md.append(
            f"| {val('Mt_target'):.2f} | {int(val('snapshot_count')) if np.isfinite(val('snapshot_count')) else 0} | "
            f"{100*val('chi_d'):.2f} ± {100*val('chi_d_temporal_std'):.2f} | "
            f"{val('Re_lambda_mean'):.1f} ± {val('Re_lambda_std'):.1f} | "
            f"{100*val('WENO_mean'):.2f} ± {100*val('WENO_std'):.2f} | "
            f"{val('dilatation_skewness_temporal_mean'):.2f} ± {val('dilatation_skewness_temporal_std'):.2f} | "
            f"{val('dilatation_flatness_temporal_mean'):.1f} ± {val('dilatation_flatness_temporal_std'):.1f} |"
        )
    (output_dir/"final_uncertainty_statistics.md").write_text("\n".join(md)+"\n")

    if field_records:
        with (output_dir/"final_field_normalization_summary.csv").open("w", newline="") as f:
            keys2 = ["Mt", "vorticity_rms", "dilatation_rms", "density_fluctuation_rms", "local_mach_mean", "local_mach_max"]
            w = csv.DictWriter(f, fieldnames=keys2); w.writeheader()
            for r in field_records:
                w.writerow({
                    "Mt": r["mt"], "vorticity_rms": r["vort_rms"], "dilatation_rms": r["div_rms"],
                    "density_fluctuation_rms": r["rho_rms"], "local_mach_mean": r["mach_mean"], "local_mach_max": r["mach_max"],
                })


def main():
    a = parse_args()
    root = a.root.resolve()
    prod = (a.production_dir or root/"production_N512").resolve()
    out = (a.output_dir or root/"thesis_postprocess").resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = discover_cases(prod)
    rows = collect_rows(cases, a.start_turnover, a.end_turnover)

    outputs = []
    for path in (
        plot_spectra(cases, out, a.dpi, a.chi_relative_floor, a.absolute_spectrum_relative_floor),
        plot_trends(rows, out, a.target_re, a.dpi),
        plot_pdf_moments(rows, out, a.dpi),
    ):
        if path is not None:
            outputs.append(path)
    outputs.extend(plot_stationarity(cases, tuple(a.stationarity_machs), out, a.dpi))
    field_path, field_records = plot_normalized_fields(cases, tuple(a.field_machs), out, a.dpi)
    if field_path is not None: outputs.append(field_path)
    write_uncertainty_tables(rows, field_records, out)

    print("Final polished outputs:")
    for p in outputs:
        print(f"  {p}")
    print(f"  {out/'final_uncertainty_statistics.csv'}")
    print(f"  {out/'final_uncertainty_statistics.md'}")
    print(f"  {out/'final_field_normalization_summary.csv'}")


if __name__ == "__main__":
    main()
