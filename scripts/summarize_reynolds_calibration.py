#!/usr/bin/env python3
"""Summarize stationary-Re calibration runs from their run.log files.

The authoritative turnover-window post-processing remains
2D/hit2d_isotropy_diagnostics.py.  This helper uses the extremely accurate
Mach control to map the desired turnover interval to physical time:
    N_eddy ~= t * Mt / L_ref,  L_ref = pi/2.
That is sufficient for selecting the next viscosity iteration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev

LINE_RE = re.compile(
    r"step=\s*(?P<step>\d+),\s*"
    r"t=(?P<t>[0-9.eE+-]+).*?"
    r"KE=(?P<K>[0-9.eE+-]+),\s*"
    r"Mt=(?P<Mt>[0-9.eE+-]+).*?"
    r"WENO=(?P<WENO>[0-9.eE+-]+).*?"
    r"ReL2D=(?P<Re>[0-9.eE+-]+).*?"
    r"eps_nu=(?P<eps>[0-9.eE+-]+).*?"
    r"P_hv=(?P<Phv>[0-9.eE+-]+)"
)


def parse_log(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="ignore").splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        rows.append({k: float(v) for k, v in m.groupdict().items()})
    return rows


def fmean(rows, key):
    vals = [r[key] for r in rows]
    return mean(vals) if vals else float("nan")


def fstd(rows, key):
    vals = [r[key] for r in rows]
    return pstdev(vals) if len(vals) >= 2 else 0.0 if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--target-re", type=float, default=130.0)
    ap.add_argument("--start-turnover", type=float, default=4.0)
    ap.add_argument("--end-turnover", type=float, default=16.0)
    args = ap.parse_args()

    root = Path(args.root)
    Lref = math.pi / 2.0
    records = []

    for case in sorted(p for p in root.iterdir() if p.is_dir()):
        cfg_path = case / "config.json"
        log_path = case / "run.log"
        if not cfg_path.exists() or not log_path.exists():
            continue

        cfg = json.loads(cfg_path.read_text())
        mt_target = float(cfg.get("target_mach", cfg.get("mach_control_target", float("nan"))))
        mu = float(cfg.get("viscosity", cfg.get("resolved_dynamic_viscosity", float("nan"))))

        rows = parse_log(log_path)
        if not rows or not math.isfinite(mt_target) or mt_target <= 0:
            continue

        # Approximate physical-time bounds corresponding to the desired
        # turnover interval. Mach control is sufficiently tight for calibration.
        t0 = args.start_turnover * Lref / mt_target
        t1 = args.end_turnover * Lref / mt_target
        selected = [r for r in rows if t0 <= r["t"] <= t1]
        if not selected:
            selected = rows[len(rows)//2:]

        re_mean = fmean(selected, "Re")
        re_std = fstd(selected, "Re")
        mt_mean = fmean(selected, "Mt")
        k_mean = fmean(selected, "K")
        weno_mean = fmean(selected, "WENO")
        eps_mean = fmean(selected, "eps")
        phv_mean = fmean(selected, "Phv")

        # First-order viscosity correction assuming Re ~ 1/mu locally.
        mu_recommended = mu * re_mean / args.target_re
        rel_error = (re_mean - args.target_re) / args.target_re

        records.append({
            "case": case.name,
            "Mt_target": mt_target,
            "viscosity": mu,
            "stationary_Re_mean": re_mean,
            "stationary_Re_std": re_std,
            "Re_relative_error": rel_error,
            "Mt_mean": mt_mean,
            "K_mean": k_mean,
            "WENO_mean": weno_mean,
            "eps_nu_mean": eps_mean,
            "P_hv_mean": phv_mean,
            "recommended_viscosity_next": mu_recommended,
            "selected_samples": len(selected),
            "approx_t0": t0,
            "approx_t1": t1,
        })

    if not records:
        raise SystemExit("No completed calibration cases found.")

    csv_path = root / "reynolds_calibration_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

    md_path = root / "reynolds_calibration_summary.md"
    lines = [
        "# Stationary Reynolds-number calibration",
        "",
        f"Target stationary $Re_{{\\lambda,2D}} = {args.target_re:.3f}$.",
        "",
        "| Case | Mt | viscosity | mean Reλ | std Reλ | error | mean Mt | WENO | recommended μ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        lines.append(
            f"| {r['case']} | {r['Mt_target']:.2f} | "
            f"{r['viscosity']:.7g} | {r['stationary_Re_mean']:.2f} | "
            f"{r['stationary_Re_std']:.2f} | {100*r['Re_relative_error']:+.2f}% | "
            f"{r['Mt_mean']:.4f} | {r['WENO_mean']:.4f} | "
            f"{r['recommended_viscosity_next']:.7g} |"
        )
    lines += [
        "",
        "## Decision rule",
        "",
        "- Within ±3% of the target: viscosity is sufficiently calibrated for the 512² production pilot.",
        "- Between 3% and 5%: usually acceptable, but one more 256² correction is preferable.",
        "- Beyond ±5%: rerun that Mach number at 256² using the recommended viscosity.",
        "",
        "The recommended viscosity uses the local first-order estimate",
        "",
        r"\[",
        r"\mu_{\mathrm{next}}=\mu_{\mathrm{current}}\frac{\overline{Re_\lambda}}{Re_{\lambda,\mathrm{target}}}.",
        r"\]",
        "",
        "Use the full isotropy post-processing as the authoritative statistical analysis; "
        "this helper is intended for viscosity calibration.",
    ]
    md_path.write_text("\n".join(lines) + "\n")

    print(md_path)
    for r in records:
        print(
            f"{r['case']}: Re={r['stationary_Re_mean']:.2f} "
            f"({100*r['Re_relative_error']:+.2f}%), "
            f"recommended mu={r['recommended_viscosity_next']:.7g}"
        )


if __name__ == "__main__":
    main()
