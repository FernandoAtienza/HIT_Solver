#!/usr/bin/env python3
"""Create a compact thesis-campaign summary from completed run logs/postprocess logs."""

from __future__ import annotations
import argparse, csv, math, re
from pathlib import Path
from statistics import mean, pstdev

RUN_RE = re.compile(
    r"t=(?P<t>[0-9.eE+-]+).*?"
    r"KE=(?P<K>[0-9.eE+-]+),\s*"
    r"Mt=(?P<Mt>[0-9.eE+-]+).*?"
    r"WENO=(?P<WENO>[0-9.eE+-]+).*?"
    r"ReL2D=(?P<Re>[0-9.eE+-]+).*?"
    r"eps_nu=(?P<eps>[0-9.eE+-]+).*?"
    r"P_hv=(?P<Phv>[0-9.eE+-]+)"
)

ISO_RE = re.compile(
    r"isotropy mismatch:\s*E_LL=(?P<ELL>[0-9.eE+-]+)\s*"
    r"\(normalized=(?P<ELLN>[0-9.eE+-]+)\),\s*"
    r"E_NN=(?P<ENN>[0-9.eE+-]+)\s*"
    r"\(normalized=(?P<ENNN>[0-9.eE+-]+)\)"
)
MEANS_RE = re.compile(
    r"selected means:\s*K=(?P<Kiso>[0-9.eE+-]+),\s*"
    r"Mt=(?P<Mtiso>[0-9.eE+-]+),\s*"
    r"A_K=(?P<AK>[0-9.eE+-]+),\s*C_uv=(?P<Cuv>[0-9.eE+-]+)"
)
SPEC_RE = re.compile(
    r"spectral checks:.*?<chi_d>=(?P<chi>[0-9.eE+-]+)"
)
CUTOFF_RE = re.compile(
    r"cutoff ratios:\s*energy=(?P<Ecut>[0-9.eE+-]+),\s*enstrophy=(?P<Zcut>[0-9.eE+-]+)"
)

def extract_mt(case_name: str):
    m = re.search(r"Mt0?(\d+)", case_name)
    if not m:
        return float("nan")
    s = m.group(1)
    if len(s) == 2:
        return int(s) / 100.0
    if len(s) == 3:
        return int(s) / 1000.0
    return float("nan")

def read_run(case: Path, mt_target, start_turnover, end_turnover):
    Lref = math.pi / 2
    t0 = start_turnover * Lref / mt_target
    t1 = end_turnover * Lref / mt_target
    rows = []
    for line in (case / "run.log").read_text(errors="ignore").splitlines():
        m = RUN_RE.search(line)
        if not m:
            continue
        d = {k: float(v) for k, v in m.groupdict().items()}
        if t0 <= d["t"] <= t1:
            rows.append(d)
    return rows

def avg(rows, key):
    vals = [r[key] for r in rows]
    return mean(vals) if vals else float("nan")

def std(rows, key):
    vals = [r[key] for r in rows]
    return pstdev(vals) if len(vals) > 1 else 0.0 if vals else float("nan")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--target-re", type=float, default=130.0)
    p.add_argument("--start-turnover", type=float, default=4.0)
    p.add_argument("--end-turnover", type=float, default=16.0)
    a = p.parse_args()

    root = Path(a.root)
    prod = root / "production_N512"
    recs = []

    if not prod.exists():
        raise SystemExit("No production_N512 directory found.")

    for case in sorted(p for p in prod.iterdir() if p.is_dir()):
        mt = extract_mt(case.name)
        if not math.isfinite(mt):
            continue
        if not (case / "run.log").exists():
            continue

        rows = read_run(case, mt, a.start_turnover, a.end_turnover)
        if not rows:
            continue

        iso_text = (case / "isotropy_postprocess.log").read_text(errors="ignore") \
            if (case / "isotropy_postprocess.log").exists() else ""

        iso = ISO_RE.search(iso_text)
        means = MEANS_RE.search(iso_text)
        spec = SPEC_RE.search(iso_text)
        cutoff = CUTOFF_RE.search(iso_text)

        re_mean = avg(rows, "Re")
        re_std = std(rows, "Re")
        eps = avg(rows, "eps")
        phv = avg(rows, "Phv")

        rec = {
            "case": case.name,
            "Mt_target": mt,
            "Mt_mean": avg(rows, "Mt"),
            "K_mean": avg(rows, "K"),
            "Re_lambda_mean": re_mean,
            "Re_lambda_std": re_std,
            "Re_error_percent": 100*(re_mean-a.target_re)/a.target_re,
            "WENO_mean": avg(rows, "WENO"),
            "eps_nu_mean": eps,
            "P_hv_mean": phv,
            "HV_to_viscous_percent": 100*phv/eps if eps else float("nan"),
            "A_K": float(means.group("AK")) if means else float("nan"),
            "C_uv": float(means.group("Cuv")) if means else float("nan"),
            "E_LL_normalized": float(iso.group("ELLN")) if iso else float("nan"),
            "E_NN_normalized": float(iso.group("ENNN")) if iso else float("nan"),
            "chi_d": float(spec.group("chi")) if spec else float("nan"),
            "energy_cutoff_ratio": float(cutoff.group("Ecut")) if cutoff else float("nan"),
            "enstrophy_cutoff_ratio": float(cutoff.group("Zcut")) if cutoff else float("nan"),
        }
        recs.append(rec)

    if not recs:
        raise SystemExit("No completed N512 production cases found.")

    csv_path = root / "final_thesis_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)

    md = [
        "# Final 2-D HIT thesis campaign",
        "",
        f"Statistical interval: approximately {a.start_turnover:g} <= N_eddy <= {a.end_turnover:g}.",
        f"Target stationary Re_lambda,2D = {a.target_re:g}.",
        "",
        "| Mt | mean Mt | K | Reλ | Re error | χd | WENO | HV/visc | A_K | E_LL,n | E_NN,n |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(recs, key=lambda x:x["Mt_target"]):
        md.append(
            f"| {r['Mt_target']:.2f} | {r['Mt_mean']:.4f} | {r['K_mean']:.5f} | "
            f"{r['Re_lambda_mean']:.2f}±{r['Re_lambda_std']:.2f} | "
            f"{r['Re_error_percent']:+.2f}% | {100*r['chi_d']:.2f}% | "
            f"{100*r['WENO_mean']:.2f}% | {r['HV_to_viscous_percent']:.2f}% | "
            f"{r['A_K']:.3f} | {r['E_LL_normalized']:.3f} | {r['E_NN_normalized']:.3f} |"
        )

    md += [
        "",
        "## Interpretation checklist",
        "",
        "- Confirm all retained cases remain within the accepted Reynolds-number band.",
        "- Use the N=512 cases as the final Mach-number dataset.",
        "- Compare absolute and normalized density-weighted spectra across Mach number.",
        "- Compare solenoidal/dilatational spectra and integrated chi_d.",
        "- Compare WENO activity and PDF intermittency versus Mach number.",
        "- Keep the already completed N=128/256/512 Mt=0.25 study as the grid-convergence evidence.",
        "",
        "After this campaign, freeze the solver and perform only post-processing/writing unless a case has actually failed.",
    ]

    md_path = root / "final_thesis_summary.md"
    md_path.write_text("\n".join(md) + "\n")

    print(md_path)
    print(csv_path)

if __name__ == "__main__":
    main()
