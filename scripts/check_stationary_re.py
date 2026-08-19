#!/usr/bin/env python3
"""Check stationary Taylor-Reynolds number from an HIT run log."""

from __future__ import annotations
import argparse, math, re
from pathlib import Path
from statistics import mean, pstdev

RE_LINE = re.compile(
    r"t=(?P<t>[0-9.eE+-]+).*?"
    r"Mt=(?P<Mt>[0-9.eE+-]+).*?"
    r"ReL2D=(?P<Re>[0-9.eE+-]+)"
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-log", required=True)
    p.add_argument("--mach", type=float, required=True)
    p.add_argument("--target", type=float, default=130.0)
    p.add_argument("--tolerance", type=float, default=0.05)
    p.add_argument("--start-turnover", type=float, default=4.0)
    p.add_argument("--end-turnover", type=float, default=16.0)
    a = p.parse_args()

    Lref = math.pi / 2
    t0 = a.start_turnover * Lref / a.mach
    t1 = a.end_turnover * Lref / a.mach

    rows = []
    for line in Path(a.run_log).read_text(errors="ignore").splitlines():
        m = RE_LINE.search(line)
        if not m:
            continue
        d = {k: float(v) for k, v in m.groupdict().items()}
        if t0 <= d["t"] <= t1:
            rows.append(d)

    if not rows:
        print("No stationary-window Re_lambda samples found.")
        raise SystemExit(2)

    re_vals = [r["Re"] for r in rows]
    mt_vals = [r["Mt"] for r in rows]
    re_mean = mean(re_vals)
    re_std = pstdev(re_vals) if len(re_vals) > 1 else 0.0
    mt_mean = mean(mt_vals)
    error = (re_mean - a.target) / a.target

    print(
        f"stationary check: Mt={mt_mean:.5f}, "
        f"Re_lambda={re_mean:.3f} ± {re_std:.3f}, "
        f"target={a.target:.3f}, error={100*error:+.2f}%"
    )

    raise SystemExit(0 if abs(error) <= a.tolerance else 1)

if __name__ == "__main__":
    main()
