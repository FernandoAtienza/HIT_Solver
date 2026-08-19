#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

# Allow imports from repository root when executed as scripts/foo.py.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OOP.postprocess.turnover import turnover_from_history
from OOP.turbulence_statistics import helmholtz_fourier_2d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute snapshot-to-snapshot temporal variability for the final HIT "
            "thesis campaign. The reported standard deviations represent temporal "
            "variability, not confidence intervals."
        )
    )
    p.add_argument("--snapshot-dir", type=Path, required=True)
    p.add_argument("--start-turnover", type=float, default=4.0)
    p.add_argument("--end-turnover", type=float, default=16.0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def _snapshot_step(path: Path) -> int:
    m = re.search(r"step(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def _central_moments(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    centered = values - float(np.mean(values))
    variance = float(np.mean(centered**2))
    rms = math.sqrt(max(variance, 0.0))
    if variance <= np.finfo(float).tiny:
        return rms, 0.0, 0.0
    skewness = float(np.mean(centered**3) / variance**1.5)
    flatness = float(np.mean(centered**4) / variance**2)
    return rms, skewness, flatness


def _snapshot_chi_d(rho: np.ndarray, u: np.ndarray, v: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    rho = np.asarray(rho, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    u_fluct = u - float(np.mean(u))
    v_fluct = v - float(np.mean(v))
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    us_hat, vs_hat, ud_hat, vd_hat = helmholtz_fourier_2d(u_fluct, v_fluct, dx, dy)

    normalization = float((x.size * y.size) ** 2)
    rho_mean = float(np.mean(rho))
    es = 0.5 * rho_mean * float(
        np.sum(np.abs(us_hat) ** 2 + np.abs(vs_hat) ** 2) / normalization
    )
    ed = 0.5 * rho_mean * float(
        np.sum(np.abs(ud_hat) ** 2 + np.abs(vd_hat) ** 2) / normalization
    )
    total = es + ed
    return ed / total if total > np.finfo(float).tiny else 0.0


def main() -> None:
    args = parse_args()
    run_dir = args.snapshot_dir.resolve()
    if args.stride <= 0:
        raise SystemExit("--stride must be positive")
    if args.end_turnover <= args.start_turnover:
        raise SystemExit("--end-turnover must be greater than --start-turnover")

    paths = sorted(run_dir.glob("hit2d_step*.npz"), key=_snapshot_step)
    if not paths:
        raise SystemExit(f"No HIT snapshots found in {run_dir}")

    snapshot_times = []
    for path in paths:
        with np.load(path) as data:
            snapshot_times.append(float(np.asarray(data["time"])))
    snapshot_times = np.asarray(snapshot_times, dtype=float)

    _, turnover, length_ref = turnover_from_history(run_dir, snapshot_times=snapshot_times)
    selected_indices = np.flatnonzero(
        (turnover >= args.start_turnover) & (turnover <= args.end_turnover)
    )[:: args.stride]
    if selected_indices.size == 0:
        raise SystemExit(
            f"No snapshots lie in {args.start_turnover} <= N_eddy <= {args.end_turnover}"
        )

    fields = {
        "chi_d": [],
        "dilatation_rms": [],
        "dilatation_skewness": [],
        "dilatation_flatness": [],
        "vorticity_rms": [],
        "vorticity_skewness": [],
        "vorticity_flatness": [],
        "pressure_rms": [],
        "pressure_skewness": [],
        "pressure_flatness": [],
        "density_rms": [],
        "density_skewness": [],
        "density_flatness": [],
    }
    selected_times = []
    selected_turnover = []
    selected_steps = []

    for idx in selected_indices:
        path = paths[int(idx)]
        with np.load(path) as data:
            required = {"x", "y", "rho", "u", "v", "pressure", "vorticity", "divergence", "time", "step"}
            missing = required.difference(data.files)
            if missing:
                raise KeyError(f"{path} is missing keys: {sorted(missing)}")
            x = np.asarray(data["x"], dtype=float)
            y = np.asarray(data["y"], dtype=float)
            rho = np.asarray(data["rho"], dtype=float)
            u = np.asarray(data["u"], dtype=float)
            v = np.asarray(data["v"], dtype=float)
            pressure = np.asarray(data["pressure"], dtype=float)
            vorticity = np.asarray(data["vorticity"], dtype=float)
            divergence = np.asarray(data["divergence"], dtype=float)
            time = float(np.asarray(data["time"]))
            step = int(np.asarray(data["step"]))

        fields["chi_d"].append(_snapshot_chi_d(rho, u, v, x, y))

        for name, values in (
            ("dilatation", divergence),
            ("vorticity", vorticity),
            ("pressure", pressure),
            ("density", rho),
        ):
            rms, skew, flat = _central_moments(values)
            fields[f"{name}_rms"].append(rms)
            fields[f"{name}_skewness"].append(skew)
            fields[f"{name}_flatness"].append(flat)

        selected_times.append(time)
        selected_turnover.append(float(turnover[int(idx)]))
        selected_steps.append(step)
        print(
            f"{run_dir.name}: step={step}, t={time:.6g}, "
            f"N_eddy={turnover[int(idx)]:.4f}, chi_d={fields['chi_d'][-1]:.5f}"
        )

    payload: dict[str, np.ndarray] = {
        "snapshot_time": np.asarray(selected_times, dtype=float),
        "snapshot_turnover": np.asarray(selected_turnover, dtype=float),
        "snapshot_step": np.asarray(selected_steps, dtype=int),
        "number_of_snapshots": np.asarray(len(selected_times), dtype=int),
        "selected_turnover_interval": np.asarray(
            [float(selected_turnover[0]), float(selected_turnover[-1])], dtype=float
        ),
        "turnover_length": np.asarray(length_ref, dtype=float),
        "note": np.asarray(
            "Standard deviations are snapshot-to-snapshot temporal variability, not confidence intervals."
        ),
    }
    for key, values in fields.items():
        arr = np.asarray(values, dtype=float)
        payload[key] = arr
        payload[f"{key}_mean"] = np.asarray(float(np.mean(arr)))
        payload[f"{key}_std"] = np.asarray(float(np.std(arr)))

    output = args.output
    if output is None:
        output = run_dir / "thesis_uncertainty_diagnostics.npz"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)

    print(f"Saved: {output}")
    print(f"Selected snapshots: {len(selected_times)}")
    print(
        "chi_d = "
        f"{float(payload['chi_d_mean']):.6f} ± {float(payload['chi_d_std']):.6f} "
        "(temporal 1σ)"
    )


if __name__ == "__main__":
    main()
