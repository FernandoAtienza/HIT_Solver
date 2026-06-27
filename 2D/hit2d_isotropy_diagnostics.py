from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OOP.postprocess import HIT2DSpectra, IsotropyDiagnostics2D


DEFAULT_SNAPSHOT_DIR = Path(__file__).resolve().parent / "hit2d_snapshots"


def resolve_snapshot_dir(snapshot_dir: Path) -> Path:
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


def run_id_from_snapshot_dir(snapshot_dir: Path) -> str | None:
    if not snapshot_dir.name.startswith("run_"):
        return None
    return snapshot_dir.name.removeprefix("run_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute stationarity and isotropy diagnostics from saved HIT2D snapshots."
    )
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument(
        "--fluctuation-type",
        choices=("reynolds", "favre"),
        default="reynolds",
    )
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument(
        "--start-snapshot",
        type=int,
        default=None,
        help="First snapshot-list index included in the averaging interval.",
    )
    parser.add_argument(
        "--end-snapshot",
        type=int,
        default=None,
        help="Snapshot-list stop index, exclusive.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--lx", type=float, default=2.0 * np.pi)
    parser.add_argument("--ly", type=float, default=2.0 * np.pi)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="NPZ path. Defaults to <run_dir>/isotropy_diagnostics.npz.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Plot directory. Defaults to <run_dir>/postprocess.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-spectra", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_snapshot_dir(args.snapshot_dir)
    if run_dir != args.snapshot_dir:
        print(f"using latest snapshot run folder: {run_dir}")

    diagnostics = IsotropyDiagnostics2D(
        run_dir=run_dir,
        fluctuation_type=args.fluctuation_type,
        gamma=args.gamma,
        start_time=args.start_time,
        end_time=args.end_time,
        start_snapshot=args.start_snapshot,
        end_snapshot=args.end_snapshot,
        stride=args.stride,
        Lx=args.lx,
        Ly=args.ly,
        output_dir=args.plot_dir or run_dir / "postprocess",
    )
    results = diagnostics.compute()
    output_path = diagnostics.save(args.output)
    print(f"saved isotropy diagnostics: {output_path}")
    print(
        "selected interval: "
        f"t=[{results.selected_time_interval[0]:.6g}, "
        f"{results.selected_time_interval[1]:.6g}], "
        f"snapshots={results.number_of_snapshots}"
    )
    print(
        "isotropy mismatch: "
        f"E_LL={results.E_LL:.6e} "
        f"(normalized={results.E_LL_normalized:.6e}), "
        f"E_NN={results.E_NN:.6e} "
        f"(normalized={results.E_NN_normalized:.6e})"
    )
    selected = results.selected_mask
    print(
        "selected means: "
        f"K={np.mean(results.K[selected]):.6e}, "
        f"Mt={np.mean(results.Mt[selected]):.6e}, "
        f"A_K={np.mean(results.A_K[selected]):.6e}, "
        f"C_uv={np.mean(results.C_uv[selected]):.6e}"
    )

    if not args.no_plots:
        for path in diagnostics.plot_all(
            filename_suffix=run_id_from_snapshot_dir(run_dir)
        ):
            print(f"saved isotropy plot: {path}")

    if not args.no_spectra:
        spectra = HIT2DSpectra(
            run_dir=run_dir,
            fluctuation_type=args.fluctuation_type,
            start_time=args.start_time,
            end_time=args.end_time,
            stride=args.stride,
            output_dir=args.plot_dir or run_dir / "postprocess",
        )
        spectrum_results = spectra.compute()
        spectrum_output = spectra.save()
        spectrum_plot = spectra.plot()
        print(f"saved spectra diagnostics: {spectrum_output}")
        print(f"saved spectra plot: {spectrum_plot}")
        print(
            "cutoff ratios: "
            f"energy={spectrum_results.high_k_energy_ratio:.3e}, "
            f"enstrophy={spectrum_results.high_k_enstrophy_ratio:.3e}"
        )


if __name__ == "__main__":
    main()
