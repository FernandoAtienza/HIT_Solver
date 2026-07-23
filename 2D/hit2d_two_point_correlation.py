from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OOP.postprocess import TwoPointCorrelation2D


DEFAULT_SNAPSHOT_DIR = Path(__file__).resolve().parent / "hit2d_snapshots"


def resolve_snapshot_dir(snapshot_dir: Path) -> Path:
    """Use snapshot_dir, or its newest child run folder when snapshots live there."""

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
    name = snapshot_dir.name
    if not name.startswith("run_"):
        return None
    return name.removeprefix("run_")


def default_postprocess_dir(run_dir: Path) -> Path:
    """Default location for post-processing figures belonging to one run."""

    return run_dir / "postprocess"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute two-point velocity correlations from saved HIT2D snapshots."
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR,
        help="HIT2D run folder, or the parent hit2d_snapshots folder.",
    )
    parser.add_argument(
        "--fluctuation-type",
        choices=("reynolds", "favre"),
        default="reynolds",
        help="Velocity fluctuation definition.",
    )
    parser.add_argument("--start", type=int, default=None, help="First snapshot index to use.")
    parser.add_argument("--stop", type=int, default=None, help="Stop snapshot index, exclusive.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth selected snapshot.")
    parser.add_argument(
        "--tot-initial-data",
        "--tot_initial_data",
        "--ToT_initial_data",
        dest="tot_initial_data",
        type=float,
        default=None,
        help="First turnover included; defaults to the simulation metadata.",
    )
    parser.add_argument(
        "--tot-final-data",
        "--tot_final_data",
        "--ToT_final_data",
        dest="tot_final_data",
        type=float,
        default=None,
        help="Last turnover included; defaults to the simulation target.",
    )
    parser.add_argument(
        "--turnover-length",
        type=float,
        default=None,
        help="Override the turnover reference length saved by the run.",
    )
    parser.add_argument("--max-snapshots", type=int, default=None, help="Limit number of snapshots.")
    parser.add_argument("--lx", type=float, default=2.0 * np.pi, help="Domain length in x.")
    parser.add_argument("--ly", type=float, default=2.0 * np.pi, help="Domain length in y.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz path. Defaults to <run_dir>/two_point_correlation_results.npz.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for PNG plots. Defaults to <run_dir>/postprocess.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Only save the .npz result.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_snapshot_dir(args.snapshot_dir)
    if run_dir != args.snapshot_dir:
        print(f"using latest snapshot run folder: {run_dir}")

    corr = TwoPointCorrelation2D(
        run_dir=run_dir,
        fluctuation_type=args.fluctuation_type,
        Lx=args.lx,
        Ly=args.ly,
        output_dir=run_dir,
    )
    selected = corr.load_snapshots(
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        max_snapshots=args.max_snapshots,
        start_turnover=args.tot_initial_data,
        end_turnover=args.tot_final_data,
        turnover_length=args.turnover_length,
    )
    print(f"processing {len(selected)} snapshots from {run_dir}")
    if corr.selection_metadata:
        print(
            "selected interval: "
            f"Neddy=[{corr.selection_metadata['selected_turnover_start']:.6g}, "
            f"{corr.selection_metadata['selected_turnover_end']:.6g}], "
            f"t=[{corr.selection_metadata['selected_time_start']:.6g}, "
            f"{corr.selection_metadata['selected_time_end']:.6g}]"
        )
    results = corr.compute()
    output_path = corr.save(args.output)
    print(f"saved correlation results: {output_path}")
    print(
        "summary: "
        f"L_integral={results.L_integral:.6e}, "
        f"lambda_taylor={results.lambda_taylor:.6e}, "
        f"fluctuation_type={results.fluctuation_type}"
    )

    if not args.no_plots:
        plot_paths = corr.plot(
            args.plot_dir or default_postprocess_dir(run_dir),
            filename_suffix=run_id_from_snapshot_dir(run_dir),
        )
        for path in plot_paths:
            print(f"saved correlation plot: {path}")


if __name__ == "__main__":
    main()
