from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OOP.problems.riemann_config3 import plot_from_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate a Riemann Configuration 6 plot from a saved .npz file."
    )
    parser.add_argument("npz", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--density-contours", action="store_true")
    parser.add_argument("--num-contours", type=int, default=40)
    parser.add_argument("--schlieren", action="store_true")
    parser.add_argument("--schlieren-k", type=float, default=20.0)
    parser.add_argument("--zoom-center", action="store_true")
    parser.add_argument("--zoom-window", type=float, default=0.35)
    parser.add_argument("--vorticity-limit", type=float, default=100.0, help="Use 0 for automatic scaling.")
    parser.add_argument("--fixed-density-limits", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--fixed-pressure-limits", type=float, nargs=2, metavar=("MIN", "MAX"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.npz.with_suffix(".png")
    plot_from_npz(
        args.npz,
        output_path=output,
        show=args.show,
        density_contours=args.density_contours,
        num_contours=args.num_contours,
        schlieren=args.schlieren,
        schlieren_k=args.schlieren_k,
        zoom_center=args.zoom_center,
        zoom_window=args.zoom_window,
        vorticity_limit=None if args.vorticity_limit == 0.0 else args.vorticity_limit,
        fixed_density_limits=tuple(args.fixed_density_limits) if args.fixed_density_limits else None,
        fixed_pressure_limits=tuple(args.fixed_pressure_limits) if args.fixed_pressure_limits else None,
    )
    print(f"saved figure: {output}")


if __name__ == "__main__":
    main()
