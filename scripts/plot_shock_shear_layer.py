from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OOP.problems.shock_shear_layer import plot_from_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate shock--shear-layer figures from a final NPZ file."
    )
    parser.add_argument("npz", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-contours", type=int, default=31)
    parser.add_argument(
        "--density-limits",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.4, 2.60),
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = plot_from_npz(
        args.npz,
        output_dir=args.output_dir,
        num_contours=args.num_contours,
        density_limits=tuple(args.density_limits),
        show=args.show,
    )
    for label, path in paths.items():
        print(f"saved {label}: {path}")


if __name__ == "__main__":
    main()
