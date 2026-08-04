from __future__ import annotations

from pathlib import Path
import argparse
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OOP.problems.riemann_config3 import RiemannConfig3, run_riemann_config3, save_run_outputs
from OOP.run_utils import make_run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 2D Euler Riemann problem, Configuration 3, with the OOP solver.")
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--ny", type=int, default=512)
    parser.add_argument("--tfinal", type=float, default=0.3)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--backend", choices=("auto", "numpy", "cupy"), default="numpy")
    parser.add_argument("--scheme", choices=("hybrid", "weno"), default="hybrid")
    parser.add_argument("--x-split", type=float, default=0.5)
    parser.add_argument("--y-split", type=float, default=0.5)
    parser.add_argument("--sensor-width", type=int, default=4)
    parser.add_argument("--jump-threshold", type=float, default=0.025)
    parser.add_argument("--compression-threshold", type=float, default=2.5)
    parser.add_argument("--shear-threshold", type=float, default=0.0, help="Use 0 to disable the shear part of the sensor.")
    parser.add_argument("--boundary-guard", type=int, default=4)
    parser.add_argument("--mn", type=float, default=0.001)
    parser.add_argument("--hyperviscosity-interval", type=int, default=1)
    parser.add_argument("--hyperviscosity-density-weight", type=float, default=1.0)
    parser.add_argument("--hyperviscosity-momentum-weight", type=float, default=1.0)
    parser.add_argument("--hyperviscosity-energy-weight", type=float, default=1.0)
    parser.add_argument("--guard-cells", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/riemann_config3"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true")
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
    config = RiemannConfig3(
        nx=args.nx,
        ny=args.ny,
        tfinal=args.tfinal,
        cfl=args.cfl,
        gamma=args.gamma,
        backend=args.backend,
        scheme=args.scheme,
        x_split=args.x_split,
        y_split=args.y_split,
        sensor_width=args.sensor_width,
        jump_threshold=args.jump_threshold,
        compression_threshold=args.compression_threshold,
        shear_threshold=None if args.shear_threshold == 0.0 else args.shear_threshold,
        boundary_guard=args.boundary_guard,
        mn=args.mn,
        hyperviscosity_interval=args.hyperviscosity_interval,
        hyperviscosity_density_weight=args.hyperviscosity_density_weight,
        hyperviscosity_momentum_weight=args.hyperviscosity_momentum_weight,
        hyperviscosity_energy_weight=args.hyperviscosity_energy_weight,
        guard_cells=args.guard_cells,
        progress_every=args.progress_every,
    )
    run_id = args.run_id or make_run_id(f"riemann3_{args.scheme}_t{args.tfinal:.3f}".replace(".", "p"))
    q, diagnostics = run_riemann_config3(config)
    paths = save_run_outputs(
        q,
        config,
        diagnostics,
        args.output_dir,
        run_id=run_id,
        plot=not args.no_plot,
        no_show=not args.show,
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
    for label, path in paths.items():
        print(f"saved {label}: {path}")


if __name__ == "__main__":
    main()
