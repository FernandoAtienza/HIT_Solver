from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OOP.problems.shock_shear_layer import (
    ShockShearLayerConfig,
    run_shock_shear_layer,
    save_run_outputs,
)
from OOP.run_utils import make_run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the two-dimensional viscous shock--shear-layer interaction "
            "from Kang and Lee (2026), Section 3.2."
        )
    )
    parser.add_argument("--nx", type=int, default=500)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--tfinal", type=float, default=120.0)
    parser.add_argument(
        "--cfl",
        type=float,
        default=0.4,
        help="Initial convective CFL used to define one fixed time step.",
    )
    parser.add_argument("--viscous-cfl", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--prandtl", type=float, default=0.72)
    parser.add_argument("--reynolds", type=float, default=500.0)
    parser.add_argument("--backend", choices=("auto", "numpy", "cupy"), default="cupy")
    parser.add_argument("--scheme", choices=("hybrid", "weno"), default="hybrid")
    parser.add_argument(
        "--viscosity-model",
        choices=("sutherland", "constant"),
        default="sutherland",
        help=(
            "Sutherland matches the paper's solver description. Constant "
            "viscosity is available for the classical benchmark variant."
        ),
    )
    parser.add_argument("--reference-temperature-k", type=float, default=300.0)
    parser.add_argument("--sutherland-constant-k", type=float, default=110.4)

    parser.add_argument("--sensor-width", type=int, default=2)
    parser.add_argument("--jump-threshold", type=float, default=0.04)
    parser.add_argument("--compression-threshold", type=float, default=2.5)
    parser.add_argument(
        "--shear-threshold",
        type=float,
        default=0.0,
        help="Use 0 to disable WENO activation based on physical shear.",
    )
    parser.add_argument("--boundary-guard", type=int, default=4)
    parser.add_argument("--guard-cells", type=int, default=4)

    parser.add_argument("--mn", type=float, default=5.0e-4)
    parser.add_argument("--hyperviscosity-interval", type=int, default=5)
    parser.add_argument("--hyperviscosity-density-weight", type=float, default=1.0)
    parser.add_argument("--hyperviscosity-momentum-weight", type=float, default=1.0)
    parser.add_argument("--hyperviscosity-energy-weight", type=float, default=1.0)

    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/shock_shear_layer"),
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--num-contours", type=int, default=31)
    parser.add_argument(
        "--density-limits",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.4, 2.60),
    )
    parser.add_argument(
        "--vorticity-limit",
        type=float,
        default=0.0,
        help="Use 0 for percentile-based automatic scaling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ShockShearLayerConfig(
        nx=args.nx,
        ny=args.ny,
        tfinal=args.tfinal,
        cfl=args.cfl,
        viscous_cfl=args.viscous_cfl,
        gamma=args.gamma,
        prandtl=args.prandtl,
        reynolds=args.reynolds,
        backend=args.backend,
        scheme=args.scheme,
        viscosity_model=args.viscosity_model,
        reference_temperature_kelvin=args.reference_temperature_k,
        sutherland_constant_kelvin=args.sutherland_constant_k,
        sensor_width=args.sensor_width,
        jump_threshold=args.jump_threshold,
        compression_threshold=args.compression_threshold,
        shear_threshold=None if args.shear_threshold == 0.0 else args.shear_threshold,
        boundary_guard=args.boundary_guard,
        guard_cells=args.guard_cells,
        mn=args.mn,
        hyperviscosity_interval=args.hyperviscosity_interval,
        hyperviscosity_density_weight=args.hyperviscosity_density_weight,
        hyperviscosity_momentum_weight=args.hyperviscosity_momentum_weight,
        hyperviscosity_energy_weight=args.hyperviscosity_energy_weight,
        progress_every=args.progress_every,
    )
    run_id = args.run_id or make_run_id(
        f"shock_shear_{args.scheme}_N{args.nx}x{args.ny}_t{args.tfinal:g}".replace(
            ".", "p"
        )
    )
    q, diagnostics = run_shock_shear_layer(config)
    paths = save_run_outputs(
        q,
        config,
        diagnostics,
        output_dir=args.output_dir,
        run_id=run_id,
        plot=not args.no_plot,
        show=args.show,
        num_contours=args.num_contours,
        density_limits=tuple(args.density_limits),
        vorticity_limit=(
            None if args.vorticity_limit == 0.0 else args.vorticity_limit
        ),
    )
    for label, path in paths.items():
        print(f"saved {label}: {path}")


if __name__ == "__main__":
    main()
