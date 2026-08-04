from __future__ import annotations

from pathlib import Path
import argparse
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_dispatch_args() -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(
        description="Unified launcher for HIT2D and the 2D Riemann benchmarks."
    )
    parser.add_argument(
        "--problem",
        choices=("hit2d", "riemann3", "riemann3_08", "riemann6"),
        required=True,
        help="Problem to run. Remaining arguments are forwarded to that problem driver.",
    )
    args, remaining = parser.parse_known_args()
    return args.problem, remaining


def main() -> None:
    problem, remaining = parse_dispatch_args()
    sys.argv = [sys.argv[0], *remaining]
    if problem == "hit2d":
        from OOP.hit2d import main as hit_main

        hit_main()
        return
    if problem == "riemann3":
        from scripts.run_riemann_config3 import main as riemann_main

        riemann_main()
        return
    if problem == "riemann3_08":
        from scripts.run_riemann_config3_08 import main as riemann_main

        riemann_main()
        return
    if problem == "riemann6":
        from scripts.run_riemann_config6 import main as riemann_main

        riemann_main()
        return
    raise ValueError(f"Unsupported problem: {problem}")


if __name__ == "__main__":
    main()
