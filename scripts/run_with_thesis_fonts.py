#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base", type=float, default=17)
    parser.add_argument("--title", type=float, default=19)
    parser.add_argument("--label", type=float, default=18)
    parser.add_argument("--tick", type=float, default=15)
    parser.add_argument("--legend", type=float, default=15)
    parser.add_argument("--suptitle", type=float, default=21)
    parser.add_argument("--annotation", type=float, default=13)
    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument("script")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))

    from OOP.postprocess.thesis_plot_style import (
        ThesisPlotStyle,
        apply_thesis_style,
    )

    apply_thesis_style(
        ThesisPlotStyle(
            base=args.base,
            axes_title=args.title,
            axes_label=args.label,
            ticks=args.tick,
            legend=args.legend,
            figure_title=args.suptitle,
            colorbar_label=args.label,
            annotation=args.annotation,
            min_dpi=args.dpi,
        )
    )

    script = Path(args.script)
    if not script.is_absolute():
        script = repo / script

    if not script.exists():
        raise FileNotFoundError(f"Plotting script not found: {script}")

    forwarded = args.script_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]

    sys.argv = [str(script), *forwarded]

    runpy.run_path(
        str(script),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
