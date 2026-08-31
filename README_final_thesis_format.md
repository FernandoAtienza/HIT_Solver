# Final Results-figure formatting patch

This patch changes post-processing only. It does not modify or rerun the CFD solver.

## Changes

- `Mt=0.25` diagnostic figures are regenerated with larger titles, axes, ticks and legends.
- The `Mt=0.25` PDF figure contains only the four marginal PDFs + Gaussian references (2x2).
  The joint PDF and pooled moments remain saved in `pdf_diagnostics.npz` for textual/table discussion.
- Low/high Mach normalized fields use a four-row by two-column layout: one physical field per row and the two Mach cases side by side. This avoids the former four-panels-across layout while preserving all eight maps.
- The cross-Mach spectral figure uses a triangular layout: two panels on top, dilatational fraction below.
- The scale-dependent dilatational-fraction threshold note is removed from inside the plot.
- Cross-Mach PDF legends are smaller, especially in the dilatation panel.
- PDF-moment legends are smaller and the flatness legend is placed at upper left.

## Install

```bash
cd ~/github/HIT_Solver
unzip -o ~/Downloads/thesis_final_results_format_patch.zip -d .
chmod +x scripts/run_with_thesis_fonts.py scripts/postprocess_thesis_final_format.sh
```

## Run

```bash
cd ~/github/HIT_Solver
./scripts/postprocess_thesis_final_format.sh
```

No simulation is launched. Existing snapshots/diagnostics are reused.

The exact thesis-ready filenames are collected in:

```text
results/hit2d/thesis_figures_final_layout/
```

If the reference-case or campaign roots differ, override them:

```bash
REF_CASE=results/hit2d/solenoidal_Mt025_N512_CFL_0p05 \
FINAL_ROOT=results/hit2d/final_thesis_Mach_ReL130_N512 \
./scripts/postprocess_thesis_final_format.sh
```
