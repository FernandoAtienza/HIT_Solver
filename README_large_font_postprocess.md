# Thesis large-font post-processing

This package adds a plotting-only style layer to the HIT solver repository.

It does **not** change:
- solver equations;
- numerical method;
- saved simulation states;
- spectra;
- PDFs;
- diagnostics.

It only regenerates the existing thesis plots with larger fonts.

Default sizes:

```text
subplot titles = 19 pt
axis labels    = 18 pt
tick labels    = 15 pt
legends        = 15 pt
figure titles  = 21 pt
DPI            = >= 300
```

Run:

```bash
cd ~/github/HIT_Solver
./scripts/postprocess_thesis_results_large_fonts.sh
```

For even larger multi-panel figures:

```bash
TITLE_FONT=22 \
LABEL_FONT=20 \
TICK_FONT=17 \
LEGEND_FONT=17 \
SUPTITLE_FONT=24 \
ANNOTATION_FONT=15 \
./scripts/postprocess_thesis_results_large_fonts.sh
```

A convenience copy of all regenerated figures is collected in:

```text
results/hit2d/thesis_figures_large_fonts/
```
