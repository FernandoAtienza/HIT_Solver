# Final thesis post-processing — enlarged fonts update

This patch supersedes the previous `thesis_final_results_format_patch.zip`.

It changes **plotting/post-processing only**. No CFD simulation is launched and
no saved numerical results are modified.

## Why this update is needed

In the previous launcher, the detailed `M_t=0.25` diagnostics were executed
through the thesis-font wrapper, but the two cross-Mach plotting scripts were
called directly. Consequently, these figures still had comparatively small
axes/title/tick text:

- `mach_pdfs.png`
- `mach_spectra_polished.png`
- `mach_trends_uncertainty.png`
- `pdf_moments_uncertainty.png`

This version sends both cross-Mach scripts through the same font-control layer.
It also enlarges the `reference_Mt025_fields.png` rendering and colorbar tick
labels.

## Default font targets

Detailed `M_t=0.25` figures:

- subplot titles: 22 pt
- axis labels: 20 pt
- tick labels: 17 pt
- legends: 14 pt
- figure title: 24 pt

Cross-Mach figures:

- subplot titles: 21 pt
- axis labels: 19 pt
- tick labels: 16 pt
- legends: 10 pt
- figure title: 23 pt

The cross-Mach legends are intentionally smaller than the axes text because five
Mach-number curves must remain visible without the legend covering them.

## Install

```bash
cd ~/github/HIT_Solver

unzip -o \
  ~/Downloads/thesis_final_results_large_fonts_patch.zip \
  -d .

chmod +x \
  scripts/run_with_thesis_fonts.py \
  scripts/postprocess_thesis_final_format.sh
```

## Re-run all thesis post-processing

```bash
cd ~/github/HIT_Solver

./scripts/postprocess_thesis_final_format.sh
```

No simulation is rerun.

The complete updated thesis figure collection is written to:

```text
results/hit2d/thesis_figures_final_layout/
```

The files specifically affected by this update are:

```text
reference_Mt025_fields.png
mach_pdfs.png
mach_spectra_polished.png
mach_trends_uncertainty.png
pdf_moments_uncertainty.png
```

The other reference-case figures are regenerated with the same large-font style
as well.

## Optional font overrides

If the paper PDF still reduces text too much, for example:

```bash
REF_TITLE_FONT=24 \
REF_LABEL_FONT=22 \
REF_TICK_FONT=18 \
REF_LEGEND_FONT=15 \
CROSS_TITLE_FONT=23 \
CROSS_LABEL_FONT=21 \
CROSS_TICK_FONT=18 \
CROSS_LEGEND_FONT=11 \
./scripts/postprocess_thesis_final_format.sh
```
