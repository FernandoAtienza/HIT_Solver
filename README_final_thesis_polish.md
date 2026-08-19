# Final thesis figure-polishing and uncertainty pass

This package performs the **last post-processing-only pass** for the final
`512^2` Mach-number campaign. It does not modify the solver and does not launch
new CFD simulations.

## What changes

1. The scale-dependent dilatational fraction is now shown only while

\[
E_s(k)+E_d(k) > 10^{-4}\max_k[E_s(k)+E_d(k)],
\]

by default. This removes ratios formed from essentially zero energy in the
numerical cutoff tail.

2. The code computes snapshot-to-snapshot temporal variability over

\[
4\leq N_{eddy}\leq16
\]

for:

- integrated dilatational fraction `chi_d`;
- dilatation, vorticity, pressure and density RMS;
- skewness;
- flatness.

The displayed `±1 sigma` bars are **temporal standard deviations**, not
confidence intervals and not numerical-discretization error bars.

3. A normalized low/high-Mach structural comparison is generated using

\[
\omega_z/\omega_{rms},\qquad
\theta/\theta_{rms},\qquad
M(x,y)/M_t,\qquad
\rho'/\rho'_{rms}.
\]

This complements the existing common-dimensional-scale figure, which should be
kept when discussing the growth in fluctuation magnitude.

## New outputs

The script writes the following into
`results/hit2d/final_thesis_Mach_ReL130_N512/thesis_postprocess/`:

- `final_mach_spectra_polished.png`
- `final_mach_trends_uncertainty.png`
- `final_pdf_moments_uncertainty.png`
- `final_low_high_fields_normalized.png`
- `final_uncertainty_statistics.csv`
- `final_uncertainty_statistics.md`
- `final_field_normalization_summary.csv`

Each production case also receives a cached
`thesis_uncertainty_diagnostics.npz` file.

## Install

```bash
cd ~/github/HIT_Solver
unzip -o ~/Downloads/final_thesis_polish_update.zip -d .
chmod +x \
  scripts/compute_final_thesis_uncertainty.py \
  scripts/plot_final_thesis_polished.py \
  scripts/postprocess_final_thesis_polished.sh
```

## Run

```bash
cd ~/github/HIT_Solver
./scripts/postprocess_final_thesis_polished.sh
```

To force the uncertainty values to be recomputed:

```bash
RECOMPUTE_UNCERTAINTY=1 ./scripts/postprocess_final_thesis_polished.sh
```

To use a different energetic cutoff for the `chi_d(k)` figure:

```bash
CHI_FLOOR=1e-5 ./scripts/postprocess_final_thesis_polished.sh
```

The default `1e-4` is intentionally conservative for the final thesis figure.
