# Final thesis post-processing — spectrum-axis fix

This update fixes only the presentation of the absolute density-weighted kinetic-energy spectrum in `final_mach_spectra_polished.png`.

The previous uncertainty band used `np.finfo(float).tiny` whenever `E(k)-sigma_E(k)` became negative. On a logarithmic y-axis that artificial value (~1e-308) forced the absolute-spectrum panel to span hundreds of decades. The underlying spectra were not wrong.

The corrected plotting routine:

- determines the absolute-spectrum y-range from the positive **mean spectra**, not from clipped uncertainty-band values;
- draws the temporal uncertainty envelope only where `E(k)-sigma_E(k)` remains positive and inside the meaningful plotted range;
- leaves every spectral value and all saved diagnostics unchanged;
- retains the `chi_d(k)` energetic cutoff introduced in the previous polishing pass.

The default lower plotting safeguard is `1e-16` times the largest mean spectral value. It is only a display floor and does not alter data.

## Install

```bash
cd ~/github/HIT_Solver

unzip -o ~/Downloads/final_thesis_spectrum_plot_fix.zip -d .

chmod +x scripts/postprocess_final_thesis_polished.sh
```

## Rerun the polished post-processing

```bash
cd ~/github/HIT_Solver

./scripts/postprocess_final_thesis_polished.sh
```

The existing `thesis_uncertainty_diagnostics.npz` files are reused automatically, so no CFD simulation is rerun and the expensive snapshot-by-snapshot uncertainty calculation is not repeated.

To force uncertainty recomputation only if desired:

```bash
RECOMPUTE_UNCERTAINTY=1 ./scripts/postprocess_final_thesis_polished.sh
```

To manually modify the spectrum plotting floor, for example:

```bash
ABS_SPECTRUM_FLOOR=1e-15 ./scripts/postprocess_final_thesis_polished.sh
```

Outputs remain under:

```text
results/hit2d/final_thesis_Mach_ReL130_N512/thesis_postprocess/
```
