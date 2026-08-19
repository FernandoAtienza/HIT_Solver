# Final thesis HIT post-processing

This package is **post-processing only**. It does not modify the CFD solver and does not launch new simulations.

It uses the frozen final campaign under:

```text
results/hit2d/final_thesis_Mach_ReL130_N512/production_N512/
```

and evaluates the common stationary interval

\[
4 \le N_{eddy} \le 16.
\]

## Outputs

The script creates `results/hit2d/final_thesis_Mach_ReL130_N512/thesis_postprocess/` containing:

- `final_mach_spectra.png`
  - absolute density-weighted spectrum \(E_{\sqrt{\rho}u}(k)\);
  - spectrum normalized by its total kinetic energy;
  - scale-dependent Helmholtz dilatational fraction \(\chi_d(k)\).
- `final_mach_trends.png`
  - integrated dilatational fraction versus \(M_t\), including an empirical through-origin \(aM_t^2\) fit;
  - WENO node fraction versus \(M_t\);
  - stationary \(Re_{\lambda,2D}\) versus \(M_t\) with temporal standard-deviation bars.
- `final_mach_pdfs.png`
  - dilatation, vorticity, pressure-fluctuation and density-fluctuation PDFs for all Mach numbers.
- `final_pdf_moments_vs_mach.png`
  - skewness and flatness of the same four variables versus Mach number.
- `final_low_high_fields.png`
  - final-state comparison between \(M_t=0.10\) and \(M_t=0.60\), if the full snapshots remain available.
- `final_cross_mach_statistics.csv`
- `final_cross_mach_statistics.md`
- `final_cross_mach_table.tex`

The normalized density-weighted spectrum is computed as

\[
\widetilde E(k)=\frac{E_{\sqrt{\rho}u}(k)}{\sum_k E_{\sqrt{\rho}u}(k)}.
\]

To avoid plotting ratios of two near-zero spectral tails, the scale-dependent dilatational fraction is shown only where the total Helmholtz energy exceeds $10^{-6}$ of its peak by default. This threshold can be changed with `--chi-relative-floor`.

The scale-dependent dilatational fraction is

\[
\chi_d(k)=\frac{E_d(k)}{E_s(k)+E_d(k)}.
\]

The \(\chi_d\)-versus-Mach figure also displays a purely empirical fit of the form

\[
\chi_d=aM_t^2,
\]

which must be described as a fit over the investigated cases, **not as a universal scaling law**.

## Install

From the repository root:

```bash
cd ~/github/HIT_Solver
unzip -o ~/Downloads/final_thesis_postprocess.zip -d .
chmod +x scripts/postprocess_final_thesis_campaign.sh
```

## Run

```bash
cd ~/github/HIT_Solver
./scripts/postprocess_final_thesis_campaign.sh
```

To force regeneration of each case's spectra/PDF/isotropy diagnostics from the original snapshots:

```bash
REPROCESS=1 ./scripts/postprocess_final_thesis_campaign.sh
```

No simulation is rerun.
