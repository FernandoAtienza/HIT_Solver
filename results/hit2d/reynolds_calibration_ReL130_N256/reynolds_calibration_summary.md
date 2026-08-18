# Stationary Reynolds-number calibration

Target stationary $Re_{\lambda,2D} = 130.000$.

| Case | Mt | viscosity | mean Reλ | std Reλ | error | mean Mt | WENO | recommended μ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| calib_Mt040_N256_mu0p001140 | 0.40 | 0.00114 | 125.32 | 6.93 | -3.60% | 0.4001 | 0.1243 | 0.001098989 |
| calib_Mt050_N256_mu0p001316 | 0.50 | 0.00131565 | 130.25 | 6.96 | +0.19% | 0.5013 | 0.1535 | 0.00131818 |
| calib_Mt060_N256_mu0p001414 | 0.60 | 0.00141405 | 122.36 | 6.07 | -5.87% | 0.6007 | 0.2765 | 0.001330976 |

## Decision rule

- Within ±3% of the target: viscosity is sufficiently calibrated for the 512² production pilot.
- Between 3% and 5%: usually acceptable, but one more 256² correction is preferable.
- Beyond ±5%: rerun that Mach number at 256² using the recommended viscosity.

The recommended viscosity uses the local first-order estimate

\[
\mu_{\mathrm{next}}=\mu_{\mathrm{current}}\frac{\overline{Re_\lambda}}{Re_{\lambda,\mathrm{target}}}.
\]

Use the full isotropy post-processing as the authoritative statistical analysis; this helper is intended for viscosity calibration.
