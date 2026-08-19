# Final 2-D HIT thesis campaign

Statistical interval: approximately 4 <= N_eddy <= 16.
Target stationary Re_lambda,2D = 130.

| Mt | mean Mt | K | Reλ | Re error | χd | WENO | HV/visc | A_K | E_LL,n | E_NN,n |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.1004 | 0.00504 | 129.43±8.79 | -0.44% | 0.22% | 1.49% | 0.50% | 0.080 | 0.038 | 0.068 |
| 0.25 | 0.2502 | 0.03081 | 130.76±7.68 | +0.59% | 1.27% | 3.71% | 0.25% | 0.083 | 0.091 | 0.086 |
| 0.40 | 0.4003 | 0.07729 | 132.69±6.75 | +2.07% | 2.91% | 6.16% | 0.26% | 0.101 | 0.060 | 0.145 |
| 0.50 | 0.4996 | 0.11877 | 125.24±6.17 | -3.66% | 5.07% | 7.20% | 0.24% | 0.078 | 0.203 | 0.145 |
| 0.60 | 0.5989 | 0.17067 | 122.18±7.75 | -6.01% | 7.15% | 9.92% | 0.33% | 0.108 | 0.050 | 0.126 |

## Interpretation checklist

- Confirm all retained cases remain within the accepted Reynolds-number band.
- Use the N=512 cases as the final Mach-number dataset.
- Compare absolute and normalized density-weighted spectra across Mach number.
- Compare solenoidal/dilatational spectra and integrated chi_d.
- Compare WENO activity and PDF intermittency versus Mach number.
- Keep the already completed N=128/256/512 Mt=0.25 study as the grid-convergence evidence.

After this campaign, freeze the solver and perform only post-processing/writing unless a case has actually failed.
