# Thesis HIT campaign summary

All statistics use the common selected turnover-time interval stored in each case.

| Case | N | Mt target | Mt mean | Reλ mean | A_K | C_uv | E_LL,n | E_NN,n | WENO | χ_d | HV/physical | kmax η | kmax ηΩ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grid_Mt025_N128_seed1234 | 128 | 0.25 | 0.2504 | 131.87 | 0.11365 | -0.01399 | 0.078103 | 0.083423 | 0.12492 | 0.015257 | 0.11158 | 1.8931 | 3.4697 |
| grid_Mt025_N256_seed1234 | 256 | 0.25 | 0.25025 | 130.58 | 0.13544 | -0.041398 | 0.15462 | 0.24372 | 0.061448 | 0.012993 | 0.019617 | 3.7561 | 6.6442 |
| grid_Mt025_N512_seed1234 | 512 | 0.25 | 0.25011 | 130.18 | 0.082879 | -0.037222 | 0.090655 | 0.086422 | 0.03709 | 0.012688 | 0.0025439 | 7.577 | 13.396 |
| mach_Mt010_N256_seed1234 | 256 | 0.1 | 0.10008 | 131.59 | 0.10858 | -0.081241 | 0.10989 | 0.036693 | 0.021263 | 0.0020497 | 0.034884 | 3.7854 | 6.8885 |
| mach_Mt050_N256_seed1234 | 256 | 0.5 | 0.50051 | 113.11 | 0.083708 | 0.052224 | 0.034231 | 0.11163 | 0.15906 | 0.049684 | 0.013829 | 3.6429 | 6.3852 |
| mach_Mt060_N256_seed1234 | 256 | 0.6 | 0.60056 | 101.31 | 0.085886 | 0.029556 | 0.18765 | 0.24311 | 0.2235 | 0.069872 | 0.012281 | 3.4895 | 6.0703 |
| repeat_Mt025_N256_seed5678 | 256 | 0.25 | 0.24999 | 127.27 | 0.088817 | -0.038862 | 0.10768 | 0.18259 | 0.069938 | 0.016391 | 0.019978 | 3.7246 | 6.588 |

## Numerical baseline

- sensor: legacy shock sensor
- WENO splitting: local stencil Lax–Friedrichs
- conservative compact-region hyperviscosity: mn=0.005
- hyperviscosity interval: every 5 complete RK steps
- CFL: 0.10
- initial 2-D Taylor Reynolds number: 120
- forcing shell: 3 <= k <= 5
- statistics window: 4 <= N_eddy <= 16

## Interpretation checklist

- Grid convergence is supported when N=256 and N=512 overlap over their common resolved spectral range and their stationary bulk statistics agree.
- Cross-Mach comparisons are interpretable only if the stationary Re_lambda values remain reasonably close; otherwise viscosity should be recalibrated before the final 512² Mach campaign.
- A rising dilatational fraction with Mt is a compressibility result only after grid convergence and comparable Re_lambda are established.
- WENO activity should increase only when compressive structures require it; excessive domain fractions at higher Mt should trigger a sensor review.
- The independent-seed Mt=0.25 case estimates stochastic sensitivity of isotropy and spectral statistics.
