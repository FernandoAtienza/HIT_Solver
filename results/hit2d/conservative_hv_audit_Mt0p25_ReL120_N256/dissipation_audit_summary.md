# HIT dissipation-audit summary

| Case | mn | CFL | HV mode | Mean Re_lambda,2D | Mean K | Mean Mt | Mean WENO fraction | Physical dissipation | HV drain | HV / physical | HV mass change | Mean mn/(N dt) | Mean chi_d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mn0p001_CFL0p10 | 0.001 | 0.1 | conservative_flux | 130.053 | 0.0309605 | 0.250516 | 0.0618925 | 0.000587094 | 2.58703e-06 | 0.00440651 | -6.8394e-10 | 0.228608 | 0.0111382 |
| mn0p002_CFL0p05 | 0.002 | 0.05 | conservative_flux | 120.317 | 0.0308262 | 0.249938 | 0.0675527 | 0.000673989 | 1.11774e-05 | 0.016584 | -4.40923e-09 | 0.933108 | 0.0125307 |
| mn0p002_CFL0p10 | 0.002 | 0.1 | conservative_flux | 127.373 | 0.0308867 | 0.250495 | 0.0603701 | 0.00059617 | 4.95256e-06 | 0.00830729 | -1.10595e-09 | 0.463664 | 0.0122055 |
| mn0p002_CFL0p20 | 0.002 | 0.2 | conservative_flux | 123.321 | 0.0309327 | 0.250536 | 0.0695721 | 0.000650109 | 2.90849e-06 | 0.00447385 | 7.78527e-10 | 0.232335 | 0.0148871 |
| mn0p005_CFL0p10 | 0.005 | 0.1 | conservative_flux | 129.864 | 0.0309329 | 0.250414 | 0.0622428 | 0.00056959 | 1.11966e-05 | 0.0196573 | -1.11322e-09 | 1.14615 | 0.0122677 |
| mn0p010_CFL0p10 | 0.01 | 0.1 | conservative_flux | 130.155 | 0.030857 | 0.250284 | 0.0603031 | 0.000576351 | 2.24982e-05 | 0.0390355 | 4.29281e-10 | 2.32635 | 0.0116426 |

The hyperviscosity-to-physical ratio compares the measured kinetic-energy change caused by the discrete hyperviscosity filter with the physical viscous dissipation over the selected turnover interval.
