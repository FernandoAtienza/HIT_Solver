# HIT sensor / LF audit

| Case | Sensor | LF | Mean K | Mean Mt | Mean Reλ,2D | Mean A_K | Mean C_uv | WENO | WENO x | WENO y | Physical diss. | HV drain | HV/physical | High-k E | High-k Z |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| directional_global | directional | global | 0.0309575 | 0.250606 | 134.776 | 0.186186 | -0.058153 | 0 | 0 | 0 | 0.000554212 | 1.19742e-05 | 0.0216057 | 0 | 0 |
| directional_local | directional | local | 0.0309575 | 0.250606 | 134.776 | 0.186186 | -0.058153 | 0 | 0 | 0 | 0.000554212 | 1.19742e-05 | 0.0216057 | 0 | 0 |
| gated_global | compression_gated | global | 0.0309575 | 0.250606 | 134.776 | 0.186186 | -0.058153 | 0 | 0 | 0 | 0.000554212 | 1.19742e-05 | 0.0216057 | 0 | 0 |
| gated_local | compression_gated | local | 0.0309575 | 0.250606 | 134.776 | 0.186186 | -0.058153 | 0 | 0 | 0 | 0.000554212 | 1.19742e-05 | 0.0216057 | 0 | 0 |
| legacy_global | legacy | global | 0.0309523 | 0.250459 | 130.547 | 0.133671 | -0.126989 | 0.0608795 | 0.0608795 | 0.0608795 | 0.000564416 | 1.11753e-05 | 0.0197997 | 0 | 1.05557 |
| legacy_local | legacy | local | 0.0309541 | 0.250465 | 130.541 | 0.133517 | -0.127106 | 0.0610963 | 0.0610963 | 0.0610963 | 0.000564493 | 1.11735e-05 | 0.0197938 | 0 | 1.13311 |

## Selection logic

Prefer the least dissipative configuration that remains stable, preserves positive density/pressure, 
keeps high-k energy/enstrophy free of pile-up, reduces false WENO activation, and does not degrade isotropy.
Local LF should be judged from changes at resolved/intermediate k, not only the final cutoff.
