# Dilatational-forcing extension

## Purpose

This update adds a purely dilatational (compressive/curl-free) stochastic
forcing option to the 2D compressible HIT solver while preserving the existing
solenoidal behavior as the default.

## Mathematical construction

A real scalar Ornstein-Uhlenbeck potential is evolved in the forced Fourier
shell. For every nonzero wavevector **k** = (kx, ky), the acceleration is formed
as either

- solenoidal: f_hat = i (ky, -kx) phi_hat,
- dilatational: f_hat = i (kx,  ky) phi_hat.

Consequently, k dot f_hat = 0 in the solenoidal case and k cross f_hat = 0 in
the dilatational case. This is the two-dimensional pure-mode form of the
Helmholtz projection used in turbulence-forcing studies.

## Code changes

- `OOP/forcing.py`
  - adds `mode={solenoidal,dilatational}`;
  - accepts `compressive` as an alias;
  - preserves the previous solenoidal branch;
  - records the selected forcing-mode fractions in diagnostics.
- `OOP/hit2d.py`
  - adds `HIT2DConfig.forcing_mode`;
  - adds `--forcing-mode` to the CLI;
  - stores the mode in `diagnostic_history.npz`;
  - adds Helmholtz velocity-energy diagnostics:
    `solenoidal_velocity_energy_fraction`,
    `dilatational_velocity_energy_fraction`, and
    `dilatational_to_solenoidal_energy_ratio`.
- `scripts/run_dilatational_hit_campaign.sh`
  - runs the Mt=0.25 and Mt=0.50 N=512 campaigns sequentially;
  - uses the same parameters as the existing solenoidal campaign except for
    `--forcing-mode dilatational`.
- `tests/test_forcing_modes.py`
  - verifies divergence-free solenoidal forcing;
  - verifies curl-free dilatational forcing;
  - verifies the `compressive` alias.

## Initial condition

The initial velocity field remains divergence-free for both forcing campaigns.
This deliberately keeps the initial condition identical and isolates the effect
of changing the forcing. The first four turnover times should continue to be
excluded from stationary statistics.

## Literature basis

The pure solenoidal/compressive split follows the Helmholtz-projection forcing
framework used by Federrath et al. (2010), while the finite-correlation-time
stochastic spectral forcing is consistent with the forcing tradition of
Eswaran and Pope (1988). No parameter values were copied from those works: the
existing shell, correlation time, power control, and thermodynamic treatment of
this repository were retained.

References:

- Federrath, C., Roman-Duval, J., Klessen, R. S., Schmidt, W., and Mac Low,
  M.-M. (2010). *Comparing the statistics of interstellar turbulence in
  simulations and observations: Solenoidal versus compressive turbulence
  forcing*. Astronomy & Astrophysics, 512, A81.
  DOI: 10.1051/0004-6361/200912437.
- Eswaran, V., and Pope, S. B. (1988). *An examination of forcing in direct
  numerical simulations of turbulence*. Computers & Fluids, 16(3), 257-278.
