# OOP HIT2D Modules

This folder contains the solver-side code needed by the 2D forced compressible HIT case.

## Core Files

- `hit2d.py`: HIT2D configuration, initialization, conservative-variable setup, time loop, diagnostics, and snapshot writing.
- `forcing.py`: stochastic finite-correlation-time Helmholtz-projected forcing in a Fourier shell (solenoidal, dilatational, or mixed).
- `domain.py`: periodic uniform 2D grid object.
- `equations.py`: compressible Euler/Navier-Stokes primitive and conservative variable utilities.
- `spatial_operator.py`: compact/WENO hybrid inviscid operator, shock sensor, and hyperviscosity utilities.
- `time_operator.py`: SSP-RK3 time integration.
- `parallel/`: backend-aware NumPy/CuPy variants for the 2D HIT operators.
- `postprocess/`: two-point correlations, isotropy diagnostics, and spectra.

The folder still includes a small amount of general 1D infrastructure because some shared operators import those definitions, but the intended use of this copied repository is HIT2D.

## Output Location

In this isolated repo, the default HIT2D output path is:

```text
2D/hit2d_snapshots/
```

Each simulation creates a timestamped child folder unless `--no-timestamp-dir` is used.

## Minimal Python Use

```python
from OOP.hit2d import HIT2DConfig, run_simulation

config = HIT2DConfig.isotropic_128(backend="numpy")
run_simulation(config)
```

The `isotropic_128()` preset uses:

- initial velocity modes in `3 <= |k| <= 5`, avoiding direct initialization of box-scale `k = 1, 2` modes;
- stochastic solenoidal forcing in the same annular shell;
- low-wavenumber drag for `|k| <= 2` to remove inverse-cascade condensate energy;
- homogeneous pressure-relaxation cooling for long compressible forced runs.

For normal use, prefer the scripts in `../2D/`.

## Forcing modes

The default remains purely solenoidal forcing. The same solver can now be run with

```text
--forcing-mode solenoidal
--forcing-mode dilatational
--forcing-mode mixed --forcing-dilatational-fraction 0.5
```

The saved diagnostic history includes the requested dilatational fraction and the
RMS divergence and curl of the forcing field. These quantities provide a direct
check that purely solenoidal forcing is divergence-free and purely dilatational
forcing is curl-free to numerical precision.

## Turnover-controlled runs

`HIT2DConfig` now supports `turnover_final`, `turnover_data_start`, and an
optional `turnover_length`. When `turnover_final` is set, the solver integrates
`Neddy = integral u_rms/L_ref dt` online and stops at that value instead of at
`tfinal`. The saved lower bound is automatically used by the post-processing
classes, preventing the initial adjustment transient from contaminating the
statistics.
