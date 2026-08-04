# OOP solver modules

This folder contains the solver-side implementation for the compressible HIT and 2D Riemann cases.

## Main modules

- `hit2d.py`: HIT configuration, initialization, time loop, diagnostics, `config.json`, and snapshots.
- `forcing.py`: finite-correlation-time solenoidal forcing in a Fourier shell.
- `domain.py`: one- and two-dimensional grid helpers.
- `equations.py`: NumPy Euler and Navier–Stokes equations.
- `spatial_operator.py`: compact/WENO operators, shock sensors, and compact-node hyperviscosity.
- `parallel/`: NumPy/CuPy backend-aware equations and periodic HIT operators.
- `postprocess/`: spectra, turnover-time, isotropy, and correlation utilities.
- `problems/`: the non-periodic 2D Riemann configurations and the viscous shock–shear-layer benchmark.
- `run_utils.py`: JSON, run-directory, and NumPy conversion helpers.

## HIT configuration

The current HIT baseline is solenoidal-only. The initial velocity and stochastic forcing use a configurable Fourier shell. The production setup uses `3 <= |k| <= 5`, low-wavenumber drag at `|k| <= 2`, and homogeneous mean-pressure cooling.

Each call to `run_simulation()` writes a complete `config.json` into `config.output_dir` before the first snapshot.

## Hyperviscosity policy

The shock sensor marks the nodes handled by WENO. The periodic HIT filter receives the complementary mask and therefore updates only compact finite-difference nodes. The hyperviscosity classes require an explicit active mask to prevent accidental global filtering.

The non-periodic Riemann solver follows the same policy. WENO-only Riemann runs disable numerical hyperviscosity automatically.

## Minimal Python use

```python
from OOP.hit2d import HIT2DConfig, run_simulation

config = HIT2DConfig.isotropic_128(backend="numpy")
run_simulation(config)
```

## Offset Configuration 3 time stepping

`RiemannConfig3Offset08` uses `x0 = y0 = 0.8`, `tfinal = 0.85`, an initial
`CFL = 0.4`, and four protected outflow-boundary cells by default. One fixed
time step is calculated from the initial Euler spectral radius. WENO-only
right-hand-side evaluations bypass the shock sensor because every interface
already uses WENO.

## Shock--shear-layer benchmark

`problems/shock_shear_layer.py` implements the viscous two-dimensional case
from Kang and Lee (2026), Section 3.2. It includes the time-dependent supersonic
inlet, upper post-shock state, lower slip wall, outlet extrapolation,
constant/Sutherland viscosity, heat conduction at configurable Prandtl number,
and output routines for the density contour and centerline density profile.
The boundary values are imposed at the SSP-RK3 stage times, while the global
time step remains fixed during the run.
