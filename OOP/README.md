# OOP HIT2D Modules

This folder contains the solver-side code needed by the 2D forced compressible HIT case.

## Core Files

- `hit2d.py`: HIT2D configuration, initialization, conservative-variable setup, time loop, diagnostics, and snapshot writing.
- `forcing.py`: stochastic finite-correlation-time solenoidal forcing in a Fourier shell.
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

## Solenoidal and dilatational forcing

The stochastic shell forcing now supports two pure Helmholtz components through
`HIT2DConfig.forcing_mode` and the CLI option `--forcing-mode`:

- `solenoidal`: divergence-free acceleration, perpendicular to each Fourier
  wavevector;
- `dilatational`: curl-free acceleration, parallel to each Fourier wavevector.

`compressive` is accepted as a command-line alias for `dilatational`.
The default remains `solenoidal`, so existing commands and campaigns preserve
previous behavior.

Example:

```bash
python3 -B scripts/run_hit2d.py \
    --backend cupy \
    --nx 128 --ny 128 \
    --tfinal 5.0 --cfl 0.05 \
    --mach 0.25 \
    --forcing-mode dilatational \
    --kf-min 3 --kf-max 5 \
    --output-dir results/hit2d/dilatational_test
```

The diagnostic history stores `forcing_mode` and also reports the instantaneous
solenoidal and dilatational fractions of the velocity-field spectral energy.
The production launcher is `../scripts/run_dilatational_hit_campaign.sh`.
