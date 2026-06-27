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

For normal use, prefer the scripts in `../2D/`.
