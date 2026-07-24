# HIT2D / Riemann Configuration 3 Solver

This repository contains the object-oriented code used for the thesis simulations:

- 2D forced compressible homogeneous isotropic turbulence (HIT).
- 2D Euler Riemann problem, Lax--Liu Configuration 3.

The numerical strategy is the same family of methods used throughout the thesis: high-order compact finite differences in smooth regions, WENO near discontinuities, a shock/smoothness sensor, SSP-RK3 time integration, and optional numerical hyperviscosity.

## Repository layout

```text
HIT_Solver-main/
├── 2D/                         # legacy HIT execution/post-processing scripts
├── OOP/
│   ├── domain.py                # 1D/2D domain helpers
│   ├── equations.py             # NumPy equation helpers
│   ├── forcing.py               # HIT forcing
│   ├── hit2d.py                 # OOP HIT driver and CLI
│   ├── parallel/                # NumPy/CuPy backend-aware helpers
│   ├── postprocess/             # spectra, isotropy and correlation diagnostics
│   ├── problems/
│   │   └── riemann_config3.py   # OOP Riemann Configuration 3 implementation
│   └── run_utils.py             # timestamped IDs, JSON helpers, array conversion
└── scripts/
    ├── run_case.py              # unified launcher with --problem hit2d/riemann3
    ├── run_hit2d.py             # clean HIT wrapper
    ├── run_riemann_config3.py   # Riemann Configuration 3 runner
    └── plot_riemann_config3.py  # regenerate Riemann figures from .npz files
```

## Backend support

The HIT solver already supports:

```bash
--backend numpy
--backend cupy
--backend auto
```

The Riemann Configuration 3 implementation in `OOP/problems/riemann_config3.py` is also written with the shared NumPy/CuPy backend helper. It avoids SciPy sparse solves for the new non-periodic compact line derivative and uses a batched Thomas algorithm, so it can run with NumPy arrays or CuPy arrays.

CuPy mode is GPU array acceleration, not MPI or multi-GPU parallelism. Before trusting long GPU runs, compare short CPU/GPU smoke tests on a small grid.

## Run HIT2D

You can use the legacy script:

```bash
python3 -B 2D/hit2d_viewer.py --run --backend cupy --nx 128 --ny 128 --tfinal 1.0
```

or the clean wrapper:

```bash
python3 -B scripts/run_hit2d.py --backend cupy --nx 128 --ny 128 --tfinal 1.0
```

or the unified launcher:

```bash
python3 -B scripts/run_case.py --problem hit2d --backend cupy --nx 128 --ny 128 --tfinal 1.0
```

Mach control is implemented in the HIT solver through:

```bash
--mach-control --mach-control-target 0.5
```

Useful related options are:

```bash
--mach-control-memory 0.995
--mach-control-exponent 2.0
--mach-control-min-power 2.0e-4
--mach-control-max-power 1.0e-2
```

## Run 2D Riemann Configuration 3

Standard validation-type run at `t=0.3`:

```bash
python3 -B scripts/run_riemann_config3.py \
  --backend numpy \
  --scheme hybrid \
  --nx 512 --ny 512 \
  --tfinal 0.3 \
  --cfl 0.04 \
  --sensor-width 4 \
  --jump-threshold 0.025 \
  --shear-threshold 0 \
  --mn 0.001 \
  --hyperviscosity-interval 1 \
  --density-contours \
  --schlieren \
  --vorticity-limit 100
```

Unified launcher equivalent:

```bash
python3 -B scripts/run_case.py --problem riemann3 \
  --backend numpy --scheme hybrid \
  --nx 512 --ny 512 --tfinal 0.3 --cfl 0.04 \
  --sensor-width 4 --jump-threshold 0.025 --shear-threshold 0 \
  --mn 0.001 --density-contours --schlieren --vorticity-limit 100
```

WENO-only comparison:

```bash
python3 -B scripts/run_riemann_config3.py \
  --backend numpy --scheme weno \
  --nx 512 --ny 512 --tfinal 0.3 --cfl 0.04 \
  --mn 0.001 --density-contours --schlieren
```

Late-time qualitative visualization:

```bash
python3 -B scripts/run_riemann_config3.py \
  --backend numpy --scheme hybrid \
  --nx 512 --ny 512 --tfinal 0.6 --cfl 0.04 \
  --sensor-width 4 --jump-threshold 0.025 --shear-threshold 0 \
  --mn 0.001 --density-contours --schlieren --zoom-center
```

The standard benchmark comparison should remain `t=0.3`. Later times such as `t=0.6` or `t=0.85` are useful for qualitative vortex visualization, but boundary effects and domain placement must be interpreted carefully.

## Riemann output files

Every Riemann run creates a timestamped run folder under:

```text
results/riemann_config3/<run_id>/
```

Each folder contains:

```text
config.json
 diagnostics.json
 <run_id>_final.npz
 <run_id>_fields.png
```

The `.npz` file stores the final numerical state so plots can be regenerated without rerunning the simulation. It includes:

```text
x, y, q, rho, u, v, pressure, omega_z, time, steps, config_json, diagnostics_json
```

## Replot a Riemann result from `.npz`

Full-domain replot:

```bash
python3 -B scripts/plot_riemann_config3.py \
  results/riemann_config3/<run_id>/<run_id>_final.npz \
  --output results/riemann_config3/<run_id>/full_domain.png \
  --density-contours \
  --schlieren
```

Zoomed replot:

```bash
python3 -B scripts/plot_riemann_config3.py \
  results/riemann_config3/<run_id>/<run_id>_final.npz \
  --output results/riemann_config3/<run_id>/zoom.png \
  --density-contours \
  --schlieren \
  --zoom-center \
  --zoom-window 0.35
```

Use `--vorticity-limit 0` for automatic vorticity scaling. Use a fixed value such as `--vorticity-limit 100` when comparing different runs.

## Diagnostics

Riemann runs print and save:

- backend and scheme,
- min/max density,
- min/max pressure,
- maximum absolute vorticity,
- NaN/Inf status,
- positivity status,
- final sensor fraction,
- whether local hyperviscosity was active.

The solver validates raw density and pressure before positivity floors can hide a failed physical state. If a non-positive state appears, the run raises a clear `FloatingPointError`.

## CPU/GPU equivalence check

Before running production GPU cases, run quick CPU/GPU smoke tests:

```bash
python3 -B scripts/run_riemann_config3.py --backend numpy --nx 64 --ny 64 --tfinal 0.01 --no-plot --progress-every 10
python3 -B scripts/run_riemann_config3.py --backend cupy  --nx 64 --ny 64 --tfinal 0.01 --no-plot --progress-every 10
```

For HIT:

```bash
python3 -B scripts/run_hit2d.py --backend numpy --nx 64 --ny 64 --tfinal 0.05 --snapshot-every 10 --diagnostics-every 5
python3 -B scripts/run_hit2d.py --backend cupy  --nx 64 --ny 64 --tfinal 0.05 --snapshot-every 10 --diagnostics-every 5
```

If the short CPU/GPU histories diverge strongly at identical parameters, investigate backend-specific logic before launching long runs.
