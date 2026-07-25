# HIT2D / 2D Riemann Solver

This repository contains the object-oriented code used for the thesis simulations:

- 2D forced compressible homogeneous isotropic turbulence (HIT).
- 2D Euler Riemann problems, Lax--Liu Configurations 3 and 6.

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

The Riemann implementations are written with the shared NumPy/CuPy backend helper. They avoid SciPy sparse solves for the non-periodic compact line derivative and use a batched Thomas algorithm, so the evolving state and numerical operators can remain on the GPU.

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
  --cfl 0.4 \
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
  --nx 512 --ny 512 --tfinal 0.3 --cfl 0.4 \
  --sensor-width 4 --jump-threshold 0.025 --shear-threshold 0 \
  --mn 0.001 --density-contours --schlieren --vorticity-limit 100
```

WENO-only comparison:

```bash
python3 -B scripts/run_riemann_config3.py \
  --backend numpy --scheme weno \
  --nx 512 --ny 512 --tfinal 0.3 --cfl 0.4 \
  --mn 0.001 --density-contours --schlieren
```

Late-time qualitative visualization:

```bash
python3 -B scripts/run_riemann_config3.py \
  --backend numpy --scheme hybrid \
  --nx 512 --ny 512 --tfinal 0.6 --cfl 0.4 \
  --sensor-width 4 --jump-threshold 0.025 --shear-threshold 0 \
  --mn 0.001 --density-contours --schlieren --zoom-center
```

The standard benchmark comparison should remain `t=0.3`. Later times such as `t=0.6` or `t=0.85` are useful for qualitative vortex visualization, but boundary effects and domain placement must be interpreted carefully.

## Run tutor Configuration 3 with x0=y0=0.8

The tutor case uses the same four primitive states as Configuration 3, but moves the initial quadrant intersection from `(0.5, 0.5)` to `(0.8, 0.8)`. It is kept separate from the centered benchmark through `RiemannConfig3Offset08` and `scripts/run_riemann_config3_08.py`.

The dedicated runner defaults to CuPy, the hybrid compact/WENO scheme, `mn=0.001`, and one hyperviscosity application every five time steps:

```bash
MPLBACKEND=Agg python -B scripts/run_riemann_config3_08.py \
  --backend cupy \
  --scheme hybrid \
  --nx 512 --ny 512 \
  --tfinal 0.3 \
  --cfl 0.4 \
  --mn 0.001 \
  --hyperviscosity-interval 5 \
  --fixed-density-limits 0.13 1.75 \
  --progress-every 500 \
  --run-id config3_08_hybrid_hv5_gpu_512_t03
```

Unified launcher equivalent:

```bash
python -B scripts/run_case.py --problem riemann3_08 --backend cupy --nx 512 --ny 512
```

All 2D Riemann presets default to `CFL=0.4`, matching the tutor setup. You can still override it with `--cfl`; reduce it if a modified grid, scheme, or dissipation setting fails the positivity diagnostics.

## Run 2D Riemann Configuration 6 on the GPU

Configuration 6 uses the standard Lax--Liu quadrant ordering

```text
II | I
---+---
III| IV
```

with primitive states `(rho, p, u, v)`:

```text
I   = (1.0, 1.0,  0.75, -0.50)
II  = (2.0, 1.0,  0.75,  0.50)
III = (1.0, 1.0, -0.75,  0.50)
IV  = (3.0, 1.0, -0.75, -0.50)
```

The benchmark domain is `[0, 1] x [0, 1]`, uses outflow boundaries, and runs to `t=0.25`. The Configuration 6 runner defaults to CuPy:

```bash
python3 -B scripts/run_riemann_config6.py \
  --backend cupy \
  --scheme hybrid \
  --nx 512 --ny 512 \
  --tfinal 0.25 \
  --cfl 0.4 \
  --density-contours \
  --fixed-density-limits 0 3.2
```

Unified launcher equivalent:

```bash
python3 -B scripts/run_case.py --problem riemann6 \
  --backend cupy --scheme hybrid \
  --nx 512 --ny 512 --tfinal 0.25 --cfl 0.4 \
  --density-contours --fixed-density-limits 0 3.2
```

Use `--backend numpy` for a CPU reference run. CuPy keeps the evolving conservative state, sensor, flux reconstruction, compact derivative solves, Runge--Kutta stages, and hyperviscosity operations on the GPU; arrays are copied to the CPU only while saving or plotting.

The Configuration 6 entry points are:

- `OOP/problems/riemann_config6.py`
- `scripts/run_riemann_config6.py`
- `scripts/plot_riemann_config6.py`
- `scripts/run_case.py --problem riemann6`

## Riemann output files

Every Riemann run creates a timestamped run folder under the matching configuration directory:

```text
results/riemann_config3/<run_id>/
results/riemann_config6/<run_id>/
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
x, y, q, rho, u, v, pressure, omega_z, time, steps, configuration_number, config_json, diagnostics_json
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

Configuration 6 uses the corresponding replot wrapper:

```bash
python3 -B scripts/plot_riemann_config6.py \
  results/riemann_config6/<run_id>/<run_id>_final.npz \
  --density-contours \
  --fixed-density-limits 0 3.2
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
