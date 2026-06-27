# HIT2D Solver Subrepo

This folder contains the code needed to run and post-process the 2D forced compressible homogeneous isotropic turbulence case.

## Layout

- `OOP/`: solver core, HIT2D setup, forcing, NumPy/CuPy backend helpers, and post-processing classes.
- `2D/`: executable scripts for running HIT2D simulations and generating plots.
- `2D/hit2d_snapshots/`: local output folder for new runs. Simulation results are intentionally ignored by git.

## Run A Simulation

From inside `HIT_Solver`, run for NumPy:

```powershell
python -B 2D\hit2d_viewer.py --run --animate --physics-plots --physics-animate --backend numpy --nx 128 --ny 128 --tfinal 12.0 --mach 0.5 --kf-min 3 --kf-max 5 --p-target 1.0e-3 --viscosity 7.5e-4 --forcing-correlation-time 1.0 --forcing-alpha-memory 0.2 --mn 0.002 --snapshot-every 75 --diagnostics-every 25 --fps 5
```

For CuPy, change `--backend numpy` to `--backend cupy`.

Each run creates a timestamped folder:

```text
2D/hit2d_snapshots/run_YYYYMMDD_HHMMSS/
```

## Post-Process A Run

Two-point correlations:

```powershell
python -B 2D\hit2d_two_point_correlation.py --snapshot-dir 2D\hit2d_snapshots\run_YYYYMMDD_HHMMSS --fluctuation-type reynolds
```

Stationarity, isotropy, anisotropy, and spectra:

```powershell
python -B 2D\hit2d_isotropy_diagnostics.py --snapshot-dir 2D\hit2d_snapshots\run_YYYYMMDD_HHMMSS --fluctuation-type reynolds --start-time 7.0 --end-time 12.0
```

Post-processing figures are saved in:

```text
2D/hit2d_snapshots/run_YYYYMMDD_HHMMSS/postprocess/
```
