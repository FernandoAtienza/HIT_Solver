# HIT2D Solver Subrepo

This folder contains the code needed to run and post-process the 2D forced compressible homogeneous isotropic turbulence case.

## Layout

- `OOP/`: solver core, HIT2D setup, forcing, NumPy/CuPy backend helpers, and post-processing classes.
- `2D/`: executable scripts for running HIT2D simulations and generating plots.
- `2D/hit2d_snapshots/`: local output folder for new runs. Simulation results are intentionally ignored by git.

## Run A Simulation

From inside `HIT_Solver`, run a corrected 128x128 isotropy test with NumPy:

```powershell
python -B 2D\hit2d_viewer.py --run --animate --physics-plots --physics-animate --backend numpy --nx 128 --ny 128 --tfinal 15.0 --mach 0.25 --initial-kmin 3 --initial-kmax 5 --kf-min 3 --kf-max 5 --p-target 1.0e-3 --viscosity 7.5e-4 --forcing-correlation-time 0.5 --forcing-alpha-memory 0.2 --large-scale-drag 0.10 --drag-kmax 2.0 --cooling-time 5.0 --mn 0.002 --snapshot-every 75 --diagnostics-every 25 --fps 5
```

For CuPy, change `--backend numpy` to `--backend cupy`.

Each run creates a timestamped folder:

```text
2D/hit2d_snapshots/run_YYYYMMDD_HHMMSS/
```

## Adaptive Turbulent-Mach Control

The default forcing is constant-power forcing: `--p-target` fixes the target
kinetic-energy injection rate, and the statistically stationary turbulent Mach
number is an output of the energy balance.

To make the run adapt toward a prescribed turbulent Mach number, enable:

```powershell
--mach-control --mach-control-target 0.5
```

At each time step the code measures the current turbulent Mach number and slowly
updates the forcing power target according to the ratio between the requested and
measured values. If `Mt` is too low, the target power increases; if `Mt` is too
high, it decreases. The forcing field itself remains solenoidal, stochastic, and
restricted to the selected Fourier shell.

Useful controls are:

- `--mach-control-memory`: smoothing of the adaptive power target. Larger values
  change power more slowly.
- `--mach-control-exponent`: exponent in the correction
  `(Mt_target / Mt)^exponent`.
- `--mach-control-min-power` and `--mach-control-max-power`: safety bounds for
  the adaptive power target.

Recommended controlled 128x128 CuPy test:

```powershell
python -B 2D\hit2d_viewer.py --run --animate --physics-plots --physics-animate --backend cupy --nx 128 --ny 128 --tfinal 70.0 --mach 0.5 --mach-control --mach-control-target 0.5 --mach-control-memory 0.995 --mach-control-exponent 2.0 --mach-control-min-power 2.0e-4 --mach-control-max-power 1.0e-2 --initial-kmin 3 --initial-kmax 5 --kf-min 3 --kf-max 5 --p-target 1.0e-3 --viscosity 7.5e-4 --forcing-correlation-time 0.5 --forcing-alpha-memory 0.2 --large-scale-drag 0.03 --drag-kmax 2.0 --cooling-time 10.0 --mn 0.002 --snapshot-every 150 --diagnostics-every 50 --fps 10
```

The diagnostic history stores `forcing_target_power`,
`mach_control_mach`, `mach_control_target`, and
`mach_control_power_desired`, so the controller behavior can be checked after
the run.

## Why Large-Scale Drag And Cooling Are Included

The compact/WENO/hyperviscosity numerical scheme is unchanged. The extra terms are physical/source-term controls for a statistically stationary 2D forced test:

- The initial velocity and the forcing both use solenoidal annular shells instead of the box-scale `k = 1, 2` modes.
- The forcing injects solenoidal kinetic energy in the annular shell `3 <= |k| <= 5`.
- In 2D, kinetic energy tends to transfer upscale. Without a sink at `k < 3`, energy can accumulate in box-scale `k = 1` and `k = 2` modes, creating bands or jets that break isotropy.
- `--large-scale-drag 0.10 --drag-kmax 2.0` damps only those low-wavenumber momentum modes. It does not change the WENO/compact operator and does not directly damp the forced shell or the small scales.
- Forced compressible turbulence converts kinetic energy into internal energy. `--cooling-time 5.0` applies homogeneous mean-pressure relaxation so the mean pressure and turbulent Mach number do not drift during long runs.

Useful diagnostics saved in `diagnostic_history.npz` include `A_K`, `C_uv`, `A_F`, `large_scale_kinetic_fraction`, `drag_power`, and `cooling_power`.

## Post-Process A Run

Two-point correlations:

```powershell
python -B 2D\hit2d_two_point_correlation.py --snapshot-dir 2D\hit2d_snapshots\run_YYYYMMDD_HHMMSS --fluctuation-type reynolds
```

Stationarity, isotropy, anisotropy, and spectra:

```powershell
python -B 2D\hit2d_isotropy_diagnostics.py --snapshot-dir 2D\hit2d_snapshots\run_YYYYMMDD_HHMMSS --fluctuation-type reynolds --start-time 7.0 --end-time 12.0
```

Turnover-time based stationarity, isotropy, anisotropy, and spectra:

```powershell
python -B 2D\hit2d_isotropy_diagnostics.py --snapshot-dir 2D\hit2d_snapshots\run_YYYYMMDD_HHMMSS --fluctuation-type reynolds --start-turnover 4.0 --end-turnover 16.0 --history-x-axis turnover
```

The turnover coordinate is computed as:

```text
N_eddy(t) = integral_0^t u_rms(t') / L_ref dt'
```

where, by default, `L_ref = 2*pi/k_ref` and `k_ref` is the center of the forced shell. For `3 <= |k| <= 5`, this gives:

```text
k_ref = 4
L_ref = pi/2
```

The recommended statistical workflow is:

```text
0 <= N_eddy < 4      transient spin-up
4 <= N_eddy <= 16    statistics and plotted histories
```

Post-processing figures are saved in:

```text
2D/hit2d_snapshots/run_YYYYMMDD_HHMMSS/postprocess/
```
