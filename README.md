# Compressible HIT and 2D Riemann Solver

This repository contains the object-oriented numerical solver used in the master’s thesis. The current study focuses on **two-dimensional, compressible, homogeneous isotropic turbulence driven exclusively by solenoidal forcing**. The repository also retains the validated two-dimensional Euler Riemann problems used to verify the shock-capturing method.

## Current scope

The implemented cases are:

- Forced compressible HIT in a periodic square domain.
- Lax–Liu Riemann Configuration 3.
- Configuration 3 with the initial intersection at `x0 = y0 = 0.8`.
- Lax–Liu Riemann Configuration 6.
- Two-dimensional viscous shock–shear-layer interaction from Kang and Lee (2026).

Dilatational and mixed forcing are not part of the current baseline.

## Numerical method

The inviscid terms use a hybrid compact/WENO discretization:

- high-order compact finite differences in smooth regions;
- seventh-order componentwise WENO reconstruction at sensor-detected discontinuities;
- SSP-RK3 time integration;
- optional physical viscosity and thermal conduction for HIT;
- optional biharmonic numerical hyperviscosity.

The shock sensor determines which nodes use WENO. Numerical hyperviscosity follows a non-overlap policy:

- it is applied only at nodes where the compact finite-difference scheme is active;
- it is never applied at WENO nodes;
- it is disabled automatically for WENO-only simulations, even when `mn > 0` is supplied.

This avoids stacking the additional biharmonic dissipation on top of WENO dissipation.

## Repository layout

```text
HIT_Solver/
├── 2D/
│   ├── hit2d_viewer.py
│   ├── hit2d_isotropy_diagnostics.py
│   └── hit2d_two_point_correlation.py
├── OOP/
│   ├── domain.py
│   ├── equations.py
│   ├── forcing.py
│   ├── hit2d.py
│   ├── run_utils.py
│   ├── spatial_operator.py
│   ├── time_operator.py
│   ├── parallel/
│   ├── postprocess/
│   └── problems/
│       ├── riemann_config3.py
│       ├── riemann_config3_08.py
│       ├── riemann_config6.py
│       └── shock_shear_layer.py
├── scripts/
│   ├── run_hit2d.py
│   ├── run_solenoidal_hit_campaign.sh
│   ├── run_riemann_config3.py
│   ├── run_riemann_config3_08.py
│   ├── run_riemann_config6.py
│   ├── plot_riemann_config3.py
│   ├── plot_riemann_config6.py
│   ├── run_shock_shear_layer.py
│   └── plot_shock_shear_layer.py
└── tests/
```

## Backend support

The HIT and Riemann solvers support:

```bash
--backend numpy
--backend cupy
--backend auto
```

`numpy` runs on the CPU. `cupy` keeps the evolving state and grid-wide operators on one CUDA GPU. The implementation is not MPI or multi-GPU parallelism.

Before a long GPU campaign, compare short NumPy and CuPy runs using identical settings.

## Run HIT

A small CPU run is:

```bash
python3 -B scripts/run_hit2d.py \
  --backend numpy \
  --nx 64 --ny 64 \
  --tfinal 0.05 \
  --cfl 0.05 \
  --mach 0.25 \
  --initial-kmin 3 --initial-kmax 5 \
  --kf-min 3 --kf-max 5 \
  --snapshot-every 10 \
  --diagnostics-every 5 \
  --output-dir results/hit2d/smoke_test
```

A production-style solenoidal case is:

```bash
python3 -B scripts/run_hit2d.py \
  --backend cupy \
  --nx 512 --ny 512 \
  --tfinal 110.0 \
  --cfl 0.05 \
  --gamma 1.4 \
  --mach 0.25 \
  --initial-kmin 3 --initial-kmax 5 \
  --kf-min 3 --kf-max 5 \
  --p-target 1.0e-3 \
  --forcing-correlation-time 0.5 \
  --forcing-alpha-memory 0.2 \
  --min-forcing-power 1.0e-6 \
  --max-forcing-rescale 20.0 \
  --mach-control \
  --mach-control-target 0.25 \
  --mach-control-memory 0.995 \
  --mach-control-exponent 2.0 \
  --viscosity 7.5e-4 \
  --mn 0.002 \
  --hyperviscosity-interval 5 \
  --large-scale-drag 0.10 \
  --drag-kmax 2.0 \
  --cooling-time 5.0 \
  --seed 1234 \
  --diagnostics-every 1000 \
  --snapshot-every 5000 \
  --output-dir results/hit2d/solenoidal_Mt025_N512_CFL_0p05_smooth
```

The complete two-case campaign can be launched with:

```bash
./scripts/run_solenoidal_hit_campaign.sh
```

## HIT output files

Every HIT output directory contains:

```text
config.json
diagnostic_history.npz
hit2d_step0000000.npz
hit2d_stepXXXXXXX.npz
...
```

`config.json` is written before time integration begins. It contains the fully resolved `HIT2DConfig` values plus the fixed implementation descriptors:

```text
problem
forcing_type
spatial_discretization
hyperviscosity_policy
```

This makes each run directory self-describing and provides the exact parameters needed for reproduction.

`diagnostic_history.npz` stores the time histories of kinetic energy, turbulent Mach number, forcing power, anisotropy measures, low-wavenumber energy, mean thermodynamic variables, sensor fraction, and conservation diagnostics.

Each snapshot stores the primitive fields, vorticity, divergence, physical time, and time-step index.

## HIT forcing

The initial velocity is generated from a random streamfunction in the shell

```text
initial_kmin <= |k| <= initial_kmax
```

and is divergence-free to roundoff. The forcing is a finite-correlation-time Ornstein–Uhlenbeck process projected onto the solenoidal Fourier subspace and restricted to

```text
kf_min <= |k| <= kf_max.
```

The production campaign uses `3 <= |k| <= 5` for both initialization and forcing. The scalar potential follows an Ornstein--Uhlenbeck process, so the spatial pattern has a finite correlation time rather than being replaced by independent white noise every step.

The current baseline uses the original finite-correlation Ornstein--Uhlenbeck
forcing with a smoothed scalar power-rescaling coefficient. The optional Mach
controller slowly adapts the requested mean injection power using
`--mach-control-memory`. The forcing and controller are intentionally left
unchanged during the first dissipation audit so that the effects of WENO,
physical viscosity, and hyperviscosity can be isolated before another forcing
redesign is attempted.

## HIT post-processing

Generate final physical fields and the time history with:

```bash
python3 -B 2D/hit2d_viewer.py \
  --snapshot-dir results/hit2d/<case_name> \
  --physics-plots \
  --history-x-axis turnover
```

Generate isotropy, two-point-correlation, spectral, Helmholtz-decomposition, and PDF diagnostics with:

```bash
python3 -B 2D/hit2d_isotropy_diagnostics.py \
  --snapshot-dir results/hit2d/<case_name> \
  --start-turnover 4 \
  --end-turnover 16 \
  --fluctuation-type reynolds
```

The spectral output contains the total velocity spectrum, the density-weighted spectrum, and solenoidal/dilatational spectra obtained by Fourier-space Helmholtz projection. The discrete normalization is based on Parseval's identity,

```text
sum_k E(k) = 0.5 * mean(rho) * mean(u_prime^2 + v_prime^2).
```

This is an energy normalization, not a volume integral of the energy-cascade flux. The plots show both `k^-5/3` and `k^-3` guides because the solver is two-dimensional: the former is associated with the inverse energy cascade and the latter with the forward enstrophy cascade. The guides are not fitted exponents and should not be interpreted as proof of an inertial range.

The PDF output includes standardized dilatation, vorticity, pressure-fluctuation, and density-fluctuation PDFs, plus a joint dilatation--vorticity PDF and skewness/flatness values.

## Reynolds-number and DNS-resolution diagnostics

A target initial Taylor-microscale Reynolds number can be requested with:

```bash
--initial-re-lambda 60
```

The solver then computes the constant dynamic viscosity required by the initialized two-dimensional field and records the resolved value in `config.json`. The reported `Re_lambda_2d` is a two-dimensional analogue; it is not numerically interchangeable with the three-dimensional Taylor Reynolds numbers in standard HIT databases.

Every diagnostic sample includes the physical viscous dissipation, the conventional Kolmogorov-length analogue `eta_K`, the two-dimensional Kraichnan/enstrophy microscale `eta_Omega`, their ratios to the grid spacing, the WENO fraction, and the measured kinetic-energy drain from numerical hyperviscosity. A run should only be described as DNS-quality when the physical dissipative scales are resolved and the numerical sinks remain small and localized. A value of `k_max*eta_K` or `eta_K/dx` alone is not sufficient when WENO or hyperviscosity contributes appreciable dissipation.

## First-stage HIT dissipation audit

The first roadmap campaign keeps the initial two-dimensional Taylor Reynolds
number fixed and varies only the compact-node hyperviscosity strength:

```bash
./scripts/run_hit_dissipation_audit.sh
```

The default sequence uses `N=256`, `Mt=0.25`, `Re_lambda_2d=120`, and

```text
mn = 0, 0.0005, 0.001, 0.002.
```

Each case is run and post-processed before the next one starts. The campaign
also creates:

```text
dissipation_audit_summary.csv
dissipation_audit_summary.md
dissipation_audit_spectra.png
```

Defaults can be overridden without editing the script, for example:

```bash
NX=128 NY=128 TFINAL=20 CFL=0.10 \
  ./scripts/run_hit_dissipation_audit.sh
```

## Run Riemann Configuration 3

Hybrid compact/WENO run:

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

WENO-only comparison:

```bash
python3 -B scripts/run_riemann_config3.py \
  --backend numpy \
  --scheme weno \
  --nx 512 --ny 512 \
  --tfinal 0.3 \
  --cfl 0.4 \
  --mn 0.0 \
  --density-contours \
  --schlieren
```

For `--scheme weno`, hyperviscosity is disabled automatically. Setting `--mn 0.0` in the command makes that choice explicit in the saved configuration.

## Run Configuration 3 with `x0 = y0 = 0.8`

This scenario defaults to `tfinal = 0.85`, an initial `CFL = 0.4`, and four
protected outflow-boundary cells. A single fixed time step is computed from the
initial state and adjusted slightly so an integer number of equal steps reaches
the requested final time exactly.

```bash
MPLBACKEND=Agg python3 -B scripts/run_riemann_config3_08.py \
  --backend cupy \
  --scheme hybrid \
  --nx 512 --ny 512 \
  --tfinal 0.85 \
  --cfl 0.4 \
  --guard-cells 4 \
  --boundary-guard 4 \
  --mn 0.001 \
  --hyperviscosity-interval 5 \
  --fixed-density-limits 0.13 1.75 \
  --progress-every 500 \
  --run-id config3_08_hybrid_hv5_gpu_512_t085_cfl04
```

The diagnostics distinguish the shock-sensor node fraction from the actual
fraction of directional interfaces using WENO.  WENO-only runs report a WENO
fraction of exactly one and skip shock-sensor evaluation during the RHS.

## Run Riemann Configuration 6

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

## Riemann output files

Each Riemann run creates a run directory under the corresponding result folder and writes:

```text
config.json
diagnostics.json
<run_id>_final.npz
<run_id>_fields.png
```

The final `.npz` file stores the state and serialized configuration so figures can be regenerated without rerunning the simulation.

## Replot a Riemann result

Configuration 3:

```bash
python3 -B scripts/plot_riemann_config3.py \
  results/riemann_config3/<run_id>/<run_id>_final.npz \
  --output results/riemann_config3/<run_id>/replot.png \
  --density-contours \
  --schlieren
```

Configuration 6:

```bash
python3 -B scripts/plot_riemann_config6.py \
  results/riemann_config6/<run_id>/<run_id>_final.npz \
  --density-contours \
  --fixed-density-limits 0 3.2
```

## Tests

Run the complete test suite from the repository root:

```bash
python3 -m pytest -q
```

A syntax-only check is:

```bash
python3 -m compileall -q OOP scripts tests
```

## Run the two-dimensional shock–shear-layer interaction

The case in Section 3.2 of Kang and Lee (2026) uses the domain
`[0, 200] x [-20, 20]`, a `500 x 100` grid, `Re = 500`, `Pr = 0.72`, and a
final time of `t = 120`. The inlet mixing layer and time-dependent transverse
perturbation, upper post-shock boundary, lower slip wall, and zero-gradient
outlet are implemented in `OOP/problems/shock_shear_layer.py`.

A reference-grid GPU run is:

```bash
MPLBACKEND=Agg python3 -B scripts/run_shock_shear_layer.py \
  --backend cupy \
  --scheme hybrid \
  --nx 500 --ny 100 \
  --tfinal 120 \
  --cfl 0.4 \
  --reynolds 500 \
  --prandtl 0.72 \
  --viscosity-model sutherland \
  --sensor-width 2 \
  --jump-threshold 0.04 \
  --compression-threshold 2.5 \
  --shear-threshold 0 \
  --guard-cells 4 \
  --boundary-guard 4 \
  --mn 0.0005 \
  --hyperviscosity-interval 5 \
  --progress-every 100 \
  --output-dir results/shock_shear_layer \
  --run-id shock_shear_hybrid_N500x100_t120
```

The run directory contains `config.json`, `diagnostics.json`, the final NPZ
state, a density-contour figure comparable with Figure 2(c), a centerline
density profile comparable with Figure 2(d), and a four-field diagnostic
figure.

The paper states that viscosity follows Sutherland's law but does not report
the exact nondimensional reference-temperature convention for this case. The
implementation uses the upper-stream temperature as the nondimensional
reference and maps it to 300 K. The alternative classical constant-viscosity
interpretation can be selected with:

```bash
--viscosity-model constant
```

Regenerate figures without rerunning the simulation with:

```bash
python3 -B scripts/plot_shock_shear_layer.py \
  results/shock_shear_layer/<run_id>/<run_id>_final.npz
```

## Conservative compact-region hyperviscosity audit

The HIT solver now provides `--hyperviscosity-mode conservative_flux`, which
writes the compact-region biharmonic correction as a periodic face-flux
divergence. Hyperviscous faces are active only when both adjacent nodes are
compact-FD nodes. This preserves the discrete sum of each conservative variable
to roundoff in periodic runs when no positivity clipping is required.

The filter deliberately retains the historical application policy: it is
applied every `--hyperviscosity-interval` complete RK steps and is **not** scaled
by `dt`. Its effective dissipation rate therefore depends on CFL. The diagnostic
`hyperviscosity_nominal_rate = mn/(interval*dt)` is saved explicitly so this
dependence can be quantified rather than hidden.

Run the next audit with:

```bash
./scripts/run_hit_conservative_hv_audit.sh
```

The campaign tests `mn = 0.001, 0.002, 0.005, 0.01` at CFL 0.10 and then tests
CFL 0.05 and 0.20 at `mn = 0.002`.

## HIT shock-sensor and WENO flux-splitting audit

The periodic HIT branch now exposes two numerical-method controls intended for the final dissipation audit before production studies.

### Shock sensor

`--sensor-mode` accepts:

- `legacy`: previous union sensor. WENO is selected when strong compression, an optional shear criterion, or any density/pressure/internal-energy jump exceeds its threshold.
- `compression_gated`: WENO requires all of (i) strong negative dilatation, (ii) a Ducros-type compression dominance
  \(S_D=\theta^2/(\theta^2+\omega_z^2+\epsilon)\), and (iii) a thermodynamic jump.
- `directional`: applies the same gate independently to the x- and y-normal reconstructions. The mask is dilated only along the corresponding reconstruction direction.

The Ducros gate is controlled with `--ducros-threshold` (default `0.5`). The diagnostics now save total, x-direction, and y-direction WENO node fractions.

### WENO Lax--Friedrichs splitting

`--weno-flux-splitting` accepts:

- `global`: the original domain-global maximum characteristic speed;
- `local`: one Lax--Friedrichs speed per interface, computed as the maximum characteristic speed over the complete WENO7 stencil contributing to that interface.

The local option is intended to reduce unnecessary WENO dissipation while retaining a conservative flux difference.

### Final pre-production audit

Run the full sensor/LF factorial campaign with:

```bash
nohup ./scripts/run_hit_sensor_lf_audit.sh > sensor_lf_launcher.log 2>&1 &
```

The default campaign holds `Mt=0.25`, `Re_lambda,2D,0=120`, `CFL=0.10`, and conservative compact-region hyperviscosity `mn=0.005` applied every five complete RK steps. It compares all three sensor modes with both global and local LF splitting and post-processes every case automatically.

## Thesis HIT production baseline and campaign

The hyperviscosity and sensor/LF audits define the first frozen numerical baseline for thesis-oriented HIT production runs:

```text
CFL                         = 0.10
initial Re_lambda,2D        = 120
sensor                      = legacy
WENO LF splitting           = local stencil maximum
conservative hyperviscosity = mn 0.005 every 5 complete RK steps
forcing shell               = 3 <= |k| <= 5
large-scale drag            = 0.10 for |k| <= 2
cooling time                = 5.0
```

The sensor/LF audit showed that the compression-gated and directional sensors were too restrictive at `Mt=0.25`: they selected zero WENO nodes for the entire run. The legacy sensor retained approximately six percent WENO activity and produced better directional-isotropy metrics over the audit interval. Global and local LF splitting gave almost identical low-Mach results; the local stencil maximum is retained for the production baseline because it remains conservative while avoiding the domain-global wave-speed bound when WENO is active.

The thesis campaign is launched with:

```bash
nohup ./scripts/run_hit_thesis_campaign.sh \
  > thesis_campaign_launcher.log 2>&1 &
```

The campaign runs sequentially on one GPU and contains:

1. `Mt=0.25` grid convergence at `128^2`, `256^2`, and `512^2`;
2. a fixed-initial-`Re_lambda,2D` Mach pilot at `256^2` for `Mt=0.10`, `0.25`, `0.50`, and `0.60`;
3. an independent `Mt=0.25`, `256^2` realization using a second random seed.

Final times are scaled with the target Mach number to cover roughly 18 turnover times. Snapshot and diagnostic strides are also scaled with grid size and target Mach so their spacing is approximately comparable in turnover-time units. All statistical post-processing uses the actual interval `4 <= N_eddy <= 16`.

After the simulations finish, `scripts/summarize_hit_thesis_campaign.py` creates:

```text
thesis_campaign_summary.csv
thesis_campaign_summary.md
thesis_grid_convergence_spectra.png
thesis_mach_spectra_comparison.png
thesis_mach_stationary_statistics.png
thesis_repeatability_spectra.png
```

The `512^2` Mach-number production campaign should only be launched after the `256^2` pilot confirms that the stationary Reynolds numbers remain sufficiently comparable and that WENO activity remains physically reasonable as `Mt` increases.
