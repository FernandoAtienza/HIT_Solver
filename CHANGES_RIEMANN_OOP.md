# Riemann OOP integration changes

Implemented first-pass OOP integration of the 2D Euler Riemann Configuration 3 benchmark while preserving the existing HIT solver.

## Configuration 6 GPU scenario

- Added the published Lax--Liu Configuration 6 quadrant states and benchmark final time (`t=0.25`).
- Reused the backend-aware non-periodic solver so the full time evolution runs on CuPy arrays.
- Added dedicated run/replot scripts and the `--problem riemann6` unified-launcher option.
- Saved results now carry their Riemann configuration number, so shared plotting labels remain correct.
- Added regression coverage for the Configuration 6 initial data.

## Tutor Configuration 3 at x0=y0=0.8

- Added `RiemannConfig3Offset08` without changing the centered Configuration 3 preset.
- Added the dedicated `scripts/run_riemann_config3_08.py` CuPy runner.
- Defaults to hybrid compact/WENO fluxes and local hyperviscosity every five steps.
- Added the `--problem riemann3_08` unified-launcher option and regression tests.

## Added

- `OOP/problems/riemann_config3.py`
  - `RiemannConfig3` dataclass.
  - `RiemannDiagnostics` dataclass.
  - non-periodic shock sensor.
  - non-periodic compact derivative using a batched Thomas solve instead of SciPy sparse solves.
  - hybrid compact/WENO and WENO-only modes.
  - outflow/zero-gradient guard-cell treatment.
  - local hyperviscosity with `--mn 0.0` disabling it completely.
  - raw density/pressure validation before applying floors.
  - final `.npz` saving.
  - density, pressure, vorticity, schlieren and contour plotting.
  - replotting from `.npz`.

- `scripts/run_riemann_config3.py`
  - CLI runner for Riemann Configuration 3.

- `scripts/plot_riemann_config3.py`
  - Regenerates figures from saved `.npz` without rerunning the simulation.

- `scripts/run_case.py`
  - Unified dispatcher with `--problem hit2d` and `--problem riemann3`.

- `scripts/run_hit2d.py`
  - Thin wrapper around the existing HIT OOP CLI.

- `OOP/run_utils.py`
  - Timestamped run IDs and JSON helpers.

## Preserved

- Existing `2D/hit2d_viewer.py` workflow.
- Existing HIT OOP solver and post-processing modules.
- Existing Mach-control implementation in `OOP/hit2d.py`.

## Tested in this environment

- Python compile check for `OOP`, `scripts`, and `2D`.
- Riemann NumPy smoke run at `32x32`, `tfinal=0.005`.
- Riemann WENO-only smoke run through `scripts/run_case.py`.
- Replotting from the saved Riemann `.npz`.
- HIT NumPy smoke run at `16x16`, `tfinal=0.005`.

## Not tested here

- CuPy/GPU execution, because this sandbox does not expose your workstation GPU.
- Production-resolution runs.

Run CPU/GPU equivalence checks on the workstation before trusting long GPU runs.
