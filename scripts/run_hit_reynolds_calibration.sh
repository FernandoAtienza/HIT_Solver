#!/usr/bin/env bash
set -uo pipefail

# ---------------------------------------------------------------------------
# Stationary-Re_lambda calibration campaign for the thesis HIT Mach study.
#
# Numerical method is frozen from the previous audits:
#   sensor              = legacy
#   WENO flux splitting = local
#   conservative HV     = mn=0.005, every 5 complete RK steps
#   CFL                 = 0.10
#
# Purpose:
#   Calibrate explicit viscosity at Mt = 0.40, 0.50 and 0.60 so that the
#   statistically stationary 2-D Taylor-scale Reynolds number is close to 130.
#
# These are NOT automatically fed into a final 512^2 campaign. Inspect the
# calibration summary first, then freeze the final viscosities.
# ---------------------------------------------------------------------------

REPO="${REPO:-$HOME/github/HIT_Solver}"
cd "$REPO" || exit 1

ROOT="${ROOT:-results/hit2d/reynolds_calibration_ReL130_N256}"
TARGET_RE="${TARGET_RE:-130.0}"
NX="${NX:-256}"
NY="${NY:-256}"
CFL="${CFL:-0.10}"
MN="${MN:-0.005}"
HV_INTERVAL="${HV_INTERVAL:-5}"
SEED="${SEED:-1234}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "$ROOT"
CAMPAIGN_LOG="$ROOT/campaign.log"
FAILED="$ROOT/failed_cases.txt"
: > "$FAILED"

exec > >(tee -a "$CAMPAIGN_LOG") 2>&1

echo "============================================================"
echo "STATIONARY Re_lambda CALIBRATION CAMPAIGN"
echo "Started: $(date)"
echo "Target stationary Re_lambda,2D = $TARGET_RE"
echo "Grid = ${NX}x${NY}; CFL=$CFL"
echo "Frozen method: legacy sensor + local LF"
echo "HV: conservative_flux, mn=$MN, interval=$HV_INTERVAL"
echo "Statistics window: 4 <= N_eddy <= 16"
echo "============================================================"

run_case () {
    local NAME="$1"
    local MT="$2"
    local MU="$3"
    local TFINAL="$4"
    local SNAPSHOT_EVERY="$5"
    local DIAGNOSTICS_EVERY="$6"

    local OUT="$ROOT/$NAME"

    if [[ -d "$OUT" && "$OVERWRITE" == "1" ]]; then
        echo "Removing existing case because OVERWRITE=1: $OUT"
        rm -rf "$OUT"
    elif [[ -d "$OUT" ]]; then
        if find "$OUT" -maxdepth 1 -name 'hit2d_step*.npz' -type f | grep -q .; then
            echo "Existing directory detected; moving it aside before rerun."
            mv "$OUT" "${OUT}_old_$(date +%Y%m%d_%H%M%S)"
        fi
    fi

    mkdir -p "$OUT"

    echo
    echo "============================================================"
    echo "STARTING $NAME at $(date)"
    echo "Mt=$MT; viscosity=$MU; target stationary Re=$TARGET_RE"
    echo "N=${NX}x${NY}; tfinal=$TFINAL; CFL=$CFL; seed=$SEED"
    echo "============================================================"

    MPLBACKEND=Agg python3 -B scripts/run_hit2d.py \
      --backend cupy \
      --nx "$NX" \
      --ny "$NY" \
      --tfinal "$TFINAL" \
      --cfl "$CFL" \
      --gamma 1.4 \
      --mach "$MT" \
      --initial-kmin 3 \
      --initial-kmax 5 \
      --kf-min 3 \
      --kf-max 5 \
      --p-target 1.0e-3 \
      --forcing-correlation-time 0.5 \
      --forcing-alpha-memory 0.2 \
      --min-forcing-power 1.0e-6 \
      --max-forcing-rescale 20.0 \
      --mach-control \
      --mach-control-target "$MT" \
      --mach-control-memory 0.995 \
      --mach-control-exponent 2.0 \
      --viscosity "$MU" \
      --prandtl 0.72 \
      --sensor-mode legacy \
      --sensor-width 4 \
      --jump-threshold 0.04 \
      --compression-threshold 2.5 \
      --weno-flux-splitting local \
      --hyperviscosity-mode conservative_flux \
      --mn "$MN" \
      --hyperviscosity-interval "$HV_INTERVAL" \
      --large-scale-drag 0.10 \
      --drag-kmax 2.0 \
      --cooling-time 5.0 \
      --diagnostics-every "$DIAGNOSTICS_EVERY" \
      --snapshot-every "$SNAPSHOT_EVERY" \
      --seed "$SEED" \
      --output-dir "$OUT" \
      2>&1 | tee "$OUT/run.log"

    local SIM_STATUS=${PIPESTATUS[0]}
    if [[ "$SIM_STATUS" -ne 0 ]]; then
        echo "FAILED simulation: $NAME (exit=$SIM_STATUS)"
        echo "$NAME simulation_exit_$SIM_STATUS" >> "$FAILED"
        return
    fi

    echo "Post-processing $NAME at $(date)"

    MPLBACKEND=Agg python3 -B 2D/hit2d_viewer.py \
      --snapshot-dir "$OUT" \
      --physics-plots \
      --history-x-axis turnover \
      > "$OUT/viewer.log" 2>&1

    if [[ $? -ne 0 ]]; then
        echo "WARNING: viewer failed for $NAME"
        echo "$NAME viewer_failed" >> "$FAILED"
    fi

    MPLBACKEND=Agg python3 -B 2D/hit2d_isotropy_diagnostics.py \
      --snapshot-dir "$OUT" \
      --start-turnover 4 \
      --end-turnover 16 \
      --fluctuation-type reynolds \
      > "$OUT/isotropy_postprocess.log" 2>&1

    if [[ $? -ne 0 ]]; then
        echo "WARNING: isotropy post-processing failed for $NAME"
        echo "$NAME isotropy_failed" >> "$FAILED"
    fi

    echo "FINISHED $NAME at $(date)"
}

# ---------------------------------------------------------------------------
# Viscosity guesses
#
# Mt=0.50 and Mt=0.60 come directly from the previous campaign via
#     mu_new ~= mu_old * Re_stationary_old / 130.
#
# Mt=0.40 is an interpolation between the calibrated Mt=0.25 and Mt=0.50
# behavior and must be checked by this campaign.
# ---------------------------------------------------------------------------

run_case \
  "calib_Mt040_N256_mu0p001140" \
  "0.40" \
  "0.0011400" \
  "72.45298058" \
  "3100" \
  "300"

run_case \
  "calib_Mt050_N256_mu0p001316" \
  "0.50" \
  "0.00131565" \
  "57.96238446" \
  "2500" \
  "250"

run_case \
  "calib_Mt060_N256_mu0p001414" \
  "0.60" \
  "0.00141405" \
  "48.30198705" \
  "2100" \
  "200"

echo
echo "============================================================"
echo "SIMULATIONS COMPLETE. Building calibration summary..."
echo "============================================================"

python3 -B scripts/summarize_reynolds_calibration.py \
  --root "$ROOT" \
  --target-re "$TARGET_RE"

SUMMARY_STATUS=$?
if [[ "$SUMMARY_STATUS" -ne 0 ]]; then
    echo "WARNING: calibration summarizer failed (exit=$SUMMARY_STATUS)."
fi

echo
echo "============================================================"
echo "CAMPAIGN FINISHED: $(date)"
echo "Results: $ROOT"
echo "Summary: $ROOT/reynolds_calibration_summary.md"
echo "Failures: $FAILED"
echo "============================================================"
