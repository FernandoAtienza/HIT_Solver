#!/usr/bin/env bash
set -uo pipefail

# =============================================================================
# FINAL THESIS HIT CAMPAIGN
#
# Frozen numerical method:
#   hybrid compact/WENO7
#   sensor              = legacy
#   WENO splitting      = local Lax-Friedrichs
#   conservative HV     = mn 0.005, every 5 complete RK steps
#   CFL                 = 0.10
#   forcing             = solenoidal shell OU, k=3..5
#
# Stage 0:
#   Confirm corrected viscosities at N=256 for Mt=0.40 and Mt=0.60.
#   The final N=512 case is run only if |Re_lambda - 130|/130 <= 5%.
#
# Stage 1:
#   Final N=512 production cases at Mt = 0.10, 0.25, 0.40, 0.50, 0.60.
#
# Statistics:
#   post-processing uses approximately 4 <= N_eddy <= 16.
#
# No core solver parameters are tuned inside this campaign.
# =============================================================================

REPO="${REPO:-$HOME/github/HIT_Solver}"
cd "$REPO" || exit 1

ROOT="${ROOT:-results/hit2d/final_thesis_Mach_ReL130_N512}"
PREFLIGHT="$ROOT/preflight_N256"
PRODUCTION="$ROOT/production_N512"
TARGET_RE="${TARGET_RE:-130.0}"
RE_TOL="${RE_TOL:-0.05}"
CFL="${CFL:-0.10}"
MN="${MN:-0.005}"
HV_INTERVAL="${HV_INTERVAL:-5}"
SEED="${SEED:-1234}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "$ROOT" "$PREFLIGHT" "$PRODUCTION"
CAMPAIGN_LOG="$ROOT/campaign.log"
FAILED="$ROOT/failed_cases.txt"
ACCEPTED="$ROOT/accepted_viscosities.csv"

: > "$FAILED"
echo "Mt,viscosity,source" > "$ACCEPTED"

exec > >(tee -a "$CAMPAIGN_LOG") 2>&1

echo "================================================================"
echo "FINAL THESIS HIT CAMPAIGN"
echo "Started: $(date)"
echo "Target stationary Re_lambda,2D = $TARGET_RE"
echo "Acceptance tolerance = ±$(python3 - <<PY
print(100*float("$RE_TOL"))
PY
)%"
echo "Frozen scheme: legacy sensor + local LF"
echo "HV: conservative_flux, mn=$MN, every $HV_INTERVAL steps"
echo "CFL=$CFL, seed=$SEED"
echo "================================================================"

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

prepare_output () {
    local OUT="$1"

    if [[ -d "$OUT" && "$OVERWRITE" == "1" ]]; then
        rm -rf "$OUT"
    elif [[ -d "$OUT" ]]; then
        if find "$OUT" -maxdepth 1 -name 'hit2d_step*.npz' -type f | grep -q .; then
            mv "$OUT" "${OUT}_old_$(date +%Y%m%d_%H%M%S)"
        fi
    fi
    mkdir -p "$OUT"
}

run_hit () {
    local NAME="$1"
    local OUT="$2"
    local N="$3"
    local MT="$4"
    local MU="$5"
    local TFINAL="$6"
    local SNAP="$7"
    local DIAG="$8"

    prepare_output "$OUT"

    echo
    echo "================================================================"
    echo "START $NAME at $(date)"
    echo "N=${N}x${N}; Mt=$MT; mu=$MU; tfinal=$TFINAL; CFL=$CFL"
    echo "================================================================"

    MPLBACKEND=Agg python3 -B scripts/run_hit2d.py \
      --backend cupy \
      --nx "$N" \
      --ny "$N" \
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
      --diagnostics-every "$DIAG" \
      --snapshot-every "$SNAP" \
      --seed "$SEED" \
      --output-dir "$OUT" \
      2>&1 | tee "$OUT/run.log"

    local STATUS=${PIPESTATUS[0]}
    if [[ "$STATUS" -ne 0 ]]; then
        echo "FAILED $NAME (simulation exit $STATUS)"
        echo "$NAME simulation_exit_$STATUS" >> "$FAILED"
        return 1
    fi

    echo "Post-processing $NAME"

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
    return 0
}

check_re () {
    local OUT="$1"
    local MT="$2"

    python3 -B scripts/check_stationary_re.py \
      --run-log "$OUT/run.log" \
      --mach "$MT" \
      --target "$TARGET_RE" \
      --tolerance "$RE_TOL" \
      --start-turnover 4 \
      --end-turnover 16
}

# -----------------------------------------------------------------------------
# Viscosities
#
# Mt=0.10 and 0.25 are retained from the already converged campaign.
#
# Mt=0.50 is directly calibrated: mean Re_lambda = 130.25 at N=256.
#
# Mt=0.40 and 0.60 use the one-step corrections inferred from the latest
# calibration and are explicitly confirmed below before N=512 production:
#
#   Mt=0.40: 0.001140000 -> 0.001098989
#   Mt=0.60: 0.001414050 -> 0.001330976
# -----------------------------------------------------------------------------

MU_010="0.0003024280385"
MU_025="0.0007560700962"
MU_040="0.001098989"
MU_050="0.00131565"
MU_060="0.001330976"

# Production physical times give roughly 18 turnover times using the same
# scaling employed in the successful thesis pilot/calibration campaigns.
TF_010="289.81192232"
TF_025="115.92476892"
TF_040="72.45298058"
TF_050="57.96238446"
TF_060="48.30198705"

# -----------------------------------------------------------------------------
# Stage 0 — confirmation at N=256
# -----------------------------------------------------------------------------

PASS_040=0
PASS_060=0

echo
echo "==================== STAGE 0: Re confirmation ===================="

OUT040="$PREFLIGHT/Mt040_N256_mu0p001098989"
if run_hit "preflight Mt=0.40" "$OUT040" 256 0.40 "$MU_040" "$TF_040" 3100 300; then
    if check_re "$OUT040" 0.40; then
        PASS_040=1
        echo "0.40,$MU_040,confirmed_N256" >> "$ACCEPTED"
    else
        echo "Mt040_N256 stationary_Re_outside_tolerance" >> "$FAILED"
    fi
fi

OUT060="$PREFLIGHT/Mt060_N256_mu0p001330976"
if run_hit "preflight Mt=0.60" "$OUT060" 256 0.60 "$MU_060" "$TF_060" 2100 200; then
    if check_re "$OUT060" 0.60; then
        PASS_060=1
        echo "0.60,$MU_060,confirmed_N256" >> "$ACCEPTED"
    else
        echo "Mt060_N256 stationary_Re_outside_tolerance" >> "$FAILED"
    fi
fi

# Already accepted from previous campaigns.
echo "0.10,$MU_010,previous_grid_Mach_campaign" >> "$ACCEPTED"
echo "0.25,$MU_025,previous_grid_Mach_campaign" >> "$ACCEPTED"
echo "0.50,$MU_050,latest_Re_calibration" >> "$ACCEPTED"

echo
echo "Preflight result: Mt0.40 pass=$PASS_040, Mt0.60 pass=$PASS_060"

# -----------------------------------------------------------------------------
# Stage 1 — final N=512 production
# -----------------------------------------------------------------------------

echo
echo "==================== STAGE 1: FINAL N=512 ===================="

# Equal-ish sampling density in turnover time:
# snapshot_every roughly scales as N/Mt.
run_hit \
  "FINAL Mt=0.10" \
  "$PRODUCTION/Mt010_N512" \
  512 0.10 "$MU_010" "$TF_010" 24800 2400

run_hit \
  "FINAL Mt=0.25" \
  "$PRODUCTION/Mt025_N512" \
  512 0.25 "$MU_025" "$TF_025" 10000 1000

if [[ "$PASS_040" == "1" ]]; then
    run_hit \
      "FINAL Mt=0.40" \
      "$PRODUCTION/Mt040_N512" \
      512 0.40 "$MU_040" "$TF_040" 6200 600
else
    echo "SKIPPING final Mt=0.40 because the N=256 Re confirmation did not pass."
fi

run_hit \
  "FINAL Mt=0.50" \
  "$PRODUCTION/Mt050_N512" \
  512 0.50 "$MU_050" "$TF_050" 5000 500

if [[ "$PASS_060" == "1" ]]; then
    run_hit \
      "FINAL Mt=0.60" \
      "$PRODUCTION/Mt060_N512" \
      512 0.60 "$MU_060" "$TF_060" 4200 400
else
    echo "SKIPPING final Mt=0.60 because the N=256 Re confirmation did not pass."
fi

# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------

echo
echo "==================== FINAL SUMMARY ===================="

python3 -B scripts/summarize_final_thesis_campaign.py \
  --root "$ROOT" \
  --target-re "$TARGET_RE" \
  --start-turnover 4 \
  --end-turnover 16

if [[ $? -ne 0 ]]; then
    echo "WARNING: final campaign summarizer failed."
fi

echo
echo "================================================================"
echo "FINAL THESIS CAMPAIGN FINISHED: $(date)"
echo "Root: $ROOT"
echo "Production: $PRODUCTION"
echo "Summary: $ROOT/final_thesis_summary.md"
echo "Accepted viscosities: $ACCEPTED"
echo "Failures/skips: $FAILED"
echo "================================================================"
