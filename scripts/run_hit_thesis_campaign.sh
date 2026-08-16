#!/usr/bin/env bash
#
# Thesis-oriented HIT production/pilot campaign.
#
# Numerical baseline frozen after the hyperviscosity and sensor/LF audits:
#   sensor                 = legacy
#   WENO LF splitting      = local stencil maximum
#   conservative HV        = mn=0.005 every 5 full RK steps (no dt scaling)
#   CFL                    = 0.10
#   initial Re_lambda,2D   = 120
#
# The campaign runs sequentially on one GPU.  It first completes the Mt=0.25
# grid-convergence set, then the fixed-Re Mach pilot, then an independent-seed
# repeatability case.  Every completed run is post-processed automatically.
#
# Environment overrides:
#   ROOT, BACKEND, CFL, REL, MN, HV_INTERVAL, TARGET_TURNOVERS,
#   START_TURNOVER, END_TURNOVER, OVERWRITE, POSTPROCESS, CAMPAIGN
#
# Example:
#   nohup ./scripts/run_hit_thesis_campaign.sh > thesis_campaign_launcher.log 2>&1 &

set -uo pipefail
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

ROOT="${ROOT:-$HOME/github/HIT_Solver}"
BACKEND="${BACKEND:-cupy}"
CFL="${CFL:-0.10}"
REL="${REL:-120}"
MN="${MN:-0.005}"
HV_INTERVAL="${HV_INTERVAL:-5}"
TARGET_TURNOVERS="${TARGET_TURNOVERS:-18.0}"
START_TURNOVER="${START_TURNOVER:-4.0}"
END_TURNOVER="${END_TURNOVER:-16.0}"
OVERWRITE="${OVERWRITE:-0}"
POSTPROCESS="${POSTPROCESS:-1}"
RUN_GRID="${RUN_GRID:-1}"
RUN_MACH="${RUN_MACH:-1}"
RUN_REPEAT="${RUN_REPEAT:-1}"
RUN_N512="${RUN_N512:-1}"
CAMPAIGN="${CAMPAIGN:-results/hit2d/thesis_hit_campaign_v1}"

cd "$ROOT" || exit 1
mkdir -p "$CAMPAIGN"
LOG="$CAMPAIGN/campaign.log"
FAILED_FILE="$CAMPAIGN/failed_cases.txt"
: > "$FAILED_FILE"
exec > >(tee -a "$LOG") 2>&1

# The turnover reference used by the post-processor is L_ref = pi/2.
# With a_mean ~= 1 and the Mach controller holding u_rms ~= Mt, this gives a
# conservative final physical time that extends slightly beyond the requested
# number of turnover times.  Post-processing always uses the ACTUAL turnover
# history and the common 4 <= N_eddy <= 16 interval.
tfinal_for_mt() {
    python3 - "$1" "$TARGET_TURNOVERS" <<'PY'
import math, sys
mt = float(sys.argv[1])
target = float(sys.argv[2])
if mt <= 0:
    raise SystemExit("Mt must be positive")
# t ~= Neddy * L_ref / u_rms = Neddy * (pi/2) / Mt.
# Add a 2.5% margin to ensure the post-processing window is fully covered.
print(f"{1.025 * target * math.pi / (2.0 * mt):.8f}")
PY
}

# Keep approximately the same spacing in turnover time across grid and Mach
# number.  The reference is 5000 steps for N=256, Mt=0.25.
stride_for_case() {
    python3 - "$1" "$2" "$3" <<'PY'
import sys
n = int(sys.argv[1])
mt = float(sys.argv[2])
base = float(sys.argv[3])
value = base * (n / 256.0) * (0.25 / mt)
# Round to a practical multiple of 50 and enforce a minimum.
value = max(50, int(round(value / 50.0) * 50))
print(value)
PY
}

case_complete() {
    local case_dir="$1"
    local expected_tfinal="$2"
    python3 - "$case_dir" "$expected_tfinal" <<'PY'
from pathlib import Path
import sys, numpy as np
case = Path(sys.argv[1])
expected = float(sys.argv[2])
paths = sorted(case.glob("hit2d_step*.npz"))
if not paths or not (case / "diagnostic_history.npz").is_file():
    raise SystemExit(1)
try:
    with np.load(paths[-1]) as data:
        t = float(data["time"])
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if t >= expected - max(1e-8, 1e-7 * expected) else 1)
PY
}

postprocess_case() {
    local case_dir="$1"
    if [[ "$POSTPROCESS" != "1" ]]; then
        return 0
    fi

    echo "Post-processing: $case_dir"
    MPLBACKEND=Agg python3 -B 2D/hit2d_viewer.py \
      --snapshot-dir "$case_dir" \
      --physics-plots \
      --history-x-axis turnover || return 1

    MPLBACKEND=Agg python3 -B 2D/hit2d_isotropy_diagnostics.py \
      --snapshot-dir "$case_dir" \
      --start-turnover "$START_TURNOVER" \
      --end-turnover "$END_TURNOVER" \
      --fluctuation-type reynolds || return 1
}

run_case() {
    local tag="$1"
    local nx="$2"
    local mt="$3"
    local seed="$4"

    local ny="$nx"
    local tfinal
    tfinal="$(tfinal_for_mt "$mt")"
    local snapshot_every
    local diagnostics_every
    snapshot_every="$(stride_for_case "$nx" "$mt" 5000)"
    diagnostics_every="$(stride_for_case "$nx" "$mt" 500)"
    local case_dir="$CAMPAIGN/$tag"

    if [[ "$OVERWRITE" == "1" && -d "$case_dir" ]]; then
        rm -rf "$case_dir"
    fi

    if case_complete "$case_dir" "$tfinal"; then
        echo "Skipping completed case: $tag"
        postprocess_case "$case_dir" || {
            echo "WARNING: post-processing failed for completed case $tag"
            echo "$tag:postprocess" >> "$FAILED_FILE"
        }
        return 0
    fi

    if [[ -d "$case_dir" ]] && compgen -G "$case_dir/*" >/dev/null; then
        local backup="${case_dir}.incomplete_$(date +%Y%m%d_%H%M%S)"
        echo "Incomplete previous case found. Moving it to: $backup"
        mv "$case_dir" "$backup"
    fi
    mkdir -p "$case_dir"

    echo
    echo "============================================================"
    echo "STARTING $tag at $(date)"
    echo "N=${nx}x${ny}; Mt=${mt}; Re_lambda,0=${REL}; seed=${seed}"
    echo "tfinal=${tfinal}; target turnovers=${TARGET_TURNOVERS}"
    echo "CFL=${CFL}; sensor=legacy; WENO LF=local"
    echo "HV: conservative_flux, mn=${MN}, interval=${HV_INTERVAL}"
    echo "snapshot_every=${snapshot_every}; diagnostics_every=${diagnostics_every}"
    echo "============================================================"

    if ! MPLBACKEND=Agg python3 -B scripts/run_hit2d.py \
      --backend "$BACKEND" \
      --nx "$nx" \
      --ny "$ny" \
      --tfinal "$tfinal" \
      --cfl "$CFL" \
      --gamma 1.4 \
      --mach "$mt" \
      --initial-kmin 3 \
      --initial-kmax 5 \
      --initial-re-lambda "$REL" \
      --prandtl 0.72 \
      --kf-min 3 \
      --kf-max 5 \
      --p-target 1.0e-3 \
      --forcing-correlation-time 0.5 \
      --forcing-alpha-memory 0.2 \
      --min-forcing-power 1.0e-6 \
      --max-forcing-rescale 20.0 \
      --mach-control \
      --mach-control-target "$mt" \
      --mach-control-memory 0.995 \
      --mach-control-exponent 2.0 \
      --sensor-mode legacy \
      --sensor-width 4 \
      --jump-threshold 0.04 \
      --compression-threshold 2.5 \
      --ducros-threshold 0.5 \
      --shear-threshold 0 \
      --weno-flux-splitting local \
      --mn "$MN" \
      --hyperviscosity-interval "$HV_INTERVAL" \
      --hyperviscosity-mode conservative_flux \
      --large-scale-drag 0.10 \
      --drag-kmax 2.0 \
      --cooling-time 5.0 \
      --diagnostics-every "$diagnostics_every" \
      --snapshot-every "$snapshot_every" \
      --seed "$seed" \
      --output-dir "$case_dir" \
      2>&1 | tee "$case_dir/run.log"; then
        echo "ERROR: simulation failed: $tag"
        echo "$tag:simulation" >> "$FAILED_FILE"
        return 1
    fi

    if ! postprocess_case "$case_dir"; then
        echo "ERROR: post-processing failed: $tag"
        echo "$tag:postprocess" >> "$FAILED_FILE"
        return 1
    fi

    echo "COMPLETED: $tag at $(date)"
    return 0
}

run_or_continue() {
    run_case "$@" || true
}

echo "============================================================"
echo "THESIS HIT CAMPAIGN v1"
echo "Started: $(date)"
echo "Frozen numerical baseline: legacy sensor + local LF"
echo "CFL=${CFL}; Re_lambda,0=${REL}; mn=${MN}; HV interval=${HV_INTERVAL}"
echo "Statistics: ${START_TURNOVER} <= N_eddy <= ${END_TURNOVER}"
echo "============================================================"

# A. Grid convergence at the reference Mach number.
if [[ "$RUN_GRID" == "1" ]]; then
    run_or_continue grid_Mt025_N128_seed1234 128 0.25 1234
    run_or_continue grid_Mt025_N256_seed1234 256 0.25 1234
    if [[ "$RUN_N512" == "1" ]]; then
        run_or_continue grid_Mt025_N512_seed1234 512 0.25 1234
    fi
fi

# B. Fixed-initial-Re Mach-number pilot.  The Mt=0.25 point is the N=256 grid case above.
if [[ "$RUN_MACH" == "1" ]]; then
    # Make sure the shared Mt=0.25/N=256 reference exists even if RUN_GRID=0.
    run_or_continue grid_Mt025_N256_seed1234 256 0.25 1234
    run_or_continue mach_Mt010_N256_seed1234 256 0.10 1234
    run_or_continue mach_Mt050_N256_seed1234 256 0.50 1234
    run_or_continue mach_Mt060_N256_seed1234 256 0.60 1234
fi

# C. Independent realization to estimate stochastic/isotropy sensitivity.
if [[ "$RUN_REPEAT" == "1" ]]; then
    run_or_continue repeat_Mt025_N256_seed5678 256 0.25 5678
fi

echo
echo "Creating thesis campaign comparison figures and tables..."
MPLBACKEND=Agg python3 -B scripts/summarize_hit_thesis_campaign.py \
  "$CAMPAIGN" \
  --start-turnover "$START_TURNOVER" \
  --end-turnover "$END_TURNOVER" || {
    echo "WARNING: final campaign summary failed"
    echo "campaign:summary" >> "$FAILED_FILE"
}

echo "============================================================"
echo "Campaign finished: $(date)"
if [[ -s "$FAILED_FILE" ]]; then
    echo "Some items failed; see: $FAILED_FILE"
    cat "$FAILED_FILE"
else
    echo "All requested simulations and post-processing completed successfully."
fi
echo "Results: $CAMPAIGN"
echo "============================================================"
