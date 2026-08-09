#!/usr/bin/env bash

# Sequential first-stage HIT campaign from the thesis roadmap.
#
# The campaign keeps the initial two-dimensional Taylor Reynolds number fixed
# and varies only the compact-node hyperviscosity strength.  Each completed run
# is post-processed before the next one starts.  Environment variables can be
# used to override the defaults without editing this file.

set -uo pipefail
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

REPO_ROOT="${REPO_ROOT:-$HOME/github/HIT_Solver}"
cd "$REPO_ROOT" || exit 1

BACKEND="${BACKEND:-cupy}"
NX="${NX:-256}"
NY="${NY:-256}"
TFINAL="${TFINAL:-70.0}"
CFL="${CFL:-0.10}"
MT="${MT:-0.25}"
RE_LAMBDA="${RE_LAMBDA:-120}"
HV_INTERVAL="${HV_INTERVAL:-5}"
DIAGNOSTICS_EVERY="${DIAGNOSTICS_EVERY:-250}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-2500}"
START_TURNOVER="${START_TURNOVER:-4}"
END_TURNOVER="${END_TURNOVER:-10}"
SEED="${SEED:-1234}"
OVERWRITE="${OVERWRITE:-0}"
POSTPROCESS="${POSTPROCESS:-1}"

MT_TAG="$(python3 - "$MT" <<'PY'
import sys
print(f"{float(sys.argv[1]):.2f}".replace(".", "p"))
PY
)"
REL_TAG="$(python3 - "$RE_LAMBDA" <<'PY'
import sys
value = float(sys.argv[1])
print(f"{value:g}".replace(".", "p"))
PY
)"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-results/hit2d/dissipation_audit_Mt${MT_TAG}_ReL${REL_TAG}_N${NX}}"
read -r -a MN_VALUES <<< "${MN_VALUES_OVERRIDE:-0.0 0.0005 0.001 0.002}"

mkdir -p "$CAMPAIGN_DIR"
CAMPAIGN_LOG="$CAMPAIGN_DIR/campaign.log"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "$CAMPAIGN_LOG"
}

mn_tag() {
    python3 - "$1" <<'PY'
import sys
value = float(sys.argv[1])
print(f"{value:.4f}".replace(".", "p"))
PY
}

case_is_complete() {
    local output_dir="$1"
    python3 - "$output_dir" "$TFINAL" <<'PY'
from pathlib import Path
import sys
import numpy as np

run_dir = Path(sys.argv[1])
tfinal = float(sys.argv[2])
paths = sorted(run_dir.glob("hit2d_step*.npz"))
if not paths:
    raise SystemExit(1)
try:
    with np.load(paths[-1]) as data:
        final_time = float(data["time"])
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if final_time >= tfinal - 1.0e-10 else 1)
PY
}

postprocess_case() {
    local output_dir="$1"
    if [[ "$POSTPROCESS" != "1" ]]; then
        log "POSTPROCESS=$POSTPROCESS; skipping automatic post-processing for $output_dir"
        return 0
    fi
    local post_dir="$output_dir/postprocess"
    mkdir -p "$post_dir"

    log "Post-processing $output_dir"

    if ! python3 -B 2D/hit2d_viewer.py \
        --snapshot-dir "$output_dir" \
        --physics-plots \
        --history-x-axis turnover \
        --physics-output "$post_dir/physics_fields_final.png" \
        --history-output "$post_dir/time_history.png" \
        >> "$output_dir/postprocess.log" 2>&1; then
        log "WARNING: viewer post-processing failed for $output_dir; see postprocess.log"
    fi

    if ! python3 -B 2D/hit2d_isotropy_diagnostics.py \
        --snapshot-dir "$output_dir" \
        --start-turnover "$START_TURNOVER" \
        --end-turnover "$END_TURNOVER" \
        --fluctuation-type reynolds \
        --history-x-axis turnover \
        >> "$output_dir/postprocess.log" 2>&1; then
        log "WARNING: isotropy/spectra/PDF post-processing failed for $output_dir; see postprocess.log"
    fi
}

run_case() {
    local mn="$1"
    local tag
    tag="$(mn_tag "$mn")"
    local case_name="Mt${MT_TAG}_ReL${REL_TAG}_N${NX}_mn${tag}"
    local output_dir="$CAMPAIGN_DIR/$case_name"

    if [[ "$OVERWRITE" == "1" && -d "$output_dir" ]]; then
        log "Removing existing case because OVERWRITE=1: $output_dir"
        rm -rf "$output_dir"
    fi

    if case_is_complete "$output_dir"; then
        log "Case already complete; skipping simulation: $case_name"
        postprocess_case "$output_dir"
        return 0
    fi

    if [[ -d "$output_dir" ]] && [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        log "ERROR: partial/non-empty case directory exists: $output_dir"
        log "Remove it manually or relaunch with OVERWRITE=1."
        return 1
    fi

    mkdir -p "$output_dir"
    log "Starting $case_name"
    log "Parameters: Mt=$MT, Re_lambda_2D(initial)=$RE_LAMBDA, N=${NX}x${NY}, CFL=$CFL, tfinal=$TFINAL, mn=$mn"

    if ! python3 -B scripts/run_hit2d.py \
        --backend "$BACKEND" \
        --nx "$NX" \
        --ny "$NY" \
        --tfinal "$TFINAL" \
        --cfl "$CFL" \
        --gamma 1.4 \
        --mach "$MT" \
        --initial-re-lambda "$RE_LAMBDA" \
        --prandtl 0.72 \
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
        --sensor-width 4 \
        --jump-threshold 0.04 \
        --compression-threshold 2.5 \
        --mn "$mn" \
        --hyperviscosity-interval "$HV_INTERVAL" \
        --large-scale-drag 0.10 \
        --drag-kmax 2.0 \
        --cooling-time 5.0 \
        --seed "$SEED" \
        --diagnostics-every "$DIAGNOSTICS_EVERY" \
        --snapshot-every "$SNAPSHOT_EVERY" \
        --output-dir "$output_dir" \
        > "$output_dir/run.log" 2>&1; then
        log "ERROR: simulation failed for $case_name; see $output_dir/run.log"
        return 1
    fi

    log "Simulation completed: $case_name"
    postprocess_case "$output_dir"
    log "Completed and post-processed: $case_name"
}

log "============================================================"
log "Starting sequential HIT dissipation audit"
log "Campaign directory: $CAMPAIGN_DIR"
log "Hyperviscosity values: ${MN_VALUES[*]}"
log "============================================================"

for mn in "${MN_VALUES[@]}"; do
    if ! run_case "$mn"; then
        log "Campaign stopped after a simulation failure."
        exit 1
    fi
done

log "Building cross-case summary"
if ! python3 -B scripts/summarize_hit_dissipation_audit.py \
    "$CAMPAIGN_DIR" \
    --start-turnover "$START_TURNOVER" \
    --end-turnover "$END_TURNOVER" \
    >> "$CAMPAIGN_LOG" 2>&1; then
    log "WARNING: cross-case summary failed; individual cases remain available."
fi

log "============================================================"
log "All dissipation-audit simulations finished"
log "Summary: $CAMPAIGN_DIR/dissipation_audit_summary.md"
log "Spectra: $CAMPAIGN_DIR/dissipation_audit_spectra.png"
log "============================================================"
