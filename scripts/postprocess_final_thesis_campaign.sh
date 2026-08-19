#!/usr/bin/env bash
set -euo pipefail

# Final, post-processing-only thesis workflow.
# It does not modify the solver and does not launch CFD simulations.

REPO="${REPO:-$HOME/github/HIT_Solver}"
ROOT="${ROOT:-$REPO/results/hit2d/final_thesis_Mach_ReL130_N512}"
PRODUCTION="${PRODUCTION:-$ROOT/production_N512}"
START_TURNOVER="${START_TURNOVER:-4}"
END_TURNOVER="${END_TURNOVER:-16}"
TARGET_RE="${TARGET_RE:-130}"
REPROCESS="${REPROCESS:-0}"

cd "$REPO"

if [[ ! -d "$PRODUCTION" ]]; then
    echo "Production directory not found: $PRODUCTION" >&2
    exit 1
fi

mkdir -p "$ROOT/thesis_postprocess"
LOG="$ROOT/thesis_postprocess/postprocess.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "FINAL THESIS POST-PROCESSING"
echo "Started: $(date)"
echo "Root: $ROOT"
echo "Statistical window: $START_TURNOVER <= N_eddy <= $END_TURNOVER"
echo "============================================================"

for CASE in "$PRODUCTION"/Mt*_N512; do
    [[ -d "$CASE" ]] || continue

    NEED=0
    [[ ! -f "$CASE/spectra_diagnostics.npz" ]] && NEED=1
    [[ ! -f "$CASE/pdf_diagnostics.npz" ]] && NEED=1
    [[ ! -f "$CASE/isotropy_diagnostics.npz" ]] && NEED=1
    [[ "$REPROCESS" == "1" ]] && NEED=1

    if [[ "$NEED" == "1" ]]; then
        if ! find "$CASE" -maxdepth 1 -name 'hit2d_step*.npz' -type f | grep -q .; then
            echo "ERROR: $CASE needs diagnostic regeneration but no snapshots are available." >&2
            exit 2
        fi
        echo
        echo "Recomputing stationary diagnostics for $(basename "$CASE")"
        MPLBACKEND=Agg python3 -B 2D/hit2d_isotropy_diagnostics.py \
          --snapshot-dir "$CASE" \
          --start-turnover "$START_TURNOVER" \
          --end-turnover "$END_TURNOVER" \
          --fluctuation-type reynolds \
          > "$CASE/final_thesis_postprocess.log" 2>&1
    else
        echo "Using existing diagnostics for $(basename "$CASE")"
    fi
done

echo
echo "Creating cross-Mach thesis figures and tables..."
MPLBACKEND=Agg python3 -B scripts/plot_final_thesis_hit_results.py \
  --root "$ROOT" \
  --start-turnover "$START_TURNOVER" \
  --end-turnover "$END_TURNOVER" \
  --target-re "$TARGET_RE"

echo
echo "============================================================"
echo "POST-PROCESSING FINISHED: $(date)"
echo "Outputs: $ROOT/thesis_postprocess"
echo "============================================================"
