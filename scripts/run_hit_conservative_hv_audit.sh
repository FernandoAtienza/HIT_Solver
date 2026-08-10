#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/github/HIT_Solver}"
cd "$ROOT"

NX="${NX:-256}"
NY="${NY:-256}"
TFINAL="${TFINAL:-70.0}"
MT="${MT:-0.25}"
REL="${REL:-120}"
BASE_CFL="${BASE_CFL:-0.10}"
HV_INTERVAL="${HV_INTERVAL:-5}"
OVERWRITE="${OVERWRITE:-0}"
START_TURNOVER="${START_TURNOVER:-4}"
END_TURNOVER="${END_TURNOVER:-10}"

CAMPAIGN="${CAMPAIGN:-results/hit2d/conservative_hv_audit_Mt0p25_ReL120_N256}"
mkdir -p "$CAMPAIGN"
LOG="$CAMPAIGN/campaign.log"

exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "Conservative HIT hyperviscosity audit"
echo "Started: $(date)"
echo "Grid: ${NX}x${NY}; Mt=${MT}; Re_lambda,0=${REL}"
echo "HV interval: every ${HV_INTERVAL} complete RK steps"
echo "IMPORTANT: no dt scaling is applied; effective HV rate is CFL dependent."
echo "============================================================"

run_case() {
    local mn="$1"
    local cfl="$2"
    local tag="$3"
    local case_dir="$CAMPAIGN/$tag"

    if [[ "$OVERWRITE" == "1" && -d "$case_dir" ]]; then
        rm -rf "$case_dir"
    fi

    if [[ -f "$case_dir/diagnostic_history.npz" ]] && compgen -G "$case_dir/hit2d_step*.npz" >/dev/null; then
        echo "Skipping completed case: $tag"
    else
        mkdir -p "$case_dir"
        echo
        echo "------------------------------------------------------------"
        echo "Starting $tag at $(date)"
        echo "mn=$mn CFL=$cfl mode=conservative_flux interval=$HV_INTERVAL"
        echo "------------------------------------------------------------"

        MPLBACKEND=Agg python3 -B scripts/run_hit2d.py \
          --backend cupy \
          --nx "$NX" \
          --ny "$NY" \
          --tfinal "$TFINAL" \
          --cfl "$cfl" \
          --gamma 1.4 \
          --mach "$MT" \
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
          --mach-control-target "$MT" \
          --mach-control-memory 0.995 \
          --mach-control-exponent 2.0 \
          --sensor-width 4 \
          --jump-threshold 0.04 \
          --compression-threshold 2.5 \
          --mn "$mn" \
          --hyperviscosity-interval "$HV_INTERVAL" \
          --hyperviscosity-mode conservative_flux \
          --large-scale-drag 0.10 \
          --drag-kmax 2.0 \
          --cooling-time 5.0 \
          --diagnostics-every 500 \
          --snapshot-every 5000 \
          --seed 1234 \
          --output-dir "$case_dir"
    fi

    echo "Post-processing $tag"
    MPLBACKEND=Agg python3 -B 2D/hit2d_viewer.py \
      --snapshot-dir "$case_dir" \
      --physics-plots \
      --history-x-axis turnover

    MPLBACKEND=Agg python3 -B 2D/hit2d_isotropy_diagnostics.py \
      --snapshot-dir "$case_dir" \
      --start-turnover "$START_TURNOVER" \
      --end-turnover "$END_TURNOVER" \
      --fluctuation-type reynolds
}

# Stage A: coefficient sweep at one CFL. 0.01 is included deliberately to
# determine whether its stronger cutoff damping starts contaminating resolved k.
run_case 0.001  "$BASE_CFL" "mn0p001_CFL0p10"
run_case 0.002  "$BASE_CFL" "mn0p002_CFL0p10"
run_case 0.005  "$BASE_CFL" "mn0p005_CFL0p10"
run_case 0.010  "$BASE_CFL" "mn0p010_CFL0p10"

# Stage B: quantify the accepted CFL dependence of a fixed every-5-step filter.
# The BASE_CFL=0.10 result above is the middle point and is not repeated.
run_case 0.002  0.05 "mn0p002_CFL0p05"
run_case 0.002  0.20 "mn0p002_CFL0p20"

echo
echo "Creating cross-case summary..."
MPLBACKEND=Agg python3 -B scripts/summarize_hit_dissipation_audit.py \
  "$CAMPAIGN" \
  --start-turnover "$START_TURNOVER" \
  --end-turnover "$END_TURNOVER"

echo "============================================================"
echo "Campaign finished: $(date)"
echo "Summary: $CAMPAIGN/dissipation_audit_summary.md"
echo "Spectra: $CAMPAIGN/dissipation_audit_spectra.png"
echo "============================================================"
