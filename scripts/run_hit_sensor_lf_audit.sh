#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/github/HIT_Solver}"
cd "$ROOT"

NX="${NX:-256}"
NY="${NY:-256}"
TFINAL="${TFINAL:-70.0}"
MT="${MT:-0.25}"
REL="${REL:-120}"
CFL="${CFL:-0.10}"
MN="${MN:-0.005}"
HV_INTERVAL="${HV_INTERVAL:-5}"
DUCROS="${DUCROS:-0.50}"
JUMP="${JUMP:-0.04}"
COMPRESSION="${COMPRESSION:-2.5}"
OVERWRITE="${OVERWRITE:-0}"
START_TURNOVER="${START_TURNOVER:-4}"
END_TURNOVER="${END_TURNOVER:-10}"

CAMPAIGN="${CAMPAIGN:-results/hit2d/sensor_lf_audit_Mt0p25_ReL120_N256}"
mkdir -p "$CAMPAIGN"
LOG="$CAMPAIGN/campaign.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "HIT shock-sensor / WENO LF audit"
echo "Started: $(date)"
echo "Grid=${NX}x${NY}; Mt=${MT}; Re_lambda,0=${REL}; CFL=${CFL}"
echo "Conservative HV: mn=${MN}, every ${HV_INTERVAL} RK steps"
echo "Sensor controls: jump=${JUMP}, compression=${COMPRESSION}, Ducros=${DUCROS}"
echo "This is a full 3(sensor) x 2(LF) factorial comparison."
echo "============================================================"

run_case() {
    local sensor_mode="$1"
    local flux_mode="$2"
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
        echo "sensor=$sensor_mode; WENO LF=$flux_mode"
        echo "------------------------------------------------------------"

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
          --sensor-mode "$sensor_mode" \
          --sensor-width 4 \
          --jump-threshold "$JUMP" \
          --compression-threshold "$COMPRESSION" \
          --ducros-threshold "$DUCROS" \
          --shear-threshold 0 \
          --weno-flux-splitting "$flux_mode" \
          --mn "$MN" \
          --hyperviscosity-interval "$HV_INTERVAL" \
          --hyperviscosity-mode conservative_flux \
          --large-scale-drag 0.10 \
          --drag-kmax 2.0 \
          --cooling-time 5.0 \
          --diagnostics-every 5000 \
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

# Full factorial design. This isolates sensor effects and LF-splitting effects
# instead of changing both at once.
run_case legacy              global legacy_global
run_case legacy              local  legacy_local
run_case compression_gated   global gated_global
run_case compression_gated   local  gated_local
run_case directional         global directional_global
run_case directional         local  directional_local

echo
echo "Creating cross-case sensor/LF summary..."
MPLBACKEND=Agg python3 -B scripts/summarize_hit_sensor_lf_audit.py \
  "$CAMPAIGN" \
  --start-turnover "$START_TURNOVER" \
  --end-turnover "$END_TURNOVER"

echo "============================================================"
echo "Campaign finished: $(date)"
echo "Summary: $CAMPAIGN/sensor_lf_audit_summary.md"
echo "Comparison plot: $CAMPAIGN/sensor_lf_audit_comparison.png"
echo "============================================================"
