#!/usr/bin/env bash
set -euo pipefail

# Final figure-polishing / uncertainty pass.
# POST-PROCESSING ONLY: no CFD simulations and no solver modifications.

REPO="${REPO:-$HOME/github/HIT_Solver}"
ROOT="${ROOT:-$REPO/results/hit2d/final_thesis_Mach_ReL130_N512}"
PRODUCTION="${PRODUCTION:-$ROOT/production_N512}"
OUTPUT="${OUTPUT:-$ROOT/thesis_postprocess}"
START_TURNOVER="${START_TURNOVER:-4}"
END_TURNOVER="${END_TURNOVER:-16}"
TARGET_RE="${TARGET_RE:-130}"
CHI_FLOOR="${CHI_FLOOR:-1e-4}"
ABS_SPECTRUM_FLOOR="${ABS_SPECTRUM_FLOOR:-1e-16}"
UNCERTAINTY_STRIDE="${UNCERTAINTY_STRIDE:-1}"
RECOMPUTE_UNCERTAINTY="${RECOMPUTE_UNCERTAINTY:-0}"

cd "$REPO"
mkdir -p "$OUTPUT"
LOG="$OUTPUT/final_polish.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "FINAL THESIS FIGURE POLISH + TEMPORAL VARIABILITY"
echo "Started: $(date)"
echo "Stationary interval: $START_TURNOVER <= N_eddy <= $END_TURNOVER"
echo "chi_d(k) spectral floor: $CHI_FLOOR"
echo "absolute-spectrum display floor: $ABS_SPECTRUM_FLOOR x peak"
echo "============================================================"

for CASE in "$PRODUCTION"/Mt*_N512; do
    [[ -d "$CASE" ]] || continue

    # Ensure the pooled spectra/PDF files used by the final figures exist.
    NEED_DIAGNOSTICS=0
    [[ ! -f "$CASE/spectra_diagnostics.npz" ]] && NEED_DIAGNOSTICS=1
    [[ ! -f "$CASE/pdf_diagnostics.npz" ]] && NEED_DIAGNOSTICS=1
    if [[ "$NEED_DIAGNOSTICS" == "1" ]]; then
        echo "Regenerating missing stationary diagnostics for $(basename "$CASE")"
        MPLBACKEND=Agg python3 -B 2D/hit2d_isotropy_diagnostics.py \
          --snapshot-dir "$CASE" \
          --start-turnover "$START_TURNOVER" \
          --end-turnover "$END_TURNOVER" \
          --fluctuation-type reynolds \
          > "$CASE/final_polish_regeneration.log" 2>&1
    fi

    UNC="$CASE/thesis_uncertainty_diagnostics.npz"
    if [[ ! -f "$UNC" || "$RECOMPUTE_UNCERTAINTY" == "1" ]]; then
        if ! find "$CASE" -maxdepth 1 -name 'hit2d_step*.npz' -type f | grep -q .; then
            echo "ERROR: uncertainty calculation requires the original snapshots in $CASE" >&2
            exit 2
        fi
        echo "Computing temporal variability for $(basename "$CASE")"
        python3 -B scripts/compute_final_thesis_uncertainty.py \
          --snapshot-dir "$CASE" \
          --start-turnover "$START_TURNOVER" \
          --end-turnover "$END_TURNOVER" \
          --stride "$UNCERTAINTY_STRIDE" \
          --output "$UNC"
    else
        echo "Using cached uncertainty diagnostics for $(basename "$CASE")"
    fi
done

echo
echo "Creating polished final figures..."
MPLBACKEND=Agg python3 -B scripts/plot_final_thesis_polished.py \
  --root "$ROOT" \
  --output-dir "$OUTPUT" \
  --start-turnover "$START_TURNOVER" \
  --end-turnover "$END_TURNOVER" \
  --target-re "$TARGET_RE" \
  --chi-relative-floor "$CHI_FLOOR" \
  --absolute-spectrum-relative-floor "$ABS_SPECTRUM_FLOOR"

echo
echo "============================================================"
echo "FINAL POLISH FINISHED: $(date)"
echo "Outputs: $OUTPUT"
echo "No simulation was rerun."
echo "============================================================"
