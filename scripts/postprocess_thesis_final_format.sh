#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$HOME/github/HIT_Solver}"
cd "$REPO"

REF_CASE="${REF_CASE:-results/hit2d/solenoidal_Mt025_N512_CFL_0p05}"
FINAL_ROOT="${FINAL_ROOT:-results/hit2d/final_thesis_Mach_ReL130_N512}"
FINAL_OUT="$FINAL_ROOT/thesis_postprocess"
EXPORT_DIR="${EXPORT_DIR:-results/hit2d/thesis_figures_final_layout}"

START_TURNOVER="${START_TURNOVER:-4}"
END_TURNOVER="${END_TURNOVER:-16}"
DPI="${DPI:-300}"

RUNNER="scripts/run_with_thesis_fonts.py"

styled_reference () {
    MPLBACKEND=Agg python3 -B "$RUNNER" \
      --base "${REF_BASE_FONT:-18}" \
      --title "${REF_TITLE_FONT:-22}" \
      --label "${REF_LABEL_FONT:-20}" \
      --tick "${REF_TICK_FONT:-17}" \
      --legend "${REF_LEGEND_FONT:-14}" \
      --suptitle "${REF_SUPTITLE_FONT:-24}" \
      --annotation "${REF_ANNOTATION_FONT:-12}" \
      --dpi "$DPI" \
      "$@"
}

# Cross-Mach figures need large axes/titles/ticks, but deliberately slightly
# smaller legends so the five Mach-number curves are not obscured.
styled_cross () {
    MPLBACKEND=Agg python3 -B "$RUNNER" \
      --base "${CROSS_BASE_FONT:-17}" \
      --title "${CROSS_TITLE_FONT:-21}" \
      --label "${CROSS_LABEL_FONT:-19}" \
      --tick "${CROSS_TICK_FONT:-16}" \
      --legend "${CROSS_LEGEND_FONT:-10}" \
      --suptitle "${CROSS_SUPTITLE_FONT:-23}" \
      --annotation "${CROSS_ANNOTATION_FONT:-11}" \
      --dpi "$DPI" \
      "$@"
}

echo "============================================================"
echo "FINAL THESIS FIGURE POST-PROCESSING"
echo "Reference case: $REF_CASE"
echo "Final campaign: $FINAL_ROOT"
echo "No CFD simulation will be launched."
echo "Reference fonts: title=${REF_TITLE_FONT:-22}, labels=${REF_LABEL_FONT:-20}, ticks=${REF_TICK_FONT:-17}, legend=${REF_LEGEND_FONT:-14}"
echo "Cross-Mach fonts: title=${CROSS_TITLE_FONT:-21}, labels=${CROSS_LABEL_FONT:-19}, ticks=${CROSS_TICK_FONT:-16}, legend=${CROSS_LEGEND_FONT:-10}"
echo "============================================================"

# 1) Detailed Mt=0.25 reference case. This reruns plotting/diagnostics only.
styled_reference 2D/hit2d_viewer.py \
  --snapshot-dir "$REF_CASE" \
  --physics-plots \
  --history-x-axis turnover

styled_reference 2D/hit2d_isotropy_diagnostics.py \
  --snapshot-dir "$REF_CASE" \
  --start-turnover "$START_TURNOVER" \
  --end-turnover "$END_TURNOVER" \
  --fluctuation-type reynolds

# 2) Cross-Mach figures.
mkdir -p "$FINAL_OUT"
styled_cross scripts/plot_final_thesis_hit_results.py \
  --root "$FINAL_ROOT" \
  --start-turnover "$START_TURNOVER" \
  --end-turnover "$END_TURNOVER" \
  --target-re 130 \
  --dpi "$DPI" \
  --chi-relative-floor 1e-4

styled_cross scripts/plot_final_thesis_polished.py \
  --root "$FINAL_ROOT" \
  --output-dir "$FINAL_OUT" \
  --start-turnover "$START_TURNOVER" \
  --end-turnover "$END_TURNOVER" \
  --target-re 130 \
  --chi-relative-floor 1e-4 \
  --absolute-spectrum-relative-floor 1e-16 \
  --dpi "$DPI"

# 3) Collect the exact filenames expected by memoria.tex.
# Keep any already-exported thesis figures that this script does not regenerate
# (e.g. grid-convergence/repeatability figures) and overwrite the regenerated
# filenames below.
mkdir -p "$EXPORT_DIR"

copy_if () {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp -f "$src" "$EXPORT_DIR/$dst"
  else
    echo "WARNING: missing $src"
  fi
}

copy_if "$REF_CASE/postprocess/hit2d_history.png" "reference_Mt025_history.png"
copy_if "$REF_CASE/postprocess/physics_fields_final.png" "reference_Mt025_fields.png"
copy_if "$REF_CASE/postprocess/energy_enstrophy_spectra.png" "reference_Mt025_spectra.png"
copy_if "$REF_CASE/postprocess/one_point_pdfs.png" "reference_Mt025_pdfs.png"
copy_if "$REF_CASE/postprocess/component_energy_anisotropy.png" "reference_Mt025_anisotropy.png"
copy_if "$REF_CASE/postprocess/directional_isotropy_correlations.png" "reference_Mt025_correlations.png"

copy_if "$FINAL_OUT/final_low_high_fields_normalized.png" "low_high_fields_normalized.png"
copy_if "$FINAL_OUT/final_mach_spectra_polished.png" "mach_spectra_polished.png"
copy_if "$FINAL_OUT/final_mach_trends_uncertainty.png" "mach_trends_uncertainty.png"
copy_if "$FINAL_OUT/final_pdf_moments_uncertainty.png" "pdf_moments_uncertainty.png"
copy_if "$FINAL_OUT/final_mach_pdfs.png" "mach_pdfs.png"

# Retain other already-used figures if a prior Overleaf export is supplied.
if [[ -n "${EXISTING_RESULTS_DIR:-}" && -d "$EXISTING_RESULTS_DIR" ]]; then
  for f in grid_convergence_spectra.png repeatability_spectra.png stationarity_Mt025.png stationarity_Mt060.png; do
    [[ -f "$EXISTING_RESULTS_DIR/$f" ]] && cp -f "$EXISTING_RESULTS_DIR/$f" "$EXPORT_DIR/$f"
  done
fi

echo
echo "Finished. Thesis figures are in:"
echo "  $REPO/$EXPORT_DIR"
echo "Copy that directory over Overleaf's imagenes/results/."
