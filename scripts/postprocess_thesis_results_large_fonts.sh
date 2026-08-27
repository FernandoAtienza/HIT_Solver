#!/usr/bin/env bash

set -euo pipefail

REPO="${REPO:-$HOME/github/HIT_Solver}"
cd "$REPO"

ROOT="${ROOT:-results/hit2d/final_thesis_Mach_ReL130_N512}"
PRODUCTION="$ROOT/production_N512"
OUTPUT="$ROOT/thesis_postprocess"

RUNNER="scripts/run_with_thesis_fonts.py"

BASE_FONT="${BASE_FONT:-17}"
TITLE_FONT="${TITLE_FONT:-19}"
LABEL_FONT="${LABEL_FONT:-18}"
TICK_FONT="${TICK_FONT:-15}"
LEGEND_FONT="${LEGEND_FONT:-15}"
SUPTITLE_FONT="${SUPTITLE_FONT:-21}"
ANNOTATION_FONT="${ANNOTATION_FONT:-13}"
DPI="${DPI:-300}"

START_TURNOVER="${START_TURNOVER:-4}"
END_TURNOVER="${END_TURNOVER:-16}"

styled () {
    MPLBACKEND=Agg python3 -B "$RUNNER" \
        --base "$BASE_FONT" \
        --title "$TITLE_FONT" \
        --label "$LABEL_FONT" \
        --tick "$TICK_FONT" \
        --legend "$LEGEND_FONT" \
        --suptitle "$SUPTITLE_FONT" \
        --annotation "$ANNOTATION_FONT" \
        --dpi "$DPI" \
        "$@"
}

echo "===================================================="
echo "THESIS LARGE-FONT POST-PROCESSING"
echo
echo "Title font:   $TITLE_FONT"
echo "Axis labels:  $LABEL_FONT"
echo "Tick labels:  $TICK_FONT"
echo "Legends:      $LEGEND_FONT"
echo "Figure title: $SUPTITLE_FONT"
echo "DPI:          $DPI"
echo
echo "No CFD simulation will be run."
echo "===================================================="

for CASE in "$PRODUCTION"/Mt*_N512
do
    [ -d "$CASE" ] || continue

    echo
    echo "Processing $(basename "$CASE")"

    styled \
        2D/hit2d_viewer.py \
        --snapshot-dir "$CASE" \
        --physics-plots \
        --history-x-axis turnover

    styled \
        2D/hit2d_isotropy_diagnostics.py \
        --snapshot-dir "$CASE" \
        --start-turnover "$START_TURNOVER" \
        --end-turnover "$END_TURNOVER" \
        --fluctuation-type reynolds
done

mkdir -p "$OUTPUT"

if [ -f scripts/plot_final_thesis_hit_results.py ]
then
    styled \
        scripts/plot_final_thesis_hit_results.py \
        --root "$ROOT" \
        --start-turnover "$START_TURNOVER" \
        --end-turnover "$END_TURNOVER" \
        --target-re 130
fi

if [ -f scripts/plot_final_thesis_polished.py ]
then
    styled \
        scripts/plot_final_thesis_polished.py \
        --root "$ROOT" \
        --output-dir "$OUTPUT" \
        --start-turnover "$START_TURNOVER" \
        --end-turnover "$END_TURNOVER" \
        --target-re 130 \
        --chi-relative-floor 1e-4
fi

COLLECT="${COLLECT:-results/hit2d/thesis_figures_large_fonts}"

rm -rf "$COLLECT"
mkdir -p "$COLLECT"

find "$ROOT" \
    -type f \
    -name "*.png" \
    -print0 |
while IFS= read -r -d '' FILE
do
    REL="${FILE#$ROOT/}"
    SAFE="${REL//\//__}"

    cp -f \
        "$FILE" \
        "$COLLECT/$SAFE"
done

echo
echo "===================================================="
echo "POST-PROCESSING COMPLETE"
echo
echo "Large-font figures:"
echo "$COLLECT"
echo
echo "No simulation was rerun."
echo "===================================================="
