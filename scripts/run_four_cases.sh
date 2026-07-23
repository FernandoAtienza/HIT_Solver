#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   CPU: BACKEND=numpy NX=128 NY=128 bash scripts/run_four_cases.sh
#   GPU: BACKEND=cupy  NX=256 NY=256 bash scripts/run_four_cases.sh
#
# Defaults to CPU so that an installed but incomplete CuPy/CUDA stack is never
# selected accidentally.

BACKEND="${BACKEND:-numpy}"
NX="${NX:-512}"
NY="${NY:-512}"
TOT_INITIAL="${TOT_INITIAL:-4}"
TOT_FINAL="${TOT_FINAL:-16}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-75}"
DIAGNOSTICS_EVERY="${DIAGNOSTICS_EVERY:-25}"

case "$BACKEND" in
    numpy|cupy) ;;
    *)
        echo "ERROR: BACKEND must be 'numpy' (CPU) or 'cupy' (GPU)." >&2
        exit 2
        ;;
esac

run_case() {
    local mode="$1"
    local mach="$2"
    local mach_tag="${mach/./p}"
    local timestamp
    timestamp="$(date +%Y%m%d_%H%M%S)"

    local run_dir="2D/hit2d_snapshots/${mode}_Mt${mach_tag}_ToT${TOT_FINAL}_${BACKEND}_${timestamp}"
    mkdir -p "$run_dir"

    echo
    echo "============================================================"
    echo "Running forcing=${mode}, Mt=${mach}, backend=${BACKEND}"
    echo "Grid=${NX}x${NY}; statistics interval=ToT ${TOT_INITIAL}-${TOT_FINAL}"
    echo "Output: ${run_dir}"
    echo "============================================================"

    python -u 2D/hit2d_viewer.py \
        --run \
        --backend "$BACKEND" \
        --no-timestamp-dir \
        --snapshot-dir "$run_dir" \
        --nx "$NX" \
        --ny "$NY" \
        --tot-final "$TOT_FINAL" \
        --tot-initial-data "$TOT_INITIAL" \
        --mach "$mach" \
        --mach-control \
        --mach-control-target "$mach" \
        --initial-kmin 3 \
        --initial-kmax 5 \
        --kf-min 3 \
        --kf-max 5 \
        --forcing-mode "$mode" \
        --p-target 1.0e-3 \
        --forcing-correlation-time 1.0 \
        --forcing-alpha-memory 0.2 \
        --viscosity 7.5e-4 \
        --mn 0.002 \
        --large-scale-drag 0.10 \
        --drag-kmax 2.0 \
        --cooling-time 5.0 \
        --diagnostics-every "$DIAGNOSTICS_EVERY" \
        --snapshot-every "$SNAPSHOT_EVERY" \
        --diagnostic-plots-only \
        2>&1 | tee "$run_dir/simulation.log"

    python -u 2D/hit2d_isotropy_diagnostics.py \
        --snapshot-dir "$run_dir" \
        --start-turnover "$TOT_INITIAL" \
        --end-turnover "$TOT_FINAL" \
        2>&1 | tee "$run_dir/isotropy_postprocess.log"

    python -u 2D/hit2d_two_point_correlation.py \
        --snapshot-dir "$run_dir" \
        2>&1 | tee "$run_dir/correlation_postprocess.log"
}

run_case solenoidal 0.25
run_case solenoidal 0.50
run_case dilatational 0.25
run_case dilatational 0.50

echo
echo "All four simulations and postprocessing stages completed."
