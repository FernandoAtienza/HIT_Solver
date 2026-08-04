#!/usr/bin/env bash

set -euo pipefail

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd ~/github/HIT_Solver

BACKEND="cupy"
NX=128
NY=128
DIAGNOSTICS_EVERY=1000
SEED=1234

# CFL values to test.
CFLS=(
0.05
0.08
0.10
0.12
0.15
0.20
)

# Snapshot intervals selected so that all CFL cases save fields at
# approximately the same physical-time spacing.
#
# Reference case:
# CFL = 0.05 -> snapshot every 5000 steps
#
# Scaling:
# snapshot_every(CFL) ~= 5000 * 0.05 / CFL
snapshot_interval_for_cfl() {
    case "$1" in
        0.05) echo 5000 ;;
        0.08) echo 3125 ;;
        0.10) echo 2500 ;;
        0.12) echo 2083 ;;
        0.15) echo 1667 ;;
        0.20) echo 1250 ;;
        *)
            echo "ERROR: No snapshot interval defined for CFL=$1" >&2
            return 1
            ;;
    esac
}

run_and_postprocess() {

    local mt="$1"
    local tfinal="$2"
    local case_name="$3"
    local cfl="$4"
    local snapshot_every="$5"

    local output_dir="results/hit2d/${case_name}"
    local post_dir="${output_dir}/postprocess"

    mkdir -p "${output_dir}" "${post_dir}"

    echo
    echo "============================================================"
    echo "RUNNING: ${case_name}"
    echo "Mt target       : ${mt}"
    echo "Grid            : ${NX}x${NY}"
    echo "CFL             : ${cfl}"
    echo "Snapshot every  : ${snapshot_every} steps"
    echo "Diagnostics every: ${DIAGNOSTICS_EVERY} steps"
    echo "Output          : ${output_dir}"
    echo "============================================================"

    python3 -B scripts/run_hit2d.py \
        --backend "${BACKEND}" \
        --nx "${NX}" \
        --ny "${NY}" \
        --tfinal "${tfinal}" \
        --cfl "${cfl}" \
        --gamma 1.4 \
        --mach "${mt}" \
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
        --mach-control-target "${mt}" \
        --mach-control-memory 0.995 \
        --mach-control-exponent 2.0 \
        --viscosity 7.5e-4 \
        --mn 0.002 \
        --large-scale-drag 0.10 \
        --drag-kmax 2.0 \
        --cooling-time 5.0 \
        --seed "${SEED}" \
        --diagnostics-every "${DIAGNOSTICS_EVERY}" \
        --snapshot-every "${snapshot_every}" \
        --output-dir "${output_dir}" \
        2>&1 | tee "${output_dir}/run.log"

    python3 -B 2D/hit2d_viewer.py \
        --snapshot-dir "${output_dir}" \
        --physics-plots \
        --history-x-axis turnover \
        --physics-output "${post_dir}/physics_fields_final.png" \
        --history-output "${post_dir}/time_history.png" \
        2>&1 | tee "${output_dir}/viewer_postprocess.log"

    python3 -B 2D/hit2d_isotropy_diagnostics.py \
        --snapshot-dir "${output_dir}" \
        --start-turnover 4 \
        --end-turnover 16 \
        --fluctuation-type reynolds \
        2>&1 | tee "${output_dir}/isotropy_postprocess.log"

    echo
    echo "Finished ${case_name}"
    echo
}

for CFL in "${CFLS[@]}"; do

    CFL_TAG=$(printf "%.2f" "${CFL}" | tr '.' 'p')
    SNAPSHOT_EVERY=$(snapshot_interval_for_cfl "${CFL}")

    run_and_postprocess \
        0.25 \
        110.0 \
        "CFL_study_equal_sampling_${CFL_TAG}_Mt025_N128" \
        "${CFL}" \
        "${SNAPSHOT_EVERY}"

done

echo
echo "============================================================"
echo "CFL STUDY WITH EQUAL PHYSICAL-TIME SAMPLING FINISHED"
echo "============================================================"
