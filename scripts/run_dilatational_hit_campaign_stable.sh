#!/usr/bin/env bash

set -euo pipefail
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd ~/github/HIT_Solver

BACKEND="${BACKEND:-cupy}"
NX="${NX:-512}"
NY="${NY:-${NX}}"
CFL="${CFL:-0.05}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-10000}"
DIAGNOSTICS_EVERY="${DIAGNOSTICS_EVERY:-1000}"
SEED="${SEED:-1234}"

run_and_postprocess() {
    local mt="$1"
    local tfinal="$2"
    local initial_power="$3"
    local min_power="$4"
    local max_power="$5"
    local case_name="$6"
    local output_dir="results/hit2d/${case_name}"
    local post_dir="${output_dir}/postprocess"

    mkdir -p "${output_dir}" "${post_dir}"

    echo
    echo "============================================================"
    echo "RUNNING: ${case_name}"
    echo "Forcing   : purely dilatational (curl-free)"
    echo "Mt target : ${mt}"
    echo "Power     : ${initial_power} [${min_power}, ${max_power}]"
    echo "grid      : ${NX} x ${NY}"
    echo "CFL       : ${CFL}"
    echo "output    : ${output_dir}"
    echo "============================================================"

    python3 -B scripts/run_hit2d.py \
        --backend "${BACKEND}" \
        --nx "${NX}" \
        --ny "${NY}" \
        --tfinal "${tfinal}" \
        --cfl "${CFL}" \
        --gamma 1.4 \
        --mach "${mt}" \
        --forcing-mode dilatational \
        --initial-kmin 3 \
        --initial-kmax 5 \
        --kf-min 3 \
        --kf-max 5 \
        --p-target "${initial_power}" \
        --forcing-correlation-time 0.5 \
        --forcing-alpha-memory 0.2 \
        --forcing-alpha-response-time 0.25 \
        --min-forcing-power 1.0e-6 \
        --max-forcing-rescale 20.0 \
        --mach-control \
        --mach-control-target "${mt}" \
        --mach-control-min-power "${min_power}" \
        --mach-control-max-power "${max_power}" \
        --mach-control-filter-time 2.0 \
        --mach-control-response-time 10.0 \
        --mach-control-deadband 0.03 \
        --mach-control-max-log-rate 0.25 \
        --viscosity 7.5e-4 \
        --mn 0.002 \
        --large-scale-drag 0.10 \
        --drag-kmax 2.0 \
        --cooling-time 2.5 \
        --cooling-target-pressure 0.7142857142857143 \
        --seed "${SEED}" \
        --diagnostics-every "${DIAGNOSTICS_EVERY}" \
        --snapshot-every "${SNAPSHOT_EVERY}" \
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
}

# Start with NX=128 for validation before launching the 512^2 production cases.
run_and_postprocess \
    0.25 110.0 \
    3.0e-3 3.0e-4 1.2e-2 \
    "dilatational_stable_Mt025_N${NX}_CFL_0p05"

run_and_postprocess \
    0.50 55.0 \
    1.2e-2 1.2e-3 4.8e-2 \
    "dilatational_stable_Mt050_N${NX}_CFL_0p05"

echo
echo "Both stabilized dilatational HIT cases completed."
