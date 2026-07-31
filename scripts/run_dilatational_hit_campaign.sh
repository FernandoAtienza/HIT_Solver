#!/usr/bin/env bash

set -euo pipefail
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd ~/github/HIT_Solver

BACKEND="cupy"
NX=512
NY=512
CFL=0.05
SNAPSHOT_EVERY=10000
DIAGNOSTICS_EVERY=1000
SEED=1234

run_and_postprocess() {
    local mt="$1"
    local tfinal="$2"
    local case_name="$3"
    local output_dir="results/hit2d/${case_name}"
    local post_dir="${output_dir}/postprocess"

    mkdir -p "${output_dir}" "${post_dir}"

    echo
    echo "============================================================"
    echo "RUNNING: ${case_name}"
    echo "Forcing   : purely dilatational (curl-free)"
    echo "Mt target : ${mt}"
    echo "tfinal    : ${tfinal}"
    echo "grid      : ${NX} x ${NY}"
    echo "CFL       : ${CFL}"
    echo "output    : ${output_dir}"
    echo "============================================================"
    echo

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
        --snapshot-every "${SNAPSHOT_EVERY}" \
        --output-dir "${output_dir}" \
        2>&1 | tee "${output_dir}/run.log"

    echo
    echo "Simulation completed. Generating field and history plots..."
    echo

    python3 -B 2D/hit2d_viewer.py \
        --snapshot-dir "${output_dir}" \
        --physics-plots \
        --history-x-axis turnover \
        --physics-output "${post_dir}/physics_fields_final.png" \
        --history-output "${post_dir}/time_history.png" \
        2>&1 | tee "${output_dir}/viewer_postprocess.log"

    echo
    echo "Generating isotropy and spectral diagnostics..."
    echo

    python3 -B 2D/hit2d_isotropy_diagnostics.py \
        --snapshot-dir "${output_dir}" \
        --start-turnover 4 \
        --end-turnover 16 \
        --fluctuation-type reynolds \
        2>&1 | tee "${output_dir}/isotropy_postprocess.log"

    echo
    echo "============================================================"
    echo "COMPLETED AND POST-PROCESSED: ${case_name}"
    echo "Results: ${output_dir}"
    echo "============================================================"
}

# The initial velocity field remains the same divergence-free field used in the
# solenoidal campaign. Only the forcing projection changes. Statistics should
# therefore continue to exclude the initial transient (4 <= N_eddy <= 16).

# Mt = 0.25: approximately 17.5 turnover times
run_and_postprocess 0.25 110.0 "dilatational_Mt025_N512_CFL_0p05"

# Mt = 0.50: approximately 17.5 turnover times
# This starts only after the Mt=0.25 case finishes successfully.
run_and_postprocess 0.50 55.0 "dilatational_Mt050_N512_CFL_0p05"

echo
echo "============================================================"
echo "BOTH DILATATIONAL HIT SIMULATIONS AND POST-PROCESSING COMPLETED"
echo "============================================================"
