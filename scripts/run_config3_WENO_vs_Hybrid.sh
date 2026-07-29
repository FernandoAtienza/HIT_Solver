#!/usr/bin/env bash
#
# Robust WENO-only vs Hybrid comparison launcher for the current HIT_Solver repo.
#
# Uses the existing repository entry point:
#   scripts/run_riemann_config3.py
#
# It inspects that script's --help output and automatically detects whether the
# method selector is called --scheme, --method, --solver, or --mode. It also
# passes only arguments that the installed entry point actually supports.
#

set -euo pipefail

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

REPO_ROOT="${HOME}/github/HIT_Solver"
ENTRY_POINT="${REPO_ROOT}/scripts/run_riemann_config3_08.py"
RESULTS_ROOT="${REPO_ROOT}/results/WENO_vs_Hybrid"

cd "${REPO_ROOT}"
mkdir -p "${RESULTS_ROOT}"

if [[ ! -f "${ENTRY_POINT}" ]]; then
    echo "ERROR: Missing entry point: ${ENTRY_POINT}" >&2
    exit 1
fi

HELP_TEXT="$(python3 -B "${ENTRY_POINT}" --help 2>&1)"

supports_flag() {
    local flag="$1"
    grep -q -- "${flag}" <<< "${HELP_TEXT}"
}

# Detect the argument used by the repository to select Hybrid or WENO-only.
METHOD_FLAG=""
for candidate in --scheme --method --solver --mode; do
    if supports_flag "${candidate}"; then
        METHOD_FLAG="${candidate}"
        break
    fi
done

if [[ -z "${METHOD_FLAG}" ]]; then
    echo "ERROR: ${ENTRY_POINT} does not expose a recognized method selector." >&2
    echo "Expected one of: --scheme, --method, --solver, --mode" >&2
    echo >&2
    echo "Its current --help output is:" >&2
    echo "${HELP_TEXT}" >&2
    exit 1
fi

echo "============================================================"
echo "WENO-only vs Hybrid Compact-WENO comparison"
echo "Repository   : ${REPO_ROOT}"
echo "Entry point  : ${ENTRY_POINT}"
echo "Method flag  : ${METHOD_FLAG}"
echo "Results      : ${RESULTS_ROOT}"
echo "============================================================"

# Common settings
TFINAL=0.85
CFL=0.25
BACKEND="cupy"
DEVICE=0
PROGRESS_EVERY=100
VORTICITY_LIMIT=100.0

# Hybrid-only settings
MN=0.004
SENSOR_WIDTH=10
JUMP_THRESHOLD=0.01
COMPRESSION_THRESHOLD=2.5
SHEAR_THRESHOLD=2.0
HYPERVISCOSITY_INTERVAL=1

append_if_supported() {
    local -n command_ref=$1
    local flag="$2"
    local value="${3-}"

    if supports_flag "${flag}"; then
        command_ref+=("${flag}")
        if [[ -n "${value}" ]]; then
            command_ref+=("${value}")
        fi
    fi
}

run_case() {
    local method="$1"
    local nx="$2"
    local name="$3"

    local case_dir="${RESULTS_ROOT}/${name}"
    local log_file="${case_dir}/run.log"
    mkdir -p "${case_dir}"

    local cmd=(
        python3 -B "${ENTRY_POINT}"
        "${METHOD_FLAG}" "${method}"
    )

    append_if_supported cmd --nx "${nx}"
    append_if_supported cmd --ny "${nx}"
    append_if_supported cmd --tfinal "${TFINAL}"
    append_if_supported cmd --cfl "${CFL}"
    append_if_supported cmd --backend "${BACKEND}"
    append_if_supported cmd --device "${DEVICE}"
    append_if_supported cmd --progress-every "${PROGRESS_EVERY}"
    append_if_supported cmd --vorticity-limit "${VORTICITY_LIMIT}"

    # Hybrid controls are added only to the hybrid cases and only when supported.
    if [[ "${method}" == "hybrid" ]]; then
        append_if_supported cmd --mn "${MN}"
        append_if_supported cmd --sensor-width "${SENSOR_WIDTH}"
        append_if_supported cmd --jump-threshold "${JUMP_THRESHOLD}"
        append_if_supported cmd --compression-threshold "${COMPRESSION_THRESHOLD}"
        append_if_supported cmd --shear-threshold "${SHEAR_THRESHOLD}"
        append_if_supported cmd --hyperviscosity-interval "${HYPERVISCOSITY_INTERVAL}"
    fi

    # Output handling: support the most common interfaces used in this repo.
    if supports_flag --output-dir; then
        cmd+=(--output-dir "${case_dir}")
    fi

    if supports_flag --prefix; then
        cmd+=(--prefix "${name}")
    fi

    if supports_flag --save; then
        cmd+=(--save "${case_dir}/${name}.png")
    fi

    if supports_flag --save-fields; then
        cmd+=(--save-fields)
    fi

    if supports_flag --no-show; then
        cmd+=(--no-show)
    fi

    echo
    echo "============================================================"
    echo "RUNNING: ${name}"
    echo "Method : ${method}"
    echo "Grid   : ${nx} x ${nx}"
    echo "Command:"
    printf ' %q' "${cmd[@]}"
    echo
    echo "============================================================"

    "${cmd[@]}" 2>&1 | tee "${log_file}"
}

# Run sequentially: only one GPU case at a time.
run_case hybrid 128 "Hybrid_N128"
run_case weno   128 "WENO_N128"

run_case hybrid 256 "Hybrid_N256"
run_case weno   256 "WENO_N256"

run_case hybrid 512 "Hybrid_N512_reference"

echo
echo "============================================================"
echo "WENO vs Hybrid study completed successfully."
echo "Results stored in: ${RESULTS_ROOT}"
echo "============================================================"
