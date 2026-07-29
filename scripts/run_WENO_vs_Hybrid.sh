#!/usr/bin/env bash
#
# ============================================================
#  WENO vs Hybrid comparison study
#
#  Runs Configuration 3 at several resolutions using
#  both the Hybrid Compact-WENO solver and the WENO-only solver.
#
#  The highest-resolution Hybrid case is used as the reference.
# ============================================================

set -e

echo "==========================================================="
echo "  WENO vs Hybrid comparison"
echo "==========================================================="

ROOT="results/WENO_vs_Hybrid"

mkdir -p "$ROOT"

# ------------------------------------------------------------
# Common parameters
# ------------------------------------------------------------

TFINAL=0.85
CFL=0.25

X0=0.8
Y0=0.8

MN=0.004

SENSOR_WIDTH=10
JUMP_THRESHOLD=0.01
SHEAR_THRESHOLD=2.0

BACKEND=cupy

# ------------------------------------------------------------
# Helper function
# ------------------------------------------------------------

run_case () {

    SCRIPT=$1
    NX=$2
    NAME=$3

    echo
    echo "===================================================="
    echo "Running $NAME"
    echo "===================================================="

    python3 -B "$SCRIPT" \
        --nx "$NX" \
        --ny "$NX" \
        --tfinal "$TFINAL" \
        --cfl "$CFL" \
        --x0 "$X0" \
        --y0 "$Y0" \
        --mn "$MN" \
        --sensor-width "$SENSOR_WIDTH" \
        --jump-threshold "$JUMP_THRESHOLD" \
        --shear-threshold "$SHEAR_THRESHOLD" \
        --backend "$BACKEND" \
        --no-show \
        --save "$ROOT/${NAME}.png" \
        --save-data "$ROOT/${NAME}.npz"

}

# ============================================================
# 128²
# ============================================================

run_case \
scripts/riemann2d/configuration3_hybrid.py \
128 \
Hybrid_N128

run_case \
scripts/riemann2d/configuration3_weno.py \
128 \
WENO_N128

# ============================================================
# 256²
# ============================================================

run_case \
scripts/riemann2d/configuration3_hybrid.py \
256 \
Hybrid_N256

run_case \
scripts/riemann2d/configuration3_weno.py \
256 \
WENO_N256

# ============================================================
# Reference
# ============================================================

run_case \
scripts/riemann2d/configuration3_hybrid.py \
512 \
Hybrid_N512_reference

echo
echo "==========================================================="
echo "Study completed."
echo "Results stored in:"
echo "$ROOT"
echo "==========================================================="
