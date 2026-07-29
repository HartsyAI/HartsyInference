#!/usr/bin/env bash
# Compiles the custom attention kernels to PTX and copies them into the HartsyInference.Cuda Ptx folder.
# Requires nvcc (CUDA 11+) on PATH. Target SM 8.0 (Ampere+); PTX is JIT-forward-compatible.
# NOTE: the emitted PTX must say ".version 9.0" — newer nvcc emits 9.3 which the fleet driver refuses to
# JIT (see INFERENCE_ACCEL_GRIND §H1.1: pin nvidia-nvvm alongside nvidia-cuda-nvcc).
#
# Usage:  ./build.sh              # compile + install
#         ./build.sh --no-install # compile only

set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
PTX_OUT="${THIS_DIR}/../../Ptx"

KERNELS=(
    "sage_attn_int8"
    "sage_attn_int8_v1"
)

INSTALL=true
if [[ "${1:-}" == "--no-install" ]]; then
    INSTALL=false
fi

for kernel in "${KERNELS[@]}"; do
    src="${THIS_DIR}/${kernel}.cu"
    ptx="${THIS_DIR}/${kernel}.ptx"
    if [[ ! -f "$src" ]]; then
        echo "missing source: $src" >&2
        exit 1
    fi
    echo "[$(date +%H:%M:%S)] nvcc -ptx -arch=sm_80 ${kernel}.cu"
    nvcc -ptx -arch=sm_80 "$src" -o "$ptx"
    if ! head -20 "$ptx" | grep -q '^\.version 9\.0$'; then
        echo "ERROR: ${kernel}.ptx is not PTX ISA 9.0 (driver JIT ceiling) — check toolchain pin." >&2
        exit 1
    fi
    if $INSTALL; then
        cp "$ptx" "${PTX_OUT}/${kernel}.ptx"
        echo "  → ${PTX_OUT}/${kernel}.ptx"
    fi
done

echo "done."
