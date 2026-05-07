#!/usr/bin/env bash
# Compiles all dequant CUDA kernels to PTX and copies into the SharpInference.Cuda Ptx folder.
# Requires nvcc (CUDA 11+) on PATH. Target SM 7.0 — compatible with every GPU since Volta (RTX 20xx onward).
#
# Usage:  ./build.sh              # compile + install
#         ./build.sh --no-install # compile only

set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
PTX_OUT="${THIS_DIR}/../../../src/SharpInference.Cuda/Ptx"

KERNELS=(
    "dequant_q8_0_to_f16"
    "dequant_q4_k_to_f16"
    "dequant_q5_k_to_f16"
    "dequant_q6_k_to_f16"
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
    echo "[$(date +%H:%M:%S)] nvcc -ptx -arch=sm_70 ${kernel}.cu"
    nvcc -ptx -arch=sm_70 "$src" -o "$ptx"
    if $INSTALL; then
        cp "$ptx" "${PTX_OUT}/${kernel}.ptx"
        echo "  → ${PTX_OUT}/${kernel}.ptx"
    fi
done

echo "done."
