#!/usr/bin/env bash
# Compiles all dequant CUDA kernels to PTX and copies into the HartsyInference.Cuda Ptx folder.
# Requires nvcc (CUDA 13.x) on PATH. Target SM 7.5 — Turing (RTX 20xx) onward.
#
# Why sm_75 and not sm_70: CUDA 13.x dropped Volta, so `-arch=sm_70` is rejected outright ("invalid value
# for --gpu-architecture"). sm_75 keeps every consumer GPU from the RTX 20xx generation on and drops only
# V100. That is a deliberate minimum-GPU decision, not a toolchain workaround — the previously shipped
# dequant_*.ptx were `.target sm_70` CUDA 11.5 builds, so the first run of this script under CUDA 13.x
# raises the floor from Volta to Turing and rewrites them at `.version 9.0`. Validate against the CPU
# reference before committing the regenerated PTX (docs/Agents/KERNEL.md).
#
# Usage:  ./build.sh              # compile + install
#         ./build.sh --no-install # compile only

set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
PTX_OUT="${THIS_DIR}/../../Ptx"

KERNELS=(
    "dequant_q8_0_to_f16"
    "dequant_q4_0_to_f16"
    "dequant_q5_0_to_f16"
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
    echo "[$(date +%H:%M:%S)] nvcc -ptx -arch=sm_75 ${kernel}.cu"
    nvcc -ptx -arch=sm_75 "$src" -o "$ptx"
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
