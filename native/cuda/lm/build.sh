#!/usr/bin/env bash
# Compiles the language-model (decoder LLM) glue kernels to PTX and copies them into the
# HartsyInference.Cuda Ptx folder. Requires nvcc (CUDA 11+) on PATH. Target SM 8.0 (Ampere+);
# PTX is JIT-forward-compatible.
#
# Usage:  ./build.sh              # compile + install
#         ./build.sh --no-install # compile only

set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
PTX_OUT="${THIS_DIR}/../../../src/HartsyInference.Cuda/Ptx"

KERNELS=(
    "lm_f32"
    "flash_attn_f32"
    "flash_attn_f32_split"
    "flash_attn_v2_tf32"
    "mul_mat_vec_q4k_f32"
    "mul_mat_vec_q6k_f32"
    "mul_mat_vec_q8_0_f32"
    "mul_mat_vec_q5_0_f32"
    "mul_mat_vec_q4_0_f32"
    "mul_mat_vec_q5k_f32"
    "quantize_activation_q8_1_f32"
    "mul_mat_vec_q4k_q8_1"
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
    if $INSTALL; then
        cp "$ptx" "${PTX_OUT}/${kernel}.ptx"
        echo "  → ${PTX_OUT}/${kernel}.ptx"
    fi
done

echo "done."
