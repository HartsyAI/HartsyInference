#!/usr/bin/env bash
# Compiles all dequant/quant CUDA kernels to PTX and copies into the HartsyInference.Cuda Ptx folder.
# Uses nvcc (CUDA 13.x) when it is on PATH, else the committed ../nvrtc_compile frontend.
# GGUF dequant + W8A8 target SM 7.5 — Turing (RTX 20xx) onward; fp8_quant targets SM 8.0 (Ampere+),
# matching its shipped artifact.
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
NVRTC="${THIS_DIR}/../nvrtc_compile"
CUDA_LIB="${CUDA_LIB:-${HOME}/.local/lib/cuda13}"
# The headers must be a COMPLETE set: mma.h (attention) pulls crt/mma.h, which the lib-adjacent
# include dir lacks — that is why attention/audio/lm silently could not rebuild here. Prefer the first
# candidate that actually has it, so a partial set degrades to a clear error rather than a stale PTX.
if [[ -z "${CUDA_INC:-}" ]]; then
    for _cand in "${HOME}/.local/cuda-tools/nvidia/cu13/include" "${CUDA_LIB}/include"; do
        if [[ -f "${_cand}/crt/mma.h" ]]; then CUDA_INC="$_cand"; break; fi
    done
    CUDA_INC="${CUDA_INC:-${CUDA_LIB}/include}"
fi

KERNELS_SM75=(
    "dequant_q8_0_to_f16"
    "dequant_q4_0_to_f16"
    "dequant_q5_0_to_f16"
    "dequant_q4_k_to_f16"
    "dequant_q5_k_to_f16"
    "dequant_q6_k_to_f16"
    "dequant_nvfp4_to_f16"
    "w8a8"
    "convrot"
)
KERNELS_SM80=(
    "fp8_quant"
    "int8_mma_gemm"   # cp.async + mma.m16n8k32.s8 are Ampere+
)

INSTALL=true
if [[ "${1:-}" == "--no-install" ]]; then
    INSTALL=false
fi

compile_one() {
    local kernel="$1" sm="$2"
    local src="${THIS_DIR}/${kernel}.cu"
    local ptx="${THIS_DIR}/${kernel}.ptx"
    if [[ ! -f "$src" ]]; then
        echo "missing source: $src" >&2
        exit 1
    fi
    if command -v nvcc >/dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] nvcc -ptx -arch=sm_${sm} ${kernel}.cu"
        nvcc -ptx -arch="sm_${sm}" "$src" -o "$ptx"
    else
        if [[ ! -x "$NVRTC" ]]; then
            echo "no nvcc on PATH and no nvrtc helper — build it: cc -O2 -o $NVRTC ${NVRTC}.c -ldl" >&2
            exit 1
        fi
        echo "[$(date +%H:%M:%S)] nvrtc_compile compute_${sm} ${kernel}.cu"
        LD_LIBRARY_PATH="$CUDA_LIB" "$NVRTC" "$src" "$ptx" "compute_${sm}" "$CUDA_INC"
    fi
    if ! head -20 "$ptx" | grep -q '^\.version 9\.0$'; then
        echo "ERROR: ${kernel}.ptx is not PTX ISA 9.0 (driver JIT ceiling) — check toolchain pin." >&2
        exit 1
    fi
    if $INSTALL; then
        cp "$ptx" "${PTX_OUT}/${kernel}.ptx"
        echo "  → ${PTX_OUT}/${kernel}.ptx"
    fi
}

for kernel in "${KERNELS_SM75[@]}"; do compile_one "$kernel" 75; done
for kernel in "${KERNELS_SM80[@]}"; do compile_one "$kernel" 80; done

echo "done."
