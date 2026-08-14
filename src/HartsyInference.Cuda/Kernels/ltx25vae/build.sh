#!/usr/bin/env bash
# Compiles the LTX-2.5 NA diffusion-decoder kernels to PTX and copies them into the HartsyInference.Cuda Ptx
# folder. Uses nvcc (CUDA 11+) when it is on PATH, else the committed ../nvrtc_compile frontend. These
# kernels include no CUDA headers, so CUDA_INC is only passed through for the nvrtc frontend's benefit.
# Target SM 8.0 (Ampere+); PTX is JIT-forward-compatible.
#
# Usage:  ./build.sh              # compile + install
#         ./build.sh --no-install # compile only

set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
PTX_OUT="${THIS_DIR}/../../Ptx"
NVRTC="${THIS_DIR}/../nvrtc_compile"
CUDA_LIB="${CUDA_LIB:-${HOME}/.local/lib/cuda13}"
if [[ -z "${CUDA_INC:-}" ]]; then
    for _cand in "${HOME}/.local/cuda-tools/nvidia/cu13/include" "${CUDA_LIB}/include"; do
        if [[ -d "$_cand" ]]; then CUDA_INC="$_cand"; break; fi
    done
    CUDA_INC="${CUDA_INC:-${CUDA_LIB}/include}"
fi

KERNELS=(
    "ltx25_na_decoder"
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
    if command -v nvcc >/dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] nvcc -ptx -arch=sm_80 ${kernel}.cu"
        nvcc -ptx -arch=sm_80 "$src" -o "$ptx"
    else
        if [[ ! -x "$NVRTC" ]]; then
            echo "no nvcc on PATH and no nvrtc helper — build it: cc -O2 -o $NVRTC ${NVRTC}.c -ldl" >&2
            exit 1
        fi
        echo "[$(date +%H:%M:%S)] nvrtc_compile compute_80 ${kernel}.cu"
        LD_LIBRARY_PATH="$CUDA_LIB" "$NVRTC" "$src" "$ptx" compute_80 "$CUDA_INC"
    fi
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
