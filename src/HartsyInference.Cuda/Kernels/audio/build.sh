#!/usr/bin/env bash
# Compiles the audio codec/TTS kernels to PTX and copies them into the HartsyInference.Cuda Ptx folder.
# Uses nvcc (CUDA 11+) when it is on PATH, else the committed ../nvrtc_compile frontend,
# which needs the CUDA headers ($CUDA_INC). Target SM 8.0 (Ampere+); PTX is JIT-forward-compatible.
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

KERNELS=(
    "conv1d_f32"
    "audio_activations_f32"
    "adain1d_f32"
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
