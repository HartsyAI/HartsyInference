#!/usr/bin/env bash
# Compiles the audio codec/TTS kernels to PTX and copies them into the HartsyInference.Cuda Ptx folder.
# Requires nvcc (CUDA 11+) on PATH. Target SM 8.0 (Ampere+); PTX is JIT-forward-compatible.
#
# Usage:  ./build.sh              # compile + install
#         ./build.sh --no-install # compile only

set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
PTX_OUT="${THIS_DIR}/../../../src/HartsyInference.Cuda/Ptx"

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
    echo "[$(date +%H:%M:%S)] nvcc -ptx -arch=sm_80 ${kernel}.cu"
    nvcc -ptx -arch=sm_80 "$src" -o "$ptx"
    if $INSTALL; then
        cp "$ptx" "${PTX_OUT}/${kernel}.ptx"
        echo "  → ${PTX_OUT}/${kernel}.ptx"
    fi
done

echo "done."
