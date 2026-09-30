#!/usr/bin/env bash
# The single-latent sparse attention, indexer, hyper-connection, latent-quantization and position kernels of
# DeepSeek-V4.1-Flash. Kernel lists only — the build itself is ../build_common.sh. latent_codec.cuh is shared by
# latent_attention.cu and latent_quant.cu (the kernel directory is an nvrtc include path).
set -euo pipefail
THIS_DIR="$(cd "$(dirname "$0")" && pwd)"

KERNELS=(
    "latent_attention"
    "latent_quant"
    "hc_mix"
    "latent_positions"
)

# shellcheck source=../build_common.sh
. "${THIS_DIR}/../build_common.sh"
build_all "$@"
