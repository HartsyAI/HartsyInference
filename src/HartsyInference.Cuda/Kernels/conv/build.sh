#!/usr/bin/env bash
# The convolution helper kernels (cuda_fp16.h/cuda_bf16.h). Kernel lists only — the build itself is ../build_common.sh.
set -euo pipefail
THIS_DIR="$(cd "$(dirname "$0")" && pwd)"

KERNELS=(
    "depthwise_conv2d"
    "im2col_banded"
    "maxpool2d"
)

# shellcheck source=../build_common.sh
. "${THIS_DIR}/../build_common.sh"
build_all "$@"
