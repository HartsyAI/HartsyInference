#!/usr/bin/env bash
# The custom attention kernels (VSA, SageAttention INT8). Kernel lists only — the build itself is ../build_common.sh.
set -euo pipefail
THIS_DIR="$(cd "$(dirname "$0")" && pwd)"

KERNELS=(
    "h3_vsa"
    "sage_attn_int8"
    "sage_attn_int8_v1"
)
# h3_vsa ships nvcc-built PTX that nvrtc does not reproduce; a routine rebuild must not overwrite it (--install-tuned does).
TUNED=("h3_vsa")

# shellcheck source=../build_common.sh
. "${THIS_DIR}/../build_common.sh"
build_all "$@"
