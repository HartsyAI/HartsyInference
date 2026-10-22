#!/usr/bin/env bash
# The custom attention kernels (VSA, SageAttention INT8). Kernel lists only — the build itself is ../build_common.sh.
set -euo pipefail
THIS_DIR="$(cd "$(dirname "$0")" && pwd)"

KERNELS=(
    "flash_attn_f16"
    "h3_vsa"
    "sage_attn_int8"
)

# shellcheck source=../build_common.sh
. "${THIS_DIR}/../build_common.sh"
build_all "$@"
