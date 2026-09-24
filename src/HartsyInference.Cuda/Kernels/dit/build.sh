#!/usr/bin/env bash
# The DiT kernels. Kernel lists only — the build itself is ../build_common.sh.
set -euo pipefail
THIS_DIR="$(cd "$(dirname "$0")" && pwd)"

KERNELS=(
    "dit_f32"
    "dit_f16"
    "dit_bf16"
    "stepcache"
    "dit_rope"
    "dit_fp8emit"
    "mg3_action"
)

# shellcheck source=../build_common.sh
. "${THIS_DIR}/../build_common.sh"
build_all "$@"
