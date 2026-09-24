#!/usr/bin/env bash
# The Wan video kernels. Kernel lists only — the build itself is ../build_common.sh.
set -euo pipefail
THIS_DIR="$(cd "$(dirname "$0")" && pwd)"

KERNELS=(
    "wan_rope"
    "wan_vae_conv3d"
    "wan_vae_frames"
    "wan_vae_norm"
)

# shellcheck source=../build_common.sh
. "${THIS_DIR}/../build_common.sh"
build_all "$@"
