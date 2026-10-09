#!/usr/bin/env bash
# The mixture-of-experts routing, dispatch and combine kernels. Kernel lists only — the build itself is ../build_common.sh.
set -euo pipefail
THIS_DIR="$(cd "$(dirname "$0")" && pwd)"

KERNELS=(
    "moe_route"
    "moe_dispatch"
)

# The expert FFN runner's kernels: compiled for sm_75 so they load on Turing and on every later card.
KERNELS_SM75=(
    "expert_f32"
)

# shellcheck source=../build_common.sh
. "${THIS_DIR}/../build_common.sh"
build_all "$@"
