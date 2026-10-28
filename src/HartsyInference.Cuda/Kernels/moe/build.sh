#!/usr/bin/env bash
# The mixture-of-experts routing, dispatch and combine kernels. Kernel lists only — the build itself is ../build_common.sh.
set -euo pipefail
THIS_DIR="$(cd "$(dirname "$0")" && pwd)"

KERNELS=(
    "moe_route"
    "moe_dispatch"
    "moe_id_q4k"
    "moe_id_q6k"
    "moe_id_q8_0"
    "moe_id_q4_0"
    "moe_combine_slots"
    "moe_prefill"
)
# Shipped as CUDA 13.0.88 nvcc output, which the expert GEMV timings were taken on; nvrtc emits different PTX for these.
NVCC_BUILT=("moe_id_q4k" "moe_id_q6k" "moe_id_q8_0")
# Compiles for sm_75, but its grouped-prefill path has not run on Turing, so it stays on the sm_80 build until a Turing run
# verifies it. Until then the host-routed MoE path serves Turing.
SM80_ONLY=("moe_prefill")

# The expert FFN runner's kernels: compiled for sm_75 so they load on Turing and on every later card.
KERNELS_SM75=(
    "expert_f32"
)

# shellcheck source=../build_common.sh
. "${THIS_DIR}/../build_common.sh"
build_all "$@"
