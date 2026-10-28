#!/usr/bin/env bash
# The custom attention kernels (VSA, SageAttention INT8). Kernel lists only — the build itself is ../build_common.sh.
set -euo pipefail
THIS_DIR="$(cd "$(dirname "$0")" && pwd)"

KERNELS=(
    "flash_attn_f16"
    "h3_vsa"
    "sage_attn_int8"
    "int8_attn_v2"
)
# CUDA 13.0.88 nvcc output: the ptxas register and spill check needs the toolchain the PTX came from.
NVCC_BUILT=("int8_attn_v2")
# --arch sm_89 adds int8_attn_v2.sm89.ptx, which the 4090 loads instead of the baseline.
ARCH_VARIANTS=("int8_attn_v2")
PTXAS_SPILL_CHECK=("int8_attn_v2")
# cp.async and m16n8k16/k32 MMA: no Turing build. The Turing card takes the dense and tiled attention rows instead.
SM80_ONLY=("flash_attn_f16" "sage_attn_int8" "int8_attn_v2" "h3_vsa")

# shellcheck source=../build_common.sh
. "${THIS_DIR}/../build_common.sh"
build_all "$@"
