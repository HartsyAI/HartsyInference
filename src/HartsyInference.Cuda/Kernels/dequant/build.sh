#!/usr/bin/env bash
# GGUF dequant, W8A8, ConvRot, the fp8 and block-scaled activation quantizers, the int8 mma GEMM. GGUF dequant and W8A8 keep SM 7.5 (Turing): CUDA 13 dropped Volta, so sm_70 is rejected outright and the floor moved from V100 to the RTX 20xx generation. Kernel lists only — the build itself is ../build_common.sh.
set -euo pipefail
THIS_DIR="$(cd "$(dirname "$0")" && pwd)"

KERNELS_SM75=(
    "dequant_q8_0_to_f16"
    "dequant_q4_0_to_f16"
    "dequant_q5_0_to_f16"
    "dequant_q2_k_to_f16"
    "dequant_q3_k_to_f16"
    "dequant_q4_k_to_f16"
    "dequant_q5_k_to_f16"
    "dequant_q6_k_to_f16"
    "dequant_nvfp4_to_f16"
    "w8a8"
    "convrot"
)
KERNELS_SM80=(
    "fp8_quant"
    "block_quant"
    "int8_mma_gemm"   # cp.async + mma.m16n8k32.s8 are Ampere+
)
# Carries arch-conditional device code (the e2m1 cvt on sm_100a/sm_120a); --arch builds its suffixed variant.
ARCH_VARIANTS=("block_quant")

# shellcheck source=../build_common.sh
. "${THIS_DIR}/../build_common.sh"
build_all "$@"
