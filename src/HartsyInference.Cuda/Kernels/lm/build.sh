#!/usr/bin/env bash
# The LLM decode kernels. Kernel lists only — the build itself is ../build_common.sh.
set -euo pipefail
THIS_DIR="$(cd "$(dirname "$0")" && pwd)"

KERNELS=(
    "lm_f32"
    "flash_attn_f32"
    "flash_attn_f32_split"
    "flash_attn_v2_tf32"
    "mul_mat_vec_q4k_f32"
    "mul_mat_vec_q6k_f32"
    "mul_mat_vec_q8_0_f32"
    "mul_mat_vec_q5_0_f32"
    "mul_mat_vec_q4_0_f32"
    "mul_mat_vec_q5k_f32"
    "mul_mat_vec_f16_bf16_f32"
    "lm_attn_mask"
    "quantize_activation_q8_1_f32"
    "mul_mat_vec_q4k_q8_1"
    "mul_mat_vec_q8_0_q8_1"
    "mul_mat_vec_q6k_q8_1"
    "mul_mat_vec_q4_0_q8_1"
    "mul_mat_vec_q5_0_q8_1"
    "mul_mat_vec_q5k_q8_1"
)
# LLM-decode hot paths tuned against llama.cpp: their sources are current and only register allocation differs
# between toolchains, so a routine rebuild must not overwrite the shipped artifacts (see --install-tuned).
TUNED=("lm_f32" "mul_mat_vec_q6k_q8_1")

# shellcheck source=../build_common.sh
. "${THIS_DIR}/../build_common.sh"
build_all "$@"
