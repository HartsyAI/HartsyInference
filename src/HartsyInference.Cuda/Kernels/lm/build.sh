#!/usr/bin/env bash
# The LLM decode kernels. Kernel lists only — the build itself is ../build_common.sh.
set -euo pipefail
THIS_DIR="$(cd "$(dirname "$0")" && pwd)"

KERNELS=(
    "lm_f32"
    "lm_history_f32"
    "lm_embed_gather"
    "lm_cfg_topk_guided"
    "flash_attn_f32"
    "flash_attn_f32_split"
    "flash_attn_v2_tf32"
    "flash_attn_causal_f16"
    "flash_attn_decode_gqa"
    "kv_scatter_f16"
    "lm_sample_topk"
    "lm_norm_q8_fast"
    "lm_decode_fused"
    "mul_mat_vec_q4k_f32"
    "mul_mat_vec_q6k_f32"
    "mul_mat_vec_q8_0_f32"
    "mul_mat_vec_q5_0_f32"
    "mul_mat_vec_q4_0_f32"
    "mul_mat_vec_q5k_f32"
    "mul_mat_vec_q2k_f32"
    "mul_mat_vec_q3k_f32"
    "mul_mat_vec_f16_bf16_f32"
    "lm_attn_mask"
    "lm_topk_f32"
    "quantize_activation_q8_1_f32"
    "mul_mat_vec_q4k_q8_1"
    "mul_mat_vec_q8_0_q8_1"
    "mul_mat_vec_q6k_q8_1"
    "mul_mat_vec_q4_0_q8_1"
    "mul_mat_vec_q5_0_q8_1"
    "mul_mat_vec_q5k_q8_1"
    "mul_mat_vec_q2k_q8_1"
    "mul_mat_vec_q3k_q8_1"
    "ssm_prefill_f32"
)
# Shipped as CUDA 13.0.88 nvcc output, which the decode timings were taken on; nvrtc emits different PTX for these.
NVCC_BUILT=(
    "lm_f32"
    "flash_attn_causal_f16"
    "flash_attn_decode_gqa"
    "lm_norm_q8_fast"
    "lm_decode_fused"
    "lm_topk_f32"
    "mul_mat_vec_q4k_q8_1"
    "mul_mat_vec_q6k_q8_1"
    "mul_mat_vec_q5_0_q8_1"
    "mul_mat_vec_q2k_q8_1"
    "mul_mat_vec_q3k_q8_1"
)

# shellcheck source=../build_common.sh
. "${THIS_DIR}/../build_common.sh"
build_all "$@"
