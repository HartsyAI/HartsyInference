#!/usr/bin/env bash
# HartsyInference Vulkan / SPIR-V kernel build.
# Compiles every *.comp.glsl in this dir (src/HartsyInference.Vulkan/Shaders) to ../Spirv/*.spv
# using glslangValidator. Each output is also validated via spirv-val.
#
# Per [SPIRV_COMPUTE_SHADERS.md], we target Vulkan 1.3 with GL_KHR_shader_subgroup_*
# and explicit FP16 types. -O is safe (driver re-optimizes after spec consts anyway).
#
# Specialized variants are produced via -DUSE_FP16=1 / -DUSE_FP16=0 so a single
# .comp.glsl file emits both <name>_f32.spv and <name>_f16.spv.

set -euo pipefail
cd "$(dirname "$0")"

GLSLANG="${GLSLANG:-glslangValidator}"
SPIRVVAL="${SPIRVVAL:-spirv-val}"
TARGET="${TARGET:-vulkan1.3}"
OUT="${OUT:-../Spirv}"
SRC="."

if ! command -v "$GLSLANG" >/dev/null; then
    echo "error: glslangValidator not found in PATH (set GLSLANG env to override)" >&2
    echo "       on Debian/Ubuntu/Mint: sudo apt install glslang-tools spirv-tools" >&2
    echo "       or install the LunarG Vulkan SDK: https://vulkan.lunarg.com/" >&2
    exit 127
fi

mkdir -p "$OUT"

compile_one() {
    # Args: <basename> <define-flags...> -- <output-suffix>
    local base="$1"; shift
    local defs=()
    while [[ "$#" -gt 0 && "$1" != "--" ]]; do defs+=("$1"); shift; done
    shift   # consume --
    local suffix="$1"

    local src="$SRC/${base}.comp.glsl"
    local dst="$OUT/${base}${suffix}.spv"

    if [[ ! -f "$src" ]]; then
        echo "  skip $base${suffix}.spv  (missing $src)"
        return
    fi

    "$GLSLANG" --target-env "$TARGET" -S comp -V --quiet \
        "${defs[@]}" -o "$dst" "$src"

    if command -v "$SPIRVVAL" >/dev/null; then
        "$SPIRVVAL" "$dst"
    fi

    local sz
    sz=$(stat -c%s "$dst" 2>/dev/null || stat -f%z "$dst")
    printf "  %s/%-40s  %5d bytes\n" "$OUT" "${base}${suffix}.spv" "$sz"
}

# Kernels with FP32 + FP16 variants
DTYPE_KERNELS=(
    elementwise
    transpose
    permute_0213
    geglu
    broadcast_add
    groupnorm
    groupnorm_silu
    layernorm
    rmsnorm
    softmax
    im2col
    col2bias_add
    upsample_nearest2d
    upsample_bilinear2d
    matmul_tiled
    mask_add
    maxpool2d
    depthwise_conv2d
    conv1d
    conv_transpose1d
    snake
    slice_last_dim
    apply_rope
    kv_cache_append
    sdpa_flash
    affine_broadcast_last_dim
    wan_rope_interleaved
    repeat_kv_heads
    gated_residual_last_dim
    slice_rows
)

SINGLE_KERNELS=(
    cfg_euler
    wan_rms_norm_channel
    cast_f32_f16
    cast_f16_f32
    cast_f8e4m3_f16
    cast_bf16_f32
    cast_f32_bf16
    matmul_coopmat
    matmul_coopmat_partial_m
    matmul_int8
    dequant_q4_0
    dequant_q5_0
    dequant_q8_0
    dequant_q4_k
    dequant_q5_k
    dequant_q6_k
    embed_gather_decode
    argmax_lastdim
    history_append
    repetition_penalty
    kv_cache_append_dev
)

for k in "${DTYPE_KERNELS[@]}"; do
    compile_one "$k" -DUSE_FP16=0 -- "_f32"
    compile_one "$k" -DUSE_FP16=1 -- "_f16"
done

for k in "${SINGLE_KERNELS[@]}"; do
    compile_one "$k" -- ""
done

# snake-beta (BigVGAN-v2): USE_BETA gates a #if-compiled binding, not a spec constant, so it
# needs its own SPIR-V module distinct from the vanilla-snake build above.
compile_one "snake" -DUSE_FP16=0 -DUSE_BETA=1 -- "_beta_f32"
compile_one "snake" -DUSE_FP16=1 -DUSE_BETA=1 -- "_beta_f16"

# sdpa_flash with an optional additive mask: HAS_MASK gates a #if-compiled binding (like USE_BETA
# above), so the masked variant needs its own SPIR-V module too.
compile_one "sdpa_flash" -DUSE_FP16=0 -DHAS_MASK=1 -- "_mask_f32"
compile_one "sdpa_flash" -DUSE_FP16=1 -DHAS_MASK=1 -- "_mask_f16"

# affine_broadcast_last_dim: HAS_SHIFT=0 is Ideogram 4's scale-only adaLN (shift is null) — a distinct
# #if-compiled binding layout (like USE_BETA/HAS_MASK above), needing its own SPIR-V module.
compile_one "affine_broadcast_last_dim" -DUSE_FP16=0 -DHAS_SHIFT=0 -- "_noshift_f32"
compile_one "affine_broadcast_last_dim" -DUSE_FP16=1 -DHAS_SHIFT=0 -- "_noshift_f16"

# rope_decode_step: INTERLEAVED selects a #if-compiled code path (like USE_BETA/HAS_MASK above), so
# the two pairing conventions need their own SPIR-V modules. F32-only (decode-graph state is F32).
compile_one "rope_decode_step" -DINTERLEAVED=0 -- "_splithalf_f32"
compile_one "rope_decode_step" -DINTERLEAVED=1 -- "_interleaved_f32"

# sdpa_flash device-position variant (FlashAttentionDev): HAS_DEVICE_POS reads skv/qOffset from a device
# buffer instead of push constants. Mutually exclusive with HAS_MASK (FlashAttentionDev has no mask param).
# F32-only (decode-graph state is F32).
compile_one "sdpa_flash" -DUSE_FP16=0 -DHAS_DEVICE_POS=1 -- "_dev_f32"

echo "Done. SPIR-V files in $(realpath "$OUT")"
