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

# Kernels this run could not build. Collected rather than fatal; see compile_one.
FAILED=()

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

    # Not fatal on its own. `set -e` used to abort the whole run on the first kernel a given glslang could
    # not build, and every kernel listed after it was then silently left stale -- the script printed no error
    # naming them and exited as if it had done its job. Ubuntu's packaged glslang cannot build matmul_int8
    # (no GL_EXT_integer_dot_product), which sits partway down the list, so on an ordinary dev box the last
    # five kernels had not been rebuilt by this script in a long time. Failures are collected and reported at
    # the end instead, and the exit status still says the run was incomplete.
    if ! "$GLSLANG" --target-env "$TARGET" -S comp -V --quiet "${defs[@]}" -o "$dst" "$src"; then
        FAILED+=("${base}${suffix}")
        return
    fi

    if command -v "$SPIRVVAL" >/dev/null && ! "$SPIRVVAL" "$dst"; then
        FAILED+=("${base}${suffix} (spirv-val)")
        return
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
    layernorm_noaffine
    qkv_split_norm
    qkv_split_norm_head_major
    layernorm_modulate
    apply_rope_single
    unpatchify_tokens
    prelu
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
    fp8_absmax
    quant_e4m3
)

SINGLE_KERNELS=(
    chw_f32_to_hwc_u8
    affine_mix
    fill_bias
    quant_int8_rowwise
    pixel_shuffle2d
    modulation_split4
    affine_broadcast_row_indexed
    gated_residual_row_indexed
    cfg_euler
    wan_rms_norm_channel
    cast_f32_f16
    cast_f16_f32
    cast_f8e4m3_f16
    cast_bf16_f32
    cast_f32_bf16
    matmul_coopmat
    matmul_coopmat_partial_m
    matmul_coopmat2
    sdpa_flash_cm2
    matmul_int8
    matmul_fp8_coopmat
    dequant_q4_0
    dequant_q5_0
    dequant_q8_0
    dequant_q2_k
    dequant_q3_k
    dequant_q4_k
    dequant_q5_k
    dequant_q6_k
    dequant_iq4_xs
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

# sdpa_flash_cm2 with an F32 [Sq,Skv] additive mask; HAS_MASK adds a binding, so it is its own module.
compile_one "sdpa_flash_cm2" -DHAS_MASK=1 -- "_mask"

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

if [[ "${#FAILED[@]}" -gt 0 ]]; then
    echo >&2
    echo "error: ${#FAILED[@]} kernel(s) did not build with $GLSLANG:" >&2
    for f in "${FAILED[@]}"; do echo "         $f" >&2; done
    echo "       Every OTHER kernel above was rebuilt; these keep whatever is committed." >&2
    echo "       A glslang that lacks an extension a shader requires is the usual cause — the LunarG SDK" >&2
    echo "       builds all of them where a distribution package may not." >&2
    exit 1
fi

echo "Done. SPIR-V files in $(realpath "$OUT")"
