#!/usr/bin/env bash
# SharpInference Vulkan / SPIR-V kernel build.
# Compiles every native/vulkan/shaders/*.comp.glsl to native/vulkan/build/*.spv
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
OUT="build"
SRC="shaders"

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
    printf "  build/%-40s  %5d bytes\n" "${base}${suffix}.spv" "$sz"
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
)

SINGLE_KERNELS=(
    cast_f32_f16
    cast_f16_f32
    cast_f8e4m3_f16
    matmul_coopmat
)

for k in "${DTYPE_KERNELS[@]}"; do
    compile_one "$k" -DUSE_FP16=0 -- "_f32"
    compile_one "$k" -DUSE_FP16=1 -- "_f16"
done

for k in "${SINGLE_KERNELS[@]}"; do
    compile_one "$k" -- ""
done

echo "Done. SPIR-V files in $(realpath "$OUT")"
