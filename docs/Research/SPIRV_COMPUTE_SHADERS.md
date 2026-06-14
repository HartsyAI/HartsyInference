# SPIR-V Compute Shaders — Research Notes

SPIR-V is the binary intermediate language consumed by Vulkan, OpenCL 2.1+, and OpenGL 4.6. HartsyInference authors compute kernels in **GLSL 4.50** with the `GL_KHR_*` subgroup extensions, compiles them at build time to SPIR-V via `glslangValidator`, ships the `.spv` files as content, and loads them at runtime via `vkCreateShaderModule`. The driver re-optimizes after applying our specialization constants and JITs to vendor ISA (NVIDIA SASS, AMD GCN/RDNA, Intel Gen ISA).

This document is the kernel-design counterpart to [VULKAN_COMPUTE_API.md](VULKAN_COMPUTE_API.md). It mirrors the role of [CUDA_AND_PTX.md](CUDA_AND_PTX.md) for the CUDA backend: shader skeletons, tiling strategies, subgroup-reduction patterns, validation tolerances, and one detailed design per kernel family the engine needs.

The single largest engineering challenge: **Vulkan has no cuBLAS.** Every GEMM call in the CUDA backend (`cublasGemmEx` for Linear / Conv2D-1×1 / attention QK^T) becomes a hand-written tiled compute shader. We define a single template shader (`matmul_tiled.comp`) parameterized by spec constants that covers FP32 and FP16, with subgroup-tiled reductions and shared-memory blocking — see [§ Tiled GEMM (cuBLAS Replacement)](#tiled-gemm-cublas-replacement).

Sources: [SPIR-V 1.6 Specification](https://registry.khronos.org/SPIR-V/specs/1.6/SPIRV.html), [GLSL 4.60 Specification](https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.pdf), [GL_KHR_shader_subgroup](https://github.com/KhronosGroup/GLSL/blob/main/extensions/khr/GL_KHR_shader_subgroup.txt), [Vulkan GLSL extensions](https://github.com/KhronosGroup/GLSL/tree/main/extensions/khr), [glslang](https://github.com/KhronosGroup/glslang), [SPIRV-Tools (`spirv-opt`/`spirv-val`/`spirv-dis`)](https://github.com/KhronosGroup/SPIRV-Tools), [llama.cpp Vulkan backend](https://github.com/ggerganov/llama.cpp/tree/master/ggml/src/ggml-vulkan), [VkFFT source](https://github.com/DTolm/VkFFT), [Khronos Vulkan Guide — Compute](https://docs.vulkan.org/guide/latest/computeshader.html), [AMD GPU Performance Guide](https://gpuopen.com/learn/concurrent-execution-asynchronous-queues/), [NVIDIA Vulkan Tips](https://developer.nvidia.com/blog/vulkan-tips/).

---

## Toolchain & Build Flow

| Tool | Use | Source |
|---|---|---|
| `glslangValidator` (Khronos) | GLSL → SPIR-V; primary compiler | [glslang](https://github.com/KhronosGroup/glslang) |
| `glslc` (Google, in shaderc) | GLSL → SPIR-V; alternative | [shaderc](https://github.com/google/shaderc) |
| `spirv-opt` | Optimization passes (mostly redundant when relying on driver JIT) | [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools) |
| `spirv-val` | SPIR-V validator | same |
| `spirv-dis` / `spirv-cross` | Disassemble / convert SPIR-V → GLSL/HLSL/MSL | [SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross) |

Install on Linux: `apt install glslang-tools spirv-tools` (or download Vulkan SDK from LunarG). On the dev container we already use, the SDK is the most reliable source for matched versions.

### Compile invocation

```
glslangValidator \
    --target-env vulkan1.3 \
    -S comp \
    -V \
    -O \
    --quiet \
    -o native/vulkan/build/groupnorm_silu.spv \
    native/vulkan/shaders/groupnorm_silu.comp.glsl
```

Flags:
- `--target-env vulkan1.3` — required for sync2, full subgroup features, FP16 storage
- `-S comp` — stage = compute
- `-V` — generate SPIR-V (default in this mode but explicit is safer)
- `-O` — optimize (uses `spirv-opt -O` internally; safe; driver will optimize again anyway)
- `-Os` — optimize for size if .spv binary size matters
- `-g` — keep debug info (use for `spirv-dis` debugging only; remove for ship)
- `-DNAME=value` — preprocessor define (we use this for backend variants like `-DUSE_FP16=1`)

### Build script (`native/vulkan/build.sh`)

```bash
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p build
for f in shaders/*.comp.glsl; do
    name=$(basename "$f" .comp.glsl)
    glslangValidator --target-env vulkan1.3 -S comp -V -O \
        --quiet -o "build/${name}.spv" "$f"
    spirv-val "build/${name}.spv"
    echo "  build/${name}.spv  ($(stat -c%s build/${name}.spv) bytes)"
done
```

Mirror the CUDA pattern in `native/cuda/build.sh`. The MSBuild target in `HartsyInference.Vulkan.csproj` invokes this script during build and copies `.spv` files into the package's `Spirv/` content directory.

### MSBuild integration

```xml
<Target Name="BuildSpirv" BeforeTargets="BeforeBuild" Condition="'$(SkipNativeBuild)' != 'true'">
    <Exec Command="bash $(MSBuildProjectDirectory)/../../native/vulkan/build.sh"
          WorkingDirectory="$(MSBuildProjectDirectory)/../../native/vulkan/" />
</Target>
<ItemGroup>
    <Content Include="..\..\native\vulkan\build\*.spv" Link="Spirv\%(Filename)%(Extension)">
        <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </Content>
</ItemGroup>
```

Identical pattern to the CUDA Ptx/ content inclusion in `HartsyInference.Cuda.csproj`.

---

## GLSL Compute Shader Anatomy

```glsl
#version 460
#extension GL_KHR_shader_subgroup_basic     : require
#extension GL_KHR_shader_subgroup_arithmetic: require
#extension GL_KHR_shader_subgroup_shuffle   : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require   // FP16
#extension GL_EXT_shader_explicit_arithmetic_types_int8    : require   // FP8 storage
#extension GL_EXT_buffer_reference_uvec2    : enable                   // optional pointer-style

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
//   ^^^^ workgroup size driven by specialization constants

layout(constant_id = 10) const uint TILE_M    = 64;
layout(constant_id = 11) const uint TILE_N    = 64;
layout(constant_id = 12) const uint TILE_K    = 16;
layout(constant_id = 13) const bool USE_FP16  = true;
layout(constant_id = 14) const uint SUBGROUP_SIZE = 32;

layout(set = 0, binding = 0) readonly  buffer InA  { float dataA[]; }   ssboA;
layout(set = 0, binding = 1) readonly  buffer InB  { float dataB[]; }   ssboB;
layout(set = 0, binding = 2) writeonly buffer OutC { float dataC[]; }   ssboC;

layout(push_constant) uniform Push {
    uint M, N, K;
    uint flags;
} pc;

shared float tileA[TILE_M * TILE_K];   // shared memory (driver allocates per-workgroup)
shared float tileB[TILE_K * TILE_N];

void main() {
    // ... kernel body ...
}
```

Key points:

1. **`local_size_*_id`** — workgroup size is **specialization-constant**, not literal. Pipeline creation picks the actual values. This is essential: optimal `local_size` differs per vendor (e.g. 256 on NV, 64 on AMD wave64, 32 or 64 on Intel).
2. **Bindings** must match the `VkDescriptorSetLayout` used at pipeline creation. Use a stable convention across the engine:
   - `binding = 0` always reserved for the primary output
   - `binding = 1..` inputs in fixed order (A, B, bias, ...)
3. **`readonly` / `writeonly`** qualifiers let the driver elide hazards/coherency you don't need. Use them — but only after verifying every spec-const path. Example: the matmul output binding *cannot* be `writeonly` if the kernel supports `C = alpha*A*B + beta*C` (the `beta != 0` path reads C). See [PHASE_3_5_DEVIATIONS.md #8](../Checklists/PHASE_3_5_DEVIATIONS.md). Also note: GLSL has no built-in `erf`; the exact-GELU path uses an inline Abramowitz & Stegun 7.1.26 approximation — see [PHASE_3_5_DEVIATIONS.md #7](../Checklists/PHASE_3_5_DEVIATIONS.md).
4. **`shared` memory** — per-workgroup scratch. Vulkan minimum: 16 KB; modern desktop ≥ 32 KB. Statically sized at SPIR-V compile time (use `layout_size_id` spec consts to scale at pipeline creation).
5. **`push_constant`** struct mapped to the `vkCmdPushConstants` blob — exact byte layout enforced; `std430` rules apply. Keep ≤ 128 bytes.

### Built-in inputs

| Builtin | Type | Meaning |
|---|---|---|
| `gl_GlobalInvocationID` | `uvec3` | Global thread index = `gl_WorkGroupID * gl_WorkGroupSize + gl_LocalInvocationID` |
| `gl_LocalInvocationID` | `uvec3` | Index within workgroup |
| `gl_LocalInvocationIndex` | `uint` | Flattened: `lid.z * size.x*size.y + lid.y * size.x + lid.x` |
| `gl_WorkGroupID` | `uvec3` | Index of this workgroup in the dispatch grid |
| `gl_WorkGroupSize` | `uvec3` | Workgroup size (constant per pipeline, defaulted by spec consts) |
| `gl_NumWorkGroups` | `uvec3` | The (groupX, groupY, groupZ) passed to `vkCmdDispatch` |
| `gl_SubgroupSize` | `uint` | Lanes in the current subgroup (32 NV / 32 or 64 AMD / 8–32 Intel) |
| `gl_SubgroupInvocationID` | `uint` | This thread's lane within the subgroup `[0, gl_SubgroupSize)` |
| `gl_NumSubgroups` | `uint` | Subgroups per workgroup |
| `gl_SubgroupID` | `uint` | This subgroup's index within the workgroup |

`gl_SubgroupSize` is **dynamic** without `requiredSubgroupSize` set. Always pin via `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` so we can shadow it with a `constant_id` and the optimizer treats it as a literal.

---

## Mapping CUDA Concepts to Vulkan/GLSL

| CUDA concept | Vulkan / GLSL equivalent |
|---|---|
| Grid (`gridDim.x`) | `gl_NumWorkGroups.x` |
| Block (`blockDim.x`) | `gl_WorkGroupSize.x` |
| `threadIdx.x` | `gl_LocalInvocationID.x` |
| `blockIdx.x` | `gl_WorkGroupID.x` |
| `__shared__ float buf[N]` | `shared float buf[N];` |
| `__syncthreads()` | `barrier(); memoryBarrierShared();` |
| Warp = 32 threads | Subgroup = `gl_SubgroupSize` (variable!) |
| `__shfl_sync(mask, v, lane)` | `subgroupShuffle(v, lane)` |
| `__shfl_xor_sync(mask, v, mask)` | `subgroupShuffleXor(v, mask)` |
| `__shfl_down_sync(mask, v, delta)` | `subgroupShuffleDown(v, delta)` |
| `__ballot_sync(mask, pred)` | `subgroupBallot(pred)` |
| `__reduce_add_sync` (PTX warp reduce) | `subgroupAdd(v)` (one-shot reduction) |
| `atomicAdd` | `atomicAdd` (via `GL_ARB_shader_atomic_*`) |
| `cuda::std::__nv_bfloat16` | `float16_t` (GLSL `GL_EXT_shader_explicit_arithmetic_types`) |
| `__half2` packed | `f16vec2` |
| `wmma::fragment` (Tensor Cores) | `coopMatLoadNV` / `coopMatMulAddNV` (`VK_KHR_cooperative_matrix` 2024+) |
| `cudaMemcpyAsync` | `vkCmdCopyBuffer` |
| `cudaStreamSynchronize` | fence wait or timeline-semaphore wait |

---

## Subgroup Operations (Warp Primitives)

The `GL_KHR_shader_subgroup_*` family — split into 8 sub-extensions:

| Extension | Use |
|---|---|
| `_basic` | `gl_SubgroupSize`, `gl_SubgroupInvocationID`, `subgroupBarrier()`, `subgroupElect()` |
| `_vote` | `subgroupAll`, `subgroupAny`, `subgroupAllEqual` |
| `_arithmetic` | `subgroupAdd`, `subgroupMul`, `subgroupMin`, `subgroupMax`, `subgroupOr`, `subgroupAnd`, `subgroupXor`, plus `Inclusive*` and `Exclusive*` scan variants |
| `_ballot` | `subgroupBallot`, `subgroupBallotBitCount`, `subgroupBroadcastFirst` |
| `_shuffle` | `subgroupShuffle(v, lane)`, `subgroupBroadcast(v, lane)` |
| `_shuffle_relative` | `subgroupShuffleUp`, `subgroupShuffleDown`, `subgroupShuffleXor` (butterfly) |
| `_clustered` | `subgroupClusteredAdd(v, clusterSize)` etc. — power-of-2 ≤ subgroupSize |
| `_quad` | `subgroupQuadBroadcast` etc. — fixed 4-lane |

**For inference we need:** `_basic`, `_arithmetic`, `_shuffle`, `_shuffle_relative`. All are guaranteed on every desktop GPU since 2018; query at startup and fail fast otherwise.

### Reduction patterns

```glsl
// One-line subgroup reduction (preferred when subgroup-shaped)
float v = subgroupAdd(localValue);    // every lane gets the sum

// Manual butterfly reduction (when you need explicit control / cross-vendor predictability)
float v = localValue;
for (uint i = SUBGROUP_SIZE / 2; i > 0; i /= 2)
    v += subgroupShuffleXor(v, i);
// every lane now holds the sum across the subgroup
```

The one-line form may produce identical assembly to manual butterfly on NVIDIA but slightly different code on AMD; both are correct. **Prefer `subgroupAdd`/`subgroupMin`/`subgroupMax`** for readability; fall back to manual `Xor` only when the spec-constant-pinned subgroup size lets the compiler unroll the loop.

### Cross-subgroup reduction (workgroup-wide)

When the workgroup spans more than one subgroup (e.g. 256-wide group on NVIDIA = 8 subgroups of 32):

```glsl
shared float warp_sums[32];   // one slot per subgroup, max 32 (≥ 1024/32)

float v = subgroupAdd(localValue);          // intra-subgroup
if (subgroupElect())                         // one lane per subgroup writes
    warp_sums[gl_SubgroupID] = v;
barrier();

// Single subgroup reduces the per-subgroup sums
if (gl_SubgroupID == 0u) {
    float w = (gl_SubgroupInvocationID < gl_NumSubgroups)
            ? warp_sums[gl_SubgroupInvocationID] : 0.0;
    w = subgroupAdd(w);
    if (subgroupElect()) warp_sums[0] = w;
}
barrier();
float total = warp_sums[0];                  // every thread reads
```

Used in GroupNorm, LayerNorm, softmax, attention rowwise reductions. Identical in spirit to the CUDA `__shfl_xor_sync` + `__shared__` cross-warp reduce in [CUDA_AND_PTX.md](CUDA_AND_PTX.md#warp-shuffle-instructions-shflsync).

---

## FP16 in GLSL

Two extensions:

```glsl
#extension GL_EXT_shader_16bit_storage                    : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
```

Plus the corresponding device features: `storageBuffer16BitAccess` (Vulkan 1.1 core, optional feature) and `shaderFloat16` (Vulkan 1.2 core, optional feature). All RDNA, Pascal+, Intel Arc support both; pre-Pascal NVIDIA and pre-Vega AMD do not.

```glsl
layout(set=0, binding=0) readonly buffer InA  { float16_t dataA[]; }  ssboA;
layout(set=0, binding=1) writeonly buffer OutC { float16_t dataC[]; } ssboC;

float16_t v = ssboA.dataA[gid];
float     accum = float(v) * 2.0;       // accumulate in FP32
ssboC.dataC[gid] = float16_t(accum);
```

**Always accumulate in FP32 even for FP16 inputs.** This matches the CUDA path's `cublasGemmEx(... CUBLAS_COMPUTE_32F ...)` and the PTX kernels' explicit `cvt.f32.f16` before `add.f32`. FP16 accumulation introduces ~1e-2 error in deep reductions like 1024-element softmax — unacceptable.

### Packed `f16vec2`

GLSL has `f16vec2`, `f16vec4` for vectorized FP16 access. The driver maps these to a 32/64-bit `b32`/`b64` register. Use for memory-bandwidth-bound elementwise ops:

```glsl
f16vec2 a = unpackHalf2x16(uintA);    // 32-bit load split into two halves
f16vec2 b = unpackHalf2x16(uintB);
f16vec2 sum = a + b;
uint packed = packHalf2x16(sum);
ssboOut[gid] = packed;
```

Equivalent to PTX `add.rn.f16x2`. Useful for elementwise add/scale/silu where the kernel is memory-bound.

---

## Kernel Catalog

Every kernel in [src/HartsyInference.Cuda/Ptx/](../../src/HartsyInference.Cuda/Ptx/) needs a Vulkan counterpart in `native/vulkan/shaders/`. The mapping:

| CUDA PTX file | GLSL shader | Notes |
|---|---|---|
| `elementwise_f32.ptx` / `elementwise_f16.ptx` | `elementwise.comp.glsl` | One shader, dispatch via spec const for op (Add/Mul/Scale/Silu/GeLU/Sigmoid/Clamp) |
| `groupnorm_f32.ptx` / `groupnorm_f16.ptx` | `groupnorm.comp.glsl` | Per-(batch, group) workgroup; subgroup reduce → cross-warp |
| `groupnorm_silu_f32.ptx` / `_f16.ptx` | `groupnorm_silu.comp.glsl` | Fused: skip intermediate write |
| `layernorm_f32.ptx` / `_f16.ptx` | `layernorm.comp.glsl` | Per-token workgroup |
| `softmax_f32.ptx` / `_f16.ptx` | `softmax.comp.glsl` | Online (Welford-style) softmax for stability |
| `spatial_f32.ptx` / `_f16.ptx` | `im2col.comp.glsl` + `upsample_nearest2d.comp.glsl` + `upsample_bilinear2d.comp.glsl` + `col2bias_add.comp.glsl` | Split into per-op shaders |
| `transpose_f32.ptx` / `_f16.ptx` | `transpose.comp.glsl` | 32×32 tile, padded to avoid bank conflicts |
| `geglu_f32.ptx` / `_f16.ptx` | `geglu.comp.glsl` | Last-dim split (see PHASE_3_DEVIATIONS #16) |
| `broadcast_add_f32.ptx` / `_f16.ptx` | `broadcast_add.comp.glsl` | Channel-aware indexing |
| `cast_f32_f16.ptx` / `cast_f8e4m3_f16.ptx` | `cast_f32_f16.comp.glsl`, `cast_f8e4m3_f16.comp.glsl` | Dtype casts |
| (cuBLAS) | `matmul_tiled.comp.glsl` | **NEW** — replaces `cublasGemmEx`, the biggest piece of work |
| (cuBLAS via im2col) | reuses `matmul_tiled.comp.glsl` | Conv2D = im2col → tiled GEMM |
| (PTX SDPA, Phase 4) | `sdpa.comp.glsl` (FlashAttention-2 style) | Tiled QKV, online softmax |

Plus utility shaders that have no CUDA counterpart because cuBLAS handled them:
- `permute_0213.comp.glsl` — already needed; existing `permute_0213_f32.ptx` is the model
- `concat.comp.glsl`, `split.comp.glsl` — currently CPU-only on the CUDA backend; nice to add

Total: **~16 GLSL files**, one source per family with `#define` variants for FP32/FP16, all compiled to ~32 `.spv` files (one per dtype variant).

---

## Tiled GEMM (cuBLAS Replacement)

This is the central piece. We need: `C[M,N] = A[M,K] * B[K,N] + (optional bias)` with FP16 inputs, FP32 accumulation, optional transpose flags. Performance target: **≥ 60% of cuBLAS HGEMM** on the same NVIDIA hardware (validates the Vulkan backend is competitive); on AMD RDNA2/3, the baseline is hipBLAS / rocBLAS (we won't beat it without cooperative-matrix, but should match it within 30%).

### Algorithmic baseline

Standard CUTLASS-style 3-level tiling:

| Level | Per | Tile size | Storage |
|---|---|---|---|
| Workgroup | one workgroup → one `(BM, BN)` block of C | BM = 128, BN = 128, BK = 16 | Shared memory |
| Subgroup (warp) | one subgroup → one `(WM, WN)` sub-tile within the BM×BN block | WM = 64, WN = 32 | Registers |
| Thread | one invocation → an inner `(TM, TN)` micro-tile | TM = 8, TN = 8 | Registers |

Workgroup config: `local_size = (16, 16, 1)` = 256 threads → 8 subgroups of 32 on NVIDIA / Intel-Arc, or 4 subgroups of 64 on AMD-wave64. Each thread holds an 8×8 `float[64]` accumulator → 64 32-bit registers per thread, 16K register pressure per workgroup, plenty of headroom.

Shared memory:
- `tileA`: `BM × BK × 2 B = 128 × 16 × 2 = 4 KB` (FP16)
- `tileB`: `BK × BN × 2 B = 16 × 128 × 2 = 4 KB` (FP16)
- Total 8 KB — well under the 16 KB minimum guaranteed by Vulkan, fits comfortably alongside two-stage software pipelining (16 KB).

Per K-loop iteration: load 4 KB + 4 KB into shared, `barrier()`, FMA the sub-tile, `barrier()`. Outer K loop runs `K/BK` times.

### GLSL skeleton

```glsl
#version 460
#extension GL_KHR_shader_subgroup_basic    : require
#extension GL_KHR_shader_subgroup_arithmetic: require
#extension GL_EXT_shader_16bit_storage     : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 10) const uint BM = 128;
layout(constant_id = 11) const uint BN = 128;
layout(constant_id = 12) const uint BK = 16;
layout(constant_id = 13) const uint TM = 8;
layout(constant_id = 14) const uint TN = 8;
layout(constant_id = 15) const bool USE_FP16   = true;
layout(constant_id = 16) const bool TRANSPOSE_A = false;
layout(constant_id = 17) const bool TRANSPOSE_B = false;
layout(constant_id = 18) const bool HAS_BIAS    = false;
layout(constant_id = 19) const uint ACTIVATION  = 0u;   // 0=none 1=silu 2=gelu

layout(set=0, binding=0) readonly  buffer SsboA    { float16_t a[]; }    bufA;
layout(set=0, binding=1) readonly  buffer SsboB    { float16_t b[]; }    bufB;
layout(set=0, binding=2) writeonly buffer SsboC    { float16_t c[]; }    bufC;
layout(set=0, binding=3) readonly  buffer SsboBias { float16_t bias[]; } bufBias;

layout(push_constant) uniform Push {
    uint M, N, K;
    uint lda, ldb, ldc;   // leading dimensions
    uint flags;
} pc;

shared float16_t tileA[BM * BK];
shared float16_t tileB[BK * BN];

void main() {
    const uint blockRow = gl_WorkGroupID.y * BM;
    const uint blockCol = gl_WorkGroupID.x * BN;

    const uint threadsPerRow = BN / TN;     // 16 when BN=128, TN=8
    const uint threadId      = gl_LocalInvocationID.y * gl_WorkGroupSize.x
                              + gl_LocalInvocationID.x;
    const uint threadRow     = (threadId / threadsPerRow) * TM;
    const uint threadCol     = (threadId % threadsPerRow) * TN;

    float acc[TM][TN];
    for (uint i = 0; i < TM; ++i)
        for (uint j = 0; j < TN; ++j) acc[i][j] = 0.0;

    const uint kBlocks = (pc.K + BK - 1) / BK;
    for (uint kb = 0; kb < kBlocks; ++kb) {
        // Cooperative load A tile (BM × BK) and B tile (BK × BN) — each thread loads multiple elements
        const uint loadsPerThreadA = (BM * BK + gl_WorkGroupSize.x*gl_WorkGroupSize.y - 1) /
                                     (gl_WorkGroupSize.x * gl_WorkGroupSize.y);
        for (uint i = 0; i < loadsPerThreadA; ++i) {
            uint idx = threadId + i * gl_WorkGroupSize.x*gl_WorkGroupSize.y;
            if (idx < BM * BK) {
                uint row = idx / BK;
                uint col = idx % BK;
                uint gRow = blockRow + row;
                uint gCol = kb * BK + col;
                tileA[idx] = (gRow < pc.M && gCol < pc.K)
                           ? bufA.a[gRow * pc.lda + gCol]
                           : float16_t(0.0);
            }
        }
        // ...same for tileB...
        barrier();

        // Inner FMA — TM x TN micro-tile per thread
        for (uint k = 0; k < BK; ++k) {
            float aReg[TM];
            float bReg[TN];
            for (uint i = 0; i < TM; ++i)
                aReg[i] = float(tileA[(threadRow + i) * BK + k]);
            for (uint j = 0; j < TN; ++j)
                bReg[j] = float(tileB[k * BN + threadCol + j]);
            for (uint i = 0; i < TM; ++i)
                for (uint j = 0; j < TN; ++j)
                    acc[i][j] = fma(aReg[i], bReg[j], acc[i][j]);
        }
        barrier();
    }

    // Bias + activation + write back
    for (uint i = 0; i < TM; ++i) {
        for (uint j = 0; j < TN; ++j) {
            uint gRow = blockRow + threadRow + i;
            uint gCol = blockCol + threadCol + j;
            if (gRow < pc.M && gCol < pc.N) {
                float v = acc[i][j];
                if (HAS_BIAS) v += float(bufBias.bias[gCol]);
                if      (ACTIVATION == 1u) v = v * (1.0 / (1.0 + exp(-v)));      // SiLU
                else if (ACTIVATION == 2u) v = 0.5 * v * (1.0 + tanh(0.7978845608 * (v + 0.044715 * v*v*v))); // GELU-tanh
                bufC.c[gRow * pc.ldc + gCol] = float16_t(v);
            }
        }
    }
}
```

This is the *baseline* shader. Optimizations to apply once correctness is proven:

1. **Vectorized 128-bit loads** — read 8 FP16 values per thread per inner load with `f16vec4` packed into `uvec2`. Reduces shared-memory write count by 8×.
2. **Double-buffered K loop** — load tile `kb+1` into a second `shared` slot while computing on tile `kb`. Hides global-memory latency behind FMA.
3. **Subgroup-cooperative inner loop** — one subgroup computes one WM×WN sub-tile of the workgroup tile via `subgroupShuffle` of A operands.
4. **Cooperative-matrix path** — when `VK_KHR_cooperative_matrix` is supported (RDNA3, Turing+, Intel Arc), replace inner FMA with `coopMatMulAdd`. 4–8× speedup on supporting hardware.

Initial implementation skips (3) and (4); land (1) and (2) as Phase-3.5 optimizations.

### Variant table (compiled at build time)

Each variant produces a separate `.spv` file. We exploit specialization constants to keep the source single while still getting fully-optimized binaries.

| Variant | TRANSPOSE_A | TRANSPOSE_B | USE_FP16 | HAS_BIAS | ACTIVATION |
|---|---|---|---|---|---|
| `matmul_nn_f32.spv` | F | F | F | F | 0 |
| `matmul_nn_f16.spv` | F | F | T | F | 0 |
| `matmul_nt_f16.spv` | F | T | T | F | 0 |
| `matmul_tn_f16.spv` | T | F | T | F | 0 |
| `linear_nn_f16.spv` | F | F | T | T | 0 |
| `linear_nn_f16_silu.spv` | F | F | T | T | 1 |
| `linear_nn_f16_gelu.spv` | F | F | T | T | 2 |

The CUDA backend has these variants implicit in cuBLAS — Vulkan needs them explicit. Bias + activation fusion into the matmul (variants 5–7) is **the** Phase-3.5 fusion to land first because it eliminates one full-tensor write+read for every Linear layer (≈ 100 layers in SDXL → measurable speedup).

### Tile-size tuning (per vendor)

| Vendor | BM × BN | BK | local_size | Notes |
|---|---|---|---|---|
| NVIDIA Turing/Ampere/Ada | 128 × 128 | 16 | (16, 16, 1) | default; matches PTX practice |
| AMD RDNA2/3 (wave32) | 128 × 64 | 16 | (16, 16, 1) | smaller BN — wave32 prefers shorter rows |
| AMD GCN/RDNA wave64 | 64 × 64 | 16 | (16, 16, 1) | 64-lane wave |
| Intel Arc (variable) | 64 × 64 | 16 | (8, 8, 1) — 64 threads | smaller workgroup |
| Apple M-series (MoltenVK) | 32 × 32 | 16 | (8, 8, 1) | smallest tile, low shared mem |

Picked at pipeline-creation time based on `vendorID`. Pre-build all reasonable variants; the auto-tuner (Phase 4) picks the fastest at first call.

---

## Per-Kernel Designs

### `groupnorm.comp.glsl` (and fused `groupnorm_silu`)

Math: per-(batch, group) → mean, variance over channels-in-group × spatial; then normalize + affine.

```glsl
#version 460
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x_id = 0) in;       // typically 256, scaled by spec const
layout(constant_id = 1) const uint CHANNELS_PER_GROUP = 32;
layout(constant_id = 2) const bool USE_FP16 = true;
layout(constant_id = 3) const bool FUSE_SILU = false;

layout(set=0, binding=0) readonly  buffer InX     { float16_t x[]; }     bufX;
layout(set=0, binding=1) writeonly buffer OutY    { float16_t y[]; }     bufY;
layout(set=0, binding=2) readonly  buffer Weight  { float weight[]; }    bufW;   // FP32 affine
layout(set=0, binding=3) readonly  buffer Bias    { float bias[]; }      bufB;

layout(push_constant) uniform Push {
    uint N, C, H, W, GROUPS;      // C = channels (multiple of GROUPS)
    float eps;
} pc;

shared float warp_sums[32];
shared float warp_sqsums[32];
shared float gMean;
shared float gInvStd;

void main() {
    uint nGroup    = gl_WorkGroupID.x;          // dispatch (B*GROUPS, 1, 1)
    uint b         = nGroup / pc.GROUPS;
    uint g         = nGroup % pc.GROUPS;
    uint cPerGroup = pc.C / pc.GROUPS;
    uint groupSize = cPerGroup * pc.H * pc.W;
    uint baseOff   = (b * pc.C + g * cPerGroup) * pc.H * pc.W;

    // Phase 1: thread-local sum + sqsum
    float sum = 0.0, sqsum = 0.0;
    for (uint i = gl_LocalInvocationIndex; i < groupSize; i += gl_WorkGroupSize.x) {
        float v = float(bufX.x[baseOff + i]);
        sum   += v;
        sqsum += v * v;
    }

    // Phase 2: subgroup reduce
    sum   = subgroupAdd(sum);
    sqsum = subgroupAdd(sqsum);

    // Phase 3: cross-subgroup via shared memory
    if (subgroupElect()) {
        warp_sums[gl_SubgroupID]   = sum;
        warp_sqsums[gl_SubgroupID] = sqsum;
    }
    barrier();
    if (gl_SubgroupID == 0u) {
        float w  = (gl_SubgroupInvocationID < gl_NumSubgroups)
                 ? warp_sums[gl_SubgroupInvocationID] : 0.0;
        float w2 = (gl_SubgroupInvocationID < gl_NumSubgroups)
                 ? warp_sqsums[gl_SubgroupInvocationID] : 0.0;
        w  = subgroupAdd(w);
        w2 = subgroupAdd(w2);
        if (subgroupElect()) {
            float invN = 1.0 / float(groupSize);
            float mean = w * invN;
            float var  = w2 * invN - mean * mean;
            gMean   = mean;
            gInvStd = inversesqrt(var + pc.eps);
        }
    }
    barrier();

    // Phase 4: normalize + affine + (optional) SiLU
    for (uint i = gl_LocalInvocationIndex; i < groupSize; i += gl_WorkGroupSize.x) {
        uint cIdx = (i / (pc.H * pc.W)) + g * cPerGroup;
        float v = float(bufX.x[baseOff + i]);
        float n = (v - gMean) * gInvStd;
        n = n * bufW.weight[cIdx] + bufB.bias[cIdx];
        if (FUSE_SILU) n = n * (1.0 / (1.0 + exp(-n)));
        bufY.y[baseOff + i] = float16_t(n);
    }
}
```

Direct port of the existing `groupnorm_silu_f16.ptx`. **Tolerance vs CPU reference: 1e-3 (FP16), 1e-5 (FP32).**

### `layernorm.comp.glsl`

Same structure as GroupNorm but per-token: `dispatch(B*S, 1, 1)`, reduce over hidden dim.

### `softmax.comp.glsl`

Online softmax (Welford-style) — single pass, numerically stable. One workgroup per row.

```glsl
// Per row of length N, single workgroup
shared float sMaxVal;
shared float sSumExp;

void main() {
    uint row    = gl_WorkGroupID.x;
    uint tid    = gl_LocalInvocationIndex;
    uint stride = gl_WorkGroupSize.x;

    // Pass 1: row max
    float localMax = -1e30;
    for (uint j = tid; j < pc.N; j += stride)
        localMax = max(localMax, float(buf.x[row * pc.N + j]));
    localMax = subgroupMax(localMax);
    if (subgroupElect()) warp_max[gl_SubgroupID] = localMax;
    barrier();
    if (gl_SubgroupID == 0u) {
        float w = (gl_SubgroupInvocationID < gl_NumSubgroups)
                ? warp_max[gl_SubgroupInvocationID] : -1e30;
        w = subgroupMax(w);
        if (subgroupElect()) sMaxVal = w;
    }
    barrier();

    // Pass 2: sum exp(x - max)
    float localSum = 0.0;
    for (uint j = tid; j < pc.N; j += stride)
        localSum += exp(float(buf.x[row * pc.N + j]) - sMaxVal);
    localSum = subgroupAdd(localSum);
    // ...cross-warp reduce into sSumExp...

    // Pass 3: write normalized
    float invSum = 1.0 / sSumExp;
    for (uint j = tid; j < pc.N; j += stride) {
        float v = exp(float(buf.x[row * pc.N + j]) - sMaxVal) * invSum;
        buf.y[row * pc.N + j] = float16_t(v);
    }
}
```

Tolerance: 1e-3 FP16. Same algorithm as `softmax_f16.ptx`.

### `im2col.comp.glsl` + tiled GEMM (Conv2D)

Conv2D = im2col + GEMM. The im2col shader is purely an indexed copy:

```glsl
void main() {
    uint outH = pc.outH, outW = pc.outW;
    uint kHkW = pc.kH * pc.kW;
    uint colsPerImage = outH * outW;
    uint rowsPerImage = pc.C_in * kHkW;

    uint tid = gl_GlobalInvocationID.x;        // dispatch (rows*cols, 1, 1)
    if (tid >= rowsPerImage * colsPerImage * pc.N) return;

    uint n  = tid / (rowsPerImage * colsPerImage);
    uint rc = tid % (rowsPerImage * colsPerImage);
    uint row = rc / colsPerImage;
    uint col = rc % colsPerImage;

    uint c     = row / kHkW;
    uint kIdx  = row % kHkW;
    uint kh    = kIdx / pc.kW;
    uint kw    = kIdx % pc.kW;
    uint outY  = col / outW;
    uint outX  = col % outW;
    int  iy    = int(outY * pc.strideH + kh) - int(pc.padH);
    int  ix    = int(outX * pc.strideW + kw) - int(pc.padW);

    float v = 0.0;
    if (iy >= 0 && iy < int(pc.H) && ix >= 0 && ix < int(pc.W))
        v = float(bufIn.x[((n * pc.C_in + c) * pc.H + uint(iy)) * pc.W + uint(ix)]);
    bufCol.col[((n * rowsPerImage + row) * colsPerImage) + col] = float16_t(v);
}
```

**Use 64-bit indexing for resolutions ≥ 1024**. Same pitfall as PTX: products like `C × kH × kW × outH × outW` overflow `uint32` for SDXL's largest layer. Either split the dispatch or compute the index in `uint64_t` (`GL_EXT_shader_explicit_arithmetic_types_int64`) — see [PHASE_3_DEVIATIONS.md #12](../Checklists/PHASE_3_DEVIATIONS.md).

After im2col, dispatch the tiled GEMM kernel (`matmul_tiled.comp.glsl`) with A = weight `[C_out, C_in*kH*kW]`, B = col `[C_in*kH*kW, N*outH*outW]`, C = output `[C_out, N*outH*outW]`, then use `col2bias_add.comp.glsl` (or fused into the GEMM) to add bias and reshape to NCHW.

### `sdpa.comp.glsl` (FlashAttention-2 style, Phase 4)

One workgroup per (batch, head, Br query rows). Tile Q rows into Br=64 blocks; loop K/V tiles of Bc=64 rows; accumulate softmax-weighted V using online softmax. Shared memory: 64×D Q, 64×D K, 64×D V (D=64 typical for SD1.5; D=128 for SDXL).

For SD1.5 / SDXL Phase-3.5 we accept the simpler **3-pass naive SDPA** (Q×Kᵀ → softmax → ×V) with `matmul_tiled.comp.glsl` chained twice plus the softmax shader. FlashAttention is a Phase 4+ optimization — the same path the CUDA backend is taking.

### `geglu.comp.glsl`

```glsl
// Input: [..., 2*D]   Output: [..., D]   y[i] = x[i] * gelu(x[i + D])
void main() {
    uint flat = gl_GlobalInvocationID.x;
    uint outerD = pc.D;             // last-dim half
    uint outer  = flat / outerD;
    uint d      = flat % outerD;
    uint baseIn = outer * 2u * outerD;
    if (flat >= pc.OuterCount * outerD) return;

    float xVal  = float(bufIn.x[baseIn + d]);
    float gate  = float(bufIn.x[baseIn + outerD + d]);
    float gelu  = 0.5 * gate * (1.0 + tanh(0.7978845608 * (gate + 0.044715 * gate*gate*gate)));
    bufOut.y[outer * outerD + d] = float16_t(xVal * gelu);
}
```

**Last-dim split**, not flat midpoint — exactly the same pitfall fixed in [PHASE_3_DEVIATIONS.md #16](../Checklists/PHASE_3_DEVIATIONS.md). The bug bit us in PTX; transferring the bug to GLSL would be regression. Test with multi-row inputs.

### `transpose.comp.glsl`

Per-tile 32×32 with +1 padding to dodge bank conflicts:

```glsl
shared float16_t tile[32][33];   // +1 padding

void main() {
    uvec2 ti = gl_LocalInvocationID.xy;
    uvec2 wg = gl_WorkGroupID.xy * 32u;

    uint inRow = wg.y + ti.y;
    uint inCol = wg.x + ti.x;
    if (inRow < pc.H && inCol < pc.W)
        tile[ti.y][ti.x] = bufIn.x[inRow * pc.W + inCol];
    barrier();

    uint outRow = wg.x + ti.y;
    uint outCol = wg.y + ti.x;
    if (outRow < pc.W && outCol < pc.H)
        bufOut.y[outRow * pc.H + outCol] = tile[ti.x][ti.y];
}
```

### `broadcast_add.comp.glsl`, `permute_0213.comp.glsl`, `cast_*.comp.glsl`

Direct ports of the existing PTX with the same indexing — straightforward.

### `elementwise.comp.glsl`

One shader, op selected by spec const:

```glsl
layout(constant_id = 1) const uint OP = 0u;   // 0 add 1 mul 2 scale 3 silu 4 gelu 5 sigmoid 6 clamp

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.N) return;
    float a = float(bufA.x[i]);
    float b = (OP <= 1u) ? float(bufB.x[i]) : 0.0;
    float r;
    if      (OP == 0u) r = a + b;
    else if (OP == 1u) r = a * b;
    else if (OP == 2u) r = a * pc.scalar;
    else if (OP == 3u) r = a * (1.0 / (1.0 + exp(-a)));
    else if (OP == 4u) r = 0.5 * a * (1.0 + tanh(0.7978845608 * (a + 0.044715*a*a*a)));
    else if (OP == 5u) r = 1.0 / (1.0 + exp(-a));
    else /* 6 */       r = clamp(a, pc.minVal, pc.maxVal);
    bufC.y[i] = float16_t(r);
}
```

---

## Validation Tolerances

Same table as KERNEL.md, restated with measured GPU-vs-CPU error from the CUDA backend (lower is better; Vulkan should match):

| Kernel | Reference | Tolerance | Measured (CUDA F16) |
|---|---|---|---|
| Elementwise FP32 | CPU scalar | 1e-5 | < 1e-6 |
| Elementwise FP16 | CPU scalar | 1e-3 | ~5e-4 |
| GroupNorm + affine FP16 | CPU FP32 | 1e-3 | ~1e-3 |
| LayerNorm FP16 | CPU FP32 | 1e-3 | ~1e-3 |
| Softmax FP16 | CPU FP32 | 1e-3 | — |
| Tiled GEMM FP16 | CPU GEMM FP32 | 1e-3 | (cuBLAS HGEMM) ~5e-4 |
| Conv2D FP16 (im2col + GEMM) | CPU FP32 | 1e-3 | ~1e-3 |
| SDPA FP16 | CPU FP32 | 1e-3 | — |
| Vulkan kernel vs CUDA kernel (same dtype) | CUDA result | 1e-3 | gate for Phase-3.5 acceptance |

**End-to-end gate (Phase 3.5 acceptance):** SD1.5 Vulkan vs CUDA same seed → same image → SSIM > 0.99 (visually indistinguishable). Same gate as CUDA's vs CPU.

---

## Performance Targets & Pitfalls

### Targets (RTX 3060 12 GB, FP16)

| Pipeline | CUDA (current) | Vulkan target |
|---|---|---|
| SD1.5 512×512 / 20 steps | ~5 s total | ≤ 8 s (60% of CUDA) |
| SDXL 1024×1024 / 20 steps | ~110 s total | ≤ 180 s (60% of CUDA) |

On AMD RX 7900 XTX (no CUDA reference), **target = match within 30% of estimated peak** (peak FP16 throughput ÷ kernel arithmetic intensity).

### Pitfalls (real bugs to design around)

1. **Subgroup-size assumption** — code that assumes 32 will mis-reduce on AMD GCN/wave64. Always pin via `requiredSubgroupSize` and shadow as a spec constant.
2. **Variable subgroup size on Intel** — Intel Arc can pick 8/16/32 per pipeline. Use `requiredSubgroupSize = 32` (always supported) so cross-warp reduction shared-memory size is a known compile-time constant.
3. **`barrier()` + `memoryBarrierShared()`** — the GLSL `barrier()` only blocks invocations; the `memoryBarrierShared()` flushes shared memory. **Both** are required after writing shared memory if a different invocation will read it. (Drivers happen to imply the second on most GPUs, but the spec doesn't guarantee it.)
4. **64-bit indexing for SDXL** — the `int64_t` extension (`GL_EXT_shader_explicit_arithmetic_types_int64`) is mandatory for spatial kernels at 1024+ resolution. Same fix as the PTX side.
5. **Shared-memory bank conflicts** — Vulkan shared mem has 32 banks of 4 bytes (matching CUDA). Apply the same +1 padding trick on every transposed access.
6. **FP16 accumulation** — accumulate in `float`, never `float16_t`. The driver may *try* to fuse reductions; explicit FP32 widening prevents lossy paths.
7. **Spec-constant `local_size`** — must be set at pipeline creation time. Forgetting to pass `VkSpecializationInfo` makes the workgroup default to (1,1,1) — kernel runs but is 256× slower. Sentinel: assert local_size > 1 in tests.
8. **Push-constant overflow** — > 128 bytes fails on the spec minimum; we'd never know on a 256-byte device until shipping. Hard-cap 128 in the layout assembler.
9. **`vkCmdDispatch` group counts** — `group_x * group_y * group_z` capped per dim at `maxComputeWorkGroupCount[i]` (≥ 65535 guaranteed). For a flat dispatch of 4M threads we still need to chunk into a 2D grid because group_x is bounded.
10. **Coherent vs non-coherent staging** — forgetting `vkFlushMappedMemoryRanges` on `HOST_VISIBLE` non-coherent memory results in zeroed weight uploads. Always flush before submit.
11. **Pipeline barrier scope** — covering "all buffers" with `ALL_COMMANDS` kills concurrency. Use buffer-scoped `VkBufferMemoryBarrier2` per dispatch output (see [VULKAN_COMPUTE_API.md § Pipeline Barriers](VULKAN_COMPUTE_API.md#pipeline-barriers-synchronization-2)).
12. **Mesa RADV pipeline cache** — was buggy on certain RDNA driver versions; treat the cache as best-effort, fall back to recompile on `VK_INCOMPLETE`.

---

## Cooperative Matrix (Optional Phase 4+ Optimization)

`VK_KHR_cooperative_matrix` (2024, finalized) gives access to vendor matrix accelerators: NVIDIA Tensor Cores, AMD WMMA (RDNA3+), Intel XMX. Equivalent in spirit to PTX `wmma`. Cooperative matrices are *subgroup-cooperative* fixed-size matrices held in registers; you load, multiply-accumulate, and store them.

```glsl
#extension GL_KHR_cooperative_matrix : require

coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>     A;
coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>     B;
coopmat<float,     gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> C;

coopMatLoad(A, ssboA, offsetA, strideA, gl_CooperativeMatrixLayoutRowMajor);
coopMatLoad(B, ssboB, offsetB, strideB, gl_CooperativeMatrixLayoutColumnMajor);
C = coopMatMulAdd(A, B, C);
coopMatStore(C, ssboC, offsetC, strideC, gl_CooperativeMatrixLayoutRowMajor);
```

Supported sizes are vendor-specific (queried via `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`). Typical: NVIDIA 16×16×16, AMD 16×16×16. Expected gain: 4–8× on supported HW. **Phase 3.5 ships without this**; Phase 4 adds an alternate `matmul_coopmat.comp.glsl` selected at pipeline-creation time when the extension is supported.

---

## Per-Vendor Driver Notes

| Vendor | Notes |
|---|---|
| NVIDIA | Excellent Vulkan compute throughput, ~95% of CUDA peak in tiled GEMM tests. Subgroup ops mature. Pipeline cache works. Vulkan 1.3 since R465 (2021). |
| AMD RADV (Mesa) | Best AMD driver. Wave32 default on RDNA, wave64 selectable. Performance has caught up to ROCm in 2024+. Set `RADV_PERFTEST=gpl` for best pipeline-creation perf. |
| AMD AMDVLK | AMD's official Vulkan driver. Slightly different perf profile from RADV; we don't optimize for it but it should work. |
| Intel ANV (Mesa) | Variable subgroup size — must pin. Modest VRAM bandwidth on iGPU; treat as fallback target. |
| Intel Arc | Strong FP16 perf, XMX matrix ops. Cooperative-matrix supported as of Mesa 24.x. |
| Apple MoltenVK | Vulkan-on-Metal translation layer. Subgroup ops mostly OK; no cooperative-matrix; 32 KB shared-memory limit. Out of scope for Phase 3.5 — verify in Phase 7+. |

---

## Implementation Notes

1. **One `.comp.glsl` source per kernel family.** Use `#define`/spec consts for dtype and op variants. Compile to multiple `.spv` at build time.
2. **Spec consts for everything tunable.** `local_size`, `TILE_*`, dtype flags. Pipeline creation picks the values per (vendor, problem size).
3. **Shared memory size is compile-time-static in GLSL.** If we need vendor-different sizes, pre-build per-vendor variants of the GEMM shader.
4. **Always FP32-accumulate** — even for FP16 inputs, even when the GLSL would let you accumulate in `float16_t`.
5. **Never trust `gl_SubgroupSize` at runtime.** Pin it at pipeline creation and shadow into a spec const.
6. **Last-dim split for gated activations.** GEGLU/SwiGLU bug from PTX must not regress.
7. **64-bit indexing for SDXL spatial kernels.** Use `GL_EXT_shader_explicit_arithmetic_types_int64`.
8. **Always pad shared-memory tiles** to avoid bank conflicts (32 banks × 4 bytes).
9. **Bias + activation fuse into matmul.** Single inner write per output element.
10. **Validate every shader against CPU reference.** Same test harness as CUDA; reuse [`HartsyInference.Diffusion.Tests`] kernel-correctness tests, switch backend.
11. **Persist `VkPipelineCache`** to disk per-`deviceUUID`; cuts cold-start by 0.5–2 s.
12. **`spirv-val` every shader** in CI; fails fast on malformed SPIR-V before runtime.
13. **`spirv-opt -O` after build** unless explicitly disabled — driver still re-optimizes but starting from optimized SPIR-V is faster JIT.

---

## Open Questions

- [ ] Whether to ship pre-specialized `.spv` per common vendor tile config or always JIT spec consts at startup. Initial plan: spec consts.
- [ ] Whether `VK_KHR_buffer_device_address` simplifies SDPA enough to be worth the device-feature requirement (need on RDNA3, Turing+, Arc).
- [ ] Whether to exploit subgroup-uniform branching (`subgroupBroadcastFirst`) in the softmax max-broadcast.
- [ ] Best 2-stage software pipelining shape for the GEMM K-loop on AMD wave32 vs wave64.
- [ ] Whether to expose a `VulkanKernelTuner` that auto-runs all tile-size variants on first launch and picks the winner, persisting choice to disk.
- [ ] How aggressive to be with cooperative-matrix in Phase 4 — wait for Mesa coverage to stabilize across RDNA3/4 + Intel Arc?

---

## References

- [SPIR-V 1.6 Specification](https://registry.khronos.org/SPIR-V/specs/1.6/SPIRV.html) — definitive
- [GLSL 4.60 Specification](https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.pdf)
- [GL_KHR_shader_subgroup family](https://github.com/KhronosGroup/GLSL/blob/main/extensions/khr/GL_KHR_shader_subgroup.txt)
- [GL_EXT_shader_explicit_arithmetic_types_float16](https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GL_EXT_shader_explicit_arithmetic_types.txt)
- [Khronos Vulkan Guide — Compute Shader](https://docs.vulkan.org/guide/latest/computeshader.html)
- [Khronos Vulkan Guide — Subgroups](https://docs.vulkan.org/guide/latest/subgroups.html)
- [VK_KHR_cooperative_matrix proposal](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_cooperative_matrix.html)
- [glslang](https://github.com/KhronosGroup/glslang) — `glslangValidator` source
- [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools) — `spirv-opt`, `spirv-val`, `spirv-dis`
- [Vulkan-Samples — compute samples](https://github.com/KhronosGroup/Vulkan-Samples/tree/main/samples/api)
- [llama.cpp — ggml-vulkan backend](https://github.com/ggerganov/llama.cpp/tree/master/ggml/src/ggml-vulkan) — production reference: tiled HGEMM, dequant, K-cache; mature, vendor-tested, well-commented C++
- [VkFFT](https://github.com/DTolm/VkFFT) — large production Vulkan-compute project; shared-memory + subgroup tactics
- [NCNN Vulkan backend](https://github.com/Tencent/ncnn/tree/master/src/layer/vulkan) — Tencent's mobile-first Vulkan inference; relevant for Mali/Adreno
- [AMD wave32 vs wave64 guide](https://gpuopen.com/learn/wave_intrinsics_unleashed/) — RDNA tuning
- [NVIDIA Vulkan compute tips](https://developer.nvidia.com/blog/vulkan-tips/)
- [Intel Vulkan compute samples](https://github.com/intel/compute-runtime) — variable subgroup-size patterns
