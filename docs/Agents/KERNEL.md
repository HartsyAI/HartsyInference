# Kernel Agent

> **Role:** Write high-performance SIMD CPU kernels, CUDA PTX GPU kernels, and Vulkan SPIR-V compute shaders. This is the most performance-critical code in SharpInference. Follow dotLLM's kernel patterns exactly (SIMD dispatch, PTX management from disk, `nint` field handles, `stackalloc` args, `int` returns) and extend them to image/audio/vision domains and Vulkan compute.

---

## Before You Start

Read these files:
- `docs/CODE_STYLE.md` -- **MANDATORY** code style and guidelines (follow this always)
- `docs/Design/CORE_DESIGN.md` -- design pillars (pure C#, PTX via CUDA Driver API, SPIR-V via Vulkan API)
- `docs/Design/IMPLEMENTATION_DETAILS.md` -- CPU, CUDA, and Vulkan backend architecture
- `docs/Research/DOTLLM_ARCHITECTURE.md` -- **CRITICAL** dotLLM's kernel patterns (SIMD dispatch, PTX loading from disk, function handle fields, fusion strategy, `int` returns, `CudaException` pattern)
- `docs/Research/SIMD_INTRINSICS_DOTNET.md` -- .NET SIMD API surface
- `docs/Research/PTX_KERNELS.md` -- PTX ISA reference
- `docs/Research/VULKAN_COMPUTE_API.md` -- Vulkan compute pipeline setup
- `docs/Research/SPIRV_COMPUTE_SHADERS.md` -- SPIR-V compute shader patterns
- `docs/Research/IM2COL_CPU.md` -- Conv2D CPU algorithm (if writing conv kernels)
- `docs/Research/GROUPNORM_MATH.md` -- normalization math (if writing norm kernels)
- `docs/Research/FLASH_ATTENTION.md` -- tiled attention algorithm (if writing attention)
- `docs/Research/CONV2D_CUDA.md` -- cuDNN bindings (if writing CUDA conv)
- Existing kernels in `src/SharpInference.Cpu/Kernels/`, `src/SharpInference.Cuda/Ptx/`, and `src/SharpInference.Vulkan/Spirv/`

## CPU Kernel Workflow (follows dotLLM's SIMD dispatch pattern)

1. **Understand the math** -- read the research doc, understand the operation precisely
2. **Write the scalar reference** -- simple, obviously-correct C# loop implementation
3. **Write the AVX2 path** -- `Vector256<float>` with cross-platform vector types (not platform-specific intrinsics)
4. **Write the AVX-512 path** (if beneficial) -- `Vector512<float>`
5. **Wire up SimdDispatch** -- `Vector512.IsHardwareAccelerated` -> `Vector256.IsHardwareAccelerated` -> scalar (same as dotLLM)
6. **Validate** -- output must match scalar reference within tolerance (dotLLM's SIMD-vs-scalar test pattern)

### CPU Kernel Standards

```csharp
// Pattern: SimdDispatch routing (from dotLLM)
public static void GroupNorm(Span<float> output, ReadOnlySpan<float> input,
    ReadOnlySpan<float> weight, ReadOnlySpan<float> bias, int groups, float eps)
{
    if (Vector512.IsHardwareAccelerated)
        GroupNormAvx512(output, input, weight, bias, groups, eps);
    else if (Vector256.IsHardwareAccelerated)
        GroupNormAvx2(output, input, weight, bias, groups, eps);
    else
        GroupNormScalar(output, input, weight, bias, groups, eps);
}
```

- Always provide a scalar fallback -- **mandatory** (dotLLM rule)
- Use `Span<T>` / `ReadOnlySpan<T>` for all data parameters -- never raw pointers in the public API
- Use `TensorPrimitives` when it provides the operation you need (dotLLM convention)
- Handle tail elements (when data length isn't a multiple of vector width)
- Accumulate in FP32 even when inputs are FP16 -- prevent precision loss
- Use `[MethodImpl(MethodImplOptions.AggressiveInlining)]` on small helpers (dotLLM convention)
- Comment the math -- what each SIMD operation corresponds to in the algorithm
- Cross-platform vector types (`Vector256<float>`) preferred over platform-specific (dotLLM convention)

### Common CPU Pitfalls
- Forgetting tail elements after the vectorized loop
- FP16 accumulation causing precision loss (always accumulate in FP32)
- Cache-unfriendly access patterns in im2col
- Not handling padding/stride edge cases in Conv2D
- AVX-512 downclocking on Intel CPUs -- benchmark both paths

## PTX Kernel Workflow (follows dotLLM's CudaKernels pattern)

1. **Understand the math** -- same as CPU
2. **Design the tiling** -- how threads map to output elements, shared memory usage
3. **Write the CUDA C source** -- `.cu` file in `native/cuda/kernels/`
4. **Compile to PTX** -- `nvcc -ptx -arch=compute_80` (dotLLM baseline)
5. **Ship as content file** -- PTX file goes in a directory alongside .NET assemblies (NOT embedded as resource)
6. **Write the launch wrapper** -- C# code that sets grid/block dims and calls `cuLaunchKernel` with `stackalloc void*[]` args and **local variables for stable addresses** (dotLLM pattern)
7. **Store function handle as `nint` field** -- resolved once in the `CudaKernels` constructor, NOT dictionary-cached (dotLLM pattern)
8. **Validate** -- output must match CPU kernel within FP16 tolerance

### PTX Kernel Standards

```
// File naming: operation_dtype_variant.ptx
// Example: conv2d_f16_3x3.ptx, group_norm_f16.ptx, group_norm_silu_fused.ptx

.version 8.0
.target sm_80
.address_size 64
```

- Target `sm_80` minimum (Ampere) -- covers RTX 30xx, 40xx, A100, H100. PTX is forward-compatible -- GPU driver JIT-compiles to native SASS (same as dotLLM)
- Use FP16 arithmetic (`add.f16x2`, `mul.f16x2`) for throughput
- Use shared memory tiling for memory-bound kernels (conv2d, attention)
- Avoid bank conflicts in shared memory access
- Use warp shuffle (`shfl.sync`) for reductions (norm kernels)
- Keep register pressure manageable -- check occupancy
- Document grid/block dimension expectations in comments
- **Consider kernel fusion** -- follow dotLLM's fusion philosophy (GroupNorm+SiLU, Conv2D+bias+activation, fused attention). Memory bandwidth is the bottleneck.

### PTX Launch Wrapper (from dotLLM -- source verified)

```csharp
// dotLLM's exact pattern: local variables for stable addresses, stackalloc args
public void LaunchGroupNormSiLU(nint output, nint input, nint weight, nint bias,
    int channels, int groups, float eps, nint stream)
{
    // Local variables ensure pointer stability during kernel launch
    nint outputArg = output, inputArg = input, weightArg = weight, biasArg = bias;
    int channelsArg = channels, groupsArg = groups;
    float epsArg = eps;

    void** args = stackalloc void*[] {&outputArg, &inputArg, &weightArg,
        &biasArg, &channelsArg, &groupsArg, &epsArg};

    // int return type, NOT CuResult enum. BlockSize = 256.
    CudaDriverApi.cuLaunchKernel(
        _groupNormSiluFusedFunc,  // nint field, NOT dictionary lookup
        (uint)groups, 1, 1, 256, 1, 1,
        0, stream, (nint)args, 0).ThrowOnError();
}
```

### PTX Loading Pattern (from dotLLM -- source verified)

```csharp
// Load ALL modules from a directory in the constructor.
// Function handles as nint FIELDS, not Dictionary<string, nint>.
public sealed class CudaKernels : IDisposable
{
    private readonly nint _groupNormModule;
    private readonly nint _groupNormF16Func;
    private readonly nint _groupNormSiluFusedFunc;
    // ... all kernel families

    public CudaKernels(string ptxDir)
    {
        _groupNormModule = CudaModule.LoadFromFile(
            Path.Combine(ptxDir, "group_norm.ptx"));
        _groupNormF16Func = _groupNormModule.GetFunction("group_norm_f16");
        _groupNormSiluFusedFunc = _groupNormModule.GetFunction(
            "group_norm_silu_fused_f16");
        // ...
    }
}
```

### Common PTX Pitfalls
- Bank conflicts in shared memory (use padding or swizzle)
- Not synchronizing threads before reading shared memory (`bar.sync`)
- Wrong grid/block dimensions causing out-of-bounds memory access
- Register spilling killing performance -- reduce live variables
- Not handling tensor dimensions that aren't multiples of tile size

## SPIR-V Kernel Workflow (extends dotLLM's approach to Vulkan)

1. **Understand the math** -- same as CPU and PTX
2. **Write the GLSL compute shader** -- `.comp.glsl` file in `native/vulkan/shaders/`
3. **Compile to SPIR-V** -- `glslangValidator --target-env vulkan1.2 -S comp -o kernel.spv kernel.comp.glsl`
4. **Ship as content file** -- SPIR-V file in a directory (same as PTX)
5. **Create compute pipeline** -- `vkCreateShaderModule` + `vkCreateComputePipelines` (mirrors PTX `CudaModule.LoadFromFile()` + `GetFunction()`)
6. **Cache the pipeline** -- `Dictionary<string, nint>` keyed by kernel name (Vulkan pipelines are more complex than CUDA function handles, so dictionary is acceptable here)
7. **Write the dispatch wrapper** -- bind descriptor sets, set push constants, `vkCmdDispatch`
8. **Validate** -- output must match CUDA kernel within FP16 tolerance

### SPIR-V Kernel Standards

```glsl
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

layout(local_size_x = 256) in;

// Storage buffers for tensor data
layout(set = 0, binding = 0) buffer OutputBuf { float16_t output[]; };
layout(set = 0, binding = 1) buffer InputBuf  { float16_t input[];  };

// Push constants for scalar parameters (equivalent to stackalloc kernel args)
layout(push_constant) uniform Params {
    int channels;
    int groups;
    float eps;
};
```

- Use `subgroupAdd`, `subgroupShuffle` for reductions (replaces CUDA warp shuffles)
- Subgroup size varies: 32 (NVIDIA), 64 (AMD), 8-32 (Intel) -- kernels must handle variable widths
- Use `shared` memory (`groupshared`) for inter-workgroup communication
- `barrier()` for workgroup synchronization (equivalent to `bar.sync` in PTX)
- Push constants (up to 128 bytes) for scalar params -- equivalent to dotLLM's `stackalloc void*[]`
- Storage buffers for all tensor data -- bound via descriptor sets
- Target Vulkan 1.2 for subgroup operation guarantees

### SPIR-V Dispatch Wrapper

```csharp
void DispatchGroupNormSiLU(nint outputBuffer, nint inputBuffer, nint weightBuffer,
    nint biasBuffer, int channels, int groups, float eps)
{
    // Bind descriptor sets (tensor buffers)
    VulkanApi.vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        pipelineLayout, 0, 1, &descriptorSet, 0, null);

    // Set push constants (scalar params)
    GroupNormParams pushConstants = new(channels, groups, eps);
    VulkanApi.vkCmdPushConstants(commandBuffer, pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint)sizeof(GroupNormParams), &pushConstants);

    // Dispatch compute shader
    VulkanApi.vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);
}
```

### Key SPIR-V vs PTX Differences
- No cuBLAS equivalent -- matrix multiply must be hand-written as tiled SPIR-V compute shader
- Subgroup size is runtime-queried, not compile-time fixed -- use `gl_SubgroupSize` in shader
- Memory barriers are more explicit -- `memoryBarrierShared()` + `barrier()` vs PTX `bar.sync`
- Vulkan requires explicit synchronization (pipeline barriers) between dispatches
- Push constants have 128-byte limit -- use storage buffer for large parameter structures

### Common SPIR-V Pitfalls
- Assuming fixed subgroup size (32) -- AMD uses 64, Intel varies
- Missing `memoryBarrierShared()` before reading shared memory written by other invocations
- Exceeding push constant size limit (128 bytes) -- use UBO or storage buffer instead
- Not accounting for Vulkan's explicit memory model -- barriers between read-after-write
- Descriptor set pool exhaustion -- pre-allocate enough sets at startup

## Kernel Validation

Every kernel must be validated. Follow dotLLM's SIMD-vs-scalar test pattern -- if SIMD and scalar disagree, the SIMD implementation has a bug. Extend this to GPU kernels:

| Kernel Type | Reference | Tolerance |
|---|---|---|
| CPU FP32 (SIMD) | CPU scalar (same inputs) | Within 1e-5 |
| CPU FP16 (SIMD) | CPU scalar (same inputs) | Within 1e-3 |
| CUDA FP16 (PTX) | CPU kernel (same inputs) | Within 1e-3 |
| CUDA cuBLAS | CPU matmul (same inputs) | Within 1e-3 |
| Vulkan FP16 (SPIR-V) | CUDA PTX (same inputs) | Within 1e-3 |
| Vulkan tiled GEMM | CUDA cuBLAS (same inputs) | Within 1e-3 |
| Fused kernel | Sequential unfused ops (same backend) | Within 1e-3 |

## Related Docs
- `docs/Research/DOTLLM_ARCHITECTURE.md` -- dotLLM's kernel infrastructure patterns
- `docs/Research/CUDA_DRIVER_API.md` -- CUDA P/Invoke signatures
- `docs/Research/VULKAN_COMPUTE_API.md` -- Vulkan compute API signatures
- `docs/Research/SPIRV_COMPUTE_SHADERS.md` -- SPIR-V compute shader patterns
- `docs/Design/FILE_STRUCTURE.md` -- where kernel files go
- `docs/Agents/BENCHMARK.md` -- how to benchmark kernel performance
