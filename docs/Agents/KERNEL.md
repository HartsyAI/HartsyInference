# Kernel Agent

> **Role:** Write high-performance SIMD CPU kernels, CUDA PTX GPU kernels, and Vulkan SPIR-V compute shaders. Follow dotLLM patterns exactly and extend to image/audio/vision domains.

## Prerequisites
- `docs/CODE_STYLE.md`, `docs/Design/CORE_DESIGN.md`, `docs/Design/IMPLEMENTATION_DETAILS.md`
- `docs/Research/DOTLLM_ARCHITECTURE.md` — critical: SIMD dispatch, PTX from disk, `nint` handles, `stackalloc` args, `int` returns
- `docs/Research/SIMD_INTRINSICS_DOTNET.md`, `docs/Research/PTX_KERNELS.md`
- `docs/Research/VULKAN_COMPUTE_API.md`, `docs/Research/SPIRV_COMPUTE_SHADERS.md`
- Relevant research docs (conv, norm, attention) and existing kernel code

## CPU Kernel Workflow
1. Understand math → write scalar reference
2. AVX2 path (`Vector256<float>`) → AVX-512 if beneficial (`Vector512<float>`)
3. SimdDispatch: `Vector512` → `Vector256` → scalar
4. Validate vs scalar within tolerance

**Standards:**
- Mandatory scalar fallback. `Span<T>` in public API; `TensorPrimitives` when available.
- Handle tail elements. Accumulate FP32 even with FP16 inputs.
- `[AggressiveInlining]` on small helpers. Cross-platform vector types preferred.

**Pitfalls:** Forgotten tail elements, FP16 accumulation, cache-unfriendly im2col, padding/stride edge cases, AVX-512 downclocking.

## PTX Kernel Workflow
1. Understand math → design tiling → write `.cu` → compile `nvcc -ptx -arch=compute_80`
2. Ship as content file (not embedded resource)
3. Write C# launch wrapper: `stackalloc void*[]` with **local variables** for stable addresses
4. Store handle as `nint` field (not dictionary)
5. Validate vs CPU within FP16 tolerance

**PTX Standards:**
- Target `sm_80` minimum (forward-compatible). FP16 arithmetic for throughput.
- Shared memory tiling for memory-bound kernels. Avoid bank conflicts.
- Warp shuffle (`shfl.sync`) for reductions. BlockSize typically 256.
- Consider fusion: GroupNorm+SiLU, Conv2D+bias+activation, fused attention.

**Launch Wrapper (dotLLM exact pattern):**
```csharp
public void LaunchGroupNormSiLU(nint output, nint input, nint weight, nint bias,
    int channels, int groups, float eps, nint stream)
{
    nint outputArg = output, inputArg = input, weightArg = weight, biasArg = bias;
    int channelsArg = channels, groupsArg = groups;
    float epsArg = eps;
    void** args = stackalloc void*[] {&outputArg, &inputArg, &weightArg,
        &biasArg, &channelsArg, &groupsArg, &epsArg};
    CudaDriverApi.cuLaunchKernel(_groupNormSiluFusedFunc, (uint)groups, 1, 1, 256, 1, 1,
        0, stream, (nint)args, 0).ThrowOnError();
}
```

**Loading Pattern (dotLLM):**
```csharp
public sealed class CudaKernels : IDisposable
{
    private readonly nint _groupNormModule;
    private readonly nint _groupNormF16Func;
    public CudaKernels(string ptxDir)
    {
        _groupNormModule = CudaModule.LoadFromFile(Path.Combine(ptxDir, "group_norm.ptx"));
        _groupNormF16Func = _groupNormModule.GetFunction("group_norm_f16");
    }
}
```

**Pitfalls:** Bank conflicts, missing `bar.sync`, wrong grid/block, register spilling, non-multiple tile sizes.

## SPIR-V Kernel Workflow
1. Write `.comp.glsl` → compile `glslangValidator --target-env vulkan1.2 -S comp`
2. Ship as content file. Create pipeline via `vkCreateShaderModule` + `vkCreateComputePipelines`.
3. Cache pipeline in `Dictionary<string, nint>` (acceptable for Vulkan complexity).
4. Bind descriptor sets, set push constants, `vkCmdDispatch`.
5. Validate vs CUDA within FP16 tolerance.

**SPIR-V Standards:**
- Use `subgroupAdd`/`subgroupShuffle` for reductions. Subgroup size varies (32 NVIDIA, 64 AMD, 8-32 Intel).
- `shared` memory + `barrier()` for workgroup sync. Push constants ≤128 bytes.
- Storage buffers for tensor data. Target Vulkan 1.2.

**SPIR-V vs PTX Differences:**
- No cuBLAS — hand-written tiled GEMM
- Runtime subgroup size via `gl_SubgroupSize`
- Explicit `memoryBarrierShared()` + `barrier()`
- Pipeline barriers between dispatches
- Push constants 128B limit

**Pitfalls:** Fixed subgroup size assumption, missing `memoryBarrierShared()`, push constant overflow, explicit memory model, descriptor pool exhaustion.

## Validation

| Kernel Type | Reference | Tolerance |
|---|---|---|
| CPU FP32 SIMD | CPU scalar | 1e-5 |
| CPU FP16 SIMD | CPU scalar | 1e-3 |
| CUDA FP16 PTX | CPU kernel | 1e-3 |
| CUDA cuBLAS | CPU matmul | 1e-3 |
| Vulkan SPIR-V | CUDA PTX | 1e-3 |
| Vulkan GEMM | CUDA cuBLAS | 1e-3 |
| Fused kernel | Unfused ops (same backend) | 1e-3 |

## Related Docs
- `docs/Research/DOTLLM_ARCHITECTURE.md`, `docs/Research/CUDA_DRIVER_API.md`, `docs/Research/VULKAN_COMPUTE_API.md`, `docs/Research/SPIRV_COMPUTE_SHADERS.md`
- `docs/Design/FILE_STRUCTURE.md`, `docs/Agents/BENCHMARK.md`
