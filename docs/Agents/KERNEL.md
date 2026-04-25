# Kernel Agent

> Write high-performance SIMD CPU kernels, CUDA PTX GPU kernels, and Vulkan SPIR-V compute shaders.

## Extra Reading
- `docs/Design/IMPLEMENTATION_DETAILS.md`
- `docs/Research/DOTLLM_ARCHITECTURE.md` — SIMD dispatch, PTX from disk, `nint` handles, `stackalloc` args
- `docs/Research/SIMD_INTRINSICS_DOTNET.md`, `docs/Research/PTX_KERNELS.md`
- `docs/Research/VULKAN_COMPUTE_API.md`, `docs/Research/SPIRV_COMPUTE_SHADERS.md`
- Relevant research docs and existing kernel code

## CPU Kernel Workflow
1. Understand math → write scalar reference
2. AVX2 path (`Vector256<float>`) → AVX-512 if beneficial (`Vector512<float>`)
3. SimdDispatch: `Vector512` → `Vector256` → scalar
4. Validate vs scalar within tolerance

**Standards:** Mandatory scalar fallback. `Span<T>` in public API; `TensorPrimitives` when available. Handle tail elements. Accumulate FP32 even with FP16 inputs. `[AggressiveInlining]` on small helpers. Cross-platform vector types preferred.

**Pitfalls:** Forgotten tail elements, FP16 accumulation, cache-unfriendly im2col, padding/stride edge cases, AVX-512 downclocking.

## PTX Kernel Workflow
1. Understand math → design tiling → write `.cu` → compile `nvcc -ptx -arch=compute_80`
2. Ship as content file (not embedded resource)
3. Write C# launch wrapper (see `AGENTS.md` CUDA Launch Pattern)
4. Store handle as `nint` field (not dictionary)
5. Validate vs CPU within FP16 tolerance

**Standards:** Target `sm_80` minimum (forward-compatible). FP16 arithmetic for throughput. Shared memory tiling for memory-bound kernels. Avoid bank conflicts. Warp shuffle (`shfl.sync`) for reductions. BlockSize typically 256. Consider fusion: GroupNorm+SiLU, Conv2D+bias+activation, fused attention.

**Loading Pattern:**
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

**Pitfalls:** Bank conflicts, missing `bar.sync`, wrong grid/block, register spilling, non-multiple tile sizes. See "Known PTX Pitfalls" below for bugs that have bitten us.

## SPIR-V Kernel Workflow
1. Write `.comp.glsl` → compile `glslangValidator --target-env vulkan1.2 -S comp`
2. Ship as content file. Create pipeline via `vkCreateShaderModule` + `vkCreateComputePipelines`.
3. Cache pipeline in `Dictionary<string, nint>` (acceptable for Vulkan complexity).
4. Bind descriptor sets, set push constants, `vkCmdDispatch`.
5. Validate vs CUDA within FP16 tolerance.

**Standards:** `subgroupAdd`/`subgroupShuffle` for reductions. Subgroup size varies (32 NVIDIA, 64 AMD, 8-32 Intel). `shared` memory + `barrier()` for workgroup sync. Push constants ≤128 bytes. Storage buffers for tensor data. Target Vulkan 1.2.

**SPIR-V vs PTX:** No cuBLAS — hand-written tiled GEMM. Runtime subgroup size via `gl_SubgroupSize`. Explicit `memoryBarrierShared()` + `barrier()`. Pipeline barriers between dispatches. Push constants 128B limit.

**Pitfalls:** Fixed subgroup size assumption, missing `memoryBarrierShared()`, push constant overflow, descriptor pool exhaustion.

## GPU Performance Optimization

See `docs/Research/CUDA_PERFORMANCE.md` for the full optimization roadmap.

### Current State: Phase 2 Complete

The `CudaBackend` uses lazy-sync activation caching with GPU kernels for all major ops. Per-op `cuStreamSynchronize` has been removed. Weight cache + activation cache give **82.6% hit rate** (~7,300 hits, ~1,500 misses per SDXL step). Current: **~53s/step at 1024x1024** (down from ~93s Phase 1, ~100s Phase 0).

**GPU kernels added in Phase 2**: `transpose_2d_f32` (batched 2D transpose), `permute_0213_f32` (multi-head attention reshape), `geglu_f32` (gated activation), `broadcast_add_f32` (broadcast add with channel indexing).

### Priority Optimizations (Next)

1. **Kernel Fusion** (~1.5-2x): GroupNorm+SiLU, Conv2D+Bias+SiLU, Residual Add in-place. Reduces kernel launch overhead and memory bandwidth.
2. **FP16 Inference** (~1.5-2x): HGEMM, `f16x2` packed PTX arithmetic, FP32 accumulation. Unlocks Tensor Cores.
3. **Memory Pool** (moderate): `cuMemPool` for activation memory. Eliminates per-op alloc/free overhead.
4. **cuDNN Conv2D** (optional, ~2-3x for conv): Winograd 3x3, eliminates large im2col temp buffers.
5. **FlashAttention-style SDPA** (moderate): Tiled attention with online softmax, reduced memory bandwidth.

### Known PTX Pitfalls

These are bugs that have actually bitten us. Check for them first when writing new kernels.

- **64-bit indexing required** for spatial kernels at 1024+ resolution. Products of `channels × kernel × outH × outW` overflow `u32`. Use `cvt.u64.u32` + `mul.lo.u64` for thread ID and element count. See `PHASE_3_DEVIATIONS.md` #12.

- **Last-dimension split for gated activations (GEGLU/SwiGLU/GLU)**: When a kernel splits `[..., 2*D]` along the last dim, you CANNOT use flat indexing (`input[i]` and `input[i + totalOutput]`). Decompose each thread's output index: `outerIdx = i / D`, `d = i % D`, then `inputX = outerIdx * 2*D + d`, `inputGate = inputX + D`. A flat midpoint split is only correct for 1D/single-row tensors. Test with multi-row inputs like `[2, 2, 2*D]` to catch this. See `PHASE_3_DEVIATIONS.md` #16.

- **Non-blocking streams** cause race conditions with synchronous `cuMemcpyHtoD` (operates on NULL stream). Use blocking streams or async copies on the compute stream. See `PHASE_3_DEVIATIONS.md` #19.

- **`cuMemFreeAsync` deferred cleanup**: Memory freed with `FreeAsync` is NOT immediately available. At pipeline stage boundaries (e.g., freeing model weights before loading another model), must `cuStreamSynchronize` first to flush pending frees. Add OOM retry logic in allocators. See `PHASE_3_DEVIATIONS.md` #18.

- **In-place GPU ops and activation cache**: When a kernel modifies a tensor's GPU buffer in-place, the tensor's old `_gpuSyncCallback`/`_gpuDisposeCallback` (from prior `CacheActivation`) must be cleared to `null` BEFORE calling `CacheActivation` again. Otherwise the old callback fires and `FreeAsync`s the pointer being re-cached. See `PHASE_3_DEVIATIONS.md` #17.

- **Weight DataPointer access**: After GPU weight preloading + CPU disposal, model code must NEVER access `weight.DataPointer` directly. All weight access must go through `IBackend` ops (which use GPU cache). See `PHASE_3_DEVIATIONS.md` #14-15.

## Validation Tolerances

| Kernel Type | Reference | Tolerance |
|---|---|---|
| CPU FP32 SIMD | CPU scalar | 1e-5 |
| CPU FP16 SIMD | CPU scalar | 1e-3 |
| CUDA FP32 PTX | CPU kernel | 1e-5 |
| CUDA FP16 PTX | CPU kernel | 1e-3 |
| CUDA cuBLAS | CPU matmul | 1e-3 |
| Vulkan SPIR-V | CUDA PTX | 1e-3 |
| Fused kernel | Unfused ops (same backend) | 1e-3 |

### Measured GPU Accuracy

| Model | avg_err | max_err |
|---|---|---|
| SD1.5 UNet (GPU vs CPU) | 5.188E-007 | — |
| SDXL UNet (GPU vs CPU) | 5.510E-007 | 8.821E-006 |
