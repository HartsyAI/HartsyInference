# CUDA Performance — Research & Optimization Roadmap

This document tracks GPU performance findings, bottlenecks, and the optimization roadmap for closing the gap with ComfyUI/PyTorch diffusion inference.

---

## Current State (2026-04-25, after Phase 2)

### Working GPU Generation

| Resolution | Steps | Time | Per-Step | Backend | Cache Hit Rate |
|---|---|---|---|---|---|
| 256x256 | 10 | ~49s | ~4.2s | CudaBackend (Phase 2) | 44% |
| 1024x1024 | 20 | ~19min | ~53s (steady) | CudaBackend (Phase 2) | 82.6% |

### Previous Baselines

| Phase | Resolution | Per-Step | Cache Hit Rate |
|---|---|---|---|
| Phase 0 (weight cache only) | 1024x1024 | ~100s | N/A |
| Phase 1 (activation cache) | 1024x1024 | ~93s | 77% |
| **Phase 2 (GPU kernels + async exec)** | **1024x1024** | **~53s** | **82.6%** |

### ComfyUI Reference (RTX 3060 12GB, SDXL, same model)

| Resolution | Steps | Time | Per-Step |
|---|---|---|---|
| 1024x1024 | 20 | ~60s | ~3s |

**Gap: ~18x slower than ComfyUI for 1024x1024 SDXL** (down from ~33x at Phase 0).

### Phase 2 Changes Summary

1. **Removed per-op `cuStreamSynchronize`** from all 15 CudaBackend ops — CUDA stream ordering guarantees correctness
2. **Switched `FreeDevice` to `cuMemFreeAsync`** — stream-ordered memory cleanup, no sync needed
3. **Added 4 GPU PTX kernels** replacing CPU-side operations:
   - `transpose_2d_f32` — batched 2D transpose `[B,D1,D2] → [B,D2,D1]`
   - `permute_0213_f32` — multi-head attention reshape `[B,S,H,D] ↔ [B,H,S,D]`
   - `geglu_f32` — gated activation split + GELU (last-dimension aware)
   - `broadcast_add_f32` — broadcast add `[B,C] + [B,C,H,W]` in-place
4. **UNet weight eviction** before VAE decode — frees ~7.5GB VRAM for large im2col buffers
5. **OOM retry** in `CudaMemory.Allocate` — syncs stream to flush pending FreeAsync, then retries

---

## Architecture: Lazy-Sync Transfer Pattern

The `CudaBackend` uses a **lazy-sync** pattern with two GPU caches:

```
For every IBackend op (Linear, Conv2D, GroupNorm, etc.):
  1. CopyToDevice(input)     — weight cache hit / activation cache hit / fresh H2D
  2. CopyToDevice(weights)   — weight cache hit (permanent)
  3. AllocateDevice(output)  — alloc GPU output buffer
  4. Run kernel              — cuBLAS SGEMM / PTX kernel
  5. CacheActivation(output) — store GPU ptr + set lazy-sync callbacks on Tensor
  6. FreeDevice(inputs)      — FreeAsync for non-cached pointers (stream-ordered)

No per-op cuStreamSynchronize. CUDA stream ordering guarantees kernel B launched
after kernel A on the same stream won't begin until A completes. Sync only needed:
  - Before D2H copy (lazy-sync callback handles this)
  - At pipeline stage boundaries (UNet → VAE weight eviction)
  - In OOM retry (flush pending FreeAsync to reclaim memory)
```

When subsequent ops consume the output tensor, `CopyToDevice` finds it in the activation cache (zero-copy reuse). When CPU code accesses `DataPointer`, the lazy-sync callback fires (D2H + GPU free). When `Dispose()` is called without CPU access, GPU memory is freed directly (zero D2H).

### Evolution

1. **Auto-transfer (initial)**: Every op copies H2D → kernel → D2H. Correct but slow.
2. **Weight cache (Phase 0)**: Weights preloaded to GPU. Eliminates ~3,360 weight H2D per step.
3. **Activation cache (Phase 1)**: Op outputs stay on GPU between consecutive ops. Eliminates ~2,372 activation H2D+D2H per step. But ~1,673 misses remain from CPU-side operations.
4. **GPU kernels + async exec (Phase 2, current)**: Removed per-op Sync, added GPU reshape/GEGLU/BroadcastAdd kernels, FreeAsync for memory cleanup. Eliminates most CPU-side round-trips. 82.6% cache hit rate, ~53s/step (down from ~93s).

### GPU Weight Cache

Weights are preloaded to GPU memory and cached by `Tensor` object reference:

```csharp
backend.PreloadWeights(unet.EnumerateWeights());  // ~10GB UNet weights → GPU
backend.PreloadWeights(vae.EnumerateWeights());    // ~332MB VAE weights → GPU
```

### GPU Activation Cache

Op outputs are cached on GPU via `GpuTransferHelper.CacheActivation()`:

```csharp
// After kernel completes:
GpuTransferHelper.CacheActivation(output, pOut, outBytes);
// Sets tensor._gpuSyncCallback (D2H + free) and tensor._gpuDisposeCallback (free only)
```

**Combined impact (Phase 1)**: 77% cache hit rate at 1024x1024 (~5,732 hits, ~1,673 misses per step). ~7% step time reduction (93s vs 100s).

**Combined impact (Phase 2)**: 82.6% cache hit rate at 1024x1024 (~7,300 hits, ~1,500 misses per step). 43% step time reduction (53s vs 93s).

---

## Bottleneck Analysis

### Per-Step Breakdown (1024x1024 SDXL, Phase 2)

| Category | Phase 1 | Phase 2 | Status |
|---|---|---|---|
| Weight cache hits | ~3,360 | ~3,360 | Eliminated (GPU pointer lookup) |
| Activation cache hits | ~2,372 | ~7,300 | Major improvement via GPU kernels |
| Activation cache misses (H2D) | ~1,673 | ~1,500 | Reduced — some remain from CFG/scheduler/batch ops |
| Lazy D2H syncs | ~1,673 | ~1,500 | Reduced — remaining from pipeline-level CPU ops |
| cuStreamSynchronize per step | ~4,045 | ~1-2 | Eliminated per-op sync (only at stage boundaries) |
| GPU kernel launches | ~4,045 | ~4,300+ | Slightly more (new GPU kernels replace CPU ops) |

### Where GPU Chains Now Work (Phase 2)

**ResNet blocks** — fully GPU-resident chain:
```
GroupNorm → SiLU → Conv2D → BroadcastAdd(timestep) → GroupNorm → SiLU → Conv2D → Add(residual)
  All on GPU, no CPU round-trips. BroadcastAdd replaces CPU AddTimestepEmbedding.
```

**Attention blocks** — fully GPU-resident chain:
```
LayerNorm → Linear(Q,K,V) → Permute0213(Q,K,V) → SDPA → Permute0213(out) → Linear → Add
  Permute0213 replaces CPU ReshapeToMultiHead/ReshapeFromMultiHead.
```

**FeedForward blocks** — fully GPU-resident chain:
```
LayerNorm → Linear → GeGlu → Linear → Add
  GeGlu replaces CPU ApplyGeGlu.
```

### Remaining Bottlenecks (Phase 2)

1. **~1,500 cache misses/step** — from CFG step, scheduler step, batch slicing (all small tensors, ~1 call/step each)
2. **Kernel launch overhead** — ~4,300 individual kernel launches per step (no fusion yet)
3. **FP32 compute** — not using Tensor Cores (FP16 would give ~2x throughput)
4. **Im2col temporary buffers** — large GPU memory allocations per Conv2D (no memory pooling)
5. **No cuDNN** — manual im2col + SGEMM instead of optimized Winograd for 3x3 convolutions

ComfyUI/PyTorch achieve ~3s/step because they combine async execution + FP16 + cuDNN + memory pooling + fused kernels.

---

## Optimization Roadmap

### Phase A: GPU-Resident Activations — COMPLETE

**Status**: Implemented via lazy-sync activation cache. Zero model code changes.

**Implementation**:
- `GpuTransferHelper.CacheActivation()` stores GPU pointer and sets `_gpuSyncCallback` / `_gpuDisposeCallback` on the Tensor
- `Tensor.DataPointer` calls `EnsureCpuData()` which triggers lazy D2H if GPU data is cached
- `CopyToDevice()` checks activation cache after weight cache (3-tier lookup)
- All 15 `CudaBackend` ops modified: `CacheActivation(output)` instead of `CopyToHost(output)`
- `InternalsVisibleTo("HartsyInference.Cuda")` on Core csproj for callback access

**Results**: 77% cache hit rate, ~7% step time reduction (93s vs 100s). Modest because per-op Sync still dominated.

### Phase A2+A3: GPU Kernels + Remove Per-Op Sync — COMPLETE

**Status**: Implemented. 4 new PTX kernels, per-op Sync removed, FreeAsync for memory cleanup.

**Implementation**:
- Removed `Sync()` from all 15 CudaBackend ops — CUDA stream ordering guarantees correctness
- Switched `FreeDevice` to `CudaMemory.FreeAsync` (stream-ordered cleanup)
- Added `GpuTransferHelper.SetStream(nint)` for stream handle propagation
- New PTX kernels: `transpose_2d_f32`, `permute_0213_f32`, `geglu_f32`, `broadcast_add_f32`
- New `IBackend` methods: `Transpose2D`, `Permute0213`, `GeGlu`, `BroadcastAdd`, `Sync`, `FreeWeights`
- Model code updated: `CrossAttentionBlock`, `TransformerSubBlock`, `FeedForwardBlock`, `UNetResNetBlock`, `VaeAttention` all use backend ops instead of CPU-side methods
- UNet weights evicted before VAE decode to fit 1024x1024 in 12GB VRAM
- OOM retry in `CudaMemory.Allocate`: sync stream on `CUDA_ERROR_OUT_OF_MEMORY`, then retry

**Bugs found and fixed during Phase 2**:
- **GEGLU flat-split bug**: Kernel split input at flat midpoint instead of along last dimension. Fixed by decomposing output index into (outerIdx, d) and computing correct per-row offsets. See `PHASE_3_DEVIATIONS.md` #16.
- **BroadcastAdd in-place caching**: Old `_gpuSyncCallback` freed the GPU pointer being re-cached. Fixed by clearing callbacks before `CacheActivation`. See `PHASE_3_DEVIATIONS.md` #17.
- **OOM during VAE decode**: `FreeAsync` deferred frees weren't reclaimed for new allocations. Fixed with weight eviction + OOM retry. See `PHASE_3_DEVIATIONS.md` #18.

**Results**: 82.6% cache hit rate, 43% step time reduction (53s vs 93s at 1024x1024). Output verified visually correct.

### Phase B: Kernel Fusion (Medium Impact)

**Goal**: Reduce kernel launch overhead and memory bandwidth by fusing adjacent ops.

**Priority Fusions**:

| Fused Kernel | Ops Combined | Benefit |
|---|---|---|
| GroupNorm + SiLU | GroupNorm → SiLU | 1 read + 1 write vs 2 reads + 2 writes |
| Conv2D + Bias + SiLU | Conv2D → BiasAdd → SiLU | Eliminate 2 intermediate tensors |
| Linear + Bias | SGEMM → BiasAdd | Already done (cuBLAS + PTX) |
| Residual Add | Add two tensors in-place | Eliminate output allocation |
| Scale + Add (scheduler step) | Scheduler arithmetic | Reduce 3 ops to 1 |

**Expected Impact**: ~1.5-2x additional speedup from reduced memory bandwidth.

### Phase C: FP16 Inference (Medium Impact)

**Goal**: Run UNet in FP16 (half precision) for ~2x throughput on Tensor Cores.

**Approach**:
- Keep weights in FP16 (already stored as FP16 in safetensors, currently cast to FP32)
- Use cuBLAS HGEMM instead of SGEMM
- FP32 accumulation for numerical stability (cuBLAS default)
- PTX kernels use `f16x2` packed arithmetic
- GroupNorm/LayerNorm: FP16 I/O with FP32 internal accumulation

**Expected Impact**: ~1.5-2x speedup for compute-bound ops (GEMM, attention).

### Phase D: Memory Pool + Async Transfers (Lower Impact)

**Goal**: Reduce GPU malloc/free overhead and overlap compute with transfers.

**Approach**:
- `cuMemPool` for activation memory (async alloc/free, pool reuse)
- `cuMemcpyHtoDAsync` / `cuMemcpyDtoHAsync` on the compute stream
- Pinned host memory (`cuMemAllocHost`) for faster H2D/D2H when needed
- Double-buffering for overlapped execution

### Phase E: cuDNN Conv2D (Optional)

**Goal**: Use cuDNN's optimized Conv2D (Winograd, implicit GEMM) instead of manual im2col + SGEMM.

**Trade-off**: Adds cuDNN dependency but provides 2-3x Conv2D speedup for 3x3 kernels via Winograd. The current im2col approach allocates large temporary buffers (~300MB for 512ch × 1024×1024) and is memory-bandwidth-limited.

---

## Bugs Found and Fixed

### Integer Overflow in Im2Col (2026-04-24)

**Problem**: At 1024x1024 with 512 channels, the im2col total element count exceeds `uint32` max:
```
512 × 3 × 3 × 128 × 128 = 301,989,888 (fits u32)
512 × 3 × 3 × 1024 × 1024 (intermediate products overflow)
```

The im2col buffer allocation in `CudaBackend.cs` used `int` arithmetic: `colRows * colCols * sizeof(float)` overflowed to a negative value, causing `CUDA_ERROR_ILLEGAL_ADDRESS`.

**Fix**: Cast to `long` before multiplication:
```csharp
// CudaBackend.cs — 3 locations
colBuf = CudaMemory.Allocate((nuint)((long)colRows * colCols * sizeof(float)));
pInput + (ulong)((long)b * inCh * inH * inW * sizeof(float));
```

```csharp
// CudaKernels.cs — 2 locations (LaunchIm2Col, LaunchUpsampleNearest2D)
long totalElements = (long)channels * kH * kW * outH * outW;
```

```ptx
// spatial_f32.ptx — im2col kernel
// Widened thread ID, total element count, bounds check, and decomposition to 64-bit
// using cvt.u64.u32, mul.lo.u64, setp.ge.u64, rem.u64, div.u64
```

**Lesson**: Any product of spatial dimensions × channels × kernel size can exceed `int`/`uint` for 1024+ resolution. Always use `long` for buffer size calculations and 64-bit PTX arithmetic for thread indexing in spatial kernels.

### GPU Weight Cache Eviction (2026-04-24)

**Problem**: `SdxlPipeline.cs` called `EvictBackendCache("CLIP")` after CLIP encoding, which invoked `GpuTransferHelper.FreeAllCached()` — destroying ALL preloaded UNet+VAE weights.

**Fix**: Removed `EvictBackendCache` calls. These were vestigial from an earlier design with no persistent cache.

### Linear Bias CPU Access (2026-04-24)

**Problem**: `CudaBackend.Linear` accessed `bias.DataPointer` directly on CPU, bypassing the GPU cache. After CPU weight disposal, this caused `ObjectDisposedException`.

**Fix**: Routed bias through `GpuTransferHelper.CopyToDevice` (cache-aware) and used the `col2bias_add` PTX kernel on GPU.

### VaeAttention CPU Weight Access (2026-04-24)

**Problem**: `VaeAttention.ProjectLinear` transposed weight matrices and added bias using direct CPU pointer access (`TransposeMatrix`, `AddBiasBroadcast`). After GPU preload + CPU disposal, accessing disposed weight `DataPointer` crashed.

**Fix**: Replaced manual CPU transpose + matmul + bias with `backend.Linear`, which handles weight transpose (cuBLAS `CUBLAS_OP_T`) and bias addition via GPU-cached tensors.

**Lesson**: After implementing GPU weight preloading, ALL weight tensor access must go through the backend (which uses the GPU cache). Any model code that directly accesses `weight.DataPointer` will crash after CPU disposal. Audit all model code for direct `DataPointer` access on weight tensors.

### CUDA Non-Blocking Stream Race Condition (2026-04-23)

**Problem**: `CudaStream(nonBlocking: true)` creates a stream that does NOT synchronize with the NULL stream. `GpuTransferHelper` used synchronous `cuMemcpyHtoD` (operates on NULL stream). With pageable host memory, DMA may still be in-progress when a kernel launches on the non-blocking stream.

**Fix**: Changed to `CudaStream(nonBlocking: false)`.

**Future**: Switch to `cuMemcpyHtoDAsync`/`cuMemcpyDtoHAsync` on the compute stream for proper ordering with non-blocking streams.

---

## Files Modified for GPU Optimization

### Phase 0: Weight Cache (2026-04-24)

| File | Changes |
|---|---|
| `CudaBackend.cs` | `PreloadWeights()`, `FreePreloadedWeights()`, `GetGpuCacheStats()`, Linear bias GPU migration, integer overflow fixes |
| `GpuTransferHelper.cs` | Weight cache (`Dictionary<Tensor, ulong>`), `PreloadWeight()`, `FreeAllCached()`, modified `CopyToDevice`/`FreeDevice` |
| `CudaKernels.cs` | `long totalElements` in `LaunchIm2Col`, `LaunchUpsampleNearest2D` |
| `spatial_f32.ptx` | 64-bit thread ID, bounds check, decomposition in im2col kernel |
| `SdxlPipeline.cs` | Removed `EvictBackendCache` calls |
| `VaeAttention.cs` | Replaced CPU `TransposeMatrix`+`AddBiasBroadcast` with `backend.Linear` |
| 10 model files | Added `EnumerateWeights()` to UNet, blocks, VAE, attention, embeddings |

### Phase 1: Activation Cache (2026-04-24)

| File | Changes |
|---|---|
| `Tensor.cs` | Added `_gpuSyncCallback`, `_gpuDisposeCallback` fields, `EnsureCpuData()` method, modified `DataPointer`/`AsSpan`/`AsReadOnlySpan`/`AsRef`/`Dispose`/finalizer |
| `HartsyInference.Core.csproj` | Added `InternalsVisibleTo Include="HartsyInference.Cuda"` |
| `GpuTransferHelper.cs` | Added activation cache (`Dictionary<Tensor, (ulong, nuint)>`), `CacheActivation()`, modified `CopyToDevice`/`FreeAllCached` |
| `CudaBackend.cs` | All 15 ops: `CacheActivation(output)` instead of `CopyToHost(output)`, `cachedOutput` flag in finally blocks |

### Phase 2: GPU Kernels + Async Execution (2026-04-25)

| File | Changes |
|---|---|
| `CudaBackend.cs` | Removed per-op `Sync()`, added `Transpose2D`/`Permute0213`/`GeGlu`/`BroadcastAdd` ops, `FreeWeights()`, in-place callback clearing |
| `GpuTransferHelper.cs` | `SetStream()`, `SyncStream()`, `FreeWeights(IEnumerable<Tensor>)`, `FreeDevice` → `FreeAsync` |
| `CudaMemory.cs` | OOM retry: sync stream on `CUDA_ERROR_OUT_OF_MEMORY`, then retry |
| `CudaKernels.cs` | Load 4 new PTX modules, launch methods for transpose/permute/geglu/broadcast_add |
| `IBackend.cs` | Added `Transpose2D`, `Permute0213`, `GeGlu`, `BroadcastAdd`, `Sync()`, `FreeWeights()` (default interface methods) |
| `CpuBackend.cs` | Implemented new ops (moved CPU code from model classes) |
| `transpose_f32.ptx` | New — batched 2D transpose kernel |
| `permute_0213_f32.ptx` | New — multi-head attention reshape kernel |
| `geglu_f32.ptx` | New — gated activation with last-dimension split |
| `broadcast_add_f32.ptx` | New — broadcast add `[B,C] + [B,C,H,W]` |
| `CrossAttentionBlock.cs` | Replaced CPU `ReshapeSpatialToSequence`/`ReshapeSequenceToSpatial` with `backend.Transpose2D` |
| `TransformerSubBlock.cs` | Replaced CPU `ReshapeToMultiHead`/`ReshapeFromMultiHead` with `backend.Permute0213` |
| `FeedForwardBlock.cs` | Replaced CPU `ApplyGeGlu` with `backend.GeGlu` |
| `UNetResNetBlock.cs` | Replaced CPU `AddTimestepEmbedding` with `backend.BroadcastAdd` |
| `VaeAttention.cs` | Replaced CPU `TransposeBCtoBC` with `backend.Transpose2D` |
| `SdxlPipeline.cs` | Added `Sync()` + `FreeWeights(unet)` before VAE decode |

---

## Key Architectural Insights

### Lazy-Sync Pattern

The lazy-sync pattern preserves correctness while progressively eliminating transfers:
- Same model code runs on CPU and GPU — zero model changes (Phase 1)
- Phase 2 moved some model code to backend ops, but the pattern is consistent: model code calls `IBackend`, backend handles GPU
- Activation cache transparently reuses GPU data between consecutive ops
- CPU-side operations trigger lazy sync automatically via `DataPointer`
- Bit-identical results within FP32 tolerance

### Stream-Ordered Memory (Phase 2)

With `cuMemFreeAsync`, GPU memory lifetime is managed by the stream:
- No per-op sync needed — CUDA stream ordering guarantees ops execute sequentially
- `FreeAsync` defers the actual free until the stream processes it
- **Gotcha**: Deferred frees mean memory isn't immediately available. At pipeline stage transitions, explicitly sync and free to reclaim VRAM
- **Gotcha**: In-place ops must clear old `_gpuSyncCallback`/`_gpuDisposeCallback` before re-caching (old callbacks close over the GPU pointer and would free it)

### Remaining Gap to ComfyUI/PyTorch (~18x)

ComfyUI/PyTorch achieve ~3s/step because:
1. ~~All tensors live on GPU~~ — Done (Phase 0+1)
2. ~~Ops chain without CPU round-trips~~ — Done (Phase 2 GPU kernels)
3. ~~Only sync at step boundaries~~ — Done (Phase 2 removed per-op Sync)
4. FP16 inference with Tensor Cores — Phase C (target ~2x)
5. cuDNN Winograd for 3x3 Conv2D — Phase E (~2-3x for conv ops)
6. Memory pooling eliminates alloc/free overhead — Phase D
7. Kernel fusion (GroupNorm+SiLU, Conv2D+Bias+SiLU) — Phase B (~1.5-2x)
8. Optimized attention (FlashAttention-style tiled SDPA) — future

---

## Native FP8 GEMM (Ada+ / SM 8.9+)

### Path overview
On Ampere (SM 8.0/8.6/8.7) the FP8 GEMM dispatch in [`CudaBackend.Linear`](../../src/HartsyInference.Cuda/CudaBackend.cs) casts FP8 weights to F16 once per call and runs `cublasGemmEx` in F16. The cast adds an extra kernel launch and a transient F16 buffer the size of the weight tensor; the GEMM itself runs on F16 tensor cores.

On Ada (SM 8.9: RTX 40-series, L40, L4) and Hopper (SM 9.0: H100 / GH200) the same operation can run as a native FP8 tensor-core GEMM via `cublasLtMatmul`, eliminating both the cast and the F16 staging buffer. Expected speedup vs the cast-then-F16 path is roughly **1.6× – 2×** for backbone Linear ops (weight-bound).

### Implementation
- [`CublasLtApi.cs`](../../src/HartsyInference.Cuda/CublasLtApi.cs) — P/Invoke for cuBLASLt: handle, matmul-desc, matrix-layout, preference, dispatch.
- [`Fp8GemmExecutor.cs`](../../src/HartsyInference.Cuda/Fp8GemmExecutor.cs) — single-handle wrapper. Allocates a 4 MiB workspace at construction (Hopper recommends 32 MiB; tune via constants if a future Hopper benchmark warrants it). `IsSupported` is `(smMajor == 8 && smMinor >= 9) || smMajor >= 9`.
- [`CudaBackend.Linear`](../../src/HartsyInference.Cuda/CudaBackend.cs) — gated dispatch: when `EnableNativeFp8Gemm == true` AND both operands are FP8 AND output is F16 AND `Fp8Executor.IsSupported`, dispatch via `Fp8Executor.Run`. Otherwise fall through to the existing cast-to-F16 path.

### Usage
```csharp
using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: "...");
if (backend.Fp8Executor.IsSupported)
{
    backend.EnableNativeFp8Gemm = true;
}
```
The `EnableNativeFp8Gemm` flag is opt-in because the path has not yet been end-to-end validated on Ada hardware in CI (the project's primary dev box is a 12 GB RTX 3060 / SM 8.6). Once an Ada GPU is added to CI, flip the default to `true` after benchmarking against the F16 fallback for at least Flux Dev FP8 + SD3.5 Medium.

### Per-tensor scale handling
ComfyUI fp8_scaled and BFL distilled checkpoints carry a single scalar `Fp8ScaleFactor` per weight tensor. The executor folds this into the cuBLAS `alpha` parameter — exact for per-tensor scale, no extra kernel launch. If a future checkpoint format ships per-row or per-column scales, wire them via `CUBLASLT_MATMUL_DESC_A_SCALE_POINTER` (constants already in `CublasLtApi.cs`).

### Validation
- [`Fp8GemmExecutorTests`](../../tests/HartsyInference.Cuda.Tests/Fp8GemmExecutorTests.cs) — gating tests confirming Ampere reports `IsSupported = false` and `Run` throws on unsupported hardware.
- Pending on Ada hardware: accuracy test vs F16 reference (target avg_err < 1e-3), end-to-end SSIM on Flux Dev FP8 vs the F16 fallback, peak-VRAM measurement.
