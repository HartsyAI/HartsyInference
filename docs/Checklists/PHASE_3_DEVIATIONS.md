# Phase 3 — Deviations from Design Plan

This document tracks every case where the C# implementation diverged from the reference Python (diffusers/PyTorch) behavior, how the bug was found, and how it was fixed. It serves as a debugging journal and a guide for future model ports.

---

## CPU Pipeline Deviations

### 1. BatchedMatMul — 2D Right Operand Silently Produced Zeros

**Design assumption**: `BatchedMatMul(a[B,M,K], b[K,N])` would correctly handle a 2D weight matrix broadcast across the batch dimension.

**Deviation**: `MatMulKernels.BatchedMatMul` read `N = b.Shape[2]`, but for a 2D tensor `[K, N]`, `Shape[2]` returns 0 (uninitialized `_dim2` in `TensorShape`). This made N=0, causing every matmul to produce an all-zeros output. The issue affected:
- **CLIP text encoder**: All 12 transformer layers were no-ops (residual passthrough only). The UNet received raw token+position embeddings instead of contextual text representations — the text prompt was effectively ignored.
- **VAE mid-block attention**: Attention projections produced zeros, making attention a residual passthrough.

**Fix**: Added 2D detection: `bool bIs2D = b.Shape.Rank == 2; long N = bIs2D ? b.Shape[1] : b.Shape[2];` and set `bSliceSize = 0` for 2D to reuse the same weight pointer across all batch slices.

**Impact**: This was the primary cause of the "brownish blob" output — without functional text encoding, the UNet had no semantic conditioning and produced essentially random noise predictions.

### 2. UNet Self-Attention — K/V Projected from Un-Normed Hidden

**Design assumption**: `TransformerSubBlock` would correctly apply LayerNorm before Q, K, and V projections for self-attention.

**Deviation**: The `TransformerSubBlock.Forward` method applied LayerNorm to produce `normed`, then projected Q from `normed` but K and V from the raw `context` parameter. For self-attention (where `context == hidden`), this meant K/V came from un-normed input while Q came from normed input. In diffusers, `attn1(norm_hidden_states)` passes the normed tensor for all of Q, K, V.

**Fix**: Added `ReferenceEquals(hidden, context)` check to detect self-attention and route K/V through the normed tensor: `Tensor kvSource = ReferenceEquals(hidden, context) ? normed : context;`

### 3. UNet Attention Head Count — Inverted Head/Dim Interpretation

**Design assumption**: `UNetConfig.AttentionHeadDim` values (8 for SD1.5) represent the per-head dimension, so `numHeads = channels / headDim`.

**Deviation (original)**: Originally passed 8 directly as numHeads — which was actually correct for diffusers semantics.

**Deviation (incorrect fix)**: Changed to `numHeads = channels / AttentionHeadDim[i]` = 320/8 = 40 heads with headDim=8. This was wrong — diffusers uses `attention_head_dim=8` to mean **8 attention heads** (when `num_attention_heads` is not specified). The confusing naming in diffusers led to the misinterpretation.

**How it was found**: Layer-by-layer binary comparison against Python reference tensors. The error was invisible at the pipeline level (images were "plausible but bad") but obvious when comparing per-layer outputs:
- `down_blocks.0.resnets.0`: avg_err=3.5e-7 (perfect)
- `down_blocks.0.attentions.0`: avg_err=0.127 (first divergence!)
- Errors compounded through all subsequent layers to avg_err=1.808 at mid_block

Running `dump_attn_sublayers.py` confirmed: Python uses 8 heads with head_dim=40, while C# used 40 heads with head_dim=8. This caused:
- Wrong attention scale: `1/sqrt(8)=0.354` vs correct `1/sqrt(40)=0.158` (2.24x too large)
- Wrong multi-head split pattern: 40 tiny 8-dim heads instead of 8 larger 40-dim heads
- Completely different attention distributions, causing ~56% signal dampening over 20 denoising steps

**Fix**: Renamed config property `AttentionHeadDim` → `NumAttentionHeads` for clarity. Changed UNet constructor to use the value directly: `numHeads = config.NumAttentionHeads[i]` instead of `outCh / config.AttentionHeadDim[i]`. Applied at all three sites (down blocks, mid block, up blocks).

**Result**: First attention block error dropped from avg_err=0.127 to avg_err=4.3e-5 (2,940x improvement). All layers now match Python within float32 accumulation tolerance. Pipeline produces coherent images.

**Lesson**: Diffusers' `attention_head_dim` parameter is confusingly named — it specifies **head count**, not head dimension, when `num_attention_heads` is not provided. Always verify multi-head attention shapes against the reference by printing `attn.heads` and checking the actual Q/K/V reshape dimensions.

### 4. VAE Attention — 3D Tensors Passed to 4D SDPA Kernel

**Design assumption**: The VAE attention layer could pass tensors directly to `ScaledDotProductAttention`.

**Deviation**: `VaeAttention` passed 3D tensors `[B, seqLen, C]` to the SDPA kernel which expects 4D `[B, H, S, D]`. With a 3D tensor, `Shape[3]` returned 0 (uninitialized `_dim3`), making the head dimension D=0. The attention kernel's inner loops iterated zero times, producing all-zeros output.

**Fix**: Added reshape to 4D before SDPA: `[B, seqLen, C]` → `[B, 1, seqLen, C]` (single-head attention), then reshape back after.

### 5. Timestep Embedding — Sin/Cos Order and Frequency Divisor

**Deviation (sin/cos order)**: SD1.5 diffusers uses `flip_sin_to_cos=True` (default), producing `[cos, sin]` layout. Our code had `[sin, cos]`. Since every ResNet block conditions on the timestep embedding, this corrupted all noise predictions.

**Deviation (frequency divisor)**: Diffusers uses `/ (half_dim - 1)` in `get_timestep_embedding()`. Our code used `/ halfDim`, causing the highest frequency component to be ~6% off. This ensures the frequency range spans exactly `[1, 1/10000]`.

**Fix**: Swapped to `[cos, sin]` order and changed divisor to `(halfDim - 1)`.

### 6. Euler Scheduler — Missing scale_model_input

**Deviation**: Diffusers' `EulerDiscreteScheduler.scale_model_input` divides the latent by `sqrt(sigma^2 + 1)` before each UNet call. Without this, the UNet receives inputs at the wrong scale.

**Fix**: Added `ScaleModelInput(stepIndex)` to `IScheduler` interface and implemented in `EulerDiscreteScheduler`.

### 7. Euler Step — Division by Zero at Final Timestep

**Deviation**: At the final timestep (t=0), sigma approaches 0, causing division by zero in `derivative = (sample - pred_x0) / sigma`. The algebraic simplification for epsilon prediction eliminates this: `derivative = model_output` (the division cancels).

**Fix**: Simplified the epsilon-prediction path to avoid division. Added sigma guard for v-prediction path.

---

## CUDA Backend Deviations

### 8. CLIP Text Encoder — Missing Final LayerNorm

**Before**: `Encode()` returned raw last transformer layer output without applying `final_layer_norm`.
**After**: `Encode()` applies `final_layer_norm` matching HuggingFace `CLIPTextTransformer.forward()`.

**Impact**: Without this, text embeddings had std ~5 instead of ~1, causing 5x amplified conditioning signals that produced abstract patterns instead of coherent images.

### 9. CUDA SDPA Softmax — PTX Kernel

**Previous deviation**: The softmax step used a CPU roundtrip (download scores → host softmax → upload). Replaced with pure-PTX numerically stable per-row softmax using shared memory reductions (3-pass: max → exp+sum → normalize). One block per row, blockDim=256. Uses `ex2.approx.f32` for exp and `rcp.approx.f32` for 1/sum.

### 10. CUDA Conv2D — Im2Col + cuBLAS SGEMM (No cuDNN)

Conv2D is implemented via im2col (PTX kernel) + cuBLAS SGEMM, rather than cuDNN. Temporary column buffer allocated per forward pass. For 1x1 convolutions, im2col is skipped and input is used directly. Avoids cuDNN dependency, keeping the project pure C# + CUDA Driver API + cuBLAS.

### 11. CUDA GroupNorm/LayerNorm — Three-Pass Kernels

Both normalization kernels use a three-pass approach (mean → variance → normalize+affine) with shared memory reductions. Simpler to implement correctly than online Welford single-pass. Performance impact minimal since not bottleneck vs GEMM.

### 12. Integer Overflow in Im2Col at 1024x1024

**Problem**: At 1024x1024 with 512 channels and 3x3 kernels, the im2col total element count exceeds `uint32` max (4,294,967,295). Three overflow sites in `CudaBackend.cs` (buffer allocation, batch offsets), two in `CudaKernels.cs` (launch grid calculation), and one in `spatial_f32.ptx` (thread ID computation used `mul.lo.u32`).

**Symptoms**: `CUDA_ERROR_ILLEGAL_ADDRESS` during Conv2D at 1024x1024. 256x256 and 512x512 worked fine (products fit in 32-bit).

**Fix**: Cast to `long` before multiplication in C#. Use `cvt.u64.u32` + `mul.lo.u64` + `setp.ge.u64` + `rem.u64`/`div.u64` in PTX for 64-bit thread indexing and bounds checks.

**Lesson**: Any product of `channels × kernelH × kernelW × outH × outW` can overflow `int`/`uint` at 1024+ resolution. Always use `long` for GPU buffer size calculations. PTX kernels handling spatial data must use 64-bit arithmetic for thread IDs and element counts.

### 13. GPU Weight Cache Eviction by Pipeline

**Problem**: `SdxlPipeline.cs` called `EvictBackendCache("CLIP")` after CLIP encoding and `EvictBackendCache("UNet")` after denoising. These called `GpuTransferHelper.FreeAllCached()`, destroying ALL preloaded weights between pipeline stages. First UNet step crashed with `ObjectDisposedException` when accessing a disposed weight tensor.

**Fix**: Removed both `EvictBackendCache` calls. They were vestigial from an earlier design where no persistent GPU cache existed.

**Lesson**: When adding GPU weight caching, audit all existing code paths that clear or evict GPU memory. Pipeline stage transitions must not destroy weights needed by subsequent stages.

### 14. CudaBackend.Linear CPU Bias Access

**Problem**: `CudaBackend.Linear` accessed `bias.DataPointer` directly on CPU (line 149) to copy bias values, bypassing the GPU weight cache. After preloading weights to GPU and disposing CPU copies, the disposed bias tensor's `DataPointer` was zero, causing a crash.

**Fix**: Routed bias through `GpuTransferHelper.CopyToDevice` (which checks the GPU cache before accessing `DataPointer`) and used the `col2bias_add` PTX kernel with `spatial=1` for GPU-side bias addition.

**Lesson**: After implementing GPU weight caching, ALL weight/bias access in backend ops must go through the cache-aware `CopyToDevice` path. Direct `DataPointer` access on weight tensors bypasses the cache and will crash after CPU disposal.

### 15. VaeAttention CPU-Side Weight Manipulation

**Problem**: `VaeAttention.ProjectLinear` performed weight transpose (`TransposeMatrix`) and bias addition (`AddBiasBroadcast`) using direct CPU pointer access. After GPU preload + CPU disposal, accessing disposed weight `DataPointer` crashed during VAE decode.

**Fix**: Replaced manual CPU transpose + matmul + bias with `backend.Linear`, which handles weight transpose (cuBLAS `CUBLAS_OP_T`) and bias addition entirely on GPU with cached weights.

**Lesson**: Model code must NEVER directly access `weight.DataPointer` for computation when GPU weight preloading is enabled. All weight operations must route through `IBackend` methods, which use the GPU cache. Audit all model code for direct `DataPointer` access on weight tensors — these are latent bugs when weights move to GPU.

---

## Troubleshooting Methodology

The following approach was developed during SD1.5 debugging and should be reused for all future model ports.

### Step 1: Build a Python Reference Pipeline

Create a Python script (`tests/python-reference/dump_reference_stats.py`) that runs the full pipeline with known inputs and saves:
- Initial noise tensor (binary)
- Text embeddings (binary)
- Per-step latent tensors (binary + JSON stats: mean, std, min, max)
- Final latent tensor (binary)

Use a **venv** to avoid system Python conflicts: `python -m venv tests/python-reference/.venv`

### Step 2: Run C# Pipeline with Python's Noise

Write a test that loads Python's saved initial noise and text embeddings, runs the C# pipeline, and compares per-step latent statistics. This eliminates RNG differences and isolates model/scheduler bugs.

### Step 3: Single Forward Pass Comparison

If per-step stats diverge, isolate a single UNet forward pass. Save Python's step-0 inputs and outputs, feed the same inputs to C#, compare element-wise. If this diverges, the bug is in the model (not the scheduler).

### Step 4: Layer-by-Layer Binary Comparison

Hook every layer in Python and save outputs. Step through C# one layer at a time, comparing each output. This pinpoints the **first divergent layer**:
```
time_embedding:              avg_err=3.4e-8  (PERFECT)
conv_in:                     avg_err=0.0     (PERFECT)
down_blocks.0.resnets.0:    avg_err=3.5e-7  (PERFECT)
down_blocks.0.attentions.0: avg_err=0.127   ← FIRST DIVERGENCE
```

### Step 5: Sub-Layer Decomposition

Once the divergent layer is identified, manually execute each sub-operation in Python and save intermediates. For a CrossAttentionBlock: GroupNorm, reshape, proj_in, LayerNorm, Q/K/V projections, multi-head reshape, attention logits, softmax, output projection, residuals, FFN, etc. Compare C# sub-operations against these to find the exact bug.

### Step 6: Fix and Verify

Re-run layer-by-layer comparison to confirm all layers match (avg_err < 1e-3, ideally < 1e-4). Then run full pipeline comparison for end-to-end correctness.

---

## Phase 2 — GPU Kernel & Async Execution Deviations

### 16. GEGLU Kernel — Flat Midpoint Split Instead of Last-Dimension Split

**Problem**: The GEGLU PTX kernel split the input tensor at the flat midpoint: `x = input[i]`, `gate = input[i + outputElements]`. For a tensor shaped `[B, S, 2*D]`, the correct split is along the **last dimension** — the first D elements of each row are x values, the next D are gate values.

**Symptoms**: 256x256 and 1024x1024 images were completely garbled (vertical banding, smeared colors). The output structure was recognizable as "something happened" but bore no resemblance to the prompt. The bug was invisible in per-op unit tests because those typically use 1D or single-row inputs where flat split equals last-dim split.

**How it was found**: Compared Phase 2 output images against known-good pre-Phase 2 output. The pre-Phase 2 images were correct; Phase 2 images were garbled. Systematic review of all new GPU kernel call sites identified GEGLU as the only kernel that could cause global corruption — it runs in every FeedForward block (~46 times per step).

**Root cause**: For output index `i` in a `[B, S, D]` output tensor:
- **Wrong (flat split)**: `x = input[i]`, `gate = input[i + B*S*D]`
- **Correct (last-dim split)**: decompose `i → (outerIdx, d)` where `outerIdx = i / D`, `d = i % D`, then `x = input[outerIdx * 2*D + d]`, `gate = input[outerIdx * 2*D + d + D]`

For a concrete example with shape `[1, 2, 6]` (B=1, S=2, D=3, input has 2*D=6 per row):
- Output index 3 maps to row 1, col 0. Correct: `x = input[1*6 + 0] = input[6]`. Wrong: `x = input[3]` (which is row 0, col 3 — a gate value from the wrong row).

**Fix**: Rewrote `geglu_f32.ptx` to accept `innerDim` (D) as a kernel parameter. The kernel decomposes each thread's output index into `(outerIdx, d)` via integer division/modulo, then computes correct input offsets. Updated `CudaKernels.LaunchGeGlu` to pass 4 args (output, input, innerDim, outputElements). Applied same fix to `CpuBackend.GeGlu`.

**Lesson**: When splitting a tensor along a specific dimension, you cannot use flat indexing. The split point depends on the stride of that dimension. For last-dim split of `[..., 2*D]`, each contiguous chunk of 2*D elements must be split at offset D — not at the global midpoint. **Always think in terms of the logical tensor layout, not flat memory offsets.** This trap is especially dangerous in PTX where there's no shape metadata — the kernel only sees raw pointers.

**Lesson for future models**: GEGLU/SwiGLU/GLU variants all require this same last-dimension split pattern. When porting any gated activation from Python to a GPU kernel, verify the split logic handles multi-dimensional inputs correctly. A simple test with shape `[2, 2, 2*D]` will catch this — if flat split gives the same result as last-dim split, the test input is too simple.

### 17. BroadcastAdd In-Place — Old GPU Callbacks Free Re-Cached Pointer

**Problem**: `BroadcastAdd` modifies the hidden tensor in-place on GPU (adds bias into the existing buffer). After the kernel runs, `CacheActivation(hidden, pHidden, bytes)` is called to update the activation cache. But `CacheActivation` internally accesses `tensor.DataPointer` to store the CPU-side reference, which triggers the **old** `_gpuSyncCallback` — that callback calls `FreeAsync` on the GPU pointer that was just modified in-place. The pointer is freed before it can be re-cached.

**Symptoms**: Intermittent corruption or crashes. The freed GPU memory might be reallocated for a different tensor, causing data from unrelated operations to appear in subsequent computations. Hard to reproduce because it depends on the GPU memory allocator's reuse pattern.

**Fix**: Clear `_gpuSyncCallback` and `_gpuDisposeCallback` to `null` before calling `CacheActivation` for any in-place GPU operation:
```csharp
// BroadcastAdd modifies hidden in-place. Clear old GPU callbacks before re-caching.
hidden._gpuSyncCallback = null;
hidden._gpuDisposeCallback = null;
GpuTransferHelper.CacheActivation(hidden, pHidden, hiddenBytes);
```

**Lesson**: Any `IBackend` op that modifies a tensor in-place on GPU must clear the tensor's `_gpuSyncCallback` and `_gpuDisposeCallback` before calling `CacheActivation`. The old callbacks hold a closure over the previous GPU pointer — if that's the same pointer being re-cached, the old callback will free it. **This applies to ALL in-place GPU operations**, not just BroadcastAdd. When adding new in-place ops (e.g., in-place Add, in-place Scale), always clear callbacks first.

### 18. OOM During VAE Decode at 1024x1024 — FreeAsync Deferred Cleanup

**Problem**: After removing per-op `cuStreamSynchronize`, GPU memory frees switched to `cuMemFreeAsync` (stream-ordered). The freed memory is not actually reclaimed until the stream processes the free command. When VAE decode tries to allocate large im2col buffers (~300MB–4.8GB for 512ch at 1024x1024), the allocation fails with `CUDA_ERROR_OUT_OF_MEMORY` because ~7.5GB of UNet weight memory has pending `FreeAsync` calls that haven't been processed yet.

**Symptoms**: `CUDA_ERROR_OUT_OF_MEMORY` during the first Conv2D of VAE decode at 1024x1024. Lower resolutions (256x256) work fine because the im2col buffers are smaller.

**Fix (three-part)**:
1. **UNet weight eviction**: Added `IBackend.FreeWeights(IEnumerable<Tensor>)`. Before VAE decode, the pipeline calls `backend.Sync()` then `backend.FreeWeights(unet.EnumerateWeights())` to synchronously free UNet weights (~7.5GB), reclaiming VRAM.
2. **OOM retry with stream sync**: In `CudaMemory.Allocate`, if `cuMemAlloc` returns `CUDA_ERROR_OUT_OF_MEMORY`, call `GpuTransferHelper.SyncStream()` to flush all pending `FreeAsync` ops, then retry. This catches any remaining deferred frees.
3. **Pipeline-level sync**: `backend.Sync()` before weight eviction ensures all pending GPU work completes before freeing weight memory.

**Lesson**: When switching from synchronous `cuMemFree` to `cuMemFreeAsync`, memory is NOT immediately available for reallocation. Plan for this:
- At pipeline stage boundaries (e.g., UNet → VAE), explicitly sync and free weights from the previous stage
- Add OOM retry logic that syncs the stream before giving up — pending frees may release enough memory
- Monitor VRAM usage at transitions between models that share GPU memory
- This is especially critical on consumer GPUs (12GB) running large models (~10GB weights)

### 19. Non-Blocking Stream Race Condition (Phase 0, revisited)

**Original fix**: Changed `CudaStream(nonBlocking: true)` to `CudaStream(nonBlocking: false)` because synchronous `cuMemcpyHtoD` operates on the NULL stream, and non-blocking streams don't synchronize with NULL stream operations.

**Phase 2 relevance**: With per-op Sync removed, race conditions between memory operations and kernel launches would be even more severe with non-blocking streams. The blocking stream ensures CUDA's implicit synchronization guarantees apply. When future optimization moves to async copies (`cuMemcpyHtoDAsync`/`cuMemcpyDtoHAsync` on the compute stream), non-blocking streams can be revisited — but ALL memory operations must be on the same stream.

---

## Lessons for Future Model Ports

### Attention Configuration is the #1 Trap

Every framework names attention parameters differently:

| Framework/Model | Parameter | Meaning |
|---|---|---|
| diffusers SD1.5 | `attention_head_dim=8` | **8 heads** (confusing! not head dim) |
| diffusers SDXL | `attention_head_dim=[5,10,20]` | Per-block head counts |
| Some configs | `num_heads=8` | 8 heads (clear) |
| Some configs | `head_dim=64` | 64-dim per head (clear) |

**Always verify** by printing `model.attn.heads` and checking Q/K/V reshape shapes in Python before writing C# config code.

### Weight Shape vs Usage Mismatches

A `proj_in.weight` with shape `[320, 320, 1, 1]` is a Conv2d but equivalent to linear for 1x1 kernels. For non-1x1 kernels, im2col/GEMM must be used. Always check weight shape.

### GELU Variant Differences

C# uses tanh-approximated GELU. PyTorch default `F.gelu` uses exact erf-based GELU. Difference is ~1e-4 and acceptable, but check which variant the reference uses if FFN diverges.

### RNG Differences

C# Box-Muller vs PyTorch algorithm. Same seed = different noise. Always compare with **shared noise tensors**, never by matching seeds.

### Expected FP32 Tolerances

| Layer type | Expected avg_err |
|---|---|
| Element-wise (Add, SiLU) | < 1e-7 |
| GroupNorm, LayerNorm | < 1e-6 |
| Linear/Conv (GEMM) | < 1e-5 |
| Full attention block | < 1e-4 |
| Full UNet/DiT pass | < 1e-3 |

If a layer exceeds these by 10x+, there's a real bug — not FP noise.

### Gated Activations (GEGLU/SwiGLU/GLU) — Last-Dimension Split

All gated activation functions split a tensor along the last dimension, NOT at the flat midpoint. For input `[..., 2*D]`:
- First D elements per row = x (value path)
- Next D elements per row = gate (gating path)

In a GPU kernel with flat thread indexing, decompose the output index: `outerIdx = i / D`, `d = i % D`, then `inputX = outerIdx * 2*D + d`, `inputGate = inputX + D`. A flat midpoint split (`input[i]` and `input[i + totalOutput]`) is WRONG for any multi-row tensor.

**Test pattern**: Always test with `[2, 2, 2*D]` or similar multi-row input. Single-row inputs (`[1, 1, 2*D]`) won't catch the bug because flat split coincidentally equals last-dim split.

### In-Place GPU Operations — Callback Cleanup

When a `CudaBackend` op modifies a tensor's GPU buffer in-place (BroadcastAdd, future in-place Add/Scale/etc.):
1. The tensor may already have `_gpuSyncCallback` / `_gpuDisposeCallback` from a previous `CacheActivation` call
2. Those callbacks hold closures over the GPU pointer
3. Calling `CacheActivation` again triggers `DataPointer` access → fires old callback → `FreeAsync` on the pointer you just modified
4. **Always clear both callbacks to `null` before calling `CacheActivation` for in-place ops**

### FreeAsync and OOM at Pipeline Stage Boundaries

When using `cuMemFreeAsync` instead of `cuMemFree`:
- Memory is NOT immediately reclaimed — it's deferred until the stream processes the free
- At pipeline stage transitions (UNet → VAE), explicitly `Sync()` and `FreeWeights()` for the previous model
- Add OOM retry logic: on `CUDA_ERROR_OUT_OF_MEMORY`, sync the stream (flushes pending frees) and retry
- Consumer GPUs (8-16GB) hit this hard when large models (~10GB) share VRAM with large temporary buffers

### Visual Output Validation

Never trust that "tests pass" means "output is correct". Always visually inspect output images after major changes:
- A numerically plausible output (reasonable value ranges, no NaN/Inf) can still be completely wrong
- GEGLU bug produced images with correct tensor statistics but garbled visual content
- Keep known-good reference images and compare after every significant change
- Quick sanity check: does the output resemble the prompt? If not, something is broken regardless of what metrics say

---

## Phase 3 — FP16 Inference Deviations

### 20. Normalization Kernels — F32 Weight/Bias Read as F16

**Problem**: F16 PTX normalization kernels (`groupnorm_f16`, `layernorm_f16`, `groupnorm_silu_f16`) use `ld.global.b16` to load weight and bias parameters. CudaBackend dispatches the F16 kernel path based on `input.DType == DType.F16`, not weight dtype. SDXL safetensors checkpoints have mixed dtypes — some norm weights are F32 even when linear weights are F16. When F32 weight data is loaded as F16, the kernel reads garbage values, causing completely wrong normalization output.

**Symptoms**: Output images were pure noise — all spatial structure destroyed. Every normalization layer (GroupNorm, LayerNorm, fused GroupNorm+SiLU) produced wrong output.

**Fix**: In `CudaBackend.GroupNorm`, `GroupNormSilu`, and `LayerNorm`: before launching the F16 kernel, check `weight.DType` and `bias.DType`. If either is F32, allocate a temporary F16 buffer, cast using `LaunchCastF32ToF16`, pass the cast pointer to the kernel, and `FreeAsync` the temp buffer in the finally block.

**Lesson**: PTX kernels that load data at a specific width (`ld.global.b16` vs `ld.global.b32`) are dtype-sensitive. When dispatching based on `input.DType`, always verify that ALL other tensor operands (weights, biases) match the expected kernel dtype. If not, cast before launch.

### 21. Mixed-Dtype Safetensors — Inconsistent Per-Tensor DTypes

**Problem**: SDXL safetensors checkpoints (e.g., Juggernaut XL) have inconsistent dtypes across tensors: conv weights are mostly F32 (60/62), linear weights are mostly F16 (867/869), norm weights are mixed. The original F16 test loaded raw weights without ensuring uniform dtype.

**Analysis of Juggernaut XL checkpoint**: F16=2 conv weights, F32=60. F16=867 linear weights, F32=2. F16=492 norm params, F32=104.

**Fix**: Added `CastWeightsToF16` helper that iterates all weights and casts any non-F16 tensors to F16 using `Tensor.CastTo(DType.F16)`. Applied before loading weights into models for F16 inference.

**Lesson**: Never assume safetensors files have uniform dtype. Different model components may have been saved at different precisions. When running F16 inference, explicitly cast all weights to F16 before loading. Consider making this a utility in the model handler package.

### 22. cuBLAS GemmEx — Does Not Support Mixed A/B Operand Types

**Problem**: `cublasGemmEx` requires operands A and B to have the **same data type** (both F16 or both F32). When `TimestepEmbedding` and `AdditionEmbedding` create F32 intermediate tensors and call `backend.Linear` with F16 weights, the Linear method's F32 path called `cublasSgemm` which read F16 weight data as F32 (garbage), and the F16 path is unreachable since dispatch was based on `input.DType == DType.F16`.

After fixing to always use `cublasGemmEx` with per-operand dtype detection (`CUDA_R_32F` for input, `CUDA_R_16F` for weight), cuBLAS returned `CUBLAS_STATUS_NOT_SUPPORTED` (error 15) because mixed A=F32, B=F16 is not a supported combination.

**Fix**: When weight dtype differs from input dtype, cast the weight to match the input dtype before calling `cublasGemmEx`. Allocate temp buffer, launch `CastF32ToF16` or `CastF16ToF32` kernel, use cast pointer for GEMM, `FreeAsync` temp buffer in finally. Applied same pattern to bias add dispatch (check `output.DType` vs `bias.DType`, cast if mismatched). Applied to all GEMM call sites: `MatMul`, `Linear`, `BatchedMatMul`, `Conv2D`.

**Lesson**: cuBLAS `GemmEx` supports these type combinations with `CUBLAS_COMPUTE_32F`:
- A=F16, B=F16, C=F16 (half precision)
- A=F16, B=F16, C=F32 (mixed precision upcast)
- A=F32, B=F32, C=F32 (single precision)

It does NOT support A=F32, B=F16 or A=F16, B=F32. When operands have different dtypes, always cast the mismatched one to match before calling GEMM. The cast is cheap (elementwise kernel) compared to the GEMM itself.

### Mixed-Dtype Pattern for Future Ops

When adding new CudaBackend ops that use cuBLAS or PTX kernels with dtype-specific code:
1. Check ALL tensor operands' dtypes, not just the input
2. If any operand has a different dtype than the kernel expects, cast it first
3. Track cast buffer pointers and `FreeAsync` them in the finally block
4. For cuBLAS: A and B must always have identical dtypes
5. For PTX: `ld.global.b16` vs `ld.global.b32` — the load width must match the actual data format

---

### Diagnostic Script Inventory

All scripts in `tests/python-reference/` using venv at `tests/python-reference/.venv/`:

| Script | Purpose |
|---|---|
| `dump_reference_stats.py` | Full pipeline: noise, embeddings, per-step latents, final output |
| `dump_layer_outputs.py` | Per-layer model outputs with index.json |
| `dump_attn_sublayers.py` | Sub-operation breakdown of first CrossAttentionBlock |
| `compare_layers.py` | Utility for comparing binary tensor files |
