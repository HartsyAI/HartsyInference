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

## Phase 4 — Flux Deviations

### 23. BFL→Diffusers Final-Layer AdaLN Swap (FluxCheckpointConverter)

**Problem**: BFL Flux's `LastLayer` chunks its AdaLN-Continuous modulation as `[shift, scale]`:
```python
shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
x = (1 + scale) * norm(x) + shift
```
Diffusers' `AdaLayerNormContinuous.forward` chunks as `[scale, shift]` — the opposite. The C# `FluxTransformer.ApplyFinalLayer` matches the diffusers convention (first half of `modParams` = scale, second half = shift). So when loading a BFL-format checkpoint (any single-file Flux .safetensors using `model.diffusion_model.*` keys, including all ComfyUI/BFL distributions and `flux1-dev-fp8.safetensors`), the final-layer adaLN weight rows must be **swapped along dim 0** to match.

The original `FluxCheckpointConverter.ConvertFinalLayerKey` just renamed `final_layer.adaLN_modulation.1.weight/bias` → `norm_out.linear.weight/bias` without rearranging rows. Result: at every denoising step, the network applied BFL's shift coefficients as scale and vice versa, producing per-channel skew that compounded across steps.

**Symptoms**: Image structure was correct (subject visible at the right pose/position) but the output had a strong color cast — typically magenta/purple, with the green channel pushed strongly negative. Per-channel velocity statistics showed monotonic growth in absolute magnitude across denoising steps (e.g. \|v_c13\| grew 3.4× from step 1 to step 10), instead of the constant-magnitude profile flow matching expects.

**How it was found**: Per-latent-channel mean diagnostic logging on the packed velocity at every step showed channel 13 with monotonically increasing positive velocity (and channel 9 similarly), driving the latent ch13 to mean=−2.29 by step 10 while other channels stayed near zero. Visual output had R=166/G=69/B=171 (magenta cast). Static review of BFL's `LastLayer` vs diffusers' `AdaLayerNormContinuous` revealed the chunk-order mismatch. The official diffusers `convert_flux_to_diffusers.py` confirms the swap via `swap_scale_shift()` applied to `norm_out.linear.weight/bias`.

**Fix**: Added `SwapScaleShiftHalves(Tensor)` helper in `FluxCheckpointConverter`. In `ConvertFinalLayerKey`, when converting BFL `final_layer.adaLN_modulation.1.weight` and `.bias`, swap the two halves of dim 0 before storing under the diffusers key. Byte-wise `Buffer.MemoryCopy` so it's dtype-agnostic (handles F8/F16/BF16/F32 weights). Applied to both weight (2D `[2*H, H]`) and bias (1D `[2*H]`). The swap is only needed for the BFL path — diffusers-format checkpoints already have the correct layout.

**Result**: Flux Dev FP8 at 512×512, 10 steps, seed=42, prompt="A photograph of an astronaut riding a horse", guidance=3.5. RGB means went from 166/69/171 (magenta) to 59/54/55 (color-balanced). Per-step velocities became roughly constant in magnitude across steps (flow-match ideal). Output is now a clean photorealistic image.

**Lesson**: When porting checkpoint conversions that involve `.chunk()` or `.split()` on a fused tensor, always verify the chunk order in BOTH the source framework and the destination framework. Naming alone (e.g. "shift", "scale") doesn't guarantee positional consistency — frameworks order them differently. The official converter scripts (in `diffusers/scripts/`) are the authoritative reference for these reorderings and worth grepping before writing custom converters. Future `swap_scale_shift`-style fixes may be needed for other DiT models that also have a final AdaLN-Continuous layer (Flux.2, AuraFlow, HunyuanImage, QwenImage, etc.) — verify each.

### 24. ComfyUI `fp8_scaled` Checkpoint Format (Krea, Kontext, Flux.2-Klein)

**Problem**: ComfyUI repackages BFL checkpoints into an `fp8_scaled` format that stores each linear-layer weight as raw FP8 E4M3 (full ±448 dynamic range, mean abs ~14) plus two F32-scalar companion tensors:
- `<weight_key>.scale_weight` — so that `real_weight = fp8_byte_decoded × scale_weight`
- `<weight_key>.scale_input` — activation-quantization scale for true FP8 GEMM with FP8 activations

Plus a marker tensor `scaled_fp8` (FP8 dtype, no relevant content) at the top level. Files we have in this format: `flux1-krea-dev_fp8_scaled.safetensors`, `flux1-dev-kontext_fp8_scaled.safetensors`. Plain Comfy-Org Dev FP8 (`flux1-dev-fp8.safetensors`) is NOT scaled — it stores already-pre-scaled FP8 values directly (mean abs ~0.014) using a tiny fraction of FP8's range.

The previous converter ignored these scale companions, producing garbage output.

**Symptoms** (depend on which scale was missing):
- Initial implementation (`alpha` not wired in): output severely under-magnitude → barely-visible silhouette in noise.
- After wiring `alpha = scale_weight` for Linear, but missing scale propagation through `SwapScaleShiftHalves`: AdaLN-Continuous final layer ran with `alpha=1.0` instead of `~3.5e-4`, multiplying the modulation ~3000× too large → all-but-saturated white background with a blocky black silhouette where the saturation tipped (each 32×32 packed token producing a uniform 16×16 image patch because `proj_out`'s output was clamped at the F32 range tail).

**Fix** (three coordinated pieces):
1. Added `Tensor.Fp8ScaleFactor` (default 1.0) — a per-tensor scalar that GEMM call sites fold into cuBLAS' `alpha` parameter at zero memory + zero perf cost. Way cheaper than dequanting the 12B-param transformer to F16 at load (which would have OOMed our 31GB-RAM box at ~22GB anonymous RSS).
2. Added `FluxCheckpointConverter.ApplyFp8ScaledDequant` pre-pass: detects the format by presence of any `*.scale_weight` key, attaches each scale to its sibling `.weight` tensor's `Fp8ScaleFactor`, drops the `.scale_weight` and `.scale_input` keys (we run F16 GEMM not FP8 GEMM, so `scale_input=1.0` activations are fine).
3. **Critical**: every site that creates a NEW Tensor by copying bytes from an FP8 weight must propagate `Fp8ScaleFactor`. Updated:
   - `SplitQkvWeight` (fused QKV → q, k, v)
   - `SplitSingleLinear1Weight` (single-stream linear1 → q, k, v, mlp)
   - `SwapScaleShiftHalves` (BFL→diffusers final-layer adaLN row swap from deviation #23)
4. `CudaBackend.Linear`: `float alpha = weight.Fp8ScaleFactor;` (was hardcoded 1.0).

**Result**: Flux.1 Krea Dev fp8_scaled at 512×512, 10 steps, seed=42, guidance=3.5. RGB means went from 180/180/180 (saturation pattern) → 141/137/132 (natural distribution, full 0-250 dynamic range). Output is now a sharp photoreal image of the prompt, with Krea's signature aesthetic richness (more detail than Dev FP8 produces at the same step count).

**Lesson for future ports**: When introducing a per-tensor metadata field on `Tensor` (like `Fp8ScaleFactor` or any future quant scale / zero-point / activation-stat), audit *every* converter site that creates a new Tensor by copying bytes — splits, transposes, half-swaps, permutations. The metadata won't be set on the new tensor by default and the bug surfaces only at that specific layer's GEMM, which makes it hard to catch. A grep for `new Tensor(` inside checkpoint converters is a good audit checkpoint after adding any such field.

---

### Diagnostic Script Inventory

All scripts in `tests/python-reference/` using venv at `tests/python-reference/.venv/`:

| Script | Purpose |
|---|---|
| `dump_reference_stats.py` | Full pipeline: noise, embeddings, per-step latents, final output |
| `dump_layer_outputs.py` | Per-layer model outputs with index.json |
| `dump_attn_sublayers.py` | Sub-operation breakdown of first CrossAttentionBlock |
| `compare_layers.py` | Utility for comparing binary tensor files |

---

## Phase 4 — Z-Image Deviations

### 25. Z-Image Single-File Naming Differs From Diffusers Reference

**Problem**: Initial implementation followed the diffusers naming spec (`all_x_embedder.2-1`, `all_final_layer.2-1`, separate `to_q/to_k/to_v` per attention) but the SwarmUI single-file FP8Mix checkpoint and the official Tongyi diffusers index use simpler, fused names. Loading with the diffusers-spec keys would `KeyNotFoundException` on every transformer block.

**Discovery method**: Read the safetensors header with `python3 -c 'import json,struct; ...'` and grouped keys by prefix. Saw `attention.qkv.weight` `[11520, 3840]` (fused), `attention.out.weight`, `attention.{q_norm,k_norm}`, `x_embedder.{weight,bias}`, `final_layer.{adaLN_modulation.1,linear}.*` — all different from what the C# code expected. Also discovered each FP8 weight has companion `.weight_scale` (F32 scalar) and `.comfy_quant` (U8 metadata blob) — the ComfyUI fp8_scaled format.

**Fix**: Rewrote `ZImageBlock` and `ZImageContextRefinerBlock` to load `attention.qkv.weight` as one fused `[3*hidden, hidden]` tensor and split Q/K/V at runtime in `SplitQkv`. Renamed loaders for `attention.out`, `attention.q_norm`, `attention.k_norm`. Updated `ZImageTransformer` for `x_embedder.*` and `final_layer.*` (no patch suffix). `ZImageCheckpointConverter` now folds `.weight_scale` into `Tensor.Fp8ScaleFactor` (cf. deviation #24) and drops `.comfy_quant` keys.

**Lesson**: Always inspect the actual safetensors header before writing the loader. The diffusers source naming is canonical for the Python class, but **single-file checkpoints distributed by SwarmUI / ComfyUI / BFL routinely simplify naming** — fused QKV, no patch-size suffix, different scale-companion key names. Header inspection takes 10 seconds; debugging mismatched keys takes hours.

### 26. Z-Image PadCaption / PadImage No-Op Disposal

**Problem**: `PadCaption` and `PadImage` return the input tensor unchanged when `paddedLen == realLen` (already at the multiple-of-32 boundary). The caller did `capProjected.Dispose()` after `capPadded = PadCaption(capProjected, ...)` — disposing the still-aliased `capPadded` along with it. Next op got `ObjectDisposedException`.

**Fix**: Added `if (!ReferenceEquals(capPadded, capProjected)) capProjected.Dispose();` guards to all three pad/trim sites in `ZImageTransformer.Forward`.

**Lesson**: Pass-through helpers that *might* allocate are disposal traps. Either always allocate (so the caller's disposal pattern works), or guard with `ReferenceEquals` at the call site. The same pattern applies to any `if (x_already_correct) return x;` early-return helper.

### 27. Z-Image Pipeline Semantic Bugs (Eight Deviations From Diffusers)

**Problem**: First end-to-end run produced pure colored static. After fixes 25 and 26 the pipeline ran clean for 8 NFE but the output was noise. Comparing line-by-line against `huggingface/diffusers/src/diffusers/{models/transformers/transformer_z_image.py, pipelines/z_image/pipeline_z_image.py}` revealed eight semantic deviations from diffusers — every one of them shifted the output measurably, in roughly this order of magnitude:

1. **Pipeline must negate the transformer output** (`noise_pred = -noise_pred`) before the flow-match Euler step. Without it, integration runs in the noise-direction and output is pure RGB static. (Diffusers flow-match Euler uses `x_next = x + dt * noise_pred` with `dt = sigma_next - sigma < 0`, so the negation gives `x + |dt| * v` — descent in the +v direction. **Without** the negation we ascend.)
2. **Sequence concat order is `[image, caption]`, not `[caption, image]`**. Per `transformer_z_image.py` line 859 (`unified.append(torch.cat([x[i][:x_len], cap[i][:cap_len]]))`). Affects both attention pattern and which slice we take after the main blocks.
3. **Gates pass through `tanh()`**: `gate_msa = mod[1].tanh()` and `gate_mlp = mod[3].tanh()`. Without this, gates blow up at init and the modulated residuals dominate the signal.
4. **Patchify inner-dim order is `[pH, pW, C]` (channel last/fastest), not `[C, pH, pW]`**. Diffusers permutation is `permute(1, 3, 5, 2, 4, 6, 0)` on `[C, F_tokens, pF, H_tokens, pH, W_tokens, pW]` → `[F, H, W, pF, pH, pW, C]`. `Unpatchify` mirrors this (permute(6, 0, 3, 1, 4, 2, 5)).
5. **Position IDs are not all-zero for caption**. Caption tokens get pos_start `(1, 0, 0)` with grid `(capRealLen, 1, 1)` — i.e. frame-axis runs `1..capRealLen`. Image tokens get pos_start `(capRealLen+1, 0, 0)` with grid `(1, hPacked, wPacked)` — same frame for all image tokens, varying h/w. Pad slots stay `(0,0,0)` (overwritten via `where(mask, pad, feat)` semantics).
6. **Timestep is inverted**. Pipeline computes `t = (1000 - sigma_in_0_to_1000) / 1000 = 1 - sigma`, then transformer multiplies by `t_scale=1000` internally for the sinusoidal embedding. Without this, the model conditions on the OPPOSITE point in the schedule at every step.
7. **Final-layer `norm_final` is `LayerNorm(elementwise_affine=False, eps=1e-6)`**, not RMSNorm and not eps=1e-5 (which is `_config.NormEps`). Use the literal 1e-6 here.
8. **Final-layer modulation is scale-only, single chunk**. The checkpoint shape is `[hidden, ADALN_EMBED_DIM] = [3840, 256]` (one output) — formula is `out = norm(x) * (1 + scale)`, no shift, no gate. Lumina2's stock NextDiT outputs 2*hidden (scale+shift); Z-Image diverges by removing the shift.

**Status**: After all eight fixes, output went from pure noise → blobby colored regions → recognizable central humanoid silhouette with heavy high-frequency glitch banding (~80% coherent). Pipeline plumbing is now correct end-to-end; remaining glitch artifacts likely come from one of: (a) attention-mask not enforced for pad slots (pad tokens currently participate fully in attention), (b) RoPE phase convention mismatch for the consecutive-pair complex multiplication, (c) dynamic-vs-static scheduler shift mismatch (config says static 3.0, diffusers pipeline may compute dynamic shift via `calculate_shift` like Flux). Further binary-comparison debugging vs a Python reference run is the remaining work.

**Lesson**: For DiT models with a flow-match scheduler, an end-to-end pipeline can produce pure noise from any single one of these bugs — sign flip, wrong position IDs, wrong patchify, wrong timestep, missing tanh, wrong concat order. Each fix moves the output a step closer (noise → static → blobs → bands → silhouette), and you need them ALL to converge. Build a diagnostic run that dumps at least: caption embeddings stats, transformer-output stats per step, final latent before VAE, decoded RGB stats. Compare every one of those against a Python reference run at the same seed before suspecting your VAE or your CUDA backend. The deviations are almost always in the new model's pipeline plumbing.

### 28. Z-Image Context-Refiner Was Missing RoPE (THE root cause of the glitch banding)

**Discovery method**: Built a layer-by-layer F32 binary diff against the diffusers reference. `tests/python-reference/dump_zimage_full_forward.py` runs `ZImageTransformer2DModel.from_pretrained('Tongyi-MAI/Z-Image-Turbo')` on deterministic synthetic inputs (`torch.manual_seed(42)`, latent `[1,16,32,32]`, caption `[1,64,2560]`, sigma=0.5) and hooks every embedder, refiner, main block, and the final layer to dump 38 F32 tensors. The C# side (`ZImageDiffTests.Transformer_Matches_PythonReference_LayerByLayer`) loads the diffusers BF16 shards, fuses to_q/to_k/to_v into the C# fused-qkv layout, and runs Forward on the SAME bytes via the CPU backend with `Z_IMAGE_DEBUG_DIR` set so each layer dumps at the matching name. Then `diff_zimage_layers.py` prints |ref − cs| stats per layer.

The first run showed:

```
t_embedder         8.270e-09   <-- noise floor
x_embedder         2.757e-08
noise_refiner.0    3.057e-08
noise_refiner.1    3.252e-08
cap_embedder       7.754e-08
context_refiner.0  1.493e-01   <-- BUG (jump of 7 orders of magnitude)
context_refiner.1  2.101e-01
layers.0           4.314e-02   (compounding)
... rest contaminated
```

Noise_refiner being clean while context_refiner blew up was the smoking gun: both blocks share the same architecture, but `noise_refiner` had `ZImageRope` wired up (image tokens) while `context_refiner` had no rope arg at all.

**Root cause**: In diffusers, `ZImageTransformerBlock` takes `freqs_cis` regardless of the `modulation` flag — `ZSingleStreamAttnProcessor` always applies `apply_rotary_emb(query/key, freqs_cis)` if non-null. The block class is reused for context_refiner, noise_refiner, and main `layers` — **all three apply RoPE**. The C# implementation split the modulation case into two classes (`ZImageContextRefinerBlock` for modulation=False, `ZImageBlock` for modulation=True), and the context_refiner variant simply omitted the rope plumbing.

Why noise_refiner was clean despite my noise_refiner using `BuildImagePositionIds` with `frame=1` instead of diffusers' `frame=cap_padded_len+1`: image-only attention with all tokens sharing the same frame index is rotation-invariant — the per-token rotation `R` cancels in `softmax((R·Q)(R·K)ᵀ) = softmax(QKᵀ)`. So the constant offset doesn't matter for image-only attention. (It DOES matter for cross-attention in the main layers — see fix 5 in deviation 27.)

Why context_refiner couldn't tolerate this: caption tokens have **per-token frame indices** `1..capPaddedLen` along axis 0 (h=w=0). Different positions → different rotations → attention scores depend on relative position. Without RoPE, every caption token attends as if from the same point — losing all positional structure.

**Fix**:
1. Added `ZImageRope.BuildCaptionPositionIds(capPaddedLen)` returning `[(1,0,0), (2,0,0), …, (capPaddedLen,0,0)]`.
2. Added a `ZImageRope?` parameter to `ZImageContextRefinerBlock.Forward` and applied it before SDPA, mirroring `ZImageBlock`.
3. `ZImageTransformer.Forward` now builds a separate `captionRope` (caption positions only) and passes it to every context_refiner block. The full-sequence `_rope` is still built later for the main layers.

After the fix, the diff ran clean end-to-end — every layer 1e-7 to 1e-4 (F32 numerical accumulation):

```
t_embedder         8.270e-09
x_embedder         2.757e-08
noise_refiner.0    3.057e-08
cap_embedder       7.754e-08
context_refiner.0  2.782e-07   <-- now matches reference
context_refiner.1  4.391e-07
layers.0           1.226e-07
... clean through layers.29
```

**Also fixed in this round** — `BuildPositionIds` was using `capRealLen+1` for the image frame and `1..capRealLen` for caption frames. Diffusers uses `capPaddedLen+1` and `1..capPaddedLen` (every caption slot, real + pad, gets a position). The synthetic test masked this because `capRealLen == capPaddedLen` for a 64-token caption, but real prompts will hit the difference. Updated the signature to drop `capRealLen` (no longer needed); caller passes `capPaddedLen` only.

**Lesson**: When porting a transformer where the same block class is reused with different module flags (modulation=True/False, with/without cross-attention, etc.), do NOT split it into separate C# classes that share architecture by accident. The diffusers `ZImageTransformerBlock` was the same class used in 3 places — context_refiner / noise_refiner / layers — and **all three branches** of the original code-path inherit the same `freqs_cis` / `attention_mask` / `processor` plumbing. Carving the modulation case into a separate class encouraged me to "clean up" the rope plumbing for the simpler path. The simpler path needed the rope just as much. Either keep the unified block (dispatch on a runtime flag), or after splitting, run a layer-by-layer diff to catch what the split silently dropped.

### 29. Z-Image Caption Encoding Pipeline — Three Bugs in How the Test Calls Qwen3

**Problem**: With deviation 28 fixed, the Z-Image transformer matched the diffusers reference byte-for-byte across all 38 layers (max err 4.5e-5 at layer 29). But the end-to-end image was still glitched — recognizable astronaut shape with heavy banding. The transformer math was correct; the *inputs* to the transformer were wrong.

**Discovery**: Re-read `pipeline_z_image.py:198-247` (`_encode_prompt`) and compared to the test code in `ZImageGenerationTests.cs`. Three deviations from the reference, each subtle and silent:

1. **No chat template applied**. Diffusers does:
   ```python
   prompt_item = self.tokenizer.apply_chat_template(
       [{"role": "user", "content": prompt}],
       tokenize=False, add_generation_prompt=True, enable_thinking=True,
   )
   ```
   This wraps the user prompt in `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`. The test was calling `tok.Encode(prompt)` (raw text + EOS), not `tok.EncodeChat(prompt)` — the model was conditioning on a text distribution it had never seen during training.

2. **Final hidden state instead of penultimate**. Diffusers does `text_encoder(...).hidden_states[-2]` — the second-to-last layer's output, raw (no final RMSNorm). The test was using `qwenEncoder.Encode(...)` which returns the FINAL hidden state through the model's `model.norm` RMSNorm. That changes the statistics dramatically: post-norm `abs_mean ≈ 1.25`, penultimate `abs_mean ≈ 6.4` with a much wider min/max range. The Z-Image transformer's `cap_embedder` was trained to consume the wider distribution; feeding it the post-norm distribution sends every downstream layer off-spec.

3. **Padding tokens treated as real**. Diffusers filters via `prompt_embeds[i][prompt_masks[i]]` — drops every padding-position hidden state. The test was passing the full padded `[1, 64, 2560]` tensor to `GenerateFromEmbeddings`. With `capRealLen = 64` (and `capPaddedLen` therefore also 64), `PadCaption` was a no-op, and the 51 garbage pad-position hidden states (Qwen3 hallucinations at causally-blind pad positions) flowed through `cap_embedder` and `context_refiner` as if they were real caption tokens. The Z-Image transformer expects: `realLen` real tokens → `cap_embedder` → pad to `capPaddedLen` with the **learned** `cap_pad_token` parameter.

**Fix** (all in the test, no encoder/pipeline changes): use `tok.EncodeChat(prompt)`, compute `realLen` from `Qwen3Tokenizer.CreateAttentionMask`, call `qwenEncoder.EncodeMultiLayer(backend, [tokens], [NumLayers - 1])` to get the penultimate hidden state in HF indexing, and slice the result to `[1, realLen, 2560]` with a small `SliceFirstSeq` helper before handing it to `pipeline.GenerateFromEmbeddings`. The pipeline naturally derives `capRealLen = realLen`, pads to the next multiple of 32 with `cap_pad_token`, and the transformer sees exactly what diffusers feeds it. Also added a `NumLayers` accessor to `LlamaStyleEncoder` so the test doesn't have to hardcode 36.

**Result**: First end-to-end run after the fix produced a clean 512×512 image of an astronaut riding a horse on the moon (8 NFE, CFG=1.0, seed=42) — no banding, no glitches, prompt fully respected. Total e2e time: 2 min 40 s on RTX 3060.

**Lesson**: A perfect transformer with the wrong caption signal looks identical to a buggy transformer. After verifying the model math via layer-by-layer diff against synthetic inputs, the next failure mode is the pipeline plumbing around it — tokenization template, hidden-layer selection, attention/pad masking. For DiT models that take "caption embeddings" as input, treat the caption-encoding pipeline as a separate component that needs its own diff against the reference. Print the embedding stats (shape, abs_mean, min/max) and compare to Python; a wide divergence in `abs_mean` is a clean signal that you're reading the wrong layer or applying the wrong norm.

### 30. CudaBackend.RmsNorm Reads Weight as `float*` Without DType Check

**Problem**: When porting Z-Image-Base (BF16 weights for everything except FP8 Linear projections), the GPU end-to-end image came out as a 16-px-grid of disconnected color blobs — exact patch-boundary noise. Same code path that produces a clean image for Z-Image-Turbo. The transformer math, RoPE, AdaLN, attention, FFN, scheduler, CFG, caption encoding — all verified identical to Turbo, yet the output looked like every patch was being processed in isolation with no cross-patch attention.

**Root cause**: [CudaBackend.cs:604-633](../../src/HartsyInference.Cuda/CudaBackend.cs#L604) implements `RmsNorm` as a CPU fallback (T5 is a once-per-generation cost, not a hot path) and reads the weight pointer as `float*` directly:

```csharp
float* pWeight = (float*)weight.DataPointer;     // ← assumes F32
...
pOut[baseIdx + i] = pIn[baseIdx + i] * invRms * pWeight[i];
```

There is no dtype check. If the weight tensor is BF16 (or F16), the bytes get bit-reinterpreted: every two consecutive BF16 values are read as one F32. Each F32 read is the upper 16 bits of one BF16 value concatenated with the upper 16 bits of the next BF16 value, producing essentially random garbage scale factors.

Why Turbo (FP8Mix) didn't surface this: `LogDTypeDistribution` shows `F32=283, F8_E4M3=170` — Turbo stores its norms in F32 natively, FP8 is only used for the heavy Linear weights (which go through cuBLAS GEMM with proper FP8→F16 cast). The F32 norm path is correct.

Why Base BF16 (`benjiaiplayground/z-image-base-repacked`) and Base nvfp8-mixed (`RamonGuthrie/z_image_base-nvfp8-mixed`) hit the bug: both store norm scales as BF16 (`BF16=453` for the pure BF16 file, `BF16=281, F8_E4M3=172` for the nvfp8-mixed). Every RMSNorm call across the 4 norms × 30 main layers + 2×2 refiners + final cap_embedder.0 was producing garbage modulated activations, breaking attention coherence across patches.

**Why the noise looked "blocky"**: each image patch (after `Patchify`) has its own independent token. Once attention can't propagate scale information across patches (because the RMSNorm scale per-channel is random per layer), each patch evolves into its own per-channel noise distribution, giving a clean 16-px-grid of differently-tinted blobs.

**Fix**: cast non-F32 RMSNorm scales to F32 at load time. The norm scales are tiny (`[hidden]` = 3840 floats per scale, 8 KB) and there are ~5 per block × 32 blocks = ~1.3 MB total — negligible cost for a one-time conversion. Same pattern was already used for `QkNorm.LoadWeights` (head-norms) and `LlamaStyleEncoder.CastToF32IfNeeded` (Qwen3 RMS scales). Added `LoadAsF32` helpers in `ZImageBlock` and `ZImageContextRefinerBlock`, applied to the 4 norms per block; cap_embedder.0 cast in `ZImageTransformer.LoadWeights`. After the fix, Z-Image-Base produces a clean astronaut-on-horse-on-moon image at 512×512, 28 steps, CFG=4.0.

**Lesson**: any C# kernel that does `(float*)weight.DataPointer` is a latent dtype bug — either fix the kernel to accept all supported dtypes, or guarantee F32 at load time and document the guarantee. Mixed-dtype checkpoints (FP8 Linear weights + BF16 norms is the modern norm) will surface these bugs the moment a non-F32-storing checkpoint hits a CPU-fallback op. When porting a model variant, audit every `weight.DataPointer` cast against the actual tensor dtypes the new checkpoint ships, not just the ones the previous variant happened to use.

A second smaller bug: BF16-only checkpoints exposed that `CudaBackend.CastOnGpu` had no `BF16↔F32` path, so `Linear()` with F32 input + BF16 weight would throw `NotSupportedException`. Added [cast_bf16_f32.ptx](../../src/HartsyInference.Cuda/Ptx/cast_bf16_f32.ptx) — BF16 is just the upper 16 bits of F32 (lossless one-way cast, RTNE on the way back). Wired through `LaunchCastBf16ToF32` / `LaunchCastF32ToBf16` and the `CastOnGpu` dispatch table. Same kernel exposes BF16↔F16 via two-step temp F32. This was a prerequisite for Base-BF16 to reach `RmsNorm` at all — without it the test crashed earlier at the t_embedder Linear.

---

## Phase 4 — SD3.5 Deviations

### 31. CLIP Tokenizer Producing Character-Level Garbage Tokens (CLIP-L + CLIP-G)

**Problem**: `Microsoft.ML.Tokenizers.BpeTokenizer.Create(vocabStream, mergesStream)` — the simple two-arg overload — runs **generic byte-pair encoding without CLIP's `</w>` end-of-word suffix or CLIP's regex pre-tokenizer**. Tokenizing "A photograph of an astronaut riding a horse" produced `[49406, 64, 1688, 684, 514, 7982, 627, 83, 553, 7545, 64, 8562]` — character/prefix-level fragments (`'a'`, `'ph'`, `'ot'`, `'og'`, `'raph'`, `'of'`, `'an'`, `'astron'`, `'au'`, `'t'`, `'a'`, `'horse'`) — instead of the HF reference `[49406, 320, 8853, 539, 550, 18376, 6765, 320, 4558, 49407]` (`'a</w>'`, `'photograph</w>'`, `'of</w>'`, `'an</w>'`, `'astronaut</w>'`, `'riding</w>'`, `'a</w>'`, `'horse</w>'`). The `EOS` was at position 12 instead of position 9 because there were 11 phantom sub-tokens.

A second, related defect: `Encode()` zero-padded after the EOS instead of padding with the EOS token. HF `CLIPTokenizer(padding="max_length")` uses `pad_token == eos_token`, so positions 10..76 should all be 49407, not 0.

**Why this stayed hidden through SDXL**: SDXL is robust enough to extract some semantic content from corrupted tokens — the images "look fine" superficially, even though the prompt is ~half-ignored. SD3 is much less tolerant: combined with deviation #32 below, the wrong tokens produced enough conditioning drift to drive the model into a degenerate "patch grid" output that we could no longer wave off.

**Symptoms**: image at the patch granularity (16-pixel cells for SD3 at patch_size=2 + 8× VAE), uniform purple cast (R≈82, G≈60, B≈103 per channel mean), looking like a textured surface — the model "denoised something" but never coherently propagated cross-patch attention.

**Fix**: rewrote [`ClipTokenizer.cs`](../../src/HartsyInference.ModelAssets.Tokenizers/ClipTokenizer.cs) to use the long-form `BpeTokenizer.Create(vocab, merges, preTokenizer, normalizer, specialTokens, unknownToken, continuingSubwordPrefix, endOfWordSuffix, fuseUnknownTokens)` with:
- `preTokenizer = new RegexPreTokenizer(ClipPreTokenRegex, ClipSpecialTokens)` where `ClipPreTokenRegex` matches `<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+` (mirrors `huggingface/transformers` `CLIPTokenizer.pat`).
- `normalizer = LowercaseNormalizer.Instance` — a tiny custom `Normalizer` that just calls `string.ToLowerInvariant()`.
- `specialTokens = { "<|startoftext|>": 49406, "<|endoftext|>": 49407 }`.
- `endOfWordSuffix = "</w>"` — the missing piece that drives every CLIP BPE merge.
- Padding loop changed to fill with `EndOfTextId` (49407) after the appended EOT.

**Result**: tokens now match HF byte-for-byte; CLIP-L hidden state diff vs HF reference is at F32 noise (avg_err 9.4e-7, max 1.7e-3 across all 77 × 768 elements).

**Lesson**: any port of an OpenAI/CLIP-family tokenizer needs the regex pre-tokenizer **and** `</w>` suffix **and** EOS-padding, not just vocab+merges. The two-arg `BpeTokenizer.Create(stream, stream)` overload from Microsoft.ML.Tokenizers is a trap for CLIP — it's correct for byte-level GPT-2-style BPE but silently wrong for word-level BPE with EOW markers. Quick smoke test for any future tokenizer port: encode "a photograph of an astronaut riding a horse" with HF, encode with C#, assert exact equality on the first 10 token IDs. Cost: 5 minutes. Coverage: catches every form of this bug class.

### 32. CLIP `text_projection` Stored in PyTorch `nn.Linear` Format Read as Transposed (CLIP-L + CLIP-G Pooled Outputs)

**Problem**: `ClipTextEncoder.ExtractPooledOutput` (used to compute the pooled vector that feeds AdaLN modulation in SDXL/SD3) implemented the projection as

```csharp
sum += ePtr[inOffset + i] * wPtr[i * projDim + o];   // x @ W where W treated as [hidden, proj]
```

This reads `text_projection.weight` as if it were stored row-major in `[hidden_size, projection_dim]` order. PyTorch `nn.Linear(hidden, proj).weight` stores `[out_features, in_features] = [proj, hidden]` and the forward is `output = x @ weight.T` — i.e. `output[o] = Σᵢ x[i] · weight[o, i] = wPtr[o*hidden + i]`. The two access patterns `wPtr[i*proj + o]` and `wPtr[o*hidden + i]` are **always** different for non-symmetric square matrices, which CLIP-L (768×768) and CLIP-G (1280×1280) both are.

**Why it survived for so long**: a transposed dense matmul on a roughly-balanced learned weight produces output with the right ballpark norm and a plausible distribution. The pooled vector "looks fine" in any quick check — `abs_mean ≈ 0.6`, `std ≈ 0.97`, no NaN/Inf. The model accepts it as a conditioning input, AdaLN modulates with whatever it received, and the image is degraded but coherent. Smoke tests pass. Visual eyeballing on SDXL passes — the prompt is partially honored. Nobody questions it.

**How it surfaced**: building the SD3.5 layer-by-layer diff harness (deviation methodology from #28) and comparing the C# `clip_g_pooled` against the HF `text_embeds` output element-by-element: avg_err = **0.96**, max_err = **4.44**, ref_abs_mean=0.77 vs cs_abs_mean=0.61. That's 100% relative error on the pooled — clearly not numerical drift.

**Fix**: swapped the inner-loop indexing in [`ClipTextEncoder.ExtractPooledOutput`](../../src/HartsyInference.Diffusion/Models/TextEncoders/ClipTextEncoder.cs):

```csharp
for (int o = 0; o < projDim; o++)
{
    float sum = 0f;
    int wRow = o * hiddenSize;             // row-major access of [proj, hidden]
    for (int i = 0; i < hiddenSize; i++)
        sum += ePtr[inOffset + i] * wPtr[wRow + i];
    pPtr[b * projDim + o] = sum;
}
```

After the fix, `clip_g_pooled` avg_err drops to **3.6e-3** (270× improvement), `final_pooled` avg_err drops to **2.3e-3** (260× improvement). The remaining drift is the GELU-tanh-vs-erf approximation accumulating across CLIP-G's 32 layers, which is acceptable for visual quality.

**Lesson**: PyTorch `nn.Linear.weight` is canonically `[out, in]` (i.e., transposed relative to "input dim, output dim" intuition). Any C# port that reads weights via raw `(float*)weight.DataPointer` for projection has to encode the **stored** order `[out, in]` and do the GEMM as `Σᵢ x[i] * w[o*in + i]`. If the matrix is square it'll silently work — until you hit a non-symmetric square (text_projection, certain attention projections in models with `kv_dim == q_dim`). Always verify against a 2×3 or 3×2 reference matrix in a unit test, not 768×768. Better still, route through `backend.Linear` which goes through cuBLAS with explicit `CUBLAS_OP_T` — there's no ambiguity. We should consider migrating `ExtractPooledOutput` to use `backend.Linear` in a future cleanup so this class of bug is structurally impossible.

### 33. SD3 / SD3.5 Pipeline VAE OOM at 512×512 (UNet→VAE Eviction Missing)

**Problem**: At 512×512, the SD3 VAE Conv2D im2col temporary buffers exceed available VRAM when the transformer is still resident. SwarmUI was holding ~10 GB at the same time, leaving us ~1.5 GB for the test — not enough for the 2.5 GB SD3.5 Medium FP8 transformer plus VAE im2col.

**Fix**: mirrored the SDXL/Flux pattern from deviation #18 — `Sd3Pipeline.GenerateFromTokens` now calls `_backend.Sync()` followed by `_backend.FreeWeights(_transformer.EnumerateWeights())` (and `_t5.EnumerateWeights()` if T5 is in play) **after** the denoise loop, **before** VAE decode. Backends without a weight cache treat both as no-ops via the default-method fallback on `IBackend`.

### 34. SD3 PatchEmbed `pos_embed` Read as F32 from F16 Storage (Pre-Emptive)

The SD3.5 single-file checkpoint stores `pos_embed.pos_embed` as F16 (1×147456×1536 = 432 MB). `PatchEmbed.AddPositionalEmbedding` does `float* posPtr = (float*)posEmbed.DataPointer` and adds component-by-component to F32 image tokens. With F16 stored as F32, the bytes get bit-reinterpreted (same bug class as #30 in Z-Image's RmsNorm).

This was identified during the layer-by-layer diff investigation but turned out **not** to be load-bearing for SD3.5 Medium specifically — the cosmetic fix (`PatchEmbed.LoadWeights` now does `posEmbed.DType != DType.F32 ? posEmbed.CastTo(DType.F32) : posEmbed`) was applied before we found the real root causes (#31, #32) and didn't change the visual output on its own. Kept because the next checkpoint that lands with non-F32 `pos_embed` would silently break.

### 35. `Sd3ClipL` Config Preset Was Missing — CLIP-L Pooled Returned Null

**Problem**: SD3 needs CLIP-L's pooled output (concatenated with CLIP-G's pooled into a 2048-dim conditioning vector). The Sd35 generation tests were instantiating CLIP-L with `ClipTextEncoderConfig.SdxlClipL`, which reuses `Sd15` and has `ProjectionDim = 0`. With `ProjectionDim = 0`, `ExtractPooledOutput` is gated off and `EncodePenultimate` returns `(hidden, null)`. The pipeline's `ConcatPooled(clipLPooled!, clipGPooled!)` then dereferenced null and threw a `NullReferenceException` at line 382 of `Sd3Pipeline.cs`.

**Fix**: added [`ClipTextEncoderConfig.Sd3ClipL = Sd15 with { ProjectionDim = 768 }`](../../src/HartsyInference.Diffusion/Models/TextEncoders/ClipTextEncoderConfig.cs) and updated [`Sd35GenerationTests.cs`](../../tests/HartsyInference.Diffusion.Tests/Sd35GenerationTests.cs) to use it.

A subtle aside: in the actual SD3.5 Medium FP8 checkpoint, `clip_l.text_projection.weight` is **99.9999% zero** by design — Stability didn't train it (mean = 4.66e-10, std = 3.58e-7, zero fraction = 0.999998). So CLIP-L's contribution to `final_pooled` is intentionally ~zero. The preset still has to project (otherwise pooled is null and `ConcatPooled` crashes), but the actual pooled values are tiny. The model was trained with this layout.

**Lesson**: `ProjectionDim = 0` is "I don't have a text_projection" (SD1.5 CLIP-L has none); `ProjectionDim > 0` is "I do, please compute pooled". Using SDXL's CLIP-L preset for SD3 quietly reuses the SD1.5 fallback. Adding a model-specific preset (`Sd3ClipL`) prevents future ports from making the same mistake. CLIP-L for newer models (SD3, FLUX-text — though they don't use it pooled the same way) all need their own preset that matches what the checkpoint was actually trained with.

---

## Phase 4 — Z-Image CUDA Bring-up Deviations

### 36. Z-Image SwiGLU FFN F16 GEMM Overflow on F32 Activations (CUDA-only, all-black output from step 1 onward)

**Problem**: Z-Image generation on the CUDA backend produced all-black images for both Turbo (4-step, CFG=1) and Base (20-step, CFG=4) at every resolution we tried (512², 1024²). The same generation on the Vulkan backend produced clean photographic output. Pipeline ran without errors — the pre-VAE latent logging revealed the latent had become fully NaN by the end of denoising (`nan=16384` per channel for a 128×128 latent), which then propagated through the VAE to all-black RGB.

**Symptom-level evidence**:
- `[Pre-VAE latent] ch0..15: nan=16384` (every element NaN, all 16 channels)
- `[VAE output] ch0..2: nan=1048576` (NaN propagated through VAE for 1024² output)
- Pipeline completed all denoising steps without exception (no OOM, no kernel error)
- **Vulkan backend produced a real image** at the same prompt/seed (mean=65.9, std=83.4, 256 distinct byte values) — confirming the bug was in CUDA-specific code, not in the Z-Image transformer logic itself

**How it was found** — full troubleshooting journey, in order:

1. **Initial confusion: was it model-specific or extension-specific?** First attempt was to compare `z_image_base-nvfp8-mixed.safetensors` against `z_image_base-bf16.safetensors` against `SwarmUI_Z-Image-Turbo-FP8Mix.safetensors` — all three produced black output. The bug was not checkpoint-format specific. Wasted ~30 min hypothesizing the NVFP8 format had a missing weight_scale handler before noticing the trace was identical across all three checkpoints.

2. **The CUDA-vs-Vulkan A/B test** — the moment of clarity. Ran `ZImageGenerationTests.Turbo_Fp8Mix_GenerateImage_Gpu` (CUDA) → all-black assertion failure. Ran `ZImageVulkanGenerationTest.Turbo_Fp8Mix_GenerateImage_Vulkan` → passed, real image. The bug was definitively isolated to CUDA-specific code.

3. **First wrong probe**: hypothesized a stream-ordered allocator race introduced by `bc23474 stream cua to GPU` — specifically the `cuMemPoolSetAttribute(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, 0)` call in `CudaStreamingWeightCache`'s constructor. Disabled it. **Did not fix the bug**. Reverted the probe. Hypothesis A wrong.

4. **The real localization: `ZImageNaNTrace`** ([src/HartsyInference.Diffusion/Models/Denoisers/ZImageNaNTrace.cs](../../src/HartsyInference.Diffusion/Models/Denoisers/ZImageNaNTrace.cs)). Wrote a stat-logger that runs inside `ZImageTransformer.Forward` at every phase boundary, gated to step 0 only initially, writing one line per phase to `/tmp/zimage_layer_trace.log`. First trace showed step 0 completely clean — no NaN at any of the 38 phases (`t_embedder`, `cap_embedder`, `context_refiner.{0,1}`, `x_embedder`, `noise_refiner.{0,1}`, `concat`, `layers.0` through `layers.29`, `imageSlice`, `final_layer`, `velocity(out)`).

5. **Key clue from step-0-clean**: NaN was being introduced *between* steps. Extended the trace to all 4 steps. Step 1's first 7 phases (caption embedder path, x_embedder path, noise refiners, concat) were clean — first NaN appeared in `layers.0` of step 1, with **only ~3.6% of values being NaN**. That fraction matched 1-of-30 attention heads (1/30 = 3.33%), so the initial guess was a degenerate-head softmax in SDPA. Wrong guess.

6. **Drilling into `ZImageBlock.Forward`**: added a `TracePrefix` field on `ZImageBlock` and instrumented every sub-step of `Forward` (mod{0..3}, norm1, modulated, qkv, q/k/v, qN/kN, q/k/vMh post-rope, attnOut, attnFlat, projected, postAttnNorm, afterAttn, normF1, modulatedF, ffnOut, postFfnNorm). Wired the prefix from `ZImageTransformer.Forward` to enable tracing only on the first 2 main layers (avoiding 30× log size).

7. **The smoking gun**, step 1 layer 0:
   ```
   [ZT s1] L0.modulatedF   abs_mean=0.286   max=14.82                            (clean)
   [ZT s1] L0.ffnOut       abs_mean=1053.91 max=12242  *** NAN=0 INF=9024 ***   (Inf!)
   [ZT s1] L0.postFfnNorm  abs_mean=0.07    max=4.99   *** NAN=9024 ***          (Inf→NaN via RmsNorm)
   ```
   The SwiGLU FFN output (`w2(silu(w1(x)) * w3(x))`) had **9024 +Inf values**, no NaN yet. The next op (CPU `RmsNorm`) sees Inf, computes `sumSq = Inf`, `invRms = 1/sqrt(Inf) = 0`, then `output = Inf × 0 = NaN`. From there NaN propagated through layer 1's QKV Linear and consumed every downstream activation.

**Root cause**: commit `bc23474 stream cua to GPU` (May 3, 2026) replaced `CudaBackend`'s per-operand dtype resolution with a joint `ResolveGemmDtype(a, b)`:

```csharp
// bc23474 version (THE BUG):
private DType ResolveGemmDtype(DType a, DType b)
{
    if (a.IsFp8 || b.IsFp8) return DType.F16;        // ← forces F16 even when input is F32
    if (a == DType.F16 || b == DType.F16) return DType.F16;
    return a == DType.F32 || b == DType.F32 ? DType.F32 : a;
}
```

The intent was to save memory: previously, an F32 activation × FP8 weight resolved to a *F32* GEMM, which forced the FP8 weight to be expanded to a full F32 buffer (151 MB for `proj_mlp` at 1024×1024). The new joint rule cuts that in half by running the whole GEMM in F16, casting the F32 activation down too.

But there's a hidden assumption: that the F32 activation always fits in F16's range (max 65504). For Z-Image's SwiGLU FFN this is **false** in step 1 onward. The intermediate `gated = silu(w1(x)) * w3(x)` is a non-linear amplification — for typical Z-Image activations:
- `w1(x)` produces values up to ~1300 (`sqrt(K=3840) × FP8_max(2.25) × x_max(15) × Fp8ScaleFactor(0.005)` order-of-magnitude)
- `silu(w1(x)) ≈ w1(x)` for large positive inputs
- `w3(x)` similar magnitude
- `gated = silu(w1) × w3` reaches **>65504** for some positions

When `CudaBackend.Linear` then casts `gated` from F32 to F16 to run w2's GEMM, **those positions become +Inf**. The cuBLAS F16 × F16 GEMM with `COMPUTE_32F` accumulates in F32 (which would otherwise be safe), but the *operand* is already Inf, so each affected output element gets Inf × weight = Inf. Step 1's `L0.ffnOut` had 9024 Inf values; step 0's stayed under 65504 by luck (different activations, max ffnOut=3876).

Why Vulkan didn't hit it: `VulkanBackend`'s `DispatchMatmul` derives `gemmDtype` from `output.DType`, not from the inputs (per [PHASE_3_5_DEVIATIONS.md #1](PHASE_3_5_DEVIATIONS.md)). Z-Image's SwiGLU output is F32, so Vulkan ran the GEMM in F32 (or with F32 widening for the gated input via `CastIfNeeded`), no overflow.

Why Flux/SDXL didn't hit it on CUDA: Flux's hot-path activations come out of `LayerNorm` and stay tighter in distribution. Z-Image's gated SwiGLU has no upstream normalization on the gating multiply specifically.

**Fix**: `ResolveGemmDtype(F32, FP8)` now returns **BF16** instead of F16 ([CudaBackend.cs:1134-1148](../../src/HartsyInference.Cuda/CudaBackend.cs#L1134-L1148)). BF16 has F32's full dynamic range (3.4e38), so the F32→16-bit activation cast cannot produce ±Inf. Memory savings vs F32 are preserved (BF16 weight cast is the same size as F16). Ampere's BF16 Tensor Cores are as fast as F16. The new rule:

```csharp
private DType ResolveGemmDtype(DType a, DType b)
{
    if (a.IsFp8 || b.IsFp8)
    {
        // BF16 when paired with F32 — full F32 range, no overflow on SwiGLU's gated tensor.
        // F16 when paired with F16 — keeps the existing fast path; F16 inputs are already in-range.
        return (a == DType.F32 || b == DType.F32) ? DType.BF16 : DType.F16;
    }
    if (a == DType.F16 || b == DType.F16) return DType.F16;
    if (a == DType.BF16 || b == DType.BF16) return DType.BF16;
    return a == DType.F32 || b == DType.F32 ? DType.F32 : a;
}
```

Three supporting changes:
1. Added F8↔BF16 cast paths in `CastOnGpu` (F8→F32→BF16 via temp F32; BF16→F32→F16→F8 on the way back, so over-range values surface as FP8 NaN rather than wrapping silently).
2. Replaced four hardcoded `gemmDtype == DType.F16 ? CUDA_R_16F : CUDA_R_32F` ternaries with a `CublasDataType(DType)` helper that maps F16, BF16, and F32 to their cuBLAS constants.
3. Conv2D's `inputIsF16` flag was left alone — Conv2D paths in Flux/SD3 VAEs don't combine F32 inputs with FP8 weights (VAEs are pure F32 or F16), so the BF16 path is unreachable there. If a future model introduces a mixed FP8 + F32 Conv2D, the im2col kernel dispatch needs an additional BF16 branch.

**Result**: Z-Image-Turbo CUDA generation produces a clean photographic image. `L0.ffnOut` on step 1 stays finite (no Inf), `layers.0..29` stay in-range across all 4 denoising steps, pre-VAE latent has 0 NaN per channel.

**Lesson**: when introducing precision-saving dtype rules for mixed-precision GEMMs, **the dynamic range of the activation matters, not just its declared dtype**. F32 is permissive: any model that accumulates residual signal across many blocks can produce intermediate F32 values that exceed F16's 65504 ceiling, especially through gated multiplications (SwiGLU, GeGLU) where two activations multiply each other. F16's narrow range (5-bit exponent, ±65504) makes it the wrong target for F32→16-bit casts on activations of unknown bound. **BF16 is the correct fallback**: same byte count, F32-equivalent range, Tensor-Core-fast on Ampere+. Default to BF16 for any "narrow the GEMM" optimization that touches an F32 activation.

**Methodology note**: this took 8+ trace-and-rebuild iterations because the bug only appeared at step 1 — step 0 was always clean, so a single-step trace showed nothing wrong. The debugging approach that finally worked:

1. **Backend A/B**: confirmed CUDA-only by running the same test on Vulkan. This eliminates "the model logic is wrong" as a hypothesis class.
2. **Per-phase tracing across all denoising steps**: gating the trace to step 0 was a wrong instinct — bugs that depend on accumulation only appear in step 1+. Default to all-step tracing for any "works once, fails on iteration" symptom.
3. **Drill into the failing phase**: once `layers.0` was identified as the failing phase, sub-instrumented `ZImageBlock.Forward` to trace its internals.
4. **Read what the trace actually says**: `INF=9024` (not NaN!) was the diagnostic clue. NaN can come from many sources; Inf in a cuBLAS output narrows it to "an operand was Inf going in", which immediately points at the F32→F16 cast as the only place an Inf could be introduced from finite inputs.
5. **Don't trust the previous step's clean trace**: step 0's `ffnOut max=3876` was ALSO above the "expected" range — it was just luck that step 0's max stayed under 65504. The fix had to address the structural overflow risk, not the specific values from one step.

Future debugging of similar "compounding instability" bugs should default to the same shape: backend A/B → all-step trace at phase boundaries → drill into the failing phase → look for the dtype/range condition that differs from the working backend.

## Phase 4 — Bucket A Closeout Deviations

### 37. OmniGen 2 RoPE Joint-Mode Missing (joint stack would silently skip rotation)

**Problem**: After the 6-model parallel agent push, OmniGen 2's `OmniGen2Block` exposed three RoPE modes — `None`, `Text`, `Image` — and the joint main-block stack was supposed to use `None` with the comment "RoPE already applied externally (joint stack uses pre-rotated Q/K because text and image positions differ)". But Q/K are computed *inside* the block from the input tensor; the transformer has no way to inject a pre-rotation before the block's Q/K projection runs. The `None` mode in practice meant "no positional encoding at all" for the joint stack — silently wrong, no compile error, no test failure.

**Fix**: added a real `Joint` mode + corresponding `OmniGen2Rope.ApplyJoint(q, k, batch, numQHeads, numKvHeads, txtSeqLen, hPacked, wPacked)`. Block derives `txtSeqLen = seqLen - hPacked * wPacked`, builds the joint cos/sin table (text positions `(s,s,s)`, image positions `(textSeqLen, row, col)`), and rotates both Q and K with GQA-aware head counts. `None` mode kept as a diagnostic-only escape hatch.

**Lesson**: when a Block's design says "do this step externally", verify there's actually a place to do it. RoPE applies to Q/K specifically, after the projection — if the block computes Q/K, the block must rotate them. External RoPE only works when the *upstream* values carry the position info (not the case here).

### 38. OmniGen 2 Block GQA Double-Call Bug

**Problem**: The block called `rope.ApplyText(qMh, kMh, batch, _numQHeads, seqLen)` followed by `rope.ApplyText(qMh, kMh, batch, _numKvHeads, seqLen)`. Each call rotated *both* qMh and kMh with the same head count, but qMh has `_numQHeads` heads and kMh has `_numKvHeads` (different in GQA: 21 vs 7 in the V1 preset). The first call rotated kMh with 21 heads (out-of-bounds writes past the kMh buffer) AND the second call rotated qMh with 7 heads (only 1/3 of qMh got rotated, the rest unchanged).

**Fix**: collapsed to a single `ApplyText(q, k, batch, numQHeads, numKvHeads, seqLen)` call that rotates Q with `numQHeads` heads and K with `numKvHeads` heads using the same precomputed table. Same fix for `ApplyImage` and `ApplyJoint`.

**Lesson**: GQA-aware kernels need separate head counts for Q and K. Functions that take a single `numHeads` parameter and rotate "Q and K" are silently wrong on any non-MHA model.

### 39. Hunyuan Image 2.1 CLIP Encoder Is a Phantom Dependency

**Problem**: The Hunyuan Image 2.1 pipeline ctor signature took a `ClipTextEncoder` parameter, but the transformer has no pooled-conditioning input — the CLIP encoder was never used. The original pipeline body had a TODO comment "produce pooled embedding for AdaLN conditioning" that misdescribed the architecture: Hunyuan Image conditions on timestep + optional distilled-guidance (no pooled vector). The model uses two *per-token* text encoders: a primary 4096-dim encoder (MLLM in the upstream model, T5-XXL in our scaffolding) and an optional 1472-dim secondary (ByT5).

**Fix**: changed the ctor to accept `ClipTextEncoder?` (nullable) and ignored it in the pipeline body. Pipeline now encodes via T5-XXL only and passes `encoderHidden2: null` to the transformer. Tests and downstream code don't break because the parameter is still accepted.

**Lesson**: when porting a model, verify what the transformer's `Forward` actually takes — don't infer text-encoder requirements from the diffusers pipeline class signature, which often pre-encodes more than the transformer needs.

### 40. Gemma 2 Has 4 Norms Per Block + Offset-From-1 RMSNorm Scale

**Problem**: Lumina-Image-2.0 uses Gemma 2 as text encoder, but `LlamaStyleEncoder` was Llama/Qwen-shaped: 2 norms per block (input + post-attn / pre-MLP), SiLU/SwiGLU MLP, RMSNorm scale used as-is. Gemma 2 differs in three load-bearing ways:
1. **4 norms per block**: pre-attn (`input_layernorm`), **post-attn sandwich** (`post_attention_layernorm`, applied to attention output before residual add — Llama applies this name to the pre-MLP norm), **pre-FFN** (`pre_feedforward_layernorm`, the actual pre-MLP norm), **post-FFN sandwich** (`post_feedforward_layernorm`).
2. **GeluTanh activation**, not SiLU. The gated wiring is the same: `down(act(gate) * up)`.
3. **Offset-from-1 RMSNorm scale convention**: stored weights are scale - 1, runtime applies `(1 + weight)`.

**Fix**: added `MlpActivation` enum + `Activation` field, `HasFfnSandwichNorms` bool, `RmsNormScalePlusOne` bool to `LlamaStyleEncoderConfig`. Block conditionally loads two extra norm tensors and applies them around attention output and FFN output respectively. `AddOneInPlace` helper folds the +1 offset at load time so the runtime `RmsNorm` path stays unchanged. Activation switch picks GeluTanh vs SiLU at runtime.

**Not yet implemented** (deferred — small numerical drift on long prompts):
- Attention logit soft-capping (cap softmax pre-softmax dot-products at 50.0 — Gemma 2 specific; needs a custom attention path that doesn't use `ScaledDotProductAttention`).
- Alternating local/global attention with sliding window 4096 (Gemma 2 alternates per-layer).
- Per-query `sqrt(head_dim)` pre-attention scalar.

**Lesson**: "Llama-family" models often disagree on small details that look like cosmetic norm-naming changes but encode real architectural differences. Sandwich-norm patterns (Gemma 2, Cosmos-Predict2) and offset-from-1 scale conventions are common enough that a config-driven flag pattern is a better generalization than per-model encoder classes.

### 41. Hunyuan / OmniGen Patchify In-Patch Order: Channel-Inner

**Problem**: When patchifying `[B, C, H, W]` → `[B, S, p²·C]`, the in-patch ordering matters. The diffusers reference uses `einops.rearrange(x, 'B C (H p) (W q) -> B (H W) (p q C)')` — channel is the **innermost** dimension within each patch (for each `(py, px)` position, all C channels are contiguous). A naive C# `for (c) for (py) for (px)` loop produces the inverse layout, which silently misaligns the `patch_embed.weight` columns.

**Fix**: explicit nested loops `for (py) for (px) for (c)` in both `OmniGen2Transformer.PatchifyLatent` and `HunyuanImagePipeline.PatchifyLatent`. Inverse `UnpatchifyTokens` mirrors the same ordering.

**Lesson**: when porting a DiT, the patch_embed weight layout is a load-bearing convention. Read the einops string (or the equivalent reshape+permute sequence) and translate the dimension order directly.

---

## Phase 4 / 7 — Ideogram 4 Deviations (9.3B single-stream DiT)

Ideogram 4 went from "loads, but 57 min/gen and ignores the prompt" to "36 s/gen, prompt-faithful" via three separate bugs — two performance (GPU residency, then host-bound allocation) and one correctness (encoder concat order). The correctness debugging methodology below generalizes to any DiT whose image is coherent but wrong.

### 42. Ideogram 4 CPU "Glue" Ops Broke GPU Residency (57 min/gen)

**Problem**: First real-weights run took ~57 min / 20 steps on an A100 (~170 s/step) versus seconds in ComfyUI. The big GEMMs and SDPA ran on GPU, but the per-block "glue" — modulation split, scale, gated residual, QK-norm, RoPE, slice/reshape/permute — and the per-step CFG/Euler update ran as CPU loops over `Tensor.DataPointer`. Every `.DataPointer` read on a GPU-resident tensor fires a `cuStreamSynchronize` + D2H copy inside `GpuTransferHelper` (the lazy-sync path), stalling the GPU ~370× per forward.

**Symptoms**: GPU compute-bound work was fast, but wall-clock was dominated by hundreds of tiny D2H syncs/forward; `nvidia-smi` showed the GPU mostly idle, waiting on the host between micro-ops.

**Fix**: added a PTX kernel family [`native/cuda/dit/dit_f32.cu`](../../native/cuda/dit/dit_f32.cu) → `Ptx/dit_f32.ptx` (rmsnorm, affine_broadcast_lastdim, gated_residual_lastdim, modulation4, cfg_euler, tanh, rope, slice_lastdim, row_scale, add_scalar, layernorm_noaffine, index_add, scatter_rows_after, slice_rows), exposed as `IBackend` default-CPU methods + `CudaBackend` GPU overrides. Rewrote `Ideogram4Block`, `Ideogram4Transformer`, and `Ideogram4Pipeline` so every op is a backend op (reshape-to-heads is free — allocate the output with the consumer shape; head permutes reuse `Permute0213`). The forward now does **0 internal D2H syncs**. Added `CudaBackend.GetD2hSyncCount()`/`ResetD2hSyncCount()` as the residency metric, logged per step.

**Lesson**: a model is only GPU-resident if **every** op on the hot path is a backend op. A single `.DataPointer` / `AsSpan` read on a GPU tensor silently forces a full device sync + D2H copy — invisible in correctness tests, lethal to throughput. When porting a DiT, audit the block + pipeline glue (modulation, gating, slicing, CFG/Euler) for direct pointer access, and assert `GetD2hSyncCount() == 0` per forward. This same CPU-glue pattern likely afflicts other DiT ports (Wan/LTX/Lance) — the new kernels are generic and reusable.

### 43. Eager Host-Buffer Allocation Starved the Idle GPU (engine-wide; ~45 s/step → ~2 s/step)

**Problem**: After deviation #42 (0 D2H syncs, weights resident), generation was *still* ~45 s/step on an RTX PRO 6000 Blackwell. `nvidia-smi dmon -s u` during a gen showed the GPU **idle** — 0–10% util, ~90 W (near idle), idle clocks — i.e. host/dispatch-bound, not compute-bound. Root cause was in **Core, not the model**: `Tensor`'s constructor eagerly did `new NativeBuffer(byteSize)`, and `NativeBuffer` does `NativeMemory.AlignedAlloc` **+ `NativeMemory.Clear`** (zeroes every byte, first-touch page-faulting the whole allocation). This ran for *every* tensor, including the thousands of GPU-resident intermediates whose data only ever lives on the device. An Ideogram 4 FFN output is `[1, 4121, 12288]` F32 ≈ **202 MB**; several per block × 34 blocks × 2 transformers × 20 steps = **tens of GB of host malloc + memset + free per step**, all for buffers the host never reads. The CPU memset wall is what the idle GPU was waiting behind.

**Symptoms**: GPU idle (`nvidia-smi` low util / near-idle power) throughout a gen; per-op host time ~16 ms; step time flat regardless of kernel/cuBLAS tuning (the GEMMs were never the bottleneck). Host RAM churns hard during gen.

**Fix**: lazy host-buffer allocation in [`Tensor`](../../src/HartsyInference.Core/Tensors/Tensor.cs). The constructor no longer allocates; a new `internal EnsureHostBuffer()` allocates (zeroed) on the **first CPU access**, and `DataPointer`/`AsSpan`/`AsReadOnlySpan`/`AsRef` route through it (`_ownsLazy`/`_disposed` track ownership/lifetime; `OwnsMemory` now means `_ownsLazy`). Crucially, `GpuTransferHelper.CacheActivation` (both CUDA and Vulkan) no longer reads `tensor.DataPointer` up front — doing so would force the lazy alloc for every activation; instead the D2H **sync callback** calls `EnsureHostBuffer()` only if/when CPU code actually reads the tensor. Net: GPU-resident activations allocate **zero** host memory; CPU-backend zeroing semantics are preserved (the buffer is still zeroed on first access). Result on Blackwell: **~45 s/step → ~2.0 s/step, full gen 50+ min → 36 s** (1024², 12-step turbo), D2H syncs 0/step.

**Lesson**: **GPU idle during a gen (low `nvidia-smi` util) means host/dispatch-bound — diagnose that before tuning kernels or cuBLAS.** Eager host-side allocation + zeroing for device-resident tensors is a silent, engine-wide throughput killer: the cost scales with intermediate tensor sizes, not with anything visible in the model code. Allocate host buffers lazily and never materialize them for tensors that stay on the device. Decisive probe: `nvidia-smi dmon -s u` (idle = this class of bug; pegged = genuinely compute-bound).

### 44. Ideogram 4 Ignored the Prompt — 13-Layer Qwen3-VL Concat Was Transposed (tap-major vs hidden-major)

**Problem**: Images were coherent but ignored the prompt entirely (e.g. "a photo of a cat" → an unrelated scene), **even with a hand-authored structured-JSON caption** in the exact format the model trains on — so it was a real conditioning bug, not the "plain prompts underperform" usage caveat.

**How it was found** (the methodology matters more than the bug):
1. **`cfgΔ` is an invalid conditioning test for asymmetric CFG.** The per-step `cfgΔ` probe (RMS of conditional − unconditional velocity) looked "healthy" (~0.1), which falsely suggested conditioning worked. But Ideogram's two passes use **different networks** (`transformer/` vs `unconditional_transformer/`), so `cfgΔ` is nonzero from the weight difference alone — even if the text is completely ignored. Discard "output changes ⇒ conditioning works" for any dual-network / asymmetric CFG.
2. **Verified the whole architecture against the upstream raw source + the live Qwen3-VL-8B-Instruct HF config** — chat template, tap-layer indices (`[1,4,…,36]` HF), encoder config (θ=5e6, M-RoPE collapses to 1D RoPE for text-only), the masked-add conditioning math, the block-diagonal attention mask (== full attention for a single sample, so `mask=null` is correct), `IMAGE_POSITION_OFFSET=65536`, the indicator constants, and the CFG combine `v = gw·pos + (1−gw)·neg`. **All correct.**
3. **Ran the synthetic-weight parity harness** (`dump_ideogram4_full_forward.py` + `Ideogram4DiffTests`): every transformer tap matched ~1e-8 — the DiT **math is proven correct**. But that harness uses synthetic weights and **skips the encoder**, so it cannot catch encoder or real-weight-load bugs.
4. **Fixed-seed two-prompt test**: a different prompt produced a different image → conditioning was **weak, not dead** (text reaches the model but fails to bind).
5. **Diffed our encoder path against ComfyUI's** [`comfy/text_encoders/ideogram4.py`](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/text_encoders/ideogram4.py) — that surfaced the difference in minutes.

**Root cause**: the channel ordering of the 13-layer Qwen3-VL concat. Both upstream (`permute(torch.stack(taps), (1,2,3,0)).reshape(B,L,-1)`) and ComfyUI (`permute(0,2,3,1).reshape(b,seq,h*n)`) build it **hidden-major**: channel `c = hidden*13 + tap` (the 13 tap values of hidden-dim 0, then of hidden-dim 1, …). Our shared `LlamaStyleEncoder.EncodeMultiLayer` / `ScatterLayerSlice` wrote **tap-major**: `c = tap*4096 + hidden` (all of tap 0, then all of tap 1) — the **transpose**. `llm_cond_norm`'s per-channel weight and `llm_cond_proj`'s input columns were trained on the hidden-major order, so every input channel of the projection was scrambled → weak, garbled conditioning while each individual velocity prediction stayed coherent.

**Fix**: added an opt-in `interleavedLayout` (hidden-major) path to [`EncodeMultiLayer`](../../src/HartsyInference.Diffusion/Models/TextEncoders/LlamaStyleEncoder.cs) (`ScatterLayerSlice` writes `dst[h*K + layerSlot] = src[h]`). Default stays **tap-major** so Flux.2 Klein (which per diffusers uses tap-major) and the single-tap callers (Z-Image, Ernie — for `K=1` the layouts are identical) are untouched; [`Ideogram4Pipeline`](../../src/HartsyInference.Diffusion/Pipelines/Ideogram4Pipeline.cs) passes `interleavedLayout: true`. Also corrected the chat template: Qwen3-VL-Instruct's `apply_chat_template(add_generation_prompt=True)` emits **no `<think>` block** and no default system message (verified against the HF tokenizer_config + upstream `pipeline_ideogram4.py` + ComfyUI). `Qwen3Tokenizer.EncodeChat` got an `includeThinkBlock` flag (default `true` for Flux.2 Klein's Qwen3-text template); the Ideogram 4 loader passes `false`. The think-block was a real mismatch but minor — it's causally appended after the prompt, so it can't corrupt the prompt-token hidden states; the concat transpose was the actual prompt-killer.

**Lesson**:
- **For multi-layer hidden-state taps, the concat ORDER (tap-major vs hidden-major) is a load-bearing convention the projection weight was trained on.** A transpose here produces "coherent image, ignored prompt" and **survives every layer-by-layer math diff** — because the math is right; only the input channel ordering is wrong. Verify the exact `stack/permute/reshape` in the upstream encoder AND cross-check ComfyUI's `comfy/text_encoders/*.py`.
- **A passing synthetic-weight parity harness proves the transformer math, not the encoder or the real-weight load.** When the math is proven but the image is wrong, suspect (in order) the encoder concat/tap order, the tokenizer/chat-template, then the real-weight converter — not the DiT.
- **Diffing directly against ComfyUI's implementation is one of the fastest ways to localize a porting bug** when the upstream is terse or its own docs are ambiguous. (ComfyUI's `permute(0,2,3,1)` vs our `permute(0,2,1,3)` was the whole bug, visible at a glance.)
- For asymmetric / dual-network CFG, build a *real* conditioning probe (same network, real vs zeroed text) — `cfgΔ` across the two networks proves nothing.

### 45. VAE Encoder Downsample Padding Parity — Flux Kontext Maze/Speckle Texture Corruption

**Problem**: Flux.1 Kontext edits through the SwarmUI extension produced deterministic texture corruption: the edit itself succeeded (subject recolored, composition preserved) but flat regions came out wrong — a smooth beige wall rendered as gray maze/squiggle patterns, wood grain as salt-and-pepper dots, speckles on the subject. Byte-stable across every perf-flag combination (F16 and F32 block loops both corrupted, so not precision), and an earlier Kontext repro with a *different* reference photo (dark, textured, no large flat regions) was visually clean — the trigger was image content, not a code path.

**How it was found**:
1. Reproduced at engine level with the exact Swarm inputs (kontext fp8_scaled + side clip_l/t5xxl/ae, same ref PNG, prompt, seed 7, guidance 3.5, 1024²) — pixel-identical maze pattern, so the extension was exonerated.
2. VAE encode→decode round trip of the ref image through our own encoder+decoder looked *clean* — a same-engine round trip cannot see an encode-grid misalignment because the decoder is misalignment-agnostic; only the DiT (trained on reference-encoder latents) is sensitive.
3. **A/B'd the encoded latent against ComfyUI's** (`comfy.sd.VAE` on CPU, same `ae.safetensors`, same pixels): global corr **0.871**, meanAbsDiff 0.44 against a latent std of 1.13 — and the correlation *improved* to 0.919 when our latent was shifted by (+1, +1), the smoking gun for a sampling-grid offset.

**Root cause**: `VaeEncoder`'s stride-2 downsamplers used **symmetric** `Conv2d(k=3, s=2, padding=1)`, while diffusers/BFL/Comfy apply **asymmetric** `F.pad((0,1,0,1))` (right/bottom only) + `Conv2d(k=3, s=2, padding=0)`. Both give the same output SIZE for even inputs, but they sample opposite pixel parities: symmetric centers output pixel *i* on input `2i`, asymmetric on `2i+1`. Each of the three stride-2 stages shifts the sampling grid by one input pixel — compounding to a ~1-latent-pixel phase error and, worse, evaluating every downsample conv half a pixel off the grid its weights were trained on. The resulting latents are subtly off-distribution; Kontext conditions on them as ground-truth reference tokens and reproduces the off-grid texture error as maze/speckle artifacts wherever the reference is flat or finely textured. (Plain img2img hides the same error because denoising pulls the latent back onto the manifold.) The original class doc *knew* about the deviation and dismissed it as "sub-pixel … still in-distribution" — it is neither.

**Fix**: [`VaeEncoder`](../../src/HartsyInference.Diffusion/Models/Vae/VaeEncoder.cs) now zero-pads right/bottom by 1 (two device-side `Concat`s with `Fill`-zeroed edge tensors — no host glue) and runs the downsample conv with `padding=0`. Encoder latent corr vs ComfyUI: **0.871 → 0.999993** (meanAbsDiff 0.44 → 0.0004); the Kontext repro's flat-wall gradient metric dropped from 5.07 (maze) to 1.49 (clean; source is 0.60). Applies to every AutoencoderKL config (SD1.5/SDXL/SD3/Flux share the diffusers encoder recipe), so img2img/inpaint/Fill/Tools encodes all gained accuracy. `VaeTiledEncoder` wraps `VaeEncoder` and inherits the fix. Regression test: [`VaeEncoderDownsamplePaddingTests`](../../tests/HartsyInference.Diffusion.Tests/VaeEncoderDownsamplePaddingTests.cs) locks the pad shape and the odd-pixel-parity sampling.

**Lesson**:
- **"Same output shape" is not "same operator."** A stride-2 conv with symmetric vs asymmetric padding is a different sampling grid — a phase error that no same-engine round trip can reveal. Validate encoders against an external reference implementation, not against your own decoder.
- **Conditioning paths are stricter than denoising paths.** img2img tolerated this encoder for months because the sampler re-projects latents onto the model manifold; Kontext/Fill-style reference conditioning trusts the encode verbatim and amplifies any off-distribution component into visible artifacts.
- Content-dependent triggers (flat/bright regions) can make a systematic numeric bug look like a flaky one-off. When a "clean" repro exists, diff the *inputs*, and pick test images with large flat regions plus fine texture.
### 46. SDXL/SD1.5 IP-Adapter Black Output — Checkpoint Layer Order Is down → up → MID LAST, Not Traversal Order

**Problem**: SDXL base + `ip-adapter_sdxl_vit-h` (standard, weight 0.8) produced a fully **black** 1024×1024 image; the identical request without IPA was perfect. First real-weight run of the IPA forward (only construction/shape unit tests existed).

**How it was found**: engine-level repro (512², 10 steps) confirmed black with healthy projected image tokens (std ≈ 0.32, no NaN) → the corruption was inside the UNet ip-attention branch. A single-UNet-forward diagnostic with **per-layer scale masks** (all / first-half / second-half / single layer) bisected it in four forwards: layers 0–34 finite, layers 35–69 → 100% NaN, and even `scale=1e-6` NaN'd (so the branch itself, not the magnitude).

**Root cause**: the flat `ip_adapter.{i}.to_k_ip/to_v_ip` list in every IPA checkpoint follows diffusers' `attn_processors` enumeration order, which is **down_blocks → up_blocks → mid_block LAST** — diffusers registers the (empty) `up_blocks` ModuleList before constructing `mid_block`, so `named_children()` yields mid after up. Our `UNet.Forward` advanced its IPA cursor in **traversal order** (down → mid → up). From the mid block onward every cross-attn layer got the wrong K_ip/V_ip pair; where the layer widths differ (640 vs 1280) the `Linear` GEMM either **under-filled its output buffer (uninitialized device memory)** or **wrote past it (heap corruption)** → NaN → black. Proof by checkpoint shapes: SDXL `to_k_ip` row dims run `[640×4, 1280×20 | 1280×30, 640×6 | 1280×10]` = down | up | mid; SD1.5 runs `[320×2, 640×2, 1280×2 | 1280×3, 640×3, 320×3 | 1280×1]`.

**Fix** ([UNet.cs](../../src/HartsyInference.Diffusion/Models/Denoisers/UNet.cs), [CrossAttentionBlock.cs](../../src/HartsyInference.Diffusion/Models/Denoisers/UNetBlocks/CrossAttentionBlock.cs), [IpAdapterScaleSchedule.cs](../../src/HartsyInference.Diffusion/Adapters/IpAdapterScaleSchedule.cs)):
- `UNet.Forward` maps its traversal onto the checkpoint order: mid uses the LAST `NumTransformerBlocks` entries; up blocks continue directly after down.
- `TransformerSubBlock` now **fails fast** when `to_k_ip.Shape[0] != layer inner dim` — a misordered list becomes an exception instead of silent NaN/heap corruption.
- The host `AccumulateScaled` (mid-forward `DataPointer` FMA — the documented CPU-glue trap: 2 stream drains per layer per CFG branch) was replaced with device-side `backend.Scale` + `backend.Add`; with-IPA step time dropped 1.4 s → 0.24 s at 512² on the 3060.
- `IpAdapterScaleSchedule.Build` now takes the down/mid segmentation (`UNet.Down/MidCrossAttentionLayerCount`) so the weight-type profiles target the intended blocks under checkpoint ordering.
- Real-weight Integration tests: `IpAdapterGenerationTests` (SDXL standard + Plus, SD1.5 standard; env-gated, assert non-black output stats).

**Verified**: SDXL standard/Plus + SD1.5 standard at 512² produce coherent images clearly driven by the reference (a red-astronaut photo reference reproduced the red suit + desert sand); IPA scale 0 matches no-IPA output; adapter unit tests green.

**Lesson**: **The flat per-layer weight list order in adapter checkpoints is diffusers' `attn_processors` module-registration order (down → up → mid), NOT the forward traversal order.** Verify by dumping the per-index weight shapes and matching the width runs against the UNet's block widths — it proves the ordering in seconds, no Python needed. And when a per-layer injection can carry wrong-shaped weights, validate dims at the injection site: an N-mismatched GEMM silently under-fills or overruns its output buffer.

### 47. Parallel-Suite Native Stomp in LanceLatentPatchTests — Premature Finalization of Undisposed Tensors, Not a Cross-Test Writer

**Problem**: `LanceLatentPatchTests.Patchify_FeatureOrderMatchesUpstreamEinops` failed under xunit class parallelism (`tokens[1]` read `9.8e-45` = raw bits `0x7` instead of `1.0`) whenever IpAdapter/ControlNet/Lens test classes ran concurrently; serial runs (`ParallelizeTestCollections=false`) always passed. Looked exactly like another test class committing a write-after-free into reused native memory.

**How it was found**: class-set bisection kept implicating *every* co-running class (leave-one-out never converged) — the "offender" was interchangeable, which fits a GC-pressure trigger, not a specific stale pointer. A standalone loop repro of the suspect bodies ran 22M iterations clean; the loop version differed from the real test in exactly one way: it Disposed its tensors. A one-shot repro of the test body with **no Dispose** and a forced `GC.Collect()+WaitForPendingFinalizers()` between the `feat=0` and `feat=1` asserts corrupted `tokens[1]` to bits `0x00000007` **deterministically, single-threaded** — byte-identical to the suite failure.

**Root cause**: the test never disposed `latent`/`tokens`. After `tokens.DataPointer` is captured into a raw `float*`, the `Tensor` object is JIT-dead; a GC (triggered in-suite by a parallel class's ~100MB safetensors-fixture allocations landing between two `Assert.Equal` calls) finalizes it, `~Tensor → NativeBuffer.Dispose → AlignedFree` frees the live 32-byte buffer on the finalizer thread, and glibc's `free()` writes the tcache mangled-next pointer into bytes 0–7 of the chunk — `(addr >> 12)` of a `0x7fxx…` thread-heap address has high dword `0x7`, which lands precisely in `tokens[1]` (`tokens[0]` was asserted before the GC, so the failure always surfaced at feature index 1). The co-running classes were never writers — they were only allocation pressure, which is why bisection couldn't converge and why sibling branches "passed" on codegen/timing luck.

**Fix**: `using` declarations on every Tensor in [LanceLatentPatchTests](../../tests/HartsyInference.Diffusion.Tests/LanceLatentPatchTests.cs) — the pending `Dispose()` roots each tensor for the whole scope (and honors the engine's "creator disposes Tensor" contract); plus `GC.KeepAlive` hardening in [LanceLatentPatch](../../src/HartsyInference.Diffusion/Models/Denoisers/DiTBlocks/LanceLatentPatch.cs) `Patchify`/`Unpatchify`, whose copy loops read through the input tensor's raw pointer after its last object use (same premature-finalization class as `Tensor.Reshape`'s Qwen3-TTS fix). Verified: original 103-test parallel repro filter green 5×/5; minimal 3-test repro green 5×/5; full Diffusion unit tier no worse than baseline (0–1 fallout failures vs 10–11 on unfixed main; the tier's pre-existing heap-corruption abort in the ACE-Step region is untouched/out of scope).

**Lesson**: **A raw `DataPointer`/`AsSpan` does not root a Tensor — an undisposed tensor can be finalized (and its buffer freed) while its pointer is still being read.** When a "parallel-only" native stomp implicates whichever class co-runs, suspect GC pressure + premature finalization of the *victim's own* objects before hunting a cross-test writer; a forced GC between the write and the read of the victim buffer proves it in one run. glibc fingerprint: corrupt value = tcache metadata (`addr>>12` high dword ≈ `0x7`) at chunk offset 0–7.
