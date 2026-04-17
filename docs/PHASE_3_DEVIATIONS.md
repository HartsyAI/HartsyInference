# Phase 3 Deviations — CUDA Backend & Inference Fixes

## 1. TimestepEmbedding Frequency Divisor Fix

**File:** `src/SharpInference.Diffusion/Models/Denoisers/UNetBlocks/TimestepEmbedding.cs`

**Before:** `float logBase = -MathF.Log(10000.0f) / halfDim;`
**After:** `float logBase = -MathF.Log(10000.0f) / (halfDim - 1);`

**Reason:** Diffusers uses `/ (half_dim - 1)` in `get_timestep_embedding()`. Our code used `/ halfDim`, which caused the highest frequency component to be ~6% off from the reference. This ensures the frequency range spans exactly `[1, 1/10000]` matching the original sinusoidal position encoding formulation.

## 2. CLIP Text Encoder — Missing Final LayerNorm

**File:** `src/SharpInference.Diffusion/Models/TextEncoders/ClipTextEncoder.cs`

**Before:** `Encode()` returned the raw last transformer layer output without applying `final_layer_norm`.
**After:** `Encode()` applies `final_layer_norm` to the encoder output before returning, matching HuggingFace `CLIPTextTransformer.forward()`.

**Reason:** The original code incorrectly assumed SD1.5 conditioning uses un-normed hidden states. In reality, HuggingFace `CLIPTextModel` applies `self.final_layer_norm()` to the encoder output before returning `last_hidden_state`. Without this, text embeddings had std ~5 instead of ~1, causing 5x amplified conditioning signals that produced abstract patterns instead of coherent images.

## 3. CUDA SDPA Softmax — PTX Kernel (Resolved)

**Files:** `src/SharpInference.Cuda/Ptx/softmax_f32.ptx`, `src/SharpInference.Cuda/CudaKernels.cs`

**Previous deviation:** The softmax step used a CPU roundtrip (download scores → host softmax → upload). This has been replaced with a pure-PTX numerically stable per-row softmax kernel using shared memory reductions (3-pass: max → exp+sum → normalize). One block per row, blockDim=256. Uses `ex2.approx.f32` for exp and `rcp.approx.f32` for 1/sum.

## 4. CUDA Conv2D — Im2Col + cuBLAS SGEMM

**File:** `src/SharpInference.Cuda/CudaBackend.cs`, `src/SharpInference.Cuda/Ptx/spatial_f32.ptx`

**Deviation:** Conv2D is implemented via im2col (PTX kernel) + cuBLAS SGEMM, rather than cuDNN. This approach allocates a temporary column buffer per forward pass (freed after each call). For 1x1 convolutions (stride=1, no padding), the im2col step is skipped and input data is used directly.

**Reason:** Avoids cuDNN dependency, keeping the project pure C# + CUDA Driver API + cuBLAS. Performance is reasonable for inference workloads, though cuDNN's optimized implementations would be faster for production.

## 5. CUDA GroupNorm/LayerNorm — Three-Pass Kernels

**Files:** `src/SharpInference.Cuda/Ptx/groupnorm_f32.ptx`, `src/SharpInference.Cuda/Ptx/layernorm_f32.ptx`

**Deviation:** Both normalization kernels use a three-pass approach (compute mean, compute variance, normalize+affine) with shared memory reductions. This requires 3 global memory traversals per normalization.

**Reason:** Simpler to implement correctly than online (Welford) single-pass algorithms. Performance impact is minimal since these kernels are not the bottleneck compared to GEMM operations.
