# Phase 3 Deviations — CUDA Backend & Inference Fixes

## 1. TimestepEmbedding Frequency Divisor Fix

**File:** `src/SharpInference.Diffusion/Models/Denoisers/UNetBlocks/TimestepEmbedding.cs`

**Before:** `float logBase = -MathF.Log(10000.0f) / halfDim;`
**After:** `float logBase = -MathF.Log(10000.0f) / (halfDim - 1);`

**Reason:** Diffusers uses `/ (half_dim - 1)` in `get_timestep_embedding()`. Our code used `/ halfDim`, which caused the highest frequency component to be ~6% off from the reference. This ensures the frequency range spans exactly `[1, 1/10000]` matching the original sinusoidal position encoding formulation.

## 2. CUDA SDPA Softmax — CPU Roundtrip

**File:** `src/SharpInference.Cuda/CudaBackend.cs`

**Deviation:** The CUDA ScaledDotProductAttention implementation uses a CPU roundtrip for the softmax step. QK^T and attention @ V are done via cuBLAS SGEMM, but the softmax is computed by downloading scores to host, running softmax in a tight loop, then uploading back.

**Reason:** Writing a correct, numerically stable softmax PTX kernel with proper shared memory reductions is non-trivial. This is a temporary measure to get the full pipeline running on GPU. A pure-PTX softmax kernel should be implemented for production performance.

## 3. CUDA Conv2D — Im2Col + cuBLAS SGEMM

**File:** `src/SharpInference.Cuda/CudaBackend.cs`, `src/SharpInference.Cuda/Ptx/spatial_f32.ptx`

**Deviation:** Conv2D is implemented via im2col (PTX kernel) + cuBLAS SGEMM, rather than cuDNN. This approach allocates a temporary column buffer per forward pass (freed after each call). For 1x1 convolutions (stride=1, no padding), the im2col step is skipped and input data is used directly.

**Reason:** Avoids cuDNN dependency, keeping the project pure C# + CUDA Driver API + cuBLAS. Performance is reasonable for inference workloads, though cuDNN's optimized implementations would be faster for production.

## 4. CUDA GroupNorm/LayerNorm — Three-Pass Kernels

**Files:** `src/SharpInference.Cuda/Ptx/groupnorm_f32.ptx`, `src/SharpInference.Cuda/Ptx/layernorm_f32.ptx`

**Deviation:** Both normalization kernels use a three-pass approach (compute mean, compute variance, normalize+affine) with shared memory reductions. This requires 3 global memory traversals per normalization.

**Reason:** Simpler to implement correctly than online (Welford) single-pass algorithms. Performance impact is minimal since these kernels are not the bottleneck compared to GEMM operations.
