# Phase 3 — First Image (Cuda + SD1.5 Pipeline)

> **Goal:** Generate an actual image from text with SD1.5 on CUDA GPU.
> **Packages:** SharpInference.Cuda, SharpInference.Diffusion (full)
>
> **Status:** CUDA backend functional. SD1.5 + SDXL generating correct images on GPU via auto-transfer pattern. Performance optimization in progress (see `docs/Research/CUDA_PERFORMANCE.md`).

---

## 1. Research

- [x] CUDA_DRIVER_API, PTX_KERNELS, CONV2D_CUDA
- [x] CUDA_PERFORMANCE — bottleneck analysis, optimization roadmap
- [ ] SD15_ARCHITECTURE — **still Draft** (implementation complete despite this)
- [ ] dotLLM CUDA backend source review

## 2. Planning

- [x] SD1.5 UNet block structure mapped (8 heads/block, headDim=channels/8)
- [x] End-to-end pipeline data flow implemented
- [x] PTX kernel JIT flow — embedded resources, `cuModuleLoadData`, `nint` function handles
- [ ] CUDA memory pool strategy — see CUDA_PERFORMANCE.md Phase D
- [ ] cuDNN version target — see CUDA_PERFORMANCE.md Phase E (optional)

## 3. Implementation — SharpInference.Cuda

- [x] `CudaDriverApi.cs` — P/Invoke (cuInit, cuDeviceGet, cuCtxCreate, cuModuleLoadData, cuLaunchKernel, cuMemAlloc/Free, cuMemcpy)
- [x] `CudaStream.cs` — stream lifecycle, blocking mode (non-blocking causes race conditions, see CUDA_PERFORMANCE.md)
- [x] `CublasApi.cs` — cuBLAS SGEMM via P/Invoke, handle bound to stream
- [x] `CudaBackend.cs` — `IBackend` implementation (auto-transfer pattern: H2D → kernel → D2H per op)
- [x] `GpuTransferHelper.cs` — device memory management + GPU weight cache (`Dictionary<Tensor, ulong>`)
- [x] `CudaKernels.cs` — kernel function handles as `nint` fields, launch wrappers
- [x] PTX kernels (FP32): `elementwise_f32.ptx` (add, scale, silu, gelu, sigmoid, clamp), `spatial_f32.ptx` (im2col, upsample_nearest2d, col2bias_add), `norm_f32.ptx` (groupnorm, layernorm), `sdpa_f32.ptx` (scaled dot-product attention with softmax)
- [x] Conv2D via im2col (PTX) + cuBLAS SGEMM (no cuDNN dependency)
- [x] GPU weight preloading API: `PreloadWeights()`, `FreePreloadedWeights()`, `EnumerateWeights()` on all model classes
- [x] Integer overflow fixes for 1024x1024 resolution (64-bit arithmetic in C# and PTX)
- [ ] FP16 PTX kernels (conv2d_f16, group_norm_f16, sdpa_f16, etc.)
- [ ] GPU-resident activations (eliminate per-op H2D/D2H round-trips)
- [ ] Kernel fusion (GroupNorm+SiLU, Conv2D+Bias+SiLU)
- [ ] `CudaMemoryPool.cs` — `cuMemPool` async allocator for activation memory

## 4. Implementation — SD1.5 Pipeline (CPU path) — COMPLETE

- [x] `ClipTextEncoder.cs`, `UNetResNetBlock.cs`, `CrossAttentionBlock.cs`
- [x] `DownBlock.cs`, `UpBlock.cs`, `UNet.cs` (4 down, 1 mid, 4 up, skip connections, timestep embedding)
- [x] `EulerDiscreteScheduler.cs` (leading timestep, epsilon prediction, scale_model_input)
- [x] `StableDiffusion15Pipeline.cs` — end-to-end: tokenize → encode → noise → denoise → VAE → image
- [x] `VaeDecoder.cs`, `TextToImageRequest.cs`, `ImagePostProcessor.cs`
- [ ] `PipelineFactory.cs`, `ImageToImageRequest.cs`, `InpaintRequest.cs`

## 5. Testing & Validation

- [x] CLIP text encoder matches diffusers hidden states
- [x] UNet forward pass matches diffusers (avg_err < 1e-3, all layers < 1e-4)
- [x] Full SD1.5 pipeline generates coherent images (CPU)
- [x] SD1.5 UNet GPU forward pass: avg_err=5.188E-007 (vs CPU reference)
- [x] SDXL UNet GPU forward pass: avg_err=5.510E-007, max_err=8.821E-006
- [x] SDXL GPU 256x256 generation: passes, ~64s for 10 steps
- [x] SDXL GPU 1024x1024 generation: passes, ~36min for 20 steps (auto-transfer limited)
- [ ] CUDA kernel unit tests (matmul, conv2d, groupnorm vs CPU) — individual op tests
- [ ] Full pipeline SSIM > 0.95 vs Python reference
- [ ] img2img, inpainting, memory leak test
- [ ] All tests pass on GPU CI
- [ ] Benchmark after GPU-resident activations: target <5s/step for 1024x1024

## 6. Deviations

See [PHASE_3_DEVIATIONS.md](PHASE_3_DEVIATIONS.md) — 15 deviations documented (7 CPU pipeline bugs + 4 CUDA backend decisions + 4 GPU weight cache bugs) with full troubleshooting methodology.

## 7. Review & Merge

- [ ] Code review (CUDA error handling, GPU memory safety)
- [ ] Benchmark SD1.5 512x512 20-step it/s vs Python
- [ ] Performance optimization: GPU-resident activations (see CUDA_PERFORMANCE.md)
- [ ] Merge to main branch
