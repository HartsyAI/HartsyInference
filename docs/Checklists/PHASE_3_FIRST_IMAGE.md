# Phase 3 — First Image (Cuda + SD1.5 Pipeline)

> **Goal:** Generate an actual image from text with SD1.5 on CUDA GPU.
> **Packages:** HartsyInference.Cuda, HartsyInference.Diffusion (full)
>
> **Status:** CUDA backend functional with FP16 inference + fused kernels. SD1.5 + SDXL generating correct images on GPU. SDXL 1024x1024 at ~5.5s/step (11x faster than Phase 2).

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

## 3. Implementation — HartsyInference.Cuda

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
- [x] GPU-resident activations — `CacheActivation` + lazy-sync, `FreeAsync` stream-ordered cleanup
- [x] GPU reshape/permute kernels — `transpose_2d_f32`, `permute_0213_f32`, `geglu_f32`, `broadcast_add_f32`
- [x] Remove per-op Sync + async execution — eliminated `cuStreamSynchronize` from all 15 ops
- [x] FP16 PTX kernels (11 files): elementwise, groupnorm, layernorm, softmax, spatial, transpose, geglu, broadcast_add, cast, groupnorm_silu_f16, groupnorm_silu_f32
- [x] FP16 cuBLAS — `cublasGemmEx` with `CUDA_R_16F` + `CUBLAS_COMPUTE_32F` for all GEMM ops
- [x] Mixed-dtype safety — automatic operand casting when input/weight dtypes differ (cuBLAS requires uniform A/B types)
- [x] Kernel fusion — fused GroupNorm+SiLU kernel (~40 fusions per UNet step)
- [x] Model dtype propagation — all model code uses `input.DType` instead of hardcoded `DType.F32`
- [x] Pipeline F16/F32 boundaries — scheduler stays F32, CLIP stays F32; cast at UNet/VAE entry/exit
- [ ] `CudaMemoryPool.cs` — `cuMemPool` async allocator for activation memory (optimization)

## 4. Implementation — SD1.5 Pipeline (CPU path) — COMPLETE

- [x] `ClipTextEncoder.cs`, `UNetResNetBlock.cs`, `CrossAttentionBlock.cs`
- [x] `DownBlock.cs`, `UpBlock.cs`, `UNet.cs` (4 down, 1 mid, 4 up, skip connections, timestep embedding)
- [x] `EulerDiscreteScheduler.cs` (leading timestep, epsilon prediction, scale_model_input)
- [x] `StableDiffusion15Pipeline.cs` — end-to-end: tokenize → encode → noise → denoise → VAE → image
- [x] `VaeDecoder.cs`, `TextToImageRequest.cs`, `ImagePostProcessor.cs`
- [x] `ImageToImageRequest.cs` — exists, used by every pipeline that supports img2img/inpaint. Inpaint is enabled by setting `ImageToImageRequest.Mask` (no separate request type). Validation centralized in `Utilities/Img2ImgSetup.cs`.
- [ ] `PipelineFactory.cs` — scaffolding only ([Pipelines/PipelineFactory.cs](../../src/HartsyInference.Diffusion/Pipelines/PipelineFactory.cs)). `LoadAuto` throws `NotImplementedException` because a real factory needs 5 unresolved design decisions (model-type detection, on-disk layout discovery, tokenizer ownership, quality profile, instance caching). Documented in the class header. Callers currently construct pipelines directly via per-pipeline constructors — every test in `HartsyInference.Diffusion.Tests` demonstrates the pattern. Reopen this item once there's a real consumer (the SwarmUI backend extension) ready to drive the design.

## 5. Testing & Validation

- [x] CLIP text encoder matches diffusers hidden states
- [x] UNet forward pass matches diffusers (avg_err < 1e-3, all layers < 1e-4)
- [x] Full SD1.5 pipeline generates coherent images (CPU)
- [x] SD1.5 UNet GPU forward pass: avg_err=5.188E-007 (vs CPU reference)
- [x] SDXL UNet GPU forward pass: avg_err=5.510E-007, max_err=8.821E-006
- [x] SDXL F32 GPU 256x256 generation: passes, ~4.2s/step steady-state
- [x] SDXL F32 GPU 1024x1024 generation: passes, ~62s/step steady-state
- [x] SDXL F16 GPU 256x256 generation: passes, ~580ms/step steady-state (7.2x faster than F32)
- [x] SDXL F16 GPU 1024x1024 generation: passes, ~5.5s/step steady-state (11x faster than F32), 173s total for 20 steps
- [x] F16 output visually matches F32 reference (same composition, colors, structure)
- [x] F32 regression test: passes after all F16 changes (no breakage)
- [ ] CUDA kernel unit tests (matmul, conv2d, groupnorm vs CPU) — individual op tests
- [ ] Full pipeline SSIM > 0.95 vs Python reference
- [ ] img2img, inpainting, memory leak test
- [ ] All tests pass on GPU CI

## 6. Deviations

See [PHASE_3_DEVIATIONS.md](PHASE_3_DEVIATIONS.md) — 22 deviations documented (7 CPU pipeline bugs + 4 CUDA backend decisions + 4 GPU weight cache bugs + 4 Phase 2 async/kernel bugs + 3 Phase 3 FP16 mixed-dtype bugs) with full troubleshooting methodology.

## 7. Review & Merge

- [ ] Code review (CUDA error handling, GPU memory safety)
- [ ] Benchmark SD1.5 512x512 20-step it/s vs Python
- [ ] Further optimization: CUDA memory pool, async H2D copies (see CUDA_PERFORMANCE.md)
- [ ] Merge to main branch
