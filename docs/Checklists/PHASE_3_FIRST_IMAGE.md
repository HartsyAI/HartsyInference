# Phase 3 — First Image (Cuda + SD1.5 Pipeline)

> **Goal:** Generate an actual image from text with SD1.5 on CUDA GPU.
> **Packages:** SharpInference.Cuda, SharpInference.Diffusion (full)
>
> **Status:** CPU pipeline functional and generating coherent images. CUDA backend not yet started.

---

## 1. Research

- [x] CUDA_DRIVER_API, PTX_KERNELS, CONV2D_CUDA
- [ ] SD15_ARCHITECTURE — **still Draft** (implementation complete despite this)
- [ ] dotLLM CUDA backend source review

## 2. Planning

- [x] SD1.5 UNet block structure mapped (8 heads/block, headDim=channels/8)
- [x] End-to-end pipeline data flow implemented
- [ ] CUDA memory pool strategy
- [ ] PTX kernel embedding and JIT flow
- [ ] cuDNN version target

## 3. Implementation — SharpInference.Cuda

- [ ] `CudaDriver.cs` — P/Invoke (cuInit, cuDeviceGet, cuCtxCreate, cuModuleLoadData, cuLaunchKernel)
- [ ] `CudaStream.cs`, `CudaMemoryPool.cs`, `PtxKernelLoader.cs`
- [ ] `CuBlasWrapper.cs` (HGEMM, SGEMM), `CuDnnWrapper.cs` (Conv2D)
- [ ] `CudaBackend.cs` — `IBackend` implementation
- [ ] PTX kernels: conv2d_f16_3x3, conv2d_f16_1x1, group_norm_f16, layer_norm_f16, sdpa_f16, upsample2d_nearest, elementwise_f16, dequant_q8

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
- [x] Full SD1.5 pipeline generates coherent images
- [ ] CUDA kernel unit tests (matmul, conv2d, groupnorm vs CPU)
- [ ] Full pipeline SSIM > 0.95 vs Python reference
- [ ] img2img, inpainting, memory leak test
- [ ] All tests pass on GPU CI

## 6. Deviations

See [PHASE_3_DEVIATIONS.md](PHASE_3_DEVIATIONS.md) — 11 deviations documented (7 CPU pipeline bugs + 4 CUDA backend decisions) with full troubleshooting methodology.

## 7. Review & Merge

- [ ] Code review (CUDA error handling, GPU memory safety)
- [ ] Benchmark SD1.5 512x512 20-step it/s vs Python
- [ ] Merge to main branch
