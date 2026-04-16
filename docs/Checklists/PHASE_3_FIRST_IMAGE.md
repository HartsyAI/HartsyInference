# Phase 3 — First Image (Cuda + SD1.5 Pipeline)

> **Goal:** Generate an actual image from text with Stable Diffusion 1.5 on a CUDA GPU.
> **Packages:** SharpInference.Cuda, SharpInference.Diffusion (full)

---

## 1. Research

- [x] Complete [CUDA_DRIVER_API.md](../Research/CUDA_DRIVER_API.md) research — done and verified
- [x] Complete [PTX_KERNELS.md](../Research/PTX_KERNELS.md) research — done and verified
- [x] Complete [CONV2D_CUDA.md](../Research/CONV2D_CUDA.md) research — done and verified
- [ ] Complete [SD15_ARCHITECTURE.md](../Research/SD15_ARCHITECTURE.md) research — **still Draft**
- [ ] Review dotLLM CUDA backend source for patterns to follow

## 2. Planning

- [ ] Map exact SD1.5 UNet block structure (channels, attention heads per level)
- [ ] Plan CUDA memory pool strategy (pre-allocate vs on-demand)
- [ ] Plan PTX kernel embedding and JIT compilation flow
- [ ] Determine cuDNN version to target
- [ ] Plan end-to-end pipeline data flow (text → tokens → embeddings → latent → denoise → VAE → image)
- [ ] Write agent instructions for Phase 3

## 3. Implementation — SharpInference.Cuda

- [ ] `CudaDriver.cs` — P/Invoke for cuInit, cuDeviceGet, cuCtxCreate, cuModuleLoadData, cuLaunchKernel
- [ ] `CudaStream.cs` — stream creation, synchronization, lifecycle
- [ ] `CudaMemoryPool.cs` — cuMemPool-based async allocation
- [ ] `PtxKernelLoader.cs` — embed PTX as resources, JIT compile, cache CuFunction handles
- [ ] `CuBlasWrapper.cs` — HGEMM, SGEMM P/Invoke bindings
- [ ] `CuDnnWrapper.cs` — Conv2D forward via cuDNN P/Invoke
- [ ] `CudaBackend.cs` — `IBackend` implementation routing to PTX + cuBLAS + cuDNN
- [ ] PTX: `conv2d_f16_3x3.ptx` — 3×3 convolution, FP16, tiled shared memory
- [ ] PTX: `conv2d_f16_1x1.ptx` — 1×1 convolution (fused matmul)
- [ ] PTX: `group_norm_f16.ptx` — GroupNorm forward
- [ ] PTX: `layer_norm_f16.ptx` — LayerNorm forward
- [ ] PTX: `sdpa_f16.ptx` — scaled dot-product attention, tiled
- [ ] PTX: `upsample2d_nearest.ptx` — nearest-neighbor upsample
- [ ] PTX: `elementwise_f16.ptx` — fused add/mul/scale/gelu/silu
- [ ] PTX: `dequant_q8.ptx` — Q8_0 dequantize on GPU

## 4. Implementation — SD1.5 Pipeline

- [ ] `ClipTextEncoder.cs` — full CLIP transformer forward pass
- [ ] `ResNetBlock.cs` — GroupNorm → SiLU → Conv2D → GroupNorm → SiLU → Conv2D + residual
- [ ] `CrossAttentionBlock.cs` — LayerNorm → self-attn → LayerNorm → cross-attn → LayerNorm → FFN
- [ ] `DownSampleBlock.cs` — strided Conv2D downsample
- [ ] `UpSampleBlock.cs` — nearest upsample + Conv2D
- [ ] `UNet.cs` — full SD1.5 UNet (4 down, 1 mid, 4 up, skip connections)
- [ ] `StableDiffusion15Pipeline.cs` — end-to-end: tokenize → encode → noise → denoise loop → VAE decode
- [ ] `PipelineFactory.cs` — auto-detect SD1.5 from model metadata
- [ ] `TextToImageRequest.cs` — prompt, negative prompt, size, steps, cfg, seed, scheduler
- [ ] `ImageToImageRequest.cs` — source image, strength, prompt
- [ ] `InpaintRequest.cs` — image, mask, prompt
- [ ] `ImagePostProcessor.cs` — tensor → PNG bytes conversion

## 5. Testing & Validation

- [ ] CUDA kernel unit tests — matmul output matches CPU backend
- [ ] CUDA kernel unit tests — conv2d output matches CPU backend
- [ ] CUDA kernel unit tests — groupnorm output matches CPU backend
- [ ] CLIP text encoder — same tokens → same hidden states as diffusers (within 1e-3 FP16)
- [ ] UNet forward pass — same inputs → same output as diffusers (within 1e-3 FP16)
- [ ] Full SD1.5 pipeline — fixed seed + prompt → visually identical to diffusers
- [ ] Full SD1.5 pipeline — SSIM > 0.95 vs Python reference image
- [ ] img2img pipeline test with known input/output pair
- [ ] Inpainting pipeline test with known input/mask/output
- [ ] Memory leak test — run 100 generations, verify VRAM usage stable
- [ ] All tests pass on GPU CI

## 6. Review & Merge

- [ ] Code review — CUDA error handling (check every CUDA call return code)
- [ ] Code review — memory safety (GPU allocations properly freed)
- [ ] Benchmark: measure it/s for SD1.5 512×512 20-step, compare to Python diffusers
- [ ] Document any performance gaps and optimization opportunities
- [ ] Merge to main branch
