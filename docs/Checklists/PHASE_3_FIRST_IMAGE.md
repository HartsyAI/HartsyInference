# Phase 3 — First Image (Cuda + SD1.5 Pipeline)

> **Goal:** Generate an actual image from text with Stable Diffusion 1.5 on a CUDA GPU.
> **Packages:** SharpInference.Cuda, SharpInference.Diffusion (full)
>
> **Status:** CPU pipeline is functional and generating coherent images. CUDA backend not yet started.

---

## 1. Research

- [x] Complete [CUDA_DRIVER_API.md](../Research/CUDA_DRIVER_API.md) research — done and verified
- [x] Complete [PTX_KERNELS.md](../Research/PTX_KERNELS.md) research — done and verified
- [x] Complete [CONV2D_CUDA.md](../Research/CONV2D_CUDA.md) research — done and verified
- [ ] Complete [SD15_ARCHITECTURE.md](../Research/SD15_ARCHITECTURE.md) research — **still Draft** (implementation complete despite this)
- [ ] Review dotLLM CUDA backend source for patterns to follow

## 2. Planning

- [x] Map exact SD1.5 UNet block structure (channels, attention heads per level) — verified against diffusers: 8 heads per block, headDim=channels/8
- [ ] Plan CUDA memory pool strategy (pre-allocate vs on-demand)
- [ ] Plan PTX kernel embedding and JIT compilation flow
- [ ] Determine cuDNN version to target
- [x] Plan end-to-end pipeline data flow (text → tokens → embeddings → latent → denoise → VAE → image) — implemented and working
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

## 4. Implementation — SD1.5 Pipeline (CPU path)

- [x] `ClipTextEncoder.cs` — full CLIP transformer forward pass
- [x] `UNetResNetBlock.cs` — GroupNorm → SiLU → Conv2D → GroupNorm → SiLU → Conv2D + time_emb + residual
- [x] `CrossAttentionBlock.cs` — GroupNorm → proj_in → self-attn → cross-attn → GEGLU FFN → proj_out → residual
- [x] `DownBlock.cs` — ResNet layers + optional CrossAttention + optional strided Conv2D downsample
- [x] `UpBlock.cs` — ResNet layers (with skip concat) + optional CrossAttention + optional nearest upsample + Conv2D
- [x] `UNet.cs` — full SD1.5 UNet (4 down, 1 mid, 4 up, skip connections, timestep embedding)
- [x] `EulerDiscreteScheduler.cs` — leading timestep spacing, epsilon prediction, scale_model_input
- [x] `StableDiffusion15Pipeline.cs` — end-to-end: tokenize → encode → noise → denoise loop → VAE decode
- [x] `VaeDecoder.cs` — latent → RGB decode
- [x] `TextToImageRequest.cs` — prompt, negative prompt, size, steps, cfg, seed, scheduler
- [x] `ImagePostProcessor.cs` — tensor → BMP conversion
- [ ] `PipelineFactory.cs` — auto-detect SD1.5 from model metadata
- [ ] `ImageToImageRequest.cs` — source image, strength, prompt
- [ ] `InpaintRequest.cs` — image, mask, prompt

## 5. Testing & Validation

- [ ] CUDA kernel unit tests — matmul output matches CPU backend
- [ ] CUDA kernel unit tests — conv2d output matches CPU backend
- [ ] CUDA kernel unit tests — groupnorm output matches CPU backend
- [x] CLIP text encoder — same tokens → same hidden states as diffusers
- [x] UNet forward pass — same inputs → same output as diffusers (avg_err < 1e-3, all layers < 1e-4)
- [x] Full SD1.5 pipeline — generates coherent images matching prompt semantics
- [ ] Full SD1.5 pipeline — SSIM > 0.95 vs Python reference image (requires shared noise)
- [ ] img2img pipeline test with known input/output pair
- [ ] Inpainting pipeline test with known input/mask/output
- [ ] Memory leak test — run 100 generations, verify VRAM usage stable
- [ ] All tests pass on GPU CI

## 6. Known Issues & Deviations

See [PHASE_3_DEVIATIONS.md](PHASE_3_DEVIATIONS.md) for the full list of bugs found and fixed (7 deviations documented with troubleshooting methodology).

Key items resolved:
- Attention head count inverted (8 heads not 40) — see deviation #3
- Self-attention K/V from un-normed hidden — see deviation #2
- BatchedMatMul silent zeros on 2D weights — see deviation #1
- GELU: C# uses tanh approximation, Python uses erf — small but acceptable difference

## 7. Review & Merge

- [ ] Code review — CUDA error handling (check every CUDA call return code)
- [ ] Code review — memory safety (GPU allocations properly freed)
- [ ] Benchmark: measure it/s for SD1.5 512×512 20-step, compare to Python diffusers
- [ ] Document any performance gaps and optimization opportunities
- [ ] Merge to main branch
