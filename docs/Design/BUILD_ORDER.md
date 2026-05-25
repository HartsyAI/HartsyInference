# Build Order & Phases

> Back to [Core Design](CORE_DESIGN.md)

## Phase Dependencies
```
Phase 1: Core → ModelHandler → Cpu
Phase 2: Tokenizers → Diffusion (schedulers + VAE only)
Phase 3: Cuda → Diffusion (full UNet + pipelines)
Phase 3.5: Vulkan backend
Phase 4: SDXL pipeline → Flux pipeline
Phase 5: Audio (Whisper → TTS)
Phase 6: Vision (CLIP → detection)
Phase 7: Server (OpenAI-compatible API)
Phase 8: SwarmUI extension
Phase 9: Video (LTX-Video → Wan → Lance → Cosmos-Predict V2W) + shared interactive infra
Phase 10: Interactive / World Models (Matrix-Game 2/3, Oasis, Hunyuan-GameCraft)
```

## Phase 1 — Foundation (Core + ModelHandler + Cpu)
**Goal:** Tensor types, model loading, basic CPU math.

| Deliverable | Package | Description |
|---|---|---|
| Tensor/TensorView/IBackend | Core | N-D tensor, non-owning views, interface |
| NativeBuffer + MmapHandle | Core | Memory primitives |
| SafeTensors/GGUF loader | ModelHandler | mmap + JSON header, dequantize Q4_0/Q8_0 |
| Model registry | ModelHandler | In-memory + disk cache |
| MatMul/Conv2D/Norm/Activation/ElementWise | Cpu | GEMM, im2col, GroupNorm, LayerNorm, GELU, SiLU |

**Validation:** Unit tests for every kernel. Tensor round-trip.

## Phase 2 — Math Validation (Tokenizers + Schedulers + VAE)
**Goal:** Prove math correct before full UNet.

| Deliverable | Package | Description |
|---|---|---|
| CLIP/T5 tokenizer | Tokenizers | BPE (CLIP), SentencePiece (T5) |
| Euler/DPM++/DDIM scheduler | Diffusion | Core schedulers |
| VAE decoder + tiled decode | Diffusion | Latents → pixels, large image support |

**Validation:** Tokenizer matches OpenAI CLIP Python. Schedulers match diffusers. VAE within 1e-3.

## Phase 3 — First Image (Cuda + SD1.5 Pipeline) — COMPLETE
**Goal:** Generate image with SD1.5/SDXL on CUDA.
**Status:** CUDA backend fully functional with FP16 inference, fused kernels, and async execution. SDXL 1024x1024 at ~5.5s/step.

| Deliverable | Package | Status |
|---|---|---|
| CUDA P/Invoke, stream, cuBLAS | Cuda | Done — `CudaDriverApi`, `CublasApi`, `CudaStream` |
| PTX kernels (FP32): elementwise, spatial, norm, SDPA | Cuda | Done — im2col, GroupNorm, LayerNorm, SiLU, GELU, SDPA |
| PTX kernels (FP16): 11 F16 kernel files | Cuda | Done — elementwise, norms, spatial, transpose, geglu, broadcast_add, cast, fused GroupNorm+SiLU |
| Conv2D via im2col + cuBLAS GEMM | Cuda | Done — no cuDNN, `cublasGemmEx` for F16 |
| GPU weight cache + preload API | Cuda | Done — `PreloadWeights`, `EnumerateWeights` on all models |
| GPU-resident activations (lazy-sync) | Cuda | Done — `CacheActivation`, 85%+ hit rate |
| GPU reshape/permute kernels | Cuda | Done — `transpose_2d`, `permute_0213`, `geglu`, `broadcast_add` |
| Remove per-op Sync + async execution | Cuda | Done — `cuMemFreeAsync`, no per-op sync |
| Kernel fusion (GroupNorm+SiLU) | Cuda | Done — fused PTX kernel, ~40 fusions per UNet step |
| FP16 inference | Cuda | Done — full F16 UNet/VAE, F32 scheduler/CLIP, mixed-dtype casting |

**Performance (SDXL on RTX 3060 12GB):**
- F16 1024x1024: ~5.5s/step (173s total for 20 steps, 5GB VRAM for weights)
- F16 256x256: ~580ms/step (10.7s total for 10 steps)
- F32 1024x1024: ~62s/step (for comparison)
- Speedup: 7-11x from FP16 + fused kernels + async execution
- Target: ~3s/step (ComfyUI) — remaining gap is activation transfer overhead

**Validation:** F16 output visually matches F32 reference. GPU UNet forward: avg_err=5e-7. See `docs/Research/CUDA_PERFORMANCE.md`.

## Phase 3.5 — Vulkan Backend
**Goal:** Port CUDA PTX to SPIR-V; SD1.5 on AMD/Intel/NVIDIA via Vulkan.
**Status:** Research complete (3 docs, 1 phase checklist). Implementation not started — ready to build.

**Research:** [VULKAN_COMPUTE_API.md](../Research/VULKAN_COMPUTE_API.md), [SPIRV_COMPUTE_SHADERS.md](../Research/SPIRV_COMPUTE_SHADERS.md), [VULKAN_MEMORY_MANAGEMENT.md](../Research/VULKAN_MEMORY_MANAGEMENT.md)
**Plan:** [PHASE_3_5_VULKAN_BACKEND.md](../Checklists/PHASE_3_5_VULKAN_BACKEND.md) — sequential build order, every deliverable mapped to a research section.

| Deliverable | Package | Description |
|---|---|---|
| `VulkanApi` (~55 P/Invokes), `VulkanLibraryResolver`, `VulkanInstance`, `VulkanDevice` | Vulkan | Vulkan 1.3 instance/device; FP16 + subgroupSizeControl required; vendor scoring |
| `VulkanMemoryAllocator`, `VulkanBuffer`, `VulkanGpuTransferHelper` | Vulkan | 256 MB / 16 MB slab allocator; weight cache (reference equality); ReBAR fast path; OOM retry |
| `VulkanCommandPool`, `VulkanCommandStream`, `VulkanBarriers` | Vulkan | Single timeline semaphore; sync2 per-buffer barriers; lazy-sync activation cache port |
| `SpirVShaderLoader`, `VulkanPipelineCache`, `VulkanKernels`, `VulkanDescriptorManager` | Vulkan | `.spv` from disk; persistent pipeline cache; push-descriptor preferred; pool ring fallback |
| ~16 GLSL kernels: matmul (the cuBLAS replacement), conv2d, groupnorm/silu, layernorm, sdpa, softmax, transpose, geglu, broadcast_add, upsample, casts, elementwise | native/vulkan/shaders | Tiled GEMM is the centerpiece (≥ 60% of cuBLAS HGEMM target) |
| `VulkanBackend : IBackend` | Vulkan | Op dispatch via `Dispatch(kernel, descriptors, pushConstants, group*)` — mirror of `CudaBackend` |
| SD1.5 / SDXL pipelines run unchanged | Diffusion | `IBackend` abstraction validated by zero source changes in Diffusion package |

**Validation gates:**
1. Every op matches CPU reference within 1e-3 (FP16) / 1e-5 (FP32).
2. Every op matches CUDA reference within 1e-3 on the same NVIDIA hardware.
3. SD1.5 512×512 same seed → SSIM > 0.99 vs CUDA.
4. SD1.5 runs end-to-end on AMD RDNA2/3 (Mesa RADV) producing visually correct output.
5. RTX 3060 Vulkan SD1.5 ≤ 8 s wall-clock (≤ 1.6× CUDA).
6. No memory leaks (validation-layer clean, 100-step loop returns to baseline VRAM).

**Risk:** No cuBLAS equivalent — hand-written tiled HGEMM is the largest single piece of work; ~16 .spv variants to ship. Subgroup size varies per vendor (32 NV, 32/64 AMD, 8–32 Intel) — pin via `requiredSubgroupSize`. See [SPIRV_COMPUTE_SHADERS.md § Performance Targets & Pitfalls](../Research/SPIRV_COMPUTE_SHADERS.md#performance-targets--pitfalls).

## Phase 4 — Model Breadth (SDXL + Flux + FP8 + Extended Models)

| Deliverable | Package | Description |
|---|---|---|
| CLIP-G, T5 encoder | Diffusion | SDXL dual CLIP, Flux T5 |
| SDXL UNet + refiner pipeline | Diffusion | Larger UNet, dual conditioning |
| DiT/Flux pipeline + flow-match scheduler | Diffusion | Double/single-stream MMDiT |
| LoRA loader | Diffusion | SD + Flux formats |
| FP8 (E4M3) dtype + loading | Core + ModelHandler | DType.F8E4M3, safetensors FP8 support |
| FP8 CUDA kernels | Cuda | Cast F8↔F16, cuBLAS FP8 GEMM (Ada+), Ampere fallback |
| FP8 mixed-precision pipeline | Diffusion | FP8 backbone + FP16 VAE/CLIP, quality presets |
| DiTUtils shared helpers | Diffusion | LayerNormNoAffine, SinusoidalTimestepEmbedding, linear projections |
| Chroma config + pipeline | Diffusion | Flux fork with standard CFG |
| AuraFlow config + transformer + pipeline | Diffusion | MMDiT with T5-XXL, SDXL VAE |
| Hunyuan Image 2.1 config + transformer + pipeline | Diffusion | 17B MMDiT, 32×32 VAE |
| Flux.2 config + pipeline | Diffusion | Next-gen 32B DiT, 16×16 VAE, Mistral/Qwen text enc |
| Flux.1 Tools + Kontext configs | Diffusion | Fill/Redux/Canny/Depth/Kontext conditioning variants |
| Qwen-Image config + transformer | Diffusion | 7B–20B MMDiT with unified editing |
| SDXL Inpaint pipeline | Diffusion | 9-channel UNet inpainting |
| ControlNet adapter | Diffusion | Parallel encoder + zero convs (SD1.5/SDXL/Flux) |
| IP-Adapter | Diffusion | Image prompt conditioning with per-layer K/V projections |
| LCM scheduler | Diffusion | 1–4 step distilled scheduler |
| VaeConfig presets | Diffusion | Flux2, Chroma, AuraFlow, HunyuanImage, QwenImage |

## Phase 5 — Audio

| Deliverable | Package | Description |
|---|---|---|
| FFT/STFT, mel spectrogram | Cpu + Cuda | Cooley-Tukey radix-2 |
| Whisper encoder/decoder/pipeline | Audio | Conv1D + transformer, autoregressive decode |
| Whisper streaming | Audio | Real-time chunked STT |
| Kokoro pipeline + HiFiGAN | Audio | Phoneme encoder, mel → waveform |

**Validation:** Whisper WER < 1% vs whisper.cpp. Kokoro mel matches reference.

## Phase 6 — Vision

| Deliverable | Package | Description |
|---|---|---|
| CLIP image encoder + scorer | Vision | ViT + cosine similarity |
| Image/text embeddings | Vision | Standalone pipelines |
| YOLO pipeline + NMS | Vision | Object detection end-to-end |

## Phase 7 — Server

| Deliverable | Package | Description |
|---|---|---|
| Image/audio endpoints | Server | `/v1/images/*`, `/v1/audio/*` |
| SSE streaming | Server | Step-by-step progress |
| Model management + queue + auth | Server | `/v1/models`, FIFO queue, optional API key |

## Phase 8 — SwarmUI Extension
**Goal:** Register as in-process SwarmUI backend.

## Phase 9 — Video + Interactive Infra (Future)

| Deliverable | Package | Description |
|---|---|---|
| Temporal attention, video VAE | Video | Cross-frame consistency |
| `IBackend.Conv3D` + `CausalConv3d` streaming wrapper | Cpu / Cuda / Vulkan | 3D conv kernel family; per-conv frame cache for chunked decode (shared with Lance, Wan, LTX, Matrix-Game) |
| `IBackend.PackedAttention` (variable-length) | Cpu / Cuda / Vulkan | Packed attention with cu_seqlens (shared with Lance + AR video models) |
| LTX-Video / Wan / Lance video / Cosmos-Predict V2W pipelines | Video | First video models. Cosmos-Predict V2W's discrete video tokenizer + AR transformer is the reusable infra for Phase 10 world models. |
| `IActionEncoder` + action embedding plumbing | Diffusion (shared) | Generic action-conditioning abstraction landing in Phase 9 for reuse by Phase 10 world models |
| `DenoiseKvCache` utility | Diffusion (shared) | First-pass KV-cache for the (text + clean cond) prefix across denoise steps; used by Lance video and reusable in Phase 10 |
| Distilled few-step schedulers (DMD, CM) | Diffusion (shared) | 3-8 step samplers required by Matrix-Game 2/3 and GameCraft distilled — land in Phase 9 to keep schedulers in one place |
| Discrete video tokenizer (Cosmos DV / VQ-GAN) | Video (shared) | Cosmos DV first; VQ-GAN follows for Oasis. Shared `IDiscreteVideoTokenizer` interface |
| Streaming VAE decode helper | Video (shared) | Per-frame / per-chunk VAE decode on a separate compute stream — enables 25-40 FPS interactive output in Phase 10 |

## Phase 10 — Interactive / World Models (Future)

| Deliverable | Package | Description |
|---|---|---|
| `SharpInference.Interactive` (new package) | Interactive | New package for action-conditioned, real-time, frame-by-frame world models. Depends on Video + Diffusion + ModelHandler. |
| `IInteractiveSession` streaming loop | Interactive | Real-time event pump: (read action → encode → step → decode → present) at 25-40 FPS |
| Action vocabs: keyboard, mouse, gamepad, camera-pose | Interactive | Per-model `IActionEncoder` implementations; reuse the Phase 9 abstraction |
| Matrix-Game 2.0 pipeline (Skywork, MIT, 1.8B) | Interactive | First interactive world model. 540p @ 25 FPS, SkyReels-V2/Wan lineage. Apache/MIT-style permissive. |
| Matrix-Game 3.0 pipeline (Skywork, Apache-2.0, 5B + MoE 28B) | Interactive | Flagship. 720p @ 40 FPS, memory-augmented DiT finetuned from Wan2.2-TI2V-5B (shares VAE with Lance video path). |
| Oasis-500m pipeline (Decart+Etched, MIT, ~500M) | Interactive | Tiny Minecraft world model. Pedagogical / CI smoke-test target. Likely uses a discrete video tokenizer (VQ family). |
| Hunyuan-GameCraft pipeline (Tencent, **restricted license**) | Interactive | Optional / gated on license acceptance — Tencent Hunyuan Community License excludes EU/UK/SK + 100M MAU cap. SharpInference does NOT bundle weights; user opts in. |
| Memory-augmented DiT cross-attention | Interactive | Matrix-Game 3.0 specific (extra cross-attn stream over stored past-frame latents); designed for reuse if future models add similar memory paths |
| History-mask channel | Interactive | Binary mask channel (1=history, 0=predict) injected into latent input — GameCraft style |
| Deferred-foundation backlog | Interactive (docs) | Explicit list of foundational pieces (AR KV-cache over interleaved video/action tokens, long-context spacetime RoPE, license-acceptance plumbing) deferred until a model that needs them is selected |
