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
Phase 9: Video (stub → LTX-Video)
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

## Phase 3 — First Image (Cuda + SD1.5 Pipeline) — CUDA FUNCTIONAL
**Goal:** Generate image with SD1.5/SDXL on CUDA.
**Status:** CUDA backend working. SD1.5 + SDXL generating correct images on GPU. Performance optimization in progress.

| Deliverable | Package | Status |
|---|---|---|
| CUDA P/Invoke, stream, cuBLAS | Cuda | Done — `CudaDriverApi`, `CublasApi`, `CudaStream` |
| PTX kernels (FP32): elementwise, spatial, norm, SDPA | Cuda | Done — im2col, GroupNorm, LayerNorm, SiLU, GELU, SDPA |
| Conv2D via im2col + cuBLAS SGEMM | Cuda | Done — no cuDNN dependency |
| GPU weight cache + preload API | Cuda | Done — `PreloadWeights`, `EnumerateWeights` on all models |
| GPU-resident activations (lazy-sync) | Cuda | Done — `CacheActivation`, 77% hit rate, ~7% speedup. Per-op Sync still limits gains. |
| GPU reshape/permute kernels | Cuda | **Next** — eliminate CPU-side reshape round-trips (~1,673 misses/step) |
| Remove per-op Sync + async execution | Cuda | Planned — deferred cleanup, `cuMemFreeAsync` |
| Kernel fusion (GroupNorm+SiLU, etc.) | Cuda | Planned |
| FP16 inference | Cuda | Planned |

**Current performance:** ~93s/step at 1024x1024 (per-op Sync + CPU reshapes dominate). Target: ~3-5s/step.
**Validation:** GPU UNet forward matches CPU within avg_err=5e-7. See `docs/Research/CUDA_PERFORMANCE.md`.

## Phase 3.5 — Vulkan Backend
**Goal:** Port CUDA PTX to SPIR-V; SD1.5 on AMD/Intel.

| Deliverable | Package | Description |
|---|---|---|
| Vulkan P/Invoke, SPIR-V loader | Vulkan | `VulkanApi`, `SpirVShaderLoader` |
| Vulkan memory allocator, descriptor manager | Vulkan | Sub-allocation, staging buffers |
| Conv2D/GroupNorm/SDPA/matmul/dequant | Vulkan | Tiled compute shaders |
| SD1.5 pipeline on Vulkan | Diffusion | `IBackend` abstraction |

**Validation:** Vulkan output matches CUDA within 1e-3. Same seed → same image.

**Risk:** No cuBLAS equivalent; tiled GEMM via subgroup ops. Slower than CUDA but still faster than CPU.

## Phase 4 — Model Breadth (SDXL + Flux)

| Deliverable | Package | Description |
|---|---|---|
| CLIP-G, T5 encoder | Diffusion | SDXL dual CLIP, Flux T5 |
| SDXL UNet + refiner pipeline | Diffusion | Larger UNet, dual conditioning |
| DiT/Flux pipeline + flow-match scheduler | Diffusion | Double/single-stream MMDiT |
| LoRA loader | Diffusion | SD + Flux formats |

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

## Phase 9 — Video (Future)

| Deliverable | Package | Description |
|---|---|---|
| Temporal attention, video VAE | Video | Cross-frame consistency |
| LTX-Video / Wan pipelines | Video | First video models |
