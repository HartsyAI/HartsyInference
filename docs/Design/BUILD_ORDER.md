# Build Order & Phases

> Back to [Core Design](CORE_DESIGN.md)

---

## Phase Dependencies

Implementation must follow this order to avoid blocked work:

```
Phase 1: Core → ModelHandler → Cpu
             (all other packages depend on these three)
                        │
Phase 2: Tokenizers → Diffusion (schedulers + VAE only)
             (validate math before building full UNet)
                        │
Phase 3: Cuda → Diffusion (full UNet + pipelines)
             (get SD1.5 working end-to-end on CUDA, following dotLLM patterns exactly)
                        │
Phase 3.5: Vulkan backend
             (port CUDA kernels to SPIR-V compute shaders, same IBackend interface)
             (AMD/Intel GPU support — extends dotLLM's P/Invoke approach to Vulkan)
                        │
Phase 4: SDXL pipeline → Flux pipeline
             (build on proven SD1.5 foundation, validated on both CUDA and Vulkan)
                        │
Phase 5: Audio (Whisper first, then TTS)
                        │
Phase 6: Vision (CLIP first, then detection)
                        │
Phase 7: Server (OpenAI-compatible image + audio API)
                        │
Phase 8: SwarmUI extension
                        │
Phase 9: Video (stub → LTX-Video)
```

---

## Phase 1 — Foundation (Core + ModelHandler + Cpu)

**Goal:** Tensor types work, models can be loaded from disk, basic CPU math is operational.

| Deliverable | Package | Description |
|---|---|---|
| Tensor type | Core | N-D tensor with unmanaged memory, shape, strides, slicing |
| TensorView | Core | Zero-alloc non-owning views |
| IBackend interface | Core | Full interface definition |
| NativeBuffer + MmapHandle | Core | Memory management primitives |
| SafeTensors loader | ModelHandler | Load safetensors files via mmap |
| GGUF loader | ModelHandler | Load GGUF files, dequantize Q4_0/Q8_0 |
| Model registry | ModelHandler | In-memory cache + disk cache |
| MatMul kernel | Cpu | GEMM, GEMV with AVX2/AVX-512 |
| Conv2D kernel | Cpu | im2col + GEMM |
| Norm kernels | Cpu | GroupNorm, LayerNorm, RMSNorm |
| Activation kernels | Cpu | GELU, SiLU |
| ElementWise kernels | Cpu | Add, Mul, Scale, Concat |

**Validation:** Unit tests for every kernel against known-good values. Tensor round-trip tests.

---

## Phase 2 — Math Validation (Tokenizers + Schedulers + VAE)

**Goal:** Prove the math is correct before tackling the full UNet.

| Deliverable | Package | Description |
|---|---|---|
| CLIP tokenizer | Tokenizers | BPE matching OpenAI CLIP exactly |
| T5 tokenizer | Tokenizers | SentencePiece for SD3/Flux |
| Euler scheduler | Diffusion | First scheduler implementation |
| DPM++ 2M scheduler | Diffusion | Most popular scheduler |
| DDIM scheduler | Diffusion | Classic deterministic scheduler |
| VAE decoder | Diffusion | Decode latents → pixels |
| VAE tiled decoder | Diffusion | Large image decode without OOM |

**Validation:** Tokenizer output matches OpenAI CLIP Python. Scheduler step sequences match diffusers. VAE output matches diffusers within 1e-3.

---

## Phase 3 — First Image (Cuda + SD1.5 Pipeline)

**Goal:** Generate an actual image from text with SD1.5 on a CUDA GPU, following dotLLM's CUDA patterns exactly.

| Deliverable | Package | Description |
|---|---|---|
| CUDA Driver P/Invoke | Cuda | `CudaDriverApi.cs` with ~34 `[LibraryImport("nvcuda")]` declarations (mirrors dotLLM) |
| PTX kernel loader | Cuda | `PtxKernelLoader.cs` — embed + JIT compile PTX at runtime (mirrors dotLLM `CudaModule.cs`) |
| cuBLAS HGEMM | Cuda | `CuBlasWrapper.cs` — FP16 GEMM with Tensor Core auto-usage (mirrors dotLLM `CublasApi.cs`) |
| CUDA Conv2D (cuDNN) | Cuda | Correct Conv2D via cuDNN P/Invoke — fallback before custom PTX |
| CUDA GroupNorm/LayerNorm | Cuda | PTX kernels with shared memory reduction |
| CUDA GroupNorm+SiLU fused | Cuda | Fused PTX kernel — key bandwidth optimization for UNet (follows dotLLM fusion philosophy) |
| CUDA SDPA | Cuda | PTX Flash Attention kernel for spatial + cross-attention |
| CUDA dequant (Q8_0, Q4_K) | Cuda | On-the-fly dequant PTX kernels (same pattern as dotLLM's dequant kernels) |
| Function handle cache | Cuda | `Dictionary<string, nint>` for process-lifetime caching (from dotLLM) |
| `CudaLibraryResolver` | Cuda | Cross-platform nvcuda.dll / libcuda.so.1 resolution (from dotLLM) |
| CLIP text encoder | Diffusion | Full transformer forward pass |
| UNet (SD1.5) | Diffusion | All blocks: ResNet, CrossAttention, Up/Down |
| SD1.5 pipeline | Diffusion | End-to-end text → latents → image |

**Validation:** Same seed + prompt → visually identical output to Python diffusers. SSIM > 0.95.

---

## Phase 3.5 — Vulkan Backend (AMD/Intel GPU Support)

**Goal:** Port all CUDA PTX kernels to Vulkan SPIR-V compute shaders. SD1.5 works on AMD and Intel GPUs.

| Deliverable | Package | Description |
|---|---|---|
| Vulkan API P/Invoke | Vulkan | `VulkanApi.cs` with ~40 `[LibraryImport("vulkan-1")]` declarations (extends dotLLM's CUDA approach) |
| `VulkanLibraryResolver` | Vulkan | Cross-platform vulkan-1.dll / libvulkan.so.1 resolution (mirrors `CudaLibraryResolver`) |
| SPIR-V shader loader | Vulkan | `SpirVShaderLoader.cs` — load `.spv`, create compute pipelines, cache in `Dictionary<string, nint>` (mirrors `PtxKernelLoader`) |
| Vulkan memory allocator | Vulkan | Sub-allocate from large `vkAllocateMemory` blocks, device-local for tensors, host-visible for staging |
| Vulkan descriptor manager | Vulkan | Descriptor set layouts + pools for tensor buffer bindings, push constants for scalar params |
| SPIR-V Conv2D | Vulkan | 3×3 and 1×1 convolution compute shaders with shared memory tiling |
| SPIR-V GroupNorm/LayerNorm | Vulkan | Subgroup reduction-based normalization kernels |
| SPIR-V GroupNorm+SiLU fused | Vulkan | Fused kernel matching CUDA equivalent |
| SPIR-V SDPA | Vulkan | Tiled attention with shared memory, subgroup shuffles |
| SPIR-V matmul tiled | Vulkan | Tiled GEMM via subgroup operations (no cuBLAS equivalent — must be hand-written) |
| SPIR-V dequant | Vulkan | Q8_0 and Q4_K dequantization compute shaders |
| SD1.5 on Vulkan | Diffusion | End-to-end pipeline running on AMD/Intel via `IBackend` abstraction |

**Validation:** SD1.5 Vulkan output matches CUDA output within FP16 tolerance (1e-3). Same seed → same image on both backends.

**Key risk:** No cuBLAS equivalent for Vulkan — matrix multiply must be implemented as a tiled SPIR-V compute shader using subgroup operations. Performance will be lower than cuBLAS + Tensor Cores but still dramatically faster than CPU.

---

## Phase 4 — Model Breadth (SDXL + Flux)

**Goal:** Support the two most popular model families beyond SD1.5.

| Deliverable | Package | Description |
|---|---|---|
| CLIP-G text encoder | Diffusion | Second CLIP encoder for SDXL |
| SDXL UNet | Diffusion | Larger UNet with dual conditioning |
| SDXL pipeline | Diffusion | Including refiner pipeline |
| T5 text encoder | Diffusion | For Flux (and SD3 later) |
| DiT (Flux) | Diffusion | Double-stream + single-stream blocks |
| Flow-match Euler scheduler | Diffusion | Flux scheduler |
| Flux pipeline | Diffusion | End-to-end Flux inference |
| LoRA loader | Diffusion | SD + Flux LoRA formats |

**Validation:** SDXL and Flux output matches reference implementations.

---

## Phase 5 — Audio

**Goal:** Whisper STT and Kokoro TTS working.

| Deliverable | Package | Description |
|---|---|---|
| FFT / STFT | Cpu + Cuda | Cooley-Tukey radix-2 |
| Mel spectrogram | Audio | Whisper-compatible preprocessing |
| Whisper encoder | Audio | Conv1D + transformer encoder |
| Whisper decoder | Audio | Autoregressive with cross-attention |
| Whisper pipeline | Audio | Full transcription pipeline |
| Whisper streaming | Audio | Chunk-by-chunk real-time STT |
| Kokoro pipeline | Audio | TTS with phoneme encoder |
| HiFiGAN vocoder | Audio | Mel → waveform synthesis |

**Validation:** Whisper output matches whisper.cpp (WER < 1%). Kokoro mel matches reference.

---

## Phase 6 — Vision

**Goal:** CLIP embeddings and object detection.

| Deliverable | Package | Description |
|---|---|---|
| CLIP image encoder | Vision | ViT patch embedding + transformer |
| CLIP scorer | Vision | Text-image cosine similarity |
| Image embeddings | Vision | Standalone embedding pipeline |
| Text embeddings | Vision | Nomic-Embed / E5 / BGE |
| YOLO pipeline | Vision | Object detection end-to-end |
| YOLO NMS | Vision | Non-maximum suppression post-processing |

---

## Phase 7 — Server

**Goal:** OpenAI-compatible REST API for all modalities.

| Deliverable | Package | Description |
|---|---|---|
| Image generation endpoints | Server | `/v1/images/generations`, `/v1/images/edits` |
| Audio endpoints | Server | `/v1/audio/transcriptions`, `/v1/audio/speech` |
| SSE streaming | Server | Step-by-step progress for image generation |
| Model management | Server | `/v1/models` CRUD |
| Request queue | Server | FIFO with concurrency control |
| Auth middleware | Server | Optional API key validation |

---

## Phase 8 — SwarmUI Extension

**Goal:** Register SharpInference as an in-process SwarmUI backend.

---

## Phase 9 — Video (Future)

**Goal:** Video generation starting with LTX-Video.

| Deliverable | Package | Description |
|---|---|---|
| Temporal attention | Video | Cross-frame attention for video consistency |
| Video VAE decoder | Video | Decode video latents to frames |
| LTX-Video pipeline | Video | First video generation pipeline |
| Wan pipeline | Video | Second video model |
