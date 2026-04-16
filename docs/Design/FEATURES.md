# Features List

> Back to [Core Design](CORE_DESIGN.md)

---

## 1. Core Infrastructure

| Feature | Rationale |
|---|---|
| **N-D Tensor type (up to 6D)** | Diffusion needs NCHW (4D), video needs NCTHW (5D), audio needs NCL (3D). LLM-centric engines only do 2D. |
| **Memory-mapped model loading** | Multi-GB models load in milliseconds via OS demand-paging. No upfront RAM allocation. |
| **Unmanaged memory tensor storage** | Eliminates GC pressure entirely on the inference hot path. Critical for real-time generation. |
| **Dual tensor types (from dotLLM)** | `Tensor` (sealed class, `IDisposable`) for lifecycle. `TensorRef` (readonly record struct) for zero-alloc kernel signatures. Same pattern dotLLM proved for LLM inference. |
| **Backend abstraction (`IBackend`)** | Swap CPU/CUDA/Vulkan backends without changing model code. Same interface pattern used by dotLLM. |
| **SIMD CPU kernels (from dotLLM)** | Tiered dispatch: AVX-512 → AVX2 → NEON → scalar fallback. `System.Runtime.Intrinsics` for hot loops, `TensorPrimitives` for standard ops. Mandatory scalar fallback for every kernel. |
| **CUDA GPU backend (from dotLLM)** | PTX kernels via CUDA Driver API P/Invoke + cuBLAS HGEMM. Same `CudaDriverApi.cs` pattern, same `cuModuleLoadData`/`cuLaunchKernel` approach. NVIDIA GPUs only. |
| **Vulkan GPU backend (extending dotLLM)** | SPIR-V compute shaders via Vulkan API P/Invoke. Same pure-C# philosophy applied to Vulkan for cross-vendor GPU support. AMD, Intel, and NVIDIA. |
| **Adaptive thread pool (from dotLLM)** | `ComputeThreadPool` with SpinWait (latency-critical) and EventBased (throughput) modes. Switches automatically based on operation type. |
| **Safetensors loader** | Virtually all modern diffusion models ship as `.safetensors`. Primary model format. |
| **GGUF diffusion model loader** | sd.cpp quantized models (Q4_0, Q8_0) enable running diffusion on consumer hardware with limited VRAM. |
| **Quantization (FP16, BF16, Q8_0)** | Flux.1-dev in FP16 needs ~24GB VRAM. Q8_0 brings it to ~12GB. Q4 is too lossy for diffusion. |
| **Streaming progress callbacks** | Users expect step-by-step previews during generation. Needed for any real UI integration. |
| **Request cancellation** | Long inference jobs (video, large images) must be cancellable mid-run. |
| **VRAM budget management** | Automatically tile large inputs, offload to CPU RAM, or reject requests that would OOM. |
| **Model caching / registry** | Downloaded models cached to `~/.sharpinference/models/`. CLI tool `sharpinference model pull`. |
| **Concurrent multi-model serving** | Server mode must handle simultaneous requests for different models without full reload. |

---

## 2. Image Generation

| Feature | Rationale |
|---|---|
| **Text-to-image** | Core use case. Prompt → image using diffusion loop. |
| **Image-to-image (img2img)** | Refine or restyle an existing image. Used extensively in art workflows. |
| **Inpainting** | Fill masked regions of an image. Essential for editing workflows. |
| **Outpainting** | Extend image beyond its borders. Popular for wallpaper/landscape generation. |
| **Classifier-free guidance (CFG)** | Positive + negative prompt control. Every real workflow uses this. |
| **ControlNet** | Guided generation from depth maps, edges, poses, etc. Heavy use in professional workflows. |
| **LoRA weight loading** | Runtime-loadable style/character adapters with no weight merging. Critical for customization. |
| **Tiled generation** | Generate images larger than VRAM allows by processing in tiles with overlap blending. |
| **VAE tiled decode** | Decode large latents in tiles to avoid OOM in VAE decoder. |
| **Multiple schedulers** | Different schedulers give different quality/speed tradeoffs. Users expect choice. |
| **Batch generation** | Generate multiple images in one forward pass for throughput. |
| **SDXL refiner pipeline** | SDXL two-stage pipeline (base + refiner). Needed for highest quality SDXL output. |
| **Flux LoRA / adapter support** | Flux has a distinct LoRA format from SD-family. |
| **Preview decoding (latent preview)** | Decode low-res latent previews every N steps for streaming UI. |

---

## 3. Audio

| Feature | Rationale |
|---|---|
| **Speech-to-text (Whisper)** | Most widely used OSS STT. Used in voice assistant workflows. |
| **Streaming STT (chunked audio)** | Real-time voice input needs chunk-by-chunk transcription. |
| **Text-to-speech (Kokoro)** | Fast, high-quality, Apache 2.0 licensed. Primary TTS engine. |
| **TTS voice selection** | Multiple speaker voices. Critical for avatar voice systems. |
| **TTS streaming output** | Stream audio as it is synthesized rather than waiting for full clip. |
| **Voice conversion** | Change the timbre of a voice while preserving content. Used in avatar systems. |
| **Audio codec tokenization** | Needed for models like ACE-Step, MusicGen. Tokenizes waveform to discrete codes. |

---

## 4. Vision

| Feature | Rationale |
|---|---|
| **CLIP image encoding** | Text-image similarity, prompt-image matching, embedding search. Core to gallery features. |
| **Image embeddings** | CLIP / SigLIP embeddings for semantic image search and clustering. |
| **Text embeddings** | Sentence embeddings for RAG, semantic search. |
| **Object detection (YOLO)** | Detect and crop subjects for inpainting, auto-focus, content moderation. |
| **Image segmentation** | Segment subjects from background. Used in avatar compositing. |
| **Face detection / landmark** | Needed for face conditioning in portrait generation. |

---

## 5. Server / API

| Feature | Rationale |
|---|---|
| **OpenAI-compatible image API** | `/v1/images/generations`, `/v1/images/edits` — works with any OpenAI client library. |
| **OpenAI-compatible audio API** | `/v1/audio/transcriptions`, `/v1/audio/speech` — drop-in for existing integrations. |
| **Server-sent events (SSE) streaming** | Stream generation progress (step previews, partial audio) over HTTP. |
| **Model management endpoints** | `/v1/models`, download, list, inspect — same pattern as dotLLM. |
| **Health and readiness probes** | `/health`, `/ready` for Docker/Kubernetes deployments. |
| **Rate limiting** | Prevent a single client from monopolizing the GPU. |
| **Request queue** | FIFO queue with configurable depth. Reject or queue excess requests. |
| **Authentication (optional)** | API key header validation for public deployments. |
