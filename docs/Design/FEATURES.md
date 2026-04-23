# Features List

> Back to [Core Design](CORE_DESIGN.md)

## 1. Core Infrastructure

| Feature | Rationale |
|---|---|
| N-D Tensor (up to 6D) | Diffusion NCHW (4D), video NCTHW (5D), audio NCL (3D) |
| Memory-mapped model loading | Multi-GB models in milliseconds via OS demand-paging |
| Unmanaged tensor storage | Zero GC pressure on inference hot path |
| Dual tensor types (dotLLM) | `Tensor` (sealed, `IDisposable`) + `TensorRef` (readonly record struct) |
| `IBackend` abstraction | Swap CPU/CUDA/Vulkan without changing model code |
| SIMD CPU kernels (dotLLM) | AVX-512 → AVX2 → NEON → scalar. Mandatory scalar fallback |
| CUDA backend (dotLLM) | PTX + cuBLAS HGEMM, pure C# P/Invoke |
| Vulkan backend | SPIR-V compute shaders, cross-vendor GPU support |
| Adaptive thread pool (dotLLM) | SpinWait (latency) / EventBased (throughput), auto-switch |
| Safetensors/GGUF loaders | Primary formats; Q4_0/Q8_0 for consumer VRAM |
| Quantization (FP16/BF16/Q8_0) | Flux FP16 ~24GB → Q8_0 ~12GB |
| Streaming progress | Step-by-step previews for UI integration |
| Cancellation | Mid-run abort for long jobs |
| VRAM budget management | Auto-tile, offload, or reject OOM requests |
| Model caching/registry | `~/.sharpinference/models/`, CLI `sharpinference model pull` |
| Concurrent multi-model serving | Simultaneous requests without full reload |

## 2. Image Generation

| Feature | Rationale |
|---|---|
| Text-to-image, img2img, inpainting, outpainting | Core diffusion workflows |
| CFG (classifier-free guidance) | Positive/negative prompt control |
| ControlNet | Depth/edge/pose-guided generation |
| LoRA loading | Runtime style/character adapters |
| Tiled generation + VAE tiled decode | Large images without OOM |
| Multiple schedulers | Quality/speed tradeoffs |
| Batch generation | Throughput |
| SDXL refiner | Two-stage quality pipeline |
| Flux LoRA/adapters | Distinct LoRA format |
| Latent preview decoding | Streaming UI previews |

## 3. Audio

| Feature | Rationale |
|---|---|
| Whisper STT | OSS STT standard |
| Streaming STT | Real-time chunked transcription |
| Kokoro TTS | Fast, high-quality, Apache 2.0 |
| TTS voice selection | Multiple speakers |
| TTS streaming | Audio as synthesized |
| Voice conversion | Timbre change, avatar systems |
| Audio codec tokenization | ACE-Step, MusicGen support |

## 4. Vision

| Feature | Rationale |
|---|---|
| CLIP image encoding | Text-image similarity, embedding search |
| Image/text embeddings | Semantic search, clustering, RAG |
| YOLO detection | Subject detection, content moderation |
| Image segmentation | Subject/background separation |
| Face detection/landmarks | Portrait conditioning |

## 5. Server / API

| Feature | Rationale |
|---|---|
| OpenAI-compatible image/audio API | `/v1/images/*`, `/v1/audio/*` — drop-in replacement |
| SSE streaming | Step previews, partial audio over HTTP |
| Model management | `/v1/models`, download, list, inspect |
| Health/ready probes | Docker/Kubernetes support |
| Rate limiting | Prevent GPU monopolization |
| Request queue | FIFO with configurable depth |
| Auth (optional) | API key for public deployments |
