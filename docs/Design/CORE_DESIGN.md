# SharpInference — Core Design Overview

SharpInference is a **native C#/.NET 10 AI inference engine** for non-LLM modalities — image generation, speech-to-text, text-to-speech, vision, object detection, and video. It works alongside **dotLLM** ([kkokosa/dotLLM](https://github.com/kkokosa/dotLLM)) as a complete AI platform with zero Python dependencies, zero C++ wrappers, and no external processes.

## Why SharpInference?

Every existing .NET AI inference solution is either a Python wrapper, C++ wrapper, ONNX Runtime, or NVIDIA-only. No pure C# engine loads `.safetensors` diffusion models, runs on any GPU (NVIDIA via CUDA, AMD/Intel via Vulkan), and streams results through ASP.NET — without external processes.

dotLLM proved pure C# with PTX achieves ~98-100% native CUDA performance for LLMs. SharpInference extends this to non-LLM modalities and adds cross-vendor GPU via Vulkan.

## Design Pillars

| Pillar | Description |
|---|---|
| **Pure C#** | CUDA via PTX + Driver API P/Invoke; Vulkan via SPIR-V + Vulkan API P/Invoke |
| **Zero-allocation hot paths** | `NativeMemory.AlignedAlloc`; mmap weights; `TensorRef` on kernels; `Span<T>` on hot paths |
| **Modular NuGet** | Pull only what you need (see `NUGET_PACKAGE_DESIGN.md`) |
| **Multi-GPU backend** | CUDA (NVIDIA) + Vulkan (AMD/Intel/NVIDIA) + CPU fallback |
| **OpenAI-compatible API** | Drop-in replacement for OpenAI image/audio endpoints |
| **dotLLM alignment** | Same tensor memory, P/Invoke, PTX, SIMD dispatch, thread pool patterns |

## Architecture

```
+--------------------------------------------------------------+
|                     SharpInference.Server                     |
|            (OpenAI-compatible REST API + SSE)                 |
+------------+--------------+----------------------------------+
|  Diffusion |    Audio     |           Vision                 |
|  SD/SDXL   |  Whisper STT |  CLIP Embeddings                 |
|  Flux/SD3  |  Kokoro TTS  |  YOLO Detection                  |
+------------+--------------+----------------------------------+
|                  SharpInference.Core                          |
|    Tensor + TensorRef . IBackend . Schedulers . Pipelines    |
+--------------+---------------------+------------------------+
| CPU Backend  |    CUDA Backend      |   Vulkan Backend       |
| AVX2/512/NEON| PTX Kernels + cuBLAS | SPIR-V + VkCompute     |
+--------------+---------------------+------------------------+
|                     Model Handler                             |
|        Safetensors . GGUF . HuggingFace . Registry            |
+--------------------------------------------------------------+
```

Model code programs against `IBackend` only. CPU dispatches to SIMD kernels; CUDA to PTX + cuBLAS; Vulkan to SPIR-V compute shaders. Backend selected at runtime.

## Key Decisions

- **Eager execution** — no computation graph. Each op executes immediately. Fusion is manual at kernel level.
- **Multi-type tensor system** — `Tensor` (owns memory), `TensorView` (non-owning), `TensorRef` (zero-alloc kernel struct). See `AGENTS.md` for details.
- **IBackend op-dispatch** — deliberate divergence from dotLLM. dotLLM uses `IBackend` for memory management only; SharpInference uses it for op-dispatch because 3 backends × many model types would be unmaintainable with direct calls. Virtual dispatch (~2ns) is negligible vs kernel runtime (ms).
- **GPU weight cache** — weights preloaded to GPU via `PreloadWeights()`, cached by `Tensor` object reference. CPU copies can be disposed after preload. Cache-aware `CopyToDevice` returns GPU pointer without H2D transfer on cache hit. See `docs/Research/CUDA_PERFORMANCE.md`.
- **Auto-transfer pattern** — current CUDA backend auto-transfers activation tensors H2D/D2H per op. Correct but slow (~33x vs ComfyUI). GPU-resident activations planned as primary optimization. See CUDA_PERFORMANCE.md for roadmap.
- **Pipeline factory** — model metadata drives automatic pipeline selection.
- **Three-tier options** — flat properties (simple), explicit composition (advanced), custom injection (full control).

## dotLLM Relationship

dotLLM handles LLM text generation; SharpInference covers everything else. Shared patterns are documented in `docs/CODE_STYLE.md` and `docs/Agents/AGENTS.md`. Integration points: shared CUDA context, unified model registry, prompt enhancement, multimodal pipelines, composable server (`/v1/chat/completions` + `/v1/images/*` + `/v1/audio/*`).

**Licensing:** dotLLM is GPLv3. SharpInference uses clean-room implementations. Architectural patterns are not copyrightable.

## Image vs LLM Inference — Why Separate Engines

| LLM (dotLLM) | Image/Audio (SharpInference) |
|---|---|
| RMSNorm | GroupNorm (32-group) |
| 1D RoPE | 2D RoPE (spatial) |
| Causal attention | Bidirectional + cross-attention |
| SwiGLU | SiLU (UNet ResNet blocks) |
| Embedding lookup | Conv2D (3x3, 1x1, depthwise) |
| KV-cache decode | Full forward pass per step |
| Single-token GEMV | Batch GEMM (spatial maps) |
| — | Upsample 2D, FFT/STFT/Mel, timestep conditioning |

## SwarmUI Backend Angle

| Before (Python backend) | After (SharpInference) |
|---|---|
| Python + pip + torch + diffusers | NuGet reference only |
| Separate process, HTTP round-trips | In-process inference |
| 10-30s startup | Milliseconds |
| No C# debuggability | Full C# debuggability |

## Design Documents Index

| Document | Description |
|---|---|
| [Build Order](BUILD_ORDER.md) | Phase dependencies and sequencing |
| [File Structure](FILE_STRUCTURE.md) | Project layout |
| [NuGet Package Design](NUGET_PACKAGE_DESIGN.md) | Package breakdown, dependency graph |
| [Implementation Details](IMPLEMENTATION_DETAILS.md) | Per-component technical approach |
| [Validation Strategy](VALIDATION_STRATEGY.md) | References and validation methods |
| [Research Requirements](RESEARCH_REQUIREMENTS.md) | Research docs needed before implementation |
| [Model Support Roadmap](MODEL_SUPPORT_ROADMAP.md) | Phase 1-3 model support plan |
