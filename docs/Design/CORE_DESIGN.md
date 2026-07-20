# HartsyInference — Core Design Overview

HartsyInference is a **native C#/.NET AI inference engine** (targets **.NET 8 and .NET 10**) spanning image generation, speech-to-text, text-to-speech, vision, object detection, video, **interactive world models** (action-conditioned, real-time, frame-by-frame video generation for games / sims / agents), and **native LLM text generation** (the `HartsyInference.LLM` package). It runs with zero Python dependencies, zero C++ wrappers, and no external processes.

The **recommended way to use the engine** is the **[SwarmUI HartsyInference backend extension](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend)**, which registers HartsyInference as an alternative to SwarmUI's ComfyUI backend and drives the engine's pipelines. Secondary supported paths are the per-modality **NuGet libraries** and the bundled **sample CLIs** under `samples/` plus `src/HartsyInference.Cli` (developer and verification tools).

## Why HartsyInference?

Every existing .NET AI inference solution is either a Python wrapper, C++ wrapper, ONNX Runtime, or NVIDIA-only. No other pure C# engine loads `.safetensors` diffusion models, runs on any GPU (NVIDIA via CUDA, AMD/Intel via Vulkan), and covers image, audio, video, vision, 3D, world models, and LLM text generation in one process.

Pure C# with PTX can reach near-native CUDA performance; HartsyInference applies that approach across every modality and adds cross-vendor GPU via Vulkan. LLM text generation is now first-party (see [LLM_LANGUAGE_PACKAGE.md](LLM_LANGUAGE_PACKAGE.md)); the engine no longer depends on any external LLM runtime.

## Goals & Non-Goals

**Goals**

1. **Pure managed .NET.** No Python, no native inference libraries, no external processes — GPU access is PTX/SPIR-V via P/Invoke only.
2. **Broad, correct model coverage.** Match a Python/C++ reference within documented tolerances, verified against real weights (not just "finite floats") — tracked in [`../Checklists/PARITY_VERIFICATION.md`](../Checklists/PARITY_VERIFICATION.md).
3. **The best pure-C# performance we can reach.** We are transparent that we are not yet as fast as the fastest native runners (see [the benchmarks](../../benchmarks/README.md)); closing that gap (flash-attention, CUDA graphs, F16 activation paths) is an ongoing, in-the-open effort.
4. **First-class SwarmUI integration.** New model support is not "done" until it runs end-to-end through the SwarmUI extension.
5. **Modular packaging.** Pull in only the modality and backend you need.
6. **Zero-GC hot paths.** Unmanaged aligned tensor storage, memory-mapped weights, `Span<T>` throughout.

**Non-goals**

- **A first-party UI / web app.** SwarmUI is the front-end; we build the backend for it.
- **An OpenAI-compatible REST server as a *product*.** SwarmUI is the recommended surface. `HartsyInference.API` does exist and works (OpenAI-shaped `/v1/chat/completions` with continuous batching + paged KV cache, `/v1/images/generations`, model management), but it ships as a runnable sample (`IsPackable=false`), not a supported/published product.
- **A dependency on dotLLM.** LLM text generation is native in `HartsyInference.LLM`; [`../Research/DOTLLM_ARCHITECTURE.md`](../Research/DOTLLM_ARCHITECTURE.md) is retained only as a historical study that informed the native design.
- **Training / fine-tuning.** Inference engine only.

**Audience:** SwarmUI users wanting a no-Python backend; .NET developers embedding inference without a Python sidecar; contributors porting new architectures (see [`BUILD_ORDER.md`](BUILD_ORDER.md) and the agent files under [`../Agents/`](../Agents/)).

## Design Pillars

| Pillar | Description |
|---|---|
| **Pure C#** | CUDA via PTX + Driver API P/Invoke; Vulkan via SPIR-V + Vulkan API P/Invoke |
| **Zero-allocation hot paths** | `NativeMemory.AlignedAlloc`; mmap weights; `TensorRef` on kernels; `Span<T>` on hot paths |
| **Modular NuGet** | Pull only what you need (see `NUGET_PACKAGE_DESIGN.md`) |
| **Multi-GPU backend** | CUDA (NVIDIA) — implemented, FP16 SD1.5/SDXL/Flux<br>Vulkan (AMD/Intel/NVIDIA) — implemented; Flux Schnell FP8 verified end-to-end on Linux + NVIDIA. SD1.5/SDXL integration tests, AMD cross-vendor verification, and perf tuning are the remaining acceptance gates — see [Phase 3.5 checklist](../Checklists/PHASE_3_5_VULKAN_BACKEND.md)<br>CPU fallback — implemented (AVX-512/AVX2/NEON via SIMD dispatch) |
| **SwarmUI-first** | Consumed primarily through the SwarmUI backend extension; also usable as NuGet libraries and sample CLIs |
| **Native LLM** | `HartsyInference.LLM`: config-driven generic decoder transformer, GGUF quantized inference, device-resident KV cache |

## Architecture

```
+--------------------------------------------------------------+
|   Consumers: SwarmUI backend extension (recommended) ·        |
|              NuGet libraries · sample CLIs                    |
+------+----------+----------+--------+----------+-------------+
| Diff | Audio    | Vision   | Video  | Inter-   | LLM         |
| SD/  | Whisper  | CLIP     | LTX /  | active   | Qwen/Llama  |
| Flux | Kokoro   | YOLO     | Wan /  | Matrix-  | Mistral     |
| SD3  | F5/Bark  | SAM      | Lance  | Game /   | (GGUF)      |
|      |          |          | Cosmos | Oasis    |             |
+------+----------+----------+--------+----------+-------------+
|                  HartsyInference.Core                          |
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
- **IBackend op-dispatch** — deliberate divergence from dotLLM. dotLLM uses `IBackend` for memory management only; HartsyInference uses it for op-dispatch because 3 backends × many model types would be unmaintainable with direct calls. Virtual dispatch (~2ns) is negligible vs kernel runtime (ms).
- **GPU weight cache** — weights preloaded to GPU via `PreloadWeights()`, cached by `Tensor` object reference. CPU copies can be disposed after preload. Cache-aware `CopyToDevice` returns GPU pointer without H2D transfer on cache hit. See `docs/Research/CUDA_PERFORMANCE.md`.
- **Auto-transfer pattern** — current CUDA backend auto-transfers activation tensors H2D/D2H per op. Correct but slow (~33x vs ComfyUI). GPU-resident activations planned as primary optimization. See CUDA_PERFORMANCE.md for roadmap.
- **Pipeline factory** — model metadata drives automatic pipeline selection.
- **Three-tier options** — flat properties (simple), explicit composition (advanced), custom injection (full control).

## Native LLM Text Generation

LLM text generation ships in the first-party **`HartsyInference.LLM`** package: one config-driven generic decoder transformer (Qwen2/Qwen3/Llama/Mistral), GGUF quantized inference, a device-resident KV cache, a composable sampler chain, and chat templates. See [LLM_LANGUAGE_PACKAGE.md](LLM_LANGUAGE_PACKAGE.md) for the design.

The engine used to plan on the external **dotLLM** project for LLMs. That is no longer the case: dotLLM is GPLv3 (linking it would relicense the engine and the SwarmUI extension), so the LLM package is a clean-room native implementation. `docs/Research/DOTLLM_ARCHITECTURE.md` remains only as a historical study of the patterns that informed the native design; it is not a live dependency. Architectural patterns are not copyrightable.

## Image/Audio vs LLM Inference — Different Compute Shapes

The image and audio stack differs from decoder LLM inference in almost every op; both live in one engine but reuse little at the kernel level:

| LLM decoder | Image/Audio |
|---|---|
| RMSNorm | GroupNorm (32-group) |
| 1D RoPE | 2D RoPE (spatial) |
| Causal attention | Bidirectional + cross-attention |
| SwiGLU | SiLU (UNet ResNet blocks) |
| Embedding lookup | Conv2D (3x3, 1x1, depthwise) |
| KV-cache decode | Full forward pass per step |
| Single-token GEMV | Batch GEMM (spatial maps) |
| — | Upsample 2D, FFT/STFT/Mel, timestep conditioning |

## SwarmUI Backend (Recommended Path)

The primary way to run the engine is the [SwarmUI HartsyInference backend extension](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend). It installs as a SwarmUI backend and registers HartsyInference as an alternative to the ComfyUI backend:

| ComfyUI backend (Python) | HartsyInference backend |
|---|---|
| Python + pip + torch + diffusers | NuGet reference only |
| Separate process, HTTP round-trips | In-process inference |
| 10-30s startup | Milliseconds |
| No C# debuggability | Full C# debuggability |

## Capabilities by Modality

A capability-level view; for per-model **status** (verified end-to-end vs built-but-pending) see the
modality status docs indexed in [`../Checklists/MODEL_STATUS.md`](../Checklists/MODEL_STATUS.md), with
[`../Checklists/PARITY_VERIFICATION.md`](../Checklists/PARITY_VERIFICATION.md) the real-weight parity
authority.

- **Core engine** — three backends behind one `IBackend` (CUDA PTX+cuBLAS, Vulkan SPIR-V, CPU AVX2/512/NEON); eager execution; direct `.safetensors`/`.gguf`/`.pt`/`.ckpt` loading incl. sharded + diffusers layouts with architecture auto-detection; GGUF + block-scaled (MXFP4/8, NVFP4) quantization with fused dequant/GEMV; HuggingFace auto-download; LoRA.
- **LLM text generation (`HartsyInference.LLM`)** — native config-driven decoder transformer (Qwen2/Qwen3, Llama-3.x, Mistral, …) + GGUF; device-resident KV cache, sampler chain, chat templates; also powers diffusion/audio text encoders; fused Q4_K/Q6_K/Q8_0 decode + quantized `lm_head` + split-K flash-decode.
- **Image (`HartsyInference.Diffusion`)** — UNet (SD1.5, SDXL+Refiner, inpaint) and DiT/MMDiT/NextDiT (Flux.1/.2, Chroma/Radiance, SD3, Qwen-Image, HunyuanImage, HiDream, AuraFlow, Lumina 2, ERNIE-Image, Kandinsky 5, OmniGen 2, Ideogram 4, …); t2i/i2i/inpaint + tiled VAE; text encoders CLIP/T5/UMT5/Pile-T5/Gemma-2/Qwen2.5-VL/Qwen3; full sampler set; prompt weighting/BREAK/scheduling/regional/textual-inversion/clip-skip; ControlNet + IP-Adapter loaders.
- **Audio (`HartsyInference.Audio`)** — STT (Whisper tiny→large-v3, Moonshine); TTS (Kokoro, StyleTTS2, Bark, Spark-TTS, CosyVoice, VibeVoice, Piper/VITS, MeloTTS, F5-TTS cloning, …); music (ACE-Step, MusicGen, YuE); codecs (Vocos, EnCodec, DAC, SNAC, Mimi, WavTokenizer, BiCodec, XCodec, Oobleck); pure-C# DSP (STFT/mel/FFT, HiFi-GAN vocoders, streaming); G2P via `HartsyInference.Audio.Phonemizer` (pure-C# espeak-ng port).
- **Vision (`HartsyInference.Vision`)** — embeddings (CLIP ViT-L/H/bigG, SigLIP/2, DINOv2/3); detection (YOLO8/11); segmentation (SAM/2/2.1, CLIPSeg); RetinaFace; PNG codec helpers.
- **Video (`HartsyInference.Video`)** — t2v/i2v (LTX-Video, Wan 2.x T2V+I2V, Lance, Kandinsky 5 Video); shared CausalConv3d + Wan-family VAE with streaming + N-axis RoPE; ffmpeg muxing via SwarmUI.
- **3D (`HartsyInference.ThreeD`)** — image/text→mesh (TripoSR, Hunyuan3D-2); marching cubes + glTF/OBJ/PLY export.
- **Interactive / world (`HartsyInference.World`)** — real-time action-conditioned generation (Hunyuan-GameCraft, Matrix-Game 2.0/3.0, Oasis); `IInteractiveSession` with background compute, action/camera/FOV memory.

## Design Documents Index

| Document | Description |
|---|---|
| [Build Order](BUILD_ORDER.md) | Phase dependencies and sequencing |
| [File Structure](FILE_STRUCTURE.md) | Project layout |
| [NuGet Package Design](NUGET_PACKAGE_DESIGN.md) | Package breakdown, dependency graph |
| [Implementation Details](IMPLEMENTATION_DETAILS.md) | Per-component technical approach |
| [LLM Language Package](LLM_LANGUAGE_PACKAGE.md) | Native LLM text generation design |
| [Validation Strategy](VALIDATION_STRATEGY.md) | References and validation methods |
| [Model Support Roadmap](MODEL_SUPPORT_ROADMAP.md) | Model support plan |
