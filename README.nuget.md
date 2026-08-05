# HartsyInference

**A pure C#/.NET AI inference engine for LLM text generation, image generation, speech, music, vision, video, 3D, and interactive world models. No Python, no native runtime DLLs.**

HartsyInference loads `.safetensors` and `.gguf` checkpoints directly and runs them on NVIDIA CUDA, cross-vendor Vulkan, or CPU SIMD, entirely in managed .NET. GPU kernels are PTX/SPIR-V shipped with the package and JIT-compiled at runtime; there are no C++ wrappers, no bundled native inference library, and no external Python process to manage. Just NuGet packages.

It is a complete pure-.NET AI stack, including **native LLM text generation** (Qwen, Llama, Mistral, quantized GGUF) in the `HartsyInference.LLM` package.

> **The easiest way to use HartsyInference is inside [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) via the [HartsyInference backend extension](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend)**, a pure-C# alternative to the ComfyUI backend. These NuGet packages are for embedding the engine directly in your own .NET code.

---

## ⚠️ Alpha software

**This is `2.0.0-alpha`, an early, fast-moving preview.** Use it to experiment, not in production.

- **APIs will change without notice** between alpha releases. Pin an exact version.
- **Model coverage is broad but maturity varies.** Many architectures are implemented and load/run end-to-end but are still being validated numerically against their reference implementations. Treat output quality per-model as "verify before you rely on it."
- **No support guarantees, no semver stability** until `2.0.0`.
- The **sample CLIs are not published as packages** in this alpha; they live in the source repository as developer/validation tools.

Found a bug or a mismatch against a reference? Please [open an issue](https://github.com/HartsyAI/HartsyInference/issues).

---

## Install

One package pulls in the whole stack (all backends + every modality):

```sh
dotnet add package HartsyInference --prerelease
```

Or reference only the pieces you need (see [Packages](#packages)):

```sh
dotnet add package HartsyInference.Audio --prerelease
dotnet add package HartsyInference.Cpu   --prerelease
```

**Requires .NET 8 or .NET 10.**

---

## Quick start: speech-to-text

The Whisper pipeline downloads a checkpoint from HuggingFace on first use and runs on whichever backend you pass:

```csharp
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;          // or HartsyInference.Cuda / HartsyInference.Vulkan

using WhisperPipeline pipeline = await WhisperPipeline.LoadAsync("openai/whisper-base");
using IBackend backend = new CpuBackend();

WhisperOptions options = new() { Language = "en", WithTimestamps = false };
string text = pipeline.TranscribeWav(backend, "audio.wav", options);

Console.WriteLine(text);
```

Swap `new CpuBackend()` for `new CudaBackend()` or `new VulkanBackend()`; the pipeline is backend-agnostic. Audio pipelines that auto-download (`WhisperPipeline`, `KokoroPipeline`) share this `LoadAsync` convention, while image and video pipelines (`StableDiffusion15Pipeline`, `WanVideoPipeline`) are constructed from pre-loaded components. See the [samples in the repo](https://github.com/HartsyAI/HartsyInference/tree/main/samples) for image, video, and TTS walkthroughs.

---

## What it can do

| Modality | Highlights |
|---|---|
| **LLM text generation** | Qwen2/Qwen3, Llama-3.x, Mistral, and quantized GGUF (Q4/Q8) inference, with a config-driven generic transformer, device-resident KV cache, samplers, and chat templates |
| **Image generation** | SD 1.5, SDXL, SD3 / SD3.5, Flux.1 / Flux.2, AuraFlow, Chroma, HiDream, Qwen-Image, Lumina 2, OmniGen2, HunyuanImage, Ideogram, Kandinsky 5, and more, with LoRA, img2img, tiling, ControlNet, and IP-Adapter |
| **Video generation** | LTX-Video, Wan 2.x, Lance, Kandinsky 5 video, HunyuanVideo |
| **Interactive / world models** | Oasis, DIAMOND: action-conditioned, frame-by-frame generation |
| **Speech-to-text** | Whisper (tiny → large-v3), Moonshine, with streaming and timestamps |
| **Text-to-speech & voice** | Kokoro, F5-TTS, StyleTTS2, Bark, CosyVoice, Spark-TTS, VibeVoice, CSM |
| **Music** | ACE-Step, MusicGen, YuE |
| **Vision** | CLIP & SigLIP embeddings, YOLO detection, SAM segmentation, face detection, Grounding DINO, RT-DETR, depth estimation (Depth-Anything-V2) |
| **3D generation** | Hunyuan3D-2 (flow-match DiT + ShapeVAE) & TripoSR (feed-forward triplane/NeRF) image to mesh, via marching cubes to glTF/OBJ/PLY |

Checkpoints load directly from `.safetensors` / `.gguf`, including quantized weights (GGUF, MXFP4/8, NVFP4, block-scaled). Low-VRAM weight streaming (`HARTSY_LOWVRAM`) lets large image models fit on smaller cards by sliding weights through a bounded window instead of holding them fully resident.

**Multi-GPU sharding** pools VRAM across cards for one model — LLM layer splits (`PlacementConfig.ShardDevices` or a `"cuda:0+cuda:1"` device key; a 32B GGUF that OOMs a 24 GB card runs split across two consumer cards), DiT block sharding for large image/video models, un-quantized audio LMs pooled instead of quantized to fit, text-encoder/VAE placement on a second GPU, and concurrent CFG branches. No NVLink or P2P required — cross-GPU hand-offs host-stage over plain PCIe. All opt-in via `EngineOptions.Placement`; an unconfigured engine is byte-identical to single-GPU. See the repo's [Multi-GPU guide](https://github.com/HartsyAI/HartsyInference/blob/main/docs/MULTI_GPU.md).

> Coverage is wide because the engine shares a common core (tensors, schedulers, VAEs, text encoders, DSP) across architectures. Per-model numerical validation is ongoing; see the alpha note above.

---

## Coming soon

HartsyInference is moving fast, and the roadmap is broad. On deck:

| Area | Planned |
|---|---|
| **Image** | ControlNet tile / inpaint modes, LCM/Turbo distillation across more architectures, regional prompting |
| **Vision** | YOLO-World, OWLv2, Florence-2, pose estimation, OCR, tracking |
| **Video** | CogVideoX, longer-context temporal generation |
| **3D** | Gaussian-splat output, texture synthesis, multi-view to mesh |
| **World models** | Matrix-Game 2.0 / 3.0 and Hunyuan-GameCraft (catalogued, checkpoint loaders still landing), broader action spaces, longer memory horizons, multiplayer state |
| **Performance** | SPIR-V/Vulkan flash-attention parity with the CUDA path, further closing the speed gap vs native runners on non-NVIDIA GPUs |
| **Multi-GPU** | Tensor parallel (NCCL) for NVLink boxes, expert parallel for MoE, >2-way DiT sharding *(the layer-split / DiT-shard / placement / CFG-parallel set shipped 2026-08)* |
| **Tooling** | Wider quantized inference (MXFP4 / MXFP8 / NVFP4), broader SwarmUI-extension model coverage |

Track progress and releases on the [GitHub repo](https://github.com/HartsyAI/HartsyInference).

---

## Design pillars

| Pillar | What it means |
|---|---|
| **Pure C#** | GPU access via PTX (CUDA Driver API) and SPIR-V (Vulkan), with no native shared inference libraries |
| **Eager execution** | Ops run immediately; no computation graph to compile |
| **Zero-allocation hot paths** | Tensor storage in `NativeMemory.AlignedAlloc`; weights memory-mapped; `Span<T>` throughout |
| **Modular packages** | Pull in only the modality and backend you need |

---

## Packages

| Package | Description |
|---|---|
| `HartsyInference` | Meta-package: one reference for the core, all three backends, and every modality package including `HartsyInference.LLM` and `HartsyInference.Engine` (only the sample `Cli`, the unpublished `HartsyInference.API` HTTP server, and the abandoned `Server` project are excluded) |
| `HartsyInference.Core` | Tensor types, `IBackend`, schedulers, pipeline base types |
| `HartsyInference.ModelAssets` | Safetensors/GGUF/PyTorch-pickle loaders, quant dequant, HuggingFace download, model registry, plus CLIP/T5/Whisper/LLM-style tokenizers |
| `HartsyInference.Engine` | Service layer: model lifecycle (registry, download, cache), the `InferenceEngine` facade + per-modality dispatch, backend factory — the CLI, HTTP server, and SwarmUI extension are thin wrappers around it |
| `HartsyInference.Cpu` | CPU backend with AVX2 / AVX-512 / NEON SIMD kernels |
| `HartsyInference.Cuda` | CUDA backend with PTX kernels + cuBLAS |
| `HartsyInference.Vulkan` | Cross-vendor Vulkan backend (NVIDIA / AMD / Intel) via SPIR-V |
| `HartsyInference.LLM` | Native LLM text generation: config-driven transformer (Qwen2/Qwen3/Llama/Mistral), GGUF inference, KV cache, samplers, chat templates |
| `HartsyInference.Diffusion` | Image + music diffusion pipelines, VAEs, text encoders, LoRA |
| `HartsyInference.Audio` | Whisper/Moonshine STT, TTS, voice conversion, music, plus a built-in pure-C# grapheme-to-phoneme (espeak-ng port) for TTS front-ends |
| `HartsyInference.Vision` | CLIP/SigLIP embeddings, YOLO, SAM, face detection |
| `HartsyInference.Video` | LTX-Video, Wan, Lance, Kandinsky 5 video, HunyuanVideo |
| `HartsyInference.World` | Action-conditioned world models (Oasis, DIAMOND) |
| `HartsyInference.ThreeD` | 3D asset generation: mesh/splat foundation (marching cubes, glTF/OBJ/PLY) + Hunyuan3D-2 image to mesh |

---

## Requirements

- **.NET 8 or .NET 10 SDK**

**CUDA backend** (NVIDIA, fastest)
- CUDA 12.x runtime
- NVIDIA GPU, compute capability 8.0+ (RTX 30xx/40xx, A100, H100)

**Vulkan backend** (NVIDIA / AMD / Intel, cross-vendor)
- Vulkan 1.3+ runtime (ships with the GPU vendor driver)
- GPU with FP16 compute (`shaderFloat16`), most discrete GPUs from 2019+

**CPU backend**
- Any x86-64 (AVX2+) or ARM64 (NEON) machine, no GPU required

---

## Links

- **Source & docs:** https://github.com/HartsyAI/HartsyInference
- **Issues:** https://github.com/HartsyAI/HartsyInference/issues
- **SwarmUI extension (recommended way to run it):** [SwarmUI-HartsyInference-Backend](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend)

---

## License

MIT © 2026 kalebbroo
