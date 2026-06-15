# HartsyInference

**A pure C#/.NET AI inference engine for image generation, speech, vision, video, and interactive world models — no Python, no native runtime DLLs.**

HartsyInference loads `.safetensors` and `.gguf` checkpoints directly and runs them on NVIDIA CUDA, cross-vendor Vulkan, or CPU SIMD — entirely in managed .NET. GPU kernels are PTX/SPIR-V shipped with the package and JIT-compiled at runtime; there are no C++ wrappers, no bundled native inference library, and no external Python process to manage. Just NuGet packages.

It is the non-LLM companion to [dotLLM](https://github.com/kalebbroo/dotLLM): together they form a complete AI stack in pure .NET.

---

## ⚠️ Alpha software

**This is `1.0.0-alpha` — an early, fast-moving preview.** Use it to experiment, not in production.

- **APIs will change without notice** between alpha releases. Pin an exact version.
- **Model coverage is broad but maturity varies.** Many architectures are implemented and load/run end-to-end but are still being validated numerically against their reference implementations. Treat output quality per-model as "verify before you rely on it."
- **No support guarantees, no semver stability** until `1.0.0`.
- The OpenAI-compatible **server and CLI are not published as packages** in this alpha — they live in the source repository.

Found a bug or a mismatch against a reference? Please [open an issue](https://github.com/kalebbroo/HartsyInference/issues).

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

## Quick start — speech-to-text

The Whisper pipeline downloads a checkpoint from HuggingFace on first use and runs on whichever backend you pass:

```csharp
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;          // or HartsyInference.Cuda / HartsyInference.Vulkan

using WhisperPipeline pipeline = await WhisperPipeline.LoadAsync("openai/whisper-base");
using IBackend backend = new CpuBackend();

var options = new WhisperOptions { Language = "en", WithTimestamps = false };
string text = pipeline.TranscribeWav(backend, "audio.wav", options);

Console.WriteLine(text);
```

Swap `new CpuBackend()` for `new CudaBackend()` or `new VulkanBackend()` — the pipeline is backend-agnostic. The same `LoadAsync` / pipeline pattern applies across modalities (`StableDiffusion15Pipeline`, `WanVideoPipeline`, `KokoroPipeline`, …); see the [samples in the repo](https://github.com/kalebbroo/HartsyInference/tree/main/samples) for image, video, and TTS walkthroughs.

---

## What it can do

| Modality | Highlights |
|---|---|
| **Image generation** | SD 1.5, SDXL, SD3, Flux.1 / Flux.2, AuraFlow, Chroma, HiDream, Qwen-Image, Lumina 2, OmniGen2, HunyuanImage, Ideogram, Kandinsky 5, and more — with LoRA, img2img, and tiling |
| **Video generation** | LTX-Video, Wan 2.x, Lance, Kandinsky 5 video |
| **Interactive / world models** | Matrix-Game 2 & 3, Oasis — action-conditioned, frame-by-frame generation |
| **Speech-to-text** | Whisper (tiny → large-v3), Moonshine — with streaming and timestamps |
| **Text-to-speech & voice** | Kokoro, F5-TTS, StyleTTS2, Bark, CosyVoice, Spark-TTS, VibeVoice, CSM |
| **Music** | ACE-Step, MusicGen, YuE |
| **Vision** | CLIP & SigLIP embeddings, YOLO detection, SAM segmentation, face detection |
| **3D generation** | Hunyuan3D-2 (flow-match DiT + ShapeVAE) & TripoSR (feed-forward triplane/NeRF) image→mesh → marching cubes → glTF/OBJ/PLY |

Checkpoints load directly from `.safetensors` / `.gguf`, including quantized weights (GGUF, MXFP4/8, NVFP4, block-scaled).

> Coverage is wide because the engine shares a common core (tensors, schedulers, VAEs, text encoders, DSP) across architectures. Per-model numerical validation is ongoing — see the alpha note above.

---

## Design pillars

| Pillar | What it means |
|---|---|
| **Pure C#** | GPU access via PTX (CUDA Driver API) and SPIR-V (Vulkan) — no native shared inference libraries |
| **Eager execution** | Ops run immediately; no computation graph to compile |
| **Zero-allocation hot paths** | Tensor storage in `NativeMemory.AlignedAlloc`; weights memory-mapped; `Span<T>` throughout |
| **Modular packages** | Pull in only the modality and backend you need |

---

## Packages

| Package | Description |
|---|---|
| `HartsyInference` | Meta-package — references everything below |
| `HartsyInference.Core` | Tensor types, `IBackend`, schedulers, pipeline base types |
| `HartsyInference.ModelHandler` | Safetensors/GGUF loaders, quant dequant, HuggingFace download, model registry |
| `HartsyInference.Tokenizers` | CLIP, T5, Whisper, and LLM-style tokenizers |
| `HartsyInference.Cpu` | CPU backend with AVX2 / AVX-512 / NEON SIMD kernels |
| `HartsyInference.Cuda` | CUDA backend — PTX kernels + cuBLAS |
| `HartsyInference.Vulkan` | Cross-vendor Vulkan backend (NVIDIA / AMD / Intel) via SPIR-V |
| `HartsyInference.Diffusion` | Image + music diffusion pipelines, VAEs, text encoders, LoRA |
| `HartsyInference.Audio` | Whisper/Moonshine STT, TTS, voice conversion, music |
| `HartsyInference.Vision` | CLIP/SigLIP embeddings, YOLO, SAM, face detection |
| `HartsyInference.Video` | LTX-Video, Wan, Lance, Kandinsky 5 video |
| `HartsyInference.Interactive` | Action-conditioned world models (Matrix-Game, Oasis) |
| `HartsyInference.ThreeD` | 3D asset generation — mesh/splat foundation (marching cubes, glTF/OBJ/PLY) + Hunyuan3D-2 image→mesh |

---

## Requirements

- **.NET 8 or .NET 10 SDK**

**CUDA backend** (NVIDIA, fastest)
- CUDA 12.x runtime
- NVIDIA GPU, compute capability 8.0+ (RTX 30xx/40xx, A100, H100)

**Vulkan backend** (NVIDIA / AMD / Intel, cross-vendor)
- Vulkan 1.3+ runtime (ships with the GPU vendor driver)
- GPU with FP16 compute (`shaderFloat16`) — most discrete GPUs from 2019+

**CPU backend**
- Any x86-64 (AVX2+) or ARM64 (NEON) machine — no GPU required

---

## Links

- **Source & docs:** https://github.com/kalebbroo/HartsyInference
- **Issues:** https://github.com/kalebbroo/HartsyInference/issues
- **LLM companion:** [dotLLM](https://github.com/kalebbroo/dotLLM)

---

## License

MIT © 2026 kalebbroo
