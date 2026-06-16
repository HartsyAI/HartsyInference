# HartsyInference

**A pure C#/.NET 10 AI inference engine for image, audio, vision, video, 3D, and interactive world models — with zero Python dependencies.**

HartsyInference loads `.safetensors`, `.gguf`, and PyTorch `.pt`/`.ckpt` checkpoints directly and runs inference on **CUDA**, **Vulkan**, or **CPU**. No Python. No C++ wrappers. No external processes. Just NuGet packages. (An OpenAI-compatible REST server is [in progress](#openai-compatible-server-in-progress).)

Designed to pair with [dotLLM](https://github.com/your-org/dotLLM) for LLM inference, forming a complete AI platform in pure .NET.

> [!NOTE]
> HartsyInference covers **everything that isn't an LLM** — diffusion image models, speech, music, vision, video, 3D mesh generation, and real-time playable world models. For text generation, pair it with dotLLM.

---

## Table of Contents

- [Why HartsyInference](#why-hartsyinference)
- [Design Pillars](#design-pillars)
- [Quick Start (CLI)](#quick-start-cli)
- [Quick Start (Library)](#quick-start-library)
- [OpenAI-Compatible Server](#openai-compatible-server)
- [Supported Models](#supported-models)
- [Future Features](#future-features)
- [Packages](#packages)
- [Requirements](#requirements)
- [Documentation](#documentation)
- [Project Structure](#project-structure)

---

## Why HartsyInference

| | |
|---|---|
| **No Python** | The entire stack is C#. No `pip`, no `venv`, no subprocess marshalling, no GIL. |
| **Pure C# CUDA** | GPU kernels are PTX, JIT-compiled through the CUDA Driver API via P/Invoke — no native shared libraries. |
| **Modular NuGet** | Pull in only the modality you need. `HartsyInference.Diffusion` for images, `HartsyInference.Audio` for speech, etc. |
| **World models** | Real-time, action-conditioned interactive generation (keyboard / mouse / camera-pose → streamed frames). |
| **Zero-alloc hot paths** | Tensor data in `NativeMemory.AlignedAlloc`, weights memory-mapped, `Span<T>` everywhere. |
| **Drop-in API** | OpenAI-compatible endpoints (in progress) and an in-process SwarmUI backend. |

---

## Design Pillars

| Pillar | What It Means |
|---|---|
| **Pure C#** | CUDA accessed via PTX through the CUDA Driver API P/Invoke — no native shared libraries |
| **Eager execution** | No computation graphs; ops execute immediately for predictable memory and debugging |
| **Zero-allocation hot paths** | Tensor data in `NativeMemory.AlignedAlloc`; model weights memory-mapped; `Span<T>` everywhere |
| **Multi-backend** | One `IBackend` abstraction over CUDA, Vulkan, and SIMD CPU |
| **Validated** | Every component matches a Python/C++ reference within documented tolerances |
| **Production-grade** | Streaming progress, memory budgeting, VRAM monitoring, model hot-swap |

---

## Quick Start (CLI)

The bundled CLI generates images with Stable Diffusion 1.5 and is the fastest way to verify your setup end-to-end.

```bash
# Build the solution
dotnet build -c Release

# Generate an image on the CPU backend (default)
dotnet run -c Release --project src/HartsyInference.Cli -- \
  --prompt "a painting of a cat sitting on a windowsill" \
  --width 512 --height 512 --steps 20 --cfg 7.5 --seed 42
```

Run it on the GPU instead:

```bash
# NVIDIA — CUDA backend (fastest)
dotnet run -c Release --project src/HartsyInference.Cli -- \
  --backend cuda \
  --prompt "a castle on a mountain at sunset, oil painting" \
  --negative "blurry, low quality" \
  --width 768 --height 768 --steps 25 --cfg 7.0 --seed 1234

# AMD / Intel / NVIDIA — cross-vendor Vulkan backend
dotnet run -c Release --project src/HartsyInference.Cli -- \
  --backend vulkan --prompt "a fox in autumn leaves, studio ghibli style"
```

<details>
<summary><b>All CLI options</b></summary>

```text
Usage: HartsyInference.Cli [options]

Options:
  --backend cpu|vulkan|cuda   Backend to run on (default: cpu)
  --prompt "..."              Positive prompt
  --negative "..."            Negative prompt
  --width N                   Image width  (default: 256)
  --height N                  Image height (default: 256)
  --steps N                   Diffusion steps (default: 20)
  --cfg N.N                   CFG scale (default: 7.5)
  --seed N                    RNG seed (default: 42)
  --models <dir>              Override Models root (default: <repo>/Models)
  --output <dir>              Override Output dir   (default: <repo>/Output)
  -h, --help                  Show this help
```

</details>

> [!TIP]
> Models are resolved from a ComfyUI-style layout under `<repo>/Models` (e.g. `Models/StabilityAI/sd-v1-5`). Override the root with `--models /path/to/models`. Generated images land in `<repo>/Output`.

> [!IMPORTANT]
> The CLI is a focused SD1.5 demo. The **full breadth of models** (audio, vision, video, 3D, world models) is exposed through the library API and the OpenAI-compatible server described below.

---

## Quick Start (Library)

Each modality is its own NuGet package. Expand a section below for the install reference and a minimal end-to-end example.

> [!NOTE]
> There is no one-line auto-loader yet — `PipelineFactory` is still scaffolding. Pipelines are constructed explicitly from pre-loaded components, exactly as the bundled CLIs under [`samples/`](samples/) and [`src/HartsyInference.Cli`](src/HartsyInference.Cli) do. Those programs are the authoritative, compile-tested usage references.

<details>
<summary><b>Image Generation</b> — diffusion text-to-image (SD1.5)</summary>

```xml
<PackageReference Include="HartsyInference.Diffusion" />
<PackageReference Include="HartsyInference.ModelHandler" />
<PackageReference Include="HartsyInference.Tokenizers" />
<PackageReference Include="HartsyInference.Cuda" />
```

```csharp
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

// Resolve a ComfyUI-style layout: tokenizer vocab/merges, text encoder, unet, vae
ModelPaths paths = ModelPaths.FromComfyLayout("Models", "StabilityAI/sd-v1-5");
paths.Validate();

using IBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: "Ptx");

using ClipTokenizer tokenizer = new ClipTokenizer(paths.VocabPath, paths.MergesPath);
int[] promptTokens = tokenizer.Encode("A castle on a mountain at sunset, oil painting");
int[] negativeTokens = tokenizer.Encode("blurry, low quality");

// Build each component and load its weights (cast F16/BF16 -> F32)
ClipTextEncoder textEncoder = new ClipTextEncoder(ClipTextEncoderConfig.Sd15);
textEncoder.LoadWeights(LoadF32(paths.TextEncoderPath), "text_model");

UNet unet = new UNet(UNetConfig.Sd15);
unet.LoadWeights(LoadF32(paths.UNetPath));

VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Sd15);
vaeDecoder.LoadWeights(LoadF32(paths.VaePath));

using StableDiffusion15Pipeline pipeline =
    new StableDiffusion15Pipeline(backend, textEncoder, unet, vaeDecoder);

TextToImageRequest request = new()
{
    Prompt = "A castle on a mountain at sunset, oil painting",
    NegativePrompt = "blurry, low quality",
    Width = 512,
    Height = 512,
    Steps = 20,
    CfgScale = 7.0f,
    Seed = 42,
};

(byte[] rgb, int width, int height, int usedSeed) = pipeline.GenerateFromTokens(
    promptTokens, negativeTokens, request,
    progress => Console.WriteLine($"Step {progress.Step}/{progress.TotalSteps}"));

ImagePostProcessor.SaveBmp("output.bmp", rgb, width, height);

static Dictionary<string, Tensor> LoadF32(string path)
{
    using SafeTensorsLoader loader = new SafeTensorsLoader();
    loader.Load(path);
    Dictionary<string, Tensor> weights = new();
    foreach (KeyValuePair<string, Tensor> kv in loader.GetAllTensors())
        weights[kv.Key] = kv.Value.DType is DType.F16 or DType.BF16 ? kv.Value.CastTo(DType.F32) : kv.Value;
    return weights;
}
```

</details>

<details>
<summary><b>Speech-to-Text</b> — Whisper transcription</summary>

```xml
<PackageReference Include="HartsyInference.Audio" />
<PackageReference Include="HartsyInference.Cpu" />
```

```csharp
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;

// Downloads the checkpoint from HuggingFace on first use
using WhisperPipeline pipeline = await WhisperPipeline.LoadAsync("openai/whisper-base");
using IBackend backend = new CpuBackend();

WhisperOptions options = new() { Language = "en" };
string text = pipeline.TranscribeWav(backend, "audio.wav", options);
Console.WriteLine(text);
```

</details>

<details>
<summary><b>3D Generation</b> — image → mesh</summary>

```xml
<PackageReference Include="HartsyInference.ThreeD" />
<PackageReference Include="HartsyInference.Vision" />
<PackageReference Include="HartsyInference.Cpu" />
```

```csharp
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Requests;
using HartsyInference.Vision.Codec;

(byte[] rgb, int width, int height) = PngDecoder.DecodeFromFile("photo.png");

using IBackend backend = new CpuBackend();
using TripoSrPipeline pipeline = TripoSrPipeline.LoadFromPath(backend, "/models/TripoSR");

ImageTo3DRequest request = new() { ImageRgb = rgb, Width = width, Height = height };
ThreeDResult result = pipeline.Generate(request);

if (result.Mesh is not null)
    GlbWriter.Save("output.glb", result.Mesh);   // glTF binary; OBJ/PLY writers also available
```

</details>

<details>
<summary><b>Interactive World Model</b> — real-time, action-conditioned</summary>

```xml
<PackageReference Include="HartsyInference.Interactive" />
<PackageReference Include="HartsyInference.Cuda" />
```

```csharp
using HartsyInference.Conditioning;
using HartsyInference.Interactive.Sessions;
using HartsyInference.Video;

// A session pushes one ActionInput per frame in and streams VideoFrames out indefinitely.
// Build the concrete session from a model-specific IFrameStepper (e.g. GameCraftFrameStepper);
// see the HartsyInference.Interactive.Tests for full wiring.
await using IInteractiveSession session = new BackgroundComputeSession(stepper, targetFps: 24);

await foreach (VideoFrame frame in session.ReadFramesAsync())
{
    // Feed the next action (keyboard / mouse / camera-pose, encoded into the payload)
    await session.SubmitActionAsync(new ActionInput(actionPayload, frame.Index + 1, timestampNanos: 0));
    Display(frame);
}
```

</details>

> [!NOTE]
> 3D and world-model pipelines are built end-to-end and structurally complete; numerical validation against reference outputs is in progress. See [Supported Models](#supported-models) for per-model status.

---

## OpenAI-Compatible Server

Host an OpenAI-compatible REST API in process — a drop-in replacement for `/v1/images/generations`, `/v1/audio/transcriptions`, and `/v1/audio/speech`, with SSE streaming for progress.

<details>
<summary><b>Host the server</b> — ASP.NET setup</summary>

```xml
<PackageReference Include="HartsyInference.Server" />
<PackageReference Include="HartsyInference.Cuda" />
```

```csharp
var builder = WebApplication.CreateBuilder(args);
builder.Services.AddHartsyInference(options =>
{
    options.ModelsDirectory = "~/.hartsyinference/models";
    options.DefaultImageModel = "stabilityai/sdxl-base-1.0";
});

var app = builder.Build();
app.MapHartsyInferenceEndpoints();
app.Run();
```

</details>

<details>
<summary><b>Call it</b> — curl and OpenAI client examples</summary>

```bash
curl http://localhost:5000/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local" \
  -d '{
        "model": "sdxl",
        "prompt": "A cat in space, photorealistic",
        "size": "1024x1024"
      }'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:5000/v1", api_key="local")
response = client.images.generate(prompt="A cat in space", model="sdxl", size="1024x1024")
```

</details>

---

## Supported Models

> [!NOTE]
> **Status legend** — ✅ Complete · 🧪 Validation-pending (built end-to-end, numerics being verified) · 🏗️ Structural (interfaces wired, forward pass in progress)

### Image Generation

| Model | Architecture | Status |
|---|---|---|
| Stable Diffusion 1.5 | UNet | ✅ |
| SDXL · SDXL Refiner | UNet (dual CLIP) | ✅ |
| SDXL Inpaint | UNet | 🏗️ |
| Flux.1-dev · Flux.2 | Single-stream DiT, flow-matching | ✅ |
| Chroma · Chroma Radiance | Flux-derivative DiT | ✅ |
| SD3 | MMDiT (3 text encoders) | ✅ |
| Qwen-Image | MMDiT (Qwen2.5-VL) | ✅ |
| Hunyuan Image 2.1 | 17B MMDiT | ✅ |
| HiDream i1 | MMDiT (quad encoder + MoE) | ✅ |
| AuraFlow | MMDiT + single-DiT hybrid (Pile-T5-XL) | ✅ |
| Lumina 2.0 | NextDiT (Gemma-2) | ✅ |
| ERNIE-Image | Single-stream DiT (Ministral-3B) | ✅ |
| Kandinsky 5 | DiT (Qwen2.5-VL + CLIP) | ✅ |
| OmniGen 2 | MLLM-based DiT | ✅ |
| Ideogram 4 | 9.3B single-stream DiT | ✅ |
| F-Lite | DiT (Qwen) | 🧪 |
| Lance (Image) | Unified multimodal DiT | 🧪 |
| Z-Image Turbo | NextDiT (Qwen3-4B) | 🏗️ |
| Anima | Cosmos-Predict2-2B (T=1) | 🏗️ |

### Audio

| Model | Category | Status |
|---|---|---|
| Whisper (tiny → large-v3) | Speech-to-text | ✅ |
| Moonshine | Speech-to-text | ✅ |
| Kokoro-82M | Text-to-speech | ✅ |
| Bark | Text-to-speech | ✅ |
| StyleTTS2 | Text-to-speech | ✅ |
| Spark-TTS | Text-to-speech (BiCodec) | ✅ |
| CosyVoice | Text-to-speech (Qwen LM + flow) | ✅ |
| VibeVoice | Text-to-speech (diffusion) | ✅ |
| MusicGen | Music generation | ✅ |
| F5-TTS | Voice cloning | 🏗️ |
| YuE | Music (StableLM) | 🏗️ |
| Codecs — Vocos · EnCodec · DAC · SNAC · Mimi · WavTokenizer · BiCodec · XCodec · Oobleck | Neural audio codecs | ✅ |

### Vision

| Model | Category | Status |
|---|---|---|
| CLIP (ViT-L/14, H/14, bigG/14) | Embeddings | ✅ |
| SigLIP · SigLIP2 | Embeddings | ✅ |
| DINOv2 · DINOv3 | Dense features | ✅ |
| YOLO8 · YOLO11 (n → xl) | Object detection | ✅ |
| SAM · SAM 2 · SAM 2.1 | Segmentation | ✅ |
| RetinaFace-style face detection + landmarks | Face | ✅ |

### Video

| Model | Status |
|---|---|
| LTX-Video | 🧪 |
| Wan 2.2 (T2V + I2V) | 🧪 |
| Lance (Video, T2V) | 🧪 |
| Kandinsky 5 Video | 🧪 |

### 3D

| Model | Task | Status |
|---|---|---|
| TripoSR | Image → mesh (triplane/NeRF) | 🏗️ |
| Hunyuan3D-2 (Shape) | Image/Text → mesh | 🏗️ |

> Built on a reusable mesh / splat / triplane foundation — marching cubes, plus glTF / OBJ / PLY export.

### World / Interactive Models

Real-time, action-conditioned generators — keyboard / mouse / camera-pose input streams in, frames stream out.

| Model | Scale / Target | Status |
|---|---|---|
| Hunyuan-GameCraft 1.0 | 12.5B · 704×1216 @ 33 fps | 🧪 |
| Matrix-Game 3.0 | 5B (+28B MoE) · 720p @ 40 fps · memory-augmented | 🧪 |
| Matrix-Game 2.0 | 1.8B · 540p @ 25 fps | 🧪 |
| Oasis-500m | ~500M · Minecraft world model | 🧪 |

See the [Model Support Roadmap](docs/Design/MODEL_SUPPORT_ROADMAP.md) for the full plan.

---

## Future Features

> [!WARNING]
> These are planned and **not yet implemented**. Tracking lives in the [roadmap](docs/Design/MODEL_SUPPORT_ROADMAP.md).

- **Image** — ControlNet, IP-Adapter, LCM/Turbo distillation across more architectures, regional prompting
- **Vision** — Grounding DINO, YOLO-World, OWLv2, Florence-2, RT-DETR, depth & pose estimation, OCR, tracking
- **Video** — HunyuanVideo, CogVideoX, longer-context temporal generation
- **3D** — Gaussian-splat output, texture synthesis, multi-view → mesh
- **World models** — broader action spaces, longer memory horizons, multiplayer state
- **Tooling** — quantized inference (MXFP4 / MXFP8 / NVFP4), model hot-swap, expanded CLI subcommands per modality

---

## Packages

| Package | Description |
|---|---|
| `HartsyInference.Core` | Tensor types, `IBackend`, schedulers, pipeline interfaces |
| `HartsyInference.ModelHandler` | Safetensors / GGUF / PyTorch loaders, quantization, LoRA, model registry, HuggingFace download |
| `HartsyInference.Tokenizers` | CLIP, T5, Whisper, SentencePiece tokenizers |
| `HartsyInference.Cpu` | CPU backend with AVX2 / AVX-512 / NEON SIMD kernels |
| `HartsyInference.Cuda` | CUDA backend with PTX kernels and cuBLAS |
| `HartsyInference.Vulkan` | Cross-vendor Vulkan backend (SPIR-V compute) |
| `HartsyInference.Diffusion` | Image pipelines — SD/SDXL/Flux/SD3/MMDiT/NextDiT, VAE, LoRA |
| `HartsyInference.Audio` | STT, TTS, music, and neural audio codecs |
| `HartsyInference.Vision` | CLIP/SigLIP/DINO embeddings, YOLO detection, SAM segmentation, face |
| `HartsyInference.Video` | LTX-Video, Wan, Lance, Kandinsky video generation |
| `HartsyInference.ThreeD` | Image/text → 3D mesh; glTF/OBJ/PLY export, marching cubes |
| `HartsyInference.Interactive` | Action-conditioned world models, sessions, action encoders |
| `HartsyInference.Server` | OpenAI-compatible REST API with SSE streaming |
| `HartsyInference.Meta` | Metadata and model-registry utilities |
| `HartsyInference.Cli` | Command-line interface |

See [NuGet Package Design](docs/Design/NUGET_PACKAGE_DESIGN.md) for the dependency graph and minimum install examples.

---

## Requirements

- **.NET 10** (SDK 10.0+)

### CUDA backend (NVIDIA, fastest)
- **CUDA 12.x**
- **NVIDIA GPU** with compute capability 8.0+ (RTX 30xx/40xx, A100, H100)

### Vulkan backend (NVIDIA / AMD / Intel, cross-vendor)
- **Vulkan 1.3+ runtime** — almost always pre-installed by the GPU vendor driver
- **GPU with FP16 compute** (`shaderFloat16`). Most discrete GPUs from 2019+ qualify.

<details>
<summary>Vulkan setup details &amp; validation layers</summary>

- **Vulkan 1.3+ runtime** — almost always pre-installed by the GPU vendor driver
  - **Linux:** `sudo apt install mesa-vulkan-drivers vulkan-tools` (AMD/Intel; NVIDIA blob ships its own ICD)
  - **Windows:** the AMD / Intel / NVIDIA driver ships Vulkan; no extra install
- Validation layers (optional, for debugging) — install the [LunarG Vulkan SDK](https://www.lunarg.com/vulkan-sdk/) and set `HARTSYINFERENCE_VK_VALIDATION=1`.
- See [PHASE_3_5_VULKAN_BACKEND.md](docs/Checklists/PHASE_3_5_VULKAN_BACKEND.md) for current model support and acceptance status.

</details>

### CPU backend (any platform)
- No GPU required — AVX2 / AVX-512 / NEON SIMD with scalar fallback. Slowest, but universally available.

---

## Documentation

<details>
<summary><b>Design documents</b></summary>

| Document | Description |
|---|---|
| [Core Design](docs/Design/CORE_DESIGN.md) | Architecture overview, design pillars, key decisions |
| [Vision & Goals](docs/Design/VISION_AND_GOALS.md) | Why this project exists, the SwarmUI angle |
| [Features](docs/Design/FEATURES.md) | Complete feature list across all modalities |
| [Model Support Roadmap](docs/Design/MODEL_SUPPORT_ROADMAP.md) | Full model support plan |
| [NuGet Package Design](docs/Design/NUGET_PACKAGE_DESIGN.md) | Package breakdown, dependencies, install examples |
| [File Structure](docs/Design/FILE_STRUCTURE.md) | Full project layout |
| [Implementation Details](docs/Design/IMPLEMENTATION_DETAILS.md) | Per-component technical approach |
| [Build Order](docs/Design/BUILD_ORDER.md) | Phase-by-phase implementation sequence |
| [Validation Strategy](docs/Design/VALIDATION_STRATEGY.md) | Reference implementations and tolerances |

</details>

**Research & Checklists** — technical research notes live in [`docs/Research/`](docs/Research/) (model formats, GPU/compute, diffusion architectures, text encoders, audio, vision). Phase-by-phase progress is tracked in [`docs/Checklists/`](docs/Checklists/). AI coding-agent instruction files are in [`docs/Agents/`](docs/Agents/) — see [CLAUDE.md](CLAUDE.md) for the dispatcher.

---

## Project Structure

<details>
<summary><b>Repository layout</b></summary>

```
HartsyInference/
├── CLAUDE.md                  AI agent dispatcher
├── README.md                  ← You are here
├── src/                       Source code (one folder per NuGet package)
├── tests/                     Test projects
├── samples/                   Example applications
├── benchmarks/                Performance benchmarks
├── docs/
│   ├── Design/                Architecture and design documents
│   ├── Research/              Technical research notes
│   ├── Checklists/            Phase progress tracking
│   └── Agents/                AI agent instruction files
└── native/cuda/               CUDA C++ source for PTX generation
```

</details>

See [File Structure](docs/Design/FILE_STRUCTURE.md) for the complete layout.

---

## License

TBD

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon).
