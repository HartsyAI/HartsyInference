# SharpInference

**A pure C#/.NET 10 AI inference engine for image generation, speech-to-text, text-to-speech, vision, and video — with zero Python dependencies.**

SharpInference loads `.safetensors` and `.gguf` models directly, runs inference on CUDA GPUs (or CPU via SIMD), and exposes an OpenAI-compatible REST API. No Python. No C++ wrappers. No external processes. Just NuGet packages.

Designed to pair with [dotLLM](https://github.com/your-org/dotLLM) for LLM inference, forming a complete AI platform in pure .NET.

---

## Features

- **Image Generation** — Stable Diffusion 1.5, SDXL, Flux, SD3 with LoRA, ControlNet, inpainting, tiling
- **Speech-to-Text** — Whisper (tiny through large-v3) with streaming and timestamps
- **Text-to-Speech** — Kokoro with voice selection and streaming audio output
- **Vision** — CLIP embeddings, YOLO object detection, SAM segmentation
- **Video** — LTX-Video, Wan (future)
- **OpenAI-Compatible API** — drop-in replacement for `/v1/images/generations`, `/v1/audio/transcriptions`, `/v1/audio/speech`
- **SwarmUI Backend** — in-process backend extension, no Python process to manage

## Design Pillars

| Pillar | What It Means |
|---|---|
| **Pure C#** | CUDA accessed via PTX through CUDA Driver API P/Invoke — no native shared libraries |
| **Zero-allocation hot paths** | Tensor data in `NativeMemory.AlignedAlloc`; model weights memory-mapped; `Span<T>` everywhere |
| **Modular NuGet packages** | Pull in only what you need |
| **Production-grade** | Streaming progress, memory budgeting, VRAM monitoring, model hot-swap |

---

## Quick Start

### Image Generation

```xml
<PackageReference Include="SharpInference.Diffusion" />
<PackageReference Include="SharpInference.Cuda" />
<PackageReference Include="SharpInference.ModelHandler" />
```

```csharp
using SharpInference.Diffusion;
using SharpInference.ModelHandler;
using SharpInference.Cuda;

// Load a Stable Diffusion model
var model = await ModelRegistry.LoadAsync("stabilityai/stable-diffusion-xl-base-1.0");
var pipeline = PipelineFactory.Create(model, new CudaBackend());

// Generate an image
var request = new TextToImageRequest
{
    Prompt = "A castle on a mountain at sunset, oil painting",
    NegativePrompt = "blurry, low quality",
    Width = 1024,
    Height = 1024,
    Steps = 20,
    CfgScale = 7.0f,
    Seed = 42
};

await foreach (var progress in pipeline.GenerateAsync(request))
{
    Console.WriteLine($"Step {progress.Step}/{progress.TotalSteps}");
}
```

### Speech-to-Text

```xml
<PackageReference Include="SharpInference.Audio" />
<PackageReference Include="SharpInference.Cpu" />
```

```csharp
using SharpInference.Audio;

var whisper = await WhisperPipeline.LoadAsync("openai/whisper-base");
var transcript = await whisper.TranscribeAsync("audio.wav");
Console.WriteLine(transcript.Text);
```

### OpenAI-Compatible Server

```xml
<PackageReference Include="SharpInference.Server" />
<PackageReference Include="SharpInference.Cuda" />
```

```csharp
var builder = WebApplication.CreateBuilder(args);
builder.Services.AddSharpInference(options =>
{
    options.ModelsDirectory = "~/.sharpinference/models";
    options.DefaultImageModel = "stabilityai/sdxl-base-1.0";
});

var app = builder.Build();
app.MapSharpInferenceEndpoints();
app.Run();
```

Then use any OpenAI client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:5000/v1", api_key="local")
response = client.images.generate(prompt="A cat in space", model="sdxl", size="1024x1024")
```

---

## Packages

| Package | Description |
|---|---|
| `SharpInference.Core` | Tensor types, `IBackend`, schedulers, pipeline interfaces |
| `SharpInference.ModelHandler` | Safetensors/GGUF loaders, model registry, HuggingFace download |
| `SharpInference.Tokenizers` | CLIP, T5, Whisper tokenizers |
| `SharpInference.Cpu` | CPU backend with AVX2/AVX-512/NEON SIMD kernels |
| `SharpInference.Cuda` | CUDA backend with PTX kernels and cuBLAS |
| `SharpInference.Diffusion` | SD1.5, SDXL, Flux, SD3 pipelines, VAE, LoRA, ControlNet |
| `SharpInference.Audio` | Whisper STT, Kokoro TTS, voice conversion |
| `SharpInference.Vision` | CLIP embeddings, YOLO detection, SAM segmentation |
| `SharpInference.Video` | LTX-Video, Wan video generation |
| `SharpInference.Server` | OpenAI-compatible REST API with SSE streaming |
| `SharpInference.SwarmUI` | In-process SwarmUI backend extension |

See [NuGet Package Design](docs/Design/NUGET_PACKAGE_DESIGN.md) for dependency graph and minimum install examples.

---

## Supported Models

### Phase 1 (Current)

| Category | Models |
|---|---|
| **Image** | SD 1.5, SDXL, SDXL Turbo/Lightning, Flux.1-dev, Flux.1-schnell |
| **Audio** | Whisper (tiny → large-v3), Kokoro-82M |
| **Vision** | CLIP ViT-L/14, CLIP ViT-H/14 |

### Phase 2 (Planned)

| Category | Models |
|---|---|
| **Image** | SD3, ControlNet, IP-Adapter, LCM, AuraFlow |
| **Audio** | Parler-TTS, F5-TTS, RVC v2 |
| **Vision** | SigLIP, YOLOv8/v11, Florence-2, SAM 2, DINO v2 |

### Phase 3 (Future)

| Category | Models |
|---|---|
| **Image** | Flux variants, HiDream, LUMINA-Next |
| **Audio** | ACE-Step, MusicGen, Stable Audio, Fish TTS, Orpheus TTS |
| **Video** | LTX-Video, Wan, HunyuanVideo, CogVideoX |

See [Model Support Roadmap](docs/Design/MODEL_SUPPORT_ROADMAP.md) for full details.

---

## Documentation

### Design

| Document | Description |
|---|---|
| [Core Design](docs/Design/CORE_DESIGN.md) | Architecture overview, design pillars, key decisions |
| [Vision & Goals](docs/Design/VISION_AND_GOALS.md) | Why this project exists, the SwarmUI angle |
| [Features](docs/Design/FEATURES.md) | Complete feature list across all modalities |
| [Model Support Roadmap](docs/Design/MODEL_SUPPORT_ROADMAP.md) | Phase 1–3 model support plan |
| [NuGet Package Design](docs/Design/NUGET_PACKAGE_DESIGN.md) | Package breakdown, dependencies, install examples |
| [File Structure](docs/Design/FILE_STRUCTURE.md) | Full project layout |
| [Implementation Details](docs/Design/IMPLEMENTATION_DETAILS.md) | Per-component technical approach |
| [Build Order](docs/Design/BUILD_ORDER.md) | 9-phase implementation sequence |
| [Validation Strategy](docs/Design/VALIDATION_STRATEGY.md) | Reference implementations and tolerances |
| [Research Requirements](docs/Design/RESEARCH_REQUIREMENTS.md) | Research needed before each component |

### Research

Technical research documents for each component — see [Research Requirements](docs/Design/RESEARCH_REQUIREMENTS.md) for the full index.

| Topic | Document |
|---|---|
| Model Formats | [Safetensors](docs/Research/SAFETENSORS_FORMAT.md) · [GGUF](docs/Research/GGUF_FORMAT.md) · [Quantization](docs/Research/QUANTIZATION_DIFFUSION.md) |
| GPU / Compute | [CUDA Driver API](docs/Research/CUDA_DRIVER_API.md) · [PTX Kernels](docs/Research/PTX_KERNELS.md) · [Conv2D CUDA](docs/Research/CONV2D_CUDA.md) · [SIMD .NET](docs/Research/SIMD_INTRINSICS_DOTNET.md) |
| CPU Algorithms | [im2col](docs/Research/IM2COL_CPU.md) · [GroupNorm](docs/Research/GROUPNORM_MATH.md) · [Flash Attention](docs/Research/FLASH_ATTENTION.md) |
| Diffusion Architectures | [SD1.5](docs/Research/SD15_ARCHITECTURE.md) · [SDXL](docs/Research/SDXL_ARCHITECTURE.md) · [Flux](docs/Research/FLUX_ARCHITECTURE.md) · [SD3](docs/Research/SD3_ARCHITECTURE.md) · [VAE](docs/Research/VAE_ARCHITECTURE.md) |
| Diffusion Techniques | [Schedulers](docs/Research/DIFFUSION_SCHEDULERS.md) · [CFG](docs/Research/CFG_AND_GUIDANCE.md) · [LoRA](docs/Research/LORA_FORMAT.md) · [ControlNet](docs/Research/CONTROLNET.md) |
| Text Encoders | [CLIP](docs/Research/CLIP_ARCHITECTURE.md) · [T5](docs/Research/T5_ARCHITECTURE.md) |
| Audio | [Whisper](docs/Research/WHISPER_ARCHITECTURE.md) · [Mel Spectrogram](docs/Research/MEL_SPECTROGRAM.md) · [Kokoro](docs/Research/KOKORO_ARCHITECTURE.md) · [HiFiGAN](docs/Research/HIFIGAN_VOCODER.md) |
| Vision | [YOLO](docs/Research/YOLO_ARCHITECTURE.md) |
| Server | [OpenAI Image API](docs/Research/OPENAI_IMAGE_API.md) |

### Checklists

Phase-by-phase progress tracking:

| Phase | Checklist |
|---|---|
| Phase 1 — Foundation | [Core + ModelHandler + Cpu](docs/Checklists/PHASE_1_FOUNDATION.md) |
| Phase 2 — Math Validation | [Tokenizers + Schedulers + VAE](docs/Checklists/PHASE_2_MATH_VALIDATION.md) |
| Phase 3 — First Image | [Cuda + SD1.5 Pipeline](docs/Checklists/PHASE_3_FIRST_IMAGE.md) |
| Phase 4 — Model Breadth | [SDXL + Flux + LoRA](docs/Checklists/PHASE_4_MODEL_BREADTH.md) |
| Phase 5 — Audio | [Whisper + Kokoro](docs/Checklists/PHASE_5_AUDIO.md) |
| Phase 6 — Vision | [CLIP + YOLO](docs/Checklists/PHASE_6_VISION.md) |
| Phase 7 — Server | [OpenAI-Compatible API](docs/Checklists/PHASE_7_SERVER.md) |
| Phase 8 — SwarmUI | [Backend Extension](docs/Checklists/PHASE_8_SWARMUI.md) |
| Phase 9 — Video | [LTX-Video + Wan](docs/Checklists/PHASE_9_VIDEO.md) |
| Release | [NuGet Publication](docs/Checklists/RELEASE_NUGET.md) |

### Agent Instructions

AI coding agent instruction files for each role — see [CLAUDE.md](CLAUDE.md) for the dispatcher.

| Agent | Purpose |
|---|---|
| [Research](docs/Agents/RESEARCH.md) | Deep-dive topics, produce research documents |
| [Architect](docs/Agents/ARCHITECT.md) | Design implementation plans with file breakdowns |
| [Builder](docs/Agents/BUILDER.md) | Write implementation code |
| [Kernel](docs/Agents/KERNEL.md) | Write SIMD CPU and PTX GPU kernels |
| [Tester](docs/Agents/TESTER.md) | Write tests, validate against references |
| [Reviewer](docs/Agents/REVIEWER.md) | Review code for safety, correctness, performance |
| [Docs](docs/Agents/DOCS.md) | Keep documentation in sync with code |
| [Checklist](docs/Agents/CHECKLIST.md) | Track progress, flag blockers |
| [Benchmark](docs/Agents/BENCHMARK.md) | Performance testing and comparison |
| [Convert](docs/Agents/CONVERT.md) | Model format conversion and quantization |
| [Deploy](docs/Agents/DEPLOY.md) | NuGet packaging and publication |
| [Debug](docs/Agents/DEBUG.md) | Diagnose and fix failures |
| [Refactor](docs/Agents/REFACTOR.md) | Optimize and clean up code |
| [API](docs/Agents/API.md) | Build server REST endpoints |
| [Integration](docs/Agents/INTEGRATION.md) | Wire cross-package boundaries |

---

## Requirements

- **.NET 10** (SDK 10.0+)
- **CUDA 12.x** (for GPU inference — optional, CPU works without it)
- **NVIDIA GPU** with compute capability 8.0+ (RTX 30xx/40xx, A100, H100) for CUDA backend

---

## Project Structure

```
SharpInference/
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

See [File Structure](docs/Design/FILE_STRUCTURE.md) for the complete layout.

---

## License

TBD

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon).
