# HartsyInference

**A pure C#/.NET 10 AI inference engine for image, audio, vision, video, 3D, and interactive world models, with zero Python dependencies.**

HartsyInference loads `.safetensors`, `.gguf`, and PyTorch `.pt`/`.ckpt` checkpoints directly and runs inference on **CUDA**, **Vulkan**, or **CPU**. No Python. No C++ wrappers. No external processes. Just NuGet packages. It's a complete pure-C# inference engine: LLMs + diffusion image models + speech/music + vision + video + 3D + interactive worlds.

> [!IMPORTANT]
> **The recommended way to run HartsyInference is inside [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) via the [HartsyInference backend extension](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend).** It registers HartsyInference as a SwarmUI backend (a pure-C# alternative to the ComfyUI backend), so you get a full generation UI, model management, LoRA, and video/audio output with no Python install. HartsyInference is not building its own front-end. You can also consume the engine directly as [NuGet libraries](#quick-start-library) or through the bundled [sample CLIs](#quick-start-cli-developer-tool).

> [!NOTE]
> HartsyInference ships with **native LLM text generation** (Qwen, Llama, Mistral, quantized GGUF inference), plus diffusion image models, speech-to-text, text-to-speech, music generation, vision embeddings & detection, video generation, 3D mesh, and real-time interactive world models, all in C#.

---

## Table of Contents

- [Why HartsyInference](#why-hartsyinference)
- [Design Pillars](#design-pillars)
- [How to Use It (SwarmUI recommended)](#how-to-use-it)
- [Quick Start (Library)](#quick-start-library)
- [Quick Start (CLI, developer tool)](#quick-start-cli-developer-tool)
- [Benchmarks](#benchmarks)
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
| **Pure C# CUDA** | GPU kernels are PTX, JIT-compiled through the CUDA Driver API via P/Invoke, with no native shared libraries. |
| **Modular NuGet** | Pull in only the modality you need. `HartsyInference.Diffusion` for images, `HartsyInference.Audio` for speech, etc. |
| **World models** | Real-time, action-conditioned interactive generation (keyboard / mouse / camera-pose → streamed frames). |
| **Zero-alloc hot paths** | Tensor data in `NativeMemory.AlignedAlloc`, weights memory-mapped, `Span<T>` everywhere. |
| **SwarmUI-native** | Ships as a first-class [SwarmUI backend extension](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend), a pure-C# alternative to the ComfyUI backend, for a full UI with no Python. |

---

## Design Pillars

| Pillar | What It Means |
|---|---|
| **Pure C#** | CUDA accessed via PTX through the CUDA Driver API P/Invoke, with no native shared libraries |
| **Eager execution** | No computation graphs; ops execute immediately for predictable memory and debugging |
| **Zero-allocation hot paths** | Tensor data in `NativeMemory.AlignedAlloc`; model weights memory-mapped; `Span<T>` everywhere |
| **Multi-backend** | One `IBackend` abstraction over CUDA, Vulkan, and SIMD CPU |
| **Validated** | Every component matches a Python/C++ reference within documented tolerances |
| **Production-grade** | Streaming progress, memory budgeting, VRAM monitoring, model hot-swap |

---

## How to Use It

There are three ways to run HartsyInference, in order of how most people should reach for them.

### 1. SwarmUI + the HartsyInference backend extension (recommended)

HartsyInference does not ship its own front-end. The recommended way to actually generate with it is through **[SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI)** using the **[SwarmUI-HartsyInference-Backend](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend)** extension. The extension registers HartsyInference as a SwarmUI backend, a pure-C# alternative to the ComfyUI backend, so you get:

- SwarmUI's full generation UI, model browser, and parameter controls with **no Python environment**.
- Per-architecture model loaders (SD/SDXL/Flux/SD3/Qwen-Image/Ideogram/Kandinsky and more), plus **video** (Wan 2.x, LTX) with ffmpeg muxing, **audio/music** (ACE-Step), LoRA passthrough, live previews, and automatic checkpoint conversion.
- The same engine and kernels this repo builds, consumed as pinned `HartsyInference` NuGet packages.

Install it like any SwarmUI extension: clone `SwarmUI-HartsyInference-Backend` into your SwarmUI `src/Extensions/` folder (as `SwarmUI-HartsyInference`), rebuild SwarmUI, then add a **HartsyInference** backend under Server → Backends. See that repo's README for the current model-support matrix and setup.

### 2. Library (NuGet)

Embed the engine directly in a .NET app. Each modality is its own package. See [Quick Start (Library)](#quick-start-library) below.

### 3. Sample CLIs (developer tool)

The bundled CLIs under [`samples/`](samples/) and [`src/HartsyInference.Cli`](src/HartsyInference.Cli) are compile-tested references for verifying a checkpoint or a pipeline end-to-end from the command line. They are development and validation tools, not the intended end-user surface. See [Quick Start (CLI, developer tool)](#quick-start-cli-developer-tool) below.

---

## Quick Start (CLI, developer tool)

The bundled CLI generates images with Stable Diffusion 1.5 and is the fastest way to verify your setup end-to-end from a terminal. It is a developer/validation tool; for day-to-day generation use the [SwarmUI extension](#how-to-use-it).

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
# NVIDIA: CUDA backend (fastest)
dotnet run -c Release --project src/HartsyInference.Cli -- \
  --backend cuda \
  --prompt "a castle on a mountain at sunset, oil painting" \
  --negative "blurry, low quality" \
  --width 768 --height 768 --steps 25 --cfg 7.0 --seed 1234

# AMD / Intel / NVIDIA: cross-vendor Vulkan backend
dotnet run -c Release --project src/HartsyInference.Cli -- \
  --backend vulkan --prompt "a fox in autumn leaves, studio ghibli style"
```

The CLI is a multi-task dispatcher (`--task image|text|music|vision|video|3d|interactive`). Pick the model for
**any** task with the unified `--model` flag (and `--model-path` for a local checkpoint):

```bash
# Text generation with a local Qwen3 safetensors checkpoint
dotnet run -c Release --project src/HartsyInference.Cli -- \
  --task text --model qwen3 --model-path /models/Qwen3-0.6B \
  --backend cuda --prompt "In one sentence, what is a transformer?" --text-max-tokens 80

# Image generation, selecting the model by its ComfyUI-layout name
dotnet run -c Release --project src/HartsyInference.Cli -- \
  --task image --model StabilityAI/sd-v1-5 --prompt "a fox in autumn leaves"
```

<details>
<summary><b>All CLI options</b></summary>

```text
Usage: HartsyInference.Cli --task <task> [options]

Tasks:
  --task image|text|music|vision|video|3d|interactive   (default: image)

Global:
  --model <name>              Model for ANY task (unified selector; overrides the per-task --*-model flags)
  --model-path <path>         Path to a model checkpoint/dir (any task)
  --backend cpu|vulkan|cuda   Backend to run on (default: cpu)
  --models <dir>              Override Models root (default: <repo>/Models)
  --output <dir>              Override Output dir   (default: <repo>/Output)
  -h, --help                  Show this help

Image task:
  --prompt "..."  --negative "..."  --width N  --height N  --steps N  --cfg N.N  --seed N

Text task:
  --text-model qwen2|qwen3|gguf   --text-model-path <path>   --prompt "..."   --text-max-tokens N
```

</details>

> [!TIP]
> Image models are resolved from a ComfyUI-style layout under `<repo>/Models` (e.g. `Models/StabilityAI/sd-v1-5`). Override the root with `--models /path/to/models`. Generated images land in `<repo>/Output`. Every per-task `--*-model` flag still works as an alias for `--model`.

> [!IMPORTANT]
> The `image` and `text` tasks run end-to-end today (image is wired to the SD1.5 pipeline; text drives the config-driven LLM transformer). `music`, `vision`, `video`, `3d`, and `interactive` are placeholders in this dispatcher CLI. That **full breadth of models** is exposed through the [SwarmUI extension](#how-to-use-it), the library API, and the per-modality samples under [`samples/`](samples/).

---

## Quick Start (Library)

Each modality is its own NuGet package. Expand a section below for the install reference and a minimal end-to-end example.

> [!NOTE]
> `PipelineFactory.DetectArchitecture(path)` and `PipelineFactory.LoadAuto(path, backend)` give a one-line auto-loader today for **SDXL**; other detected families throw a clear `NotSupportedException` naming the architecture, so those pipelines are still constructed explicitly from pre-loaded components. The bundled CLIs under [`samples/`](samples/) and [`src/HartsyInference.Cli`](src/HartsyInference.Cli) are the authoritative, compile-tested usage references for the explicit path.

<details>
<summary><b>Image Generation</b>: diffusion text-to-image (SD1.5)</summary>

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
<summary><b>Text Generation</b>: LLM text generation (Qwen, Llama, Mistral, quantized GGUF)</summary>

```xml
<PackageReference Include="HartsyInference.LLM" />
<PackageReference Include="HartsyInference.Tokenizers" />
<PackageReference Include="HartsyInference.Cuda" />
```

```csharp
using HartsyInference.Core.Backends;
using HartsyInference.Cuda;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;

// CUDA backend (use `new CpuBackend()` for CPU).
using IBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: "Ptx");

// Load a quantized GGUF model. Its tokenizer and chat template are read from the file.
using GgufLanguageModel model = GgufLanguageModel.Load("models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf");
backend.PreloadWeights(model.Transformer.EnumerateWeights());   // keep weights GPU-resident (CUDA)

// Pipeline order is (model, tokenizer, backend, template).
TextGenerationPipeline pipeline = new(model.Transformer, model.Tokenizer, backend, model.Template);

GenerationRequest request = new()
{
    Prompt = "What is the capital of France?",
    MaxTokens = 100,
    Sampling = SamplingOptions.Default with { Temperature = 0.7f, TopP = 0.95f },   // or `with { Greedy = true }`
};

GenerationResult result = pipeline.Generate(request);
Console.WriteLine(result.Text);
```

> For an F32/bf16 **safetensors** checkpoint (no GGUF), build a `GenericTransformer` from a `TransformerConfig`
> preset, call `LoadWeights`, and pair it with a `Qwen2Tokenizer(vocabPath, mergesPath)` — see
> [`samples/HartsyInference.TextGen.Cli`](samples/HartsyInference.TextGen.Cli) for the exact, compile-tested flow.
> On the CPU backend, cast weights to F32 first (the CPU kernels are F32-only).

</details>

<details>
<summary><b>Speech-to-Text</b>: Whisper transcription</summary>

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
<summary><b>3D Generation</b>: image to mesh</summary>

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
<summary><b>Interactive World Model</b>: real-time, action-conditioned</summary>

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

## Benchmarks

We publish real numbers and we are honest about where we stand. HartsyInference is a young pure-C# engine: it is correct across a very wide model set, and it is **not yet as fast as the best native runners**. We are closing that gap in the open. Every number below is reproducible from `benchmarks/` and the committed result files under [`benchmarks/results/`](benchmarks/results/).

### LLM decode vs llama.cpp

RTX 3060 12GB, CUDA, batch=1, 128-token greedy decode, warm, tokens/sec. Same GGUF file and quant on both sides. After a focused kernel optimization pass (fused quantized GEMV, quantized `lm_head`, split-K flash-decode attention, vectorized loads), engine decode went from **20-54× slower to 1.94-2.88× off llama.cpp**:

| Model | Quant | llama.cpp t/s | HartsyInference t/s | Gap |
|---|---|---:|---:|---:|
| Llama-3.2-1B | Q8_0 | 215.9 | ~111.5 | **1.94×** |
| Mistral-7B-v0.3 | Q4_K_M | 66.5 | ~30.7 | **2.12×** |
| Qwen3-0.6B | Q4_K_M | 354.5 | ~157 | **2.26×** |
| Gemma-3-1B | Q4_K_M | 229.8 | ~79.7 | **2.88×** |

Prefill (prompt processing) is much faster and not the bottleneck. The remaining decode gap is launch-overhead on small models; the next lever is CUDA graphs. Details: [`LLM_THROUGHPUT_BENCHMARK.md`](docs/Checklists/LLM_THROUGHPUT_BENCHMARK.md) + [`LLM_DECODE_PERF_GRIND.md`](docs/Checklists/LLM_DECODE_PERF_GRIND.md).

### Diffusion / video end-to-end vs ComfyUI

RTX 4090 24GB, full end-to-end wall-clock through the **SwarmUI API**, the identical request routed to the ComfyUI backend then the HartsyInference backend on the same GPU. Warm (model resident), 512×320, 20-25 steps. All outputs are coherent; this is a **speed** gap, not a correctness gap:

| Model | HartsyInference warm | ComfyUI warm | Gap |
|---|---:|---:|---:|
| Wan 2.1 T2V 1.3B (fp16) | ~23.7 s | 6.28 s | ~3.8× |
| LTX-0.9 2B (fp16) | ~15 s | 2.84 s | ~5.3× |
| Wan 2.2 TI2V-5B (fp16) | ~37.9 s | 4.52 s | ~8.4× |
| Wan 2.1 T2V 14B (fp8) | ~180 s | 30.6 s | ~5.9× |

Image architectures (Flux, SD3, Ideogram) were device-ported and run much closer to ComfyUI; the video DiT blocks are the current optimization frontier. The remaining gap is well understood and documented: no full flash-attention kernel yet, some F32-only elementwise ops, and kernel-launch overhead at small token counts (needs CUDA graphs / op fusion). Full write-up: [`benchmarks/results/video_comfy-vs-hartsy_2026-07-03.md`](benchmarks/results/video_comfy-vs-hartsy_2026-07-03.md).

### GPU op microbenchmarks

Per-op MatMul / Conv2D / norm / SDPA / elementwise timings against PyTorch, with full statistical method (5 trials, 95% CI, Welch's t-test), are committed under [`benchmarks/results/run_baseline_*`](benchmarks/results/) for both RTX 3060 and RTX 4090. See [`benchmarks/README.md`](benchmarks/README.md) to reproduce.

---

## Supported Models

> [!NOTE]
> **Status legend:** ✅ Complete · 🧪 Validation-pending (built end-to-end, numerics being verified) · 🏗️ Structural (interfaces wired, forward pass in progress)

### Language / Text Generation

| Model | Category | Status |
|---|---|---|
| Qwen2.5 (0.5B → 7B) | LLM (native inference) | ✅ |
| Qwen3 (0.6B → 7B) | LLM (native inference) | ✅ |
| Llama-3.x | LLM (native inference) | 🏗️ |
| Mistral (dense) | LLM (native inference) | 🏗️ |
| Quantized GGUF (Q4/Q8) | LLM (quantized inference, all models) | ✅ |

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

### Audio & Music

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
| Fish-Speech / OpenAudio | Text-to-speech (DualAR + tiktoken tokenizer) | 🧪 |
| MusicGen | Music generation | ✅ |
| AudioGen | Sound-effect generation (MusicGen-arch, .bin + T5) | 🧪 |
| ACE-Step | Music generation (flow-matching) | ✅ |
| YuE | Music generation (dual-stage Llama) | ✅ |
| Stable Audio Open | Music generation | 🏗️ |
| F5-TTS | Voice cloning (flow-matching DiT) | 🧪 |
| Codecs (Vocos · EnCodec · DAC · SNAC · Mimi · WavTokenizer · BiCodec · XCodec · Oobleck) | Neural audio codecs | ✅ |

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

> Built on a reusable mesh / splat / triplane foundation: marching cubes, plus glTF / OBJ / PLY export.

### World / Interactive Models

Real-time, action-conditioned generators. Keyboard / mouse / camera-pose input streams in, frames stream out.

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

- **Image:** ControlNet, IP-Adapter, LCM/Turbo distillation across more architectures, regional prompting
- **Vision:** Grounding DINO, YOLO-World, OWLv2, Florence-2, RT-DETR, depth & pose estimation, OCR, tracking
- **Video:** HunyuanVideo, CogVideoX, longer-context temporal generation
- **3D:** Gaussian-splat output, texture synthesis, multi-view to mesh
- **World models:** broader action spaces, longer memory horizons, multiplayer state
- **Tooling:** quantized inference (MXFP4 / MXFP8 / NVFP4), model hot-swap, expanded CLI subcommands per modality

---

## Packages

| Package | Description |
|---|---|
| `HartsyInference.Core` | Tensor types, `IBackend`, schedulers, pipeline interfaces |
| `HartsyInference.ModelHandler` | Safetensors / GGUF / PyTorch loaders, quantization, LoRA, model registry, HuggingFace download |
| `HartsyInference.Tokenizers` | CLIP, T5, Whisper, SentencePiece, Qwen, Llama tokenizers |
| `HartsyInference.Cpu` | CPU backend with AVX2 / AVX-512 / NEON SIMD kernels |
| `HartsyInference.Cuda` | CUDA backend with PTX kernels and cuBLAS |
| `HartsyInference.Vulkan` | Cross-vendor Vulkan backend (SPIR-V compute) |
| `HartsyInference.LLM` | Native LLM text generation (Qwen, Llama, Mistral, GGUF inference, chat templates) |
| `HartsyInference.Phonemizer` | Pure-C# grapheme-to-phoneme (espeak-ng port) for TTS front-ends |
| `HartsyInference.Diffusion` | Image pipelines (SD/SDXL/Flux/SD3/MMDiT/NextDiT), VAE, LoRA |
| `HartsyInference.Audio` | STT, TTS, music generation, and neural audio codecs |
| `HartsyInference.Vision` | CLIP/SigLIP/DINO embeddings, YOLO detection, SAM segmentation, face |
| `HartsyInference.Video` | LTX-Video, Wan, Lance, Kandinsky video generation |
| `HartsyInference.ThreeD` | Image/text → 3D mesh; glTF/OBJ/PLY export, marching cubes |
| `HartsyInference.Interactive` | Action-conditioned world models, sessions, action encoders |
| `HartsyInference` | Meta-package that references the core, backends, and modality packages (add `HartsyInference.LLM` separately for text generation) |
| `HartsyInference.Cli` | Command-line sample/validation tool (not published as a package) |

See [NuGet Package Design](docs/Design/NUGET_PACKAGE_DESIGN.md) for the dependency graph and minimum install examples.

---

## Requirements

- **.NET 10** (SDK 10.0+)

### CUDA backend (NVIDIA, fastest)
- **CUDA 12.x**
- **NVIDIA GPU** with compute capability 8.0+ (RTX 30xx/40xx, A100, H100)

### Vulkan backend (NVIDIA / AMD / Intel, cross-vendor)
- **Vulkan 1.3+ runtime**, almost always pre-installed by the GPU vendor driver
- **GPU with FP16 compute** (`shaderFloat16`). Most discrete GPUs from 2019+ qualify.

<details>
<summary>Vulkan setup details &amp; validation layers</summary>

- **Vulkan 1.3+ runtime**, almost always pre-installed by the GPU vendor driver
  - **Linux:** `sudo apt install mesa-vulkan-drivers vulkan-tools` (AMD/Intel; NVIDIA blob ships its own ICD)
  - **Windows:** the AMD / Intel / NVIDIA driver ships Vulkan; no extra install
- Validation layers (optional, for debugging): install the [LunarG Vulkan SDK](https://www.lunarg.com/vulkan-sdk/) and set `HARTSYINFERENCE_VK_VALIDATION=1`.
- See [PHASE_3_5_VULKAN_BACKEND.md](docs/Checklists/PHASE_3_5_VULKAN_BACKEND.md) for current model support and acceptance status.

</details>

### CPU backend (any platform)
- No GPU required. AVX2 / AVX-512 / NEON SIMD with scalar fallback. Slowest, but universally available.

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

**Model status:** which models are built versus **verified end-to-end** is tracked per modality, indexed in [`docs/Checklists/MODEL_STATUS.md`](docs/Checklists/MODEL_STATUS.md) ([Image](docs/Checklists/MODEL_STATUS_IMAGE.md), [Audio](docs/Checklists/MODEL_STATUS_AUDIO.md), [Video](docs/Checklists/MODEL_STATUS_VIDEO.md), [World](docs/Checklists/MODEL_STATUS_WORLD.md), [3D](docs/Checklists/MODEL_STATUS_3D.md), [Vision](docs/Checklists/MODEL_STATUS_VISION.md), [LLM](docs/Checklists/MODEL_STATUS_LLM.md)). The cross-modality real-weight parity authority is [`docs/Checklists/PARITY_VERIFICATION.md`](docs/Checklists/PARITY_VERIFICATION.md).

**Research & Checklists:** technical research notes live in [`docs/Research/`](docs/Research/) (model formats, GPU/compute, diffusion architectures, text encoders, audio, vision). Phase-by-phase progress is tracked in [`docs/Checklists/`](docs/Checklists/). AI coding-agent instruction files are in [`docs/Agents/`](docs/Agents/); see [CLAUDE.md](CLAUDE.md) for the dispatcher.

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
