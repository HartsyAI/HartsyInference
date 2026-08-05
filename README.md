# HartsyInference

**A pure C#/.NET AI inference engine for LLMs, image, audio, vision, video, 3D, and interactive world models — with zero Python dependencies.**

HartsyInference loads `.safetensors`, `.gguf`, and PyTorch `.pt`/`.ckpt` checkpoints directly and runs inference on **CUDA**, **Vulkan**, or **CPU** — no Python, no C++ wrappers, no external processes, just NuGet packages. One engine spans native LLM text generation, diffusion image models, speech/music, vision, video, 3D mesh, and real-time interactive worlds. Targets **.NET 8 and .NET 10**.

> [!IMPORTANT]
> **The recommended way to run HartsyInference is inside [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) via the [HartsyInference backend extension](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend)** — it registers HartsyInference as a pure-C# alternative to the ComfyUI backend, giving you a full generation UI, model management, LoRA, and video/audio output with no Python install. The engine is not building its own front-end; you can also embed it as [NuGet libraries](#quick-start-library) or drive the [sample CLI](#quick-start-cli-developer-tool). For per-model status across every modality, see [Models](#models).

---

## Table of Contents

- [Why HartsyInference](#why-hartsyinference)
- [How to Use It (SwarmUI recommended)](#how-to-use-it)
- [Quick Start (Library)](#quick-start-library)
- [Quick Start (CLI, developer tool)](#quick-start-cli-developer-tool)
- [Multi-GPU & Sharding](#multi-gpu--sharding)
- [Benchmarks](#benchmarks)
- [Models](#models)
- [Future Features](#future-features)
- [Packages](#packages)
- [Requirements](#requirements)
- [Documentation](#documentation)
- [Project Structure](#project-structure)

---

## Why HartsyInference

| | |
|---|---|
| **No Python** | The whole stack is C# — no `pip`, no `venv`, no subprocess marshalling, no GIL. |
| **Pure-C# GPU** | CUDA kernels are PTX, JIT-compiled through the CUDA Driver API via P/Invoke; Vulkan via SPIR-V. No native shared libraries. |
| **Multi-backend** | One `IBackend` abstraction over CUDA (NVIDIA), Vulkan (AMD/Intel/NVIDIA), and SIMD CPU (AVX2/AVX-512/NEON). |
| **Eager execution** | No computation graphs; ops execute immediately for predictable memory and debugging. |
| **Zero-alloc hot paths** | Tensor data in `NativeMemory.AlignedAlloc`, weights memory-mapped, `Span<T>` everywhere. |
| **Modular NuGet** | Pull in only the modality you need — `HartsyInference.Diffusion` for images, `HartsyInference.Audio` for speech, etc. |
| **Validated** | Every component matches a Python/C++ reference within documented tolerances. |
| **World models** | Real-time, action-conditioned interactive generation (keyboard / mouse / camera-pose → streamed frames). |
| **Runs models bigger than your GPU** | VRAM-aware weight streaming: the engine measures free VRAM per generation phase and streams a denoiser's blocks from host RAM only when the model would not otherwise fit. A 12 GB card runs models needing ~20 GB resident. Automatic by default, and switchable off for operators who prefer a hard failure. |
| **Multi-GPU sharding** | One model **split across GPUs, VRAM pooled** — LLM layer splits (a 32B that OOMs a 24 GB card runs at ~12 tok/s across 4090+3060), DiT block sharding for big image/video models, un-quantized audio LMs pooled instead of crushed to fit, TE/VAE placement, and parallel CFG branches. Works over plain PCIe — **no NVLink/P2P required**. See [Multi-GPU & Sharding](#multi-gpu--sharding). |
| **Production-grade** | Streaming progress, memory budgeting, VRAM monitoring, model hot-swap. |
| **SwarmUI-native** | Ships as a first-class [SwarmUI backend extension](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend), a pure-C# alternative to the ComfyUI backend. |

---

## How to Use It

HartsyInference does not ship its own front-end. There are a few ways to run it, in order of how most people should reach for them.

**1. SwarmUI + the HartsyInference backend extension (recommended).** Install the [SwarmUI-HartsyInference-Backend](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend) extension to register HartsyInference as a SwarmUI backend — a pure-C# alternative to the ComfyUI backend. You get SwarmUI's full generation UI, model browser, and parameter controls with **no Python environment**, plus per-architecture model loaders, video (Wan 2.x, LTX) with ffmpeg muxing, audio/music (ACE-Step), LoRA passthrough, live previews, and automatic checkpoint conversion — all on the same engine and kernels this repo builds, consumed as pinned `HartsyInference` NuGet packages. Clone it into your SwarmUI `src/Extensions/` folder, rebuild, and add a **HartsyInference** backend under Server → Backends; see that repo's README for the model-support matrix and setup.

**2. Library (NuGet).** Embed the engine directly in a .NET app; each modality is its own package. See [Quick Start (Library)](#quick-start-library).

**3. Sample CLI (developer tool).** The bundled [`hartsy` CLI](#quick-start-cli-developer-tool) drives every modality from the terminal — the fastest way to verify a checkpoint end-to-end. It's a development/validation tool, not the intended end-user surface.

**4. OpenAI-compatible HTTP server.** `HartsyInference.API` hosts an OpenAI-shaped REST API: `/v1/chat/completions` (LLM/SSM chat — streaming, non-streaming, and JSON-mode — with **continuous batching** and a **paged KV cache**), `/v1/images/generations` (+ a streaming variant), and `/v1/models` load / list / pull / unload. Audio and image-edit routes are shaped for API completeness but return `501` until wired. It runs from source (`IsPackable=false`), CPU by default; set `HartsyInference:Backend=Cuda` + `HartsyInference:PtxDirectory` for GPU. See [`ROADMAP.md`](docs/Checklists/ROADMAP.md) for what's left before it's published.

```bash
dotnet run --project src/HartsyInference.API -c Release --urls http://127.0.0.1:8080
curl -X POST http://127.0.0.1:8080/v1/models/load -H "Content-Type: application/json" \
  -d '{"model":"/path/to/model.gguf"}'
curl -X POST http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"/path/to/model.gguf","messages":[{"role":"user","content":"Hello!"}]}'
```

---

## Quick Start (CLI, developer tool)

The bundled **`hartsy`** CLI ([`src/HartsyInference.Cli`](src/HartsyInference.Cli), built on Spectre.Console) drives every modality from the terminal — the fastest way to verify a checkpoint end-to-end. It's a developer/validation tool; for day-to-day generation use the [SwarmUI extension](#how-to-use-it). Run it with **no arguments** for an interactive REPL.

```bash
dotnet build -c Release

# Image (diffusion checkpoint; SDXL auto-constructs). The prompt is a positional argument.
dotnet run -c Release --project src/HartsyInference.Cli -- \
  image "a castle on a mountain at sunset, oil painting" \
  --model-path Models/sdxl.safetensors --steps 25 -b cuda

# Text — streams tokens from a local LLM (safetensors dir or .gguf)
dotnet run -c Release --project src/HartsyInference.Cli -- \
  text "In one sentence, what is a transformer?" -m qwen3 --model-path /models/Qwen3-0.6B -b cuda

# Transcribe a WAV with Whisper
dotnet run -c Release --project src/HartsyInference.Cli -- transcribe speech.wav -m whisper-base
```

Commands span every modality — `text`, `image`, `transcribe`, `speak`, `3d`, `vision`, `music`, `video`, `world`, `restore` (SeedVR2 video/image restoration: `hartsy restore old_clip.mp4` → upscaled/deartifacted PNG frames + MP4; also chainable as `hartsy video ... --restore`), `convert` (voice conversion), `fx separate` / `fx enhance` (stem separation / speech enhancement) — plus catalog helpers `list`, `models`, `pull`, and `preview` (inline terminal image display). Common flags: `-b|--backend cpu|cuda|vulkan`, `-m|--model <name>`, `--model-path <path>`. Multi-GPU placement flags are shared across generation commands — `--dit-shard-gpu`, `--lm-shard-gpu`, `--te-gpu`, `--vae-gpu`, `--cfg-parallel-gpu` (and `--device "cuda:0+cuda:1"` for text) — see [Multi-GPU & Sharding](#multi-gpu--sharding). Run `hartsy <command> --help` for a command's full option set.

> [!TIP]
> `pull` downloads a model from HuggingFace (or registers a local path) into the cache; `list` and `models` show the catalog and what's already cached. Image checkpoints also resolve from a ComfyUI-style layout under `<repo>/Models`.

---

## Quick Start (Library)

Each modality is its own NuGet package. Expand a section below for the install reference and a minimal end-to-end example.

> [!NOTE]
> `PipelineFactory.DetectArchitecture(path)` and `PipelineFactory.LoadAuto(path, backend)` give a one-line auto-loader today for **SDXL**; other detected families throw a clear `NotSupportedException` naming the architecture, so those pipelines are still constructed explicitly from pre-loaded components. The bundled CLIs under [`samples/`](samples/) and [`src/HartsyInference.Cli`](src/HartsyInference.Cli) are the authoritative, compile-tested usage references for the explicit path.

<details>
<summary><b>Image Generation</b>: diffusion text-to-image (SD1.5)</summary>

```xml
<PackageReference Include="HartsyInference.Diffusion" />
<PackageReference Include="HartsyInference.ModelAssets" />
<PackageReference Include="HartsyInference.ModelAssets.Tokenizers" />
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
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

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
<PackageReference Include="HartsyInference.ModelAssets.Tokenizers" />
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
<PackageReference Include="HartsyInference.World" />
<PackageReference Include="HartsyInference.Cuda" />
```

```csharp
using HartsyInference.Conditioning;
using HartsyInference.World.Sessions;
using HartsyInference.Video;

// A session pushes one ActionInput per frame in and streams VideoFrames out indefinitely.
// Build the concrete session from a model-specific IFrameStepper (e.g. GameCraftFrameStepper);
// see the HartsyInference.World.Tests for full wiring.
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
> 3D and world-model pipelines are built end-to-end and structurally complete; numerical validation against reference outputs is in progress. See [Models](#models) for per-model status.

---

## Multi-GPU & Sharding

HartsyInference can spread **one model across several GPUs with the VRAM pooled** — not just one
generation per GPU. It works over plain PCIe with **no NVLink and no P2P required** (cross-GPU
hand-offs host-stage automatically; mismatched consumer cards are the primary tested rig — an
RTX 4090 + RTX 3060). Everything is opt-in: an unconfigured engine is byte-identical to single-GPU.

| Mode | What it buys you | Example (measured) |
|---|---|---|
| **LLM layer split** (`--device "cuda:0+cuda:1"`) | Text models bigger than any one card: layers split proportionally to free VRAM, KV cache follows | **Qwen3-32B** Q4_K_M OOMs a 24 GB 4090 alone → runs at **~12 tok/s** split across 4090+3060 |
| **DiT block sharding** (`--dit-shard-gpu`) | Big image/video diffusion models fully resident across 2 cards instead of streaming from RAM | **Qwen-Image 20B** resident at 13.4 + 6.2 GB, SSIM 0.9734 vs single-GPU; Krea2, Flux.1, Chroma, HunyuanImage, MiniMax-H3 verified |
| **Audio-LM split** (`--lm-shard-gpu`) | Big music LMs run **un-quantized** pooled across cards instead of being quantized down to fit one | **YuE's 7B Stage-1** at checkpoint bf16 in 8.7 + 4.3 GB pooled (single-GPU default crushes it to Q4_K) |
| **TE / VAE placement** (`--te-gpu`, `--vae-gpu`) | Multi-GB text encoders / VAE off the denoiser's card — often a straight wall-clock win | **Wan TI2V-5B: 43.7 s → 32.7 s** with umT5 on the second card |
| **CFG-parallel** (`--cfg-parallel-gpu`) | Positive + negative prompt branches run **concurrently** on two cards (weights replicated) | ~1.8-1.9× per-step concurrency on Wan / Flux true-CFG; auto-falls-back when the model doesn't fit both |
| **Same-GPU multi-tenant** | Two independent engine backends share one physical GPU with isolated state | Serialized per-GPU by default; enterprise co-tenancy |

Honest framing: **sharding pools VRAM, it does not add speed** — a pipeline split runs its stages
sequentially, so per-step time is the same or a few percent slower (boundary copies). The win is that
models that *couldn't run* now run, and models that had to be quantized or streamed now run resident at
full precision. Placement and CFG-parallel, by contrast, can be outright faster.

**The full guide — every setting (SwarmUI extension / CLI / `PlacementConfig` library API), mechanics,
verified-model matrix, and limits: [`docs/MULTI_GPU.md`](docs/MULTI_GPU.md).** Measured tables:
[`benchmarks/results/2026-08-05_multigpu_speeds.md`](benchmarks/results/2026-08-05_multigpu_speeds.md).
Every mode is regression-guarded by a real-weight campaign (`tests/run-multigpu-campaign.sh`). Tensor
parallel (NCCL) and expert parallel are designed but not yet built — see [`ROADMAP.md`](docs/Checklists/ROADMAP.md).

---

## Benchmarks

We publish real numbers and we are honest about where we stand. HartsyInference is a young pure-C# engine: it is correct across a very wide model set, and on several flagship image models it is now **faster than ComfyUI on the same GPU** — while other models are still mid-optimization. We are closing the remaining gaps in the open. Each modality has one canonical scoreboard table under [`benchmarks/scoreboards/`](benchmarks/scoreboards/) — that's where the numbers, methodology, and sources live now.

**Headlines** (see each scoreboard for the full per-model table, GPU, date, and source):

| Modality | Baseline | Standout results | Scoreboard |
|---|---|---|---|
| Image (T2I / edit) | ComfyUI (same GPU), Python for a couple of no-Comfy-graph models | Several flagship turbo/distilled models faster than ComfyUI (Z-Image-Turbo, Krea2, SDXL, Flux-Schnell); larger 20-30 step models trail by 1.2-1.7× | [`IMAGE.md`](benchmarks/scoreboards/IMAGE.md) |
| LLM decode | llama.cpp, same GGUF/quant | 8 of 9 core text models at-or-ahead of llama.cpp; GLM-4-9B at 0.98× (effective parity) | [`LLM.md`](benchmarks/scoreboards/LLM.md) |
| Video (T2V) | ComfyUI (same GPU) | Still behind ComfyUI on most video DiTs (1.1×-8×, architecture-dependent); the current optimization frontier | [`VIDEO.md`](benchmarks/scoreboards/VIDEO.md) |
| Audio (TTS / STT / Music / VC / Fx) | Model-specific Python reference or self-comparison — no shared engine exists for audio | RTF up to ~15× real-time on the fastest STT models; several TTS models still sub-real-time | [`AUDIO.md`](benchmarks/scoreboards/AUDIO.md) |
| 3D mesh (image → mesh) | Python reference (`tsr`, `hy3dgen`) | Within 1.2-1.6× of the Python reference after a multi-round optimization campaign | [`THREED.md`](benchmarks/scoreboards/THREED.md) |

These times require **zero configuration**: the engine's standard performance profile (cuDNN fused flash attention, fp8 tensor-core GEMM, F16 DiT activations, resident weights, warm activation pool) is default-on with per-feature kill-switches and graceful fallbacks.

**Image conditioning features (all verified end-to-end through SwarmUI, 2026-07-16/17):** FLUX.1 Kontext instruction editing, FLUX.1 Fill inpaint/outpaint, FLUX.1 Canny / Depth (with an in-engine Depth-Anything-V2 annotator, parity 2.9e-7 vs the official implementation; the FLUX-Depth conditioning map is numerically exact to BFL's own `DepthImageEncoder`, corr 1.000000), FLUX.1 Redux image variation (SigLIP + projection numerically A/B'd vs `FluxPriorReduxPipeline`, tokens corr 1.000000), and FLUX DiT ControlNet (union + single-mode, parity 3.7e-9 vs diffusers). SDXL + SD1.5 ControlNet with a full in-engine preprocessor set (canny / depth / openpose / lineart / softedge / scribble / normal / **segmentation** — UperNet-ConvNeXt ADE20K, 100% class parity), multi-net stacking, start/end step windows, both diffusers and original LDM checkpoint layouts, plus **union-type SDXL ControlNet** (xinsir controlnet-union ProMax, all residuals corr ≥0.9999998). IP-Adapter across SDXL standard / Plus / Plus-Face, SD1.5, **FaceID** (ArcFace IR-50, cosine 1.000000) and **FaceID-Plus / FaceID-PlusV2** (SD1.5 + SDXL, projection corr 1.000000). Instruction-edit models: OmniGen2, Boogu-Edit, Qwen-Image-Edit 2511. See [`docs/Checklists/MODEL_STATUS_IMAGE.md`](docs/Checklists/MODEL_STATUS_IMAGE.md).

The server (`HartsyInference.API`) additionally supports **continuous batching** for LLMs — concurrently-submitted requests against the same model share decode rounds instead of running one at a time — and a **paged KV cache**, both independent of graph decode. LLM details: [`ROADMAP.md`](docs/Checklists/ROADMAP.md) + [`ROADMAP.md`](docs/Checklists/ROADMAP.md). Audio full-fleet verification methodology and per-model bug writeups: [`ROADMAP.md`](docs/Checklists/ROADMAP.md). GPU op microbenchmarks (MatMul / Conv2D / norm / SDPA / elementwise vs PyTorch) reproduce via [`benchmarks/README.md`](benchmarks/README.md).

---

## Models

The engine covers a very wide model set across every modality. The **[benchmark tables](#benchmarks)** above show the representative models with measured numbers; the **authoritative per-model status** — verified end-to-end vs built-but-pending, with bring-up notes and real-weight parity evidence — lives in the modality status docs:

| Modality | Breadth (representative) | Status doc |
|---|---|---|
| **Language / LLM** | Llama, Qwen2/Qwen3, Gemma 2/3/4, Phi, Mistral (dense); Qwen3.5 gated-DeltaNet hybrid; MoE/MLA giants (Mixtral, Qwen3-MoE, DeepSeek-V3, Kimi-K2, GPT-OSS); VLMs; embeddings/rerankers; Mamba/RWKV/T5 — quantized GGUF throughout | [MODEL_STATUS_LLM](docs/Checklists/MODEL_STATUS_LLM.md) · [coverage](docs/Checklists/MODEL_STATUS_LLM.md) |
| **Image** | SD1.5 / SDXL (UNet); Flux.1/.2, SD3, Chroma / Radiance, Qwen-Image (+ Edit), HunyuanImage, HiDream, AuraFlow, Lumina 2, ERNIE-Image, Kandinsky 5, OmniGen 2, Ideogram 4 (DiT / MMDiT / NextDiT) | [MODEL_STATUS_IMAGE](docs/Checklists/MODEL_STATUS_IMAGE.md) |
| **Audio & Music** | Whisper / Moonshine (STT); Kokoro, Piper, StyleTTS2, Bark, Spark-TTS, CosyVoice, VibeVoice, MeloTTS, F5-TTS clone (TTS); ACE-Step, MusicGen, YuE, HeartMuLa (music); 9 neural codecs | [MODEL_STATUS_AUDIO](docs/Checklists/MODEL_STATUS_AUDIO.md) |
| **Vision** | CLIP / SigLIP / DINOv2-3 embeddings; YOLO8 / YOLO11 / RT-DETR / Grounding DINO detection; SAM / SAM 2 / 2.1 segmentation; Depth-Anything-V2 depth estimation; face detection | [MODEL_STATUS_VISION](docs/Checklists/MODEL_STATUS_VISION.md) |
| **Video** | LTX-Video, Wan 2.x (T2V + I2V), HunyuanVideo, Lance, Kandinsky 5 Video; **SeedVR2 3B/7B video+image restoration** (`hartsy restore` — one-step upscale/deartifact/denoise, parity-verified SSIM 0.9995 vs the Python reference) | [MODEL_STATUS_VIDEO](docs/Checklists/MODEL_STATUS_VIDEO.md) |
| **3D** | TripoSR, Hunyuan3D-2 (image → mesh; glTF / OBJ / PLY export) | [MODEL_STATUS_3D](docs/Checklists/MODEL_STATUS_3D.md) |
| **World / interactive** | Oasis, DIAMOND (action-conditioned, real-time, loadable today); Matrix-Game 2.0 / 3.0 and Hunyuan-GameCraft are parity-verified but catalogued only — no `WorldService` loader yet, multi-checkpoint sets 9-51GB | [MODEL_STATUS_WORLD](docs/Checklists/MODEL_STATUS_WORLD.md) |

Index of all status docs: [`MODEL_STATUS.md`](docs/Checklists/MODEL_STATUS.md). Cross-modality real-weight parity authority: [`PARITY_VERIFICATION.md`](docs/Checklists/PARITY_VERIFICATION.md). Planned additions live in each status doc's **Remaining work** section; cross-cutting engineering work is in [`ROADMAP.md`](docs/Checklists/ROADMAP.md).

---

## Future Features

> [!WARNING]
> These are planned and **not yet implemented**. Tracking lives in [`ROADMAP.md`](docs/Checklists/ROADMAP.md) and each modality's [status doc](docs/Checklists/MODEL_STATUS.md) (Remaining work section).

- **Image:** LCM/Turbo distillation across more architectures; ControlNet tile / inpaint modes; union-type segment / tile / repaint control types (raw-map pass-through wired, dedicated preprocessing pending). *(Shipped 2026-07: IP-Adapter FaceID + FaceID-Plus/PlusV2, Flux-DiT ControlNet, union-type SDXL ControlNet, and the lineart / softedge / normal / segmentation preprocessors — see the image conditioning list above.)*
- **Vision:** YOLO-World, OWLv2, Florence-2, pose estimation, OCR, tracking. *(Shipped 2026-07: Grounding DINO open-vocab detection, RT-DETR, and Depth-Anything-V2 depth estimation, all real-weight verified end-to-end — see [MODEL_STATUS_VISION](docs/Checklists/MODEL_STATUS_VISION.md).)*
- **Video:** CogVideoX, longer-context temporal generation; **MiniMax-H3** (omni text/image/video/audio → 2K video with native stereo audio — announced 2026-07-31, weights promised but not yet published; the recipe seam and the researched capability contract are in [`MINIMAX_H3.md`](docs/Research/MINIMAX_H3.md)). *(Shipped 2026-07: HunyuanVideo 13B T2V, verified end-to-end through SwarmUI — see [Models](#models) above and [`VIDEO.md`](benchmarks/scoreboards/VIDEO.md).)*
- **3D:** texture synthesis, multi-view to mesh. *(Gaussian-splat output has an initial pipeline CLI-wired for TRELLIS, but is not yet parity-verified against the reference rasterizer — see [MODEL_STATUS_3D](docs/Checklists/MODEL_STATUS_3D.md).)*
- **World models:** broader action spaces, longer memory horizons, multiplayer state
- **Multi-GPU:** tensor parallel (NCCL) for datacenter NVLink boxes, expert parallel for MoE, >2-way DiT sharding, sequence parallel for video. *(Shipped 2026-08: the layer-split / DiT-shard / placement / CFG-parallel feature set — see [Multi-GPU & Sharding](#multi-gpu--sharding).)*
- **Tooling:** quantized inference (MXFP4 / MXFP8 / NVFP4), expanded CLI subcommands per modality

---

## Packages

| Package | Description |
|---|---|
| `HartsyInference.Core` | Tensor types, `IBackend`, schedulers, pipeline interfaces |
| `HartsyInference.ModelAssets` | Safetensors / GGUF / PyTorch loaders, quantization, LoRA, model registry, HuggingFace download |
| `HartsyInference.ModelAssets.Tokenizers` | CLIP, T5, Whisper, SentencePiece, Qwen, Llama tokenizers |
| `HartsyInference.Cpu` | CPU backend with AVX2 / AVX-512 / NEON SIMD kernels |
| `HartsyInference.Cuda` | CUDA backend with PTX kernels and cuBLAS |
| `HartsyInference.Vulkan` | Cross-vendor Vulkan backend (SPIR-V compute) |
| `HartsyInference.LLM` | Native LLM text generation (Qwen, Llama, Mistral, GGUF inference, chat templates) |
| `HartsyInference.Audio.Phonemizer` | Pure-C# grapheme-to-phoneme (espeak-ng port) for TTS front-ends |
| `HartsyInference.Diffusion` | Image pipelines (SD/SDXL/Flux/SD3/MMDiT/NextDiT), VAE, LoRA |
| `HartsyInference.Audio` | STT, TTS, music generation, and neural audio codecs |
| `HartsyInference.Vision` | CLIP/SigLIP/DINO embeddings, YOLO detection, SAM segmentation, face |
| `HartsyInference.Video` | LTX-Video, Wan, Lance, Kandinsky video generation |
| `HartsyInference.ThreeD` | Image/text → 3D mesh; glTF/OBJ/PLY export, marching cubes |
| `HartsyInference.World` | Action-conditioned world models, sessions, action encoders |
| `HartsyInference` | Meta-package: one reference that pulls in the core, all three backends, and every modality package including `HartsyInference.LLM` and `HartsyInference.Audio.Phonemizer` (`Server` and the sample `Cli` are excluded — run those from source) |
| `HartsyInference.Cli` | Command-line sample/validation tool (not published as a package) |
| `HartsyInference.API` | OpenAI-compatible HTTP API host — chat completions (continuous-batched, streaming, JSON-mode), image generation, model management. Runs from source (`IsPackable=false`), not published; see [How to Use It](#how-to-use-it). |

Each package is one folder under `src/`; the meta **HartsyInference** package pulls in the modality packages (add `HartsyInference.LLM` and `HartsyInference.Audio.Phonemizer` explicitly). GPU code stays behind `IBackend` in the backend packages — CPU-only packages never depend on CUDA/Vulkan.

---

## Requirements

- **.NET 8 or .NET 10** — the libraries target `net8.0` and `net10.0`; the sample Server and CLI target `net10.0`.

### CUDA backend (NVIDIA, fastest)
- **CUDA 13.x / 12.x** userspace libraries (cuBLAS, cuBLASLt)
- **NVIDIA GPU** with compute capability 8.0+ (RTX 30xx/40xx, A100, H100); fp8 tensor-core paths need 8.9+ (Ada)
- **cuDNN 9.21+** (optional) for fused flash attention — without it the engine falls back to materialized attention (slower, identical output).

### Vulkan backend (NVIDIA / AMD / Intel, cross-vendor)
- **Vulkan 1.3+ runtime**, almost always pre-installed by the GPU vendor driver
- **GPU with FP16 compute** (`shaderFloat16`). Most discrete GPUs from 2019+ qualify.

<details>
<summary>Vulkan setup details &amp; validation layers</summary>

- **Vulkan 1.3+ runtime**, almost always pre-installed by the GPU vendor driver
  - **Linux:** `sudo apt install mesa-vulkan-drivers vulkan-tools` (AMD/Intel; NVIDIA blob ships its own ICD)
  - **Windows:** the AMD / Intel / NVIDIA driver ships Vulkan; no extra install
- Validation layers (optional, for debugging): install the [LunarG Vulkan SDK](https://www.lunarg.com/vulkan-sdk/) and set `HARTSYINFERENCE_VK_VALIDATION=1`.
- See [ROADMAP.md](docs/Checklists/ROADMAP.md) for current model support and acceptance status.

</details>

### CPU backend (any platform)
- No GPU required. AVX2 / AVX-512 / NEON SIMD with scalar fallback. Slowest, but universally available.

---

## Documentation

<details>
<summary><b>Design documents</b></summary>

| Document | Description |
|---|---|
| [Benchmark Scoreboards](benchmarks/scoreboards/) | Per-modality HartsyInference-vs-baseline results tables (Image / Video / Audio / LLM / 3D) |
| [Multi-GPU Guide](docs/MULTI_GPU.md) | Sharding, component placement, CFG-parallel — settings, mechanics, verified models, limits |
| [Code Style](docs/CODE_STYLE.md) | Mandatory coding conventions (the standards single source of truth) |
| [Model Status](docs/Checklists/MODEL_STATUS.md) | Per-modality status + remaining work (index) |
| [Roadmap](docs/Checklists/ROADMAP.md) | Cross-cutting engineering roadmap (multi-GPU, kernel perf, quant, release) |
| [Parity Verification](docs/Checklists/PARITY_VERIFICATION.md) | Real-weight parity authority |
| [Troubleshooting](docs/Checklists/TROUBLESHOOTING.md) | Model bring-up debugging reference |

</details>

**Model status & parity:** the [Models](#models) section links the per-modality status docs (indexed in [`MODEL_STATUS.md`](docs/Checklists/MODEL_STATUS.md)); the cross-modality real-weight parity authority is [`PARITY_VERIFICATION.md`](docs/Checklists/PARITY_VERIFICATION.md).

**Research & Checklists:** technical research notes live in [`docs/Research/`](docs/Research/) (model formats, GPU/compute, diffusion architectures, text encoders, audio, vision). Phase-by-phase progress is tracked in [`docs/Checklists/`](docs/Checklists/). AI coding-agent instruction files are in [`docs/Agents/`](docs/Agents/); see [CLAUDE.md](CLAUDE.md) for the dispatcher.

---

## Project Structure

<details>
<summary><b>Repository layout</b></summary>

```
HartsyInference/
├── CLAUDE.md                  AI agent dispatcher
├── README.md                  ← You are here
├── src/                       Source code (one folder per NuGet package; GPU kernel sources in HartsyInference.Cuda/Kernels + HartsyInference.Vulkan/Shaders)
├── tests/                     Test projects
├── samples/                   Example applications
├── benchmarks/                Performance benchmarks
└── docs/
    ├── Research/              Technical research notes
    ├── Checklists/            Model status, roadmap & troubleshooting
    └── Agents/                AI agent instruction files
```

</details>

See the `src/` tree — one folder per NuGet package — for the complete layout.

---

## License

TBD

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon).
