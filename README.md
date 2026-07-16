# HartsyInference

**A pure C#/.NET AI inference engine for LLMs, image, audio, vision, video, 3D, and interactive world models — with zero Python dependencies.**

HartsyInference loads `.safetensors`, `.gguf`, and PyTorch `.pt`/`.ckpt` checkpoints directly and runs inference on **CUDA**, **Vulkan**, or **CPU** — no Python, no C++ wrappers, no external processes, just NuGet packages. One engine spans native LLM text generation, diffusion image models, speech/music, vision, video, 3D mesh, and real-time interactive worlds. Targets **.NET 8 and .NET 10**.

> [!IMPORTANT]
> **The recommended way to run HartsyInference is inside [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) via the [HartsyInference backend extension](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend)** — it registers HartsyInference as a pure-C# alternative to the ComfyUI backend, giving you a full generation UI, model management, LoRA, and video/audio output with no Python install. The engine is not building its own front-end; you can also embed it as [NuGet libraries](#quick-start-library), drive the [sample CLI](#quick-start-cli-developer-tool), or host the [OpenAI-compatible server](#how-to-use-it). For per-model status across every modality, see [Models](#models).

---

## Table of Contents

- [Why HartsyInference](#why-hartsyinference)
- [How to Use It (SwarmUI recommended)](#how-to-use-it)
- [Quick Start (Library)](#quick-start-library)
- [Quick Start (CLI, developer tool)](#quick-start-cli-developer-tool)
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
| **Production-grade** | Streaming progress, memory budgeting, VRAM monitoring, model hot-swap. |
| **SwarmUI-native** | Ships as a first-class [SwarmUI backend extension](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend), a pure-C# alternative to the ComfyUI backend. |

---

## How to Use It

HartsyInference does not ship its own front-end. There are a few ways to run it, in order of how most people should reach for them.

**1. SwarmUI + the HartsyInference backend extension (recommended).** Install the [SwarmUI-HartsyInference-Backend](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend) extension to register HartsyInference as a SwarmUI backend — a pure-C# alternative to the ComfyUI backend. You get SwarmUI's full generation UI, model browser, and parameter controls with **no Python environment**, plus per-architecture model loaders, video (Wan 2.x, LTX) with ffmpeg muxing, audio/music (ACE-Step), LoRA passthrough, live previews, and automatic checkpoint conversion — all on the same engine and kernels this repo builds, consumed as pinned `HartsyInference` NuGet packages. Clone it into your SwarmUI `src/Extensions/` folder, rebuild, and add a **HartsyInference** backend under Server → Backends; see that repo's README for the model-support matrix and setup.

**2. Library (NuGet).** Embed the engine directly in a .NET app; each modality is its own package. See [Quick Start (Library)](#quick-start-library).

**3. Sample CLI (developer tool).** The bundled [`hartsy` CLI](#quick-start-cli-developer-tool) drives every modality from the terminal — the fastest way to verify a checkpoint end-to-end. It's a development/validation tool, not the intended end-user surface.

**4. OpenAI-compatible HTTP server.** `HartsyInference.Server` hosts an OpenAI-shaped REST API: `/v1/chat/completions` (LLM/SSM chat — streaming, non-streaming, and JSON-mode — with **continuous batching** and a **paged KV cache**), `/v1/images/generations` (+ a streaming variant), and `/v1/models` load / list / pull / unload. Audio and image-edit routes are shaped for API completeness but return `501` until wired. It runs from source (`IsPackable=false`), CPU by default; set `HartsyInference:Backend=Cuda` + `HartsyInference:PtxDirectory` for GPU. See [`PRODUCTION_RELEASE_CRITERIA.md`](docs/Checklists/PRODUCTION_RELEASE_CRITERIA.md) for what's left before it's published.

```bash
dotnet run --project src/HartsyInference.Server -c Release --urls http://127.0.0.1:8080
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

Commands span every modality — `text`, `image`, `transcribe`, `speak`, `3d`, `vision`, `music`, `video`, `world` — plus catalog helpers `list`, `models`, and `pull`. Common flags: `-b|--backend cpu|cuda|vulkan`, `-m|--model <name>`, `--model-path <path>`. Run `hartsy <command> --help` for a command's full option set.

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
> 3D and world-model pipelines are built end-to-end and structurally complete; numerical validation against reference outputs is in progress. See [Models](#models) for per-model status.

---

## Benchmarks

We publish real numbers and we are honest about where we stand. HartsyInference is a young pure-C# engine: it is correct across a very wide model set, and on several flagship image models it is now **faster than ComfyUI on the same GPU** — while other models are still mid-optimization. We are closing the remaining gaps in the open. Every number below is reproducible from `benchmarks/` and the committed result files under [`benchmarks/results/`](benchmarks/results/); the exact methodology, required libraries, and the engine's default performance profile are specified in the **[Performance Guide](docs/PERFORMANCE.md)**.

### Image generation end-to-end vs ComfyUI

RTX 4090 24GB, full end-to-end wall-clock through the SwarmUI API — the identical 1024×1024 request routed to the ComfyUI backend, then the HartsyInference backend, on the same GPU. Warm median of 3 runs, randomized seeds, outputs visually verified coherent. Engine `1.0.0-alpha.46` + in-flight `44.x-local` optimization rounds, 2026-07-11 (living snapshot — updated as optimization rounds land):

| Model | HartsyInference | ComfyUI | Status |
|---|---:|---:|---|
| Flux.2 Klein 4B (4 steps) | 2.36 s | 1.85 s | 1.28× — Flux-family kit transplant done (was 3.45 s) |
| Z-Image-Turbo (8 steps) | **2.76 s** | 3.1 s | **Faster than ComfyUI** |
| SDXL (20 steps) | **2.93 s** | 3.7 s | **Faster than ComfyUI** (was 33.9 s) |
| Boogu-Turbo (4 steps) | 3.26 s | 2.54 s | 1.28× — optimization in progress |
| Flux-Schnell (4 steps) | 3.6 s | 3.04 s | 1.18× — Flux-family kit transplant done (was 10.5 s) |
| Krea2-Turbo (8 steps) | **4.52 s** | 6.5 s | **Faster than ComfyUI** |
| AuraFlow-0.3 (20 steps) | **13.93 s** | 14.0 s | **Tied with ComfyUI** (was 31.4 s) |
| Flux-Dev (20 steps) | 16.05 s | 12.5 s | 1.28× — Flux-family kit transplant done (was 72.4 s) |
| Ideogram4 (20 steps) | 19.5 s | 17.0 s | 1.15× — optimization queued |
| ERNIE-Image (20 steps) | **20.0 s** | 23.9 s | **Faster than ComfyUI** |
| Boogu-Base (20 steps) | 26.5 s | 17.8 s | 1.49× — optimization in progress |
| Chroma1-HD (20 steps) | 28.5 s | 16.6 s | Optimization in progress (1.7×, was 33×) |
| Qwen-Image (20 steps) | **39.4 s** | 54.8 s | **Faster than ComfyUI** |
| OmniGen 2 (20 steps) | 90.5 s | 13.0 s | 7.0× — first bench, pre-optimization |
| Qwen-Image-Edit 2511 (20 steps + reference) | 93 s | 87.8 s | 1.05× — full image editing with up to 3 vision-conditioned reference images |
| HiDream-i1 17B (25 steps, cfg 5) | 44.0 s | 35.2 s | 1.25× — first official row 2026-07-11 (was FAILED/OOM in the 07-05 run, then 24 min cold) |
| Flux.2 Dev 32B (20 steps, Q4 GGUF) | 52.6 s | — | First e2e 2026-07-10 — native GGUF quant + live Mistral-Small-3 encoder; multi-model switching validated 2026-07-11; no ComfyUI baseline |
| HunyuanImage 2.1 17B (2048², 20 steps, Q4 GGUF) | 74.1 s | — | Native-resolution 2048², new pixel-shuffle VAE decoder + prompt-embedding cache (was 77.0 s); no ComfyUI baseline |
| F-Lite 10B (30 steps) | 61.5 s | — | First correct run 2026-07-11 — validated block-by-block against the fal-ai reference; no ComfyUI baseline |
| Lumina-Image 2.0 (25 steps, cfg 4) | 17.7 s | — | 37× in one optimization round (was 650 s); no ComfyUI baseline |
| Chroma1-Radiance (pixel-space VAE-free, 20 steps) | 54.4 s | — | 14× in one optimization round (was 12.3 min); NeRF-head GPU port + prompt cache + resident DiT; no ComfyUI baseline |

These times require **zero configuration**: the engine's standard performance profile (cuDNN fused flash attention, fp8 tensor-core GEMM, F16 DiT activations, resident weights, warm activation pool) is default-on with per-feature kill-switches and graceful fallbacks — see the [Performance Guide](docs/PERFORMANCE.md).

### LLM decode vs llama.cpp

RTX 3060 12GB, CUDA, batch=1, 128-token greedy decode, warm, tokens/sec. Same GGUF file and quant on both sides. After a focused kernel optimization pass (fused quantized GEMV, quantized `lm_head`, split-K flash-decode attention, vectorized loads), engine decode went from **20-54× slower to 1.94-2.88× off llama.cpp**; **CUDA-graph decode** (2026-07-10, opt-in via `HARTSY_GRAPH_DECODE=1`) then collapsed the ~600-700 per-token kernel launches into one `cuGraphLaunch`/token for the plain dense GQA/RoPE decoder shape (Llama/Qwen2/Qwen3/Mistral), closing most of the remaining gap on launch-bound small models — **verified byte-identical greedy token output** with the graph on vs off:

| Model | Quant | llama.cpp t/s | Kernel-optimized t/s | + CUDA graph t/s | Gap (graph) |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | Q4_K_M | 337.6 | ~157 | **190.8** (2.57× faster than no-graph) | **1.77×** |
| Llama-3.2-1B | Q8_0 | 206.8 | ~111.5 | **120.2** (1.07×) | **1.72×** |
| Mistral-7B-v0.3 | Q4_K_M | 66.5 | ~30.7 | **32.0** (1.02×) | **2.08×** |
| Gemma-3-1B | Q4_K_M | 229.8 | ~79.7 | not eligible (sliding-window + softcap) | 2.88× |

CUDA graphs remove kernel-*launch* overhead, so the win scales inversely with model size: dramatic on small models (Qwen3), marginal on large ones already bound by GEMV memory bandwidth (Mistral — the engine reads ~22% of the bandwidth llama.cpp does there, a separate un-fixed bottleneck). Graph decode requires greedy sampling and the plain decoder shape — MoE, MLA, cross-attention, sliding-window, and SSM models fall through to the eager path unchanged — but it now correctly applies repetition penalty on-device rather than silently ignoring it (the only sampler stage that can change greedy's picked token; temperature/top-k/top-p/min-p are all argmax-preserving so they don't need a device kernel). Fused quantized GEMV now covers Q4_K/Q5_K/Q6_K/Q8_0/Q5_0/Q4_0 — every quant format in the default coverage matrix except the rarer K-quants (Q2_K/Q3_K) and IQ-formats, which still fall to a CPU dequant path. Prefill (prompt processing) is much faster and not the bottleneck. The server (`HartsyInference.Server`) additionally supports **continuous batching** — concurrently-submitted requests against the same model share decode rounds instead of running one at a time — and a **paged KV cache**, both independent of graph decode. Details: [`LLM_THROUGHPUT_BENCHMARK.md`](docs/Checklists/LLM_THROUGHPUT_BENCHMARK.md) + [`LLM_DECODE_PERF_GRIND.md`](docs/Checklists/LLM_DECODE_PERF_GRIND.md).

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

### Audio (TTS / STT), end-to-end

Real-time factor (RTF = generated-audio-seconds ÷ warm-gen-seconds; higher is faster) through the SwarmUI + AudioLab
generation path, on both GPUs. Full table + methodology + the outlier list: [`benchmarks/results/audio_tts_stt_2026-07-12.md`](benchmarks/results/audio_tts_stt_2026-07-12.md).

| Model | Type | RTX 3060 | RTX 4090 |
|---|---|---|---|
| Piper (VITS) | TTS | 8.6× | 8.3× |
| Moonshine | STT | 6.5× | 6.5× |
| Whisper (base) | STT | 5.1× | 5.4× |
| Kokoro (StyleTTS2) | TTS | 4.5× | 5.2× |
| MeloTTS (en-v3) | TTS | 1.4× | 1.4× |
| Kyutai TTS (DSM, tts-1.6b) | TTS | ~0.5× | — |
| F5-TTS (v1 base, voice clone) | TTS | ~0.4× | — |

Piper / Kokoro / MeloTTS / F5-TTS are verified word-correct via a proven STT oracle (whisper `medium.en`). F5-TTS
is zero-shot voice cloning (needs a reference clip + transcript); its RTF is below real-time by design (32
flow-matching DiT forwards), but a host grouped-conv bottleneck was removed for a **34× speedup (174.6 s → 6.4 s)**
at bit-parity. **Kyutai TTS** (Delayed-Streams Modeling, in-engine e2e word-correct 2026-07-16; measured through the
test path, not yet Swarm-deployed) generates 32 Mimi codebooks per frame through a weights-per-step depformer;
making those per-step projections device-resident cut the 3060 gen **6.5× (63.0 → 9.7 s)** — see
[`benchmarks/results/kyutai_tts_2026-07-16.md`](benchmarks/results/kyutai_tts_2026-07-16.md).

TTS measured through the canonical `GenerateText2Image` path (WAV to `/Output` like any gen); STT via `ProcessSTT`. **The small audio models are host/launch-bound, not compute-bound** — the 4090's extra compute buys ~nothing (Piper is even slightly slower on it). So the optimization lever here is **CUDA-graph capture / host-glue removal** (kills kernel-launch overhead), exactly like the LLM-decode graph win and the opposite of the compute-bound video/music DiTs where graphs are a no-op. Music (ACE-Step, YuE, HeartMuLa) e2e numbers live in their own result files and [`docs/Checklists/MODEL_STATUS_AUDIO.md`](docs/Checklists/MODEL_STATUS_AUDIO.md).

### 3D mesh generation (image → mesh), end-to-end vs the Python reference

Warm end-to-end seconds on the RTX 4090 (`examples/chair.png` → coherent `.glb`), against the upstream Python
reference on the same GPU. Full campaign write-up (Rounds 1–6, per-round diagnosis + parity gates):
[`docs/Checklists/THREED_GENPERF_PLAN.md`](docs/Checklists/THREED_GENPERF_PLAN.md).

| Model | HartsyInference | Python ref | Status |
|---|---|---|---|
| TripoSR (256³ density grid) | **2.1 s** | 0.58 s (neural) | 12.5× vs our own start (was 26.2 s); our GPU density decode now **beats** the reference — the gap is host marching-cubes + small-GEMM occupancy |
| Hunyuan3D-2 Shape (30 steps, grid 128, fp16) | **9.2 s** | 5.76 s | 1.6× — was 71.3 s (**7.75×** in-campaign); DiT per-forward (113 ms) now within 1.18× of the reference's 96 ms |

The Hunyuan3D wins were **bit-exact / coherence-gated**, not accuracy trades: a fused `Concat` kernel (a per-slice
`cuMemcpyDtoDAsync` loop was issuing ~280k memcpy nodes/forward → dit-loop 27.7 → 7.5 s, bit-identical mesh), the
DINOv2-giant conditioner's per-block host loops moved to device (cond 4.1 → 0.87 s), fused DiT adaLN + QKV-split-norm
kernels, a device FourierEmbed for the VAE, plus cuDNN fused SDPA, a DiT CUDA-graph, and F16 activations. Batched
CFG was measured and **ruled out** — the Concat fix already removed the per-forward overhead it would amortize
(graph-off ≈ graph-on), so the forward is now compute-bound. Both models produce coherent meshes verified end-to-end.

---

## Models

The engine covers a very wide model set across every modality. The **[benchmark tables](#benchmarks)** above show the representative models with measured numbers; the **authoritative per-model status** — verified end-to-end vs built-but-pending, with bring-up notes and real-weight parity evidence — lives in the modality status docs:

| Modality | Breadth (representative) | Status doc |
|---|---|---|
| **Language / LLM** | Llama, Qwen2/Qwen3, Gemma 2/3/4, Phi, Mistral (dense); Qwen3.5 gated-DeltaNet hybrid; MoE/MLA giants (Mixtral, Qwen3-MoE, DeepSeek-V3, Kimi-K2, GPT-OSS); VLMs; embeddings/rerankers; Mamba/RWKV/T5 — quantized GGUF throughout | [MODEL_STATUS_LLM](docs/Checklists/MODEL_STATUS_LLM.md) · [coverage](docs/Checklists/LLM_MODEL_COVERAGE.md) |
| **Image** | SD1.5 / SDXL (UNet); Flux.1/.2, SD3, Chroma / Radiance, Qwen-Image (+ Edit), HunyuanImage, HiDream, AuraFlow, Lumina 2, ERNIE-Image, Kandinsky 5, OmniGen 2, Ideogram 4 (DiT / MMDiT / NextDiT) | [MODEL_STATUS_IMAGE](docs/Checklists/MODEL_STATUS_IMAGE.md) |
| **Audio & Music** | Whisper / Moonshine (STT); Kokoro, Piper, StyleTTS2, Bark, Spark-TTS, CosyVoice, VibeVoice, MeloTTS, F5-TTS clone (TTS); ACE-Step, MusicGen, YuE, HeartMuLa (music); 9 neural codecs | [MODEL_STATUS_AUDIO](docs/Checklists/MODEL_STATUS_AUDIO.md) |
| **Vision** | CLIP / SigLIP / DINOv2-3 embeddings; YOLO8 / YOLO11 detection; SAM / SAM 2 / 2.1 segmentation; face detection | [MODEL_STATUS_VISION](docs/Checklists/MODEL_STATUS_VISION.md) |
| **Video** | LTX-Video, Wan 2.x (T2V + I2V), Lance, Kandinsky 5 Video | [MODEL_STATUS_VIDEO](docs/Checklists/MODEL_STATUS_VIDEO.md) |
| **3D** | TripoSR, Hunyuan3D-2 (image → mesh; glTF / OBJ / PLY export) | [MODEL_STATUS_3D](docs/Checklists/MODEL_STATUS_3D.md) |
| **World / interactive** | Hunyuan-GameCraft, Matrix-Game 2.0 / 3.0, Oasis (action-conditioned, real-time) | [MODEL_STATUS_WORLD](docs/Checklists/MODEL_STATUS_WORLD.md) |

Index of all status docs: [`MODEL_STATUS.md`](docs/Checklists/MODEL_STATUS.md). Cross-modality real-weight parity authority: [`PARITY_VERIFICATION.md`](docs/Checklists/PARITY_VERIFICATION.md). Planned additions: [Model Support Roadmap](docs/Design/MODEL_SUPPORT_ROADMAP.md).

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
| `HartsyInference` | Meta-package: one reference that pulls in the core, all three backends, and every modality package including `HartsyInference.LLM` and `HartsyInference.Phonemizer` (`Server` and the sample `Cli` are excluded — run those from source) |
| `HartsyInference.Cli` | Command-line sample/validation tool (not published as a package) |
| `HartsyInference.Server` | OpenAI-compatible HTTP API host — chat completions (continuous-batched, streaming, JSON-mode), image generation, model management. Runs from source (`IsPackable=false`), not published; see [How to Use It](#how-to-use-it). |

See [NuGet Package Design](docs/Design/NUGET_PACKAGE_DESIGN.md) for the dependency graph and minimum install examples.

---

## Requirements

- **.NET 8 or .NET 10** — the libraries target `net8.0` and `net10.0`; the sample Server and CLI target `net10.0`.

### CUDA backend (NVIDIA, fastest)
- **CUDA 13.x / 12.x** userspace libraries (cuBLAS, cuBLASLt)
- **NVIDIA GPU** with compute capability 8.0+ (RTX 30xx/40xx, A100, H100); fp8 tensor-core paths need 8.9+ (Ada)
- **cuDNN 9.21+** (optional) for fused flash attention — without it the engine falls back to materialized attention (slower, identical output). Library locations and resolution order: [Performance Guide §3](docs/PERFORMANCE.md#3-native-library-requirements-and-resolution)

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
| [Performance Guide](docs/PERFORMANCE.md) | The default performance profile, native library requirements, benchmark methodology |
| [Core Design](docs/Design/CORE_DESIGN.md) | Architecture, design pillars, key decisions, goals/non-goals, per-modality capabilities |
| [Model Support Roadmap](docs/Design/MODEL_SUPPORT_ROADMAP.md) | Full model support plan |
| [NuGet Package Design](docs/Design/NUGET_PACKAGE_DESIGN.md) | Package breakdown, dependencies, install examples |
| [File Structure](docs/Design/FILE_STRUCTURE.md) | Full project layout |
| [Implementation Details](docs/Design/IMPLEMENTATION_DETAILS.md) | Per-component technical approach |
| [Build Order](docs/Design/BUILD_ORDER.md) | Phase-by-phase implementation sequence |
| [Validation Strategy](docs/Design/VALIDATION_STRATEGY.md) | Reference implementations and tolerances |

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
