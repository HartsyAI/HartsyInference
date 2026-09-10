# HartsyInference

C#/.NET inference libraries for text, images, video, speech, music, vision, 3D, and world models.
Load safetensors, GGUF, and supported PyTorch checkpoints with CUDA, Vulkan, or CPU backends.
Inference does not require a Python runtime; reference-generation tooling may use Python.

**Alpha:** pin package versions. Model, checkpoint-format, feature, and backend coverage differ;
[model status](docs/Checklists/MODEL_STATUS.md) records verified paths and known gaps.

## Get started

For a UI, use the [SwarmUI backend extension](https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend).
For a .NET application, install the meta-package (or select individual packages):

```bash
dotnet add package HartsyInference --prerelease
```

InferenceEngine owns loading, caches, placement, and generation:

```csharp
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

using InferenceEngine engine = new("cuda");
ImageResult image = await engine.Images.GenerateAsync(
    new ModelSpec { Requested = "sdxl", Modality = Modality.Image, LocalPath = "sdxl.safetensors" },
    new ImageRequest { Prompt = "a castle on a mountain at sunset", Steps = 25 });
// image.Rgb contains image.Width * image.Height * 3 RGB24 bytes.
```

Services include Images, Text, Video, Speech, Transcribe, Music, Vision, Mesh, World, Restore,
VoiceConversion, Fx, and Embeddings. Inspect the service interfaces for their request/result contracts.

## CLI and HTTP

```bash
dotnet run -c Release -f net10.0 --project src/HartsyInference.Cli -- \
  image "a castle on a mountain at sunset" --model-path sdxl.safetensors -b cuda
dotnet run -c Release -f net10.0 --project src/HartsyInference.API
```

The CLI provides catalog/download tools and modality commands; use --help for current options.
The HTTP application wraps Engine with native routes and OpenAI-compatible routes. See
[endpoint implementations](src/HartsyInference.API/Endpoints/),
[server options](src/HartsyInference.API/HartsyInferenceServerOptions.cs), and [deployment](deploy/README.md).
The HTTP text path does not currently expose the LLM package's continuous-batching scheduler.

Video supports preflight at POST /v1/native/video/plan and SSE generation at POST /v1/native/video/stream.
Plans govern compatibility, artifacts, defaults, and release gates before loading weights. H3 expansion
features remain subject to their individual [release gates](docs/Checklists/MODEL_STATUS_VIDEO.md#minimax-h3).

## Requirements and backends

Libraries target .NET 8 and .NET 10; building/testing uses the .NET 10 SDK.
CUDA requires an NVIDIA GPU supported by the shipped PTX (baseline sm_80), a compatible driver, and
required CUDA userspace libraries. Optional cuDNN paths have additional library requirements.
Vulkan requires a compatible runtime and the features queried by the backend; AMD/Intel hardware
validation remains open. CPU has no GPU dependency.

C# model code routes math through IBackend. CUDA uses disk-loaded PTX, Vulkan uses SPIR-V, and CPU uses
SIMD with scalar fallbacks. CUDA kernels include compiled CUDA sources and legacy handwritten PTX.
Vendor-library P/Invoke is used in GPU backends; this is not a ban on native driver/math libraries.

## Community benchmarks

[![Reviewed community benchmark results](https://hartsyai.github.io/HartsyInference/overview.svg)](https://hartsyai.github.io/HartsyInference/)

Run the same frozen workloads on your GPU and submit the complete evidence by PR.
The [benchmark guide](benchmarks/README.md) covers setup, methodology and review;
the [explorer](https://hartsyai.github.io/HartsyInference/) compares accepted runs on matching workloads.
No historical scores are presented as verified community results. The explorer requires Pages activation;
see the [initial dataset snapshot](benchmarks/generated/overview.svg) until it is available.

## Documentation

- [Documentation map](docs/README.md), [contributing](CONTRIBUTING.md), [agent instructions](AGENTS.md).
- [Model support](docs/Checklists/MODEL_STATUS.md) and [numerical evidence](docs/Checklists/PARITY_VERIFICATION.md).
- [Open work](docs/Checklists/ROADMAP.md), [troubleshooting](docs/Checklists/TROUBLESHOOTING.md).
- [Historical performance scoreboards](benchmarks/scoreboards/) — hardware, settings, dates, and external baselines.
- [Multi-GPU configuration](docs/MULTI_GPU.md) and [environment controls](docs/ENV_VARS.md).

Code and packages are [MIT licensed](LICENSE). Model weights retain their publishers' licenses;
consult the checkpoint's license, including the [MiniMax-H3 license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
before use or redistribution.
