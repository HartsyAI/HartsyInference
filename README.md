# HartsyInference

C#/.NET inference libraries for text, images, video, speech, music, vision, 3D, and world models.
Load safetensors, GGUF, and supported PyTorch checkpoints with CUDA, Vulkan, or CPU backends.
Inference does not require a Python runtime; reference-generation tooling may use Python.

**Alpha: pin an exact package version.** Model, checkpoint-format, feature, and backend coverage differ, and
implementation does not imply real-weight verification —
[model status](https://github.com/HartsyAI/HartsyInference/blob/main/docs/Checklists/MODEL_STATUS.md)
records verified paths and known gaps.

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

Transcribe handles audio longer than one model window. Whisper, distil-Whisper, whisper-streaming, Moonshine and Kyutai
split the clip at silences into windows, decode each one, and join the text. Timestamps are offset onto the clip
timeline. `TranscriptResult` reports `Duration` and `WindowCount`. For these windowed models, clips longer than
`vram.transcribeMaxSeconds` (default 7200) are refused with a named error. SheetSage2 windows internally and has no
such cap.

## CLI and HTTP

```bash
dotnet run -c Release -f net10.0 --project src/HartsyInference.Cli -- \
  image "a castle on a mountain at sunset" --model-path sdxl.safetensors -b cuda
dotnet run -c Release -f net10.0 --project src/HartsyInference.API
```

The CLI provides catalog/download tools and modality commands; use --help for current options.
The HTTP application wraps Engine with native routes and OpenAI-compatible routes. See
[endpoint implementations](https://github.com/HartsyAI/HartsyInference/tree/main/src/HartsyInference.API/Endpoints),
[server options](https://github.com/HartsyAI/HartsyInference/blob/main/src/HartsyInference.API/HartsyInferenceServerOptions.cs),
and [deployment](https://github.com/HartsyAI/HartsyInference/blob/main/deploy/README.md).
The HTTP text path does not currently expose the LLM package's continuous-batching scheduler.

Image, video and music families take their defaults from a recipe: a request field left unset takes the recipe's
value where the recipe declares one, and a recipe value of null leaves that choice to the pipeline or loader. Image and video results carry provenance (`ParameterProvenance`) for the settings planning resolves: each
names its source, such as the recipe, the caller, or a normalization of the caller's value. GET /v1/native/recipes lists the image and video
recipes, and POST /v1/native/recipes/describe answers for one model without loading weights; text, speech and
transcription defaults still live in their loaders. See the
[recipe contract](https://github.com/HartsyAI/HartsyInference/blob/main/docs/Research/RECIPE_CONTRACT.md).

Video supports preflight at POST /v1/native/video/plan and SSE generation at POST /v1/native/video/stream.
Plans govern compatibility, artifacts, defaults, and release gates before loading weights. H3 expansion
features remain subject to their individual
[release gates](https://github.com/HartsyAI/HartsyInference/blob/main/docs/Checklists/MODEL_STATUS_VIDEO.md#minimax-h3).

## HairDeck: hair cards from one image

HairDeck (`hairdeck`) turns one character image into a GLB of hair cards, with per-card colours and optional spring-bone
chains. It runs through `hartsy 3d` and `POST /v1/native/mesh` on the CUDA and CPU backends. Vulkan is rejected for HairDeck
(its BiRefNet-matting preprocessing needs `LeakyRelu`, which the Vulkan backend does not implement). Weights are from
[`hyz317/HairDeck`](https://huggingface.co/hyz317/HairDeck) (Apache-2.0). Output is static (hair cards and chain tubes) or
rigged (the cards skinned to the spring chains, with head-motion clips). VRM packaging is not implemented.

### Setup

1. Pull the catalog assets. This fetches the HairDeck checkpoints (about 1.6 GB) and the Vision models it needs:

   ```bash
   dotnet run -c Release -f net10.0 --project src/HartsyInference.Cli -- pull hairdeck
   ```

   Assets are placed under `<paths.modelsRoot>/3D/HairDeck` and `<paths.modelsRoot>/Vision`. The default root is `Models/`
   in the repository. Change it with `hartsy settings set paths.modelsRoot <dir>`.
2. Two of the Vision models are gated on Hugging Face. Request access on
   [`facebook/sam3`](https://huggingface.co/facebook/sam3) (SAM License) and
   [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
   (DINOv3 License) first. Then set `HF_TOKEN` to a token with read access before `pull`. BiRefNet-matting is MIT and
   needs no approval.

### Input

- A PNG with an alpha channel uses that alpha as the person matte. Any other image (including a PNG without alpha) is
  segmented automatically with SAM 3 and BiRefNet-matting.
- One character per image. A request fails with a named error when the hair or face cannot be found.

### CLI

```bash
dotnet run -c Release -f net10.0 --project src/HartsyInference.Cli -- \
  3d character.png -m hairdeck -b cuda --seed 0 -o out/
```

Writes `out/hairdeck-0001.glb`. Options that apply to HairDeck:

| Option | Default | Effect |
|---|---|---|
| `--steps` | 50 | Euler steps of the image model |
| `--seed` | -1 (random) | RNG seed; use a fixed value to repeat a result |
| `--no-colors` | off | Skip the colour model; the cards then have no per-card colour |
| `--no-chains` | off | Skip the spring-bone chains; the GLB has only `hair_cards` |
| `-b` | `auto` | `cuda` or `cpu`. `vulkan` is rejected; `auto` may pick Vulkan on some GPUs, so pass `-b cuda` or `-b cpu` |
| `--rigged` | off | Skin the cards to the spring chains and add four clips: `turn`, `nod`, `jump`, `sway`. Needs chains |

Samples are fixed at one per request (`-n 1`). Rigged output is written to the same `hairdeck-0001.glb` name.

### HTTP

`POST /v1/native/mesh` takes the image as a base64 PNG in `imagePng` (a `data:image/png;base64,` prefix is accepted).
Set either `imagePng` or `request.image`, not both.

```bash
curl -X POST http://localhost:5000/v1/native/mesh -H "Content-Type: application/json" -d '{
  "model": "hairdeck",
  "imagePng": "<base64 PNG>",
  "request": { "prompt": "", "seed": 0, "steps": 50, "colors": true, "chains": true, "rigged": false }
}'
```

The response carries `data` (the GLB, base64), `format` (`glb`) and `savedPath`. Add `"save": false` to return the GLB
without writing it to disk. `POST /v1/native/mesh/stream` returns only the final result for HairDeck; step progress is not streamed yet. Use your server's address; the example assumes the default local port, 5000.

### Output

- `hair_cards`: the hair as ribbons, with per-card colours (`COLOR_0`) when colours are on, ready for a viewer or DCC tool.
- `spring_chains` (when chains are on): the chain tubes. Static output shows where the chains run. Rigged output skins
  the hair cards and the chain tubes to the chains, and the file has one skin and four animations (`turn`, `nod`, `jump`, `sway`) at 60 fps.

### Limits

- Verified on one demo image (`ex01`, seed 0) on CUDA (RTX 3060): the CLI and the API each wrote a 1,099,912-byte static
  GLB, and each wrote a 2,439,976-byte rigged GLB. All four clips move the hair (peak vertex displacement 0.19–0.30 m).
  Other demo images are not run.
- Per-step time and peak VRAM are not measured yet.
- Cancellation stops between sampling steps. It does not stop SAM 3, BiRefNet or DINOv3 in the middle of preprocessing.
- One generation runs at a time per loaded model. A request queued behind an unload gets HTTP 503 and can be retried.
- The cuDNN fast path is not used; convolutions use the cuBLAS fallback.

## Configuration

Settings live in one file, `~/.config/hartsyinference/settings.json`, and are read and written through
`hartsy settings list | get | set`. The engine reads no environment variables; see
[settings](https://github.com/HartsyAI/HartsyInference/blob/main/docs/SETTINGS.md) for the knobs and their scopes.

## Requirements and backends

Libraries target .NET 8 and .NET 10; building/testing uses the .NET 10 SDK.
CUDA requires an NVIDIA GPU supported by the shipped PTX (baseline sm_80, plus per-architecture variants
where a kernel ships one), a compatible driver, and
required CUDA userspace libraries. Turing cards (sm_75, such as the RTX 2060 SUPER) cannot run the shipped
baseline: several kernels use `cp.async` or sm_80 bf16 instructions. The CUDA backend still constructs there with
only the kernels built for that architecture, and `auto` selects Vulkan; use an Ampere-or-newer GPU for CUDA runs.
Optional cuDNN paths have additional library requirements.
Vulkan requires a compatible runtime and the features queried by the backend; AMD/Intel hardware
validation remains open. CPU has no GPU dependency.

C# model code routes math through IBackend. CUDA uses disk-loaded PTX, Vulkan uses SPIR-V, and CPU uses
SIMD with scalar fallbacks. CUDA kernels include compiled CUDA sources and legacy handwritten PTX.
Vendor-library P/Invoke is used in GPU backends; this is not a ban on native driver/math libraries.

## Community benchmarks

[![Reviewed community benchmark results](https://raw.githubusercontent.com/HartsyAI/HartsyInference/main/benchmarks/generated/overview.svg)](https://github.com/HartsyAI/HartsyInference/blob/main/benchmarks/README.md)

Run the same frozen workloads on your GPU and submit the complete evidence by PR.
The [benchmark guide](https://github.com/HartsyAI/HartsyInference/blob/main/benchmarks/README.md) covers setup,
methodology and review; the [explorer](https://hartsyai.github.io/HartsyInference/) compares accepted runs on
matching workloads. No historical scores are presented as verified community results. The image above is the
checked-in dataset snapshot; it is replaced by the live explorer once Pages is activated for this repository.

## Documentation

- [Documentation map](https://github.com/HartsyAI/HartsyInference/blob/main/docs/README.md),
  [contributing](https://github.com/HartsyAI/HartsyInference/blob/main/CONTRIBUTING.md),
  [agent instructions](https://github.com/HartsyAI/HartsyInference/blob/main/AGENTS.md).
- [Model support](https://github.com/HartsyAI/HartsyInference/blob/main/docs/Checklists/MODEL_STATUS.md) and
  [numerical evidence](https://github.com/HartsyAI/HartsyInference/blob/main/docs/Checklists/PARITY_VERIFICATION.md).
- [Open work](https://github.com/HartsyAI/HartsyInference/blob/main/docs/Checklists/ROADMAP.md),
  [troubleshooting](https://github.com/HartsyAI/HartsyInference/blob/main/docs/Checklists/TROUBLESHOOTING.md).
- [Historical performance scoreboards](https://github.com/HartsyAI/HartsyInference/tree/main/benchmarks/scoreboards)
  — hardware, settings, dates, and external baselines.
- [Multi-GPU configuration](https://github.com/HartsyAI/HartsyInference/blob/main/docs/MULTI_GPU.md),
  [settings](https://github.com/HartsyAI/HartsyInference/blob/main/docs/SETTINGS.md) and
  [embedding the engine in a host application](https://github.com/HartsyAI/HartsyInference/blob/main/docs/HOST_INTEGRATION.md).

Code and packages are [MIT licensed](https://github.com/HartsyAI/HartsyInference/blob/main/LICENSE).
Model weights retain their publishers' licenses; consult the checkpoint's license, including the
[MiniMax-H3 license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
before use or redistribution.
