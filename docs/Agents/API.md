# API Agent

> Own the engine's **public library API surface**: the types and methods that external consumers call. The primary consumer is the **SwarmUI + HartsyInference backend extension**; secondary consumers are direct NuGet-library users, the bundled sample CLIs, and `src/HartsyInference.API` (the HTTP adapter — see below).

> **`HartsyInference.API` is a live, supported thin HTTP adapter over `HartsyInference.Engine`** — not dropped, not abandoned scaffolding. It was rewired onto `IInferenceEngine` (previously it called a legacy pre-facade `ModelManager` that was hard-cast to SDXL-only). SwarmUI remains the *recommended* surface for end users; the HTTP API is for scripting/automation/non-.NET clients that want the engine over HTTP instead. See "The HTTP API" below for its endpoint catalog.

## How the Engine Is Consumed

1. **SwarmUI backend extension (primary).** Repo: `https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend`. It registers HartsyInference as an alternative to the ComfyUI backend inside SwarmUI, loads models, and drives the engine's pipelines. When you change or add a public API, this extension is the caller you must not break. Keep signatures stable, additive, and discoverable.
2. **NuGet libraries (secondary).** Consumers add the meta package `HartsyInference` (pulls in Core, Cpu, Cuda, Vulkan, ModelHandler, Tokenizers, Diffusion, Audio, Vision, Video, Interactive, ThreeD) plus `HartsyInference.LLM` and `HartsyInference.Audio.Phonemizer` explicitly when needed.
3. **HTTP API (secondary).** `src/HartsyInference.API` — an ASP.NET Core Minimal API process that constructs one long-lived `InferenceEngine` and maps every call through `HartsyInference.Engine.InferenceQueue` (the facade is not safely re-entrant per backend). See "The HTTP API" below.
4. **Sample CLIs (dev/verification).** `src/HartsyInference.Cli` and the projects under `samples/`. These are thin drivers over the same public API and double as usage examples; keep them compiling when the API moves.

## The HTTP API (`src/HartsyInference.API`)

Thin wrapper — every route resolves a `ModelSpec` via `HartsyInference.Engine.Registry.ModelResolver`/`ModelCatalog` (relocated from the CLI so both share one catalog) and calls the matching `IInferenceEngine` service. Route groups live under `src/HartsyInference.API/Endpoints/`, one file per modality/concern.

- **Health/settings/admin** — `GET /health`, `/ready`, `/version`, `/settings`; `GET/POST/DELETE /admin/{catalog,models,cache,queue,memory/free,backend}`.
- **Native routes** (the primary contract — pass the Engine's native request/result records through almost verbatim, as an envelope `{model, modelPath, request}`): `/v1/native/images`(+`/stream`), `/v1/native/text`(+`/stream`, `/count-tokens`), `/v1/native/{speech,transcribe,voice-convert,fx/separate,fx/enhance}`, `/v1/native/vision`, `/v1/native/mesh`(+`/stream`), `/v1/native/video/stream`, `/v1/native/world/sessions` (open/action/stream/close — a stateful session, tracked by `WorldSessionRegistry`).
- **OpenAI-compat routes** (secondary, deliberately narrow — chat + images only, since LoRA/ControlNet/regional-prompting/tool-calling don't fit that schema): `/v1/chat/completions`, `/v1/images/generations`(+`/stream`). These call the *same* handler methods as the native routes, not a parallel implementation.
- **Concurrency**: two `InferenceQueue` gates — the default (unkeyed) one for fast modalities, a keyed `"long-running"` one (`[FromKeyedServices(QueueKeys.LongRunning)]`) for video generation and opening world sessions, so one multi-minute video job can't starve every fast request behind it.

Full endpoint-by-endpoint rationale and the phased build order live in the (now-implemented) plan this API was built from — ask the session history or see the git log for `src/HartsyInference.API/` if you need the reasoning behind a specific route's shape.

## Extra Reading
- `docs/Design/FILE_STRUCTURE.md`, `docs/Design/NUGET_PACKAGE_DESIGN.md`
- `docs/Research/DOTLLM_ARCHITECTURE.md` (historical study that informed the engine's patterns; not a live dependency)
- Existing public entry points: `src/HartsyInference.Diffusion/Pipelines/PipelineFactory.cs`, the `*Pipeline` classes, `src/HartsyInference.LLM/Generation/TextGenerationPipeline.cs`, `src/HartsyInference.ModelAssets/Registry/ModelRegistry.cs`, `src/HartsyInference.Engine/InferenceEngine.cs` (the facade every consumer, including the HTTP API, goes through)
- Backends: `src/HartsyInference.Cpu/CpuBackend.cs`, `src/HartsyInference.Cuda/CudaBackend.cs`

## Public API Surface (verify against `src/` before extending)

**Backends.** Everything runs against `IBackend`. Consumers construct a concrete backend (`CpuBackend`, `CudaBackend`) and pass it into loaders and pipelines. Model code never calls a backend by concrete type.

**Model loading / registry**
- `ModelRegistry` (in `HartsyInference.ModelAssets`): `Task<LoadedModel> LoadAsync(...)`, `LoadedModel? Get(string modelId)`, `bool Unload(string modelId)`, `bool IsLoaded(...)`, `LoadedModelIds`. This is how the SwarmUI extension tracks resident models and reclaims VRAM.
- `PipelineFactory` (in `HartsyInference.Diffusion.Pipelines`): `ModelArchitecture DetectArchitecture(string path)`, `DiffusionPipelineBase LoadAuto(string path, IBackend backend)`, plus per-arch loaders (e.g. `LoadSdxl`).

**Diffusion pipelines.** These inherit `DiffusionPipelineBase`. Public shape is synchronous methods returning `(byte[] rgbData, int width, int height, int seed)` and taking an `Action<GenerationProgress>?` progress callback: `GenerateFromTokens` / `GenerateFromEmbeddings` / `InpaintFromTokens` / `RefineFromTokens`. There is no `IDiffusionPipeline` interface and no `IAsyncEnumerable` streaming contract (the old one was deleted because no pipeline implemented it). Pipelines do NOT own their components (text encoders, transformer/UNet, VAE); the caller does.

**LLM text generation** (`HartsyInference.LLM`)
- `GenericTransformer` (config-driven decoder: Qwen2/Qwen3/Llama/Mistral) plus `GgufLanguageModel` for GGUF-quantized weights.
- `TextGenerationPipeline(GenericTransformer model, ILlmTokenizer tokenizer, IBackend backend, ...)` with `GenerationResult Generate(GenerationRequest request, Action<int>? onToken = null)`. Device-resident KV cache, sampler chain, and chat templates live under `Sampling/`, `Transformer/`, and `ChatTemplates/`.

## Workflow
1. Identify the consumer (SwarmUI extension, library user, or CLI) and the exact call site.
2. Design the smallest additive change to the public type/method; prefer new overloads over breaking signatures.
3. Implement: consumer request → backend + loader → pipeline call → return the result tuple / `GenerationResult`.
4. Wire progress via the existing callback contract (`Action<GenerationProgress>?` for diffusion, `Action<int>? onToken` for LLM) for in-process library consumers. `src/HartsyInference.API` is the one place SSE/HTTP is appropriate — it wraps `IProgress<StepPreview>`/`IAsyncEnumerable<T>` from the Engine facade, it doesn't reinvent progress plumbing.
5. Update `src/HartsyInference.Cli` / `samples/` to exercise the new surface, and note the change for the SwarmUI extension maintainer.
6. Update the relevant checklist.

## Design Principles
- **Stability first.** The SwarmUI extension pins a published engine version; a broken or renamed public API is invisible to it until republished and re-pinned. Treat public signatures as a contract.
- **Backend-agnostic.** Public entry points take `IBackend`; never leak `CudaBackend`/`CpuBackend` specifics into signatures.
- **Callbacks, not transports, for the library API itself.** The public library surface (this section) stays in-process callbacks — `Action<GenerationProgress>?`, `Action<int>? onToken` — not HTTP/SSE. `src/HartsyInference.API`'s `AddHartsyInference`/`MapHartsyInferenceEndpoints` are the one sanctioned transport layer on top, and they call this same library surface (via `IInferenceEngine`) rather than duplicating it.
- **Minimal, discoverable API.** XML-doc every public type and method; keep the surface small and cohesive so the extension author can find the right call.

## Security Checklist (library-side)
- [ ] Validate model paths (no path traversal) when loading checkpoints
- [ ] Validate input sizes / shapes at public entry points before handing tensors to the backend
- [ ] No secrets in exception messages or logs
