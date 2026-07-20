# API Agent

> Own the engine's **public library API surface**: the types and methods that external consumers call. The primary consumer is the **SwarmUI + HartsyInference backend extension**; secondary consumers are direct NuGet-library users and the bundled sample CLIs.

> **The OpenAI-compatible REST server is dropped.** HartsyInference ships no first-party server or GUI. `src/HartsyInference.API/` remains as abandoned scaffolding; do not extend it, advertise it, or route new work into it. If a task asks for "server endpoints", redirect it to library-API work or the SwarmUI extension.

## How the Engine Is Consumed

1. **SwarmUI backend extension (primary).** Repo: `https://github.com/HartsyAI/SwarmUI-HartsyInference-Backend`. It registers HartsyInference as an alternative to the ComfyUI backend inside SwarmUI, loads models, and drives the engine's pipelines. When you change or add a public API, this extension is the caller you must not break. Keep signatures stable, additive, and discoverable.
2. **NuGet libraries (secondary).** Consumers add the meta package `HartsyInference` (pulls in Core, Cpu, Cuda, Vulkan, ModelHandler, Tokenizers, Diffusion, Audio, Vision, Video, Interactive, ThreeD) plus `HartsyInference.LLM` and `HartsyInference.Audio.Phonemizer` explicitly when needed.
3. **Sample CLIs (dev/verification).** `src/HartsyInference.Cli` and the projects under `samples/`. These are thin drivers over the same public API and double as usage examples; keep them compiling when the API moves.

## Extra Reading
- `docs/Design/FILE_STRUCTURE.md`, `docs/Design/NUGET_PACKAGE_DESIGN.md`
- `docs/Research/DOTLLM_ARCHITECTURE.md` (historical study that informed the engine's patterns; not a live dependency)
- Existing public entry points: `src/HartsyInference.Diffusion/Pipelines/PipelineFactory.cs`, the `*Pipeline` classes, `src/HartsyInference.LLM/Generation/TextGenerationPipeline.cs`, `src/HartsyInference.ModelAssets/Registry/ModelRegistry.cs`
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
4. Wire progress via the existing callback contract (`Action<GenerationProgress>?` for diffusion, `Action<int>? onToken` for LLM). Do not add SSE or HTTP.
5. Update `src/HartsyInference.Cli` / `samples/` to exercise the new surface, and note the change for the SwarmUI extension maintainer.
6. Update the relevant checklist.

## Design Principles
- **Stability first.** The SwarmUI extension pins a published engine version; a broken or renamed public API is invisible to it until republished and re-pinned. Treat public signatures as a contract.
- **Backend-agnostic.** Public entry points take `IBackend`; never leak `CudaBackend`/`CpuBackend` specifics into signatures.
- **Callbacks, not transports.** Progress and token streaming are in-process callbacks. No HTTP, no OpenAI schemas, no `AddHartsyInference`/`MapHartsyInferenceEndpoints`.
- **Minimal, discoverable API.** XML-doc every public type and method; keep the surface small and cohesive so the extension author can find the right call.

## Security Checklist (library-side)
- [ ] Validate model paths (no path traversal) when loading checkpoints
- [ ] Validate input sizes / shapes at public entry points before handing tensors to the backend
- [ ] No secrets in exception messages or logs
