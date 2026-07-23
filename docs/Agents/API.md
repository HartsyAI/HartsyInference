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

- **Health/settings/admin** — `GET /health`, `/ready`, `/version`, `/settings`; `GET/POST/DELETE /admin/{catalog,models,cache,queue,usage,memory/free,backend}`.
- **Native routes** (the primary contract — pass the Engine's native request/result records through almost verbatim, as an envelope `{model, modelPath, request}`): `/v1/native/images`(+`/stream`), `/v1/native/text`(+`/stream`, `/count-tokens`), `/v1/native/{speech,transcribe,voice-convert,fx/separate,fx/enhance,embeddings}`, `/v1/native/vision`, `/v1/native/mesh`(+`/stream`), `/v1/native/video/stream`, `/v1/native/world/sessions` (open/action/stream/close — a stateful session, tracked by `WorldSessionRegistry`).
- **OpenAI-compat routes** — `/v1/chat/completions`(+tool/function calling), `/v1/models`(+`/{model}`), `/v1/embeddings`, `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/images/generations`(+`/stream`). All in `CompatEndpoints.cs`; these call the *same* handler methods/services as the native routes, not a parallel implementation. Still deliberately narrower than the native contract — composition-heavy image requests (LoRA/ControlNet/regional prompting) don't fit OpenAI's schema and stay native-route-only. See "OpenAI compat details" below for the specific scoping calls (response-format limits, embeddings model scope, tool-calling mapping).
- **Concurrency**: two `InferenceQueue` gates — the default (unkeyed) one for fast modalities, a keyed `"long-running"` one (`[FromKeyedServices(QueueKeys.LongRunning)]`) for video generation and opening world sessions, so one multi-minute video job can't starve every fast request behind it.

### OpenAI compat details

- **`GET /v1/models`(+`/{model}`)** — wired to `ModelCatalog.All`/`Find`. `created` reports the server process's start time for every entry (OpenAI's schema wants a timestamp; this catalog doesn't track real per-model dates) — cosmetic, not a real per-model claim.
- **Tool/function calling** (`/v1/chat/completions`) — `tools`/`tool_choice` map onto the native `TextRequest.Tools`/`ForceToolId` (a fully-built native capability the compat DTOs previously never exposed). `tool_choice: "required"` has no native equivalent (`ForceToolId` forces one *specific* tool, not "any tool") — best-effort maps to `"auto"`. Streaming emits a tool call as one complete delta when the native `NativeToolCall` chunk arrives, not incrementally.
- **`/v1/audio/speech`** — raw binary WAV response (`Results.Bytes`), not a JSON envelope — matches OpenAI's real wire behavior and is the first non-JSON response in this API. Only `response_format: "wav"` is accepted; `AudioResult.Data` is always a pre-encoded WAV container, no mp3/opus/aac encoder exists.
- **`/v1/audio/transcriptions`** — the first multipart/form-data route in this API (every other request body is base64-in-JSON). Reads the form via `HttpRequest.ReadFormAsync()` directly rather than `[FromForm]` binding, sidestepping .NET 8+'s antiforgery requirement for form-bound minimal-API endpoints (a browser-CSRF protection this machine API doesn't need). Only WAV input (`AudioClipCodec.Decode` has no ffmpeg dependency) and `response_format: "json"`/`"text"` (not `srt`/`vtt`/`verbose_json`, which need per-segment timestamp formatting not built yet).
- **`/v1/embeddings`** — see "Text embeddings" below for the Engine-side capability. `encoding_format` only supports `"float"`. A `dimensions` request that doesn't match the model's real output width is rejected, not truncated (Matryoshka-style truncate-and-renormalize correctness hasn't been verified for the models this engine ships).

### Text embeddings (`Modality.Embedding`)

New modality added for RAG/semantic-search-style dense sentence vectors, backed by `IEmbeddingService`/`EmbeddingService` (`src/HartsyInference.Engine/Services/`) — mirrors the shape of every other per-capability Engine service. **Decoder-LLM-backed only** (`HartsyInference.LLM.Embeddings.DecoderEmbeddingModel`, which wraps the same `GgufLanguageModel`/`GenericTransformer` GGUF pipeline chat models use — Qwen3-Embedding/gte-Qwen2/e5-mistral family). `BertEmbeddingModel` (bge/gte/nomic-family, BERT-style bidirectional encoders) exists in the same namespace but is **not wired** — it needs its own WordPiece tokenizer plumbed in (`BertWordPieceTokenizer` exists, unconnected) and has no verification path; a separate, later scope.

- **Correctness-critical detail**: `DecoderEmbeddingModel.Encode` pools whichever token is literally last in the ids you pass it — it does not append EOS itself. `EmbeddingService` appends the tokenizer's EOS id before encoding, matching the reference convention (HF's tokenizer auto-appends EOS before the same last-position pool). Verified 2026-07-22 against a real `transformers.AutoModel` reference for identical token ids: full 1024-dim cosine similarity = **1.000000** (`tests/HartsyInference.LLM.Tests/DecoderEmbeddingRealCheckpointTests.cs`, env-gated on `HARTSY_EMBED_GGUF_PATH`, skips cleanly otherwise).
- **Real gap found and routed around, not fixed**: the catalog's `qwen3-embedding` entry ships the **f16** GGUF, not a quantized one. A Q8_0 quant of the same model fails on the CPU backend with `"Quantized dtype conversion (Q8_0 → F32) requires a dedicated dequantizer. Use GgufDequantizer instead."` — `GenericTransformer.Layer.Forward` → `Project` → `backend.Linear` → `Tensor.CastTo` has no path from a block-quantized dtype straight to F32 without going through `GgufDequantizer` first. This is a broader, pre-existing CPU-backend limitation for quantized-GGUF inference generally (the forward-pass code path is shared with ordinary chat generation, not embedding-specific) — worth a real fix, but out of scope for this pass.
- Not CLI-drivable yet (no `hartsy embed` command) — reachable only via `POST /v1/native/embeddings` and `POST /v1/embeddings`.

### Production hardening

- **Multi-key auth** (`ApiKeyStore`) — `HartsyInferenceServerOptions.ApiKeys` (list of `{Key, Name, RateLimitPerMinute}`) replaces the old single shared secret; each key resolves to a named caller identity, not just a bool. The legacy `HartsyInferenceServerOptions.ApiKey` string still works — it's folded into `ApiKeys` as one `"default"`-named entry at startup, so existing single-key deployments don't need to change config. Zero keys configured = auth disabled entirely, same as before. The resolved identity is stashed on `HttpContext.Items` (`HartsyInferenceServiceExtensions.ApiKeyIdentityItemKey`) for the rate limiter, usage tracker, and request tracing to share without re-parsing headers.
- **Per-key rate limiting** — `Microsoft.AspNetCore.RateLimiting`'s built-in fixed-window limiter (no new package), partitioned by the resolved identity name (or remote IP when auth is disabled). Limit comes from the key's own `RateLimitPerMinute` or falls back to `HartsyInferenceServerOptions.DefaultRateLimitPerMinute` (default 300/min). A distinct failure mode from `QueueFullException`'s `rate_limit_error` (server capacity) — a 429 from the rate limiter uses `type: "rate_limit_exceeded"` so a client can tell "you're over your quota" from "server is momentarily saturated". `/health`, `/ready`, `/version` are exempt, same as auth.
- **Usage metering** (`UsageTracker`) — in-memory per-caller counters (total requests, error count, by-modality breakdown, last-seen), recorded by one piece of middleware that wraps the rate limiter (so even a 429-rejected request counts). Read via `GET /admin/usage`. In-memory only, resets on restart — not a billing ledger, a foundation for one.
- **Observability** (`ApiMetrics`) — `GET /metrics` in Prometheus text format (`OpenTelemetry.Exporter.Prometheus.AspNetCore`), combining ASP.NET Core's own request instrumentation (duration/count/in-flight) with a domain-specific `hartsyinference_requests_total` counter (tagged route-group "modality" + outcome — covers every request, not just generation calls, hence the name) and a `hartsyinference_queue_pending` gauge over both `InferenceQueue`s. Request tracing auto-instruments every call as an `Activity` honoring the W3C `traceparent` header, tagged with the resolved caller identity (no exporter wired yet — spans stay in-process; wire one if cross-service trace correlation is ever needed). `/metrics` is gated behind the same auth as `/admin/*` (queue depth and error rates are ops-sensitive), not exempted like the liveness probes.
- **OpenAPI spec** — `GET /openapi/v1.json` via ASP.NET Core's built-in `AddOpenApi`/`MapOpenApi`. Deliberately a *thin* spec this round: request shapes are accurate (the DTOs are already documented per `docs/CODE_STYLE.md`), but most routes still return anonymous objects, so response schemas in the generated document are under-specified. A full typed-response-DTO retrofit across all ~25 endpoints is a known follow-up, not done yet.
- **Explicitly out of scope**: idempotency keys (low value for a single-process API with no side-effecting-retry problem today) and formal API versioning (routes already carry `/v1/`; a real v2 strategy is a future decision).

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
