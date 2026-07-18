# Engine Service Layer — Refactor & Debloat Plan

> **Goal.** Introduce one in-process service layer, `HartsyInference.Engine`, that is the single source of
> truth for "load a model + generate." The CLI, the future HTTP API, and the SwarmUI extension all become thin
> wrappers over it. This eliminates the two divergent proto-Engines that exist today (the server's `ModelManager`
> and the CLI's `Dispatch/Handlers`), un-pins image generation from SDXL, and gives every modality one home.

## Status

- ✅ **Phase 0** — `HartsyInference.Engine` created; 5 lifecycle files moved up out of ModelHandler.
- ✅ **Phase 1** — `ModelManager`/`InferenceQueue`/PNG relocated into Engine; native `Engine.Requests.ImageRequest`.
- ✅ **Phase 3** — all 9 modality handlers + dispatch + shared infra (`Modality`, `ParamState`, `RepoPaths`,
  `PngEncoder`, `FrameWriter`, `BackendFactory`, …) moved into Engine. The CLI is now a thin wrapper.
- ✅ **Phase 2** — `InferenceEngine` facade (owns backend + runner cache + dispatch); `CommandRunner` and
  `ReplSession` collapsed onto it. One-shot commands and the REPL both route every generation through the facade.
- ✅ **Phase 5 (safe subset)** — deleted dead `SdxlInpaintPipeline`; collapsed the two PNG encoders into one;
  added `Engine` to the Meta package.
- ⏸ **Phase 4 (un-pin architectures)** and the **risky Phase 5 items** (unify the 3 LLM generate paths, merge
  `LensPipelineFactory`, dedup `ConvertedWeights`, consolidate audio DSP) touch model runtime code that can only
  be validated by loading real weights on the GPU — **held**, because heavy diffusion loads OOM the shared GPU and
  crash the running server. Do these when the GPU is free (or drive on the CPU backend).

**Verified:** all `src/` projects build clean; text generation runs end-to-end through the relocated Engine handler
and the facade. (Image path is the identical relocation pattern; not re-run on GPU to avoid the OOM.)

## Principles (decided)

1. **In-process service is the source of truth; HTTP is one thin adapter over it** — never route in-process work
   through HTTP. A daemon deployment, if ever needed (many clients, one resident GPU model), is just the HTTP
   adapter hosting the same Engine.
2. **Native request/result records are the contract** — they express every modality (image params, video, world
   sessions, 3D) that an OpenAI schema cannot. An optional OpenAI-compat adapter layers on top for chat/images/audio.
3. **Move orchestration up, keep primitives down.** Pipelines/denoisers/codecs/kernels stay in their modality
   packages; the Engine only calls their existing `GenerateFromTokens`/`GenerateFromEmbeddings`/`Synthesize`/… .
4. **Debloat is structural, not line-count.** Net LOC is ~flat; the win is deleting the *second* implementation of
   the load→route→generate path (today diverging) and collapsing duplicated factories/encoders.

## Dependency placement

`ModelHandler` stays foundational (depends only on `Core`; used bottom-up by every modality package, and Diffusion
uses its `ModelArchitectureDetector`/`ModelLayoutResolver`). **Do not rename it Engine.** Engine sits on *top* of all
modality + backend packages.

```
Engine  ->  Core, ModelHandler, Cpu, Cuda, Vulkan, Tokenizers,
            LLM, Diffusion, Audio, Vision, Video, Interactive, ThreeD
CLI / Server / SwarmUI  ->  Engine  (thin wrappers)
Meta  ->  + Engine (15th dependency)
```

---

## Phase 0 — Engine skeleton + lifecycle move (mechanical, no behavior change)

Create `src/HartsyInference.Engine` (deps: `Core`, `ModelHandler`). Move the 5 model-lifecycle files out of
ModelHandler (they are HTTP/download/cache concerns used only by high-level consumers) into
`HartsyInference.Engine.Registry` / `.HuggingFace`:

- `Registry/ModelRegistry.cs` (carries `LoadedModel`), `Registry/ModelCacheStore.cs`, `Registry/ModelInfo.cs`
- `HuggingFace/HuggingFaceClient.cs`, `HuggingFace/HuggingFaceModelIndex.cs`

**Stays in ModelHandler** (Diffusion depends on it): `ModelArchitecture`, `ModelArchitectureDetector`,
`ModelLayoutResolver`. Repoint the 3 consumers (`Server/ModelManager.cs`, `Cli/Commands/PullCommand.cs`,
`Cli/Infra/CacheView.cs`) to the new namespaces; add the Engine `ProjectReference` to Server + CLI; add Engine to
the solution. **Verify:** ModelHandler, Engine, Server, CLI all build clean.

## Phase 1 — the service layer

Move the transport-agnostic orchestration into Engine and expose the facade:

- From Server → Engine: `ModelManager.cs` (~290), `InferenceQueue.cs` (56), `Imaging/PngImageWriter.cs` (~120,
  becomes the one PNG encoder), the backend/options plumbing.
- Add `IInferenceEngine` facade: `Load(pathOrId) -> ModelHandle`, `Unload`, cache/VRAM, plus capability services
  `Images` and `Text` first (the two that exist today). Consistent conventions: async + `CancellationToken`,
  streaming via `IAsyncEnumerable<T>`, progress via a `Progress<T>` callback. Native request/result records live here.
- Expand Engine deps to the modality/backend packages it now needs.
- **Verify:** Engine builds; SDXL image + GGUF text still generate through the new service (structural parity).

## Phase 2 — collapse the server

Point the HTTP endpoints at `IInferenceEngine`; delete `ModelManager` from Server. Server keeps only `Program.cs`,
the endpoint mapping half of `HartsyInferenceServiceExtensions.cs`, `OpenAiDtos.cs`, and the `ApiKey` option (~620
LOC of pure ASP.NET/OpenAI transport). **Verify:** `/v1/chat/completions`, `/v1/images/generations`, `/v1/models`
unchanged.

## Phase 3 — collapse the CLI

Point the 9 `Dispatch/Handlers` at the Engine. Each collapses from an orchestrator to a ~50-line mapper
(`ParamState` → engine request, engine result → `GeneratedArtifact`). Keep all CLI UI (Commands, REPL, LineEditor,
Infra discovery/picker/TerminalImage/catalog/theming). Progress/result abstractions (`IProgressSink`,
`GeneratedArtifact`) become shared where useful. **Verify:** CLI generates against on-disk weights (SDXL image, a
GGUF text) with no regression.

## Phase 4 — un-pin architectures (the payoff)

Now that construction + generate live in one place, cover every architecture there, once, for all consumers:

- Extend Engine model construction beyond SDXL: fold `PipelineFactory.LoadSdxl` + `LensPipelineFactory.*` +
  `GgufLanguageModel.Load` + `SsmLanguageModel.Load` behind one `Engine.Load`, and add per-family diffusion
  construction (the `LoadAuto` switch gains the non-SDXL cases).
- Add a per-pipeline prompt-level generate (`GenerateFromPrompt`) or an Engine-side arch→tokenize→encode dispatch,
  so `GenerateFromTokens` heterogeneity is hidden behind one call.
- Wire the modalities the CLI stubs today (SAM/face — the models are complete; only the handler lied).
- **Verify against on-disk weights**, priority order: Flux (dev/schnell/krea), SD3 / Qwen-Image, Chroma / ZImage /
  AuraFlow, then the rest.

## Phase 5 — debloat pass

- Unify the 3 LLM generate paths (`TextGenerationPipeline`, `DynamicBatchScheduler`, `SsmGenerationPipeline`) onto
  one shared prompt-build/sampling core; batched-vs-single becomes an Engine strategy.
- Merge `LensPipelineFactory` into `PipelineFactory` (one switch).
- Consolidate the two hand-rolled PNG encoders → one; two backend factories → one.
- Hoist the ~10 near-duplicate `ConvertedWeights` DTOs in `ModelHandler/CheckpointConverters/` into one generic type.
- Consolidate the Audio iSTFT/NSF-source DSP re-rolled across ~5 vocoders into one `Preprocessing/`/`Dsp` primitive.
- Delete `Diffusion/Pipelines/SdxlInpaintPipeline.cs` (throw-only, zero references) — or finish it.
- Finish-or-cut the half-wired encoders: SNAC encode, Mimi encode, PocketTts placeholder config.

## Verification

Every phase must end with a clean solution build (0 warnings). At the end, run real test generations to confirm no
regression: at minimum an SDXL image and a GGUF text completion on the on-disk weights, plus one newly-wired
architecture from Phase 4. `docs/Checklists/PARITY_VERIFICATION.md` remains the parity authority.

## Not in this refactor (follow-on)

The CLI and HTTP API thin-wrapper polish, and the OpenAI-compat adapter, come *after* the Engine backbone is in and
proven.
