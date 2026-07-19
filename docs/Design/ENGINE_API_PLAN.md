# Engine API & Full-Coverage Plan — "every model type, any model"

> How the engine's generation API gets fully rewritten into typed per-capability services with a rich contract, so
> **every modality** and **every architecture** run through one place, and the CLI, HTTP API, and SwarmUI extension all
> become thin wrappers. Successor to `ENGINE_REFACTOR_PLAN.md` (which did the package/relocation groundwork).

## The reframe (and a correction)

There are **four** copies of "load model + generate" today, not two:

| Consumer | Orchestration | Coverage |
|---|---|---|
| CLI handlers | `Engine/Dispatch/Handlers/*` | image = **SDXL only** |
| Server | `Engine/ModelManager` | image = **SDXL only** |
| **SwarmUI extension** | **`SwarmUI-HartsyInference/Generation/*`** | **~30 architectures + LoRA/ControlNet/IP-Adapter/Refiner/samplers/img2img/inpaint** |
| (direct library users) | ad hoc | — |

The earlier claim that "the extension barely changes" was wrong. The extension holds the **most complete** orchestration
in the ecosystem — per-arch `*Loader.cs` (construction) and per-feature `*Resolver.cs` (LoRA, ControlNet, IP-Adapter,
Refiner, sampling). The correct move is to **lift that into the Engine** as the single source of truth. Then the
extension collapses to a SwarmUI-params↔Engine-request mapper. This is *lift-and-centralize existing, working code*,
not write-from-scratch — which is what makes full-architecture coverage tractable.

## Part 1 — The contract: rich native request/result records

Toy DTOs (the current 8-field `ImageRequest`) are why nothing could delegate to the Engine. The contract must carry
**everything SwarmUI's `T2IParams` expresses**, using the codebase's three-tier options pattern:

- **Flat common props:** prompt, negative, width, height, steps, cfg, seed, sampler, scheduler, clipSkip, batch.
- **Composition (nullable):** `LoraStack` (list of {model, weight}), `ControlNetConditioning` (list of {type, image,
  strength, start, end}), `IpAdapter`, `Refiner` ({model, control, steps, cfg}), img2img `InitImage`+strength,
  inpaint `MaskImage`.
- **Extension bag:** arch-specific knobs (`IReadOnlyDictionary<string,object>`), so a new family needs no contract change.

One request record per capability (`ImageRequest`, `TextRequest`, `SpeechRequest`, `TranscribeRequest`, `MusicRequest`,
`VideoRequest`, `WorldRequest`, `MeshRequest`, `VisionRequest`) + matching result records. These records ARE the
contract — shared verbatim by in-process callers and the HTTP layer; the OpenAI DTOs map onto them.

## Part 2 — Per-capability services + the facade

Replace the `ParamState` string-bag dispatch with typed services on `IInferenceEngine`:

```
IInferenceEngine
  Load(pathOrId) → ModelHandle · Unload · cache/VRAM
  Images     : GenerateAsync(ImageRequest)      → ImageResult     (+ step-preview progress)
  Text       : GenerateAsync/StreamAsync(TextRequest) → tokens / TextResult
  Speech     : SynthesizeAsync(SpeechRequest)   → AudioResult
  Transcribe : RunAsync(AudioRequest)           → TranscriptResult
  Music      : GenerateAsync(MusicRequest)      → AudioResult
  Video      : GenerateAsync(VideoRequest)      → IAsyncEnumerable<VideoFrame>
  World      : OpenSession(WorldRequest)        → IWorldSession   (stateful stream)
  Mesh       : GenerateAsync(MeshRequest)       → MeshResult
  Vision     : Embed/Detect/Segment(VisionRequest) → typed results
```

Consistent conventions everywhere: `async` + `CancellationToken`, streaming via `IAsyncEnumerable<T>`, progress via
`IProgress<T>`. Batched-vs-single and backend gating are Engine strategy (fold in `ModelManager`'s paged-KV/queue).

## Part 3 — Per-architecture recipes (this is "any model")

A recipe registry keyed on `ModelArchitecture`:

```
IArchitectureRecipe
  DiffusionPipelineBase Construct(string checkpoint, IBackend backend, ModelHandle deps)
  EncodedConditioning   Encode(ImageRequest req, ...)   // arch-specific tokenizer(s) + text encoder(s)
  ImageResult           Generate(pipeline, conditioning, ImageRequest, progress)
```

`Images.GenerateAsync` = detect arch → recipe.Construct (cached in `ModelHandle`) → apply feature resolvers (LoRA /
ControlNet / IP-Adapter / Refiner) → recipe.Encode → recipe.Generate. **The recipes are lifted from the extension's
`Generation/*Loader.cs`; the feature resolvers from its `*Resolver.cs`.** Every architecture the extension drives today
(the ~30 above) becomes a recipe — that IS full coverage, and it's proven code, not new code.

## Part 4 — Consumers become thin wrappers

- **CLI:** the 9 handlers become ~50-line `ParamState` ↔ request / result ↔ `GeneratedArtifact` mappers.
- **HTTP API:** Server = endpoints + OpenAI DTO ↔ native request + auth (~620 LOC); `ModelManager` folds into Engine.
- **SwarmUI extension:** `HartsyInferenceBackend` becomes a `T2IParams` ↔ `ImageRequest`/`VideoRequest` mapper +
  result mapper; **`Generation/` moves into the Engine** (the extension shrinks by most of its 21.5k LOC).
- **Meta:** Engine already added.

## Phasing (each phase build-verifiable; heavy generation is GPU-gated)

- **A. Contract** — author the rich per-capability request/result records. *No GPU.*
- **B. Services + facade** — the typed services over the *existing* SDXL + LLM paths, so nothing regresses. *No GPU.*
- **C. Recipe registry + first lift** — the `IArchitectureRecipe` seam + lift 1–2 arches from the extension
  (start ZImage/Chroma — smallest on disk). *Build per arch; GPU-verify per arch.*
- **D. Feature resolvers** — lift LoRA / ControlNet / IP-Adapter / Refiner from the extension. *Build; GPU-verify.*
- **E. Lift remaining arches** — one at a time from the extension, verifying each. *GPU-gated.*
- **F. Rewire consumers thin** — CLI handlers, Server endpoints, and the SwarmUI extension onto the contract; delete
  the extension's `Generation/`. *Build-verifiable; e2e GPU-verify.*

## Scope reality

This is a multi-week, cross-repo (engine + SwarmUI) effort, and every arch's e2e verification needs the GPU free
(diffusion loads OOM a shared GPU). The de-risker: **the hard part — per-arch construction + encoding for all ~30
families — already exists and works in the extension.** This plan centralizes it; it does not reinvent it. Parts A, B,
and the *structure* of C/D are build-verifiable now without touching the GPU.
