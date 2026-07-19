# SwarmUI Extension Refactor Plan — collapse the three extensions onto `HartsyInference.Engine`

> **Deliverable of the "thin wrappers" handoff.** Per-extension, phased plans to collapse each SwarmUI extension
> onto the Engine's per-capability services + rich request/result contract, by **lifting** the redundant
> orchestration into the Engine (not deleting capability). Successor/companion to `ENGINE_API_PLAN.md` (Engine-side
> target API) and `ENGINE_REFACTOR_PLAN.md` (package relocation, done).
>
> **Status: PLAN ONLY.** No extension edits until reviewed. Engine-side contract work is the gating dependency.

---

## 0. The situation, precisely

Four copies of "load model + generate" exist; the SwarmUI extensions hold **three of them**, and they are the
*most complete* copies — not thin clients. None of them touch the Engine at all today; they duplicate it. The refactor
target: each extension becomes a **params ↔ Engine-request mapper + result mapper**, and every piece of orchestration
(arch construction, encoding, feature application, synthesis loops) moves *into* the Engine.

### The gating dependency (read this first)

The Engine's package relocation is **done and building** (net8.0 + net10.0), but its public API is still the
**pre-refactor toy**:

- The facade (`InferenceEngine.Generate`) is `ParamState` string-bag based, not typed services.
- `Requests/ImageRequest.cs` is **8 fields** (prompt, negative, w, h, steps, cfg, seed, clipSkip). It cannot express
  a LoRA stack, ControlNet, IP-Adapter, refiner, img2img, inpaint, regional prompts, component overrides, or video.
- `PipelineFactory.LoadAuto` wires **SDXL only**; every other architecture's construction lives in the extension's
  `Generation/*Loader.cs` (`ImageHandler` literally throws `NotSupportedException` for non-SDXL pipelines).

**Consequence:** an extension can only delegate a capability *after* the Engine grows the typed request record + the
service + (for image) the per-arch recipe for that capability. So this is a **cross-repo, dependency-ordered** effort:
Engine adds contract/service/recipe → publish alpha (or rebuild the net8.0 CLI closure) → extension bumps its pin and
deletes the lifted code. The plan is sequenced around that, and the Engine-side prerequisite phases
(`ENGINE_API_PLAN.md` Parts A/B/C/D) are called out explicitly per extension.

### Decisions this plan takes (the handoff's open questions, resolved)

| Question | Decision | Why |
|---|---|---|
| Contract schema | **Native HartsyInference request/result records** as source of truth; OpenAI DTOs map *onto* them | Only native records express video/world/3D/LoRA/ControlNet/refiner; OpenAI can't. Matches `ENGINE_REFACTOR_PLAN.md` principle #2. The `T2IParamTypes` field list in §2.2 *is* the authoritative image contract. |
| Where ControlNet preprocessors live (Canny/Depth/OpenPose/annotators) | **Stay extension-side initially** (SwarmUI `Image`-coupled + download annotator models via Swarm), expose as an Engine `Vision` preprocessing service in a *later* pass | They consume SwarmUI image types and Swarm's downloader; lifting them is orthogonal to the core "construct+generate" lift and can follow. The *conditioning tensor* they produce is what the Engine `ControlNet` composition object accepts. |
| Model acquisition (Engine `ModelAsset`/`ModelDownloader` vs extension `ModelAutoDownloader` + Swarm model manager) | **Engine owns the pure download/lock/hash/atomic-move + the `SideModels` URL/hash registry; extension keeps the thin `T2IModel`-producing + Swarm-model-set-refresh adapter** | The registry data (`SideModels.cs`, `AudioWeightsRegistry`) is portable verbatim; only the `T2IModel` production and Swarm registry refresh are SwarmUI-specific. |
| Do LLMAssistant / AudioLab thin out now or later? | **Image backend is the priority (Extension A); B and C follow, and are lower-risk** because they already have a clean handler seam (`ILLMProvider`, `IAudioHandler`) | The image backend holds the ~30-arch goldmine; B/C are already one-file / one-interface delegations. |

### Contract-schema principle used throughout — three tiers

Every request record uses the codebase's three-tier options pattern so a new model family needs **no contract change**:

1. **Flat common props** — the high-frequency fields (prompt, negative, w/h, steps, cfg, seed, sampler, …).
2. **Composition objects (nullable)** — `LoraStack`, `ControlNetConditioning`, `IpAdapter`, `Refiner`, `Img2Img`,
   `Inpaint`, `Regional`, `VariationSeed`, `ComponentOverrides`.
3. **Extension bag** — `IReadOnlyDictionary<string,object>` for arch-specific knobs.

---

## Extension A — SwarmUI-HartsyInference (the backend; ~21.5k LOC, the priority)

**Role today:** the SwarmUI `AbstractT2IBackend` implementation (`Backends/HartsyInferenceBackend.cs`, **1951 LOC**)
plus `Generation/` (~80 files) — the ecosystem's most complete diffusion/video/audio orchestration: **~33 per-arch
loaders + ~24 feature resolvers + ~9 preprocessors + infra**. This is what moves into the Engine.

### A.1 — Inventory classification (every `Generation/` file → destiny)

Coupling flag from the audit: **PURE** = 0 SwarmUI type refs (lifts verbatim), **EDGE** = thin param/model/image
shell, **COUPLED** = reads many `T2IParamTypes` / marshals `Image` (needs the param-read prologue + image epilogue
split out). The universal lift technique for COUPLED files: **the middle (construct + encode + generate) is already
engine-native; only the param-read *prologue* and the `Image`-marshal *epilogue* are SwarmUI.** Extract the middle to
the recipe; the prologue becomes the `ImageRequest` mapper in the backend; the epilogue (`RgbToImage` etc.) stays
extension-side.

#### Loaders → Engine `IArchitectureRecipe` registry (lift the middle; keep param-read in the backend mapper)

| Bucket | Files | Target |
|---|---|---|
| **Image recipes** (~22) | Sd15, Sdxl, Sd3, Flux, Flux2, Chroma, ChromaVariant, AuraFlow, FLite, Ideogram4, BooguImage, ErnieImage, Lumina2, HunyuanImage, OmniGen2, ZImage, Anima, HiDream, QwenImage, Krea2, Lens, Lance | one `IArchitectureRecipe` each, keyed on `ModelArchitecture`. Start with **ZImage / Chroma** (smallest on disk) per `ENGINE_API_PLAN.md` Phase C. |
| **Video recipes** (~7) | WanVideo, WanVace, WanAnimate, WanS2V, LtxVideo, LtxVideo2, (Lens/Lance also video) | recipes behind `Video.GenerateAsync(VideoRequest)`; `WanModelVariants` (variant discrimination) lifts PURE. |
| **Audio recipes** (~4) | AceStep, AceStep15, MusicGen, Yue | **→ Engine `Music` service** — *the same models AudioLab drives* (see §C: dedupe both onto one Music service). |
| **Refiner** | RefinerLoader (EDGE) | folds into the `Refiner` composition object + resolver. |

#### Resolvers → Engine feature resolvers (applied by `Images.GenerateAsync` before `recipe.Encode`)

| Feature | Files | Coupling | Notes |
|---|---|---|---|
| LoRA | LoraResolver (EDGE), **LoraApplier (PURE)** | LoraApplier lifts verbatim (already `LoraStack`); LoraResolver's `T2IParamTypes.Loras` read becomes backend mapping. |
| ControlNet | ControlNetResolver, FluxControlNetResolver (EDGE) | produce the `ControlNetConditioning` composition object; preprocessors stay extension-side (see A.4). |
| IP-Adapter / Redux | IpAdapterResolver (COUPLED, 866 LOC), ReduxResolver (EDGE) | biggest resolver; the `Image`→CLIP-vision embed marshalling is the coupling. |
| Refiner | RefinerResolver, SegmentRefiner (EDGE) | `Refiner` composition object. |
| Sampling / steps | SamplingParamResolver, VariationSeedResolver (EDGE) | flat props + `VariationSeed` object. |
| Img2img / inpaint | Img2ImgResolver, MaskResolver (EDGE/COUPLED) | `Img2Img` + `Inpaint` composition objects. |
| Regional / segment | **RegionalPromptResolver (PURE)**, SegmentResolver (COUPLED, 568), SegmentRefiner | RegionalPromptResolver lifts verbatim (`RegionalPlan`); Segment auto-mask is COUPLED. |
| Conditioning | **PromptConditioningResolver (PURE)**, **WeightedConditioning (PURE)**, EmbeddingResolver (EDGE) | first two lift verbatim (`ConditioningSchedule`). |
| Detection (for segment/controlnet) | ClipSegResolver, GroundingDinoResolver, RtDetrResolver | → Engine `Vision` service (Detect/Segment) — these are already vision models. |
| Video params | VideoParamResolver (COUPLED), TaesdResolver | `VideoRequest` mapping + preview decoder. |

#### Preprocessors / annotators → **stay extension-side** (this pass), later → Engine `Vision`

Canny, DepthAnything, OpenPose, ClipImage, AnnotatorControlPreprocessors, AnnotatorDownloader, WanAnimate{Pose,Face,Driving}.
They consume SwarmUI `Image` and download annotator weights via Swarm. Their **output tensor** feeds the
`ControlNetConditioning` object. Lift to a Vision preprocessing service only after the core construct+generate lift lands.

#### Infra → mixed

| File | Destiny |
|---|---|
| **ModelSupport (PURE)** — CompatClass→`ModelArchitecture` map | **lift to Engine** — becomes the recipe-registry key resolver. |
| **LoaderVaeUtils, VaePrecisionHelper (PURE)** | lift verbatim (VAE key remap + F16 precision policy). |
| **EngineGap (PURE)** | **delete on lift** — it exists *only* to stub missing Engine surface; the lift makes the real surface exist. Its call sites are the checklist of what the Engine must expose. |
| PipelineCache (612, EDGE) | lift as Engine `ModelHandle` cache (fold with `ModelManager`'s residency/LRU). |
| SideModels (483), ModelAutoDownloader (165) | data registry → Engine; download/lock/hash → Engine `ModelDownloader`; `T2IModel` production + Swarm refresh → stays (see A.4). |
| RgbToImage, PreviewEncoder, VideoOutputEncoder, AudioOutputEncoder, AudioDecoder, ControlVideoDecoder | **stay** — SwarmUI `Image`/`AudioFile` ↔ tensor marshalling; this is the residual mapper's I/O half. |
| TokenizerExport, Ideogram4MagicPrompt | Ideogram magic-prompt is an LLM expansion → could use Engine `Text` service later; TokenizerExport is a build-time helper. |

### A.2 — `T2IParamTypes` → Engine `ImageRequest`/`VideoRequest` (the authoritative contract)

This is the field-level mapping the handoff asked for: **every distinct param the extension reads is a field the
Engine contract must carry.** The current 8-field `ImageRequest` expresses only the first block; **everything else is a
net-new Engine-side addition** (flagged ⚠).

**Flat common props (`ImageRequest`)**

| SwarmUI param | Engine field | Status |
|---|---|---|
| Prompt, NegativePrompt, Width, Height, Steps, CFGScale, Seed | (same) | ✅ exists |
| ClipStopAtLayer | ClipSkip | ✅ exists |
| Sampler (custom `SamplerParam`), — | Sampler | ⚠ add |
| SigmaShift | Scheduler / SigmaShift | ⚠ add |
| EndStepsEarly | EndStepsEarly | ⚠ add |
| IP2PCFG2 | InstructPix2PixCfg | ⚠ add |
| (batch) | Batch | ⚠ add |

**Composition objects (nullable)** — all ⚠ net-new

| Composition object | Source `T2IParamTypes` |
|---|---|
| `ComponentOverrides` { vae, t5xxl, clipL, clipG, clipVision, qwen, llama, gemma } | VAE, T5XXLModel, ClipLModel, ClipGModel, ClipVisionModel, QwenModel, LLaMAModel, GemmaModel |
| `LoraStack` : list { model, weight, tencWeight, sectionConfinement } | Loras, LoraWeights, LoraTencWeights, LoraSectionConfinement |
| `ControlNetConditioning` : list { type, image, strength, start, end } | Controlnets, ControlNetParamHolder, ControlNetStrength |
| `IpAdapter` { promptImages, grouping, faceIdV2Weight } | PromptImages, GroupImagePrompting, ConcatDropdownValsClean, FaceIdV2Weight |
| `Refiner` { model, vae, method, control, steps, cfg, upscale } | RefinerModel, RefinerVAE, RefinerMethod, RefinerControl, RefinerSteps, RefinerCFGScale, RefinerUpscale |
| `Img2Img` { initImage, creativity } | InitImage, InitImageCreativity |
| `Inpaint` { mask, grow, blur, shrinkGrow } | MaskImage, MaskGrow, MaskBlur, MaskShrinkGrow |
| `Regional` { plan, sortOrder, maskGrow, maskBlur, maskOversize, steps, cfg } | (prompt `<region>`/`<segment>` syntax) + SegmentSortOrder, SegmentMaskGrow, SegmentMaskBlur, SegmentMaskOversize, SegmentSteps, SegmentCFGScale |
| `VariationSeed` { seed, strength } | VariationSeed, VariationSeedStrength |

**Extension bag** (`IReadOnlyDictionary<string,object>`) — the 10 custom-registered params + arch knobs:
`DtypeOverride`, `TileVaeThreshold`, `Ideogram4MagicPrompt(+Model)`, `AnimateReferenceImage`, `AnimateAutoPreprocess`,
`AnimatePoseVideo`, `AnimateFaceVideo`. (Registered by `SwarmUIHartsyInference.cs` via `T2IParamTypes.Register`.)

**`VideoRequest`** (all ⚠ net-new): VideoModel, VideoSwapModel, VideoSwapPercent, VideoExtendModel, VideoResolution,
VideoFPS, VideoFormat, VideoBoomerang, VideoEndFrame, VideoAudioInput, VideoAudioReference, Text2VideoFrames,
TrimVideoStartFrames, TrimVideoEndFrames.

**Text2Audio params** (Text2AudioStyle/Duration/BPM/KeyScale/TimeSignature/Language) → map onto the Engine **`MusicRequest`**
shared with AudioLab (see §C) — the main extension already drives AceStep/MusicGen/Yue, so this is the same service.

### A.3 — Residual thin `HartsyInferenceBackend` design (post-lift)

The 1951-LOC backend collapses to a mapper with four responsibilities, all SwarmUI-native:

1. **Load:** resolve `T2IModel` → checkpoint path + `ModelArchitecture` (via lifted `ModelSupport`), call
   `engine.Load(path)` → `ModelHandle` (the Engine now owns arch detection + recipe construction + the pipeline cache).
2. **Map request:** read `T2IParamInput` via `input.Get(T2IParamTypes.*)` → build `ImageRequest`/`VideoRequest`
   (the §A.2 mapping). This is the *only* place `T2IParamTypes` is read.
3. **Generate + stream:** `engine.Images.GenerateAsync(req, progress)` with an `IProgress<StepPreview>` that the
   backend renders via `PreviewEncoder` → SwarmUI `gen_progress.preview`.
4. **Map result:** `ImageResult.Rgb` → `RgbToImage`; video frames → `VideoOutputEncoder`; audio → `AudioOutputEncoder`.

Kept: `SupportedFeatures`, `IsValidForThisBackend` (honesty guard), `Init`/`Shutdown`/`FreeMemory`, the WebAPI
(`HartsyInferenceWebAPI.cs`), param registration (`SwarmUIHartsyInference.cs`), and the I/O marshalling helpers.
**Deleted (lifted):** the CompatClass→Loader dispatch chain, all `Generation/*Loader.cs`, the feature resolvers, the
pipeline cache, `EngineGap`. Estimated residual: low thousands of LOC vs 21.5k.

### A.4 — Model acquisition reconciliation

- **Lift:** `SideModels.cs` registry data (URL/hash/canonical name per side component) + `ModelAutoDownloader`'s
  lock/atomic-`.tmp`/SHA-256 logic → Engine `ModelDownloader` + a `ModelAsset` catalog (the Engine already has the
  `ModelAsset`/`ModelDownloader` shape; extend its catalog with the `SideModels` data).
- **Stays:** producing a SwarmUI `T2IModel` from the downloaded path and refreshing Swarm's model set — a thin adapter
  over the Engine download. Swarm's own model manager remains the user-facing model browser.

### A.5 — Phasing (Extension A), interlocked with `ENGINE_API_PLAN.md`

| Phase | Engine-side (HartsyInference repo) | Extension-side (SwarmUI repo) | GPU |
|---|---|---|---|
| A0 | Author rich `ImageRequest`/`VideoRequest` records + composition objects (§A.2) + `ImageResult`. (`ENGINE_API_PLAN` Part A) | — | none |
| A1 | `Images`/`Video` typed services on the facade over the *existing* SDXL path (no regression). (Part B) | — | none |
| A2 | `IArchitectureRecipe` seam + `ModelSupport` lift + lift **ZImage + Chroma** recipes from the extension. (Part C) | — | build per arch; GPU-verify per arch when free |
| A3 | Lift feature resolvers (LoRA/ControlNet/IP-Adapter/Refiner/img2img/inpaint/regional). (Part D) | — | build; GPU-verify small |
| A4 | Lift remaining image recipes, then video, then audio (one at a time). (Part E) | — | GPU-gated, one small model at a time |
| A5 | — | Add `<Reference Include="HartsyInference.Engine">` to the local-dev closure (⚠ **missing today** — see §5); bump alpha pin; rewrite `HartsyInferenceBackend` to the §A.3 mapper; **delete `Generation/`** as each arch lands upstream. | build (net8.0 green); e2e GPU-verify one small arch |

Extension deletion is **incremental**: an arch's loader is deleted only once its recipe is lifted, published, and
build-verified upstream. The backend keeps its dispatch fallback for not-yet-lifted arches until A4 completes.

---

## Extension B — SwarmUI-LLMAssistant (~12.5k LOC; delegate to Engine `Text`)

**Role:** LLM chat. All Engine consumption is concentrated in **one file**: `Backends/HartsyLocalLLMProvider.cs`
(775 LOC) — the entire local-GGUF generation orchestrator. The extension already has the ideal seam: `ILLMProvider`
(the interface the stable core talks to) with a **disabled `SwarmNativeLLMProvider` stub** that is literally the
template for this refactor.

### B.1 — Lift map

**LIFT into Engine `Text` service** (all of `HartsyLocalLLMProvider.cs`):
device/slot management + backend creation + PTX resolution + RAM guard + model load/evict; architecture probe +
transformer-vs-SSM routing (`GgufLanguageModel` vs `SsmLanguageModel`); chat-template selection + raw-completion
fallback; sampling construction (`BuildSampling`/`BuildVisionSampling`); the streaming decode-and-diff loop; multimodal
(mmproj discovery, Siglip/Qwen2.5-VL/Mllama encoder selection, `VlmImagePreprocessor`, `MultimodalGenerator`);
`CountTokens`; model enumeration (`ListModels`, vision-badge metadata — model-folder discovery stays Swarm-pathed).

**STAYS (SwarmUI glue):** backend registration + config UI (`LLMBackendPack`, settings), Swarm model-registry
integration, all of `WebAPI/`, `Services/` (25 files: threads, assistants, tools, permissions, media), `Tools/`,
`T2I/`; the `LLMs/` seam (`ILLMProvider`, registry, dispatcher, matcher, DTOs); **`LLMStreamHelper`** (WebSocket
framing, thread persistence, and the **agentic tool-calling loop** — note: tool-call *grammar masking* is engine-side
via `SamplingOptions.JsonModeSentinel`, but the loop itself is SwarmUI orchestration); **the remote providers**
(`RemoteOpenAILLMProvider`, `AnthropicLLMProvider`) — pure HTTP clients, unrelated to the Engine.

### B.2 — `Text` contract (`TextRequest`/`TextResult`)

Source of truth: `ExtendedLLMInput` (the extension's de-facto request DTO) + the backend's settings knobs.

- **Flat:** messages (`list<{role, content, media[]}>`) + system, temperature, topP, topK, minP, repetitionPenalty,
  maxTokens, seed, greedy/deterministic, device placement, JSON/grammar sentinel.
- **Decode-strategy hints:** graphDecode, speculativeDecode, lowVramQuant, alwaysFreeMemory.
- **Media:** image attachments (`{type: url|base64, data, mediaType}`) for the VLM path.
- **Streaming:** the Engine owns decode-and-diff and emits chunks. **Recommended contract:** `IAsyncEnumerable<TextChunk>`
  (or keep the `Action<TextChunk>` callback to minimize churn), where a chunk carries incremental decoded text **or** a
  terminal `stopReason`. Preserve the existing `{chunk | result | status | stopReason | native_tool_call}` event
  vocabulary — it is the de-facto wire schema `LLMStreamHelper` expects. After the lift the extension no longer touches
  `ILlmTokenizer`.

### B.3 — Phasing (Extension B)

| Phase | Engine-side | Extension-side | GPU |
|---|---|---|---|
| B0 | Author `TextRequest`/`TextResult` + streaming `TextChunk`; `Text` service wrapping the *existing* `TextGenerationPipeline` + `SsmGenerationPipeline` (fold the transformer/SSM routing in). | — | none (structural) |
| B1 | Add multimodal to the `Text` service (`MultimodalGenerator`/`MllamaGenerator` + encoder selection). | — | small-GGUF VLM verify OK |
| B2 | — | Fill in `SwarmNativeLLMProvider` (already stubbed) as `ExtendedLLMInput → TextRequest`, stream `TextResult` → `{chunk|stopReason}`; register it in place of `HartsyLocalLLMProvider`; delete the 775-LOC provider. | small-GGUF gen on CUDA — **allowed** per constraints |

**Lowest-risk extension** — small-GGUF LLM generation on CUDA is explicitly allowed, so B is fully e2e-verifiable
without waiting for a free GPU.

---

## Extension C — SwarmUI-AudioLab (~14.5k LOC; delegate to Engine `Speech`/`Transcribe`/`Music` + `Vc`/`Fx`)

**Role:** TTS / STT / music / voice-conversion / stem-separation. Already has the **cleanest seam of the three**:
`IAudioHandler.ProcessAsync(IBackend, args) → JObject` is exactly where the future services sit, and the per-capability
handlers already parse `args` into typed requests (`TtsRequest`, `SttRequest`, `MusicRequest`, `VcRequest`) that are the
**near-final form** of the Engine records. ~3,000 LOC of pure orchestration in `AudioServices/{Tts,Stt,Music,Vc,Fx}/`.

### C.1 — Lift map

**LIFT into Engine services** (`AudioServices/*`, no SwarmUI dep except `Logs`/`DownloadFile`):
- `AudioEngine.cs` device management (backend construction, gen-serialization semaphore, memory-pressure eviction) →
  Engine service infrastructure (reconcile with `ModelManager`'s residency/queue).
- The descriptor+runner registries — `Tts/TtsModels.cs` + `Tts/Models/*` (16), `Stt/SttModels.cs`, `Music/MusicModels.cs`
  (816, the largest), `Vc/VcModels.cs` + `Vc/Models/*`, `Fx/*Handler.cs` — each is repo-resolve → `AudioModelCache`
  download → loader/converter → `LoadWeights` → frontend/tokenizer → `Synthesize`/`Transcribe` closure → `float[]`.
- The typed request/result records (`TtsRequest` etc.) → move to the Engine so both sides share them.
- The generic handlers become thin (extension hands a request, gets a result).

**STAYS (SwarmUI glue):** `AudioLabParams.cs` (~110 param registrations), `DynamicAudioBackend.cs` (Swarm backend,
install/delete/reconcile, model listing) **except** `BuildEngineArgs` which retargets from untyped `args` to the typed
Engine requests; all `AudioAPI/*` (WebAPI); model browser/metadata (`AudioModels/`, `AudioProviders/` (55),
`AudioProviderTypes/`, `AudioWeights*`); **the 20 cloud-API providers** (`ApiHandlers/*`, `ApiEngineHandler`,
`AudioServerManager`'s API branch) — never touch HartsyInference, orthogonal to this refactor; ffmpeg decode + WAV
encode (`AudioIo.cs`, unless the Engine accepts/returns raw `float[]`).

### C.2 — Contracts (per capability)

- **`SpeechRequest` → `AudioResult`** (TTS): text, voice/speaker, language, speed, reference audio (clone: mono24k /
  wav path / b64 + refText), seed, sampling (temp/topP/topK/minP/repPenalty), per-model knobs (exaggeration, nfeStep,
  cfg, diffusionSteps) in the extension bag; output format/quality/volume.
- **`AudioRequest` → `TranscriptResult`** (STT): language, translate. **Gap to close:** timestamps, diarization,
  word/segment-level output, format — absent today; the new `TranscriptResult` is the place to add them.
- **`MusicRequest` → `AudioResult`** (music): prompt/lyrics, genre/style, duration, seed, shift, inferSteps, cfg, temp,
  topK; ACE metas (bpm, keyScale, timeSignature, vocalLanguage); CFG controls; 5Hz LM planner knobs. **Gap to close:**
  continuation/repaint/cover (`RepaintStart/End`, `CoverStrength`, `ACESourceAudio`, `ACEReferenceAudio`, `ACETaskType`)
  exist in the UI but aren't threaded into `MusicRequest` yet.
- **Beyond the three named services:** `Vc` (voice conversion: RVC, OpenVoice V2) and `Fx` (Demucs stems,
  Resemble-Enhance) are real capabilities that don't fit Speech/Transcribe/Music — the Engine needs a `VoiceConversion`
  and a stem-separation/enhance surface (or an extension-bag "audio-process" service) so capability isn't lost.

### C.3 — Dedup with Extension A

Both AudioLab **and** the main extension drive **AceStep / MusicGen / YuE**. The Engine `Music` service unifies them:
lift once, both extensions delegate. This is a concrete win — resolve the AceStep/MusicGen/Yue loaders in A's Music
bucket (§A.1) and C's `Music/MusicModels.cs` to **one** set of Engine recipes.

### C.4 — Phasing (Extension C)

| Phase | Engine-side | Extension-side | GPU |
|---|---|---|---|
| C0 | Author `SpeechRequest`/`AudioRequest`/`MusicRequest` + results (close the STT-timestamps / Music-continuation gaps); `Speech`/`Transcribe`/`Music` services + `VoiceConversion`/`Fx` surface, wrapping existing pipelines. | — | none |
| C1 | Lift `AudioServices/*` orchestration + `AudioEngine` device mgmt + `AudioWeightsRegistry` data → Engine. Unify AceStep/MusicGen/Yue with A. | — | GPU-gated per model, small first |
| C2 | — | Retarget `BuildEngineArgs` → typed requests; `IAudioHandler` local branch delegates to the Engine services; keep API branch + I/O. | small models e2e when GPU free |

---

## 5. Cross-repo sequencing, references, verification

### 5.1 The reference/version seam (do not break net8.0)

- **Published path:** extensions pin `<PackageReference Include="HartsyInference" Version="1.0.0-alpha.N" />` (meta-package,
  currently **alpha.50** in Extension A; B is on alpha.48). The meta includes `HartsyInference.Engine` transitively. Flow:
  Engine ships alpha.N+1 → CI logs "Publishing version:" → bump the pin. (`RestoreNoHttpCache=true` forces the live feed.)
- **Local-dev path:** `<Reference>` HintPaths into `$(HartsyRepo)/src/HartsyInference.Cli/bin/Release/**net8.0**` (the CLI
  net8.0 output = the full engine closure). **⚠ Gap:** the `UseLocalHartsy` ItemGroup lists Core/ModelHandler/…/Diffusion
  but **NOT `HartsyInference.Engine`** — add `<Reference Include="HartsyInference.Engine"><HintPath>$(HartsyLocalBin)/HartsyInference.Engine.dll</HintPath><Private>true</Private></Reference>`
  before any extension starts naming Engine types. (The DLL is already in the net8.0 closure; only the `<Reference>` entry is missing.)
- **The net8.0 build must stay green** — extensions consume the net8.0 closure, not net10.0. Every Engine-side phase ends
  with a clean net8.0 CLI build so the closure the extensions link against is valid.
- **Path note:** the csproj `HartsyRepo` default points at `../../../../SharpInference` (legacy name); this repo is
  `HartsyInference`. Local-dev builds pass `-p:HartsyRepo=…/HartsyInference` (or the default is corrected) — verify before A5/B2/C2.

### 5.2 Global ordering

```
Engine: ImageRequest/TextRequest/Speech… records (A0/B0/C0)   ← no GPU, do first, all three unblock in parallel
   ↓
Engine: typed services over existing SDXL+LLM+audio paths (A1/B0/C0)   ← no regression, no GPU
   ↓
Engine: recipe seam + per-arch/per-model lift (A2→A4, C1)   ← GPU-gated, one small model at a time
   ↓  (publish alpha or rebuild net8.0 closure after each lift)
Extension: add Engine <Reference>, bump pin, rewrite backend as mapper, delete lifted code (A5/B2/C2)
```

Recommended execution order across extensions: **B first** (one file, one interface, small-GGUF verify allowed — proves
the whole thin-wrapper pattern cheaply), then **A** (the priority payload, GPU-gated, incremental per-arch), with **C**
interleaved with A at the shared AceStep/MusicGen/Yue lift.

### 5.3 Verification plan (honoring the hard GPU/OOM rule)

- **Structural (always, every phase):** clean solution build, 0 warnings, net8.0 + net10.0; extension builds green
  against the rebuilt net8.0 closure. This validates the contract + wiring without loading weights.
- **Reason-from-parity:** each lifted recipe/resolver is *moved, proven code* — diff the lifted body against the
  extension original to confirm byte-for-byte behavior; `docs/Checklists/PARITY_VERIFICATION.md` stays the parity authority.
- **GPU-gated e2e (only when the user confirms the GPU is free, one small model at a time):** never load large diffusion
  (6–33 GB) on the shared GPU — it OOMs and crashes the live server (documented incident). Start with the smallest-on-disk
  arches (ZImage/Chroma). **Small-GGUF LLM gen on CUDA is fine** → Extension B is fully e2e-verifiable now.
- **Per-phase gate:** an extension deletes lifted code only after the corresponding Engine recipe/service is published (or
  in the rebuilt net8.0 closure) and build-verified; the backend keeps a fallback path for not-yet-lifted capabilities.

### 5.4 What this plan deliberately defers

ControlNet preprocessor lift to a Vision service (A.4 keeps them extension-side first); the OpenAI-compat DTO adapter
(maps onto the native records later); `Vc`/`Fx` service-shape finalization; and the Ideogram magic-prompt → Engine
`Text` reuse. None block the core "construct + generate" collapse.
