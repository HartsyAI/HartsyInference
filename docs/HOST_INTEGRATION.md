# Embedding HartsyInference in a host application

The engine is a set of libraries. A host (SwarmUI's extension, a service, a desktop app) drives it through the entry
points below; everything else in `HartsyInference.Core` is engine-internal and may change between alpha releases without
notice. Members in this table carry "Public for host applications" in their summary so a reader finds this page from the
code.

## Backend selection (`HartsyInference.Engine.BackendFactory`)

| Member | Use |
|---|---|
| `Create(selector)` | Builds the backend for `auto`, `cpu`, `cuda[:n]` or `vulkan[:n]`. `auto` prefers CUDA, then Vulkan, then CPU. |
| `Resolve(selector)` / `ResolveProbed(selector)` | What `auto` would pick, without or with a kernel probe (a small matmul checked against the CPU). |
| `ProbeCuda(ordinal)` / `ProbeVulkan(ordinal)` | Run the probe; `CudaProbeFailureReason` / `VulkanProbeFailureReason` say why it failed. |
| `CudaUnavailableReason` / `VulkanUnavailableReason` | Why `Resolve` passed over a device; null when it was usable. |
| `ValidSelectors`, `Validate`, `IsValidSelector`, `Kind`, `ParseOrdinal`, `WithOrdinal`, `CanonicalDeviceKey`, `Describe` | Selector parsing and display for a host's own settings UI. |
| `KernelDirOverride`, `KernelDir`, `PtxDirName`, `SpirvDirName` | Where compiled kernels live when the host deploys them away from the executable. |

## Logging (`HartsyInference.Core.Logging.Logs`)

| Member | Use |
|---|---|
| `SetLogger(Action<LogLevel, string>)` | Routes every engine log line into the host's logging framework. Unset, lines go to standard error. |
| `MinLevel` | Minimum level the engine emits. `LogLevel` orders like SwarmUI's levels so a host forwards without remapping. |

## Settings (`HartsyInference.Core.Configuration`)

| Member | Use |
|---|---|
| `KnobFile.ExplicitPath` | The settings file a host keeps elsewhere. Set it before the first setting is read. |
| `KnobFile.Reload()`, `Save(id, value)`, `Apply(json)` | Re-read, persist one setting, or apply inline JSON. |
| `KnobStore.Set(knob, value)` / `Clear(knob)` / `SourceOf(id)` | Process-wide overrides and where a value came from. |
| `KnobProfiles.ByName`, `Names`, `Default`, `Reference` | The named profiles a host can offer. |
| `RequestSettings` | Per-request `profile` + `set` as they arrive over the API. `Resolve()` throws, naming the ids, when a request would change a `Construction`-scope setting from the value the engine was built with; see [SETTINGS.md](SETTINGS.md). |
| `EngineKnobs.*` | Every setting, declared once with id, default, scope and domain; `KnobRegistry` lists them. |

## VRAM policy (`HartsyInference.Core.Memory.Vram`)

| Member | Use |
|---|---|
| `VramPolicy.For(tier)` | The expanded lever set of a `VramTier`, what a backend runs before any per-request override. |
| `VramOverrides` | Per-request lever overrides; every member nullable, null means "follow the backend's policy". |
| `LowVramPolicy.SetOverride(backend, mode)` / `ClearOverride(backend)` | Pin or release a backend's `LowVramMode` regardless of the process-wide `vram.lowVram` setting. |
| `VramGraphGuard.CanMoveMemoryFreely(backend)` | Whether a captured step graph is holding device pointers, so an optional lever may move memory now. |

## Model folders (`HartsyInference.Core.Configuration.ModelFolders`)

| Member | Use |
|---|---|
| `ModelFolders.MapName(name)` / `MapSubdir(relative)` / `Spellings(relative)` | Map a catalog folder name (any spelling, any case) to its name on disk, so a host that already has a models tree keeps one folder per kind. |
| `ModelFolders.Under(modelsRoot, folder, ...)` / `Physical(folder)` | Build paths under the models root honoring the per-folder overrides. |

## Diagnostics

| Member | Use |
|---|---|
| `IBackend.Capabilities` (`BackendCapabilities`) | What the backend supports: dtypes, quantized formats, SDPA, STFT, device name and vendor. |
| `HostMemoryInfo.AvailableBytes()` | Available host RAM, cross-platform, best effort. |
| `DeviceKind.IsCpu` / `IsCuda` / `IsVulkan` / `IsGpu` | Device family of a tensor or backend. |

## 3D preflight (`IInferenceEngine.MeshPlanning`, `POST /v1/native/mesh/plan`)

| Member | Use |
|---|---|
| `DescribeAll()` / `Describe(id)` (`GET /v1/native/mesh/describe[/{id}]`) | Per 3D model: input and output kinds, how it treats each tunable `MeshRequest` field (access, default, legal values or range, unit), licence and gated repositories, cost. A host builds its form from this. |
| `PlanAsync(spec, request)` | The validation generation runs, the settings with their provenance, ignored-field warnings, the file inventory and a warning per absent file, and whether the recorded peak fits the card (verdict `Unknown` with the reason when none was recorded); nothing loads. A plan with an error issue answers `422` over HTTP. |
| `MeshResult.Info`, `MeshRequest.PreviewTriangles`, `MeshRequest.Frame` | Every glb or ply result carries `Info`: size, triangle and vertex counts (or the splat count), bounds, and for a mesh the components, boundary and non-manifold edges and watertightness of the welded surface, plus normals, texture, skin and animation names. `PreviewTriangles > 0` also returns `Preview`, a decimated geometry-only glb in the same frame (null when the mesh is already within the budget). `Frame = GltfYUpMetres` adds one root node that centres the scene and scales its longest side to 1 m, and turns Pixal3D a half turn to glTF's +Z front; every other model keeps its front and `Warnings` says so, because no model's real front direction has been checked on output. Measuring welds the mesh, which costs seconds on a multi-million-triangle result. |
| `MeshLifecycle.Loaded()` / `Free(name)` (`GET /v1/native/mesh/loaded`, `POST /v1/native/mesh/free`) | The 3D pipelines resident now (kind and checkpoint name, never an absolute path) and freeing one by name or all, so a host can give the card back after a job (3D models do not unload on their own; `FreeMemory()` frees everything including these). A generation running on a pipeline finishes before its disposal. A generation that fails with `OutOfVramException` frees the resident 3D models and the matting model, trims the driver pool (not the backend's weight caches, which cached image pipelines still rely on), reports a `Retrying after out of memory` stage (progress restarts from 0) and runs once more; a second failure propagates as the typed exception, which the HTTP routes map as before. A free or a disposal waits for a run in progress on every kind of pipeline. For the image, scene, rig, paint and part pipelines the wait is service-wide: it covers every running generation of those, and holds new ones of those off (waiting, not blocking a thread, and cancellable) until the free is done, so `POST /v1/native/mesh/free` can take as long as the longest running generation; a host behind a proxy should allow for that. |
| `MeshConversion.Convert(data, source, target, name)` (`POST /v1/native/mesh/convert`, `GET /v1/native/mesh/formats`) | Converts a model file between `glb` and `obj` (read) and `glb`, `obj`, `ply`, `stl` (write), whichever 3D model made it: it works on the file, so a model added later needs no change here. An OBJ comes with its MTL and its texture images (base colour, normal and emissive maps have MTL slots; metallic-roughness and occlusion maps do not, and are reported as left out), and `Archive()` (or `archive: true` over HTTP) packs the files into one zip. PLY is binary with normals and vertex colours, STL is binary geometry. A skin and animations never survive: the output is the bind pose, `StaticOnly` is true and the notes say so, as they say every other loss (textures into PLY or STL, vertex colours into OBJ, primitives in strip, fan, line or point mode, textures that are not embedded PNG or JPEG, a second UV set). A primitive without an index accessor is read as a triangle soup. A Gaussian-splat PLY (the TRELLIS and SAM 3D output) is not a mesh and is refused with that reason. The result's `asset.extras` (the settings block) is carried into a glb output. Unknown formats and malformed files are HTTP 400. Extension points: a new output format is one `IMeshExporter` class in `HartsyInference.ThreeD.Export` plus one line in `MeshExporters.All`; a new input format is one case in `MeshDocumentReader.Read`; both work on `MeshDocument` (primitives in world space, materials with their embedded images, notes), which the post-processing and thumbnail services share. |
| `StepPreview.Stage`, `StageIndex`, `StageCount`, `OverallFraction` (and the same fields on the SSE `progress` event of `/v1/native/mesh/stream`) | Stage-aware 3D progress: `Step` and `TotalSteps` keep their meaning (one counter across the whole run), and a staged model (TRELLIS.2, Pixal3D, SAM 3D, HairDeck) also names the stage, its 1-based position and the stage count (the stages a run reports, nothing it skips). For TRELLIS.2 and Pixal3D `OverallFraction` weighs each stage by how long it usually takes, so it does not reset between stages and a bar driven by it moves at the pace the user waits at; the weights are approximations from the recorded timings, and SAM 3D, HairDeck and any model that names no stage report the plain step ratio. HairDeck reports its preprocessing, card sampler and chain stages and honours the cancellation token in every sampler step; a SAM 3 or BiRefNet pass in progress finishes first. |
| `MeshResult.Settings` (`MeshSettingsUsed`) and the embedded description | The settings a result was made with: model id, the resolved seed (a negative request seed is drawn by the engine, so it is known afterwards; null for TripoSR and CoLa3D, which have no seed), every setting the model reads with its value and whether the caller or the default chose it, and the engine version. The same description is embedded in the file: a glb carries it in `asset.extras.hartsyinference`, a ply in a `comment hartsyinference` header line, so a file kept outside the host still says how it was made. Hunyuan3D, TRELLIS, TripoSR and UniRig still cast a seed above `int.MaxValue` to `int` unchecked; `Settings.Seed` reports the low 32 bits they run with as an unsigned value, so sending it back reproduces the run. |
| `MeshPreparation.PrepareImageAsync(spec, image)` (`POST /v1/native/mesh/prepare`) and `MeshRequest.Preprocess` | The image a model will see: an image without alpha is matted (BiRefNet), then Hunyuan3D recentres it on white at 512 px, TripoSR crops and composites it on gray, TRELLIS (v1) crops to 518 px and premultiplies alpha, Paint takes the cutout. The result carries the cutout so a user can correct the matte and send it back. `Preprocess = Auto` in a request does the same inside the generation; the default `None` leaves the image as given. TRELLIS.2, Pixal3D, HairDeck and SAM 3D prepare their own input and return the image unchanged. |
| `MeshAssets.GetStatus(spec)` (`GET /v1/native/mesh/assets/{id}`) | Each file a 3D model needs with its role, catalog-relative path, whether it is required, present (found where the loader looks, legacy folders and SAM 3D's `.ckpt` included) and its size, and a typed blocker for what the engine cannot fetch: `Gated` (needs `HF_TOKEN`), `ManualPlacement` (Pixal3D's `naf_release.pth`), `NoCatalogEntry` (TripoSR, TRELLIS, CoLa3D and UniRig have no catalog files). |
| `ModelPrefetch.PrefetchAsync(spec)` for a mesh spec (`POST /v1/native/mesh/prefetch`) | Fetches the absent, fetchable catalog files with progress, the primary file last; the files it cannot fetch come back in `ModelPrefetchResult.Blockers` instead of an exception. |
| `IMemoryEstimationService` for a mesh spec | The recorded whole-card peak of the default request as a one-stage estimate; a model with no recorded peak throws `NotSupportedException` from `EstimateAsync` and answers `Unknown` from `AssessAsync`. |

## Realtime audio primitives (`HartsyInference.Core.Runtime`)

| Member | Use |
|---|---|
| `MonotonicClock.NowNs()` / `SleepUntil(deadlineNs)` | A clock that never jumps backwards and an absolute-deadline sleep for a thread that keeps a fixed cadence. |
| `SpscRing<T>` | Lock-free ring for one producer thread and one consumer thread, with a dropped-sample counter. |

## What is not an entry point

Backend operations (`IBackend` members), the `*Reference` host implementations, the expert cache, the attention planner
and the model classes are engine internals even where they are `public` for cross-package use. A host that needs one of
them has found a missing entry point: ask for it here rather than depend on the internal.
