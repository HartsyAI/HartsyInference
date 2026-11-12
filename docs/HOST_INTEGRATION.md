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
| `PlanAsync(spec, request)` | The validation generation runs, the settings with their provenance, ignored-field warnings, catalog files not at their catalog path, and whether the recorded peak fits the card; nothing loads. A plan with an error issue answers `422` over HTTP. |
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
