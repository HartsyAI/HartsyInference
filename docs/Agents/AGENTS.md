# Agent Core — Shared Context & Routing

> **Every agent conversation starts here.** Read this file first, then load the specialized agent file for your task.

## Before Any Task

1. Read `docs/CODE_STYLE.md` — mandatory, no exceptions
2. Read `docs/Design/CORE_DESIGN.md` — architecture overview
3. Check `docs/Checklists/` — find the active phase (earliest with unchecked items)

## Task Routing

Pick the specialized agent file that matches your task. Read it before starting work.

| Task | Agent File |
|---|---|
| Research a topic | `RESEARCH.md` |
| Plan an implementation | `ARCHITECT.md` |
| Write implementation code | `BUILDER.md` |
| Write SIMD/PTX/SPIR-V kernels | `KERNEL.md` |
| Write or run tests | `TESTER.md` |
| Review code | `REVIEWER.md` |
| Debug a failure | `DEBUG.md` |
| Refactor or optimize | `REFACTOR.md` |
| Build server/API endpoints | `API.md` |
| Wire cross-package integration | `INTEGRATION.md` |
| Convert model formats | `CONVERT.md` |
| Run benchmarks | `BENCHMARK.md` |
| Update docs/README | `DOCS.md` |
| Update checklists | `CHECKLIST.md` |
| Package for NuGet | `DEPLOY.md` |

If your task spans multiple agents (e.g., build + test), load both files.

## Shared Design Rules

These apply to ALL agents. Specialized files only add task-specific rules.

**Pure C# only** — no native shared libraries, no Python, no C++ wrappers, no ONNX Runtime, no managed GPU wrappers (ILGPU, ManagedCuda, ComputeSharp, Vortice).

**Eager execution** — no computation graphs; ops execute immediately.

**Zero GC on hot paths** — no managed allocations during inference. Use `NativeMemory.AlignedAlloc(byteCount, 64)`, `TensorPool` for temporaries, `ArrayPool<T>.Shared` only for managed metadata.

**IBackend abstraction** — model code never calls CPU/CUDA/Vulkan directly. Each backend delegates to static kernels.

**Package boundaries** — respect `docs/Design/NUGET_PACKAGE_DESIGN.md`. Don't leak CUDA/Vulkan into CPU packages.

## dotLLM Patterns (Single Source of Truth)

All code follows patterns verified from the dotLLM codebase. See `docs/CODE_STYLE.md` for full P/Invoke and disposal patterns.

### Tensor Type System

| Type | Owns Memory | Dispose | Use For |
|---|---|---|---|
| `Tensor` | Yes | `Interlocked.Exchange` + `AlignedFree` | Weights, intermediates |
| `TensorView` | No | No-op | Borrowed refs, mmap slices |
| `TensorRef` | No | N/A (value type) | Kernel hot paths |

**Rules:** Creator disposes `Tensor`. `TensorView` never outlives backing memory. `TensorRef` is stack-only.

### CUDA Launch Pattern
```csharp
nint outputArg = output, inputArg = input;
int nArg = n;
float epsArg = eps;
void** args = stackalloc void*[] {&outputArg, &inputArg, &nArg, &epsArg};
CudaDriverApi.cuLaunchKernel(func, grid, 1, 1, 256, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
```
Key: `stackalloc void*[]` with **local variables** for stable addresses. Never pass field refs directly.

### PTX Loading
PTX from disk via `CudaModule.LoadFromFile(path)`. Function handles as `nint` fields (not dictionary). Target `sm_80` minimum.

### Config & Options
- `ModelConfig`: class `record` with `required` + `init`
- Options: three-tier (flat props / explicit composition / custom injection)
- JSON: source-generated `[JsonSerializable]` contexts — no reflection

### Streaming
`IAsyncEnumerable<GenerationProgress>` where `GenerationProgress` is `readonly record struct`.

### Error Handling
- Shape mismatches: fail fast with `SharpInferenceException`
- CUDA/Vulkan: `.ThrowOnError()` on every call
- Compute threads: `Environment.FailFast` for unrecoverable errors
- Custom exceptions: `SharpInferenceException`, `OutOfVramException`, `UnsupportedModelException`

### Performance Attributes
| Attribute | When |
|---|---|
| `[AggressiveInlining]` | Small hot methods, tensor accessors, SIMD helpers |
| `[SkipLocalsInit]` | Large `stackalloc` paths |
| `[SuppressGCTransition]` | Short CUDA P/Invoke (< 1us) |

### GPU Weight Management
- Weights preloaded to GPU via `backend.PreloadWeights(model.EnumerateWeights())`
- After preload, CPU weight tensors can be `Dispose()`d to free RAM
- `GpuTransferHelper.CopyToDevice` checks cache by `Tensor` reference equality BEFORE accessing `DataPointer` — works on disposed CPU tensors
- Model code must NEVER access `weight.DataPointer` directly — always route through `IBackend` ops
- At pipeline stage transitions (e.g., UNet → VAE), call `backend.Sync()` + `backend.FreeWeights(model.EnumerateWeights())` to reclaim VRAM
- **Pair `PreloadWeights` with `FreeWeights` symmetrically.** If you `FreeWeights` a component at the end of a phase, also `PreloadWeights` it before the first heavy use — otherwise the first kernel pays a per-op cache-miss H2D transfer that defeats the bulk-upload optimization. Every diffusion pipeline follows this pattern; see `FluxPipeline` or `Sd3Pipeline` for the canonical placement (preload before text-encode, then again before the denoise loop). No-op on backends without a weight cache (CPU, Vulkan).
- See `docs/Research/CUDA_PERFORMANCE.md` for the full optimization roadmap

### Diffusion Pipeline Conventions
- All pipelines inherit `SharpInference.Diffusion.Pipelines.DiffusionPipelineBase` — provides `Backend` property, idempotent `Dispose` + `ThrowIfDisposed`, `DisposeCore()` hook for subclass cleanup.
- **Component ownership**: pipelines do NOT own their components (text encoders, transformers/UNets, VAE) — those are shared resources passed in by the caller. `Dispose()` on a pipeline only releases pipeline-internal state.
- **Public API shape**: pipelines expose synchronous `GenerateFromTokens` / `GenerateFromEmbeddings` / `InpaintFromTokens` / `RefineFromTokens` methods that return `(byte[] rgbData, int width, int height, int seed)` tuples and accept `Action<GenerationProgress>?` callbacks. They do NOT implement `IAsyncEnumerable<GenerationProgress>` — there is no `IDiffusionPipeline` interface (the old one was deleted because it didn't match what any pipeline actually does).
- **Shared utilities** under `SharpInference.Diffusion/Utilities/`:
  - `CfgHelper.SliceBatchElement` / `SliceBatchElement1D` / `ApplyCfg` / `ConcatLastDim` — every CFG pipeline routes through these.
  - `DtypeCastHelper.EnsureDtype` / `EnsureF32` — single source for F16/F32/BF16 activation casts. Don't write `new Tensor(shape, dt); backend.CastTo*(...)` inline.
  - `Img2ImgSetup.Prepare(request, h, w, steps)` — validates source/mask and computes `StartStep` for img2img/inpaint. Handles strength=0 pass-through detection.
  - `Schedulers/SchedulerFactory.Create(name)` — user-selectable scheduler dispatch (DDIM / DPM++ 2M / LCM / Euler default).
- **Why no `DenoiseLoopRunner`**: the per-step body varies meaningfully across pipelines (Flux streaming controller, Z-Image non-standard CFG, Lumina timestep inversion, F-Lite custom integrator, Anima Cosmos normalization, SDXL refiner step-swap). The genuine duplication has been extracted into the utilities above; the loops themselves stay inline where the model quirks live. See class-level docs on `DiffusionPipelineBase` for the full rationale.

### GPU Activation Cache Rules
- All `CudaBackend` ops call `CacheActivation(output)` to keep results on GPU
- No per-op `cuStreamSynchronize` — CUDA stream ordering guarantees correctness on a single blocking stream
- `FreeDevice` uses `cuMemFreeAsync` (stream-ordered) — memory is NOT immediately reclaimed
- **In-place ops**: When modifying a tensor's GPU buffer in-place (BroadcastAdd, etc.), clear `_gpuSyncCallback` and `_gpuDisposeCallback` to `null` BEFORE calling `CacheActivation`. Old callbacks close over the GPU pointer and will free it.
- **OOM retry**: `CudaMemory.Allocate` syncs the stream on `CUDA_ERROR_OUT_OF_MEMORY` to flush pending `FreeAsync` ops, then retries
- **Gated activations (GEGLU/SwiGLU)**: Split along last dimension, NOT at flat midpoint. See `PHASE_3_DEVIATIONS.md` #16.

### What NOT to Do
- No Python/C++ wrappers, ONNX Runtime, managed GPU wrappers
- No computation graphs — eager only
- No managed arrays for tensor data
- No `CuResult` enum, no `[DllImport]` (use `[LibraryImport]`)
- No `Dictionary<string, nint>` for CUDA handles
- No embedded PTX/SPIR-V resources — load from disk
- No reflection JSON — use source-gen contexts
- No direct `weight.DataPointer` access in model code — use `IBackend` ops (GPU cache bypass causes crashes after preload + CPU disposal)
- No flat-midpoint tensor splits in GPU kernels — always decompose index to logical coordinates for dimension-aware splitting
- No `cuMemFree` on the hot path — use `cuMemFreeAsync` (stream-ordered). Synchronous free after removing per-op Sync can cause use-after-free.
- No `CacheActivation` on in-place-modified tensors without clearing old `_gpuSyncCallback`/`_gpuDisposeCallback` first
