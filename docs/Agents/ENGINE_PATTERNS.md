# Engine patterns

The engine's own patterns for tensors, CUDA launches, config and disposal — native, not a dependency on any
external framework. LLM text generation is native too, in `HartsyInference.LLM` (config-driven generic decoder
transformer: Qwen2/Qwen3/Llama/Mistral, GGUF quantized inference, device-resident KV cache, sampler chain, chat
templates). Full P/Invoke and disposal patterns are in [code style](../CODE_STYLE.md); the design rules that
frame these patterns are in [shared architecture](AGENTS.md).

## Tensor Type System

| Type | Owns Memory | Dispose | Use For |
|---|---|---|---|
| `Tensor` | Yes | `Interlocked.Exchange` + `AlignedFree` | Weights, intermediates |
| `TensorView` | No | No-op | Borrowed refs, mmap slices |
| `TensorRef` | No | N/A (value type) | Kernel hot paths |

Creator disposes `Tensor`. `TensorView` never outlives backing memory. `TensorRef` is a non-owning readonly
record struct, not a stack-only `ref struct`; its lifetime must also stay within the storage lifetime.

## CUDA Launch Pattern
```csharp
nint outputArg = output, inputArg = input;
int nArg = n;
float epsArg = eps;
void** args = stackalloc void*[] {&outputArg, &inputArg, &nArg, &epsArg};
CudaDriverApi.cuLaunchKernel(func, grid, 1, 1, 256, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
```
`stackalloc void*[]` over **local variables** for stable addresses; never pass field refs. PTX loads from disk via
`CudaModule.LoadFromFile(path)`, function handles live in `nint` fields, `sm_80` minimum.

## Config, Streaming, Errors
- `ModelConfig` is a class `record` with `required` + `init`; options are three-tier (flat props / explicit
  composition / custom injection); JSON uses source-generated `[JsonSerializable]` contexts, no reflection.
- Streams are `IAsyncEnumerable<GenerationProgress>`, `GenerationProgress` a `readonly record struct`.
- Shape mismatches fail fast with `HartsyInferenceException`; other custom types are `OutOfVramException` and
  `UnsupportedModelException`. Check every CUDA/Vulkan native status, preserving void/handle-returning ABI
  signatures. `Environment.FailFast` is for unrecoverable process corruption, not ordinary request failures.

## Performance Attributes
| Attribute | When |
|---|---|
| `[AggressiveInlining]` | Small hot methods, tensor accessors, SIMD helpers |
| `[SkipLocalsInit]` | Large `stackalloc` paths |
| `[SuppressGCTransition]` | Short CUDA P/Invoke (< 1us) |

## Video Planning Contract
- `VideoService.PlanAsync` plans every video request before pipeline construction or weight loading. Recipe
  declarations supply family defaults; the resulting `VideoPlan` is then the authority for profile defaults,
  locked settings, supported features, component paths/formats, recipe-cache identity, release gates and the
  execution summary. Profile and composition decisions stay in `Engine/Planning`; a recipe consumes the plan and
  must not re-infer task, acceleration, attention or component roles from filenames.
- H3 semantics activate only from an exact full-file SHA-256 in the built-in manifest or an exact-hash-bound
  local profile/converter sidecar; a filename is never profile evidence. The durable hash cache is keyed by
  canonical path, byte length and last-write ticks, under `paths.modelCacheRoot/video-checkpoint-hashes` (or the
  same subdirectory below `~/.cache/hartsyinference`); verified catalog downloads seed it from their streamed
  verification, an arbitrary local file pays one visible exact-hash pass first.
- Planning errors and non-bypassable release gates return before backend probing or model construction, and
  execution rechecks planned artifact stamps, so a replaced checkpoint needs a new plan.

## GPU Weight Management
- Preload with `backend.PreloadWeights(model.EnumerateWeights())`; dispose CPU weight tensors afterwards only if
  the residency policy guarantees they are not needed for streaming, recaching or fallback.
- `GpuTransferHelper.CopyToDevice` checks its cache by `Tensor` reference equality BEFORE touching
  `DataPointer`, so it works on disposed CPU tensors. Model code never reads `weight.DataPointer` directly.
- At stage transitions (UNet → VAE) call `backend.Sync()` + `backend.FreeWeights(model.EnumerateWeights())`, and
  **pair the two symmetrically** — a component freed at the end of a phase must be preloaded again before its
  next heavy use or the first kernel pays per-op cache-miss H2D transfers. `FluxPipeline` and `Sd3Pipeline` show
  the canonical placement. Check each backend's cache and ownership; do not assume Vulkan has no weight cache.
- Open kernel/perf work is ROADMAP §2; `Research/CUDA_PERFORMANCE*.md` is the historical technique reference.

## Diffusion Pipeline Conventions
- All pipelines inherit `Diffusion.Pipelines.DiffusionPipelineBase` — `Backend` property, idempotent `Dispose` +
  `ThrowIfDisposed`, `DisposeCore()` hook. They do NOT own their components (text encoders, transformers/UNets,
  VAE): those are shared resources passed in, and `Dispose()` releases only pipeline-internal state.
- Public shape is synchronous `GenerateFromTokens` / `GenerateFromEmbeddings` / `InpaintFromTokens` /
  `RefineFromTokens` returning `(byte[] rgbData, int width, int height, int seed)` with
  `Action<GenerationProgress>?` callbacks. No `IDiffusionPipeline` interface, no `IAsyncEnumerable` pipelines.
- There is deliberately no `DenoiseLoopRunner`: the per-step body varies meaningfully (Flux streaming controller,
  Z-Image CFG, Lumina timestep inversion, F-Lite integrator, Anima Cosmos normalization, SDXL refiner step-swap).
  Shared parts already live in the utilities above; see `DiffusionPipelineBase` class docs.

## GPU Activation Cache Rules
- Every `CudaBackend` op calls `CacheActivation(output)` to keep results on GPU. No per-op
  `cuStreamSynchronize` — stream ordering guarantees correctness on a single blocking stream.
- `FreeDevice` uses stream-ordered `cuMemFreeAsync`, so memory is not reclaimed immediately;
  `CudaMemory.Allocate` syncs on `CUDA_ERROR_OUT_OF_MEMORY` to flush pending frees, then retries.
- **In-place ops** (BroadcastAdd, …): clear `_gpuSyncCallback` and `_gpuDisposeCallback` to `null` BEFORE
  `CacheActivation`; old callbacks close over the freed GPU pointer.
- **Gated activations (GEGLU/SwiGLU)** split along the last dimension, not at the flat midpoint. See
  TROUBLESHOOTING #16.
