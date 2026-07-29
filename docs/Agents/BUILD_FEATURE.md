# Build-a-Feature Agent

> Build a new **non-model** feature: engine service, CLI command, HTTP API route, SwarmUI-extension
> surface, or cross-package plumbing. Assumes you've read `AGENTS.md` + `docs/CODE_STYLE.md`.

## The one rule that matters most

`HartsyInference.Engine` owns load + generate. The CLI, HTTP API, and SwarmUI extension are **thin
wrappers over `IInferenceEngine`** — never re-implement orchestration in a consumer.

```csharp
// ✅ a consumer resolves a model + calls the facade
IInferenceEngine engine = ...;
await engine.GenerateImageAsync(spec, request, onProgress);
// ❌ re-deriving load/generate in the consumer (the API rewrite existed to kill exactly this —
//    it used to hard-cast a pre-facade ModelManager to SDXL-only)
var mgr = new ModelManager(); var sdxl = (SdxlPipeline)mgr.Load(path); sdxl.Run(...);
```

## Public API is a contract

The SwarmUI backend extension pins a **published** engine version, so a renamed/broken public signature is
invisible until it's republished *and* re-pinned. Treat signatures as append-only.

```csharp
// ✅ add a new overload; old callers keep compiling
public Task<Result> GenerateImageAsync(ImageSpec spec, ImageRequest req, Action<GenerationProgress>? cb);
public Task<Result> GenerateImageAsync(ImageSpec spec, ImageRequest req, LoraStack loras, Action<GenerationProgress>? cb);
// ❌ changing an existing public signature in place (silently breaks the pinned extension)
```

```csharp
// ✅ public entry points take IBackend, never a concrete backend
public sealed class FooService(IBackend backend) { }
// ❌ public void Configure(CudaBackend backend)   // leaks GPU into the surface
```

## Library surface = callbacks; API = the one transport

- Library methods report progress via `Action<GenerationProgress>?` (diffusion) / `Action<int>? onToken`
  (LLM). **Do not** put SSE/HTTP/JSON plumbing in a library package.
- `src/HartsyInference.API` is the single sanctioned transport layer — it wraps the Engine facade's
  `IProgress<>` / `IAsyncEnumerable<>` into SSE. One endpoint file per modality under `Endpoints/`; native
  routes are the primary contract (`{model, modelPath, request}` envelope), OpenAI-compat routes call the
  *same* handlers. Two `InferenceQueue` gates (default + `"long-running"` for video/world) so a multi-minute
  job can't starve fast requests.

## Cross-package boundary checklist

Shapes match at each boundary · dtypes match (FP32 into an FP16 kernel = garbage) · `DeviceKind` consistent,
cross-device only via `IBackend.CopyTo()` · memory ownership explicit, `TensorView` never outlives its
backing `Tensor` · `CancellationToken` threaded end-to-end · source-generated `[JsonSerializable]` at every
serialization boundary (no reflection JSON).

## Don't forget

- Keep `src/HartsyInference.Cli` and `samples/` compiling — they double as the usage examples for the API
  you just changed.
- Memory discipline holds here too: `NativeMemory.AlignedAlloc(bytes, 64)` / `TensorPool` for temporaries,
  `.ThrowOnError()` on every native call, `Environment.FailFast` in compute workers. (See CODE_STYLE.md.)
