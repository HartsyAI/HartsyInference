# Changelog

All notable changes to HartsyInference are recorded here. Versions follow `2.0.0-alpha.N` (the scheme moved
up from `1.0.0-alpha.N`; entries below that pre-date the change and keep their original numbers). The single
source of truth is `<VersionPrefix>`/`<VersionSuffix>` in `Directory.Build.props` — see
[`docs/Checklists/PRODUCTION_RELEASE_CRITERIA.md`](docs/Checklists/PRODUCTION_RELEASE_CRITERIA.md) for what a
stable release will require. Dates are UTC.

## [2.0.0-alpha.5] — 2026-07-27

Low-VRAM generation, a GPU-memory leak fix, and selectable devices. Full technical detail, with all
measurements and the things that turned out **not** to be true, in
[`benchmarks/results/2026-07-27_lowvram_leak_fix.md`](benchmarks/results/2026-07-27_lowvram_leak_fix.md).

### Added
- **Low-VRAM weight streaming across the image fleet** (`HARTSY_LOWVRAM`, three-state: `auto` default /
  `on` / `off`). The sliding-window machinery (`BlockStreamingController`) already existed but only one of
  ~25 image pipelines used it. **Four models that could not run on a 12 GB card now do**, all 1024²,
  quality-gate clean: **HunyuanImage-2.1** (19.7 s), **Ideogram 4** (205 s — a *pair* of 9.3 GB DiTs
  needing 19.7 GB against 9.2 GB available), **Qwen-Image** (231 s, 20B MMDiT), **Krea2** (71 s).
  `off` is a real escape hatch: the same request succeeds under `on` and raises `OutOfVramException`
  under `off`.
- **`VramPlanner`** — one place that decides resident-vs-streamed per generation phase, carries the
  `HARTSY_KEEP_MODELS` residency short-circuit, and logs the **weights-vs-activations split** (streaming
  can only move the weight term, so a phase dominated by activations needs a smaller working set, not a
  sliding window).
- **Selectable CUDA device**: `cuda:1`-style backend selectors, `InferenceEngine(selector, ordinal)`, and
  a real `GPU_ID` in the SwarmUI backend — previously logged and ignored. Verified by memory delta.
  Note the ordinal is CUDA's (fastest-first), which need not match `nvidia-smi`'s PCI order.
- **SD3.5 modular component loading** — CLIP-L / CLIP-G / T5-XXL / VAE each resolve independently when the
  checkpoint does not bundle them, which is the standard SD3.x distribution format. SD3.5-Medium now
  generates end-to-end; it previously threw before any sampling.

### Fixed
- **GPU memory leak on OOM.** `CudaBackend.PreloadWeights` had no exception path, so a mid-load OOM left
  already-uploaded weights registered against a model that would never finish — unreachable, therefore
  unfreeable. The process held ~11.5 GB with nothing running and **starved other processes on the same
  card**, including a separate ComfyUI. Now: typed `OutOfVramException`, per-batch rollback, and reclaim at
  both the generate and construct boundaries. An OOM'd process now holds **152 MiB instead of ~11.5 GB**,
  and a sequential multi-model sweep survives an OOM (3/3 models succeeded after one).
- **Streaming was inert for every GGUF model.** `DType.Q4_K.SizeInBytes` is 0 (a K-quant has no
  per-element size), so `ElementCount * SizeInBytes` totalled block weights to **zero bytes** and the
  "fits resident?" test was always trivially true. Fixed in four block implementations.
- **Lens rendered solid black** (16/16). SageAttention's INT8 path materializes V as F16; Lens does not
  RMS-norm V, and `max|V|` crossed F16's 65504 mid-generation. Verified against ComfyUI's own reference
  implementation on the same checkpoint — an engine bug, not a port bug.
- **Anima was 19-63× slower than ComfyUI**: 792 host round-trips per denoise step (14 per block × 28
  blocks × 2 CFG passes). Now **3**. Warm step 15,279 ms → 519 ms. Its documented "1024² hangs" was never
  a hang.
- **The VRAM planner under-reported free memory by ~4.6 GB**, because `cuMemGetInfo` counts the
  stream-ordered pool's reservations as used. The error is asymmetric — it biases toward streaming, which
  costs 5-8× — so a large card could silently take the slow path for a model that fits.
- `TextService.PrimaryDeviceKey()` hardcoded `"cuda:0"`, so a `cuda:1` engine would have rendered images on
  one GPU while its LLM landed on another.
- Lumina2's on-disk checkpoint was the wrong variant (`cap_embedder.*` naming vs the diffusers
  `time_caption_embed.*` the converter expects). Correct weights now load and generate — though this
  revealed a **separate, previously unreachable conditioning bug**: output is coherent but off-prompt.

### Changed
- `Chroma` checkpoint conversion is now streaming per tensor (removes a GC-timing dependence from the
  peak). **The documented "host RAM OOM" does not reproduce** — it peaks at 9.1 GB anon and completes;
  the reported 25 GB was total RSS including reclaimable file-backed page cache.

## [1.0.0-alpha.48]

Production-readiness push: closes the throughput gap toward python inference stacks (vLLM/TGI-class) and
adds the serving infrastructure a real deployment needs. Full technical detail in
[`docs/Checklists/LLM_DECODE_PERF_GRIND.md`](docs/Checklists/LLM_DECODE_PERF_GRIND.md)'s dated status
updates; this is the release-notes-level summary.

### Added
- **Fused GEMV kernels for Q4_0 and Q5_K** quantization formats — the last two of the six original
  quant types without a fused decode kernel; both previously fell to the ~10-20x-slower
  dequant-to-F16-then-cuBLAS path.
- **On-device repetition penalty for CUDA-graph decode.** Graph decode was previously greedy-only with a
  raw unpenalized argmax — a request with `RepetitionPenalty > 1.0` and graph decode enabled silently
  ignored the penalty. Fixed with two new device-resident kernels chained into the existing captured graph.
- **`/v1/chat/completions`** (OpenAI-compatible, streaming and non-streaming) on `HartsyInference.Server` —
  the server previously had no LLM chat endpoint at all (image generation only). Includes structured
  request logging (queue depth, prompt/completion tokens, latency, tokens/sec) and real cancellation that
  stops in-flight generation, not just the HTTP connection.
- **Paged KV cache** (`PagedKvPool`/`PagedKvCache`) — replaces the single-sequence `FixedKvCache` (hard
  `batch=1` restriction) with pages allocated on demand from a pool shared across sequences.
- **True continuous batching** (`DynamicBatchScheduler`/`IBatchScheduler`) — requests admit dynamically at
  any time and batch together into shared decode rounds; each sequence evicts the instant it
  finishes/stops/cancels. Replaces the old static-batch `ContinuousBatchScheduler` (fixed request list up
  front, zero production callers, removed). Backend-exclusivity is preserved via an injected gate so LLM
  batching never races with diffusion image generation on the shared GPU backend instance.
- **JSON-mode constrained decoding** (`response_format: {"type":"json_object"}`) — masks every candidate
  token so generation can only produce syntactically valid JSON. The richer `json_schema` mode is not
  implemented and is rejected with a clear 400 rather than silently ignored.
- Server integration test suite (`ChatCompletionsIntegrationTests`, in-process via `WebApplicationFactory`)
  covering chat-completions request validation — previously zero automated coverage on this HTTP surface.

### Changed
- `IBackend.SliceTimeRange` — new primitive (host default + CUDA kernel) extracting a contiguous
  time-range from a KV-shaped tensor; used by the paged KV cache.
- `GenericTransformer.ForwardBatchDecode`'s cache parameter widened from `FixedKvCache[]` to `IKvCache[]`.
- Chat-completions request validation now checks pure request-shape issues (empty messages, unsupported
  `response_format`) before consulting server state (is the model loaded) — fails fast on a malformed
  request regardless of what's currently loaded.

### Fixed
- Two real bugs in the new JSON-grammar state machine, both caught by unit tests before ever touching a
  live model: object keys didn't set the post-string parse transition (would have broken any JSON with a
  key — i.e. almost all real JSON); the state's `Clone()` was missing two fields added after it was first
  written (every candidate-token check clones the state, so this would have corrupted the container stack
  on every single trial in production).
- `ModelManager`'s diffusion-vs-LLM checkpoint routing no longer speculatively attempts the LLM loader on
  an unrecognized GGUF — a prior version of this logic (try-LLM-then-catch-fallback) fully materialized a
  multi-GB diffusion checkpoint's tensors before the fallback path could fire, causing a real OOM.
- Paged KV cache's VRAM footprint is now sized from a configurable byte budget
  (`HartsyInferenceServerOptions.KvPoolBytesBudget`, default 512MB) scaled to each loaded model's actual KV
  dimensions, replacing a fixed page count that comfortably fit a narrow-KV-dim model but eagerly
  pre-allocated several GB for a wider one — caught loading gemma-3 during a broader architecture sweep.

### Deferred (explicitly, not attempted)
- Prefix/prompt caching (share identical-content KV pages across sequences) — real additional scope (page
  reference-counting, prefix hashing, copy-on-write on divergence).
- Speculative decoding — a true stretch item, orthogonal to everything else in this release.
- `json_schema`-constrained decoding (schema-aware, not just syntax-valid JSON).
- Wider quant kernel coverage (Q2_K/Q3_K/IQx formats) — no template to adapt from, genuinely new kernel
  design (lookup-table dequant for IQx specifically).

## [1.0.0-alpha.47] and earlier

Not individually itemized here — see `git log` for the full history prior to this changelog's introduction.
