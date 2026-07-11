# Changelog

All notable changes to HartsyInference are recorded here. Versions follow `1.0.0-alpha.N` during the
pre-1.0 phase (see [`docs/Checklists/PRODUCTION_RELEASE_CRITERIA.md`](docs/Checklists/PRODUCTION_RELEASE_CRITERIA.md)
for what "1.0.0" itself will require). Dates are UTC.

## [Unreleased] — 1.0.0-alpha.48

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
