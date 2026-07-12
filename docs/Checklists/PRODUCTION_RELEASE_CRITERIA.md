# 1.0.0-stable cut criteria

What has to be true before dropping the `-alpha.N` suffix and publishing a stable `1.0.0`. Written
2026-07-11 as part of the production-readiness plan's Phase 6 (see
[`docs/Checklists/LLM_DECODE_PERF_GRIND.md`](LLM_DECODE_PERF_GRIND.md) for the phase-by-phase technical
log this criteria list summarizes). Scope: single-GPU LLM serving — multi-GPU is out of scope for this
list (already solved elsewhere as one-backend-per-GPU, manual assignment; not part of this readiness
track).

Check off an item only once it's independently re-verified at cut time, not just "was true when written."

## Correctness

- [x] Every architecture in the benchmark table (~26, including base/non-instruct models via the
      raw-completion fallback) generates real output with zero refusals/hard errors.
- [x] Repetition penalty verified correct on both the eager sampler chain and the CUDA-graph decode path
      (the two previously had different behavior; graph decode silently ignored it until fixed).
- [x] Paged KV cache verified byte-for-byte identical to the single-sequence `FixedKvCache` it replaces,
      across prefill and multi-page-spanning decode.
- [x] Continuous batching verified byte-identical to single-sequence decode on CPU's batch-invariant GEMM
      (batching is a throughput change, not a math change).
- [x] JSON-mode constrained decoding verified against real generation with two independent checks (unit
      tests on the grammar state machine; live model output independently parsed with a real JSON parser).
- [ ] **Gap**: no automated sweep re-validates ALL ~26 architectures after each phase of this work — Phases
      1-6 were spot-checked against 3-4 representative architectures (qwen2, qwen3, llama) per change, not
      the full matrix. Before cutting 1.0.0, run the full architecture benchmark once more end-to-end and
      diff against the last recorded pass. **Partial progress**: extending the sweep to gemma-3 (wider KV
      heads, fewer layers than the models tested so far) caught a real bug — the KV pool's VRAM footprint
      was sized off a FIXED page count tuned against a narrow-KV-dim model, which OOM'd on a wider one. Fixed
      (see `LLM_DECODE_PERF_GRIND.md` 2026-07-11) with a byte-budget-aware page count instead, locked in with
      unit tests reproducing the exact shape difference. This is the kind of gap the missing full sweep is
      FOR — worth treating as a concrete argument for actually running it before cut, not just a checkbox.

## Resiliency & operations

- [x] **Decode-round fault isolation.** Found 2026-07-11 via the architecture sweep: an exception escaping
      `DynamicBatchScheduler`'s decode round (native kernel bug, OOM, anything) previously killed that
      model's entire background loop silently — no crash, no log line, every future request to that model
      just hung forever (nothing was left reading the admission channel). Fixed: the round is now wrapped
      in a try/catch that fails only the sequences in that round, logs the exception, frees their KV pages,
      and lets the loop keep serving everything else. Admission failures got the same logging. A fault that
      somehow still escapes both is now at least logged via a continuation on the loop's own `Task`, instead
      of vanishing as an unobserved task exception.
- [x] **Real `/ready` check.** Was an unconditional 200 with zero dependency on server state — a model
      whose scheduler loop had died looked identical to a healthy one. Now checks
      `ModelManager.UnhealthyChatModels` (any loaded model whose `DynamicBatchScheduler.IsLoopAlive` is
      false) and returns 503 with the affected model ids. `/health` deliberately stays a cheap, dependency-free
      liveness check (standard k8s/systemd split — a real check there would make an orchestrator kill a
      process that's actually fine during a transient hiccup).
- [x] **Global exception-handler middleware** added so any route bug not already caught (chat/images had
      their own try/catch; nothing else did) returns a structured, logged 500 instead of a bare response.
- [x] **429 disambiguation**: KV-pool-exhausted (`insufficient_capacity`, a per-model capacity limit — retrying
      against a different model would work) is now distinguished from queue-full (`rate_limit_error`, a
      global limit) — previously identical error `type`, making them indistinguishable from the HTTP response.
- [x] **Time-to-first-token now logged** per chat completion (both streaming and non-streaming), not just
      total elapsed — separates prefill/queueing latency from decode throughput in the one log line ops has.
- [x] **Process supervision**: `deploy/systemd/hartsyinference-server.service` (Restart=always) and
      `deploy/run-with-restart.sh` (non-systemd fallback) added — see `deploy/README.md`. This exists because
      of a hard CLR limit, not a bug we can fix: `AccessViolationException` and other corrupted-state
      exceptions from native/unsafe code (this engine has plenty) are **uncatchable** in .NET Core — the
      process terminates before any handler runs, including the exception middleware above. Verified live
      2026-07-11: a CPU-backend MoE kernel bug crashed the whole process (confirmed via `curl` connection-refused
      immediately after); there was no restart mechanism of any kind in the repo at the time.
- [ ] **Gap, not yet addressed**: no graceful shutdown / request draining. `ModelManager.Dispose()` fires on
      SIGTERM via normal DI disposal and immediately faults every in-flight request with
      `ObjectDisposedException` rather than letting them finish or giving the load balancer time to stop
      routing new traffic first.
- [ ] **Gap, not yet addressed**: no per-request timeout or max-request-size enforcement (a client — or a
      hung generation — can hold a concurrency slot indefinitely); no `/metrics` scrape endpoint (structured
      log lines only, no counters/histograms, no aggregation without a log pipeline).

## Concurrency & serving

- [x] Thread-safety audited for the shared `IBackend` instance across diffusion + LLM chat traffic (one
      CUDA stream, non-thread-safe caches) — confirmed unsafe if naively concurrent, fixed via a shared
      exclusivity gate that still allows chat requests to batch together.
- [x] Real concurrent-load stress test: 3 waves of 10 concurrent chat requests + a mid-wave cancellation,
      live server, real model — 30/30 succeeded, zero cross-contamination between batched sequences.
- [ ] **Gap**: no sustained SOAK test (hours, not seconds) — the stress test above proves correctness under
      burst concurrency, not long-running stability (slow memory/page leaks, connection handling under a
      persistent high request rate). Worth a multi-hour run against a real workload generator before cut.
- [ ] **Gap**: no documented behavior/test for what happens when the KV pool is chronically undersized for
      the actual traffic pattern (repeated `KvPoolExhaustedException` under real load) — the exhaustion
      path itself is unit-tested, but not "what does an operator see/do about it in practice."
- [x] **CUDA-graph decode retrofitted into `DynamicBatchScheduler`, 2026-07-11.** Was wired into
      `TextGenerationPipeline.Generate` only (Phase 6 above), meaning production server traffic got none of
      the measured speedup. A request admitted while the scheduler is otherwise idle now gets a dedicated
      `FixedKvCache` and captures a graph once at admission, falling back permanently ("one-way retirement")
      to the existing eager path the moment a second request arrives while it's running. Full design in
      `LLM_DECODE_PERF_GRIND.md`'s "Phase 6b" section and `~/.claude/plans/nested-zooming-tower.md`. Verified
      byte-identical output (including a transition test that forces a graph→eager mid-generation splice) and
      a real **~32% server-side speedup** on a 500-token completion (111.0 → 146.6 tok/s, live
      `/v1/chat/completions`, not a microbenchmark). Found and fixed a genuine pre-existing bug along the way:
      a cold model's first-ever graph capture failed with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` (weight
      auto-promotion during capture) — this affected the original CLI-only feature too, just never caught
      since every prior test happened to warm up the backend with an eager call first.
- [ ] **Gap, remains out of scope**: speculative decoding (Phase 5b, also shipped and verified — see
      `LLM_DECODE_PERF_GRIND.md`) is still wired into `TextGenerationPipeline.Generate` only. Batching it
      across a dynamically-changing multi-sequence batch (variable per-sequence draft lengths reshape the
      batch every round) is a harder problem than the graph-decode retrofit above and remains real, scoped,
      follow-up work — not attempted in this pass.
- [x] **Real efficiency win found investigating the above**: `PagedKvCache.Gather`'s scratch buffer was
      reallocating a full GPU tensor on EVERY decode round for EVERY active sequence (its size check was
      "changed at all" instead of "too small," and `_physicallyWritten` — hence the required size — grows by
      exactly 1 every decode step). Fixed to grow-only, rounded to a page boundary (mirrors
      `FixedKvCache`'s already-proven oversized-buffer-plus-explicit-valid-length contract, which
      `GenericTransformer`'s attention call already honors) — cuts scratch reallocation frequency by
      `KvPageSize`× (16 by default) for every sequence going through the actual production server path
      today, at the cost of at most one page's worth of unused VRAM per sequence per layer. Verified: paged
      vs. fixed-cache parity test and the 300-round admit/evict stress test both still pass (one assertion
      updated — it had asserted the OLD exact-sizing as a correctness invariant, which was never actually
      required by any consumer, same as `FixedKvCache` already doesn't provide it).

## API stability

- [ ] **Not yet met.** This release cycle added a new scheduler abstraction (`IBatchScheduler`), a new KV
      cache implementation (`PagedKvCache`), new `IBackend` methods (`SliceTimeRange`, repetition-penalty
      device ops), and new `SamplingOptions`/DTO fields (`JsonMode`, `response_format`). The public NuGet
      surface is still actively evolving. 1.0.0 should follow at least one alpha cycle with NO further
      breaking changes to `IBackend`, `GenerationRequest`/`GenerationResult`, or the `/v1/*` HTTP shapes.

## Documentation

- [x] README, `LLM_MODEL_COVERAGE.md`, `MODEL_STATUS_LLM.md` reflect current supported architectures.
- [x] `LLM_DECODE_PERF_GRIND.md` and `QUANT_GEMM_PERF_PLAN.md` record every measured perf claim with a
      real number and a gate, not just "should be faster."
- [x] `CHANGELOG.md` exists and is kept current per release.
- [ ] **Gap**: no public-facing API reference/quickstart specifically for `HartsyInference.Server`'s HTTP
      API (the `/v1/chat/completions` shape, `response_format`, streaming) — currently only documented
      inline in code comments and this checklist family. Worth a short standalone doc before 1.0.0 since
      external consumers of the server won't read the engine's internal perf-grind docs.

## Versioning & release process

- [x] Single source of truth for the version (`Directory.Build.props`), CI auto-publishes all sub-packages
      on push via OIDC trusted publishing with skip-duplicate (safe to push without a version bump).
- [x] `CHANGELOG.md` introduced this cycle — keep it current per bump from here on.
- [ ] **Gap**: no automated check that the SwarmUI extension's pinned `HartsyInference` NuGet version is
      resynced after each engine publish — currently a manual step (bump engine, push, wait for CI publish,
      THEN bump the extension's pin) with no drift detector if it's forgotten.

## What 1.0.0 does NOT need (explicitly out of scope for this cut)

- Multi-GPU / tensor-parallel serving — already solved elsewhere (manual one-backend-per-GPU), not part of
  this readiness track.
- Prefix/prompt caching, `json_schema`-constrained decoding — explicitly deferred as separate, larger
  follow-up features (see `CHANGELOG.md`'s "Deferred" section). Neither blocks a stable cut on its own;
  they're roadmap items for a `1.1.0`-class release, not correctness/stability gaps.
- Speculative decoding is DONE and verified (byte-identical, ~35% measured speedup on repetitive content) —
  see `LLM_DECODE_PERF_GRIND.md` — but only on `TextGenerationPipeline.Generate`, not the server's batched
  path (tracked as a gap above, not "out of scope"; listed here only to correct the record from an earlier
  draft of this checklist that called it deferred).
- Wider quant kernel coverage (Q2_K/Q3_K/IQx) — falls back to a working-but-slower CPU dequant path today;
  a performance gap, not a correctness one.
