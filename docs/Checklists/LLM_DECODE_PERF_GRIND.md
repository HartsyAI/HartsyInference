# LLM Decode Performance Grind — closing the gap to llama.cpp

Status legend: ⬜ todo · 🔧 in progress · ✅ done · 📊 measured

**Goal.** Get HartsyInference LLM single-token decode from **20-54× slower than llama.cpp** to within a small factor (target: ≥50% of llama.cpp t/s, i.e. ≤2× slower) on the RTX 3060, same GGUF files and params as `LLM_THROUGHPUT_BENCHMARK.md`. Top priority.

> **STATUS (2026-07-04): gap closed 20-54× → 1.94-2.88×.** Llama-3.2-1B **1.94× (under 2×)**, Mistral-7B 2.12×, Qwen3 2.26×, Gemma-3 2.88×. All verified coherent, default path is the shipped state. Phases 1-4 done (fused Q4_K/Q6_K/Q8_0 GEMV, quantized lm_head, split-K flash-decode attention, vectorized loads). Phase 5 (dp4a) built but no gain (memory-bound). Phase 6 (CUDA graphs) foundation verified, full device-resident build deferred (multi-session). Progress table + per-phase detail below.

> **STATUS UPDATE (2026-07-10): Phase 6 (CUDA graphs) DONE for the plain dense GQA/RoPE decoder shape** (Llama/
> Qwen2/Qwen3/Mistral — opt-in `HARTSY_GRAPH_DECODE=1`, greedy only, `GenericTransformer.SupportsGraphDecode`
> gates eligibility). Result: **Qwen3-0.6B 74.3→190.8 tok/s (2.57×, gap to llama.cpp 4.5×→1.77×)**,
> **Llama-3.2-1B 112.4→120.2 (1.07×, gap 1.84×→1.72×)**, **Mistral-7B 31.4→32.0 (1.02×, gap 2.12×→2.08×)** —
> confirms the doc's own prediction below: launch-overhead removal helps small models most, marginal on big
> GEMV-bound ones. **Correctness: byte-identical greedy token sequences, graph on vs off**, verified across all
> three. See [`docs/Research/CUDA_GRAPH_FINDINGS.md`](../Research/CUDA_GRAPH_FINDINGS.md) for the underlying
> capture/replay mechanism (built for diffusion, reused here) and the Phase 6 section below for the LLM-specific
> device-indexed RoPE/embed-gather/argmax kernels this needed on top. **Also fixed same session:** a missing
> fused Q5_0 GEMV kernel — llama.cpp's Q4_K_M mixed-quant scheme substitutes Q5_0 for any tensor whose K isn't a
> multiple of 256 (common on odd hidden sizes, e.g. qwen2.5-0.5b's 896), and there was no fused kernel for it at
> all — those tensors silently ran the slow dequant-then-cuBLAS path. qwen2.5-0.5b Q4_K_M: 27.5→70.7 tok/s (2.5×).
> **Not done:** on-device sampler (temperature/top-k/top-p — graph decode is greedy-only until this exists),
> on-device MoE routing (would unlock olmoe/qwen2moe/granitemoe for graphing), the big-model GEMV
> memory-access-pattern redesign (blocked on `ncu` access last attempt).

> **STATUS UPDATE (2026-07-11): on-device repetition penalty for graph decode.** Investigated "extend
> graph decode past greedy" and found the real gap was narrower than it looked: `SamplerChain.Next`'s own
> doc comment already establishes that **temperature/top-k/top-p/min-p can never change which token wins a
> greedy argmax** (temperature is a monotonic scale; top-k/top-p/min-p only ever remove non-max candidates,
> never the max itself) — **repetition penalty is the only sampler stage whose output can differ for
> greedy**, since it can genuinely change which index has the highest logit. But graph decode's
> `ForwardGraphDecodeStep` went straight from `ProjectLogits` to `ArgMaxInto` with no sampler steps at
> all, so a request with `Greedy=true, RepetitionPenalty>1.0` and graph decode enabled was **silently
> ignoring the penalty** (raw unpenalized argmax) — a real correctness bug, not just a missing feature.
> Fixed: two new tiny single-thread kernels (`lm_history_append`, `lm_repetition_penalty_f32` in
> `native/cuda/lm/lm_f32.cu`) chained into the SAME captured graph between `ProjectLogits` and
> `ArgMaxInto`. A device history buffer (fixed capacity = KV cache capacity) is appended to once per
> replay (the current input token — mirrors `generated.Add(next)` timing in the eager loop exactly); the
> penalty kernel then walks history **sequentially, single-threaded**, replicating
> `RepetitionPenaltyStep.Apply`'s CPU semantics byte-for-byte, including a token that recurs being
> penalized cumulatively on each occurrence — a parallel scatter would race on repeated tokens and diverge,
> so single-thread is a correctness choice, not a shortcut (the loop itself is microseconds, nowhere near
> the forward pass it rides alongside). **Gate**: new `GraphDecodeRepetitionPenaltyTests` (real qwen2.5-0.5b
> Q4_K_M checkpoint, CUDA) — eager vs. graph-decode token ids are **byte-identical** with
> `RepetitionPenalty=1.3`, and confirmed to diverge from the graph-decode output with penalty=1.0 (so the
> kernel is verifiably engaging, not a false-positive match). Full Cuda.Tests suite otherwise green (5
> pre-existing, unrelated cuDNN SDPA-engagement failures on this box, nothing touched by this change).
> **Scope note for future readers:** true non-greedy on-device sampling (temperature/top-k/top-p with a
> real probabilistic multinomial draw) is a separate, much larger effort — needs GPU sort/select
> primitives for top-k/top-p and a device RNG matching the CPU xorshift exactly for determinism — and
> remains unbuilt; it was NOT needed to close the greedy correctness gap described here.

> **STATUS UPDATE (2026-07-11): fused GEMV kernel coverage completed for Q4_0 and Q5_K** — the last two
> quant types on the Phase 1 checklist's original list (`Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, Q5_0`) that didn't
> have one yet. Q4_0 is a common legacy/baseline GGUF quant; Q5_K appears in some K-quant mixed schemes.
> Both previously fell all the way to the dequant-to-F16-then-cuBLAS fallback (Q4_0 had a dequant kernel
> but no fused GEMV; Q5_K likewise). New kernels: `native/cuda/lm/mul_mat_vec_q4_0_f32.cu` (direct
> structural adaptation of the existing Q5_0 kernel, minus the high-bit plane) and
> `native/cuda/lm/mul_mat_vec_q5k_f32.cu` (the existing Q4_K kernel's super-block/warp-ownership layout,
> extended with Q5_K's extra high-bit plane, analogous to how Q5_0 extends Q4_0). Dispatched in
> `CudaBackend.LinearImpl` alongside the existing Q5_0 branch (M≤8, K%32==0 for Q4_0, K%256==0 for Q5_K).
> **Correctness**: new ground-truth test `FusedGemvGroundTruthTests` (quantize/synthesize real block data,
> dequantize independently via the canonical CPU codec, compute a plain-C# reference matmul, compare
> against the GPU fused path through the actual `CudaBackend.Linear` entry point) — `avg_err` 3.5e-7 to
> 5.9e-7 (max 1.9e-6 to 3.8e-6), 3-4 orders of magnitude under the 5e-3 tolerance; this is float-rounding
> noise, not a layout bug. Existing dequant/QuantizedMatMul regression suites (12 tests) stay green.
> **Speed**: A/B microbench at a 4096×4096 M=1 decode-shaped `Linear` call (fused vs the exact prior
> fallback code path, same process): **Q4_0 1.43× (0.123→0.086 ms/call), Q5_K 1.63× (0.117→0.072 ms/call)**.
> Caveat — this number is a floor, not the real decode-loop win: the microbenchmark's `Linear` call
> re-copies the full weight tensor host→device every call (both paths equally), which is *not* how
> production decode works (weights stay resident on-device across tokens), so the fixed per-call transfer
> cost dilutes the measured ratio. The Q5_0 kernel's real end-to-end result (qwen2.5-0.5b, resident
> weights, full decode loop) was 2.5× — expect similar or better for Q4_0/Q5_K once measured against a
> real Q4_0- or Q5_K-quantized checkpoint (none in the local model fleet currently; flagged as follow-up
> for an end-to-end t/s number, not blocking since the kernel-level correctness + speed direction are both
> independently confirmed). Remaining quant-kernel gap: Q2_K/Q3_K (new bit-packing, no template) and
> IQ2/IQ3/IQ4 (lookup-table/codebook dequant, genuinely new kernel design) still fall to the CPU dequant
> path — scoped separately as Phase 1b, out of this update.

> **STATUS UPDATE (2026-07-11): paged KV cache landed** (`src/HartsyInference.LLM/Transformer/PagedKvPool.cs`,
> `PagedKvCache.cs`) — the production-readiness plan's Phase 3. Replaces `FixedKvCache`'s single
> contiguous per-sequence buffer (hard-throws on `batch != 1` — genuinely can't serve more than one
> sequence per instance) with fixed-size pages allocated on demand from a pool SHARED across sequences: a
> short sequence no longer reserves VRAM for a worst-case max length, and a finished sequence's pages return
> to the free-list immediately for the next admission instead of sitting reserved. De-risking spike done
> first (per the plan): audited every use of `FixedKvCache` inside `GenericTransformer.ForwardBatchDecode`/
> `Layer.ForwardBatchDecode` and found it only ever touches `IKvCache` interface members — no concrete-type
> leakage — so the `FixedKvCache[] caches` parameter widened to `IKvCache[]` as a pure mechanical signature
> change (array covariance meant `ContinuousBatchScheduler`'s existing `FixedKvCache[]` needed zero changes).
> **Design: "physically paged, logically contiguous."** Pages are non-contiguous physical storage (the real
> memory-management win), but `KeyPrefix`/`ValuePrefix` gather a sequence's pages into a contiguous scratch
> tensor each call (new `IBackend.SliceTimeRange` + the existing `KvCacheAppend`, page-by-page) so
> `FlashAttention`'s existing contiguous-buffer contract needs zero changes — this defers writing a new
> paged-attention kernel (which reads scattered pages directly, no gather) to a follow-up once the block
> allocator/table are proven correct, rather than bundling a new attention kernel into the same change as a
> new memory-management scheme. **Exhaustion policy: reject** — `PagedKvPool.AllocatePage` throws
> `KvPoolExhaustedException` when the free-list is empty; queueing/eviction policy is left to the Phase 4
> scheduler, not built into the pool. **Gates**: `PagedKvCacheTests` — (1) byte-for-byte parity against
> `FixedKvCache` across a multi-page-spanning prefill + 8 page-boundary-crossing decode steps, real CUDA
> hardware; (2) a synthetic 300-round random admit/grow/evict stress harness with a deliberately undersized
> page budget (11 exhaustion hits exercised) — the load-bearing assertion is `pool.FreePageCount == maxPages`
> after everything is disposed, i.e. zero page leaks across the whole random churn. Both pass. Full
> regression suite otherwise unchanged (same 5 pre-existing unrelated cuDNN SDPA failures as prior updates).
> **Not done / explicitly deferred**: wiring `PagedKvCache` into `ContinuousBatchScheduler` (still uses
> `FixedKvCache` — that's Phase 4's replacement scheduler, which is also where dynamic mid-flight admission
> will actually exercise the pool under real multi-sequence load instead of the synthetic harness), prefix
> caching (a natural extension once pages are shared/reference-countable, noted as Phase 4 scope), and the
> gather-eliminating true paged-attention kernel mentioned above.

> **STATUS UPDATE (2026-07-11): true continuous batching landed** — the production-readiness plan's Phase
> 4. Replaces the old static-batch `ContinuousBatchScheduler` (took a fixed request list up front, ran it to
> completion, returned nothing until every sequence finished; zero production callers, so removed outright)
> with `DynamicBatchScheduler` (`src/HartsyInference.LLM/Generation/DynamicBatchScheduler.cs`, behind a new
> `IBatchScheduler` interface): requests are admitted at ANY time via `SubmitAsync`, a single background loop
> owns the model/backend/every active sequence's state exclusively, and each sequence is evicted the instant
> it finishes/stops/cancels rather than waiting for the whole cohort. Each round: drain new submissions
> (single-sequence prefill on admission — chunked/batched prefill is a further throughput optimization, not
> required for correctness, left as a follow-up), filter out cancelled/stopped/limit-reached sequences BEFORE
> decoding, then one batched `ForwardBatchDecode` round over everyone left. KV comes from Phase 3's
> `PagedKvPool`, shared across every sequence the scheduler admits.
> **The concurrency audit the plan called for found a real gap and changed the design**: naively raising
> `InferenceQueue.MaxConcurrency` would have been unsafe, because the shared `CudaBackend` instance (one CUDA
> stream, non-thread-safe activation/weight caches) is also used by diffusion image generation through the
> SAME queue — concurrent GPU calls from both workloads on one backend instance would race. Fix: `SubmitAsync`
> itself is cheap and un-gated (so concurrently-submitted chat requests all admit and batch together), but
> every GPU-touching step (prefill, each decode round) is routed through an injected gate function that wraps
> the server's existing `InferenceQueue` — one physical GPU operation at a time server-wide (diffusion or one
> LLM batch round), while every chat request that arrived since the last round still batches into that one
> operation. This gets the real concurrency win (N requests sharing 1 GPU op) without the unsafe blanket
> concurrency bump the plan's initial framing suggested.
> **Gates**: 5 new `DynamicBatchSchedulerTests` (CPU backend, synthetic weights) — (1) concurrently-submitted
> requests produce BYTE-IDENTICAL output to running each alone through `TextGenerationPipeline` (batching
> changes throughput, not answers, on CPU's batch-invariant GEMM — same guarantee the old scheduler
> documented); (2) a gpuGate that would throw on concurrent re-entry proves the exclusivity actually holds
> while still producing correct results; (3) requests submitted well after the loop starts still get admitted
> (dynamic, not a fixed up-front list); (4) cancelling one request doesn't affect concurrently-running others;
> (5) a too-small KV pool fails admission with `KvPoolExhaustedException` without corrupting other sequences.
> All pass. **Live end-to-end proof** (real qwen2.5-0.5b checkpoint, real HTTP server): two concurrent chat
> requests completed in **10.3s total vs an 8.4s single-request baseline** — not ~16.8s, which is what
> serialized-through-a-queue would have measured. Streaming and mid-flight cancellation (Phase 2) both
> reverified working through the new path. Full regression suite unchanged (same 5 pre-existing cuDNN
> failures). **Explicitly deferred, not attempted**: prefix/prompt caching (sharing identical-content pages
> across sequences) — real additional scope (page reference-counting, prefix hashing, copy-on-write on
> divergence), documented as a distinct follow-up rather than a partial bolt-on. SSM models still run
> unbatched through the shared queue directly (no KV cache to page — same scope boundary as Phase 3).

> **STATUS UPDATE (2026-07-11): JSON-mode constrained decoding landed** — the production-readiness plan's
> Phase 5a. `SamplingOptions.JsonMode` + a new `JsonGrammarStep` (`src/HartsyInference.LLM/Sampling/`) mask
> every candidate token whose decoded text would make the accumulated output an invalid JSON prefix, so the
> model can only ever emit syntactically valid JSON — exposed over HTTP as OpenAI's
> `response_format: {"type":"json_object"}` (the richer schema-constrained `json_schema` mode is explicitly
> rejected with a clear 400, not silently ignored — real additional scope, not attempted here).
> **Design**: a hand-written incremental (streaming) JSON validator (`JsonGrammarState`) tracks exactly enough
> state to answer "is this still a valid prefix" character-by-character without re-parsing from scratch each
> token (`Clone()` is a cheap stack copy) — the same technique llama.cpp's GBNF grammar sampler uses, just
> specialized to one fixed grammar instead of an arbitrary user one. Per-token vocab-to-text decoding is
> cached once per tokenizer instance (`ConditionalWeakTable`), not recomputed per request. Cost is O(vocab
> size) per generated token — the accepted cost of constrained decoding in general — and only applies when
> `JsonMode` is set.
> **Two real bugs were caught by unit tests before this ever touched a real model** — worth calling out
> because they'd have been serious if shipped: (1) object KEYS never set the "what comes after this string"
> transition, so every key would silently misroute to an invalid state right after the first `"key"` closed
> (`{"a":1}`-shaped JSON would have been unparseable — nearly all real JSON has at least one key); (2) the
> `Clone()` copy constructor forgot to copy two fields added after it was first written (`_pendingAfterContainer`,
> `_pendingAfterScalar`) — every trial-token check inside `Apply` clones the state, so this would have silently
> corrupted the container stack on literally every candidate-token evaluation in production, not just an edge
> case. Both caught by a 51-case test file (`JsonGrammarStateTests`) written BEFORE trusting the mechanism
> enough to wire it into a live model — went from 31/51 passing to 51/51 after two targeted fixes. Also fixed:
> `IsComplete` didn't originally account for numbers/literals that end simply because input stops (no
> explicit terminator character in JSON) — "123" as the whole output was incorrectly reported as "incomplete"
> forever.
> **Live end-to-end proof** (real qwen2.5-0.5b checkpoint, real HTTP server, independently verified with
> Python's `json.loads`): a flat object (`{"name":"John Doe","age":30}`) and a deliberately harder nested
> request (array of objects + a separate nested object) both produced genuinely valid, parseable JSON and
> correctly stopped generation (`finish_reason: "stop"`) — proving the "always allow the model's stop tokens
> once the JSON is grammatically complete" rule works in practice, not just in the state-machine unit tests.
> Default (non-JSON) text mode reverified unaffected. Full regression suite unchanged (same 5 pre-existing
> cuDNN failures as every prior phase). **5b (speculative decoding) remains a separate, true stretch item —
> not started.**

> **STATUS UPDATE (2026-07-11): Phase 6 (production hardening) started.** (1) Added 11 real in-process HTTP
> integration tests (`ChatCompletionsIntegrationTests`, `WebApplicationFactory<Program>`) covering
> `/v1/chat/completions`/`/v1/models` request validation — previously this HTTP layer had zero automated
> coverage beyond manual curl checks. Writing them surfaced a genuine validation-ORDER bug: the route checked
> "is the model loaded" before "is response_format even valid," so testing the json_schema-rejection path
> required a real loaded model to reach it. Fixed by reordering to check pure request-shape issues
> (messages non-empty, response_format recognized) before consulting server state (model loaded) — a
> defensible fail-fast improvement on its own, not just a testing workaround. (2) Real concurrent-load stress
> test, live server, real qwen2.5-0.5b: 3 waves of 10 concurrent chat requests each (different prompts/topics
> per request) plus a deliberately client-cancelled 11th request per wave — **30/30 succeeded, zero
> cross-contamination between batched sequences** (every response stayed on-topic for its OWN prompt, not
> mixed with another concurrent request's), zero errors logged, each wave of 10 completed in under 1 second
> wall-clock. This is the load-bearing thread-safety evidence for Phase 4's batching + backend-exclusivity
> design actually holding under sustained real concurrency, not just the earlier 2-3-request smoke tests.
> **A broader architecture sweep through the new batched server path caught a real bug**: loading gemma-3-1b
> (wider KV heads, fewer layers than the qwen2/qwen3/llama models tested so far) OOM'd on generation. Root
> cause: `PagedKvPool`'s VRAM footprint had been sized via a FIXED page count (1024, tuned against
> qwen2.5-0.5b's narrow KV dims) — since the pool pre-allocates every page up front, the exact same 1024
> pages that used a modest amount of VRAM for a narrow-KV-dim model eagerly grabbed ~3.4GB for a
> wider-KV-dim one, on a GPU that only had ~4GB free (the rest legitimately in use by the user's own running
> SwarmUI instance — confirmed via `nvidia-smi --query-compute-apps`, not a leak). Fixed by replacing the
> fixed page count with `HartsyInferenceServerOptions.KvPoolBytesBudget` (default 512MB) and a new
> `ModelManager.ComputeKvPoolPageCount` that converts the budget into a page count sized to THAT model's
> actual `numLayers × numKvHeads × headDim` — safe by construction regardless of model shape, instead of a
> number tuned against whichever model happened to be tested first. Locked in with 3 new unit tests
> (`ServerTests`) reproducing the exact narrow-vs-wide shape difference that caused the incident. Reverified
> the small models (qwen2.5-0.5b) still work identically after the fix (coherence, 4-way concurrent
> batching, JSON mode all reconfirmed). gemma-3 itself was not retested to full success after the fix — the
> real constraint is the user's concurrently-running SwarmUI instance leaving too little headroom on this
> particular shared dev GPU, not a remaining code defect; worth a retry when more VRAM is free.

**Method.** Strict measure → optimize → verify loop.

**Method.** Strict measure → optimize → verify loop. Every change must (1) show a speedup on an isolated micro-benchmark, (2) show a speedup on end-to-end t/s, and (3) preserve correctness (greedy token-id match vs the pre-change output). A faster wrong answer is a regression.

---

## The physics (why the gap is closable)

Batch-1 decode reads every weight from VRAM once per token → **memory-bandwidth bound**. 3060 ≈ 360 GB/s. Ceiling t/s ≈ bandwidth ÷ model-bytes.

| Model | quant bytes | ceiling t/s | llama.cpp t/s (% peak) | engine t/s (% peak) |
|---|---|---|---|---|
| Qwen3-0.6B | ~0.40 GB | ~900 | 354 (39%) | 17.6 (2%) |
| Llama-3.2-1B | ~1.3 GB (q8) | ~270 | 216 (80%) | 6.0 (2%) |
| Mistral-7B | ~4.4 GB | ~82 | 66 (80%) | 1.24 (1.5%) |

llama.cpp is near the wall. We are at 1.5-2% of peak → we do ~20-50× the necessary memory traffic and/or run compute-inefficient kernels. Hypothesis (pre-profile): the quantized matmul path **dequantizes each weight to F16/F32 into VRAM, then runs a generic GEMM** — that is 2+ passes over the weights plus a full-size intermediate write, versus llama.cpp's single fused dequant-GEMV pass. Confirm by profiling before building.

---

## Success metrics (tracked every phase)

1. **End-to-end tg t/s** per model (Qwen3-0.6B primary iteration target; Mistral-7B as the bandwidth-bound check). Measured via `samples/HartsyInference.TextGen.Cli` `gguf cuda 128`, warm, decode-only where possible.
2. **% of memory-bandwidth peak** (tg × model-bytes ÷ 360 GB/s).
3. **Per-op time breakdown** of one decode step (from Phase 0 profiler).
4. **Correctness:** greedy 128-token id sequence identical to the last-known-good run (per model).

---

## Progress table (fill as we go)

Baselines are **warm, 256-token, decode-dominated** (amortizes JIT+prefill; earlier 13-17 t/s figures were cold-start). Two engine configs matter: `lowVram=0` caches BF16 weights (F16 footprint, fits only small models) — the best-case; `lowVram=1` keeps weights compressed but re-dequantizes every weight every token — forced for big models.

| Date | Change | Qwen3 lv0 t/s | Qwen3 lv1 t/s | Mistral t/s | vs llama.cpp | Correct? |
|---|---|---|---|---|---|---|
| llama.cpp | reference bar | 354 | 354 | 66.5 | 1.0× | — |
| 2026-07-03 | **baseline** (current engine) | 84.3 | 26.5 | 1.24 (lv1) | Qwen3 4.2×/13× · Mistral 54× | ref |
| 2026-07-03 | **P1: fused Q4_K GEMV** (warp-per-row) | 89.3 | 67.6 | **8.52** (lv1) | Qwen3 4.0×/5.2× · **Mistral 7.8×** | coherent ✓ |
| 2026-07-03 | **P2: + fused Q6_K GEMV** (lm_head) | 90.6 | 91.6 | **16.38** (lv1) | Qwen3 **3.9×** · **Mistral 4.1×** | coherent ✓ |
| 2026-07-04 | **P2b: + fused Q8_0 GEMV** | 90.6 | 91.6 | 16.38 | **Llama-1B q8_0 6.05→83.7 (13.8×), 2.6×** | coherent ✓ |
| 2026-07-04 | **P2c: quantized lm_head** (tied embed) | 106.3 | 106.0 | 18.77 | Qwen3 **3.3×** · Llama **2.15×** · Mistral 3.5× | coherent ✓ |
| 2026-07-04 | **P3: split-K flash-decode default** | 133 | 133 | 18.6 | Qwen3 **2.66×** · Llama **1.94×** (<2×!) · Mistral 3.5× | coherent ✓ |
| 2026-07-04 | **P4: vectorized Q4_K GEMV loads** | 156.6 | 156.6 | **30.7** | Qwen3 **2.26×** · Llama 1.94× · **Mistral 2.16×** | coherent ✓ |

**Phase 4 — vectorized loads (step toward dp4a):** the Q4_K GEMV did 16 scalar loads/super-block (8 quant bytes + 8 activations). Replaced with **1× uint2 + 2× float4** (128-bit loads) + nibble extraction via shifts (`mul_mat_vec_q4k_f32.cu`). Alignment verified (quant bytes 8-aligned, activations 16-aligned). Result: **Qwen3 133→156.6 (+18%), Mistral 18.6→30.7 (+65%!)** — matmul-bound big models win most. **All models now ~2.2× off llama.cpp (from 20-54×).** Q8_0 already coalesced (1 byte/lane) — left as-is; Q6_K vectorization deferred (complex ql/qh packing).

| 2026-07-04 | **P5: dp4a int8 GEMV** (tried) | 157 | — | 31.4 | Qwen3 2.26× · Mistral 2.12× · Gemma 2.88× | dp4a=NO GAIN |

**Phase 5 — dp4a int8 GEMV: implemented, numerically exact, but NOT faster.** Built the full llama.cpp-style path (`quantize_activation_q8_1_f32.cu` → int8 + per-block scale/sum; `mul_mat_vec_q4k_q8_1.cu` → `__dp4a` int8 dot). Output byte-identical to the float kernel. **But same speed** (Qwen3 153 vs 159; Mistral 31.4 vs 31.4). **Key finding: our vectorized-float GEMV is memory-bound, not ALU-bound** — cutting ALU (dp4a) doesn't help and the extra quantize kernel offsets it. Kept behind `HARTSY_DP4A_ON` for GPUs/shapes where ALU limits. Also ruled out split-K on the GEMV: for big models N=4096-14336 → thousands of warps already, so it's **not occupancy-starved** either (unlike attention). The remaining GEMV inefficiency is **memory-access-pattern** (needs ncu, blocked by GeForce perms) — a deeper redesign.

### FINAL STATE: gap **20-54× → 1.94-2.88×**. Qwen3 2.26× · Llama **1.94×** · Mistral 2.12× · Gemma 2.88× (Gemma slowest — sliding-window attn can't use split-K yet). All coherent.
### Remaining levers: ~~CUDA graphs~~ **✅ done 2026-07-10** for the plain decoder shape (see Phase 6 below); **split-K for sliding-window/softcap attention** (helps Gemma/GPT-OSS, would also unlock those for graph decode); **GEMV memory-pattern redesign** (needs ncu access — the remaining lever for BIG models, which graphs barely helped); **on-device sampler + MoE routing** (would extend graph decode past greedy-only/dense-only).

---

## Phase 6 — CUDA Graphs (✅ done for the plain decoder shape, 2026-07-10)

**Why:** nsys shows ~30% GPU-idle gaps on small models = CPU can't issue the ~640 kernel launches/token fast enough (GPU starves between tiny kernels). A captured graph replays the whole decode step in ONE `cuGraphLaunch`, removing launch issuance from the critical path. Biggest remaining lever for small models.

**✅ Foundation VERIFIED (2026-07-04):** `CudaGraph` wrapper (already in repo, was untested) works on this GPU — `CudaBackend.GraphSmokeTest()` captures a Scale on a stable buffer, replays twice with changed input → returns (6, 15) = PASS. Proves: capture/replay works, the stream-ordered async-pool memory model is capture-compatible, and replay reads live buffer content. CLI: `hartsyinference-textgen graphtest`. Backend was already graph-ready (single compute stream, all launches on it, `cuMemAllocAsync` pool, graph P/Invokes bound).

**Design (mapped by 2 agents):** capture-once/replay-many requires the per-token varying state to be **device-resident** (host scalars/tables get baked into the graph at capture). The exact conversions needed (each has a clean device path; gate the whole thing behind `HARTSY_GRAPH_DECODE`, default OFF → zero risk to the verified default path):
1. **RoPE**: `BuildRope` is host `Math.Cos/Sin` into per-step cos/sin tensors → precompute the FULL `[maxSeq, ropeDim]` table once (device), rope-apply indexes it by a **device position**. (Handles all scaling variants since the table is built once with the existing `RopeFrequencyBuilder`.)
2. **Attention**: `kvLen`/`qOffset` are host int params in `LaunchFlashAttention` → read from a **device position counter** (kernel loops device length; grid fixed for max).
3. **KV append**: offset is a host int → device position.
4. **Embed**: `EmbedLookup` is a host gather from host-resident `_embed` → device gather (`lm_gather_rows_f32` already reads a device idx ptr) from an **on-device embed**, indexed by a **device token buffer**.
5. **Argmax**: `ArgMaxLastDim` (device, EXISTS) writes the next token to the **device token buffer** → embed of the NEXT replay reads it. Fully GPU-resident greedy: no per-token D2H (accumulate tokens in a device buffer, one D2H at the end).
6. **Fallback (no hacks):** graph path handles the dense RoPE+GQA case (Qwen/Llama/Mistral); MoE/MLA/sliding-window/softcap/abs-pos fall through to the (verified) default decode. Clean feature gate, not a monkey-patch.

**Status: ✅ DONE (2026-07-10).** All 6 conversions built exactly as designed above, plus a device-position
`{kvLen, qOffset}` buffer convention shared with the (concurrent, separate-session) audio-model CUDA-graph work —
`IBackend.GraphDecodeSupported`/`AllocDevicePos`/`WriteDevicePos`/`KvCacheAppendDev`/`FlashAttentionDev` landed
first from that session; this work added the LLM-specific remainder: `native/cuda/lm/lm_f32.cu`'s
`lm_rope_decode_splithalf`/`lm_rope_decode_interleaved`/`lm_embed_gather_decode_f32` kernels,
`IBackend.RopeApplyDecodeStep`/`EmbedGatherDecodeStep`/`ArgMaxInto`/`BuildRopeTableDevice`/device-token-id
buffer, and — since `TextGenerationPipeline` must stay backend-agnostic (CPU builds link `IBackend` only) — a
backend-agnostic `IBackend.CaptureGraph(Action)`/`LaunchGraph(object)`/`DisposeGraph(object)` (opaque handle)
so the pipeline never references `CudaGraph`/`CudaBackend` directly. `GenericTransformer.Layer.ForwardGraphStep`
reuses the EXISTING position-agnostic helpers (`PreSublayer`/`PostSublayer`/`Mlp`/`Project`) verbatim, swapping
in the device-indexed ops only for RoPE/KV-append/attention. `SupportsGraphDecode` eligibility gate: plain
pre-norm GQA/RoPE, no MLA/MoE/cross-attention/sliding-window/softcap/sink/ALiBi/parallel-residual/non-1.0
embedding-scale — covers Llama/Qwen2/Qwen3/Mistral, falls through to the untouched eager path otherwise.
Opt-in via `HARTSY_GRAPH_DECODE=1`, gated to greedy (no on-device sampler chain yet). **Verified
byte-identical greedy token output** vs the eager path on Qwen3-0.6B (split-half RoPE) and Llama-3.2-1B +
Mistral-7B (interleaved RoPE) — not just "coherent," the literal same computation replayed instead of reissued.
Results in the STATUS UPDATE block at the top of this doc.

**Phase 3 — attention split-K (nsys-guided):** re-nsys after P2c showed attention `lm_flash_attn_f32` had grown to **29.6%** of decode (72.8µs/call — one block/head, ~16 blocks under-occupy 28 SMs, sequential keys). The validated split-K flash-decode kernel existed but only engaged at `kvLen≥1024` (never for decode). Changed the dispatch (`CudaBackend.cs` ~3417): engage for the **occupancy-limited** case (`baseBlocks < 2·SM` = decode) at `kvLen≥128`, `minChunk=32`. Gated to `plain` attention so sliding-window/softcap models (Gemma, GPT-OSS) keep the monolithic path unchanged; split kernel is numerically exact. Result: **Qwen3 103→133 (+28%), Llama 100→111.5 (1.94× off llama.cpp, under 2×)**. Big models (Mistral) unaffected — they're matmul-bound, not attention-bound.

### Gap now: **Qwen3 2.66× · Llama 1.94× · Mistral 3.5×**. Small models attention-fixed; big models need GEMV throughput (dp4a int8 / R4-repack / memory-parallelism — ours reads ~22% of bandwidth vs llama.cpp ~80% on Mistral). That + CUDA graphs is the path to parity/beating.

**Phase 2c — the lm_head fix (nsys-guided):** nsys showed the tied lm_head was a **2.17 ms/token F32 cuBLAS GEMV** (28% of decode) because line 123 force-dequantized the tied embed to F32 (622 MB read/tok). Now keep the original **quantized** embed (`_lmHeadQuant`), preload *that* (not the F32 table → **~0.5 GB less VRAM**), and route the head through the fused GEMV (F32 hidden, no F16 cast). Also fixed untied lv1 heads (Mistral's Q6_K head skipped the F16-cast path). Results: **Qwen3 90.6→106 (+17%), Llama 83.7→100 (+20%, 2.15× off llama.cpp), Mistral 16.4→18.8**. `GenericTransformer.cs` load/EnumerateWeights/ProjectLogits.

### Gap so far: 20-54× → **2.15-3.5×**. Remaining (nsys): our Q4_K GEMV 36% (17µs/call, overhead-bound not bandwidth-bound → R4 repack/vectorize/dp4a), attention 16% (low occupancy), ~640 launches/token (CUDA graphs).
### ⚠ TODO before "done": formal token-parity gate vs llama.cpp (validated coherent only so far).

### Mid-grind merge (2026-07-04) — RESOLVED, NO regression. Merge `a3c6f59` landed: my LLM work (`3b034fa`), flash-attn-v2 (`26ebea3`), Wan/video (`b314a87`), music (`5b7bf24`). Verdict:
- **LLM not regressed.** `flash_attn_v2_tf32` is **opt-in** (`HARTSY_SDPA_V2`) and only in the video/diffusion `ScaledDotProductAttention` path — LLM decode uses `FlashAttention`/`lm_flash_attn_f32`, untouched. My GEMV+lm_head committed & intact. Clean idle re-measure: **Qwen3 lv0 100 / lv1 99** (the "82-88" I saw was GPU contention from SwarmUI + the user's other agents at 91% util, NOT the merge).
- **Builds clean.** Full sample (LLM+Audio+Cuda) + Diffusion(video) both compile. The Audio CS8600 errors were a transient mid-merge artifact, gone.
- Audio/video runtime owned by other agents; v2 opt-in so their default paths unchanged.

**Phase 2b:** Q8_0 GEMV (`native/cuda/lm/mul_mat_vec_q8_0_f32.cu`). Llama-3.2-1B q8_0 **6.05→83.7 t/s (13.8×)**, now only **2.6× off llama.cpp** (was 36×) — the best result, q8_0 is the simplest format. All three quant formats (Q4_K/Q6_K/Q8_0) fused. Re-profile after P2 (synced) still shows the launch-bound shape: ~640 launches/token (Linear 12805, Permute 7280, RmsNorm 7345, **H2D_MISS 7466 = per-token host RoPE/embed rebuild**). Next = attack launch/host overhead (H2D elim, fusion, CUDA graphs).

**Phase 2 result:** Q6_K fused GEMV (lm_head + Q6_K weights). Compressed `lv1` now **equals** F16-cached `lv0` (both ~91 on Qwen3) — so big models run compressed at full speed, no F16-blowup/OOM. Cumulative: **Qwen3 lv1 26.5→91.6 (3.46×)**, **Mistral lv1 1.24→16.38 (13.2×)**, gap to llama.cpp **54×→4.1×** on Mistral. Kernel `native/cuda/lm/mul_mat_vec_q6k_f32.cu`. Remaining ~4× gap = launch overhead (~700 launches/tok → CUDA graphs P5), F32 activations (→FP16 pipeline P3b), per-token host H2D (RoPE/embed → P4), unfused ops.

**Phase 1 result:** Qwen3 lv1 **26.5→67.6 (2.55×)**, Mistral lv1 **1.24→8.52 (6.9×)**, lv0 no longer regresses (84.3→89.3). Big models (forced to lv1) win most — the per-token whole-weight dequant is gone. lv0 gains little yet because its Q4_K weights were already F16-cached AND the lm_head is Q6_K (still old path → Phase 2). Kernel: `native/cuda/lm/mul_mat_vec_q4k_f32.cu` (one warp/row, shuffle reduction, F32 accumulate — more accurate than the old BF16-operand cuBLAS). Dispatched in `CudaBackend.LinearImpl` for M≤8 · Q4_K · F32 in/out.

**Phase 0 attribution (Qwen3, HARTSY_PROFILE_SYNC):** `Linear` (matmul) = **~90%** of GPU time (750 ms), then Permute0213 123, RmsNorm 78, H2D_MISS 61, Silu 16. Also **7,466 per-token H2D transfers** (RoPE cos/sin + embed rebuilt on host each token) = Phase 4/5 target. Confirmed: **no fused dequant-GEMV; no CUDA graph; ~700-1300 launches/token.**

---

## Phase 0 — Profile & attribute ⬜ (do this FIRST, no code changes)
- [ ] Get a per-op time breakdown of one decode step (HARTSY_PROFILE if it does per-kernel CUDA-event timing; else add minimal CUDA-event instrumentation around op categories: qkv-matmul / attn / o-proj / gate-up-matmul / down-matmul / lm_head / norm+rope+elementwise).
- [ ] Attribute the ~70 ms/token (Qwen3) across those categories. Confirm the matmul (esp. quant-GEMV + lm_head) share.
- [ ] Stand up the isolated micro-bench for the quant matmul at **M=1** decode shape (via `benchmarks/HartsyInference.GpuBenchmarks`), so a kernel change can be measured in isolation.
- [ ] Lock the correctness gate: capture Qwen3-0.6B greedy 128-token id sequence as golden.
- [ ] Record the attribution + baseline numbers in the progress table.

## Phase 1 — Fused dequant-GEMV kernel ✅ (expected biggest win)
- [x] Per quant type (Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, Q5_0): a single CUDA kernel that reads quantized blocks and accumulates the dot product with the input vector directly — no F16/F32 weight materialization, one pass over the weight. Model on llama.cpp `mul_mat_vec_q` (one warp per output row, int8/dp4a where applicable, F32 accumulate). **All six done** (Q4_K/Q6_K/Q8_0/Q5_0 2026-07-04..07-10, Q4_0/Q5_K 2026-07-11 — see status update above).
- [ ] Wire the decode-path Linear (M small, i.e. ≤ a few rows) to dispatch to the fused GEMV; keep the existing GEMM for prefill (M large).
- [ ] Micro-bench each kernel vs the current dequant-then-GEMM; require ≥ target GB/s.
- [ ] End-to-end t/s + correctness. Record.

## Phase 2 — lm_head / vocab projection ⬜
- [ ] Ensure the final ~150k-row projection uses the fused quant-GEMV (or an efficient F16 GEMV), not a full F32 GEMM. Measure its per-token share before/after.

## Phase 3 — Attention decode kernel ⬜
- [ ] Fused single-query flash-decode over the KV cache (online softmax, one pass over KV, no full score materialization). Compare to current SDPA path.
- [ ] Confirm KV is F16 and contiguous; no per-token realloc.

## Phase 4 — Fuse norm / RoPE / SwiGLU / residual ⬜
- [ ] Fuse RMSNorm(+residual), RoPE, and SwiGLU (gate·silu·up) to cut kernel launches and extra memory passes. At 0.6B these small ops and launch overhead are a real share once matmul shrinks.

## Phase 5 — Per-token / launch overhead ⬜
- [ ] CUDA graphs (or a captured replay) for the fixed per-token op sequence to amortize kernel-launch + host overhead (llama.cpp does this). Measure launch-overhead share first.
- [ ] Trim any residual host work per token (sampling, cache bookkeeping).

## Phase 6 — Sweep & report ⬜
- [ ] Re-run the full 7-model comparison (Tier-1 + Swarm) from `LLM_THROUGHPUT_BENCHMARK.md`; update its tables with the new ratios.

---

## dotLLM (kkokosa) review — techniques to adopt

dotLLM is the same architecture as us (pure C#, PTX-via-Driver-API, unmanaged/mmap, `IBackend`, config-driven transformer). Almost everything in its README we already have or exceed (more archs, more modalities, same sampler chain, same memory model, flash/online-softmax attention). The techniques it has that we **don't**, ranked by fit to this grind:

- [🔧] **Fused quantized GEMV for Q8_0 / Q4_K / Q6_K** — they have all three; we now have Q4_K (Phase 1). Extend to Q6_K (lm_head!) + Q8_0 → **this is Phase 2**, confirmed.
- [ ] **FP16 activation pipeline for decode** — they run decode activations in FP16 with a "custom quantized GEMV + FP16 activation pipeline"; we run **F32 activations**. FP16 halves activation bandwidth and lets the GEMV read FP16 x. → new **Phase 3b**.
- [ ] **Projection fusion: Q/K/V (3→1) and Gate/Up (2→1)** — "saves ~72 dispatches/layer" and shares the input read. → folds into **Phase 4**.
- [ ] **Row-interleaved (R4) weight repacking + 4-row GEMV** — repack 4 consecutive rows' quant blocks contiguously at load so a warp-group reads coalesced and reuses the input vector across 4 rows. Direct upgrade to our GEMV. → **Phase 1.5**.
- [ ] **Fused RMSNorm+quantize (decode)** — eliminate the norm-output intermediate; produce the (F16/quantized) activation the next GEMV wants in one pass. → **Phase 4**.
- [ ] **Speculative decoding** (draft-verify-accept + KV rollback) — large decode-latency win, orthogonal to kernels. → later phase.
- [ ] **Paged KV-cache + KV quantization (Q8_0/Q4_0)** — PagedAttention, prefix cache, 3.7-7.1× KV memory. Serving/memory, enables batching. → later phase.
- [ ] Their CPU-only tricks (AVX-512 softmax, Schraudolph exp, NUMA threading, ComputeThreadPool) matter **only if CPU inference becomes a target** — we're GPU-first, and our CPU kernels are F32-only. Deprioritize.

Note: dotLLM does **not** mention CUDA graphs — our Phase 5 (graph capture of the static decode step) may be a genuine edge, not just parity.

## Rules of the grind
- One change at a time; measure isolated + end-to-end; keep or revert on the number.
- Never regress correctness — greedy token-id match gate after every kernel change.
- Iterate on Qwen3-0.6B (fast loop); confirm on Mistral-7B (bandwidth-bound, biggest absolute gap).
- Prefer fixing the engine kernels over per-model hacks — the win must generalize across archs.
