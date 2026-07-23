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

> **STATUS UPDATE (2026-07-22): `ncu` access re-attempted, still blocked — confirmed a driver policy, not a
> toolchain gap.** This session discovered pip-installable standalone CUDA toolkit wheels
> (`nvidia-cuda-nvcc`/`nvidia-cuda-runtime`/`nvidia-cublas`) work on this box with no system CUDA install
> (see `MODEL_STATUS_LLM.md`'s glm4 section) — that includes `nsight-compute` (`ncu`), which was previously
> assumed simply absent. It isn't: `ncu --set basic ... dotnet ...` connects to the target process fine but
> fails with `ERR_NVGPUCTRPERM` — the GeForce driver restricts GPU performance-counter access to admin users
> regardless of which `ncu` binary/version launches it. This needs a system-level fix (`sudo`-gated kernel
> module parameter + reload, or profiling as root), not a code or toolchain change, and wasn't attempted
> without explicit user sign-off. **The GEMV memory-access-pattern redesign lever stays blocked** — Phase 5's
> conclusion (memory-bound, not ALU-bound; `dp4a` int8 built, numerically exact, no speedup) still stands, so
> there's no new evidence to guess a redesign from. Two new benchmark data points added this session (via
> `TextDecodeThroughputBenchmark.cs` vs a fresh llama-cpp-python build — see `LLM_THROUGHPUT_BENCHMARK.md`):
> **Llama-3.2-1B-Instruct Q8_0: 142.14 tok/s graph-on vs llama-cpp-python 190.34 (1.34× slower)** — notably
> *better* than the 2026-07-10 entry's 1.72× gap for the same model (measurement-methodology difference:
> llama-cpp-python here vs `llama-bench` there; still an improvement, not a regression). **Qwen3-4B Q4_K_M
> (new size, not previously benchmarked — only Qwen3-0.6B was): 60.05 tok/s graph-on vs 85.59 (1.43× slower)**,
> in the same range as the rest of this doc's gaps. Neither is faster than llama.cpp; no code change made this
> pass since every already-identified lever (CUDA graphs) is already on, and the next lever needs `ncu` data
> this session couldn't obtain.

> **STATUS UPDATE (2026-07-22, same session, continued): `ncu` access unblocked via `sudo`, real fix found
> and verified.** `ERR_NVGPUCTRPERM` is bypassed by running `ncu` itself under `sudo` (with `HOME` and
> `CUDA_DEVICE_ORDER`/`CUDA_VISIBLE_DEVICES` explicitly passed through `sudo env`, since `sudo` resets the
> environment and this engine resolves its native libs relative to `$HOME/.local/lib/cuda13` — see
> `CudaLibraryResolver.cs`). No persistent system change made (no kernel module parameter edited); this is a
> per-invocation elevation only.
>
> **Root-caused a real, fixable inefficiency**: `nsys` full-run kernel aggregation on Qwen3-4B showed the
> fused GEMV kernels dominate decode as expected (`mul_mat_vec_q4k_f32` 34%, `mul_mat_vec_q6k_f32` 22% of
> total GPU time) — but per-instance, `mul_mat_vec_q6k_f32` took ~135us vs `mul_mat_vec_q4k_f32`'s ~36us
> despite comparable per-layer work. `ncu --set full` on both kernels in isolation confirmed why: Q6_K's
> `Executed Ipc Active` was 1.39 vs Q4_K's 2.70, Issue Slots Busy 33.7% vs 62.5%, with ncu's own diagnostic
> reading "low compute throughput and memory bandwidth... typically indicate latency issues" — the kernel
> was **latency-bound, not bandwidth-bound**, from `mul_mat_vec_q6k_f32.cu`'s tight interleaved
> load-then-immediately-use pattern (each of the 4 unpacked values per half-super-block loads its bytes and
> computes right before the next), unlike `mul_mat_vec_q4k_f32.cu`'s upfront `uint2`/`float4` vectorized loads.
>
> **Fix**: restructured `mul_mat_vec_q6k_f32.cu` to issue every load for both half-super-blocks up front
> (into named locals) before any bit-unpacking or FMA — gives the warp scheduler many independent in-flight
> memory requests to hide latency behind, instead of serializing on each load's round trip one at a time.
> The accumulation order into `acc` is unchanged (same 8 FMAs in the same sequence, only WHEN the underlying
> loads happen moved earlier), so this is provably bit-for-bit identical floating-point output, not an
> approximation — confirmed by `CudaQuantizedMatMulTests.QuantizedMatMul_MatchesLinear(Q6_K, ...)` and the
> full `Q6_K_GpuDequant_MatchesCpu`/`Linear_Q6_K_Weight_MatchesF16Reference` tests, plus the full
> `HartsyInference.Cuda.Tests` suite (136/136) passing after the change with no other files touched.
>
> **Measured result (RTX 3060, `TextDecodeThroughputBenchmark.cs`, graph-on)**: Qwen3-4B Q4_K_M
> **60.05 to 63.9 tok/s (+6.4%)**; gap to llama-cpp-python narrowed from **1.43x to 1.34x slower**.
> Llama-3.2-1B-Instruct Q8_0 unchanged (142.14 tok/s) as expected — a pure Q8_0 quant has no Q6_K tensors to
> benefit from this fix. Q5_K's sibling kernel (`mul_mat_vec_q5k_f32.cu`) already uses the same upfront
> `uint2`/`float4` vectorized-load structure as Q4_K, so it wasn't affected by this class of bug and wasn't
> touched. Modest, verified, low-risk win — not the full "GEMV memory-access-pattern redesign" this doc
> flagged as the big remaining lever (that would mean restructuring which thread owns which elements to
> enable true vectorized loads across Q6_K's strided layout, a bigger and riskier rewrite deferred here), but
> real progress unblocked specifically by finally getting `ncu` access.

> **STATUS UPDATE (2026-07-22, same session, continued): graph-capture allocation-node arena — tried,
> measured, REVERTED (no real win).** `nsys cuda_gpu_trace` gap analysis on Qwen3-4B found a large idle gap
> around each `cuGraphLaunch`, and `cuda_api_sum` showed captured graphs held ~976 `cuMemAllocAsync`/
> `cuMemFreeAsync` allocation nodes (each decode-step temporary tensor allocates async inside the capture,
> becoming a graph memory node CUDA must re-resolve on every `cuGraphLaunch`). Built a bump-arena allocator
> (`CudaMemory.ArenaAllocate`, opt-in via `CudaBackend.CaptureGraph(..., useCaptureArena: true)`): reserve one
> 64MB buffer *outside* the capture, hand out bump-pointer offsets for everything allocated *inside* the
> capture, bulk-free the whole arena at graph disposal — collapses the ~976 per-launch allocation nodes to
> zero. Correctness fully verified (byte-identical greedy output, full `HartsyInference.Cuda.Tests` 136/136,
> `HartsyInference.LLM.Tests` 131/131). **Found and fixed a real regression along the way**: applying the
> arena unconditionally broke CSM (audio) graph decode, because `CsmModel`'s `ForwardGraphDecodeStepEmbeds`
> keeps a persistent fixed-address buffer (`gs.OutHidden`) alive and read across many replays — redirecting it
> into the transient arena corrupted that invariant. Fixed by making the arena strictly opt-in per call site
> (`IBackend.CaptureGraph`'s new parameter defaults to `false`), enabled only at the two LLM decode-step sites
> whose captured body disposes every temporary before returning; CSM's call sites were left untouched at the
> default.
>
> **The catch: `nsys`'s own numbers were misleading about real-world impact.** Under `nsys` instrumentation,
> `cuGraphLaunch`'s reported API-level cost dropped ~3× (≈1.47ms → ≈0.48ms avg). But repeated **clean,
> unprofiled** `TextDecodeThroughputBenchmark.cs` runs (the same benchmark this whole doc's numbers come from,
> and the only number that matters — nsys's own tracing overhead is known to scale with per-call complexity,
> so a 976-node graph launch being instrumented is not representative of that launch running un-instrumented)
> showed Qwen3-4B at **64.0-64.6 tok/s** — barely distinguishable from the **63.7-64.1 tok/s** already measured
> with the Q6_K fix alone, before the arena existed. In other words: a real, verified, measurable reduction in
> one specific profiler-reported metric, with **no measurable end-to-end throughput gain**. Per this doc's own
> rule ("every change must show a speedup on an isolated micro-benchmark, end-to-end t/s, AND preserve
> correctness"), rule #2 failed — **reverted** (all 5 files: `CudaMemory.cs`, `CudaBackend.cs`, `IBackend.cs`,
> `TextGenerationPipeline.cs`, `DynamicBatchScheduler.cs` restored via `git checkout`). Re-verified back to the
> Q6_K-only state with `HartsyInference.Cuda.Tests` still green.
>
> **Lesson for future profiling on this engine**: host-side/API-level overhead measured *under nsys* must be
> cross-checked against a clean unprofiled wall-clock run before treating it as a real lever — nsys's per-call
> instrumentation tax itself scales with call complexity (more graph nodes → more to trace → inflated reported
> gap), so an nsys-only diagnosis can lead to chasing an artifact of the measurement tool rather than a real
> bottleneck. This effectively **rules out the "launch/allocation-node overhead" branch of the host-overhead
> hypothesis entirely** — combined with the already-ruled-out per-token D2H sync (fundamental to greedy
> autoregressive decode: token N+1's embed depends on N's sampled id, so there's no pipelining it away, and
> llama.cpp pays the identical cost), **the remaining ~1.34× gap on Qwen3-4B is GPU-kernel-bound, not
> host-overhead-bound.** Next candidate per `nsys cuda_gpu_kern_sum`: `lm_flash_attn_f32` at ~12% of decode
> GPU time, not yet profiled in isolation with `ncu --set full` (unlike Q4_K/Q6_K GEMV, which were checked and
> found well-balanced/near-roofline and latency-bound respectively).

> **STATUS UPDATE (2026-07-22, same session, continued): attempted decode-attention profiling — blocked by
> tooling, not by findings. Two real, narrow facts surfaced; no decode-path conclusion reached.**
>
> (1) **Profiled `lm_flash_attn_f32` under `ncu --set full`, but only caught PREFILL calls (grid=736 = prompt
> tokens × heads), not decode** — the process reliably OOMs on `ProjectLogits`'s lm_head cast right after
> prefill finishes, before decode step 1's attention kernel ever launches, every time `ncu`/`nsys` is attached
> (confirmed the model itself generates fine at the same token counts with no profiler attached — this is
> profiler-induced VRAM overhead, e.g. `ncu`'s per-launch replay state, not an engine bug). The prefill-shaped
> data (42%/42% memory/compute throughput, 74% occupancy, "37.5% Est. Speedup: L1TEX scoreboard stall") is
> real but **does not license a decode-path conclusion** — decode's attention launches at a much smaller grid
> (~32 blocks vs 736) on 28 SMs, a materially different occupancy regime the prefill numbers don't transfer to.
> Do not read this as "flash-attn is latency-bound in decode."
>
> (2) **The dedicated split-K flash-decode kernels (`lm_flash_attn_f32_split`/`lm_flash_attn_f32_combine`)
> never launched at all** during any of these runs (24-128 generated tokens) — confirmed by kernel-name
> filtering, not by inference. The only "split" kernel that fired was cuBLAS's own `splitKreduce_kernel`
> (unrelated — unpacked here after an initial regex false-match). Per Phase 3's dispatch rule
> (`CudaBackend.cs`, engage at `kvLen≥128`), Qwen3-4B's short-to-medium decode runs never reach that
> threshold, so 100% of decode-time attention for a typical (say 128-token) generation runs the plain
> monolithic per-head kernel, not the split-K one. **This is a fact, not yet a diagnosis** — the threshold may
> be correctly tuned (short KV = little attention work, splitting isn't worth it there) or may be miscalibrated
> for this shape; distinguishing the two needs a wall-clock A/B (lower the threshold, measure), not more
> profiling.
>
> **Fixed as a side effect**: `~/.local/lib/cuda13/libnvToolsExt.so.1` was missing, so this engine's NVTX range
> instrumentation silently no-op'd (`CudaLibraryResolver`'s `DllNotFoundException` fallback) — every prior
> `nsys` trace this session lost the ability to visually separate prefill from decode via named ranges.
> Symlinked from the local Python venv's `nvidia-nvtx` pip package. Not yet confirmed working end-to-end
> (deprioritized once the `ncu`/`nsys` OOM-under-profiler issue above made attention profiling the blocker,
> not missing NVTX ranges) — verify before relying on it in a future session.
>
> **What's actually settled after both this update and the arena update above**: the host-overhead hypothesis
> (launch overhead, graph allocation-node overhead, per-token D2H sync) is fully ruled out. **The remaining
> ~1.34x gap is GPU-kernel-bound**, but WHICH kernel(s) and by how much remains unmeasured for the decode
> phase specifically — profiling that requires either (a) a lighter-weight decode-only isolation harness than
> attaching `ncu`/`nsys` to a full generate call (e.g. a microbenchmark that runs ONLY N decode steps with a
> pre-warmed KV cache, no prefill, sized to fit under whatever VRAM headroom the profiler needs), or (b) fixing
> the profiler-induced OOM directly. Neither attempted yet.

> **STATUS UPDATE (2026-07-22, same session, continued): split-K decode attention split-COUNT formula fixed —
> real, verified ~9% win on BOTH benchmarked models.** Chasing the "does split-K even engage?" question above
> led to a wrong intermediate conclusion (documented and corrected here rather than silently fixed): a
> `HARTSY_DEBUG_SPLITK` print statement (temporary, since removed) confirmed `FlashAttentionDev`
> (`CudaBackend.cs`) **does** engage split-K for the real benchmarked shape (`lk=193, splits=3` for Qwen3-4B
> at the standard 128-token benchmark) — the earlier "never engages" claim in this doc was an artifact of
> profiling with much shorter prompts/max-tokens specifically to dodge the ncu OOM, which shrank the KV
> capacity gate (`promptLen + MaxTokens + 1`) below the `lk≥128` threshold. Corrected.
>
> With that resolved, the real remaining question was whether the split COUNT itself was well-tuned. Added a
> temporary `HARTSY_FORCE_SPLITS` env-var override (removed after) and swept it against the real
> `TextDecodeThroughputBenchmark` on Qwen3-4B (`lk=193`, clean wall-clock, no profiler): tok/s rose
> **monotonically** from the shipped default (splits=3, 64.0 tok/s) through splits=4 (66.2), 6 (68.0), 8
> (68.9), up to a plateau at **splits=12-20 (~69.5-69.6 tok/s, +8.6%)**, then a slight decline by splits=32
> (69.4) — the fixed per-split combine-kernel overhead eventually outweighs the added parallelism. The
> existing formula's `maxG = lk/64` chunk-size floor ("keep chunks ≥64 keys") was capping splits at 3 well
> short of that plateau — an unvalidated guess from when the split-K path was first built, not a measured
> constraint.
>
> **Fix** (`CudaBackend.cs`, `FlashAttentionDev`): raised the occupancy target from `4 × SM count` to
> `16 × SM count`, and loosened the chunk-size floor from `lk/64` to `lk/16` (chunks as small as ~16 keys, not
> ≥64) — reproduces splits=12 for the measured shape, landing in the plateau with margin either side.
>
> **Measured result, clean wall-clock, `TextDecodeThroughputBenchmark` (RTX 3060, graph-on)**:
> - **Qwen3-4B Q4_K_M: 64.01 → 70.09 tok/s (+9.5%)**
> - **Llama-3.2-1B-Instruct Q8_0: 142.14 → 155.28 tok/s (+9.2%)** — notably, this model has zero Q6_K tensors
>   (pure Q8_0), so this is a clean, independent confirmation the win is real and generalizes across quant
>   types/architectures, not an artifact riding on the earlier Q6_K fix.
>
> **Correctness**: `SchedulerGraphDecodeTests` (all 4 cases) pass cleanly with real Llama-3.2-1B checkpoints,
> confirming byte-identical greedy token sequences graph-on vs graph-off vs the direct pipeline. For Qwen3-4B,
> a direct CLI A/B (`hartsy text "The capital of France is" --max-tokens 40`, graph-decode on vs off) produced
> **character-for-character identical output**. `SchedulerGraphDecodeTests`/`GraphDecodeRepetitionPenaltyTests`
> could not give a clean in-process confirmation for Qwen3-4B specifically — they hit a real
> `CUDA_ERROR_OUT_OF_MEMORY`, but **confirmed pre-existing via `git stash`** (identical failure on the
> unmodified tree, same test file, same model) — a cumulative-VRAM-across-sequential-test-cases issue in that
> test file unrelated to this change, not investigated further here (out of scope for this grind). Full
> `HartsyInference.Cuda.Tests` suite: 133/137 (4 pre-existing unrelated failures — `Fp8NativeGemmTests`
> cuBLASLt heuristic error, `Conv1dKernelTests`, `MultiBackendIsolationTests` precision — all confirmed
> pre-existing via the same stash comparison).
>
> **Gap update**: Llama-3.2-1B 1.34x → **1.23x** slower than llama-cpp-python; Qwen3-4B 1.34x → **1.22x**.
> Both real, both kernel-dispatch-formula fixes (no new kernel code, no correctness risk beyond a tuning
> constant) — the highest-value, lowest-risk lever found this session. See `LLM_THROUGHPUT_BENCHMARK.md` for
> the updated results table.

> **STATUS UPDATE (2026-07-22, same session, continued): same fix applied to the EAGER (non-graph-decode)
> split-K path — smaller but real win, and it unblocked the ncu decode-attention profiling this doc's earlier
> entries couldn't get.** Built a minimal standalone probe (`attn-probe`, scratchpad-only, not committed) that
> calls `CudaBackend.FlashAttention` directly at the real Qwen3-4B decode shape (Hq=32, Hkv=8, D=128, lk=193)
> with NO model loading — no weights, no KV cache, no scheduler — specifically to get under the VRAM headroom
> `ncu`/`nsys` need. It worked: profiled `lm_flash_attn_f32_split`/`_combine` on the real RTX 3060 for the
> first time this session without an OOM.
>
> That immediately surfaced a fact worth stating plainly: `FlashAttention` (eager) and `FlashAttentionDev`
> (graph-decode, fixed in the update above) are **two separate call sites with two separate, independently-
> tuned split-count formulas** for the identical split/combine kernel pair. The eager one was still on the old
> unvalidated constants (`target=2×SM`, `minChunk=32`) — the probe showed it landing on only 2 splits for
> `lk=193` (grid=64/32), with the same low-occupancy signature (19%/10% achieved occupancy, ~12%/~4% memory
> and compute throughput) that the graph-decode path had before its fix. Applied the identical tuning
> (`target=16×SM`, `minChunk=16`) to the eager formula (`CudaBackend.cs`, the `FlashAttention` method's
> `occLimited` branch) — same kernel, same occupancy physics, so no new sweep needed to justify it.
>
> **Measured**: the probe's isolated per-call latency dropped **216.7us → 178.5us (-17.6%)**. End-to-end,
> `TextDecodeThroughputBenchmark`'s graph-OFF numbers (which exercise this exact path — every eager decode
> step calls `FlashAttention`, not `FlashAttentionDev`) moved **Llama-3.2-1B 128.5 → 134.5 tok/s (+4.7%)**,
> **Qwen3-4B 58.7 → 60.6 tok/s (+3.4%)** — smaller than the graph-decode win because attention is one part of
> a decode step dominated by GEMV, but real and consistent. Graph-ON numbers unaffected (within run-to-run
> noise), as expected — this change only touches the separate eager call site. This path matters beyond "the
> slow fallback": every graph-INELIGIBLE architecture (MoE, sliding-window/softcap models, MLA, non-1.0
> embedding-scale — see Phase 6's eligibility gate) runs decode through this exact formula permanently, so
> this fix generalizes to model families the graph-decode fix above never touches.
>
> **Correctness**: `CudaFlashAttentionTests` (5/5) pass unchanged — these call `FlashAttention` directly and
> don't depend on which split count is chosen (numerically exact split/combine vs monolithic either way).
> Full `HartsyInference.Cuda.Tests`: 133/137, same pre-existing 4 failures as every prior checkpoint this
> session.
>
> **Method note for future profiling sessions on this box**: attaching `ncu`/`nsys` to a full `generate()`
> call on this 12GB card reliably OOMs regardless of what's being measured — not a bug, the engine's own
> footprint (weights + graph capture + activations) already uses nearly the full 12GB in normal unprofiled
> operation, and the profiler's own bookkeeping is enough to tip it over. **The fix isn't a smaller model or
> a shorter generation — it's a standalone probe that calls the target kernel directly with synthetic
> tensors at the real production shape, no model/weights/KV-cache in the picture at all.** This is the
> pattern to reach for next time a kernel needs `ncu` data and a full end-to-end run won't fit.

> **STATUS UPDATE (2026-07-22, same session, continued): attempted a kernel-level fix on
> `lm_flash_attn_f32_split` — correct, verified bit-exact, measurably NO real speedup. Reverted.** With the
> probe now able to reach real decode-shape data on the actual 3060, `ncu --set full` on
> `lm_flash_attn_f32_split` at production shape showed the same class of signature Q6_K had before its fix:
> "Est. Speedup 46.01%" from warps stalled on an L1TEX scoreboard dependency — each key's K·Q product load
> sits immediately before a `__syncthreads()`, so the whole block stalls on the slowest lane's load before the
> tree reduction can even start.
>
> **Fix attempted**: software-pipelined the per-key loop with double-buffered shared memory (ping-pong
> `buf0`/`buf1`) — issue key k+1's load right after key k's data is confirmed ready, so its latency overlaps
> with key k's reduction/softmax/V-load/accumulate instead of sitting fully exposed. Same per-key math, same
> accumulation order, only WHEN each load is issued changes — same template as the Q6_K fix.
>
> **Correctness verified rigorously before measuring speed** (this project's own rule: never trust a
> plausible-looking kernel change without proof): built OLD and NEW `Ptx/` directories (`git show HEAD:...` for
> the pre-change PTX) and ran the identical probe harness against both, dumping raw output to disk and
> `cmp`-ing byte-for-byte. **Bit-identical at every kvLen tested (130, 193, 512, 2000, 8192)** — the
> restructuring is provably correctness-neutral, not just "close."
>
> **Speed: no real win, confirmed two ways.** (1) Direct `ncu gpu__time_duration.avg` on the split kernel
> alone, old PTX vs new PTX, same inputs: **48.6-50.5us both ways** — differences smaller than run-to-run
> noise. (2) Clean end-to-end `TextDecodeThroughputBenchmark`: Llama graph-on 155.28→155.72, Qwen3-4B
> graph-on 69.4-70.09→70.33, graph-off 60.64→61.87 — all within normal run-to-run variance, no attributable
> change. **Reverted** (`flash_attn_f32_split.cu`, its `.ptx` in both locations, `CudaKernels.cs`'s shared-mem
> sizing — `git checkout` on all four, reverification: full suite back to the same 133/137 baseline).
>
> **Why the textbook fix didn't pay off (worth understanding, not just noting)**: double-buffering only hides
> load latency behind *other independent work*, and there wasn't much to hide it behind. The "other work" per
> iteration (a tree reduction that itself hits another `__syncthreads()` almost immediately, plus a handful of
> softmax ALU ops) is far shorter than a global-memory round trip, and achieved occupancy at the tuned split
> count is only ~19-38% — too few concurrently-resident warps for warp-level parallelism to cover the gap
> either. Prefetching exactly one iteration ahead only buys "one iteration's worth of non-memory work" as
> overlap headroom, nowhere close to hiding a few-hundred-cycle memory latency. This is a materially different
> situation from Q6_K's GEMV kernel, where each thread's work was embarrassingly parallel (no cross-thread
> reduction, no barriers) and there was abundant independent ALU work per thread to hide loads behind — the
> "move loads earlier" pattern generalizes only when there's real independent work to overlap with, not to
> every latency-bound kernel. A real fix for this kernel would need either much deeper prefetching (multiple
> iterations, awkward with already-small per-split chunk sizes), materially higher occupancy, or a different
> kernel design entirely (e.g. a warp-per-key layout with no cross-thread barrier at all) — a bigger rewrite,
> not attempted here.
>
> **Confirms this doc's standing conclusion once more**: the remaining gap requires a genuine
> attention-kernel-design change, not a targeted latency fix — the same "GEMV memory-access-pattern redesign"
> scope this doc has flagged for the GEMV kernels applies to attention too, and is out of scope for a
> single-session targeted fix.

> **STATUS UPDATE (2026-07-22, same session, continued): decode-step attribution via `HARTSY_PROFILE_SYNC`
> (zero VRAM overhead — sidesteps the ncu/nsys OOM entirely) + a priced, not-built, `Permute0213` lever.**
> Ran the CLI with `HARTSY_PROFILE=1 HARTSY_PROFILE_SYNC=1` (eager path — HARTSY_PROFILE_SYNC forces a stream
> sync per op, which CUDA forbids during graph capture, so this only works eager) on Llama-3.2-1B, full
> 128-token run: `Linear 494.7ms (56.5%), Permute0213 136.0ms (15.5%), RopeInterleaved 86.5ms (9.9%), RmsNorm
> 73.5ms (8.4%), H2D_MISS_SMALL 70.5ms (8.0%), Silu 14.9ms (1.7%)`. `Linear` dominating matches "GEMVs are
> already near-roofline" (confirmed separately this session: isolated `mul_mat_vec_q8_0_f32` at Llama's real
> gate/up shape hit **89.2% of both memory and compute roofline** via `ncu --set full` — same
> already-optimal signature as Q4_K found earlier, nothing to fix there either).
>
> **`Permute0213` stood out as a genuine, well-understood — not just latency-hidden — lever**: at decode
> (T=1), it transforms `[1,1,H,D] -> [1,H,1,D]`, and those two shapes are **provably bit-identical in memory**
> (both linearize to `h*D+d`) — the kernel is mathematically a no-op copy, called 4×/layer (Q/K/V in,
> attention-output out) purely to satisfy a shape label FlashAttention's indexing expects. In principle,
> eliminable entirely (a relabeled view, zero GPU work) rather than just speeding up — a different class of
> win than the two reverted latency-hiding attempts above.
>
> **But the 15.5% figure is eager-path-and-sync-profiler-shaped, not what the gap is measured against.**
> `HARTSY_PROFILE_SYNC` drains the stream after every op, so tiny ops (Permute 16us avg, RmsNorm 17us, RoPE
> 21us, Silu 7us — all landing in the same narrow band despite very different actual work) are dominated by
> that forced-sync floor, not their real GPU execution time. Graph-decode (the actually-benchmarked "ours"
> config) already eliminates that floor — that's the mechanism behind the 134→155 tok/s graph-on win. So the
> real question is the permute kernel's **own GPU execution time**, isolated, at the exact decode shape —
> not its eager-profiled wall-clock share.
>
> **Priced it before building anything** (a repeat of this session's now-standard discipline after two
> profiler-artifact false leads): a minimal probe (`permute-probe`, same no-model-loading pattern as the
> attention/Q8_0 probes) calling `Permute0213([1,1,32,64] -> [1,32,1,64])` — Llama's real decode shape —
> measured via `ncu gpu__time_duration.avg`: **~1.9-3.5us/call (median ~2.1us)**. At 4 calls/layer × 16
> layers = 64 calls/step, that's **~134us/step out of a ~6.45ms graph-on decode step (~2.1%)** for
> Llama-3.2-1B; extrapolated for Qwen3-4B (144 calls/step at a larger D=128, roughly 2.5x the per-call bytes)
> lands in a similar **~2-2.5%** range. Below the threshold where the real engineering risk is clearly worth
> it: eliminating this cleanly (not just fast-pathing the kernel, but avoiding the allocation+copy entirely)
> means teaching the GPU activation-pointer cache about aliased/borrowed views shared across two independently-
> disposed `Tensor` objects — new lifetime semantics in `GpuTransferHelper`'s pointer-cache, touched by every
> architecture, both eager and graph-decode paths. A ~2% ceiling doesn't clear that bar. **Not built.**
>
> **Session-wide pattern now confirmed three times** (CUDA-graph arena, attention-kernel prefetch, and now
> this pricing check that stopped a build before it started): an ncu/nsys number measured in one execution
> context (full-VRAM profiling, eager-path sync-forced timing, or API-trace instrumentation) does not
> automatically transfer to the graph-decode config the benchmark and the llama.cpp comparison actually use.
> Every real, kept win this session (Q6_K load-order fix, both split-K formula fixes) was verified against
> the actual benchmarked config, not just a plausible profiler reading in a different one — worth keeping as
> the standing rule for any future work on this doc.

> **STATUS UPDATE (2026-07-22, same session, continued): QKV + gate/up projection fusion — built, verified,
> shipped. Small but real win, smaller than the isolated-kernel pricing predicted (a real, understood cost —
> not another profiler-artifact false lead).** Direct answer to "why is llama.cpp faster, we should be able to
> match C++": it isn't a language issue (the GPU runs identical PTX regardless of what launched it, and
> graph-decode already erases most host-dispatch overhead) — llama.cpp fuses Q/K/V into one projection and
> gate/up into one, so every GEMV call is large enough to fill the GPU. We issued them separately: `ncu` on the
> real Llama-3.2-1B K/V projection shape (N=512) showed **0.38 full waves across 28 SMs** — the same
> class of occupancy stall split-K fixed for attention, this time in the GEMV dispatch itself.
>
> **Implementation** (`GenericTransformer.cs`): at load time, `ConcatRows` byte-concatenates the separate
> Q/K/V (and gate/up) weight/bias tensors into one fused tensor per layer — safe generically across every
> dtype including block-quantized formats, since a GGUF/our quant block never spans two output rows (each row
> is an independent whole number of blocks; confirmed against every fused-GEMV kernel's own block-layout
> assumption). Gated to `hasOwnKv` (skips Gemma-4 KV-sharing/MLA layers, which never reach this code — no
> explicit exclusion needed) and a dtype-match guard (mixed-quant GGUF schemes occasionally assign a
> different quant type per tensor by shape; falls back to the untouched separate-projection path if Q/K/V
> or gate/up ever disagree). One fused `Linear`/`QuantizedMatMul` call replaces three (QKV) or two (gate/up),
> then `IBackend.SliceLastDim` (an existing primitive, already used elsewhere for exactly this "split a fused
> tensor into contiguous chunks" job — not new kernel code) splits the output back into separate q/k/v or
> gate/up tensors. QK-norm (Qwen3) is unaffected — it's applied to the already-sliced q/k tensors exactly as
> before; fusion only changes the projection dispatch, nothing downstream. Wired into all three real decode
> paths: eager `Forward`, graph-decode `ForwardGraphStep`, and the HTTP server's batched `ForwardBatchDecode`.
>
> **Correctness**: CPU-backend `HartsyInference.LLM.Tests` (131/131, exercises `SliceLastDim`'s default
> fallback with synthetic weights), CUDA `HartsyInference.Cuda.Tests` (153/157, same 4 pre-existing unrelated
> failures as every checkpoint this session), and a direct CLI A/B on real checkpoints — Llama-3.2-1B and
> Qwen3-4B both produce **character-for-character identical output**, graph-decode on vs off, and Qwen3-4B's
> output matches the exact text from an earlier unrelated session checkpoint (independent confirmation).
>
> **Measured, clean `TextDecodeThroughputBenchmark` runs, repeated for stability**: Llama-3.2-1B graph-on
> **156.0 → 157-158 tok/s (~1-1.5%)**, graph-off **128.5 → 138-140 tok/s (~7-9%, bigger — eager has no
> graph-capture launch-overhead baseline already subtracting from the win)**; Qwen3-4B graph-on **69.9 → 70.2-
> 70.6 tok/s (~1%)**, graph-off **61.7 → 61.9-62.1 (~0.5%)**. Smaller than the isolated `ncu` GEMV-kernel-time
> pricing predicted (~4% for Llama) — the gap is real, not noise: that estimate only measured the fused-GEMV
> saving, not the cost of `SliceLastDim` splitting the output back apart afterward (3 slice calls for QKV, 2
> for gate/up, each a real kernel launch), which eats back a meaningful fraction of the raw GEMV saving. Kept
> anyway — it's a real, consistent, correctness-verified, low-risk positive, not another reverted false lead.
>
> **Known tradeoff, not yet addressed**: the fused weight is a byte-level copy, not a replacement — the
> original separate Q/K/V and gate/up weights stay resident too (both `EnumerateWeights`-preloaded), so this
> roughly doubles VRAM for the fused tensors specifically (~680MB extra for Llama-1B's non-lm_head weights,
> proportionally similar for other sizes). No OOM observed in any testing this session, but freeing the
> originals once fusion succeeds (they'd need their widths cached as plain ints first, since a few call sites
> read `.Shape[0]` off them) is a real, scoped follow-up if VRAM pressure becomes an issue on a smaller card.

> **STATUS UPDATE (2026-07-22, same session, continued): R4 row-interleaved GEMV redesign — investigated,
> premise refuted by measurement, NOT built.** This was handed over by the parallel `performance-grind` agent's
> `INFERENCE_ACCEL_GRIND.md` (its H5 item, explicitly delegated to "the LLM agent"), citing this doc's own
> stale "~22% of bandwidth vs llama.cpp's ~80%" estimate as justification. Re-measured first, per this doc's
> own rule of not building on stale numbers.
>
> **Fresh `ncu --set full` on the real Q4_K `mul_mat_vec_q4k_f32` kernel** (standalone probe, no model load,
> Qwen3-4B's actual `ffn_gate.weight` shape K=2560 N=9728): **DRAM Throughput 68.65%, Compute (SM) Throughput
> 74.59%, Memory Throughput 76.03% — compute and memory are co-limiting, not the ~22%-bandwidth-bound picture
> the delegated task assumed.** That 22% number predates this session's fused-GEMV work (Phase 0-era);
> superseded. At 68.65% DRAM with ALU already at 74.6%, a perfect coalescing fix can only push the bottleneck
> fully onto ALU (per-element nibble-unpack/dequant, which is layout-invariant) — the real achievable win is a
> fraction of the naive "41.5% Est. Speedup" `ncu` flags on the load instructions, which is an isolated-issue
> number of exactly the kind that already burned the CUDA-graph-arena and attention-prefetch attempts earlier
> this session.
>
> **Localized the flagged inefficiency anyway** (`ncu --page source --print-source sass`, matching raw
> addresses against the kernel source): the two `LDG.E.128.CONSTANT` instructions carrying the reported excess
> sectors are the activation-vector reads (`xa`/`xb2`, the two `float4` loads of `input`), not the
> `get_scale_min_k4` scattered scale-byte reads originally suspected. Root cause: `blockDim.y = WARPS_PER_BLOCK
> = 8` warps per block all cover the same batch row `m` (different output rows `n`), so all 8 independently
> re-read the identical K-float input row — L1 absorbs most of it (94.83% hit rate) but not all, and the
> residual shows up as real L2/DRAM sector traffic. This is the same redundancy R4's "input-vector reuse"
> half targets.
>
> **Built and measured the direct fix for that redundancy** (no full R4 row-repack needed): a second kernel
> entry point (`mul_mat_vec_q4k_f32_shmem`) that cooperatively stages the shared input row into shared memory
> once per block (`__syncthreads()`), then all 8 warps read from shared instead of re-hitting global/L1 —
> gated on `K * sizeof(float)` fitting a 96 KB opted-in shared-memory budget (`cuFuncSetAttribute`,
> `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`), falling back to the untouched original kernel otherwise.
> Verified correct: all 7 `FusedGemvGroundTruthTests` pass bit-exact. **Measured net regression**: clean
> probe timing at the same real shape went from **63.05us/call (baseline) to ~70.2-70.7us/call (~11% slower,
> reproduced 3×)** — the block-wide `__syncthreads()` barrier serializes all 8 warps behind the slowest
> loader, which costs more than the redundant-read savings recover. **Reverted.**
>
> **Conclusion, reported back to the coordination doc**: the R4 premise (bandwidth-bound decode GEMV) doesn't
> hold at current shapes post-fusion — this kernel is compute/memory co-limited, the input-vector-reuse half
> of R4 was tried directly (shared-memory staging) and measured a net loss, and the coalescing half would only
> capture a fraction of ALU-capped headroom even if built. Not worth the full row-interleaved weight-repack
> rewrite (new load-time layout transform, new kernel, per-quant-type surface) on this evidence. If decode GEMV
> throughput needs another pass later, the next real lever is the FP16-activation-pipeline idea already queued
> in the dotLLM list below (halves activation bytes, independent of this finding), not R4.

> **STATUS UPDATE (2026-07-22, same session, continued): full-decode-step re-profile — confirmed the
> remaining ~1.21x gap is real GEMV bandwidth efficiency (~68% vs llama.cpp's ~70%), NOT a system/host-overhead
> problem. Chased and ruled out two false leads first (both documented in detail because they looked
> compelling before being tested) — re-enabled the dp4a int8-activation GEMV path as a real, small,
> re-measured win.**
>
> **Full-decode `nsys` re-profile (Llama-3.2-1B, real production path, graph-on)**: `mul_mat_vec_q8_0_f32`
> is 71.3% of total decode GPU time across 4 shape buckets (gate/up-fused N=16384 41%, O-proj+down-proj
> N=2048 31%, lm_head N=128256 20%, QKV-fused N=3072 9%) — all four cluster at **235-270 GB/s effective**
> (3060 peak 360 GB/s), i.e. uniformly ~65-75% of peak with no single broken shape, confirming (with real
> production shapes, not just the isolated `ffn_gate` probe) the R4 block's conclusion above: this is a
> genuinely near-roofline kernel, not one hiding an easy fix.
>
> **False lead #1 — "half of decode wall-time is host overhead, not GPU work" — REFUTED, was a graph-replay
> profiler undercount.** A `cuda_gpu_kern_sum` on a graph-decode-ONLY trace (isolated via a standalone probe,
> same no-model-loading-under-profiler pattern used all session) showed only 260 `mul_mat_vec_q8_0_f32`
> instances and 13.7% overall GPU-busy time across the trace span — `nsys --trace=cuda` does not reliably
> decompose CUDA-graph-replay into one timeline entry per node the way it does for eager launches (the same
> class of graph-tracing distortion that burned the CUDA-graph-arena false lead earlier this session — now
> confirmed as a recurring pitfall of profiling this engine's graph-decode path specifically, not a one-off).
> The eager-inclusive trace's 71.3%/48,478-instance figure is the trustworthy one.
>
> **False lead #2 — "the GPU's PCIe link is downgraded to Gen1, explaining a ~6ms D2H sync tax" — REFUTED,
> was an idle-vs-load measurement confound.** `cuda_api_sum` on the same trace showed `cuMemcpyDtoH_v2` (the
> per-token sampled-token-id readback) averaging **5.95ms/call** across 516 calls — suspiciously large for a
> 4-byte transfer. `lspci -vv` on the 3060's slot showed `LnkSta: Speed 2.5GT/s (downgraded)` against an
> `LnkCap` of 8GT/s, which looked like a real hardware fault (and would explain a fixed per-call tax
> independent of workload, matching the tight 3.6-6.5ms range observed). Tried and ruled out `CU_CTX_SCHED_SPIN`
> (no primary-context scheduling flag was ever set; added one, zero measured effect, reverted — a context
> **not** shared with any other library-initiated retain in this process, so the flag did take) and GPU
> clock locking (`nvidia-smi -lgc`, also zero effect) before questioning the PCIe premise itself. **Killed by
> three independent checks**: (1) decode-window weight streaming never crosses PCIe at all — it's
> VRAM-to-SM, so even a real Gen1 downgrade could only tax the 4-byte token-id copy, not decode throughput;
> (2) re-read `LnkSta` *during* an active 800-token run — **8GT/s**, full speed; the "downgraded" reading was
> the driver's normal idle power-save state (`lspci` was run between test runs), not a fault; (3) PCIe ASPM
> L1 exit latency is documented at <4us — nothing in the PCIe spec produces a consistent multi-millisecond
> tax. Reconciled what the 5.95ms actually is: `cuMemcpyDtoH` is blocking, so it waits for that token's real
> GPU work (weight streaming) to drain — 1.32GB (Llama-1B Q8_0 file size) × 159 tok/s ≈ 210 GB/s sustained ≈
> textbook memory-bound decode, consistent with the ncu-measured 68.65% DRAM finding above. There is no
> separable host-overhead tax to cut; the wait time **is** the bandwidth-bound work. (`CU_CTX_SCHED_SPIN`
> change reverted — no measured gain, and spin-wait burns a CPU core for nothing, this doc's standing
> revert-on-no-gain rule.)
>
> **Real, positive result found while re-checking assumptions: the existing dp4a int8-activation GEMV path
> (`HARTSY_DP4A_ON`, opt-in, `CudaBackend.cs` ~line 511) was carrying a stale "not faster here" comment.**
> Re-measured on Qwen3-4B (the model that actually has Q4_K tensors — Llama-3.2-1B is Q8_0-only and never
> exercises this branch) post this session's split-K + QKV/gate-up-fusion changes: **71.35 → 72.77 tok/s,
> +2%, reproduced 3x each way, clean separation from the ~0.1-0.3 tok/s noise band in each set.** Small, but
> real and consistent with the ncu finding that this kernel is compute/memory co-limited (74.6% ALU) — cutting
> per-element ALU work via dp4a int8 dot-products should help, and now measurably does, whatever the
> now-outdated comment claimed. Comment corrected in place; **not** flipped to default-on this session — no
> dedicated correctness test exists for this path yet and the win hasn't been swept across shapes/quant types/
> models. Scoped as the next real kernel-level lever; full design handoff: `LLM_GEMV_KERNEL_HANDOFF.md`.
>
> **Bottom line for "why are we still ~1.2x slower than llama.cpp"**: decode is memory-bandwidth-bound weight
> streaming for both engines (batch=1, GPU saturated, not host-idle) — we sustain ~65-75% of the 3060's peak
> VRAM bandwidth across every real production GEMV shape, llama.cpp somewhat more. That efficiency delta,
> which lives entirely inside the GEMV kernel (ALU-side dequant cost on a co-limited kernel, per the R4
> investigation and the dp4a result above), is the whole gap. Not a language issue, not a host-dispatch issue,
> not a system/PCIe issue — confirmed by direct measurement, not assumption, on all three counts this session.

> **STATUS UPDATE (2026-07-22, later session): FASTER THAN llama.cpp — the dp4a kernel grind from
> `LLM_GEMV_KERNEL_HANDOFF.md` executed to completion. Llama-3.2-1B Q8_0 158.9 → ~195 tok/s, Qwen3-4B
> Q4_K_M 70.8 → ~90-92 tok/s (RTX 3060, graph-on); llama-cpp-python measured 190.3 / 85.6 on the same
> box — both models now BEAT the Python/llama.cpp baseline.** What landed, in handoff-task order:
>
> 1. **Ground-truth correctness test for the int8-activation path** (`Dp4aGemvGroundTruthTests`, 9 cases
>    covering Q4_K/Q6_K/Q8_0 × shapes incl. real production K × batch × bias): three gates per case —
>    (a) exact CPU bit-replica of the Q8_1 activation quantizer (amax/127 scale, round-nearest-even,
>    clamp) dotted against the CPU-dequantized weight, agreeing to float-accumulation noise (~1e-7 avg,
>    tolerance 5e-4); (b) an ANALYTIC error bound vs the unquantized reference — per-element Q8_1
>    rounding error ≤ scale/2, so |gpu − f32ref| ≤ Σ|w_i|·(scale_blk(i)/2) computed from the actual
>    data (derived, not guessed; zero violations); (c) an engagement check (dp4a output must differ
>    from the float kernel's — catches silent dispatch fall-through).
> 2. **New `mul_mat_vec_q8_0_q8_1.cu`** — Q8_0×Q8_1 dp4a GEMV (Q8_0 is symmetric: no min/int-sum term).
>    34-byte blocks are only 2-byte aligned → operands assembled from u16 loads (llama.cpp get_int_b2
>    pattern). Layout that measured best: warp-per-row, 8 blocks/warp-iteration, each lane owning an
>    8-elem chunk (2 dp4a) — the first cut (1 dp4a/lane, 4 blocks/iter) left ~10-20% on the table.
>    Isolated (probe, weight resident): gate/up 145→113 µs, down 84→63.5 µs, lm_head 1084→837 µs
>    (93% of peak DRAM bandwidth) vs the float kernel.
> 3. **New `mul_mat_vec_q6k_q8_1.cu`** — Q6_K×Q8_1 dp4a GEMV (6-bit unpack via whole-word nibble+qh
>    merge, per-byte −32 via `__vsub4`, signed per-16 scales, no min term). Final layout: 32 lanes =
>    2 halves × 2 ql-pairs × 8 u16-positions, each lane loading its ql bytes ONCE and processing both
>    nibble planes. Isolated: down 116→75 µs (1.57×), lm_head 1573→953 µs (1.63×, 93% of peak) —
>    the single biggest win; Q6_K is 22% of Qwen3-4B's weights (lm_head + half the ffn_down/attn_v).
> 4. **Q4_K dp4a kernel rewrite** (the pre-existing `mul_mat_vec_q4k_q8_1.cu`): whole-word nibble
>    extraction (`(word >> shift) & 0x0F0F0F0F` — one op pair for all four byte-lanes, replacing
>    per-byte shift/or chains) + word-based scale/min extraction (3 aligned u32s + shifts replacing
>    ggml's per-byte `get_scale_min_k4` loads) + fused d/dmin u32 load. Isolated: gate/up 119→92 µs
>    (1.33× vs float, 65→83% of peak), QKV 44→35 µs, down 67→59 µs — the fix that turned Q4_K dp4a
>    from ~neutral (the +2% of the earlier session) into a clear win.
> 5. **Activation-quantize kernel occupancy** (`quantize_activation_q8_1_f32.cu`): 8 warps/block
>    instead of one-warp blocks. Measured ~neutral e2e (launch-geometry change, no added complexity).
> 6. **Default flipped**: dp4a is now the standard profile (`CudaBackend.EnableDp4aGemv`,
>    `EnvSwitch`-gated `HARTSY_DP4A_ON=0` kill-switch, resolved once at construction — the old code
>    read the env var on every Linear call). Dispatch covers Q4_K/Q6_K (K%256) and Q8_0 (K%32) at
>    M≤8; everything else unchanged.
>
> **Tried and REVERTED (so the next reader doesn't re-try them)**: (a) block-per-row K-split variants
> (llama.cpp mmvq's occupancy shape, warps splitting one row's super-blocks + shared-mem combine) —
> built for Q4_K, swept W∈{2,4,8} at all three production shapes: net LOSS everywhere (up to −35% at
> W=8; ~equal-at-best on the long-K ffn_down it targeted) once the whole-word unpack landed, and −1.3%
> e2e on Qwen3 under the auto heuristic; (b) a byte-shared Q4_K layout (each lane processing both
> nibble planes of its uint2, 2 super-blocks/warp-iter) — ~10% slower on ffn_down, flat elsewhere.
> Both removed; the plane-per-lane + whole-word-unpack form is the keeper.
>
> **Correctness gates (all green)**: full `HartsyInference.Cuda.Tests` 166/166 (dp4a default-ON — the
> pre-existing fused-GEMV/quantized-matmul ground-truth suites now exercise the dp4a path at decode
> shapes and pass unchanged; note the 4 "pre-existing failures" of the earlier session do NOT reproduce
> under the documented `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0` pin — clean 157/157 was
> re-established BEFORE any change); CPU `HartsyInference.LLM.Tests` 131/131; real-checkpoint CLI A/B
> (`hartsy text`, greedy, 60 tokens): **byte-identical output across dp4a on/off × graph on/off** on
> BOTH Llama-3.2-1B Q8_0 and Qwen3-4B Q4_K_M (the Q8_1 rounding perturbation flipped no greedy argmax
> on either model — llama.cpp ships this exact numerics tradeoff as its only decode path). One
> pre-existing footnote: `hartsy text` WITHOUT `--low-vram-quant` OOMs loading Qwen3-4B on the shared
> 12GB card (lm_head F16-cast blowup in `ProjectLogits`) — reproduced identically with dp4a and graphs
> OFF, i.e. unrelated to this work; use `--low-vram-quant` (the benchmark harness already does).
>
> **Measured (`TextDecodeThroughputBenchmark`, 3060 pinned, medians of 5, final default-config run)**:
> Llama-3.2-1B Q8_0 graph-on 158.91 → **197.28** tok/s (+24%), graph-off 140.68 → 162.61 (+16%);
> Qwen3-4B Q4_K_M graph-on 70.82 → **92.21** (+30%), graph-off 62.47 → 75.58 (+21%). llama-cpp-python
> on the same box/pin/prompt: 190.34 (best documented quiet run; a same-hour back-to-back run measured
> 173.6 under desktop contention) / 86.83 (fresh, stable) → **we are now 1.04-1.14× FASTER on Llama
> and 1.06-1.08× FASTER on Qwen3** — faster than the Python baseline's best number on both models.
> (Desktop/rustdesk GPU contention can swing either engine ±20%+ — always compare back-to-back runs
> from a quiet GPU; see `LLM_THROUGHPUT_BENCHMARK.md` for the final side-by-side table.)
> Mechanism, for the record: the fused float GEMVs were compute/memory CO-limited (74.6% ALU on Q4_K);
> dp4a packs 4 int8 MACs/instruction and the int8 activation quarters activation-read bytes, pushing
> every major GEMV to 83-93% of the 3060's DRAM roofline — the ALU-side dequant cost this doc's 2026-07-22
> handoff identified as the whole remaining gap is now paid.

> **STATUS UPDATE (2026-07-22, later session, round 2): popular-model sweep — dp4a coverage completed
> (Q4_0/Q5_0/Q5_K), a stale lm_head dispatch gate that made two top-Ollama-family models 4-5× slower
> than llama.cpp fixed, split-K attention extended to sliding-window/softcap, and a pre-existing
> GLM-4 partial-rotary graph-decode CORRUPTION bug found and fixed.** Motivation: mapped Ollama's
> top-downloads list (llama3.x ~220M pulls, deepseek-r1 90M, gemma3/2/4 ~86M, qwen2.5 ~54M, qwen3,
> mistral, phi, gpt-oss) against engine coverage, downloaded three small representative checkpoints
> (gemma-3-1b-it, DeepSeek-R1-Distill-Qwen-1.5B, qwen2.5-0.5b-instruct, all Q4_K_M) and benchmarked
> them plus the local GLM-4-9B against llama-cpp-python, same protocol as round 1.
>
> **What the sweep found and what was fixed, in impact order:**
> 1. **`ProjectLogits`' fusedHead gate used a blanket `HiddenSize % 256 == 0`** — wrong for Q8_0/Q4_0/
>    Q5_0 heads, which only need K % 32. Every model with hidden ≡ 128 (mod 256) — qwen2.5-0.5b (896),
>    gemma3-1b (1152) — silently dequantized its ENTIRE lm_head to F16 and ran cuBLAS on it EVERY
>    token; `nsys` measured `dequant_q8_0_to_f16` at **55% (qwen2.5) / 62% (gemma3) of total decode
>    GPU time**. One-line gate fix (mirror LinearImpl's per-dtype divisibility):
>    **gemma3-1b 36.9 → 88.9 tok/s, qwen2.5-0.5b 74.1 → 158.9 tok/s (both ~2.2×)**.
> 2. **New dp4a kernels `mul_mat_vec_q4_0_q8_1` / `mul_mat_vec_q5_0_q8_1`** (offset quants: fixed −8/−16
>    folded into the packed weights via `__vsub4`, no min/int-sum term; Q5_0's high-bit word byte-
>    assembled — its 22-byte stride alternates 4-alignment) **and `mul_mat_vec_q5k_q8_1`** (Q4_K-style
>    min term + high-bit plane injected whole-word). Q5_0 matters far more than its name suggests:
>    llama.cpp's Q4_K_M scheme substitutes it on every tensor whose K isn't a multiple of 256, which is
>    MOST tensors on odd-hidden-size models — **46% of gemma3-1b, 62% of qwen2.5-0.5b, 12% of
>    GLM-4-9B by element count**. Q4_0 covers legacy Ollama default tags (llama3 era).
>    `Dp4aGemvGroundTruthTests` extended to 18 cases (exact quantizer simulation + analytic bound +
>    engagement gates, all dtypes) — all pass; raw-block synthesis for the read-only Q4_0/Q5_0 codecs.
> 3. **Split-K decode attention no longer excludes sliding-window/softcap** (`flash_attn_f32_split.cu`
>    gains the monolithic kernel's window clamp + `cap·tanh` transform; sink/ALiBi stay monolithic).
>    gemma3-1b decodes with FOUR query heads — the monolithic path put 4 blocks on 28 SMs. Also
>    engage split at kvLen ≥ 32 (was 128) when baseBlocks ≤ SM count. New
>    `FlashSplit_WindowSoftcap_MatchesMonolithic` test (4 gemma2/3-shaped cases, split vs monolithic
>    ≤ 5e-8). Measured alone: gemma3 35.1 → 36.9 (+5%) — real but small; the lm_head gate above was
>    the actual elephant. Kept (correct, tested, and required for any future windowed graph decode).
> 4. **GLM-4 graph decode was producing DEGENERATE output** (repetitive loops — caught only because
>    this sweep ran output A/B across dp4a×graph configs on every model): `BuildRopeTableDevice` filled
>    only the first rotaryDim table entries per row, leaving the rest as uninitialized alloc garbage —
>    and the graph-decode rope kernels always walk headDim/2 pairs, so partial-rotary models (GLM-4:
>    rotaryDim 64 < headDim 128; the first such model to pass the graph-eligibility gate) rotated their
>    untouched dims by garbage angles. Fixed: identity (cos=1, sin=0) entries past rotaryDim/2.
>    **GLM-4-9B graph-on output is now byte-identical to eager** (was garbage; throughput 39.2 tok/s
>    unchanged — this was a correctness fix, and it makes GLM's graph-on number legitimate for the
>    first time).
>
> **Where the fleet stands vs llama-cpp-python (same box, 3060 pinned, same-day baselines):**
> | Model | ours graph-off | ours graph-on (best) | llama-cpp-python | ratio |
> |---|---|---|---|---|
> | Llama-3.2-1B Q8_0 | 164.5 | **197.3** | 190.3 | **1.04× FASTER** |
> | Qwen3-4B Q4_K_M | 76.0 | **92.9** | 85.6-86.8 | **1.07× FASTER** |
> | DeepSeek-R1-Distill-Qwen-1.5B | 137.6 | **161.1** | 180.1 | 1.12× slower |
> | qwen2.5-0.5b-instruct | 113.8 | **158.9** | 322.7 | 2.03× slower (was 4.3×) |
> | gemma-3-1b-it | **88.9** | ineligible | 190.9 | 2.15× slower (was 5.4×) |
> | GLM-4-9B-0414 | 35.6 | **39.2** | 47.8 | 1.22× slower |
>
> **Remaining known gaps, scoped for a next session (in expected-value order):**
> (a) **gemma-family graph decode** (~86M Ollama pulls): eligibility needs device-side sliding-window
> in `FlashAttentionDev` (the split KERNEL already supports window/softcap after this round — only the
> Dev dispatch + eligibility gate + per-layer window plumbing remain), dual local/global RoPE tables
> (`RopeLocalTheta`), and per-layer global-vs-local selection. qwen2.5-0.5b's +40% graph-on delta
> suggests gemma3-1b ~89 → ~120+.
> (b) **Tiny-model graph overhead**: qwen2.5-0.5b's kernels sum ~3ms/token but graph-on decode takes
> 6.4ms — per-node replay overhead dominates at this scale (dp4a's 3 alloc + 3 free nodes per Linear
> are part of it; the round-1 capture-arena revert was measured on a 4B model where this didn't bind —
> worth re-testing at 0.5B before assuming it still doesn't). llama.cpp's 322 tok/s shows ~2× headroom.
> (c) **DeepSeek-distill residual 1.12×**: same architecture as the beaten Qwen3 — likely the same
> tiny-model overhead scaled down; no dedicated lever identified yet.
> (d) **gpt-oss (MXFP4+MoE+sink)**: untestable on this 12 GB box (20B ≈ 13 GB); MXFP4 has no fused
> GEMV of any kind — first step there is a float fused kernel, then dp4a, then sink support in split-K.
> (e) Q2_K/Q3_K/IQ-quant fused kernels: still fall to the slow dequant path; no popular default tags
> hit them, so deprioritized.

> **STATUS UPDATE (2026-07-22, later session, round 3): Gemma-family CUDA-graph decode landed —
> gemma2-2b now BEATS llama-cpp-python (129.5 vs 121.1 tok/s, 1.07×); gemma3-1b 88.2 → 115.1 tok/s
> (+30%, gap 2.15× → 1.66×).** This was roadmap item (a) from round 2: the split-K attention kernel
> already handled sliding-window/soft-cap with a device position (round 2's kernel work), so this round
> was pure plumbing + eligibility:
> 1. **`IBackend.FlashAttentionDev` gains `softcap`/`slidingWindow` params** (optional, appended —
>    existing callers unchanged). CudaBackend passes them through all three paths (split, monolithic
>    fallback, host-eager fallback); both kernels read kvLen/qOffset from the device position BEFORE
>    computing the window clamp, so the per-layer window is a capture-constant and the clamp tracks the
>    replayed position correctly.
> 2. **Dual local/global RoPE tables for graph decode** (`_graphDecodeCosLocal`/`SinLocal`, built with
>    `RopeLocalTheta` alongside the global pair in `EnsureRopeTableForGraphDecode`);
>    `ForwardGraphDecodeStep` selects per layer via `IsGlobalLayer`, mirroring the eager dual-RoPE path.
> 3. **`Layer.ForwardGraphStep` passes the per-layer window** (`SlidingWindow` on non-global layers,
>    0 on global — same rule as eager) **and `AttnLogitSoftcap`** to `FlashAttentionDev`.
> 4. **Eligibility gate**: `SlidingWindow == 0`, `AttnLogitSoftcap == 0`, and `RopeLocalTheta == 0`
>    exclusions removed; `HeadDimSwa == 0` added (Gemma-4's per-layer SWA head dim stays excluded —
>    the graph rope tables are built at a single head dim).
>
> **Correctness (the GLM-4 lesson applied: no arch enters the gate without a real-checkpoint e2e A/B):**
> gemma-3-1b-it AND gemma-2-2b-it (downloaded for exactly this check — softcap + FinalLogitSoftcap +
> single-theta, a different config corner than gemma3) both produce **byte-identical greedy output
> graph-on vs graph-off** on the float path (dp4a on either side shows only the known bounded argmax
> tie-flips, both texts coherent). Full `HartsyInference.Cuda.Tests` 179/179, `HartsyInference.LLM.Tests`
> 132/132; the 6-model regression benchmark shows every previously-benchmarked model unchanged within
> noise (Llama 196.0, Qwen3 92.6, DeepSeek 160.5, qwen2.5 157.8, GLM-4 39.2), and gemma graph-on runs
> report the required ≤1 mid-decode D2H syncs (true single-launch replay).
>
> **Measured (RTX 3060 pinned, medians of 5, llama-cpp-python same-day):**
> - gemma-2-2b-it Q4_K_M: eager 111.5 → graph-on **129.51 tok/s** vs llama-cpp-python 121.08 →
>   **1.07× FASTER** (third model past the Python baseline; this arch had NO graph decode this morning).
> - gemma-3-1b-it Q4_K_M: eager 88.2 → graph-on **115.09 tok/s** vs 192.34 → 1.67× slower (was 5.4×
>   at the start of the day). Its residual gap is NOT gemma-specific anymore: with kernels near-roofline
>   and graphs on, it sits in the same tiny-model per-node-overhead bucket as qwen2.5-0.5b (2.0×) and
>   the DeepSeek distill (1.12×) — that bucket (graph replay/node overhead at small per-kernel work,
>   incl. the dp4a per-Linear alloc/free nodes) is the next real lever, roadmap item (b) in round 2.

> **STATUS UPDATE (2026-07-22, later session, round 4): tiny-model decode overhead — three targeted
> fixes, every model in the 7-model fleet improved, nothing regressed.** A `--cuda-graph-trace=node`
> nsys pass on qwen2.5-0.5b graph decode (the mode that actually decomposes graph replays — the default
> trace undercounts them, a known pitfall here) surfaced three concrete overheads:
> 1. **Decode attention ran the MONOLITHIC kernel inside graph decode at short context** (25% of GPU
>    time, 44.6 µs/call): `FlashAttentionDev`'s split gate required capacity ≥ 128, so short
>    prompt+generation runs never split — on few-head models that's 4-14 blocks on 28 SMs. Round 3's
>    "engage at ≥32 when baseBlocks ≤ SM count" rule was only in the EAGER dispatch; now mirrored in
>    the Dev formula. Measured: attention now `lm_flash_attn_f32_split` at 15.9 µs/call (−64%).
> 2. **`lm_argmax_lastdim_f32` took ~179 µs/token** — ONE block scanning the whole vocab row (600 KB
>    at C=152k through a single SM). New two-stage argmax (`lm_argmax_lastdim_stage1/2_f32`): G≤64
>    chunk blocks then a combine block, bit-identical first-max tie-break at every level, engaged at
>    rows==1 && C≥32k. Benefits EVERY model (vocabs are 128k-262k); argmax dropped out of the top-kernel
>    list entirely.
> 3. **~6 allocation/free graph nodes per Linear** from the dp4a scratch (xq/xd/xs per call — ~750
>    nodes on a 24-layer model's captured graph): replaced with a persistent, stream-serialized,
>    256-aligned combined buffer on the backend (`EnsureDp4aScratch`). NEVER grown during stream
>    capture (`cuStreamIsCapturing` check — a capture-time cuMemAllocAsync becomes a graph-OWNED
>    allocation that must not back a cached pointer); the pre-capture warm-up forward sizes it to max
>    first, so the transient fallback never fires on the graph path. The argmax partials scratch
>    (512 B) is allocated eagerly at construction for the same reason (the warm-up doesn't call
>    ArgMaxInto, so lazy allocation would bake the slow single-block path into the first graph).
>
> **Correctness**: two-stage argmax is bit-identical by construction (verified: float-path graph-vs-
> eager CLI output byte-identical on qwen2.5-0.5b, gemma3-1b, Llama-3.2-1B, GLM-4-9B after the change);
> full `HartsyInference.Cuda.Tests` 179/179 + `HartsyInference.LLM.Tests` 132/132.
>
> **Measured (graph-on medians, vs round 3)**: Llama-3.2-1B 197.3 → **202.74** (+2.8%, now 1.07×
> FASTER than llama-cpp-python's 190.3); DeepSeek-R1-1.5B 160.5 → **166.01** (+3.4%, gap 1.12→1.09×);
> gemma3-1b 115.1 → **118.48** (+3.0%); qwen2.5-0.5b 157.8 → **162.16** (+2.8%); gemma2-2b 129.5 →
> **131.59** (+1.6%, 1.09× faster than llama.cpp); Qwen3-4B 92.9 → **93.67**; GLM-4-9B 39.2 → 39.35.
>
> **Where the tiny-model gap stands after this round**: qwen2.5-0.5b per-token GPU-busy is ~2.5 ms of
> a 6.2 ms wall — the rest is graph-replay per-node overhead across ~340 kernel nodes/step (norms,
> ropes, permutes, slices, appends, quantizes at ~1-3 µs each plus replay scheduling). The next real
> lever is NODE-COUNT REDUCTION — the original Phase 4 fusion list (RMSNorm+RoPE fusion, eliminating
> the provably-no-op t=1 Permute0213s, quantize folded into the dp4a GEMV launch) — a distinct,
> larger campaign; everything cheap-and-targeted in this bucket is now done.

> **STATUS UPDATE (2026-07-23, round 5): long-K/small-N GEMV K-split — the block-per-row split
> RE-INTRODUCED with a much tighter, probe-derived gate; GLM-4-9B and the DeepSeek distill both
> improved, nothing regressed.** Probing every production GEMV shape of the two closest-to-parity
> laggards showed the recurring weakness is the ffn_down class (long K, few rows → ~1-3 waves of
> warps at warp-per-row): DeepSeek-1.5B's 8960×1536 Q4_K ran at **51% of DRAM peak**, GLM-4-9B's
> 13696×4096 Q5_0/Q8_0 downs at 71%/82%, while every wide shape and both lm_heads sit at 89-93%.
> The round-1 K-split rejection was measured at N≥2560 — a different regime.
> **What landed**: `_ksplit` entries (one BLOCK per row, warps split the row's blocks, deterministic
> shared-memory combine) for Q4_K/Q6_K/Q8_0/Q5_0, dispatched at **K ≥ 8192 && N ≤ 4096, W=4** —
> swept: DS down 42.2→38.3 µs, GLM Q5_0 down 168.8→134.3 (−21%), GLM Q8_0 down 205→191.9; the square
> 4096×4096 attention shape and the shorter-K downs (qwen2.5's 4864, gemma3's 6912) measured
> ambiguous-to-worse under a split and are excluded by the K floor (a first, looser gate measurably
> dipped qwen2.5-0.5b e2e; tightened and re-verified back to level). A Q5_0 qh-u16-assembly tweak was
> tried and REVERTED (no measurable gain). The probe harness was also fixed to toggle
> `CudaBackend.EnableDp4aGemv` directly — its env-var toggle had been dead since the default flip, so
> post-flip float-vs-dp4a probe RATIOS were bogus (both columns ran dp4a; absolute times, which drove
> all decisions, were valid).
> **Gates**: `Dp4aGemvGroundTruthTests` extended to 22 cases (4 new long-K/small-N rows that exercise
> the ksplit path against the analytic bound); full CUDA suite 183/183, CPU 132/132; float-path
> graph-vs-eager CLI output byte-identical on DeepSeek and GLM-4.
> **Measured (graph-on medians)**: DeepSeek-R1-1.5B 166.0 → **167.87** (gap to llama-cpp-python's
> 180.1 now 1.07×); GLM-4-9B 39.35 → **40.37** (gap to 47.8 now 1.18×); qwen2.5-0.5b 157.7 and
> gemma3-1b ~115-118 unchanged within the ±3% desktop-contention noise band; Llama 202.3 / Qwen3 93.2
> / gemma2 131.3 unchanged.
> **Residual analysis**: GLM's remaining ~4.5 ms/token gap is now spread thin — gate/up Q4_K at 89%,
> the square 4096² attention projections at 68% (resistant to splitting; would need a genuinely
> different layout), and ~3 ms of 40-layer graph-node/glue overhead. DeepSeek's remaining ~0.4 ms is
> the same tiny-model node-overhead bucket as gemma3/qwen2.5. Both point at the SAME next campaign:
> per-layer node-count reduction (norm+rope fusion, t=1 permute elimination via activation-cache
> aliased views — `Tensor.Reshape` exists but is host-pointer-based and would sync/miss the cache, so
> this needs explicit alias support in `GpuTransferHelper` with borrowed-buffer lifetime semantics —
> and quantize-in-GEMV was examined and REJECTED: per-block re-quantization multiplies by grid size).

> **STATUS UPDATE (2026-07-23, round 6): the node-fusion campaign — two structural fusions, every
> graph-eligible model improved +2-4%, all outputs verified identical.**
> 1. **Permute-free graph decode step.** At t=1 the [1,1,H,D] and [1,H,1,D] layouts are byte-identical,
>    so `Layer.ForwardGraphStep` now allocates q/k/v DIRECTLY in the head-major shape attention and
>    KV-append expect — the four per-layer `Permute0213` copies and their intermediates' alloc/free
>    graph nodes (~12 nodes/layer) no longer exist on the graph path. Every producer/consumer between
>    projection and attention was audited for layout-agnosticism at t=1: Linear/SliceLastDim size by
>    element count, per-head QK-norm reduces over the last dim either way, `KvCacheAppendDev` reads
>    tNew from Shape[2] (1 in both layouts), and the decode-RoPE launcher now derives the head count
>    from elementCount/headDim instead of Shape[2] (which also fixes a latent batched-caller bug where
>    only the first batch element was rotated). The EAGER path keeps its permutes — unchanged.
> 2. **Fused gated-FFN epilogue** (`lm_glu_act_f32`, `IBackend.GluActivate` with a composed
>    slice+act+mul default for CPU): the SwiGLU/GeGLU epilogue over the load-time-fused [gate|up]
>    projection output — previously 2× SliceLastDim + activation + Mul, four kernels and three
>    intermediates — is ONE elementwise pass with zero intermediates (~9 nodes/layer). Applies to
>    eager, graph, and batched paths alike (`DenseFfn`); SiLU and GELU-tanh supported, Relu-family
>    and unequal gate/up widths fall back to the unfused path.
>
> **Correctness**: 183/183 CUDA + 132/132 CPU suites; float-path graph-vs-eager CLI output
> byte-identical on all five architectures (qwen2, gemma3, llama, qwen3, glm4) — AND byte-identical
> to the pre-fusion build (the fused GLU's intrinsic formulas reproduce the elementwise kernels'
> fast-math exactly on all tested models; permute removal is a pure copy elimination).
>
> **Measured (graph-on medians, vs round 5 → after both fusions)**: DeepSeek-R1-1.5B 167.9 →
> **175.3** (+4.4%; fresh back-to-back llama-cpp-python 177.7 vs ours 172.2 in the same window —
> true gap now ~1.03×); Llama-3.2-1B 202.3 → **208.9** (1.10× FASTER than llama.cpp); Qwen3-4B
> 93.2 → **95.3** (1.11× FASTER); gemma2-2b 131.3 → **135.4** (1.12× FASTER); qwen2.5-0.5b 157.7 →
> **163.8**; gemma3-1b ~117 → **119.4**; GLM-4-9B ~40.1-40.5 (flat — its budget is GEMV-bound, not
> node-bound). Graph-off also gained fleet-wide from the GLU fusion (e.g. Llama 164→168.5,
> DeepSeek 136.6→143.0).
>
> **Still on the table for a future pass** (diminishing but real): cross-layer fused residual-add +
> RMSNorm (−2 nodes/layer, needs restructuring the layer-boundary handoff); QKV slice elimination
> (needs activation-cache aliased-view lifetime support — the one remaining permute-class item);
> GLM-4's square 4096² attention projections at 68% of DRAM peak (resistant to K-splitting — would
> need a different thread-ownership layout); and the sub-2B models' residual gap vs llama.cpp's
> extreme small-model efficiency (qwen2.5-0.5b 163.8 vs 322.7 — llama.cpp's per-step overhead at
> 0.5B scale is simply lower; closing further means attacking total per-step node count again).

> **STATUS UPDATE (2026-07-23, round 7): the deep-fusion pass — QKV rope-scatter and fused
> residual-add+RMSNorm, all bit-exact, every graph model up again; DeepSeek is now within 1.2% of
> llama-cpp-python.** On top of round 6's permute-free step and GLU fusion:
> 1. **`lm_qkv_rope_scatter_f32`**: ONE launch consumes the QKV projection output (fused [q|k|v]
>    buffer OR three separate tensors — the kernel takes three source pointers, so mixed-dtype-QKV
>    layers that can't load-fuse still benefit) and ropes q into the attention input, ropes k into
>    the KV cache at the device position, and copies v into the cache. Replaces 3× SliceLastDim +
>    2× rope + 2× KV-append on fused layers (and 2 ropes + 2 appends on unfused/QK-norm layers) —
>    BIT-EXACT: rope is elementwise (formulas copied verbatim from lm_rope_decode_splithalf/
>    _interleaved; each thread derives its own element from the same inputs and table entries), and
>    the scatter moves the identical bytes lm_kv_append_f32 would. Exposed as
>    `IBackend.QkvRopeScatterDecodeStep` / `RopeScatterKvDecodeStep` with composed defaults.
> 2. **`lm_add_rmsnorm_f32`** (`IBackend.AddRmsNorm`): residual add + RMSNorm in one pass, reduction
>    body copied VERBATIM from dit_rmsnorm_f32 (same strided partial, shared-memory tree, launch
>    geometry) so the result is bit-identical to the two-kernel sequence. Wired at the intra-layer
>    attn-residual → pre-MLP-norm site for plain pre-norm RMS layers.
> **Ops incident during measurement**: a stuck test-host from the parallel performance-grind agent's
> worktree (2.4 h old) was holding 6.9 GB of the 3060 — killed; the first post-fusion benchmark
> round was contaminated by it (numbers understated), so the finals below are from a clean GPU.
> **Correctness**: 183/183 CUDA + 132/132 CPU; graph-vs-eager CLI output byte-identical on ALL SEVEN
> models after every change in this round.
> **Final fleet (graph-on medians, clean GPU, 2026-07-23)**: Llama-3.2-1B **211.5** (1.11× FASTER
> than llama-cpp-python's 190.3); Qwen3-4B **96.3** (1.13× FASTER); gemma-2-2b **136.6** (1.13×
> FASTER); **DeepSeek-R1-1.5B 178.7 vs a same-window Python sandwich of 180.8/180.7 — gap 1.2%,
> effectively parity**; qwen2.5-0.5b **167.5**; gemma-3-1b **120.0**; GLM-4-9B **40.8** (solo,
> clean GPU). Note: in-process 7-model benchmark runs can drop/underfeed the LAST (largest) model
> via cumulative VRAM — measure GLM solo.
> **Remaining, honestly**: GLM (1.16×) is bounded by its square 4096² Q4_K projections at ~68% of
> DRAM peak (needs a different thread-ownership layout — the one big unexplored kernel idea) plus
> 40-layer bulk; the sub-2B pair (qwen2.5 ~1.9×, gemma3 ~1.6×) are at llama.cpp's extreme
> small-model efficiency frontier — kernels near-roofline, node count now cut three times; what's
> left is llama.cpp-class per-step scheduling economics (their whole step is a handful of larger
> fused kernels). Next candidates if this reopens: WARPS_PER_BLOCK sweep for square shapes,
> partial (q+k) load-time fusion for mixed-dtype QKV layers, cross-layer add+norm fusion.**

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

## Phase 6b — Retrofit CUDA-graph decode into DynamicBatchScheduler (✅ done, 2026-07-11)

**Why:** Phase 6 above shipped graph decode for `TextGenerationPipeline.Generate` only. An audit found the
actual HTTP server's chat path (`ModelManager.GenerateChatAsync` → `DynamicBatchScheduler`) has its own
separate eager decode loop using `PagedKvCache` and never used graph decode at all — every measured speedup
above benefited nobody hitting the real server. Full design/plan in
`~/.claude/plans/nested-zooming-tower.md`'s "NEW PLAN" section (6 phases: de-risking spike, pure refactor,
admission-time capture-once, per-round dispatch + one-way retirement, CUDA correctness tests, live
measurement).

**Design, in one line:** a request admitted while the scheduler is otherwise idle (`active.Count == 0`) and
graph-eligible gets a dedicated `FixedKvCache` (not pool-backed) and captures a graph ONCE at admission;
falls back permanently ("one-way retirement") to the existing eager `PagedKvCache` path the moment a second
request arrives while it's still running. `PagedKvCache` is fundamentally graph-incompatible (variable kernel
launch count per replay AND a reallocating scratch buffer — confirmed by reading `CudaGraph`'s own "frozen
topology/addresses" contract), so this is the only design that doesn't require a much harder paged+graph
cache hybrid.

**Real bug found and fixed along the way:** a genuinely cold model's first-ever graph capture (no prior
eager decode of any kind) reliably failed with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` — prefill's
multi-token GEMM shape and a single-token GEMV-shaped forward pass promote a lazily-cast/quantized weight to
GPU via different code paths, so the FIRST-EVER single-token access to a weight (both a per-layer projection
AND, separately, the LM head) can trigger a non-stream-ordered "auto-promote" allocation mid-capture, which
CUDA forbids. This is a **pre-existing bug in the original Phase 6 `GenerateGraphDecode`** (reproduced there
identically, no scheduler code involved) that no prior test had caught — every existing graph-decode test
happened to run an eager reference call first, incidentally warming up the exact weights capture needed.
Fixed in both `DynamicBatchScheduler.CaptureGraphSession` and `TextGenerationPipeline.GenerateGraphDecode`:
one throwaway single-token forward pass + `ProjectLogits` call (discarded cache/result) immediately before
capture, forcing the promotion to happen safely outside the capture region. One-time, first-request-only
cost. Locked in with a dedicated regression test
(`SchedulerGraphDecodeTests.SoloAdmission_SucceedsColdWithNoPriorEagerWarmup`) that admits cold with zero
warm-up — the scenario every prior test accidentally avoided.

**Correctness, verified real (CUDA, qwen2.5-0.5b-instruct-q4_k_m):**
- Solo scheduler request byte-identical to `TextGenerationPipeline.Generate` with graph decode on.
- Solo scheduler request byte-identical whether graph decode is on or off (the optimization never changes
  output).
- Transition test: sequence A admitted alone (captures a graph), sequence B admitted while A is still
  running (forces the heterogeneous eager path + one-way retirement for A) — both sequences' full output
  matched their own solo, graph-decode-OFF references exactly, proving the graph→eager splice mid-generation
  doesn't corrupt KV state (this is what actually exercises the `IKvCache.AdvanceLength` fix the retrofit
  needed — `GenerateGraphDecode` never called it, since it never needed to hand off to a different decode
  path mid-generation; the scheduler does).
- Cold-start capture (see bug above) now succeeds and produces correct output.
- 116/116 existing CPU-backend LLM tests pass unchanged (Phase 1's `ActiveSeq` cache-type widen +
  uniform-gated-disposal refactor is behavior-neutral) + the pre-existing `GraphDecodeRepetitionPenaltyTests`
  still passes with the warm-up fix in place.

**Speed, measured live through the real `/v1/chat/completions` server (not the CLI), qwen2.5-0.5b, solo
request, 500-token completion, greedy:** eager **111.0 tok/s** (4505ms) vs graph decode **146.6 tok/s**
(3410ms) — **~32% faster**, real server traffic, not a microbenchmark. Short generations (~120 tokens) showed
close to zero net win — the one-time per-sequence capture cost (~70-200ms, includes the warm-up fix above)
needs enough decode rounds to amortize over; this is expected given the design captures a fresh graph per
admitted sequence (no cross-request graph reuse) and matches this doc's existing framing of one-time capture
as "expensive, once" elsewhere in this file.

**Scope cuts (v1, deliberate):** no resume-after-eager-interlude (a sequence that's ever crowded loses the
graph-decode benefit permanently, even if later solo again — resuming would need resyncing device position/
token-id/repetition-penalty-history buffers that the eager path never touches, a materially harder problem
for marginal benefit at the target light/moderate-load use case); no promoting an already-`PagedKvCache`
sequence to graph-eligible later; speculative decoding is NOT retrofitted into the scheduler (harder problem
— variable per-sequence draft lengths reshape the batch every round — explicitly separate follow-up work).

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
- [🚫] **Row-interleaved (R4) weight repacking + 4-row GEMV** — investigated 2026-07-22: fresh `ncu` shows the decode GEMV is compute/memory co-limited (68.65% DRAM / 74.6% ALU), not the bandwidth-bound picture this bullet assumed; the input-vector-reuse half was tried directly (shared-memory staging, no repack) and measured an **11% regression** (`__syncthreads()` barrier cost > redundant-read savings). Not worth the full repack on this evidence — see the dated status block above.
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
