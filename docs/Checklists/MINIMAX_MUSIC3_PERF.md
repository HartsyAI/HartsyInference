# MiniMax Music 3 — performance grind

Live plan. Referenced from `MODEL_STATUS_AUDIO.md`; numbers land in
`benchmarks/scoreboards/AUDIO.md`. Written 2026-08-14 to survive session boundaries — a future session should be
able to pick up any phase from this file alone.

## Where it stands

Already **1.87× the reference**: 26.4 s of generation for 15.0 s of audio on a 4090 at `:q8`, against diffusers
PR #14456 BF16 at 49.4 s (same card, prompt, seed, step count). The chain so far: 36.7 → 31.3 (depth KV cache) →
28.3 (CFG batching, byte-identical output) → 26.4 (depth Q8).

So this grind is "take the remaining measured headroom", not "catch up". Current split on the 4090 at `:q8`, 15 s
of audio: **LM 9.9 s · depth 3.8 s · flow 9.5 s · vocoder 0.4 s · sampling 0.3 s · ~2.6 s unaccounted loop glue.**

Speculative ceiling if every phase below lands: ~13–16 s, roughly 3× the reference. That is an estimate, not a
promise.

## Hardware protocol — read before running anything

The 3060 is the default working card. The 4090 is shared with another agent and needs a negotiated window.

Three consequences, all load-bearing:

1. **The Python reference needs ~22 GB and cannot run on the 3060, ever.** Every "vs Python" number is a 4090-window
   activity.
2. **The correctness gates are 4090-window activities too.** `MiniMaxMusic3ArParityTests` loads the BF16 language
   model (17.2 GB); the flow/step parity tests load the F32 transformer (9.6 GB); both hardcode `deviceOrdinal: 0`.
   On the 3060 you have the checkpoint-free unit and geometry tests only.
3. **`:q8` does not fit the 3060 — `:q4` is the 3060 iteration configuration.** Establish a fresh 3060/q4 baseline
   before phase work and A/B against *that*. Never compare a 3060 number to the 26.4 s chain above.

Timing hygiene, which has already caught contention twice: measure on an idle card, use flow-stage time as the
contention proxy (~9.5 s on an idle 4090), and take a same-tree A/B — revert your edits for the before-run rather
than quoting an earlier run.

## Phases, in ROI order

### 1. `ForwardGraphStepDual` shape contract — correctness first, 3060-OK
`Layer.ForwardGraphStepDual` allocates `attnSeg` token-major `[1,1,hq,d]` against `FlashAttentionDev`, which shares
the validator that required the `ForwardBatchDecode` fix in `a6fec0e5` — yet CSM's dual-graph path is believed
working. Reconcile that: either it silently falls back to a non-graph path, or the validator is not reached. Produce
evidence either way; if broken, apply the same byte-identical shape fix (at Tq=1 the layouts are identical, so
concatenating along dim 0 changes nothing downstream). **This gates all graph work below.**

### 2. CUDA-graph capture of the batch-2 language-model step — the big one
Measured: ~8.7 ms per forward is eager overhead (kernel launches across 36 layers), paid twice per frame no matter
how the branches are batched. That is why batching alone bottomed out at 9.9 s. Infrastructure already exists —
`ForwardGraphDecodeStepDualEmbeds` and the `HARTSY_CSM_GRAPH` precedent. Expect LM 9.9 → ~6–7 s.
**Landmine:** graph-decode output is only deterministic on an otherwise-idle GPU (learned on H3). Do not chase
phantom hash differences while the other agent is working.

### 2b. F16 KV lost split-K attention — FIXED, `c81c42a4`, on by default
Confirmed end to end on a 3060 at `:q4`, 15 s of audio, n=2 per arm, one build:

| stage | split-K off | split-K on |
|---|---|---|
| **language model** | 16.2, 16.3 s | **12.4, 12.4 s** |
| depth decoder | 9.6, 8.8 s | 9.3, 9.3 s |
| flow, 3 windows | 43.5, 43.5 s | 43.4, 43.9 s |
| vocoder | 1.0 s | 1.0 s |

−3.9 s, −24% on the language-model stage, with the halved KV cache kept. The controls carry the argument: the
only stage that moved is the one that reads the KV cache, and a 43.5 s flow stage sat still while it did. Each
arm is byte-identical across its runs (`89ad9c6f` off, `d5917952` on); the arms differ from each other because
split-K re-associates and the sampler forks on last bits.

Original diagnosis below, kept because the reasoning is the reusable part.

### 2b-old. F16 KV is 32% SLOWER than F32 KV — the biggest unexplained number here
Phase 2 set out to measure graph capture and found something larger by accident. On a 3060 at `:q4`, LM stage:
F16 KV eager 16.2 s vs F32 KV eager 12.3 s. Same kernels, same batching; the only change is cache dtype. A
half-width cache moving half the bytes should be *faster*. Whatever causes this is worth more than every other
item on this list combined (4.0 s vs the graph's 0.4 s), and it is not MiniMax-specific — `FixedKvCache` F16 is
shared, so any model using an F16 cache is likely paying it.

**DIAGNOSED. An F16 KV cache disqualifies itself from split-K attention** and silently falls back to the
monolithic kernel. `CudaBackend.FlashAttention` line 9132: `splitEligible = pSink == 0 && pAlibi == 0 && !f16Kv`.
The comment above it says "v1 scope" — this was a deliberate shortcut, not a bug, and nobody had measured what it
cost. It costs 18–25% of decode.

The F16 kernel is not slower. At the real decode shape, F16 and F32-monolithic are within 0.05% at every kvLen —
so the *whole* gap is the lost fast path, and disabling split-K for F32 reproduces the F16 number exactly
(15.8 s, matching F16's 15.8 s, against F32-with-split's 12.4 s).

**Not MiniMax-specific.** Same signature on Qwen3-4B text decode with no source edits: 45.3 s F32 → 53.4 s under
`HARTSY_KV_F16=1`, and 53.6 s for F32 with split forced off. Split-K engages whenever `b*hq*tq < 2*SM`, i.e.
`hq < 56` on a 3060 — effectively every decode step of every model. Prefill is unaffected.

Three F16 construction sites exist: `MiniMaxMusic3GlobalLm.cs` (the only one on by default), plus
`KvCaches.ForDecode` and `TextGenerationPipeline`, both behind `HARTSY_KV_F16`. `PagedKvCache` is F32-only.

**Fix, ~90 lines, NOT applied** — it lands in `CudaBackend.cs` / `CudaKernels.cs` / `Kernels/lm/`, which another
session is actively committing to. Add an `lm_flash_attn_f16kv_f32_split` entry point to
`flash_attn_f32_split.cu` (`__half` K/V plus `__half2float` on the two loads — character-for-character the diff
that already exists between `lm_flash_attn_f32` and `lm_flash_attn_f16kv_f32`), thread a `f16Kv` flag through
`LaunchFlashAttentionSplit`, and drop `!f16Kv` from the eligibility test. The combine kernel is untouched, and
`KvF16StorageTests.FlashAttention_F16Kv_SplitForceEnv_StillMatchesMonolithicF32Kv` — which passes vacuously
today — becomes the real gate.

Expected: MiniMax `:q4` LM 15.8 → ~12.4 s *while keeping the halved cache*. It also unlocks F16 for the
graph path (`FlashAttentionDev` uses the same split/combine pair), which is what would let phase 2 default on
without the four-minute ceiling. Bench to A/B against: `tests/HartsyInference.Cuda.Tests/KvF16DecodeAttentionBench.cs`.

### 3. Autoregressive host glue — the ~2.6 s unaccounted
Two sources, neither ever removed: the per-frame frame-emit D2H, and `DecodeDepth`'s host-built sequence plus
per-step logits/state readbacks (~7 round trips per frame per branch). Keeping the depth sequence device-resident
also *enables* phase 5. Fully 3060-friendly.

### 4. Flow-stage CFG batching — DONE, `fc163303`, opt-in
−1.63 s of a 43.5 s flow stage (−3.7%) on a 3060 at `:q4`, n=4 per arm with disjoint ranges. The ~20% estimate
below was wrong: the DiT moves ~4.84 GB of weights per forward, so halving weight traffic caps the win near
1.2 s. Nothing further for batching to take. `HARTSY_MM3_FLOW_CFG_BATCH=1` enables; it stays off until the flow
parity gate runs against it.

**Every absolute number in this file is stale.** The 26.4 s chain and the 48.4 s flow figure came from different
trees; a same-tree measurement puts the flow stage at 43.5 s. Trust same-tree A/B deltas, re-measure absolutes.

### 4-old. Flow-stage CFG batching — original estimate, kept for the record
The DiT runs conditional and unconditional as two separate forwards per step. At L=689 it is compute-bound, so the
win is launch count and weight amortization rather than 2× — expect 9.5 → ~7.5–8 s. Note the asymmetry with the
semantic head: there is **no sampling in the flow stage**, so last-bit GEMM drift cannot fork the output. It stays
inside the 5e-3 flow-parity tolerance. Same-seed WAV bytes may shift once — document it, do not treat it as a bug.

### 5. Depth-decoder graph capture
After 1 and 2 prove the infrastructure. 3.8 → ~2 s, hopeful.

### 6. Profiling gate — before ANY kernel writing
`libnvToolsExt.so.1` is missing, so every `NvtxRange.Push` in `CudaBackend` is a no-op and an nsys timeline would be
unlabelled kernels. Install NVTX first (may need sudo — ask).

**Then read this sentence before writing a kernel:** the engine's Q8 GEMV is already at-or-ahead of llama.cpp across
nine text models, and every measurement in this grind so far points at launch overhead and host glue, not kernel
quality. Write or rewrite a kernel only where a profile shows a specific kernel below its bandwidth or FLOP bound.

### 6b. VRAM audit — why long songs OOM. PLAN, not yet executed
Written 2026-08-16 after ten full-length songs peaked at **23,940 MiB of 24,564** on a 4090 (97.5%), with the
disco and synthwave tracks stopping at exactly 240.3 s because they hit the cap rather than finishing. A 150 s
batch had already failed all ten on the 3060, OOMing in the AR stage at frame 1900 of 3750.

**Do not start from the assumption that the cast cache is the cause.** Two facts from the logs cut against it:

1. `EvictAllWeightCasts` released **6336 MB / 6528 MB** on some attempts but only **80 MB** on the one that
   actually died — so at the moment of failure the cache was already small and something else held the card.
2. The q4 batch peaked **higher** than the q8 probe (23,940 vs 20,183 MiB) despite roughly 4 GB lighter
   weights. Backwards for a weights-dominated story; consistent with a frames-dominated one, since the q4 songs
   ran 2–3× the frames.

Arithmetic on those two points: q8 ≈ 8.8 GB over ~2218 frames ≈ 4 MB/frame; q4 ≈ 16 GB over ~5–6k frames ≈
3 MB/frame. The F16 KV cache accounts for **0.29 MB/frame**. So several MB/frame is unexplained, which over
36 layers is ~80–110 KB/layer/frame — transient-sized, i.e. leak-shaped. This engine has a documented history
of exactly that (an undisposed `Tensor.Reshape` view OOM'd a 24 GB card during the Music3 port).

#### Phase 0 — the discriminator. Run this BEFORE designing a fix.
One 4090 run and one 3060 run, instrumented:
- Per-frame VRAM curve: `cuMemGetInfo` sampled in the AR `onFrame` callback every ~50 frames.
- Cast-cache census at frame ~200 and at peak: entry count, total bytes, source→cast dtype, weight identity.
- Pool stats via `cuMemPoolGetAttribute` (reserved vs used high-water) — separates live allocations from pool
  retention.
- Read `LinearImpl` (CudaBackend.cs ~1769–1810 and ~2079–2247) and settle whether the batch-2 CFG input routes
  to dequant-to-F16 + GEMM (which would cache multi-GB dequants) or to a row-wise quantized GEMV. Note the CFG
  batch was byte-identical to two separate forwards, which hints GEMV and would kill the dequant story.

Decision table: **linear slope in MB/frame → hunt the leak** (audit every per-frame allocation in the AR and
depth loops for a missing Dispose, `Reshape` views included). **Flat-but-high with a large census → budget the
cache and add an m=2 quantized kernel.** **Reserved ≫ used → pool retention policy.**

#### Fix levers, ROI-ordered, conditional on Phase 0
1. **Budgeted LRU cast cache** — worth doing whichever story wins. `GpuTransferHelper.WeightCastCache` (line 48)
   is an unbounded dictionary with no budget and no LRU; the only eviction is all-or-nothing and fires *after* an
   OOM. Give it a byte budget, LRU ordering, and a headroom-aware insert that declines to cache when free VRAM is
   below a reserve (`cuMemGetInfo` costs 5.2 µs and inserts are first-touch only). Add a partial-LRU rung to the
   `CudaMemory` ladder before evict-all. Kill-switch env; `=0` reproduces today exactly.
2. **An m=2 quantized kernel for the CFG pair**, if Phase 0 confirms dequant+GEMM — reads q4 once instead of
   materializing F16, removing the cache's reason to exist for quantized models. Carries the known last-bit fork
   risk (different song at the same seed), so: kill-switch, documented, CUDA parity gate.
3. **KV grow-on-demand instead of prealloc.** `CreateCache(maxSeqLen)` allocates the full cap up front, so a
   240 s cap costs ~1.8 GB even when the song ends at 90 s. `PagedKvCache` exists but is F32-only; an F16 paged
   variant is the clean shape. Matters most on the 3060.
4. Re-derive the per-card frame ceilings, update the AudioLab provider VRAM strings, and re-run the two tracks
   that hit the cap.

#### On the two ideas the user raised
- **Unified/managed memory ("cuda map")**: a spike only if 1–3 leave a gap. Expect it to lose — weights and KV
  are read every frame, so there are no cold pages to evict and managed memory would thrash the decode hot path.
- **CPU overflow so we never OOM**: the large host-spillable things are already host-side (`frame_hiddens`) or
  rebuildable (casts — never spill them, just rebuild from the resident quantized source). KV cannot spill, since
  the full prefix is read every frame — but six minutes is ~9000 frames ≈ 2.6 GB at F16, which fits even a 12 GB
  card. So "never OOM" is reachable by **policy** — budget, evict-before-fail, grow-on-demand — not by building a
  host-overflow subsystem.

#### Guardrails
`CacheWeightCasts` is global, so any semantic change needs the cross-model throughput A/B this file already
demands, plus the LLM suite, plus attention to the auto-promote regression class (the weight cache is read before
the activation cache — cache-lifetime changes are how that bug comes back). CUDA parity gates before commit,
never CPU-only. Same-seed WAV bytes may shift once per lever; document each rather than chasing it.

### 7. BF16 cast caching in `LinearImpl` — separate track
`cacheWeightCast: true` caches a device-side dtype cast per weight, roughly doubling the 17.2 GB language model,
which is why the bare BF16 variant does not fit a 24 GB card while the reference's BF16 does. Fixing it unlocks the
dtype-matched benchmark that is still impossible on this hardware. Blast radius is every model in the engine, so it
needs a cross-model throughput A/B before any default changes. Late, or its own project.

### 8. Small wins
- Model load is ~10 s of wall time (parallel shard mmap, defer the head slice). UX, not generation time.
- One long-duration run per major phase to catch scaling regressions — the KV cache reaches ~2.6 GB at 9000 frames.

## Mempool retention — what it is, and why it is probably NOT worth a project

Recorded 2026-08-17 because KV grow-on-demand tripped over it and the next person will wonder the same thing.

**The engine deliberately runs its device mempool at a keep-everything release threshold** (`HARTSY_MEMPOOL_KEEP`,
default on, applied by `DeviceMempoolPolicy`). A zero threshold hands every freed activation back to the driver
and re-acquires it on the next allocation; that was measured at **~13 s of pure alloc/free round-trips on a
single Krea2 1024² image**. So retention is a real optimisation, not an oversight.

Two consequences worth internalising:

1. **"Used VRAM" is the pool's high-water mark, not live data.** A freed buffer stays in the pool. This is not a
   leak — the OOM ladder (`SyncStreamsAndReleasePool`) and `cuMemPoolTrimTo` both reclaim on demand — but it means
   nvidia-smi overstates what is actually in use.
2. **It punishes allocation patterns with many distinct sizes.** Uniform sizes are reused perfectly; a
   grow-and-copy pattern makes 36 layers × N chunk crossings worth of odd sizes and the pool keeps them all. That
   is exactly why growth measured worse (see `5f231b71`).

### Grow-on-demand was built, measured, and REMOVED
Growing the cache in chunks instead of preallocating cost ~4 GB of peak VRAM and ~20% of the decode stage, in the
very case it was meant to win (short song, large cap). Output was byte-identical, so it was correct — just a
pessimisation, because the pool retains every intermediate buffer size. The code is deleted rather than parked:
it would never be the right answer to a VRAM ceiling, since it makes VRAM worse.

The right shape, if a bounded footprint is ever needed, is `PagedKvCache` — fixed-size pages from a shared pool,
gathered into a contiguous scratch per call so `FlashAttention`'s contract is untouched. Uniform sizes are exactly
what a keep-everything pool rewards. It would need an F16 variant and pays a per-call gather.

**But measure before building it.** With the decode leak fixed (`19e68209`) and F16 preserved on the graph path
(`af2900fd`), six minutes of song is ~9000 frames at 288 KB/frame across the guided pair = **~2.6 GB of KV**
against ~5.7 GB of `:q4` weights. That plausibly fits a 12 GB card outright, in which case preallocation is
simply correct and paged F16 buys nothing. Generate the longest song each card can manage and write the numbers
here first.

## OPEN BUG — dual-stream device attention is wrong on high-SM cards

`GraphDecodeDualEmbedsTests` **passes on the 3060 and fails on the 4090**, diverging 1.05e-3 against a 1e-6 bar
on the very first eager dual step — before any graph capture, so this is `FlashAttentionDev` arithmetic, not
capture. Reproduced with `af2900fd`'s changes reverted, so it predates them; the test was written against the
3060 and the card difference hid it.

Prime suspect is split-K selection, which keys off SM count: eligibility is `baseBlocks < 2*SM` and the split
count is clamped against SM count, so a 128-SM card picks many more splits than a 28-SM one for the same tiny
cache, leaving far more empty chunks for the combine to merge. Note the split kernel's own header says split-K
is gated to `b==1` because "the split/combine kernels have a latent batch>1 bug".

**This blocks `HARTSY_MM3_LM_GRAPH`**, which routes through that path — its parity gate passes, but a path that
is wrong on the bigger card must not become a default. Fix the divergence first, then flip.

Reproduce: `CUDA_VISIBLE_DEVICES=0 dotnet test tests/HartsyInference.Cuda.Tests --filter GraphDecodeDualEmbeds`
(fails) versus `CUDA_VISIBLE_DEVICES=1` (passes). **Run CUDA tests on both cards from now on** — a suite that is
green on one GPU says nothing about the other, which is how this survived.

## Parked behind a 4090 window

Two shipped-but-disabled features. Both are switched off only because the gate that would clear them cannot run
on a 12 GB card, not because anything is known wrong with them. When a window opens, run
`MiniMaxMusic3ArParityTests`, `MiniMaxMusic3FlowParityTests` and `MiniMaxMusic3FlowStepParityTests`, then flip
whichever pass:

- `HARTSY_MM3_FLOW_CFG_BATCH` — flow guidance batching, −3.7% (`fc163303`).
- `HARTSY_MM3_LM_GRAPH` — graph-captured guided decode, −4.6 s but forces F32 KV (`7bbe4b24`). Do **not** flip
  this one on the strength of parity alone — it also needs the four-minute cache ceiling resolved, which is
  what phase 2b is really about.

Also un-run for want of the card: the Python reference baseline, and a HeartMuLa CFG smoke test to confirm
`98b3f54c` un-broke that model on CUDA.

## Explicitly out of scope — do not redo these

- **Semantic-head batching.** Measured at +0.2 s and backed out: cuBLAS picks a different algorithm at two rows, and
  the last-bit logit difference makes the sampler produce a *different song at the same seed*. Not worth 0.7%.
- **Vocoder and sampling micro-optimization.** 0.4 s and 0.3 s respectively; there is nothing there.
- **Speculative kernel rewrites** ahead of phase 6 evidence.
- **More CFG batching in the LM.** Bottomed out — the remaining cost is launch overhead, which is phase 2.

## Process guardrails — these have each caught a real bug

- A kill-switch env var per change (`HARTSY_MM3_CFG_BATCH`, `HARTSY_MM3_DEPTH_QUANT`, …).
- The CUDA parity gate before any commit. Never the CPU gate alone: two silent bugs were invisible on `CpuBackend`.
- Never `Reshape` a device-resident tensor and then mutate it, or use a reshaped view as an op's output —
  `Reshape` returns a HOST pointer and the device copy goes stale.
- Delegate implementation with the full trap list, and require the agent to report refuted premises. Both agents so
  far corrected a wrong premise of mine with a measurement; that is the most valuable thing they produced.
