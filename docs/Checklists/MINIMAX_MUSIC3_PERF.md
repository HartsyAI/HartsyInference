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

### 2b. F16 KV is 32% SLOWER than F32 KV — the biggest unexplained number here
Phase 2 set out to measure graph capture and found something larger by accident. On a 3060 at `:q4`, LM stage:
F16 KV eager 16.2 s vs F32 KV eager 12.3 s. Same kernels, same batching; the only change is cache dtype. A
half-width cache moving half the bytes should be *faster*. Whatever causes this is worth more than every other
item on this list combined (4.0 s vs the graph's 0.4 s), and it is not MiniMax-specific — `FixedKvCache` F16 is
shared, so any model using an F16 cache is likely paying it.

Suspect a per-access widening conversion, an F16 path that falls back to a slower attention kernel, or a
non-vectorized F16 load. Diagnose before patching. Fixing it plausibly gets the graph win *and* keeps the memory,
which is what would let phase 2 default on.

### 3. Autoregressive host glue — the ~2.6 s unaccounted
Two sources, neither ever removed: the per-frame frame-emit D2H, and `DecodeDepth`'s host-built sequence plus
per-step logits/state readbacks (~7 round trips per frame per branch). Keeping the depth sequence device-resident
also *enables* phase 5. Fully 3060-friendly.

### 4. Flow-stage CFG batching
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

### 7. BF16 cast caching in `LinearImpl` — separate track
`cacheWeightCast: true` caches a device-side dtype cast per weight, roughly doubling the 17.2 GB language model,
which is why the bare BF16 variant does not fit a 24 GB card while the reference's BF16 does. Fixing it unlocks the
dtype-matched benchmark that is still impossible on this hardware. Blast radius is every model in the engine, so it
needs a cross-model throughput A/B before any default changes. Late, or its own project.

### 8. Small wins
- Model load is ~10 s of wall time (parallel shard mmap, defer the head slice). UX, not generation time.
- One long-duration run per major phase to catch scaling regressions — the KV cache reaches ~2.6 GB at 9000 frames.

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
