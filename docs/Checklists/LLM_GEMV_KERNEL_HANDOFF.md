# LLM Decode GEMV — Kernel Handoff (2026-07-22)

## For the kernel agent picking this up

This is a scoped, evidence-backed handoff for the remaining lever on LLM decode throughput. It is **not**
a restart — read `LLM_DECODE_PERF_GRIND.md` first (especially its dated 2026-07-22 status blocks) for
everything already tried, measured, and reverted this session. Re-doing that work will waste your time;
the point of this doc is to hand you the conclusion and the next concrete step, not make you re-derive it.

**One-sentence status**: our engine and llama.cpp are both memory-bandwidth-bound at decode (batch=1); we
sustain ~65-75% of the RTX 3060's peak VRAM bandwidth across every real production GEMV shape, llama.cpp
sustains somewhat more (~1.21x gap on Llama-3.2-1B and Qwen3-4B, RTX 3060); the gap lives entirely inside
the fused dequant-GEMV kernels, which `ncu` shows are **compute/memory co-limited** (74.6% ALU, 68.65%
DRAM on Q4_K), not purely bandwidth-bound — so the lever is cutting per-element ALU cost, not memory layout.

## What's already been tried (do not repeat)

| Idea | Result | Why |
|---|---|---|
| R4 row-interleaved weight repack (coalescing half) | Rejected, not built | Kernel is co-limited; a perfect coalescing fix only reclaims a fraction of ALU-capped headroom. See `LLM_DECODE_PERF_GRIND.md` R4 block. |
| R4 input-vector-reuse (shared-memory staging) | Built, measured, **reverted** | 11% regression — `__syncthreads()` barrier cost > redundant-read savings. |
| CUDA-graph allocation arena | Reverted | Profiler artifact, no real wall-clock change. |
| Flash-attention double-buffered pipelining | Reverted | Correct, bit-exact, but no measured speedup (not enough independent work to hide latency behind). |
| `CU_CTX_SCHED_SPIN` context flag | Reverted | Zero measured effect — the ~6ms/token isn't a host-scheduling tax, it's bandwidth-bound work (see below). |
| GPU clock locking (`nvidia-smi -lgc`) | Tested, no code change | Zero measured effect. |
| PCIe ASPM / link-speed theory | Refuted, not pursued | `lspci` showed "downgraded" only at idle; retrains to full 8GT/s under load. Was a measurement confound, not a real finding. |
| dp4a int8-activation GEMV (`HARTSY_DP4A_ON`) | **Re-measured positive**, still opt-in | +2% on Qwen3-4B (71.35→72.77 tok/s). This is your starting point — see below. |

**Two profiling pitfalls specific to this engine, hit twice each this session — expect to hit them again if
you're not careful:**
1. **`nsys --trace=cuda` undercounts kernels inside CUDA-graph replay.** A graph-decode-only trace showed
   13.7% GPU-busy and 260 GEMV instances where the real count is ~33,000 (eager-inclusive traces are
   trustworthy; graph-only traces are not, at least not with the default trace flags — if you need a clean
   graph-only kernel count, find the right `--cuda-graph-trace` mode first and verify instance counts against
   an eager-inclusive trace before trusting the numbers).
2. **Isolated-kernel `ncu` "Est. Speedup: X%" numbers overstate the achievable win on a co-limited kernel.**
   They're computed against that ONE metric's own ceiling (e.g. "if this load were perfectly coalesced"),
   not against the kernel's actual co-limiting bottleneck. Always price a fix against the SM/DRAM throughput
   balance first (`ncu --set full`'s GPU Speed Of Light section), and always confirm with an end-to-end
   tok/s measurement — not just the isolated kernel's `ncu` duration — before keeping a change. Every kernel
   change this session that "looked good" in isolated `ncu` timing but wasn't validated end-to-end either
   underperformed or regressed once measured for real.

## The target: cut ALU cost in the fused dequant-GEMV kernels

**Evidence** (RTX 3060, `ncu --set full`, real Qwen3-4B `ffn_gate.weight` shape K=2560 N=9728,
`mul_mat_vec_q4k_f32`):
- Memory Throughput 76.03%, DRAM Throughput 68.65%, **Compute (SM) Throughput 74.59%** — co-limited, not
  purely bandwidth-bound.
- SASS-level source correlation (`ncu --page source --print-source sass`) localizes the flagged
  "uncoalesced global access" finding to the activation-vector `float4` reads, which are structurally
  optimal per-warp (see R4 block in the grind doc) — the ALU side is where the real headroom is.
- Kernel does, per lane, per super-block: 8× nibble-extract-and-scale-and-subtract (`w0..w7` in
  `mul_mat_vec_q4k_f32.cu`), all scalar `float` ops. This is the per-element dequant cost the co-limiting
  ALU throughput is spent on.

**The dp4a path already builds an alternative**: quantize the F32 activation to int8 (Q8_1) once per call,
then do the GEMV as int8×int8 dot products via `__dp4a` (4 int8 MACs per instruction) instead of per-element
float dequant-then-multiply. It's already implemented (`native/cuda/lm/mul_mat_vec_q4k_q8_1.cu`,
`quantize_activation_q8_1_f32.cu`), already wired (`CudaBackend.cs` LinearImpl, gated on
`HARTSY_DP4A_ON` + `weight.DType == Q4_K`), and now measured positive (+2%, this session). It is NOT
currently: correctness-tested with a dedicated ground-truth test, swept across shapes/models, extended to
Q5_K/Q6_K/Q8_0, or the default.

## Concrete tasks, in priority order

1. **Build a ground-truth correctness test for the dp4a path.** Follow the exact pattern of
   `tests/HartsyInference.Cuda.Tests/FusedGemvGroundTruthTests.cs` (synthetic Q4_K weights, CPU dequant
   reference, tolerance check) but comparing `HARTSY_DP4A_ON=1`'s int8 path against the existing F32 path's
   output — int8 activation quantization is lossy, so this needs a tolerance gate (not bit-exact), sized
   from the Q8_1 quantization error, not guessed. This is the blocker for everything else below; do it first.

2. **Sweep the +2% result**: is it consistent across shapes (small K/N like QKV projections vs huge N like
   `lm_head`) and models (Qwen3-4B confirmed; check Gemma, Mistral, GPT-OSS — anything with Q4_K tensors)?
   Use the `gemv-probe`-style standalone harness pattern (no model loading, synthetic weights at real
   production shapes — see either probe built this session, both left in scratchpad and not committed,
   rebuild following the same pattern) to isolate shape effects from end-to-end noise, but **confirm every
   claim with a real end-to-end `TextDecodeThroughputBenchmark`-style run** — isolated timing alone is not
   sufficient per the pitfalls above.

3. **If the sweep holds up, flip `HARTSY_DP4A_ON` to default-on for Q4_K** (remove the opt-in gate, or
   invert it to opt-out) — full test suite gate (CPU + CUDA test projects), both real checkpoints byte-
   identical-token-id output check (same rigor as the QKV/gate-up fusion verification in the grind doc),
   graph-decode on AND off.

4. **Extend the same int8-activation/dp4a technique to Q5_K, Q6_K (the `lm_head` quant type on non-Q8_0
   models — worth checking specifically, since `lm_head` was 20% of total decode GEMV time in this
   session's Llama-1B profile), and Q8_0.** Q8_0 weights are already int8 — an int8×int8 dp4a GEMV for
   Q8_0 should need no activation-quantization-error tradeoff analysis at all (both operands already
   integer), making it the safest and potentially highest-value extension; consider doing it before Q5_K/Q6_K.

5. **If ALU is still co-limiting after (3)+(4)**, investigate cheaper float-path dequant as a fallback for
   quant types where int8 activation loses too much accuracy: `__byte_perm`-based nibble extraction, a
   small constant-memory LUT for the `(nibble * scale - min)` mapping instead of per-element scalar math,
   or reducing the number of live registers/instructions in the `w0..w7` unrolled block. Measure each
   independently — this is exactly the kind of "looks good in isolation" change that needs the end-to-end
   gate from the pitfalls section above.

6. **Occupancy/`WARPS_PER_BLOCK` tuning** (currently a fixed `8` for every quant-type GEMV,
   `CudaKernels.cs` `LaunchMulMatVecImpl`): worth a shape-aware sweep given `lm_head`'s N=128256 and the
   fused gate/up's N=16384 are an order of magnitude apart in grid width — but do this AFTER the ALU work
   above, since the co-limiting bottleneck analysis suggests occupancy isn't the primary constraint right
   now (measure to confirm before spending time here).

## Quality gates (non-negotiable, same standard as every other change this session)

- Correctness: dedicated ground-truth test with a justified tolerance (not bit-exact for int8 paths, not
  a guessed tolerance either — derive it from Q8_1's quantization error bound).
- Full test suite: CPU-backend `HartsyInference.LLM.Tests` + CUDA `HartsyInference.Cuda.Tests`, no new
  failures beyond the pre-existing baseline (currently 4 unrelated CUDA test failures — confirm the current
  count hasn't drifted before you start, so you can tell your changes apart from pre-existing flakiness).
- Real checkpoints: Llama-3.2-1B (Q8_0 — exercises the Q8_0 extension, not Q4_K) and Qwen3-4B (Q4_K_M),
  both graph-decode on and off, output compared against the current (pre-change) output — must match
  within whatever tolerance the ground-truth test establishes, not silently drift.
- End-to-end throughput: `TextDecodeThroughputBenchmark`-style measurement (median of ≥3 reps after
  warmup), not isolated kernel timing alone. Report both.
- Revert on no gain: if a change doesn't show a real, reproducible end-to-end improvement, revert it and
  document why — same discipline as the R4 and attention-pipelining reverts in the grind doc. A change that
  "should" help per the roofline model but doesn't measure is not a change to keep.

## Reporting

Append a dated status block to `docs/Checklists/LLM_DECODE_PERF_GRIND.md` in the same format as the
existing 2026-07-22 entries (what was tried, what was measured, kept-or-reverted, why). If this work is
being coordinated with the parallel `performance-grind` agent, also update the Results ledger in
`.claude/worktrees/performance-grind/docs/Checklists/INFERENCE_ACCEL_GRIND.md` — see that file's existing
R4 entries (H5 section) for the expected format.
