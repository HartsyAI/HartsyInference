# LLM Decode Performance Grind — closing the gap to llama.cpp

Status legend: ⬜ todo · 🔧 in progress · ✅ done · 📊 measured

**Goal.** Get HartsyInference LLM single-token decode from **20-54× slower than llama.cpp** to within a small factor (target: ≥50% of llama.cpp t/s, i.e. ≤2× slower) on the RTX 3060, same GGUF files and params as `LLM_THROUGHPUT_BENCHMARK.md`. Top priority.

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

**Phase 2c — the lm_head fix (nsys-guided):** nsys showed the tied lm_head was a **2.17 ms/token F32 cuBLAS GEMV** (28% of decode) because line 123 force-dequantized the tied embed to F32 (622 MB read/tok). Now keep the original **quantized** embed (`_lmHeadQuant`), preload *that* (not the F32 table → **~0.5 GB less VRAM**), and route the head through the fused GEMV (F32 hidden, no F16 cast). Also fixed untied lv1 heads (Mistral's Q6_K head skipped the F16-cast path). Results: **Qwen3 90.6→106 (+17%), Llama 83.7→100 (+20%, 2.15× off llama.cpp), Mistral 16.4→18.8**. `GenericTransformer.cs` load/EnumerateWeights/ProjectLogits.

### Gap so far: 20-54× → **2.15-3.5×**. Remaining (nsys): our Q4_K GEMV 36% (17µs/call, overhead-bound not bandwidth-bound → R4 repack/vectorize/dp4a), attention 16% (low occupancy), ~640 launches/token (CUDA graphs).
### ⚠ TODO before "done": formal token-parity gate vs llama.cpp (validated coherent only so far).

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

## Phase 1 — Fused dequant-GEMV kernel ⬜ (expected biggest win)
- [ ] Per quant type (Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, Q5_0): a single CUDA kernel that reads quantized blocks and accumulates the dot product with the input vector directly — no F16/F32 weight materialization, one pass over the weight. Model on llama.cpp `mul_mat_vec_q` (one warp per output row, int8/dp4a where applicable, F32 accumulate).
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
