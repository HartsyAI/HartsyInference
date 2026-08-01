# Vulkan vs CUDA GPU-kernel scoreboard

Canonical, single-source-of-truth scoreboard for the Vulkan backend's raw GPU-kernel throughput
against the CUDA backend. This is the first dated benchmark artifact for Vulkan in this repo —
`docs/Checklists/ROADMAP.md` previously cited an unbacked "~6.5× CUDA" figure with no run behind it;
this table replaces that claim.

**Hardware:** RTX 4090, single card. Both backends bind device ordinal 0 by default
(`BenchmarkFixture`'s `deviceOrdinal: 0`); on this dual-GPU box (RTX 3060 + RTX 4090) **both** CUDA's
and Vulkan's own device enumeration independently put the 4090 at ordinal 0 (verified empirically —
this is NOT guaranteed in general; see `TROUBLESHOOTING.md`'s device-ordinal pitfall), so the two runs
below are apples-to-apples on the same physical card without needing `MESA_VK_DEVICE_SELECT` or
`CUDA_VISIBLE_DEVICES` overrides.

**Methodology:** `benchmarks/HartsyInference.GpuBenchmarks`, `GpuBenchmarkConfig` (1 warmup + 5 measured
iterations, `RunStrategy.Throughput`). Backend selected via `HARTSYINFERENCE_BENCH_BACKEND`
(unset/`cuda` = CUDA, `vulkan` = Vulkan) — both runs use the *same* benchmark classes and shape grids,
added in this pass (`BenchmarkFixture` previously hardcoded `CudaBackend`; see `docs/Checklists/
TROUBLESHOOTING.md`). `MemoryAllocFreeBenchmarks` and `QuantMatMulGpuBenchmarks` are CUDA-exclusive by
design (CUDA memory-pool API, GGUF k-quant dequant-GEMM — no Vulkan equivalent exists) and are excluded
from the Vulkan run rather than reported as failures.

## Results — GEMM (`MatMulGpuBenchmarks`), mean time (lower is better)

Shapes are real model hot paths (SDXL/Flux/SD3.5/Z-Image/Lumina2/Hunyuan) — see the class source for
the full (M,K,N) grid. Full per-shape numbers in `benchmarks/results/` CSVs from this run; representative
rows below.

| Shape (M,K,N) | Method | CUDA | Vulkan | Ratio (Vulkan÷CUDA) | Date |
|---|---|---:|---:|---:|---|
| (4096,1280,1280) SDXL UNet QKV | MatMul_F16 | 187.3 μs | 6,018 μs | **32×** | 2026-07-28 |
| (4096,1280,1280) SDXL UNet QKV | MatMul_F32 | 363.9 μs | 32,090 μs | **88×** | 2026-07-28 |
| (1024,3072,9216) Flux DiT QKV | MatMul_F16 | 463.7 μs | 72,923 μs | **157×** | 2026-07-28 |
| (1024,3072,12288) Flux DiT FFN | MatMul_F16 | 655.1 μs | 93,605 μs | **143×** | 2026-07-28 |
| (1024,1536,4608) SD3.5 joint-attn | MatMul_F16 | 219.2 μs | 6,680 μs | **30×** | 2026-07-28 |
| (1024,3072,9216) Hunyuan Image 2.1 | MatMul_F16 | 528.9 μs | 69,142 μs | **131×** | 2026-07-28 |

**Ratio range across all 10 shapes × {F32, F16, Linear+bias, FP8-cast} = 40 combinations: ~30×–160×.**
This is far worse than the previously-cited "~6.5×" — that figure has no backing artifact anywhere in
the repo and should be treated as superseded by this table. `MatMul_F32` (always the tiled fallback —
`TryDispatchCoopmat` requires `gemmDtype == F16`) is consistently the worst ratio, consistent with CUDA
opportunistically promoting F32 GEMMs to TF32 tensor-core throughput (`Compute32F()`) while Vulkan's F32
path has no tensor-core acceleration at all. The F16 path (which **should** hit `matmul_coopmat` — all
sampled shapes satisfy the M/N/K-multiple-of-16 gate) is *also* 30-157× slower, which is the more
concerning number: either coopmat isn't actually engaging for these shapes/dtypes at runtime (needs
verification with `HARTSYINFERENCE_VK_PROFILE=1`), or the hand-written coopmat1 kernel's real
throughput is far below cuBLAS's tuned tensor-core GEMM. **Root-cause priority for Phase 5.**

## Results — Norm/elementwise, mean time (lower is better)

| Op | Size | CUDA | Vulkan | Ratio | Date |
|---|---|---:|---:|---:|---|
| RmsNorm | [2,2,4096] (small) | 73.7 μs | 1,011 μs | 14× | 2026-07-28 (pre-fix) |
| RmsNorm | largest shape in grid | 100.1 μs | 20,924 μs | 209× | 2026-07-28 (pre-fix) |
| LayerNorm | [2,2,4096] (small) | 119.3 μs | 26,303 μs | 220× | 2026-07-28 (pre-fix) |
| Silu | [1,4096,1,1280] (5.24M elem) | 145.3 μs | 29,652 μs → **4,420 μs** | 204× → **30×** | 2026-07-28 (post-fix) |
| Silu | [1,320,128,128] (5.24M elem) | 128.6 μs | 26,846 μs → **3,733 μs** | 209× → **29×** | 2026-07-28 (post-fix) |
| Silu | [1,1280,32,32] (1.31M elem) | 28.2 μs | 956 μs → 1,328 μs | 34× → 47× | 2026-07-28 (post-fix) |
| Silu | [1,64,1,1] (64 elem, launch floor) | 33.7 μs | 160.8 μs → 72.3 μs | 5× → 2× | 2026-07-28 (post-fix) |
| BroadcastAdd | all sizes | 44–71 μs | 58–170 μs (unaffected — no fresh allocation) | 1.3–3.8× | 2026-07-28 |

**Root-caused and fixed** (see `docs/Checklists/TROUBLESHOOTING.md`): Silu/Gelu/RmsNorm/LayerNorm's
non-linear jump at the ~5.24M-element size (a ~31× time increase for only a 4× element-count increase,
which `BroadcastAdd` — no fresh output allocation — did not show) was `VulkanMemoryAllocator` destroying
any >= 16 MB ("dedicated") block the instant it emptied instead of pooling it like slab blocks, forcing a
real `vkAllocateMemory`/`vkFreeMemory` round trip on every dispatch that produced a >= 16 MB transient
output. Fixed by removing that special-casing — dedicated blocks now pool exactly like slabs. The two
5.24M-element rows above dropped ~7× (29,652→4,420 μs; 26,846→3,733 μs) and now scale roughly linearly
with element count as expected, converting a pathological cliff into a normal (if still large) kernel/
dispatch throughput gap that folds into the same coopmat/dispatch-overhead investigation as the GEMM
numbers above, rather than a separate anomaly. Regression-guarded by
`VulkanLeakTests.Vulkan_100Iter_LargeTransient_PoolsInsteadOfReallocating`.

## Results — synthetic LLM decode-step, GPU-residency closure (Vulkan-only, no CUDA baseline run)

`VulkanLinearProfileMeasurement.Measure_LlmDecodeStep_ResidencyVsDispatchOverhead` drives one synthetic
decode step (RmsNorm → QKV Linear → RoPE → KV-cache append → attention → out-proj → residual → RmsNorm →
gate-up Linear → SwiGLU → down-proj → residual) and reports wall-clock plus `GetD2hSyncCount()` /
transfer-cache hit-miss deltas around it.

| Stage | ms/step | D2H syncs/step | H2D misses/step | Date |
|---|---:|---:|---:|---|
| Baseline (audit) | 2.581 | 5.0 | 10.0 | 2026-07-28 |
| + `SliceLastDim`/`ApplyRope`/`KvCacheAppend` wired to real GPU dispatches | 1.461 | 1.0 | 5.0 | 2026-07-29 |
| + `CopyTo` device-to-device fast path (`TryGetCached`) | **1.382** | **0.0** | 4.0 | 2026-07-29 |

**~1.87× faster, D2H syncs eliminated entirely** for this step shape. None of `SliceLastDim`, `ApplyRope`,
`KvCacheAppend` had a `VulkanBackend` override before this pass — every call silently fell through to
`IBackend`'s CPU-loop default (a full device sync + host readback), and the next GPU op needing the
result paid a fresh H2D re-upload on top. The remaining 4.0 misses/step are genuinely-always-host
per-step inputs (the token embedding source, the RoPE cos/sin table) that need a device-resident RoPE
table (Phase 6) to close, not further residency work on this op set. No CUDA equivalent of this
synthetic step was run for a head-to-head comparison — this is a before/after on Vulkan alone.

## Results — fused flash attention (`sdpa_flash`), mean time, RTX 4090 (lower is better)

`SdpaGpuBenchmarks.Sdpa_F32`, same shape grid as the GEMM table above. `ScaledDotProductAttention` and
`FlashAttention` now dispatch the fused online-softmax kernel (`sdpa_flash.comp.glsl`) instead of the
old materialized 3-pass path for head dims <= 128 (all sampled shapes qualify).

| Shape | CUDA (cuDNN-fused) | Vulkan (`sdpa_flash`) | Ratio | Date |
|---|---:|---:|---:|---|
| (H=16,S=1024,D=80) SDXL self-attn | 323.9 μs | 18.9 ms | 58× | 2026-07-29 |
| (H=16,S=4096,D=80) SDXL self-attn 64×64 | 6,115.7 μs | 220.4 ms | 36× | 2026-07-29 |
| (H=24,D=64) SD3.5 joint-attn | 250.8 μs | 42.4 ms | 169× | 2026-07-29 |
| (H=24,D=128) Flux joint-attn | 375.7 μs | 68.4 ms | 182× | 2026-07-29 |
| **(H=24,S=16384,D=128) video-DiT scale** | 22.8 ms | **9.80 s** | 430× | 2026-07-29 |

**The headline result isn't the ratio — it's that the last row runs at all.** The old materialized path
needs a ~25 GB score matrix at that shape and cannot complete regardless of time budget (the documented
Wan-video full-resolution OOM); the fused kernel completes in 9.8 s using only the Q/K/V/O tensors'
own memory (~800 MB total), no intermediate score matrix at any size. The ratio vs CUDA's cuDNN-fused
path (a mature vendor-library kernel with full tensor-core tiling) is real and should NOT be read as "the
Vulkan flash kernel is broken" — this is a deliberately correctness-first design (Br=1: one query row per
workgroup, no register tiling, no coopmat) that trades throughput for a working first implementation; see
`docs/Checklists/ROADMAP.md` §3 for the Phase 5 tuning plan (larger query tiles, coopmat/tensor-core
fusion). Also new: causal masking, GQA, sliding window, and an additive mask are all supported and
numerically verified against a from-scratch CPU reference (`VulkanBackendSmokeTests`); softcap/sink/ALiBi
fall through to the CPU reference (Gemma-2/GPT-OSS/MPT-class models don't use Wan-scale attention, so
this doesn't block the OOM fix) — a documented scope boundary, not an oversight.

**A real, previously-shipping bug this work found and fixed**: the first kernel version indexed the K/V
buffer using the number of VALID kv positions as the per-head stride — correct only when the buffer is
exactly that size. Any real KV-cache buffer (over-allocated to a max sequence length, with only a prefix
valid) would have silently read the wrong memory. Caught by
`Backend_FlashAttention_GqaAndKvLenLessThanBuffer_MatchesCpu` before this ever shipped; fixed by passing
the buffer's actual capacity as a separate push-constant field from the loop-bound `skv`.

## Results — INT8 GEMM wired into `Linear`'s call surface (opt-in), correctness only, RTX 3060

`VulkanBackend.Linear` now has an opt-in INT8 dot-product GEMM path (`TryDispatchInt8Linear`, toggled via
the settable `EnableInt8Linear` property / `HARTSYINFERENCE_VK_INT8=1`). **This is op-level wiring, not
model-loading wiring** — no model's weight-loading path calls into it yet and there's no e2e SSIM/parity
gate; see `ROADMAP.md` §3's `[~]` entry for the distinction. This re-uses the already bit-exact-validated
`MatMulInt8` + `Int8Quantizer.RowwiseSymmetric` pair from the earlier INT8 GEMM effort — no new kernel,
just a new call site — so there's no separate throughput number to report yet (it dispatches the exact
same shader the GEMM table above already measures); what's new is the correctness gate at the `Linear`
call surface with real re-quantization on both operands, plus the CPU-side bias add. Tested on the RTX
3060; llvmpipe self-skips (no integer dot-product support — deviation #20 in `TROUBLESHOOTING.md`), not a
pass on that device. Measured on identical input run twice (opt-in off then on), so the numbers below
can't reflect a silent fallthrough to the exact path:

| Shape (M,K,N) | Exact path (opt-in off) relErr | INT8 opt-in relErr | Date |
|---|---:|---:|---|
| 64,128,96 | 0.0000% | 0.549% | 2026-07-29 |
| 96,256,128 | 0.0000% | 0.566% | 2026-07-29 |

Consistent with the standalone `MatMulInt8_QuantizedWeights_ApproximatesFloatMatmul` result above the
`Linear` layer didn't add — same per-row symmetric INT8 error budget carries through unchanged. Also
gated: a chained-into-a-second-GPU-op test confirms the bias-add's host-side write doesn't leave a stale
cached GPU buffer behind for a downstream consumer (`Backend_Linear_Int8OptIn_BiasSurvivesDownstreamGpuConsumption`)
— verified safe via the tensor lazy-sync callback's evict-on-read behavior, not just assumed.
**Not yet done** (explicitly deferred, not forgotten): this re-quantizes the weight from scratch on every
call; weights don't change between calls, so caching the weight's quantized form is the natural
throughput follow-up once it has its own freed-with-`FreeWeights` lifecycle. Per-shape INT8 tile selection
remains a separate open Phase 5 item.

## Results — LLM decode-graph device state (Phase 6a-d), correctness only, RTX 3060 + llvmpipe

The leaf ops a future graph replay would drive: device-resident RoPE table build + single-position apply
(`rope_decode_step`, interleaved + split-half), embed gather + on-device argmax (`embed_gather_decode`/
`argmax_lastdim`), repetition-penalty history append + apply (`history_append`/`repetition_penalty`), and
the device-position variants of KV-cache append and flash attention (`kv_cache_append_dev`,
`sdpa_flash_dev_f32`). All 14 tests in `VulkanDecodeGraphTests` pass on both the 3060 and llvmpipe — no
throughput numbers yet, since `GraphDecodeSupported` stays off (a settable property, not hardcoded) until
Phase 6f's real end-to-end decode-loop parity test validates it; see `ROADMAP.md` §3 for what's still open
and why `VulkanStepGraph` (the actual command-buffer-replay mechanism these ops would eventually run
under) was deliberately not built this pass.

Two real, would-have-shipped-silently bugs surfaced and fixed here (see `TROUBLESHOOTING.md`):
- A host-side write into a decode-graph control buffer could race ahead of an already-recorded-but-
  unsubmitted dispatch still needing the OLD value — passed on the 3060, failed on llvmpipe, the same
  cross-vendor-catches-what-one-GPU-hides pattern as every prior phase's llvmpipe sweep.
- `KvCacheAppendDev`/`FlashAttentionDev` initially forwarded the caller's placeholder host
  `offset`/`kvLen`/`qOffset` (real callers pass literal `0`s expecting the device buffer to be
  authoritative) instead of reading the real value from `devicePos` — would have silently corrupted the KV
  cache the instant `GraphDecodeSupported` flipped on. Caught in review before any test ran, not after.

## Results — Krea2 real-weight e2e (Phase 7), RTX 4090 — CORRECTNESS FIXED, speed not at parity

**Update 2026-07-30 — the earlier "OUTPUT INVALID" state below was root-caused and fixed.** `DispatchMatmul`
derived `M`/`N` from `output.Shape`'s rank structure instead of from the weight tensor — silently wrong for
any Linear whose output is shaped `[B, S, heads, headDim]` (Krea2's Q/K/V). Fixed by deriving `N` from the
weight operand, mirroring `CudaBackend.LinearImpl`. See `TROUBLESHOOTING.md`'s "Krea2 Vulkan e2e: F16
activation blowup — ROOT CAUSE FOUND AND FIXED" entry for the full writeup. Krea2 now produces a correct,
coherent image on Vulkan, verified via the CLI with the identical prompt/seed/steps/cfg as CUDA.

**Speed is measured, and is NOT comparable to CUDA** — this is a real, substantial, still-open gap, not a
residual correctness issue. `HARTSY_LOG_LEVEL=Verbose` breakdown, same Turbo/NoCfg/1024×1024/8-step config:

| Stage | CUDA (RTX 4090) | Vulkan (RTX 4090) | Ratio |
|---|---:|---:|---:|
| Text encode (preload+encode+free) | 1.05 s | ~2.4 s (2.0s preload + 0.34s encode) | ~2× |
| DiT preload | 2.04 s | ~2.2 s | ~1× |
| Denoise loop (8 steps) | 4.26 s total (~0.53 s/step steady-state; step 1 = 0.87s, warmup) | 275.9 s total (**34.4 s/step**, stable ±1.1s across all 8) | **~65×/step** |
| VAE decode | 0.017 s | ~39.0 s | ~2300× |
| **Total (cold single-shot CLI)** | **7.9 s** (`[krea2-phase]` internal total) / 11.3 s (wall, incl. process startup) | **321.5 s** | **~29–41×** |

Both breakdowns are real `HARTSY_LOG_LEVEL=Verbose` `[krea2-phase]` measurements from the identical CLI
invocation, not estimates. CUDA's VAE decode (17 ms) is essentially free — the QwenImage VAE decoder runs
through cuDNN's implicit-GEMM conv path; Vulkan's hand-written im2col+GEMM conv (see Phase 1's tiling fix
above) is the single largest per-stage ratio in this table, even larger than the per-step GEMM/attention
gap. The per-step timing's tight stability (34.4–35.6 s range across all 8 independent steps) was specifically
checked against a "stray D2H sync" or "un-batched dispatch submit" explanation before concluding this is
architectural: `HARTSYINFERENCE_VK_SUBMIT_PER_OP` defaults off (dispatches already batch, `FlushThreshold=8`),
and a fixable stall would show as an outlier step or a large host-idle gap concentrated in one place, not
uniform per-step cost. This is the same dispatch-overhead ceiling the GEMM table at the top of this file
already measures (30–160× per-op on hand-written GLSL kernels vs. cuBLAS/cuDNN) compounding across 28
blocks × 8 steps with no graph-capture/command-buffer-reuse to amortize per-dispatch host overhead — see
`docs/Checklists/ROADMAP.md`'s open item for the scoped path (Phase 5 core-primitive perf ceiling + Phase 7
denoise-loop graph capture). Not attempted this session beyond confirming it isn't a quick fix.

A `PARITY_VERIFICATION.md` row for Krea2/Vulkan is warranted now (coherent image, matches CUDA) — tracked
as follow-up, not yet added.

**Update 2026-07-30 — denoise-loop graph capture attempted, does not close the gap for Krea2.** The
`VulkanStepGraph` mechanism (Phase 6e/7, see `TROUBLESHOOTING.md`) was built and wired into Krea2 via
`HARTSY_DIT_GRAPH=1`. Four real CLI runs each hit a distinct capture-illegal op (fixed permanently:
`CfgEulerStep`, `Concat`, `AddScalar` — all three now have real GPU-resident overrides, not just
capture-time shims, so they also close real per-step D2H syncs in the EAGER path Krea2 actually runs) before
run 4 hit a structural OOM: capture retains every transient buffer touched during recording, but Krea2's
`CacheWeightCasts=false` strategy depends on freeing each block's transient fp8→F16 weight casts
immediately — retaining all ~224 of them across one capture exhausts VRAM even on the 4090. This is a
genuine conflict between two correct designs, not a bug; closing it needs an intra-capture bump-pointer
arena with slot reuse (no Vulkan equivalent to CUDA Graph's allocation-node semantics exists — see the
plan's Non-Goals). Every one of the 4 runs still completed with a correct, coherent image via the model's
existing eager fallback, proving that safety net robust including under a mid-capture OOM — no run has yet
produced an actual graph-captured-and-replayed step, so there is no capture-mode speedup number to report.

The default (no-`HARTSY_DIT_GRAPH`) CLI path was re-run after all three op-override fixes landed to confirm
they don't regress the path every user actually gets: **338.2 s total, ~36.5 s/step avg** (35.5–40.4 s
range) — within the same steady-state band as the 34.4 s/step figure measured before this work, confirming
no regression. (The per-run jitter between 34–40 s/step across different processes/times is consistent with
other background GPU load on this box, not attributable to these changes — each individual run's own
step-to-step variance stays tight, which is the signal that matters.) Output image verified pixel-composition-identical
to the pre-existing baseline. See `docs/Checklists/ROADMAP.md` for the single-block-capture experiment
recommended as the next step before building the weight-cast arena.

**Update 2026-07-31 — denoise loop confirmed kernel-bound (not dispatch-bound); VAE decode fixed (~65×);
coopmat engagement measured directly and found shape-blocked.** Three follow-on findings, RTX 4090 unless
noted:

1. **Kernel-bound, not dispatch-bound.** `HARTSYINFERENCE_VK_SUBMIT_PER_OP=1` (forces a submit per dispatch
   instead of the default `FlushThreshold=8` batching — an upper bound on what removing submit/dispatch
   overhead entirely could buy) added only ~1.3–1.5s to a ~35.5s step (~4%), and changed VAE decode by
   ~0% (33.12s vs 33.12s). This settles the open question from the Phase 7 entry above: the denoise loop's
   cost is GPU kernel time, not host dispatch/submit overhead — graph capture's ceiling here is small,
   correctly deprioritizing the weight-cast bump-arena needed to make capture fit Krea2's VRAM budget.

2. **VAE decode: 33.0s → 0.50s (~65×).** `WanRmsNormChannel` had no `VulkanBackend` override — every call
   fell through to `IBackend`'s CPU-loop default (full D2H sync + single-threaded host reduction + H2D
   re-upload) at the decoder's largest tensor shape (`[1,96,1024,1024]`, the final head norm). Fixed with a
   real GLSL kernel (`wan_rms_norm_channel.comp.glsl`) mirroring `CudaBackend`'s existing CUDA kernel for the
   same op. VAE decode now lands at **~30× vs CUDA's 17ms** — inside the normal hand-written-vs-cuDNN band
   from the rest of this scoreboard, not a separate pathology. Full generation: 331.1s → 296.2s (~10.6%
   faster). (VAE decode measured 0.50–0.65s across separate runs — run-to-run jitter consistent with this
   box's already-documented background-GPU-load pattern; either way it's a ~50–65× win, not a separate
   pathology.) See `docs/Checklists/TROUBLESHOOTING.md` for the full root-cause writeup.

3. **Coopmat engagement measured directly: ~1% (16/2112 GEMMs) on a real Krea2 run.** New engagement
   counters (`VulkanBackend`'s `_coopmatGemmCount`/`_tiledGemmCount`, printed alongside
   `HARTSYINFERENCE_VK_PROFILE=1`) answer the "needs a profile check" question the GEMM table above raised:
   coopmat is not silently broken, it is genuinely almost never reached for this model. A real dtype gap was
   found and fixed (F32-output Linears — every Linear in `Krea2Transformer` — could never reach
   `TryDispatchCoopmat`'s existing `OUTPUT_F32` support because `ResolveGemmDtype` never promoted F32 to F16
   to get there), but fixing it only moved engagement from 0/2112 to 16/2112: the real, binding constraint is
   shape, not dtype. `matmul_coopmat.comp.glsl` requires M/N/K all exact multiples of 16 (no buffer-padding
   support built), and every per-block QKV/FFN/out-proj Linear runs on the joint `[txtSeq+imgSeq]` sequence —
   measured directly on the reference prompt as `imgSeq=4096` (multiple of 16) + `txtSeq=13` (raw
   prompt-token count, unpadded) = `jointSeq=4109`, 3 short of the next multiple of 16. Since `txtSeq` is
   prompt-length-dependent, essentially no real prompt lands on a multiple of 16 by chance — this is a
   structural blocker on almost every expensive GEMM in the model, independent of the dtype question. **The
   dtype fix was reverted the same session** (kept only the zero-cost counters) since it carried a real,
   unverified precision cost (full F16 compute on FP8-dequantized-weight GEMMs) for a measured ~0 throughput
   win. See `docs/Checklists/TROUBLESHOOTING.md` and `docs/Checklists/ROADMAP.md` §3 for the next concrete
   step (M/N/K padding).

4. **M-padding attempted, also reverted, after a real GPU crash — no speedup number to report.** Built and
   extensively tested (from-scratch CPU-reference correctness, a 200-iteration no-sync leak/stress test at
   Krea2's exact real scale, two full CLI runs at 2 and 4 steps completing cleanly with coopmat engagement
   climbing to 60.7% then 74.9%). A full 8-step run then failed with `Vulkan error -4 (ErrorDeviceLost):
   vkQueueSubmit2` — a genuine GPU driver fault/reset, ruled out as VRAM pressure (an unrelated external
   ComfyUI backend process on this shared box, independently consuming ~12.7GB, was found and killed; the
   crash reproduced again on a verified-clean GPU with ~22GB free). Reverted before root-causing further —
   this needs Vulkan validation layers or `compute-sanitizer`-class tooling, not more full-model runs. See
   `docs/Checklists/TROUBLESHOOTING.md` for the two leading (unconfirmed) suspects and the recommended next
   design (ggml/llama.cpp's shared-memory-staged bounds-check, which avoids the scratch-buffer/barrier risk
   surface this attempt hit entirely).

5. **M-padding, third attempt (the ggml shared-memory design) — built, a real bug caught by testing and
   fixed, verified stable through a full 8-step run — and it definitively answers the question finding #3
   raised: engagement was never the bottleneck.** Built `matmul_coopmat_partial_m.comp.glsl`, a separate
   shader mirroring `matmul_tiled.comp.glsl`'s own bounds-checked shared-memory staging (no scratch buffer,
   no device-to-device copy, no cross-command-buffer barrier — everything inside one dispatch). A 9-case
   parameterized test caught a real, separate divergent-barrier bug before it ever reached the real model
   (N not being a multiple of the workgroup tile width BN — independent of N being a multiple of 16 — left
   some subgroups early-`return`ing past this kernel's `barrier()` while siblings continued to it; fixed by
   removing all early returns in favor of unconditional barrier participation + bounds-checked scalar
   draining). After the fix: 9/9 correctness cases pass, a pre-existing large-scale real-shape test (M=4108)
   now automatically exercises this kernel and passes unmodified, a 200-iteration no-sync stress test at
   Krea2's exact scale shows zero leak, and three consecutive real Krea2 CLI runs (2/4/8-step) all completed
   cleanly with GPU pinned at 100% utilization throughout (never idle/stuck) and correct output:

   | Steps | Total | Coopmat engagement |
   |---|---:|---:|
   | 2 | 80.5s | 60.7% (448/738) |
   | 4 | 154.9s | 74.9% (896/1196) |
   | 8 | **296.7s** | **84.8% (1792/2112)** |

   Full suite 143/143 (3060 + 4090); llvmpipe unaffected (no coopmat hardware — these tests self-skip
   there). **The result: coopmat engagement climbing from 0% to 84.8% bought ~0 wall-clock.** An A/B at
   matched shape (78.9s coopmat fully off vs. 80.5s at 60.7% engagement, 2-step) shows the delta is noise,
   not signal — consistent with the 296.7s 8-step total landing in the same steady-state band this table
   already measured pre-coopmat-fix. This confirms finding #3's open question: coopmat's own per-op
   throughput (not whether it engages) was already the real ceiling, matching the GEMM table's 30–157×
   F16 gap at the top of this file, measured well before engagement was ever fixed. **Kept in the codebase**
   (unlike the second, reverted attempt) — it's proven correct and safe, closes a real capability gap, and
   any future coopmat-kernel-throughput work now reaches these GEMMs automatically. See
   `docs/Checklists/TROUBLESHOOTING.md` for the full writeup and `docs/Checklists/ROADMAP.md` §2/§3 for
   next levers (tile size, register blocking, coopmat2 — not engagement).

## Register blocking, tested directly (2026-07-31) — a real, if modest, win from shared-memory staging;
## register blocking itself (ggml's core technique) is a net LOSS on this GPU

Built `matmul_coopmat_blocked.comp.glsl` — the ggml-inspired kernel described in the "Next levers" section
below — as a standalone, NOT-wired-into-production diagnostic (see `docs/Checklists/TROUBLESHOOTING.md`)
specifically to test whether register blocking (each subgroup computes a grid of 16×16 accumulators from
ONE shared-memory-staged tile, instead of matmul_coopmat.comp.glsl's one-accumulator-per-subgroup direct-
global-load design) closes a meaningful fraction of the GEMM table's 30–157× gap. Correctness verified
first (4/4 CPU-reference cases, including a non-16-aligned M boundary case). Then a controlled 3-way A/B
(same shapes as the GEMM table above, same Stopwatch-around-`Sync()` methodology for all three so the
comparison is internally valid) isolated two independent variables:

| Shape | Naive | **Staged only** (shared-mem tiling, occupancy UNCHANGED — WM=WN=16, 1 accumulator/subgroup, matching naive's 16 subgroups/workgroup) | **Register-blocked** (WM=WN=32, 4 accumulators/subgroup, only 4 subgroups/workgroup) |
|---|---:|---:|---:|
| (4096,1280,1280) SDXL UNet QKV | 1997.0 μs | 1982.9 μs (**1.01×**) | 2512.7 μs (0.79×) |
| (1024,3072,9216) Flux DiT QKV | 12978.0 μs | 9236.7 μs (**1.41×**) | 12391.4 μs (1.05×) |
| (1024,3072,12288) Flux DiT FFN | 15481.8 μs | 11550.2 μs (**1.34×**) | 15272.2 μs (1.01×) |
| (1024,1536,4608) SD3.5 joint-attn | 2342.9 μs | 2341.5 μs (**1.00×**) | 2765.3 μs (0.85×) |
| (1024,3072,9216) Hunyuan Image 2.1 | 9299.5 μs | 8970.4 μs (**1.04×**) | 10337.4 μs (0.90×) |

**Register blocking (ggml's headline technique) makes things WORSE on this RTX 4090** — 0.79×–1.05×, i.e.
neutral-to-slower than the already-naive kernel. Reducing subgroup count from 16 to 4 per workgroup (to
give each subgroup 4× the register-blocked work) costs more in lost occupancy/parallelism than it gains in
reduced global-memory traffic. **Plain shared-memory staging, with occupancy held constant, is a real but
modest win** (1.00×–1.41×, correlated with K-depth — the two K=3072 shapes benefit most, the K=1280/1536
shapes barely move) — consistent with staging's benefit being about amortizing GLOBAL load cost across
more reuse, which only matters when there's enough K-depth to reuse across.

**The honest conclusion: neither technique gets remotely close to closing a 30–157× gap.** Even the best
result here (1.41×) leaves the large majority of the measured CUDA gap unexplained. This means the
bottleneck is NOT primarily about memory access pattern / arithmetic intensity (the thing both of ggml's
techniques target) — it's something more fundamental: either raw tensor-core instruction throughput via
`VK_KHR_cooperative_matrix` on this specific NVIDIA driver, fixed per-dispatch/launch overhead that no
amount of in-kernel tiling can amortize, or a genuine gap between what `coopMatMulAdd` compiles to and
what CUDA's hand-tuned WMMA/MMA PTX achieves. **Caveat on the absolute numbers**: this benchmark's
methodology (batch N calls, one `Sync()` at the end) is NOT the same as the GEMM table above's presumed
per-call BenchmarkDotNet methodology, so the ABSOLUTE `naive` numbers here (1997–15482 μs) are NOT directly
comparable to that table's Vulkan column (6018–93605 μs at the same shapes) — batching amortizes host/sync
overhead the isolated-call methodology doesn't. The staged-vs-naive-vs-blocked RATIOS in the table above
are the reliable result (same methodology throughout); a rerun through the actual
`benchmarks/HartsyInference.GpuBenchmarks` harness would be needed for an apples-to-apples CUDA comparison
under matched methodology — not done this pass. **Recommended next step**: real GPU profiling (Nsight
Compute / `VK_LAYER_KHRONOS_validation` — neither available on this box, no passwordless sudo to install
them) to see where cycles actually go inside one `coopMatMulAdd`-based dispatch, rather than more blind
kernel-structure guesses; `VK_NV_cooperative_matrix2` (below) is the next concrete kernel-level lever if
profiling isn't available, since it changes the underlying instruction path rather than just the tiling
strategy around it.

## Real GPU profiling, second attempt with root access — the definitive answer (2026-07-31)

Got `sudo` on this box. First tried the "recommended next step" above directly:

- **Nsight Compute (`ncu`, already installed locally from prior CUDA work, v2026.2.1) is CUDA-only** —
  confirmed empirically, not assumed: its own `--help` output has zero mentions of "vulkan", "graphics",
  "opengl", or "d3d" anywhere, and pointing it at a `dotnet test` process running the Vulkan benchmark with
  `--target-processes all` produces `==WARNING== No kernels were profiled.` — it cannot see Vulkan compute
  dispatches at all, only CUDA driver/runtime API calls.
- **`vulkan-validationlayers` installs cleanly via apt** (now installed) but validation layers catch API
  *misuse*, not throughput — not useful for the "where do cycles go" question.
- **Nsight Graphics** (NVIDIA's actual Vulkan/D3D/OpenGL profiler, the tool that would show real SM
  occupancy / warp-state / tensor-core-utilization breakdowns) is **not apt-installable** — no package in
  Ubuntu's repos, requires a separate authenticated download from NVIDIA's developer portal not accessible
  from this environment.
- `apitrace` is apt-installable but only traces API call timing, which the next paragraph's approach already
  measures more precisely and directly.

**So the profiling was built directly instead** — the standard technique a Vulkan developer reaches for when
vendor tooling isn't cooperating: `VkQueryPool` GPU timestamps (`vkCmdWriteTimestamp2`), a core, always-
available Vulkan feature, not an optional extension. Added as a small, permanent, reusable capability
(`VulkanGpuTimer` + `VulkanBackend.MeasureGpuTimeMs`, diagnostic-only, not on any production path) that
brackets a batch of dispatches with GPU-side timestamps and reads back the elapsed GPU time directly —
completely removing the host-side submission/dispatch/wait overhead that a `Stopwatch`-around-`Sync()`
measurement (used everywhere else in this file so far) cannot separate from real kernel execution time.

Re-measured the same shapes, same three kernel variants, PLUS the `matmul_tiled` fallback for comparison,
all via pure GPU execution time (directly comparable in absolute terms to the CUDA column, unlike the
Stopwatch numbers above):

| Shape | CUDA | Naive coopmat | Staged (WM=WN=16) | Blocked (WM=WN=32) | Tiled fallback (coopmat disabled) |
|---|---:|---:|---:|---:|---:|
| (4096,1280,1280) SDXL UNet QKV | 187.3 μs | 3994.8 μs (**21.3×**) | 1978.8 μs (10.6×) | 2458.0 μs | 4259.5 μs (22.7×) |
| (1024,3072,9216) Flux DiT QKV | 463.7 μs | 10374.9 μs (**22.4×**) | 9565.3 μs (20.6×) | 10631.5 μs | — |
| (1024,3072,12288) Flux DiT FFN | 655.1 μs | 12516.3 μs (**19.1×**) | 11822.4 μs (18.0×) | 13436.6 μs | — |
| (1024,1536,4608) SD3.5 joint-attn | 219.2 μs | 2324.7 μs (**10.6×**) | 2348.9 μs (10.7×) | 2674.3 μs | — |
| (1024,3072,9216) Hunyuan Image 2.1 | 528.9 μs | 8688.1 μs (**16.4×**) | 8922.3 μs (16.9×) | 10153.1 μs | — |

**Two findings, both large:**

1. **The true kernel-level gap is 10–22×, not 30–157×.** The overwhelming majority of the previously-
   documented "30–157×" figure (top of this file) was measurement methodology, not kernel throughput: that
   number came from an isolated per-call benchmark methodology (presumed full sync-per-call), which pays a
   large fixed host/dispatch/sync cost on every single measured call. GPU-only timing removes that entirely.
   10–22× is still a real, substantial gap — but it's a normal hand-written-vs-vendor-library gap, not
   evidence of something structurally broken.

2. **Coopmat and the scalar `matmul_tiled` fallback have STATISTICALLY IDENTICAL real GPU throughput** —
   re-ran the SDXL shape with `HARTSYINFERENCE_VK_DISABLE_COOPMAT=1` (verified via the engagement counters:
   `coopmat=0 (0.0%), tiled-fallback=125 (100.0%)`) and got 4259.5 μs / 22.7× vs. coopmat-enabled's 3994.8 μs
   / 21.3× — the same result within run-to-run noise. **This is THE finding that explains everything else
   measured this session**: why raising Krea2's coopmat engagement from 0%→84.8% bought ~0 real speedup
   (`docs/Checklists/TROUBLESHOOTING.md`), why register blocking and shared-memory staging only moved the
   needle 0.8–1.4× (both techniques assume coopmat's tensor-core path has more room to exploit than the
   scalar path — if coopmat isn't meaningfully faster than scalar FMA to begin with, tuning ITS tiling
   strategy can't get much further than tuning the scalar kernel's). The likely explanation: `VK_KHR_
   cooperative_matrix` (coopmat1) on this NVIDIA driver either isn't compiling `coopMatMulAdd` down to real
   tensor-core (HMMA) instructions, or is doing so with far less scheduling/register-allocation
   sophistication than cuBLAS's decades-tuned WMMA/MMA PTX — a driver/compiler-level gap, not something
   fixable by restructuring the GLSL source further. Confirming which requires SASS/ISA-level inspection
   (Nsight Graphics or equivalent), which this box doesn't have access to.

**Recommended next step, in priority order** (as of this writing): (1) `VK_NV_cooperative_matrix2` — a
genuinely different instruction/memory path from coopmat1, not just different tiling of the same
instructions; if it shows a real win where coopmat1 didn't, that CONFIRMS coopmat1's tensor-core engagement
specifically is the problem. (2) Get Nsight Graphics installed (needs an NVIDIA developer account login this
environment can't complete) for direct SM/tensor-core utilization metrics — the only way to get a definitive
yes/no on "is coopMatMulAdd actually issuing HMMA instructions here." **(1) was done immediately after —
see the next section: it confirmed the hypothesis and delivered the first real speedup found all session.)**

## `VK_NV_cooperative_matrix2` — the first real speedup found this session (2026-07-31)

Built `matmul_coopmat2.comp.glsl` against `VK_NV_cooperative_matrix2` (confirmed supported, revision 1, on
both the RTX 4090 and RTX 3060 in this box via `vulkaninfo`) — architecturally distinct from coopmat1:
WORKGROUP scope (the whole workgroup cooperates on one big matrix multiply, not one independent 16×16 tile
per subgroup), addressed via `tensorLayoutNV` descriptors + `coopMatLoadTensorNV`/`coopMatStoreTensorNV`
directly against global memory (no manual shared-memory staging — the driver handles the cooperative load
internally), with built-in bounds CLAMPING (`gl_CooperativeMatrixClampModeConstantNV`) so M/N/K need not be
multiples of the tile size at all. Enumerating `vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV`
on the RTX 4090 surfaced a family of configs (32/64/128/256-invocation workgroups); the largest FP16-in/
FP32-out workgroup-scope one — 32×32 M/N tile granularity, K granularity 16, 256 invocations/workgroup — was
selected automatically (host picks the config with the largest `workgroupInvocations` among matching dtype/
scope entries; see `VulkanDevice.CoopMat2Supported`).

Correctness: verified against a from-scratch CPU reference across aligned shapes AND shapes where none of
M/K/N are multiples of the tile granularity (129×130×144) and shapes smaller than one tile in every dimension
(17×33×5) — all pass, with zero manual bounds-checking code in the shader (the clamp mode handles it). This
is a meaningfully more robust addressing model than every coopmat1 kernel in this codebase, which all need
hand-written partial-tile logic (`matmul_coopmat_partial_m.comp.glsl` exists solely because coopmat1 has no
equivalent to this).

K-block size (`BK`) was swept {16, 32, 64, 128, 256} on two representative shapes: BK=16 (the bare minimum —
the device's reported K-granularity) was measurably WORSE than larger values on the K=3072 shape (14983 μs
vs. ~9260–9925 μs for BK∈{32,64,128}) — each `coopMatLoadTensorNV` is itself an expensive workgroup-
cooperative op, so smaller/more-frequent K-steps pay that overhead more often. BK=64 was best-or-near-best
on both swept shapes and is now the default.

GPU-only execution time (same `VkQueryPool`-timestamp methodology as the section above, directly comparable
to the CUDA column) against the naive coopmat1 kernel and CUDA, same 5 shapes:

| Shape | CUDA | Naive coopmat1 | coopmat2 (BK=64) | coopmat2/naive | coopmat2/CUDA |
|---|---:|---:|---:|---:|---:|
| (4096,1280,1280) SDXL UNet QKV | 187.3 μs | 3881.3 μs (21.7×) | 1561.6 μs | **0.40×** | 8.3× |
| (1024,3072,9216) Flux DiT QKV | 463.7 μs | 11624.4 μs (25.1×) | 7233.5 μs | 0.62× | 15.6× |
| (1024,3072,12288) Flux DiT FFN | 655.1 μs | 12982.6 μs (19.8×) | 9256.1 μs | 0.71× | 14.1× |
| (1024,1536,4608) SD3.5 joint-attn | 219.2 μs | 2293.2 μs (10.5×) | 1877.6 μs | 0.82× | 8.6× |
| (1024,3072,9216) Hunyuan Image 2.1 | 528.9 μs | 8635.0 μs (16.3×) | 7109.3 μs | 0.82× | 13.4× |

**This is the first technique measured this session that beats the naive coopmat1/scalar baseline by a real,
substantial margin** (1.2–2.5× faster, best on the smallest-K shape) — confirming the hypothesis from the
section above: coopmat1's tensor-core engagement specifically was the bottleneck, not GEMM tiling strategy in
the abstract. It does NOT close the gap to CUDA — a large 8.3–15.6× residual remains, roughly half the
previous 10.5–25.1× naive gap. Diagnostic-only; not wired into `DispatchMatmul` — see
`VulkanBackend.TryDispatchCoopMat2Diagnostic` and `VulkanCoopmatBlockedBenchmark.Compare_Naive_Vs_CoopMat2_GpuOnlyTime`.

Not yet tried on coopmat2 specifically (carried over from the "Next levers" section below): ggml's PR #10942
diffusion-shaped tile-size tuning, `split_k`, and Nsight Graphics SASS-level inspection to see exactly what's
still eating the remaining 8–16×. **Superseded by the real e2e result below — read that before acting on
these isolated-benchmark numbers.**

## Wired into `Linear`/`DispatchMatmul` (opt-in) and run against real Krea2 weights — a real REGRESSION, not
## the win the isolated benchmark predicted (2026-07-31)

Added bias support (a follow-up `BroadcastAdd` dispatch — coopmat2's shader doesn't fuse it, matching ggml's
own `mul_mm_cm2.comp`, which doesn't either), wired `TryDispatchCoopMat2` into `DispatchMatmul` (tried before
`TryDispatchCoopmat`) behind a new opt-in `VulkanBackend.EnableCoopMat2` property
(`HARTSYINFERENCE_VK_COOPMAT2=1`, off by default — mirrors `EnableInt8Linear`'s pattern), and ran it against
real Krea2 fp8 weights on the RTX 4090 (`HARTSY_DIT_GRAPH=1`, 2-step, seed 42, same prompt) five times across
capture-on/off and cold/warm pipeline-cache states.

**Correctness: rock solid.** All 5 runs (baseline coopmat1-only, coopmat2 on ×2, coopmat2 with
`HARTSY_DIT_GRAPH=0`, coopmat2 with a warm on-disk pipeline cache) produced **byte-for-byte identical PNG
output** (same MD5). The bias epilogue, the transpose-combination gating, and the fallback wiring are all
correct.

**Performance: WORSE, not better.** Baseline (coopmat1 + tiled fallback, `coopmat=60.7%`): 58.6s total,
Linear-only 58.1s. With coopmat2 opted in (`coopmat2=60.7%`, exactly the same GEMMs coopmat1 was already
reaching — see below): 65.7–65.9s total, Linear-only ~61.0–61.2s, reproducible across 4 separate coopmat2
runs. **Two compounding causes found:**

1. **Wrong baseline in the isolated benchmark.** `matmul_coopmat_partial_m.comp.glsl` (a shared-memory-staged
   coopmat1 variant, built in an EARLIER pass this session specifically to handle non-16-aligned M) already
   handles most of Krea2's real GEMM shapes — coopmat1's real engagement here is 60.7%, not the 0.8% figure
   from *before* that kernel existed. The coopmat2-vs-coopmat1 isolated benchmark above only compared against
   `matmul_coopmat` (the plain aligned-M-only kernel) — never against `matmul_coopmat_partial_m`, which is
   what coopmat1 actually uses for most of these shapes. coopmat2 was never benchmarked against the kernel it
   would actually be replacing in practice.
2. **Un-fused bias costs a real extra dispatch.** Every bias'd Linear call now pays a full second dispatch
   (`BroadcastAdd`, with its own unconditional global `VkMemoryBarrier2` — see ROADMAP.md §2's "per-dispatch
   barrier scoping" entry) that coopmat1's fused `HAS_BIAS` epilogue avoids entirely. The isolated GPU-only-
   time benchmark always passed `bias: null`, so it never paid this cost.
3. **An unexplained ~4-second one-time stall per run**, always exactly once, landing on whatever op happens
   to be executing when it resolves (`ScaledDotProductAttention` with `HARTSY_DIT_GRAPH=1`, `RepeatKvHeads`
   with it off) — NOT tied to graph-capture mode (reproduces either way) and NOT a cold-pipeline-cache effect
   (reproduces with a warm on-disk `.pipeline_cache` too). Leading hypothesis: a genuine driver/hardware
   first-use latency specific to the `VK_NV_cooperative_matrix2` workgroup-scope instruction path, distinct
   from ordinary SPIR-V→ISA pipeline compilation (which the on-disk cache does cover) — plausible because the
   isolated benchmark's 5-iteration warmup loop would absorb exactly this kind of cost, hiding it completely.
   Unconfirmed; would need Nsight Graphics or a bisection harness to pin down definitively.

**Conclusion at this point: `EnableCoopMat2` stays off by default**, pending the follow-up below — the
wiring is correct (opt-in, zero effect on default behavior — confirmed via
`VulkanCoopMat2LinearTests.Backend_Linear_CoopMat2OptIn_DefaultsOff`) but the causes weren't fully understood
yet.

## Follow-up: benchmarked against the correct competitor, fused bias, and the "~4s stall" turned out to be a
## pre-existing, coopmat2-UNRELATED memory-pressure bug on this shared box (2026-07-31, same day)

Chased all three open items from the section above:

**(a) Correct competitor.** Re-ran the GPU-only-time comparison against `matmul_coopmat_partial_m`
specifically (M shifted +3 off alignment to force it, same K/N as the scoreboard shapes, going through the
real `Linear`/`DispatchMatmul` path so bias-epilogue cost is included). **coopmat2 still wins**: 0.74–0.98×
of coopmat1's time across 20 shape/alignment/bias combinations — the "wrong baseline" theory doesn't explain
the real-run regression after all.

**(b) Fused bias.** Bias is now added directly in the shader via a broadcast `tensorLayoutNV` (stride 0 on
the M dimension, so every row's `coopMatLoadTensorNV` reads the same `bias[n]` — no shared memory, no extra
dispatch), loaded straight into an Accumulator-typed coopmat at full F32 precision. Real wins: correctness
tolerance improved from 1.5% to 0.05% max relative error (bias no longer round-trips through F16 via a
second dispatch), and the isolated GPU-only-time benchmark's biased cases dropped sharply (e.g. the aligned
SDXL-QKV-with-bias case went from 0.87× to 0.31× of coopmat1's time). **But it barely moved the real Krea2
e2e number** (65.7s before vs. 65.7s after, Linear-only ~61.0ms both times) — meaning bias dispatch overhead,
while real, was NOT the dominant cause of the earlier regression. Most of Krea2's actual coopmat2-eligible
Linears apparently don't carry bias (or carry little enough that fusing it doesn't move the aggregate).

**(c) The "~4s stall" — solved, and it's not about coopmat2 at all.** A 4-step real Krea2 run
(`HARTSY_DIT_GRAPH=1`, coopmat2 ON) logged: `[WRN] [Krea2 graph] capture invalidated — falling back to
eager: HartsyInference.Vulkan.VulkanException: Vulkan error -2 (ErrorOutOfDeviceMemory):
vkAllocateMemory(size=134643712, typeIdx=1)`, inside `TryDispatchCoopMat2 → VulkanMemoryAllocator.
AllocateDedicated`, during a `Krea2Block.SwiGlu` Linear call on step 2 — a genuine ~128MB VRAM allocation
failure, caught by the existing (pre-existing, not something built this session) step-graph-capture
fallback-to-eager exception handling. **Ran the identical 4-step generation with coopmat2 OFF as a
control — it hit the EXACT SAME OOM, same allocation size, same step, same call site pattern
(`SwiGlu`'s Linear).** This is a real, pre-existing memory-pressure issue on this box — most likely the
GPU contention already noted earlier (SwarmUI + a ComfyUI Python process + rustdesk all resident on the
4090; `nvidia-smi --query-compute-apps` shows ~1.8GB held by other processes even at idle) — completely
unrelated to coopmat2. It doesn't reproduce at 2 steps (less accumulated VRAM pressure over the run) but
reliably does at 4. Output correctness held perfectly through the OOM+recovery in both configs: all three
4-step runs (baseline, coopmat2 run 1, coopmat2 run 2) produced **byte-identical PNG output**.

**Corrected picture once this shared, coopmat2-unrelated cost is factored out as common-cause noise**: at
2 steps (no OOM event in any run), coopmat2 with fused bias is still ~5% slower in aggregate Linear time
than the coopmat1 baseline (60,952ms vs. 58,085ms) — a real, if now much smaller, regression.

**4-step variance sweep (2026-07-31, same day): 6 more runs (4 coopmat2-ON, 2 coopmat2-OFF), added to the 2
existing ON runs and 1 existing OFF run for 6 ON / 3 OFF total.** Every single one of the 9 runs hit the
IDENTICAL `ErrorOutOfDeviceMemory` at the identical allocation and step, confirming it's a fully
deterministic, pre-existing property of this box at 4+ steps — not intermittent, and identical in both
configs, so it's valid common-mode noise to factor out. Aggregate Linear-only time:

| | n | mean | stdev | min | max |
|---|---:|---:|---:|---:|---:|
| coopmat2 ON | 6 | 121,486 ms | 6,271 ms | 113,653 ms | 131,534 ms |
| coopmat2 OFF (baseline) | 3 | 126,456 ms | 8,199 ms | 117,104 ms | 132,404 ms |

Mean difference: coopmat2 is 4,970ms (3.9%) faster on average. **Not statistically significant at this
sample size** (Welch's t ≈ 0.92, |t| well under the ~2 threshold for significance — the two groups'
distributions substantially overlap: the OFF group's minimum, 117,104ms, is below 4 of the 6 ON runs).
**Honest conclusion: at 4 steps, coopmat2 is roughly on par with baseline — plausibly a small win, but not
confidently distinguishable from run-to-run noise on this shared, contended GPU with only 9 total samples.**
Getting a statistically confident answer would need many more runs than is a reasonable use of shared GPU
time to chase further right now. Correctness held perfectly: all 9 4-step runs (6 ON, 3 OFF) produced
byte-identical PNG output.

**`EnableCoopMat2` stays off by default.** The 2-step regression (clean, no OOM confound, ~5% slower) is
the more reliable signal available and is still real. At 4 steps the picture is now "not worse, maybe
slightly better, not confidently proven either way" — genuine progress from "clear regression," but not
yet a case for enabling this by default. See `docs/Checklists/TROUBLESHOOTING.md` for the same writeup and
`VulkanCoopMat2LinearTests.cs` / `VulkanCoopmatBlockedBenchmark.cs` for the tests/benchmarks backing this
up.

## Next levers, informed by ggml/llama.cpp's mature Vulkan backend (research pass, 2026-07-31)

A research pass (reading `ggml-vulkan`'s `mul_mm.comp`/`mul_mm_cm2.comp` source directly, plus PR history)
surfaced concrete, higher-leverage directions than anything attempted so far in this file — not implemented
this session, listed here so the next pass doesn't have to re-derive them:

- **Diffusion-shaped GEMM tile-size tuning has prior art with real numbers — but note ggml's win was on
  `coopmat2` (`VK_NV_cooperative_matrix2`), not plain `coopmat1`.** ggml PR #10942 ("vulkan: im2col and
  matmul optimizations for stable diffusion") took RTX 4070 sd.cpp throughput from 3.68→4.65 it/s via a
  larger coopmat2 tile size alone, then 4.65→5.07 it/s combined with im2col kernel tuning (more elements per
  thread, workgroup 256→512). **This session tested the coopmat1 register-blocking/tiling idea directly (see
  the results table above this section) and found it does NOT transfer here**: register blocking was a net
  loss (0.79–1.05×) and plain shared-memory staging only a modest win (1.00–1.41×) on an RTX 4090 with
  `matmul_coopmat.comp.glsl`'s `VK_KHR_cooperative_matrix` (coopmat1) path. This doesn't contradict ggml's
  result — it's evidence the win specifically needs `coopmat2`'s different instruction/memory path, not that
  "tile size tuning" as a general idea is wrong. `TryDispatchCoopmat`'s current BM/BN selection (64×64,
  dropping to 32×32/16×16 only for tiny shapes) still hasn't been tuned on coopmat2 against Krea2/Flux/SDXL's
  actual shapes the way ggml tuned theirs — that's the version of this idea still worth trying.
- **`split_k`** (ggml PR #10637): splits the K dimension across workgroups to improve occupancy specifically
  for small-M / imbalanced matmuls. Less obviously relevant here than for LLM decode (M=1 GEMVs), since
  Krea2's M (~4096+) isn't tiny, but worth a real measurement before ruling out.
- **`VK_NV_cooperative_matrix2`** (NVIDIA-only, already flagged as deferred in ROADMAP.md §3): direct
  global-memory loads (skips shared-memory staging entirely), tensor layout descriptors, K-loop unrolling.
  ggml uses it for both `mul_mat` and FlashAttention2. Higher-effort, NVIDIA-only, but the highest-ceiling
  single item on this list given coopmat1's throughput is the now-confirmed bottleneck.
- **Cross-vendor coopmat reliability is not uniform** — relevant to this project's AMD/Intel roadmap even
  without hardware to test on yet: ggml maintains a device-specific coopmat allow/deny list (PR #11074
  disables coopmat entirely on the AMD proprietary driver; on Strix Halo/gfx1151, disabling coopmat improves
  AMDVLK performance by 17% while RADV handles it fine). Budget for a per-vendor/per-driver capability check
  before assuming `HasCooperativeMatrix` being true means it's actually fast.
- **Quantized (Q4/Q8) GEMM is not a universal win** — vendor-dependent. NVIDIA RTX 30/40-series: Q8_0 is
  <5% slower than Q4 (both hit INT8 tensor cores, close to free). Intel Arc B70 (Xe2): Q8_0 only reaches
  21–24% of theoretical bandwidth vs. Q4_K_M's 53–64% — quant kernel maturity varies a lot by backend/vendor,
  don't assume NVIDIA's numbers transfer.
- **Rough magnitude context** (LLM, not diffusion, weak signal — old GT 1030): ggml's Vulkan backend runs
  prompt-processing at roughly 2–3× slower than its CUDA backend, not orders of magnitude. This is a
  meaningfully SMALLER gap than this project's current 30–157×, suggesting a well-tuned Vulkan GEMM path can
  land much closer to CUDA than what's shipped here today — the ceiling is not architectural.

## What this does NOT show yet
  target, not a perf number to chase on the current naive path.
- AMD/Intel hardware: none available on this box. Mesa llvmpipe (software) was used for small-subgroup
  *correctness* checks elsewhere in this cycle, not for these perf numbers — llvmpipe throughput is not
  representative of real AMD/Intel silicon.

## Raw artifacts

Full BenchmarkDotNet output (Markdown/JSON/CSV per benchmark class) for this run is not committed to
the repo (BenchmarkDotNet's own `BenchmarkDotNet.Artifacts/` convention) — re-run via:
```
dotnet build benchmarks/HartsyInference.GpuBenchmarks -c Release
dotnet run --project benchmarks/HartsyInference.GpuBenchmarks -c Release --no-build -- --filter "*MatMulGpuBenchmarks*"
HARTSYINFERENCE_BENCH_BACKEND=vulkan dotnet run --project benchmarks/HartsyInference.GpuBenchmarks -c Release --no-build -- --filter "*MatMulGpuBenchmarks*"
```
