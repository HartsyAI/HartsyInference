# Vulkan Performance Plan (dispatch-overhead first, measure every step)

> **Scope**: the `HartsyInference.Vulkan` compute backend's inference hot path — Linear/GEMM dispatch
> overhead, weight-cast cost, kernel fusion, and the cross-backend gap vs CUDA. Sibling to
> [`QUANT_GEMM_PERF_PLAN.md`](QUANT_GEMM_PERF_PLAN.md) (CUDA) and the broader
> [`PHASE_3_5_VULKAN_BACKEND.md`](PHASE_3_5_VULKAN_BACKEND.md) bring-up checklist. Reuses the
> established Phase-C profiling methodology (`HARTSYINFERENCE_VK_PROFILE=1`).
>
> **Execute at will**: each stage is self-contained — run it, A/B, record in the ledger, accept or revert
> on the gate. The ledger is the durable record.

---

## Why this exists (the diagnosis is already known and data-backed)

Phase C ([`PHASE_3_5_VULKAN_BACKEND.md` §8](PHASE_3_5_VULKAN_BACKEND.md)) established the bottleneck on an
RTX 3060 (Flux Schnell FP8, 512², 4 steps): Vulkan was **129.5 s vs CUDA's ~20 s — ~6.5× off**, and

- **Linear is ~93.6% of profiled host time.**
- **~94% of per-Linear time is per-dispatch overhead** (descriptor binding, command-buffer recording,
  barriers) — *not* GEMM compute. Bigger tiles / cooperative-matrix only shrink the ~6% compute slice.

**Refreshed on current code, 2026-06-30 (this box has BOTH a 3060 and a 4090; Vulkan default-selects the
4090 — `ScoreDevice` ranks by VRAM):**

| Probe | Vulkan (4090) | CUDA (4090) | Gap |
|---|---|---|---|
| Flux FFN FP8 Linear, M=1024 K=3072 N=12288 | **~3.9 ms/call, 3 dispatches** | 0.59 ms (cached F16) / 1.79 ms (re-dequant FP8) | **~2.2–6.6×** |

(`VulkanLinearProfileMeasurement` + the CUDA `QuantMatMulGpuBenchmarks` probe.) Current code is down to
**3 dispatches/Linear** (the Phase-C notes said ~5): **FP8→F16 weight cast + coopmat matmul + bias add**.
The gap and the overhead-bound diagnosis hold on the faster GPU — in fact the *fraction* of time that is
per-dispatch overhead is **larger** on the 4090 (its compute is faster, so the fixed CPU/launch cost
dominates more). The two open structural levers, confirmed in the current source:

1. **The weight cast is NOT cached** — every Linear re-runs the FP8→F16 dequant (1 of the 3 dispatches),
   which is also a full weight-sized memory pass (~75 MB write). Mirrors CUDA's `CacheWeightCasts`.
2. **Bias is a separate dispatch** — `matmul_coopmat.comp.glsl` has no `HAS_BIAS` path; bias runs as a
   follow-up `BroadcastAdd`.
3. **Q/K/V are three separate Linears** — no fused QKV projection.

The command stream already **batches dispatches into one submit per `Sync`** (Phase C1), so the overhead
is genuinely per-*dispatch* (descriptor update + barrier + `vkCmdDispatch` record), not per-submit. The
lever is therefore **fewer dispatches per Linear** and **cheaper per-dispatch binding**, in that order.

---

## Guardrails (every stage)

1. **Correctness gate before speed.** A change ships only if the Vulkan correctness tests stay green on
   the target GPU: `VulkanKernelTests` (vs CPU), `VulkanVsCudaTests` (vs CUDA within 1e-3, dual-GPU box),
   and the SD1.5/Flux end-to-end SSIM gate where it exists. New fused kernels need a vs-unfused diff test.
2. **Measure on vs off, same process, same shapes.** Every claim is an A/B with `HARTSYINFERENCE_VK_PROFILE=1`
   (per-op time **and dispatch count**). Accept a win only when both wall-clock and dispatch count move the
   predicted way and the correctness gate is green.
3. **Record everything in the ledger.** Wall-clock, per-Linear ms, dispatches/Linear, the gate result, GPU,
   driver/Vulkan version, git SHA. A win not saved did not happen.
4. **One change at a time.** Never fuse two things in the same measurement.
5. **Revert cleanly.** Failed gate or regression → leave the path off/unfused, write the negative result,
   move on. (Push descriptors are the cautionary tale: a *regression* on NVIDIA, kept opt-in — see
   [`PHASE_3_5_DEVIATIONS.md` #12](PHASE_3_5_DEVIATIONS.md).)

---

## Measurement protocol

- **Per-op profile**: `HARTSYINFERENCE_VK_PROFILE=1 [HARTSYINFERENCE_VK_PROFILE_FILE=...]` → top-N op table
  with **dispatch counts** (the key metric — this plan is about cutting dispatches).
- **Microbench (current code, no model)**: [`VulkanLinearProfileMeasurement`](../../tests/HartsyInference.Vulkan.Tests/VulkanLinearProfileMeasurement.cs)
  drives Flux-shape FP8 Linears; extend with the shapes each stage targets.
- **End-to-end**: `samples/BasicImageGeneration --backend vulkan` (or the CLI `--backend vulkan`) on a Flux
  Schnell FP8 / SD1.5 checkpoint, the Phase-C reference workload. Needs the model checkpoint.
- **CUDA reference**: the same op via `CudaBackend` on the *same* physical GPU — the gap target.
- **Devices**: 4090 (Vulkan default-selected here) **and** 3060 (`new VulkanBackend(deviceOrdinal: 1)` or
  whichever ordinal — Vulkan order ≠ CUDA order; verify by device name). The existing Phase-C data is
  3060-only; the 4090 is the new and harder target.

---

## Stage 0 — Lock the baseline (do first)

- [ ] Confirm Vulkan device enumeration; record which ordinal is the 4090 vs 3060 (Vulkan order differs
      from CUDA's PCI order — print `VkPhysicalDeviceProperties.deviceName`).
- [ ] Run the Flux Schnell FP8 e2e (`--backend vulkan`, `HARTSYINFERENCE_VK_PROFILE=1`) on **both** GPUs;
      record wall-clock, Linear total, dispatches/Linear. Run the CUDA backend on the same workload/GPU for
      the gap.
- [ ] Run `VulkanLinearProfileMeasurement` on both GPUs for the per-Linear microbench baseline.
- [ ] Commit numbers to the ledger. This is the immutable reference for every later "Nx".

Acceptance: baseline captured on 3060 + 4090, Vulkan and CUDA, e2e + microbench.

---

## Stage 1 — Fuse FP8 dequant into the coopmat matmul (highest value)

Hypothesis: read the FP8 weight directly in `matmul_coopmat.comp.glsl` and dequant per-tile in registers,
removing the separate FP8→F16 weight-cast dispatch **and** the weight-sized memory pass **and** keeping
the weight FP8-resident in VRAM. One change attacks dispatch count, memory traffic, *and* VRAM together —
strictly better than caching the F16 cast (Stage 1b) where it's feasible. This is the Vulkan analogue of
CUDA's deferred "fused dequant-GEMM."

- [ ] Add an FP8 (and later GGUF-quant) load path to the coopmat GEMM shader (spec-const `WEIGHT_FP8`),
      dequant in the tile load. Keep a non-fused fallback for unsupported dtypes.
- [ ] Gate: new diff test (fused FP8-matmul vs cast-then-coopmat) within FP8 noise (~1e-2 rel); existing
      `VulkanVsCuda` Linear within 1e-3 stays green.
- [ ] A/B microbench: Flux FFN FP8 Linear, off vs on — expect **3 → 2 dispatches** and a per-call drop ≈
      the weight-cast memory pass (~1 ms at this shape).
- [ ] A/B e2e Flux Schnell; gate SSIM; log wall-clock + VRAM (weights stay FP8 → ~½ the F16 footprint).
- [ ] Commit, fill ledger.

### Stage 1b — ✅ DONE (2026-06-30): cache the F16 weight cast (mirror CUDA `CacheWeightCasts`)
Landed first because the in-shader FP8 dequant (Stage 1) needs `glslangValidator` to recompile the shader,
which is not installed on this box (sudo-gated). This is the pure-C# lever.

- [x] `VulkanGpuTransferHelper` weight-cast cache (`_weightCastCache`, keyed by weight Tensor ref → target
      dtype), wired into `VulkanBackend.CastIfNeeded` via `TryGetWeightCast`/`ShouldCacheCast`/`StoreWeightCast`.
      Only **preloaded weights** are cached (activations change every step). Freed in `FreeWeights`/`FreeAllCached`.
- [x] Opt-out for low VRAM: `HARTSYINFERENCE_VK_NO_WEIGHT_CAST_CACHE=1` (default = cache on). Costs ~2× weight
      VRAM (FP8 original + F16 cast) — same trade-off as CUDA's `CacheWeightCasts`.
- [x] Gate: new `Backend_Linear_FP8Weight_CachedCast_Matches_Cpu` (preloaded FP8 weight, two consecutive
      Linears — miss then hit — both vs CPU dequant reference) + full smoke/leak suite (31 tests) green.
- [x] A/B microbench (4090): **3.65 → 2.82 ms/call (−22.7%), 3 → 2 dispatches**. See ledger.

> The remaining 2 dispatches are coopmat-matmul + bias `BroadcastAdd`. Stage 2 (bias fusion into the coopmat
> shader) collapses that to 1 — but needs `glslangValidator` to recompile `matmul_coopmat.comp.glsl`.

---

## Stage 2 — Fuse bias into the coopmat matmul

Hypothesis: a `HAS_BIAS` spec-const path in `matmul_coopmat.comp.glsl` removes the follow-up `BroadcastAdd`
dispatch (~1 of the 3 per Linear, ~1,000 dispatches/generation per the Phase-C estimate).

- [ ] Add `HAS_BIAS` to the coopmat shader; bind the bias buffer; add in the epilogue.
- [ ] Gate: vs-unfused diff (bit-close); `VulkanVsCuda` green.
- [ ] A/B microbench (with Stage 1 on): **2 → 1 dispatch** per Linear.
- [ ] A/B e2e; gate SSIM. Commit, fill ledger.

> After Stages 1 + 2: a FP8 Linear goes **3 dispatches → 1** (fused dequant-matmul-bias). That is the
> headline target — it removes the two non-compute dispatches that the overhead-bound diagnosis blames.

---

## Stage 3 — QKV projection fusion (3 Linears → 1)

Hypothesis (Phase-C carryover): concat the Q/K/V weight matrices at load so the three sequential Linears
become one matmul `[batch, seq, 3·hidden]`, then tensor-view-slice into Q/K/V. Cuts ~2/3 of the Q/K/V
Linear dispatches (Q/K/V are ~21% of Linears → ~14% of total). Needs a tensor view/slice API (only
`Reshape` exists today) + per-block weight-load changes in the Flux/Flux2/SDXL attention blocks.

- [ ] Tensor view/slice API (no-copy sub-view) on `Tensor`.
- [ ] Concat QKV weights at load in the affected blocks.
- [ ] Gate: attention output unchanged (vs-unfused diff + SSIM).
- [ ] A/B e2e: dispatch count + wall-clock. Commit, fill ledger.

---

## Stage 4 — Per-dispatch binding overhead

The remaining lever once dispatch count is minimized: make each surviving dispatch cheaper.

- [ ] Profile the per-dispatch cost breakdown (descriptor update vs barrier vs record) with NVTX/Vulkan
      timestamps.
- [ ] Evaluate descriptor-update batching / a descriptor cache keyed by (pipeline, buffer set).
- [ ] Coalesce compute-to-compute barriers where dependencies allow (avoid a full barrier between
      independent dispatches).
- [ ] Re-evaluate push descriptors **per vendor** — a measured regression on NVIDIA (deviation #12),
      possibly a win on AMD/Intel. Keep `HARTSYINFERENCE_VK_PUSH_DESCRIPTORS` opt-in.
- [ ] A/B each independently; commit, fill ledger.

---

## Stage 5 — Cross-vendor (AMD / Intel) — not testable on this box

This box is NVIDIA-only (3060 + 4090). The cross-vendor proof (AMD RDNA2/3, Intel Arc variable subgroup
size) from the Phase-3.5 checklist needs other hardware. Note tuned tile tables / subgroup-size paths here;
do not attempt blind.

---

## Lower priority — attention

Phase-C profile: **SDPA is only ~4.4% of host time** on Flux Schnell, so a flash-SDPA shader
(`sdpa_flash.comp.glsl`, carryover) is **low priority** versus the Linear-dispatch levers above. Revisit
only if a long-context or LLM-on-Vulkan workload shifts the profile.

---

## Results ledger

`Δ%` = (new − baseline)/baseline; negative time / fewer dispatches = better. Keep negative results.

| Date (UTC) | GPU | Stage | Config | Shape / Model | Metric | Baseline | New | Δ% | Gate | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-30 | 4090 | 0 baseline (refresh) | current code | Flux FFN FP8 Linear | ms/call, dispatches | — | 3.9 ms, **3 disp** | (ref) | n/a | vs CUDA 0.59 ms (cached F16) / 1.79 ms (FP8): **2.2–6.6× off** |
| 2026-06-30 | 4090 | (probe) coopmat vs tiled | `HARTSYINFERENCE_VK_DISABLE_COOPMAT=1`, cache on | Flux FFN FP8 Linear, M=1024 K=3072 N=12288 | ms/call, dispatches | 2.82 ms, 2 disp (coopmat+bias) | **7.61 ms, 1 disp (tiled, fused bias)** | **+170% (worse)** | n/a | **Negative result**: dropping coopmat to save the bias dispatch is 2.7× SLOWER — at this FFN shape tensor-core compute dominates, NOT dispatch count. ⇒ keep coopmat; fuse bias *into* the coopmat shader (Stage 2), don't fall back to tiled. |
| 2026-06-30 | 4090 | 1b weight-cast cache | `HARTSYINFERENCE_VK_NO_WEIGHT_CAST_CACHE` off (default ON) | Flux FFN FP8 Linear, M=1024 K=3072 N=12288 | ms/call, dispatches | 3.65 ms, 3 disp | **2.82 ms, 2 disp** | **−22.7%** | **pass** | Cache the FP8→F16 weight dequant per preloaded weight (removes 1 dispatch + ~75 MB temp alloc+write per call). Gate: `Backend_Linear_FP8Weight_CachedCast_Matches_Cpu` + full smoke/leak (31) green. Microbench re-uploads input each call (synthetic); real inference (cached activations) should gain more. Costs ~2× weight VRAM (opt-out flag). |
| (hist) | 3060 | 0 baseline | Phase C2.3 | Flux Schnell e2e | wall-clock | 178 s | 129.5 s | −27% | — | **6.5× off CUDA ~20 s**; 94% per-dispatch overhead |
| | 3060 | 0 baseline | current code | Flux Schnell e2e | wall-clock | | | | | (re-measure — Stage 0) |
| | 4090 | 0 baseline | current code | Flux Schnell e2e | wall-clock | | | | | (NEW data point — Stage 0) |
| | | 1 fp8-fuse | WEIGHT_FP8 on | Flux FFN Linear | dispatches | 3 | | | pass/fail | + VRAM (weights stay FP8) |
| 2026-06-30 | 4090 | 2 bias-fuse | `HAS_BIAS` spec-const in matmul_coopmat | Flux FFN FP8 Linear, M=1024 K=3072 N=12288 | ms/call, dispatches | 2.82 ms, 2 disp | **2.45 ms, 1 disp** | **−13.1%** (vs 1b) | **pass** | Fuse per-column bias into the coopmat epilogue via a stride-0 broadcast accumulator load (keeps tensor cores). Bias cast to FP32 once + cached. Gate: `Backend_Linear_FP8Weight_CachedCast_Matches_Cpu` (F16 out) + `Backend_Linear_FluxShape_F32_Matches_Cpu` (F32 out) + full smoke/leak/int8 (39) green. **Cumulative 1b+2: 3.65 → 2.45 ms (−32.9%), 3 → 1 dispatch.** |
| | | 3 qkv-fuse | — | Flux attn | dispatches/wall | | | | pass/fail | ~14% of Linears |
| | | 4 dispatch | — | per-dispatch | µs | | | | | descriptor/barrier |

---

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-06-30 | Plan created; dispatch-count-first ordering | Phase-C diagnosis (per-dispatch overhead = 94% of Linear) confirmed on current code + the 4090; the lever is fewer dispatches/Linear, biggest first |
| 2026-06-30 | Stage 1 = fuse FP8 dequant into matmul (not just cache the F16 cast) | Fusion removes the dispatch AND the weight-sized memory pass AND keeps weights FP8-resident (low VRAM) — solves dispatch + VRAM together; caching the F16 cast (1b) only removes the dispatch and doubles VRAM |
| 2026-06-30 | Attention deprioritized | SDPA is ~4.4% of host time on Flux; Linear dispatch overhead is the ceiling |
| 2026-06-30 | Cross-vendor deferred | NVIDIA-only box; AMD/Intel needs other hardware |
| 2026-06-30 | Landed Stage 1b first (before Stage 1) | In-shader FP8 dequant + bias fusion both need `glslangValidator` to recompile shaders — not installed on this box (sudo-gated). The weight-cast cache is the pure-C# lever and lands the dispatch-count + memory-pass win now. Measured −22.7% on the 4090 FFN microbench. |
| 2026-06-30 | Keep coopmat; do NOT fall back to tiled to cut the bias dispatch | Probe: tiled (1 fused-bias dispatch) is 2.7× slower than coopmat (2 dispatches) at the Flux FFN shape — tensor-core compute dominates here, not host dispatch overhead. The remaining win is fusing bias into the coopmat shader (Stage 2), which needs glslang. |

---

## Quick reference: run order

```
# 0. baseline (both GPUs, Vulkan + CUDA)
HARTSYINFERENCE_VK_PROFILE=1 <flux-schnell-fp8 e2e --backend vulkan>     # 3060 + 4090
dotnet test ...VulkanLinearProfileMeasurement  (HARTSYINFERENCE_VK_PROFILE=1)

# 1. fuse FP8 dequant into matmul_coopmat (shader + diff gate), A/B microbench + e2e
# 2. fuse bias (HAS_BIAS spec const), A/B  -> target 1 dispatch/Linear
# 3. QKV fusion (needs tensor view API), A/B e2e
# 4. per-dispatch binding overhead, A/B each lever
```
