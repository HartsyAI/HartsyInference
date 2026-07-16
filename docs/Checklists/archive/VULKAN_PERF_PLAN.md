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

## Status (2026-06-30 session) — FP8 Linear 3 dispatches → 1, −32.9%

| Stage | Outcome | 4090 microbench |
|---|---|---|
| **1b** weight-cast cache (C#) | ✅ shipped | 3.65 → 2.82 ms, 3→2 disp (−22.7%) |
| **2** bias fused into coopmat (shader) | ✅ shipped | 2.82 → 2.45 ms, 2→1 disp (−13.1%) |
| **4** submit batching (C#) | ✅ shipped | tiny-op host −15.2% (48→41 µs/disp) |
| 4 descriptor recycling | 🔬 tried → reverted (no-win on NVIDIA) | ~0% |
| 4 pipeline-bind dedup | 🔬 tried → reverted (no-win on NVIDIA) | ~0% |
| (probe) coopmat vs tiled | negative result recorded | tiled 2.7× slower — keep coopmat |
| **1** in-shader FP8 dequant | ⤵ demoted (VRAM-only; cache already cut the dispatch) | — |
| **3** QKV fusion | ⛔ blocked (concurrent model-code rewrite + no e2e) | — |
| **5** cross-vendor — preemptive crash fixes | ✅ shipped (3 latent AMD/Intel pipeline-failure bugs), perf tuning needs HW | no-op on NVIDIA (verified) |

Cumulative **3 dispatches → 1**, **−32.9%** on the Flux FFN FP8 Linear, coopmat tensor-core path kept. All
gates green (44 Vulkan correctness/leak/int8 tests). Vulkan still ~2–6× off CUDA — the remaining gap is the
coopmat compute itself plus the per-dispatch host floor that needs a deeper, e2e-validated rework. Tooling:
`glslangValidator` fetched as a Khronos prebuilt binary (no root) — see decision log.

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

> **Re-scoped after Stage 1b + 2 (2026-06-30):** the dispatch-count win Stage 1 targeted is **already
> captured** — Stage 1b caches the FP8→F16 cast (0 extra dispatches after warmup) and Stage 2 fused bias, so
> a FP8 Linear is already at **1 dispatch**. In-shader FP8 dequant no longer reduces dispatches; its only
> remaining benefit is **VRAM** (weights stay FP8-resident instead of doubling with the F16 cast cache). It is
> also the most complex/risky shader change (coopmat can't `coopMatLoad` FP8 directly — needs a shared-memory
> stage: load FP8 → dequant → `coopMatLoad` F16 from shared). **Now LOWER priority than Stage 4**; do it only
> when VRAM is the binding constraint (large models near capacity), or pair with the low-VRAM flag.

- [ ] Add an FP8 (and later GGUF-quant) load path to the coopmat GEMM shader (spec-const `WEIGHT_FP8`),
      dequant via a shared-memory tile stage. Keep a non-fused fallback for unsupported dtypes.
- [ ] Gate: new diff test (fused FP8-matmul vs cast-then-coopmat) within FP8 noise (~1e-2 rel).
- [ ] A/B microbench + e2e: expect **VRAM ↓ ~½ weight footprint**, wall-clock ≈ flat (dispatch already 1).
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

## Stage 2 — ✅ DONE (2026-06-30): Fuse bias into the coopmat matmul

A `HAS_BIAS` spec-const path in `matmul_coopmat.comp.glsl` removes the follow-up `BroadcastAdd` dispatch.

- [x] Added `HAS_BIAS` (constant_id = 16) to the coopmat shader. Bias added in the epilogue via a
      **stride-0 row-major `coopMatLoad`** — broadcasts the 16 per-column bias values across all 16 rows of
      the accumulator fragment, then `acc = acc + biasFrag`. Binding 3 redeclared `float bias[]` (FP32, the
      accumulator type — no per-element f16 conversion). Compiled with glslangValidator 16.3.0.
- [x] C#: `SpecConstant.Bool(16, bias is not null)` + bind the FP32 bias buffer at slot 3 in
      `TryDispatchCoopmat`; bias cast to FP32 once and cached (preloaded weight). Removed the `BroadcastAdd`.
- [x] Gate: `Backend_Linear_FP8Weight_CachedCast_Matches_Cpu` (F16 output) + `Backend_Linear_FluxShape_F32_Matches_Cpu`
      (F32 output, fused F32 bias) + full smoke/leak/int8 suite (39 tests) green.
- [x] A/B microbench (4090, with 1b on): **2.82 → 2.45 ms (−13.1%), 2 → 1 dispatch** per Linear.
- [ ] A/B e2e Flux Schnell + SSIM — NOT run this session (e2e needs the Diffusion project, which a concurrent
      edit was breaking, plus a model checkpoint). Re-run when the tree is clean.

> **Headline target achieved (1b + 2): a FP8 Linear goes 3 dispatches → 1** (cached dequant + matmul + fused
> bias), keeping the coopmat tensor-core path. Cumulative −32.9% on the FFN microbench. The remaining cost is
> the coopmat compute itself (probe: tensor-core-bound at this shape) + per-dispatch host overhead (Stage 4).

---

## Stage 3 — QKV projection fusion (3 Linears → 1) — DEFERRED (blocked)

Hypothesis (Phase-C carryover): concat the Q/K/V weight matrices at load so the three sequential Linears
become one matmul `[batch, seq, 3·hidden]`, then tensor-view-slice into Q/K/V. Cuts ~2/3 of the Q/K/V
Linear dispatches (Q/K/V are ~21% of Linears → ~14% of total). Needs a tensor view/slice API (only
`Reshape` exists today) + per-block weight-load changes in the Flux/Flux2/SDXL attention blocks.

**Status (2026-06-30): not started — two hard blockers, deliberately not forced.**
1. **Touches model code under active concurrent rewrite.** The Flux/Flux2/SDXL attention blocks live in
   `HartsyInference.Diffusion`, which a parallel effort was rewriting this session (commits `288396e`,
   `591e6ad`, `ded0ef0` swept denoiser/block files). Editing the same attention blocks now would conflict.
2. **Can't validate.** The gate is "attention output unchanged + e2e SSIM", and e2e is unavailable here
   (no model checkpoint + the Diffusion build was intermittently broken by the concurrent edits). Shipping a
   weight-concat + view-slice change to attention without the SSIM gate violates guardrail #1.

- [ ] Tensor view/slice API (no-copy sub-view) on `Tensor` — the reusable prerequisite, safe to land in
      `Core` with unit tests once the tree is quiet; do this first.
- [ ] Concat QKV weights at load in the affected blocks (after the tree settles + a checkpoint is available).
- [ ] Gate: attention output unchanged (vs-unfused diff + SSIM); A/B e2e dispatch count + wall-clock.

---

## Stage 4 — Per-dispatch binding overhead — submit-batching DONE; finer levers no-win on NVIDIA

The remaining lever once dispatch count is minimized: make each surviving dispatch cheaper. Measured floor
(2026-06-30, 4090, tiny Linears, no per-op Sync): **~48 µs/dispatch** of host overhead before this stage.
**Net outcome:** submit-batching landed (−15%); descriptor recycling and pipeline-bind dedup were both
implemented, measured, and **reverted as no-wins on NVIDIA** (cheap pool-alloc + cheap bind; the floor is
update/bind/barrier-record/C# marshal, which need a deeper rework with e2e validation to move).

- [x] **Submit batching (DONE):** the per-op path was issuing a `vkQueueSubmit2` at the end of *every* op
      (`OpScope.Dispose`). Now transients drain per op (correctness) but the submit is batched until
      `FLUSH_THRESHOLD` dispatches accumulate; `Sync()`/lazy-sync force-flush. **−15.2% on tiny-op host wall
      (97.4 → 82.6 ms, 48 → 41 µs/dispatch).** Escape hatch `HARTSYINFERENCE_VK_SUBMIT_PER_OP=1`. Gate: 39
      smoke/leak/int8 green. (Big-shape FFN bench Syncs per call, so it's unaffected — this targets the
      attention/norm-heavy parts and the 3060.)
- [x] **Descriptor-set recycling — TRIED, REVERTED (negative result):** implemented tick-gated set reuse
      (free-list + in-flight-by-tick, reclaimed once `stream.CompletedTick` passes the set's submit tick — so a
      reused set is never referenced by in-flight GPU work). Correctness-safe (leak + SDPA-multihead green), but
      **0% measurable win** (81 ms → 79.6–90.5 ms, within run-to-run noise): on NVIDIA `vkAllocateDescriptorSets`
      is a cheap pool bump-allocator, and the recycle bookkeeping + per-refill timeline read cancel the saving.
      Reverted to keep the codebase clean. The remaining ~41 µs/dispatch is update/bind/barrier-record/C# marshal,
      not the alloc. Revisit only on AMD/Intel (costlier descriptor alloc) with e2e validation.
- [ ] Coalesce compute-to-compute barriers where dependencies allow (the global barrier after every dispatch
      is conservative). Lower host-side value (one barrier struct is cheap); mainly a GPU-serialization win, so
      only matters where GPU-bound.
- [ ] Re-evaluate push descriptors **per vendor** — a measured regression on NVIDIA (deviation #12),
      possibly a win on AMD/Intel. Keep `HARTSYINFERENCE_VK_PUSH_DESCRIPTORS` opt-in.

---

## Stage 5 — Cross-vendor (AMD / Intel) — preemptive correctness fixes DONE; perf tuning needs HW

This box is NVIDIA-only (3060 + 4090), so the cross-vendor *proof* (AMD RDNA2/3, Intel Arc) still needs
other hardware. But the audit surfaced three latent **crash/garbage** bugs where the fast paths assumed
NVIDIA's generous limits — those are fixable blind and **verified behaviour-neutral on NVIDIA** (39 tests
green + FFN microbench unchanged at 2.41 ms / 1 disp), so they're done now:

- [x] **`matmul_tiled` shared memory exceeded the 32 KB spec minimum (the only GEMM fallback when coopmat is
      absent → device unusable).** The shared `Asub`/`Bsub` were statically `MAX_BM/MAX_BK/MAX_BN`-sized FP32 =
      **33,280 B** (the in-code comment even mis-counted, ignoring the `+1` pad). On AMD GCN / Intel Arc /
      iGPU (exactly 32 KB) every tiled pipeline was invalid. **Fix:** size the shared arrays by the
      *specialized* `BM/BK/BN` (valid Vulkan GLSL), so a 64×64×16 tile costs 8,448 B and 32×32×16 costs
      4,224 B. Recompiled with glslangValidator 16.3.0.
- [x] **`PickMatmulTile` ignored `maxComputeSharedMemoryBytes`.** Now picks the largest tile whose FP32
      shared footprint fits the device (128→64→32), so low-shared-mem devices degrade gracefully instead of
      failing pipeline creation. No-op on NVIDIA (48–64 KB → still 128).
- [x] **Coopmat workgroup invocations could exceed device limits.** `(BM/16)*(BN/16)*subgroupSize` is 512 on
      NVIDIA (subgroup 32) but **1024 on AMD wave64** for a 64×64 tile, over `maxComputeWorkGroupInvocations`/
      `sizeX` on smaller parts. Now shrinks the coopmat tile (never below the 16×16 fragment = one subgroup,
      always ≤ limit) until it fits. No-op on NVIDIA.
- [ ] **Perf tuning (needs HW):** vendor tile tables (AMD wave32 128×64, wave64 64×64, Intel Arc 64×64), the
      `requiredSubgroupSize` pin across RDNA wave32/64, and re-evaluating push descriptors on AMD/Intel.
- [ ] **Watch-item:** the Stage-2 fused-bias uses a stride-0 `coopMatLoad` broadcast — spec-legal and fine on
      NVIDIA; re-verify on the first AMD RDNA3 / Intel Arc run (some early coopmat impls were picky about
      non-unit strides). If it misbehaves, gate `HAS_BIAS` off there and fall back to the BroadcastAdd path.

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
| 2026-06-30 | 4090 | 4 submit-batch | `HARTSYINFERENCE_VK_SUBMIT_PER_OP` off (default batched) | 2000 tiny Linears (M=16 K=64 N=64), no per-op Sync | host-wall, µs/dispatch | 97.4 ms, 48 µs/disp (submit-per-op) | **82.6 ms, 41 µs/disp** | **−15.2%** | **pass** | Was submitting one `vkQueueSubmit2` per op (`OpScope.Dispose`). Now drains transients per op but batches the submit until `FLUSH_THRESHOLD`(8) dispatches; `Sync()`/lazy-sync still force-flush. Helps small-op-heavy workloads (attention/norms, the 3060). Big-shape FFN bench unaffected (it Syncs per call). Gate: smoke/leak/int8 (39) green. Escape hatch: `HARTSYINFERENCE_VK_SUBMIT_PER_OP=1`. |
| 2026-06-30 | 4090 | 5 cross-vendor crash fixes | spec-const shared arrays + shmem/invocation-aware tile pick | Flux FFN coopmat + tiled GEMM | ms/call, gate | 2.45 ms | 2.41 ms (noise) | **~0% (no-op on NVIDIA)** | pass (39) | Correctness, not perf: `matmul_tiled` shared mem was 33,280 B > 32 KB spec floor (unusable on AMD GCN/Intel Arc); tile pick ignored shmem; coopmat invocations could exceed limits on wave64. All fixed, verified behaviour-neutral on NVIDIA. |
| 2026-06-30 | 4090 | 4 descriptor-recycle | tick-gated set reuse (reverted) | 2000 tiny Linears | host-wall | 81 ms | 79.6–90.5 ms (3 runs) | **~0% (noise)** | pass (42 tests) | **Negative result, reverted.** Tick-safe descriptor-set recycling (free-list + in-flight-by-tick, reclaim on completed timeline) removed per-dispatch `vkAllocateDescriptorSets` but gave NO measurable win — on NVIDIA that alloc is a cheap pool bump-allocator, and the recycle bookkeeping + timeline read cancel it. Correctness-safe (leak + SDPA-multihead green) but pure added complexity, so reverted per guardrail #5. The ~41 µs/dispatch floor is elsewhere (update/bind/barrier-record/C# overhead). |

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
| 2026-06-30 | Stage 2 bias via stride-0 broadcast coopmat load | Per-column bias add to a cooperative-matrix accumulator can't index elements (lane→(i,j) mapping is opaque). A stride-0 row-major `coopMatLoad(biasFrag, bias, outCol, 0, RowMajor)` broadcasts bias[outCol+j] to every row; `acc + biasFrag` then matches BroadcastAdd. Works on NVIDIA (verified F16 + F32 output). |
| 2026-06-30 | glslang obtained via Khronos prebuilt binary (no root) | `glslang-tools` apt package is sudo-gated and unavailable non-interactively. Downloaded the official `glslang-main-linux-Release.zip` (v16.3.0) from KhronosGroup/glslang releases to a local dir and ran build.sh with `GLSLANG=<path>`. spirv-val absent → build.sh skips validation (still emits valid SPIR-V; verified by NVIDIA driver accepting the module + green correctness gates). |
| 2026-06-30 | Stage 1 (in-shader FP8) demoted below Stage 4 | With 1b caching the cast, Stage 1 no longer cuts dispatches — VRAM-only benefit, highest shader complexity. |
| 2026-06-30 | Stage 4 finer levers (descriptor recycle, pipeline-bind dedup) reverted | Both correctness-safe but ~0% on NVIDIA (cheap pool-alloc + cheap bind); guardrail #5 says don't carry unmeasured complexity. Submit-batching is the kept win. |
| 2026-06-30 | Stage 5 done as preemptive *correctness* fixes, not perf | Can't perf-tune blind, but the audit found 3 latent bugs where the fast paths assumed NVIDIA limits (tiled shared-mem > 32 KB spec floor; tile selection ignoring shmem; coopmat invocations > limit on wave64) that would crash/garbage on AMD/Intel. Fixed + verified no-op on NVIDIA (39 tests + microbench). Cross-vendor perf tuning + the stride-0-bias watch-item still need real HW. |

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
