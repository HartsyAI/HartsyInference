# Krea2-Turbo → sub-6.5s — Arena / CUDA-graph / F16 rewrite plan

**Goal:** Krea2-Turbo warm gen < 6.5s (Comfy baseline), coherent, on RTX 4090.
**Start state (2026-07-05, `alpha.43.131-local`):** 27.7s warm. cuDNN SDPA banked (`HARTSY_SDPA_CUDNN=1`),
host-glue block port done, patched-latent residency + device Euler step done. See
`benchmarks/results/image_comfy-vs-hartsy_2026-07-05.md` and memory `krea2-flash-attention-cudnn`.

## Diagnosis (measured, not guessed)

At 1024²/27.7s: ~6s GPU transformer + ~2s VAE + ~2.2s host op-issue + **~18s CUDA alloc/free churn**.
NOT GEMM-bound (`HARTSY_FP8_NATIVE` moved wall 0%). NOT host-dispatch-bound (all host op-issue ≈2.2s).
Wall scales ~linearly with tokens (256²=3.9 / 512²=9.2 / 1024²=30) = bandwidth + large-buffer-alloc bound.

The dominant ~18s is **stream-ordered `cuMemAllocAsync`/`cuMemFreeAsync`** per-op (`CudaMemory.cs:165,187`),
serializing against compute even with `HARTSY_MEMPOOL_KEEP`. F16 only halves buffer *size*, not this traffic.
Therefore: **Arena first (bit-identical, kills 18s) → Graph → F16.** Each phase unlocks the next.

Key enabling facts:
- Activation device pointers live in `GpuTransferHelper.ActivationCache` `Dictionary<Tensor,(ptr,bytes)>` — the
  alloc/free layer can be swapped without touching `Tensor`, the ops, or block code.
- `ForwardPatched` issues an identical op sequence every step → a deterministic allocator returns identical
  addresses every step → precondition for graph capture.
- `CudaGraph.cs` (Capture/TryUpdate/Launch, `AUTO_FREE_ON_LAUNCH`) exists + smoke-tested, never wired in.

## Phase 1 — Persistent activation arena  (flag `HARTSY_ACT_ARENA=1`)  → target ~11–13s
- Size-keyed free-list in `GpuTransferHelper`/`CudaMemory`. Alloc: exact-size bucket, pop free-list else
  `cuMemAlloc` once. Free: push to free-list, never `cuMemFreeAsync` during a gen. Release on teardown/OOM-retry.
- Weights untouched (separate `AllocatePersistent`).
- Debug poison-on-free mode to catch latent use-after-Dispose.
- **DoD:** numerically bit-identical to baseline (hard assert) + coherent + warm median recorded.
- [x] **Implemented** (2026-07-05): exact-size LIFO free-list at the `CudaMemory.AllocateAsync`/`FreeAsync` choke point
  (`HARTSY_ACT_ARENA=1`). Zero op/block/Tensor changes. Recycle on free, pop on alloc, `cuMemAllocAsync` fires only
  the first time a size is seen. OOM drains idle blocks (via `SyncStreamsAndReleasePool→DrainArena`); teardown releases
  via `ReleaseArenaForStream` (no double-free: free-list blocks are always out of `CachedPointers` before recycling).
  `HARTSY_ARENA_MAX_MB` caps retained VRAM. Hit/miss stats logged at teardown. Compiles net8.0.
- [x] **Validated 2026-07-05 — NEGATIVE RESULT.** `alpha.43.132-local`, `HARTSY_ACT_ARENA=1`: Krea2 warm **28.9s vs
  27.7s baseline (~0 change)**, VRAM 17.9→21.4 GB, image coherent. Allocation was NOT the bottleneck — `HARTSY_MEMPOOL_KEEP`
  already made allocs a cheap warm-pool pop. **Arena is safe + correct (bit-identical, coherent) but not a speed win;**
  kept in-tree default-off as CUDA-graph-capture substrate. See benchmark doc "NEGATIVE RESULT" section.

**PIVOT (evidence-based): the bottleneck is memory-bandwidth on F32 elementwise/norm activation kernels, not allocation
or graph overhead.** Four eliminations converge (fp8 0%, host-issue 2.2s, arena 0%, op-fusion 0.5s). → **Phase 3 (F16
activations) is promoted to the next action** — it is the lever the evidence supports (halves the bytes those kernels
move). Phase 2 (graph capture) is deferred: it attacks launch overhead, which is already only ~2.2s, so it can't close a
~21s gap. Reorder: **F16 (+ optional fused elementwise kernels) → then graph capture for the last mile.**

**Research-informed refinement (later / Phase 2 substrate):** GGML `gallocr` (`ggml-alloc.c:717`) is the memory-optimal
form — a *static offset plan* over the op DAG via a refcount liveness walk (free a tensor's span the instant its last
consumer runs), best-fit + coalesce into ONE arena buffer sized to the peak-live high-water mark, plus inplace aliasing
of elementwise chains (`ggml_op_can_inplace`). Our exact-size free-list is the pragmatic first cut (higher footprint —
per-size buckets, no cross-size reuse or inplace aliasing — but zero op-graph tracing and bit-identical). Adopt the
gallocr static plan when building capture if footprint or determinism needs it. Key capture prereqs the research flagged:
(1) **no `cuMemAllocAsync` mid-step** — our warm free-list already guarantees this (pops are pure C#); fold the fp8
activation-quant scratch into the arena too (ggml's one wart: MMQ scratch `cudaMalloc`s outside its plan). (2) Capture on
step ≥1, never step 0 (pointers settle). (3) Validate fp8 numerics BEFORE capture — a captured graph replays a scale bug
8× silently. (4) Honor sole-consumer + same-layout + not-output before any inplace alias. Sources: `ggml-alloc.c`,
`ggml-cuda.cu:4468` (capture, `cudaGraphExecUpdate` pointer-swap), `comfy/ops.py:812` (`_scaled_mm` fp8, weights packed).

## Phase 2 — CUDA-graph capture of the denoise step  (flag `HARTSY_GRAPH=1`)  → target ~8–9s
- Capture boundary after timestep embedding. Compute `temb`/`tembMod` on host, upload to fixed device buffers.
  Capture `img_in → concat → 28 blocks → final layer` reading `tembMod`/`txt`/`patchLatent` from fixed buffers,
  output velocity to a fixed buffer.
- CFG: one graph, launched twice/step with `txt` fixed-buffer swapped cond/uncond (same latent). Euler on-device.
- Capture on thread-local-mode stream; confirm `D2hSyncs≈0` across captured region.
- **DoD:** coherent + warm median recorded; fallback path intact.
- [ ] Not started

## Phase 3 — F16 activations WITHOUT losing packed-fp8 weights  (flag TBD)  → target ~5–6s (sub-6.5)
**VRAM constraint (do not regress):** today native fp8 (`HARTSY_FP8_NATIVE`, SM 8.9) keeps weights PACKED at
1 byte — the fp8 Linear path quantizes the *activation* to e4m3 and runs fp8×fp8, weights never cast up
(Krea2 peak ~17.9GB, not ~26GB). Naive F16 activations SKIP that path (`LinearImpl:441` needs fp8-or-F32 input)
→ either per-step fp8→F16 weight recast (slow, the Axis-B pathology) or cached F16 weights = 2× weight VRAM
→ OOM on <24GB cards. **Keep fp8 packed weights for VRAM-poor users.**

- **Fp8 activation-quant GEMM (the enabler):** add an F16→e4m3 input-quant branch to the native fp8 Linear path
  (`LinearImpl` ~line 448), generalizing `Fp8Executor`, so F16 activations quantize to fp8 at each Linear and
  multiply against still-packed fp8 weights (ComfyUI/`torch._scaled_mm`/GGML-MMQ pattern). Weights stay 1 byte.
- **Widen 6 F32-only glue ops** Krea2 uses/block so F16 activations flow between GEMMs (this is where the
  bandwidth win lives): `GatedResidualLastDim`, `AffineBroadcastLastDim`+`AddScalar`, `WanRopeInterleaved`,
  `SliceRows`, `Sigmoid`, `RepeatKvHeads` (SliceRows/RepeatKv/Sigmoid are byte-copies/simple); add native F16
  `RmsNorm`. Already-F16: Linear, SDPA, Add/Mul/Scale/Silu, Concat, Permute.
- Flip Krea2Block/Transformer/Attention activations to F16; cast boundaries only at transformer input
  (patchLatent F32→F16) + velocity output (F16→F32 for Euler).
- Numerical watch: F16 65504 ceiling in SwiGLU (BF16 fallback if it overflows); QK-norm makes attention F16-safe.
- **DoD:** coherent (per-stage relL2 vs F32 during bring-up) + warm median < 6.5s + peak VRAM ≤ today's ~17.9GB.
- [ ] Not started

## VRAM notes (all phases)
- Phases 1–2 touch only ACTIVATION buffers, not weight storage → no weight-VRAM regression.
- Arena free-list holds a bounded working set of activation blocks; add a trim/cap knob so it doesn't balloon
  activation VRAM on small cards (release-on-teardown + optional max-reserved).
- Keep native fp8 packed-weight path intact throughout; it is a VRAM feature (enabling it moved wall 0%).

## Per-phase validation / deploy
Deploy per handoff §4 (`KREA2_CUDNN_SDPA_HANDOFF.md`): build → pack new `-local` → bump pin → `rm` built ext
folder → relaunch with `LD_LIBRARY_PATH=~/.local/lib/cuda13` + `HARTSY_SDPA_CUDNN=1 HARTSY_MEMPOOL_KEEP=1`.
Bench: `benchmarks/swarm_image_bench/bench_t2i.py --only Krea2-Turbo`. Always view the output PNG (coherent
astronaut-on-horse). Record in `benchmarks/results/image_comfy-vs-hartsy_2026-07-05.md`.
