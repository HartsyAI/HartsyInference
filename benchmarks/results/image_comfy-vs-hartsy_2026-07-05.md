# Image models — HartsyInference vs ComfyUI end-to-end + optimization plan (2026-07-05)

End-to-end wall-clock through the **SwarmUI API**, same generation routed to the ComfyUI backend then the
HartsyInference backend on the **same GPU (RTX 4090, SM 8.9, `GPU_ID:0`)**. Companion to the video benchmark
([`video_comfy-vs-hartsy_2026-07-03.md`](video_comfy-vs-hartsy_2026-07-03.md)); same methodology.

**Method:** 1024² (native fallback where an arch can't), identical params both backends, per-model step counts
(turbo/distilled models get their low step counts). 1 warmup + 3 warm gens, **random seed each** (defeats
SwarmUI's identical-params result cache), warm = median. Backends toggled one-at-a-time via `/API/ToggleBackend`
so routing is deterministic. Peak VRAM sampled during each gen. Harness: `scratchpad/bench_t2i.py` + `models.json`.

**Backend config (4090):** `ComputeBackend=cuda`, `NativeFp8Gemm=auto→enabled` (SM 8.9), `CacheWeightCasts=off`.
Launch via `launch-linux-dev.sh` with `LD_LIBRARY_PATH=/home/hartsy/.local/lib/cuda13` (cuBLAS 13.1) — **without
that env the backend errors `libcublas.so.13 not found`.**

## Results — warm generation (model resident)

| Model | Steps | Comfy warm | Hartsy warm | **Gap** | Hartsy peak VRAM |
|---|---|---|---|---|---|
| AuraFlow-0.3 | 20 | 14.0 s | 34.5 s | **2.5×** | 11.8 GB |
| Flux-Dev (fp8) | 20 | 12.5 s | 74.8 s | **6.0×** | 6.8 GB |
| Qwen-Image (Q4_K) | 20 | 54.8 s | 380.7 s | **7.0×** | 16.6 GB |
| Krea2-Turbo (fp8) | 8 | 6.5 s | 70.6 s | **10.9×** | 17.9 GB |
| ERNIE-Image (fp8) | 20 | 24.0 s | 296.4 s | **12.4×** | 17.9 GB |
| SDXL | 20 | 3.7 s | 46.3 s | **12.6×** | 7.3 GB |
| Ideogram4 (fp8) | 20 | 17.0 s | 42.3 s | **2.5×** | 23.7 GB |
| Z-Image-Turbo (fp8) | 8 | 3.1 s | 52.1 s | **16.8×** | 16.8 GB |
| **Chroma1-HD (fp8)** | 20 | 16.6 s | **550.0 s** | **33.2×** ⚠️ | 13.0 GB |
| HiDream-i1 (fp8, 17B) | 25 | 35.2 s | **FAILED** (cold 1462 s) | — | ~24 GB (OOM) |

_Ideogram4 needs a structured JSON caption (`high_level_description`+`compositional_deconstruction`) or it renders
the grey safety-filter placeholder; both backends were given the JSON prompt._

**HiDream-i1:** cold gen took **1462 s (24 min)**, then the first warm rep **failed** ("Something went wrong") —
17B fp8 at 1024² hit the 24 GB ceiling (Ideogram4 already peaked at 23.7 GB). Both catastrophically slow *and*
unstable.

## Models SwarmUI refused (not a speed result — routing/wiring gaps)

| Model | Cause | Fix |
|---|---|---|
| **Boogu Base/Turbo/Edit** | `arch_id:null` — this stale SwarmUI core (0.9.8.1) predates the built-in `boogu` detector; the extension intentionally relies on core (`SwarmUIHartsyInference.cs:77`) | **Update SwarmUI core** (Boogu is supported in current Swarm) |
| **Lumina2** | Mis-detected as `omnigen-2`; real `.safetensors` missing (only diffusers variant present) | Detector fix + weights |
| **OmniGen2** | `omnigen-2` compat "pending" — engine pipeline exists + e2e-verified, but the extension per-arch loader (TE/VAE/tokenizer) is a TODO | Wire the `omnigen-2` loader |

## Root-cause taxonomy (4 independent investigations + engine profiling — all converge)

The gap is **host/launch-bound, not raw compute.** For Chroma@1024²: 277 s wall, **only ~35 s is GPU** (Linear
19.5 s + SDPA 12 s); the other **~240 s is host-side** (`TODO_CHROMA_GPU_RESIDENCY.md`). Two axes, by model:

### Axis A — host-glue: DiT blocks run ~20 CPU ops/forward (the "GPU-not-CPU" problem)
The dominant gap on most models. Block code (reshape/concat/split/QK-norm/AdaLN-modulation/RoPE/final-norm) runs
nested CPU loops over `Tensor.DataPointer` — each read forces a full GPU→host sync of the Q/K/V tensor, then a
re-upload (cache miss). GPU sits ~5–11 % utilized. **Flux/SD3/Qwen/Ideogram4 got the GPU-residency port; Chroma,
Krea2, ERNIE, HiDream, Lumina2 did not** — that is exactly the gap ordering (ported models 2.5–7×, un-ported
10–33×). Proven fix: port blocks to device ops → **2.85× on Wan** with no new kernels, bit-identical.
- Un-ported hot-path `DataPointer` counts: Chroma (14 Transformer + 5+4+3 blocks), ERNIE (11), Krea2 (5+4), HiDream (4).
- Reference template: `Ideogram4Block` (the one already-clean block) — uses `backend.RmsNorm`/`Permute0213`/
  `SliceLastDim`/`AffineBroadcastLastDim`/`GatedResidualLastDim`/`ApplyRope`/`ScaledDotProductAttention`, never
  touches `DataPointer`.
- **NOT just RoPE:** `HARTSY_SKIP_ROPE=1` saved only 5 s on Chroma — the whole block must be ported, not one op.

### Axis B — fp8 per-step weight re-cast (the "proper fp8 / casts" problem, large models)
`CudaBackend.LinearImpl:403`: the native fp8 tensor-core path fires only when activations are **fp8 or F32** — but
the DiTs run **F16 activations**, so every fp8 weight is **re-cast fp8→F16 every step** (with `CacheWeightCasts=off`,
needed to fit VRAM). For HiDream 17B that's re-casting the whole model 25×. This is the large-fp8 differentiator
(HiDream), *not* Chroma (Flux pays the same fp8 path and is only 6×). ComfyUI/GGML avoid it: **quantize the
activation to fp8/int8 and multiply with weights kept packed, dequantizing only in the accumulate** (`torch._scaled_mm`
/ GGML MMQ). Hartsy's `Fp8Executor` already does this for HunyuanVideo (`HARTSY_FP8_NATIVE`) — generalize it as the
default fp8 matmul path so image DiTs stop upcasting.

### Cross-cutting levers (universal, from the reference engines)
| Lever | What | Expected | Status |
|---|---|---|---|
| **Fused flash-attention PTX** | one kernel, online-softmax, never materialize `[H,S,S]` scores; tensor cores | ~half of DiT GPU time; 2–5× on attention | tiled SDPA exists (mem fix); true fused kernel written but slower than cuBLAS — open |
| **fp8×act fused MMA GEMM** | GGML MMQ pattern (Axis B) | ~2× on fp8 step; removes recast + OOM | `Fp8Executor` exists, not default |
| **Standing fused norm/AdaLN/SwiGLU kernels** | reusable RMSNorm+scale+shift, GroupNorm+SiLU, Linear+bias+act epilogue | 1.3–1.8×/block; kills the per-model glue firefight | partial (GroupNormSilu) |
| **Op-recorder + graph scratch arena → CUDA-graph capture** | GGML-style: record ops, one pooled alloc, capture the fixed per-step graph, relaunch | removes ~1/3 launch overhead; diffusion is an ideal graph target | CudaGraph capture **proven**, not wired |
| **fp16 cuDNN-quality VAE conv, untiled default** | im2col→implicit-GEMM, fp16; tile only on OOM | 2–4× VAE decode | im2col + F32; tiled-decode has a separate 1024² bug |

## Ranked execution plan

**Phase 1 — host-glue block ports (Axis A), one model at a time, each parity-checked.** Highest leverage,
lowest risk, proven playbook, template exists.
1. **Chroma** (33× → target ~6×) — execute `TODO_CHROMA_GPU_RESIDENCY.md` (already scoped, op-by-op table + template)
   + build the SDPA mask once/forward on GPU instead of per-block. Worst gap, clearest fix.
2. **ERNIE** (12.4×), **Krea2** (10.9×) — same port, replicate the Chroma pattern.
3. **HiDream** (failed) — port blocks **and** address VRAM/fp8 (Axis B) so it fits + stops re-casting.

**Phase 2 — fp8 activation-quant GEMM as default (Axis B).** Generalize `Fp8Executor`/`HARTSY_FP8_NATIVE` so fp8
DiTs stop the per-step F16 weight re-cast. Unblocks HiDream stability + speeds every fp8 model.

**Phase 3 — cross-cutting kernels.** Fused flash-attention → standing fused norm/AdaLN kernels → CUDA-graph
capture of the denoiser step. These lift the whole fleet (incl. the already-ported Flux/SDXL band toward ~2×).

**Phase 4 — SwarmUI wiring.** Update core (unblock Boogu), wire `omnigen-2` loader (OmniGen2/Lumina2), fix Lumina2 detection.

## Optimization results (host-glue GPU-residency ports)

Fixing the per-block host-CPU RoPE excursion (move rotation to the device `WanRopeInterleaved` kernel on the
pre-permute `[B,S,H,D]` layout — bit-identical, mirrors the Flux/Ideogram GPU-resident pattern). Engine
`alpha.43.119/.120-local`.

| Model | Fix | Comfy | Before | After | Speedup | Gap now | Output |
|---|---|---|---|---|---|---|---|
| **Krea2-Turbo** | GQA GPU-RoPE (`FluxRope.ApplyGpuGqa`, per-tensor head counts) | 6.5s | 70.6s (10.9×) | **36.7s** | **1.9×** | **5.6×** | ✅ verified coherent |
| **Chroma1-HD** | MHA GPU-RoPE both blocks (`ApplyGpu` pre-permute) + SDPA mask built once/forward in `ChromaTransformer` (was 85 MB `[B,1,S,S]` host build × 57 blocks) | 16.6s | 550.0s (33.2×) | **119s** | **4.6×** | **7.2×** | ✅ verified coherent |
| **ERNIE-Image** | GPU rotate_half RoPE (`ErnieImageRope.ApplyRotaryEmbGpu`: slice packed freqs → cos/sin on-device, `backend.ApplyRope`) — was host `ApplyRotaryEmb` per block | 24.0s | 296.4s (12.4×) | **53.3s** | **5.6×** | **2.2×** | ✅ verified coherent |

Chroma breakdown: 550s → 270s (GPU-RoPE, 2.0×) → **119s** (+ mask-once, another 2.3×). It went from the **worst
model in the set (33×) to mid-pack (7.2×)**.

_Remaining host-glue not yet ported: Krea2 transformer final-norm + patchify; ERNIE 3D-RoPE (`ErnieImageRope`) +
AdaLN/patchify host loops (a larger, different port); HiDream (host-glue **and** needs fp8 activation-quant GEMM
to fit 24 GB + stop the per-step fp8→F16 weight recast — it OOM-failed the benchmark). Cross-cutting levers
(fused flash-attention, CUDA-graph capture) lift the whole fleet including the already-ported models._

## Recommended first target: **Chroma**
Worst gap (33×), the fix is already scoped in `TODO_CHROMA_GPU_RESIDENCY.md` with an op-by-op conversion table and a
working reference block (`Ideogram4Block`), it's bit-identical (zero numerical risk), Chroma-only (no blast radius),
and it validates the block-port playbook that then replicates to ERNIE/Krea2/HiDream.

## Optimization results (cuDNN fused flash-attention SDPA) — engine `alpha.43.124-local`

Wired cuDNN 9.24's fused flash-attention engine as a new SDPA fast path (`HARTSY_SDPA_CUDNN=1`, env-gated,
self-disables on any cuDNN failure). Pure-C# backend-graph P/Invoke (`CudnnApi.cs` + `CudnnSdpa.cs`), no C++
shim. The graph mirrors cudnn-frontend's plain-inference SDPA: `bmm1 → pointwise-scale → unified SOFTMAX op →
bmm2`, 4D `[B,H,S,D]`, fp16 I/O + fp32 accum, all intermediates virtual (never materializes the score matrix).
The **unified backend SOFTMAX op** (cuDNN ≥ 9.21) is the key — the decomposed max/sub/exp/sum/div graph does
**not** match the fused engine (all engines return NOT_SUPPORTED).

**Per-call SDPA GPU cost (profiled, `HARTSY_PROFILE_SYNC`, Krea2 B=1 H=24 S=4608 D=128):**

| | cuBLAS materialized (F16) | cuDNN fused | speedup |
|---|---|---|---|
| per SDPA call | **62.7 ms** | **5.5 ms** (1.8 ms execute + ~3.6 ms F32↔F16 casts) | **11.4×** |
| SDPA share of GPU time | 52% | ~12% | — |
| score-matrix workspace | ~2 GB | **0** | — |

Prototype pure-execute at this shape = **1.82 ms** (34.5× over 62.7 ms). Validated vs numpy + CPU reference
(relL2 2.7e-4; C# unit test `CudnnSdpaTests` D∈{64,128}, `CudnnSdpaEngaged` asserted). Output **coherent**
(clean astronaut-on-horse @ 1024², seeds 111/222/83865+).

**Wall time (warm, RTX 4090, no profiler):** Krea2-Turbo **36.7s → 33.0s** (≈1.1×). Gap vs Comfy 6.5s: 5.6× → **5.1×**.

**Why the wall barely moved despite an 11× SDPA win — Krea2 is HOST-BOUND, not GPU-bound.** Summing all GPU
ops (serialized profile) ≈ **15 s/gen**, but wall is 33 s → ~18 s is host launch overhead / non-overlapped H2D
that reducing GPU work can't touch. Post-cuDNN profile top ops (2 gens): `Linear` 12.4 s (4714 calls),
`SDPA` 3.2 s (584), `GatedResidual` 3.1 s (896), `RmsNorm` 2.7 s (2360), `Conv2D` 2.6 s, `Permute0213` 2.2 s,
`H2D_MISS_BIG` 1.7 s (126). The cuDNN win is **real and banked** but only surfaces in wall time once the
host-launch tail is attacked. **Next = "B":** CUDA-graph capture of the denoise step + fuse the DiT block glue
(`GatedResidual`/`RmsNorm`/`AffineBroadcast`/`Permute` = the still-pending Krea2 block-GPU-residency) + kill the
`H2D_MISS_BIG` re-uploads. SDPA is no longer the bottleneck; the per-op host launch overhead is.

Kept env-gated (not default-on) because it is not yet a clear wall win on this host-bound pipeline (it adds the
3+1 F32↔F16 cast launches); flip to default-for-RMS-normed-archs once B lands and the GPU savings surface.

## Krea2-Turbo host-overhead reduction ("B") — engine `alpha.43.131-local`

After cuDNN SDPA banked the attention GPU win but wall stayed ~35s, profiling showed Krea2 is **not** GEMM-bound
(enabling `HARTSY_FP8_NATIVE` fp8 tensor cores changed wall 0%) and **not** host-op-dispatch-bound (non-blocking
profile: total host issue time for ALL ops ≈ **2.2 s**; SplitModulation fusion removing ~2.5k ops/gen gave ~0.5 s).
Resolution sweep (256²=3.9s, 512²=9.2s, 1024²=30s) proves it scales ~linearly with token count = **memory/bandwidth
+ large-buffer allocation bound**. SYNC profile: profiled GPU kernels ≈ **6 s/gen** (transformer) + ~2 s VAE. The
remaining ~18 s is the CUDA driver cost of allocating/freeing **thousands of 113–300 MB F32 activation buffers per
generation** (eager per-op `cuMemAllocAsync`/`FreeAsync`) plus non-overlapped memory ops — not addressable by any
single op-level change.

Shipped, all coherent (verified each step), cumulative **36.7s → 27.7s** on top of prior work:

| Change | Effect | Δ |
|---|---|---|
| Text path cached across steps (was recomputed every step: ~95 ops + 4 text SDPAs) | fewer ops/gen | ~2s |
| RoPE precompute + head/tail (final-layer modulate, SliceTail) made device-resident (removed host loops + D2H drains) | fewer drains | ~1s |
| Patched-latent residency: latent kept in `[1,imgSeq,64]` token space across the whole loop, on-device Euler step (Scale+Add) — patchify/unpatchify once, no per-step D2H | removed 8 drains/gen | ~0s (drains weren't the bottleneck) |
| `cuMemAllocAsync` pool release-threshold raised (`HARTSY_MEMPOOL_KEEP`, opt-in) | warm activation reuse | ~1.4s |
| Device `Concat` (was host `ConcatAlongSeqDim`: per-step 113MB D2H+memcpy+H2D) | removed per-step drain | ~1s |
| SplitModulation fused 18→7 ops/block (flattened table + 1 Add + 6 slices) | −2.5k ops/gen | ~0.5s |

**Wall now 27.7s vs Comfy 6.5s (gap 5.6× → 4.3×).** The two remaining levers, both **major rewrites**, are what
close the rest: (1) **F16/BF16 activations** — halves every activation buffer (alloc cost) AND all bandwidth-bound
elementwise/attention traffic (the dominant GPU cost; fp8 doesn't touch it); (2) **CUDA-graph capture / persistent
activation arena** — eliminates the per-op alloc/free churn by reusing buffers and replaying the denoise step.
Estimated: F16 → ~16–18s, + graph/arena → GPU-floor ~8s, then within reach of 6.5s. `Krea2Transformer.ForwardPatched`
+ device Euler step already give a clean, drain-free per-step region to capture.

## 2026-07-07 — Krea2-Turbo BEATS ComfyUI: 27.7s → **5.83s** (Comfy 6.5s) — engine `alpha.43.137-local`

**Final: Hartsy 5.83s warm median vs Comfy 6.5s = 1.11× FASTER. From 70.6s (10.9× slower) at the start of the effort.**
Coherent astronaut-on-horse verified at every stage. Flags: `HARTSY_SDPA_CUDNN=1 HARTSY_FP8_NATIVE=1
HARTSY_DIT_F16=1 HARTSY_KEEP_MODELS=1 HARTSY_MEMPOOL_KEEP=1`. Peak VRAM 23.9 GB (fits 24 GB, weights stay packed fp8).

### The decisive discovery: the gap was the VAE, not the DiT
Wall-clock **phase probes** in `Krea2Pipeline` (op profiles only summed ~5.5s of the 27s — the rest was invisible
to op-level profiling) attributed the 27s: TE 0.9s / DiT re-upload 1.5s / **denoise 8 steps = 3.9s** / **VAE decode
= 20.1s** / rgb 0.01s. The transformer was ALREADY Comfy-class after the cumulative optimizations; the entire
remaining gap was `QwenImageVaeDecoder` running its RMS norms + residual adds as **host CPU loops** (a comment
claimed per-gen amortization made GPU kernels unnecessary — false at 1024²): ~35 × (D2H drain of up-to-400 MB
conv output + CPU triple-loop + H2D re-upload). This also explains why the arena (0%) and F16 (≈0% wall) "failed":
they correctly optimized a transformer that was already fast. Op-level elimination experiments cannot see
un-instrumented host phases — **instrument wall-clock phases first.**

### What shipped (alpha.43.133 → .137)
| Change | Effect |
|---|---|
| **F16 activations for the Krea2 DiT** (`HARTSY_DIT_F16`): 8 new `__half` PTX kernels (dit_f16.cu via nvrtc — RmsNorm/AffineBroadcast/GatedResidual/AddScalar/Sigmoid/RopeInterleaved/RepeatKv/SliceRows) + F16 branches in CudaBackend + block/attention conversion + zero-cast native-F16 cuDNN SDPA + **F16→e4m3 activation-quant** (`absmax_f16`/`quant_f16_e4m3`) so the native fp8 GEMM keeps weights PACKED (no per-step recast, no VRAM regression) | norm/residual/permute GPU time ↓8–19×; wall ≈0 (DiT wasn't the wall) — but it's what makes the 3.9s denoise loop this fast, and F16-kernel parity is unit-tested (`DitF16KernelTests`) |
| **Qwen VAE GPU-residency port**: `WanRmsNormChannel` (existing kernel) replaces host `RmsNormPerPixelAcrossChannels` in residual blocks + attention + head; device `Add` residuals; device SliceRows+Transpose2D QKV split; device Clamp | **VAE decode 20,139ms → 16ms (~1250×)**; wall 27.0 → **7.4s** |
| **`HARTSY_KEEP_MODELS=1`**: DiT weights stay GPU-resident across gens (skip post-loop free + next-gen ~1.8s re-upload). TE still freed each gen — its VRAM is required by the VAE decode's 6.9 GB im2col (keeping all three OOM'd) | wall 7.4 → **5.83s** |

### Warm-gen phase profile at 5.83s (probe logs)
TE preload+encode+free 1.13s · DiT preload 0ms (resident) · denoise 8 steps 3.95s (~525ms/step, step2 ~220ms) ·
VAE decode 271ms · rgb 148ms.

### Remaining headroom (next targets, in order)
1. **TE churn ~1.1s/gen** — cache prompt embeddings across gens (same prompt = free), or pin the TE on the second GPU.
2. **~525ms/step → ~450ms** — CUDA-graph capture of `ForwardPatched` (scaffolding proven; arena in-tree default-off
   provides deterministic addresses), fuse remaining per-step ops.
3. **VAE decode 271ms → ~100ms** — F16 conv + fold the 6.9 GB im2col peak (would also let TE stay resident → kills #1).
4. Replicate the VAE-host-glue fix fleet-wide: any model using `QwenImageVaeDecoder`/host-loop VAE norms
   (Qwen-Image, Anima, …) inherits the same win; re-bench SDXL/Flux/Chroma VAEs for the same pathology.

## 2026-07-07 (later) — sub-4s push: 5.83s → **4.68s** — engine `alpha.43.143-local`

Three more shipped, all coherent (astronaut verified after each):

| Change | Δ | Notes |
|---|---|---|
| **Prompt-embedding cache** (`Krea2Pipeline`: tapped TE hiddens keyed on token ids + drop index; reusing the same tensor reference keeps the transformer's txt-fusion cache warm too) | **−1.1s** → 4.72s | Repeat-prompt gens skip the whole TE phase (preload+encode+free = 0ms), matching Comfy's conditioning cache |
| Device rgb conversion (`ChwF32ToHwcU8` kernel + IBackend op: CHW F32 → HWC u8 on-GPU, 3 MB D2H instead of 12 MB + host loop) | ~0 | The 142ms "rgb" phase was mostly absorbing the VAE's async tail — cleaner, not faster |
| **CUDA-graph step capture** (`HARTSY_DIT_GRAPH=1`): `ForwardCore` (img_in→28 blocks→final layer) captured once, replayed per step via one `cuGraphLaunch`; per-step-varying inputs (temb/tembMod/latent) refreshed into FIXED buffers (`CopyInto`, in-place `CfgEulerStep` Euler); velocity lands in a pre-capture normal buffer via a captured CopyInto; self-disables on failure | ~0 wall (**4.68s**) | Step ISSUE went ~4.2s → **6ms** (host fully free during the loop) — but the GPU is genuinely busy ~550ms/step, so wall is unchanged. Capture is correct + banked. |

**Two capture-blockers found + fixed (general lessons):** (1) the host-materialized txt cache re-uploaded from
PAGEABLE memory every step (auto-promotion blocked by the VRAM headroom gate at 99% full) — an internally-syncing
copy that invalidates capture; fix = explicit `PreloadWeights([txt])` pin in graph mode. (2) **`Concat` used
synchronous `cuMemcpyDtoD`** — capture-illegal AND a hidden per-step host serialization; fix = `cuMemcpyDtoDAsync`
on the compute stream (better for everyone, not just capture).

**Standing: Hartsy 4.68s vs Comfy 6.5s = 1.39× faster.** Sub-4s needs ~0.7s more and the profile is now purely
GPU-compute: 8×~550ms steps + VAE 275ms + rgb/Swarm tail. The honest menu (each a real kernel-engineering session):
fused QKV+gate GEMM (blocked on per-tensor fp8 scale handling — requantize to a common scale at load), VAE F16 convs
(halves its im2col + tensor-core rate), cuBLASLt algo tuning for the M=4608 shapes, VAE→second-GPU overlap.

## Persistent activation arena (`HARTSY_ACT_ARENA`) — engine `alpha.43.132-local` — **NEGATIVE RESULT**

Implemented a size-keyed in-process free-list at the `CudaMemory.AllocateAsync`/`FreeAsync` choke point: recycle freed
device blocks instead of `cuMemAllocAsync`/`cuMemFreeAsync` per op (bit-identical, zero op/block/Tensor changes,
`HARTSY_ARENA_MAX_MB` cap, OOM-drains idle blocks). Hypothesis: the ~18s residual was stream-ordered alloc/free driver
churn. **Disproven.**

| | Comfy | Baseline (43.131) | Arena on (43.132) | Δ |
|---|---|---|---|---|
| Krea2-Turbo warm | 6.5s | 27.7s | **28.9s** | ~0 (noise) | 
| peak VRAM | — | 17.9 GB | **21.4 GB** | +3.5 GB |

Image coherent (astronaut-on-horse, clean) → the free-list reuse is **safe** (stream-ordering guarantee holds; recycling
never corrupted a live buffer). But **wall did not move**, and VRAM rose. Interpretation: `HARTSY_MEMPOOL_KEEP` (shipped
in the 27.7s baseline) already made `cuMemAllocAsync` a cheap warm-pool pop — the arena replaced a cheap pool-pop with a
cheap dict-pop, and its exact-size buckets are *less* memory-efficient than the driver pool's coalescing (hence +3.5 GB).
**Allocation is NOT the bottleneck.**

### Reconciled diagnosis — it is memory-bandwidth / elementwise-kernel bound, not allocation
Three independent eliminations now converge: `HARTSY_FP8_NATIVE` → 0% (not GEMM-compute-bound); host op-issue ≈2.2s
(not host-dispatch-bound); **arena → 0% (not allocation-bound)**; SplitModulation −2.5k ops → 0.5s (not op-count-bound).
By elimination the ~18s residual is **GPU execution of the many bandwidth-heavy F32 elementwise/norm/attention activation
kernels** (RmsNorm, GatedResidual, Modulate, Concat, Permute, Silu, Mul, Add over `[1,~4600,3072]` F32 = ~56 MB each),
which the NvtxRange profile under-counts (many of these ops have no range). This is exactly what **F16 activations**
halve (bytes read/written per kernel) and what **fused elementwise kernels** reduce (fewer passes over the data).
**Next = F16 activation path** (keep fp8 weights packed via activation-quant GEMM — see plan) ± fused norm/AdaLN kernels.
The arena stays in-tree, default-off: harmless, and it is the right substrate for later CUDA-graph capture (deterministic
addresses, zero mid-step `cuMemAllocAsync`).

## 2026-07-07 — fleet rerun on `alpha.43.144` + Z-Image round 1 (`alpha.43.145`)

**Fleet inheritance from the Krea2 work** (shared VAE port, async Concat, cuDNN SDPA; env
`HARTSY_SDPA_CUDNN/MEMPOOL_KEEP/FP8_NATIVE/DIT_F16/DIT_GRAPH`, no KEEP_MODELS):

| Model | Comfy | 07-05 | now | note |
|---|---|---|---|---|
| SDXL | 3.7 | 46.3 | **36.9** | free |
| Flux-Dev | 12.5 | 74.8 | **72.9** | ~flat |
| Z-Image-Turbo | 3.1 | 52.1 | 40.5 → **6.6** | round 1 below |
| AuraFlow | 14.0 | 34.5 | **31.4** | free |
| Qwen-Image Q4_K | 54.8 | 380.7 | **192** | shared Qwen-VAE port (~2×) |
| Chroma1-HD | 16.6 | 119 | **110** | modest |
| Krea2-Turbo | 6.5 | 4.68 (KEEP) | **6.56 no-KEEP** | see crash fix below |
| ERNIE / HiDream / Ideogram4 | — | 53.3 / fail / 42.3 | invalid | collateral of the crash below — rerun |

**Step-graph weight-eviction crash (FOUND + FIXED, 43.145):** in the fleet env (no KEEP_MODELS) Krea2's
post-loop `FreeWeights` freed the DiT weights whose device pointers the captured step graph had BAKED; the
next gen's replay hit CUDA 700 ILLEGAL_ADDRESS and **poisoned the context** (the ERNIE/HiDream/Ideogram4
"load failures" were collateral). Fix: `Krea2Transformer.InvalidateStepGraph` called whenever transformer
weights are freed — proven by 3 clean no-KEEP warm gens. Rule: **a captured graph dies with any weight
eviction.**

### Z-Image-Turbo round 1: 40.5s → **6.6s** (vs Comfy 3.1s; was 16.8× → now 2.1×) — coherent ✓
Recon killed two stale beliefs (Z-Image HAS QK-norm; blocks already GPU-resident) and found the real costs:
| Fix | What |
|---|---|
| **GPU RoPE** (`ZImageRope.ApplyGpu` → `WanRopeInterleaved`, pre-permute, bit-identical) | was a ~63 MB × 2 host D2H/H2D round-trip per block × 34 blocks × 8 steps — the dominant cost |
| **Device AdaLN** (`ModulationSplit4` — its (1+scale, tanh(gate)) convention matches Z-Image exactly; block applies scales via direct AffineBroadcast) | 34 D2H sync barriers/step gone |
| `allowF16: true` on both SDPAs (QK-normed → bounded) | engages cuDNN flash attention |
| Caption-path cache (cap_embedder + 2 context refiners, timestep-independent, keyed on encoder-output ref) + rope-table caching (3 host rebuilds/step → sig-cached) | −14 block-forwards +3 table builds per gen |
| Device concat + single-SliceRows output slice (B=1) | 2 more per-step drains gone |
| `FreeActivations(trimPool: false)` per step | stops a stream-sync + multi-GB pool release/re-map every step |

**Round 2 menu (→ sub-3.1s):** the pipeline still host-syncs every step by design (host scheduler.Step +
NegateInPlace + latent D2H) — build a `ForwardPatched`-style drain-free fast path (persistent latent tokens,
on-device negate+Euler via CfgEulerStep with delta=−dt), then `DitDtype` F16 (audit ffn=10240 SwiGLU) +
`DitStepGraph` capture + KEEP_MODELS/prompt-cache/device-rgb pipeline parity with Krea2. Then Ideogram4.

### Z-Image-Turbo round 2 (`alpha.43.146-local`): 6.6s → **5.15s** — coherent ✓, Krea2 regression clean (4.68s)
Drain-free packed fast path (the Krea2 `ForwardPatched` pattern): `ZImageTransformer.ForwardPacked` keeps the
latent in `[1, imgRealLen, C·p²]` token space across the whole loop (patchify/unpatchify once); the Euler update
is one in-place device op — `CfgEulerStep(packed, v, v, 1, −dt)` folds diffusers' `noise_pred = -noise_pred`
negation into the sign (NegateInPlace + host scheduler.Step + per-step latent D2H + per-step FreeActivations all
gone). Also: final layer fully device (host LayerNormNoAffine → backend op; host (1+scale) loop → Modulate),
device rgb (`ChwF32ToHwcU8`), per-gen latent-stats scans gated behind `HARTSY_ZIMAGE_STATS`.

**Standing: Z-Image-Turbo 5.15s vs Comfy 3.1s (1.66×; was 16.8×).** Remaining menu to beat 3.1s:
1. **TE encode cost** (Qwen3-4B runs per gen in the extension loader — needs a prompt-embedding cache there,
   extension-side; attribution first via the `[zimage-phase]` Verbose probes, run Swarm at verbose log level).
2. **`DitStepGraph` capture** — the loop is now drain-free + fixed-shape; copy the Krea2Transformer pattern
   (fixed latent/tEmb/velocity buffers, sig = caption ref ⊕ shape, weight-eviction invalidation).
3. **`DitDtype` F16 activations** — REAL SwiGLU overflow history here (ffn=10240, L0 ffnOut INF): audit with FFN
   inner kept F32 first.

## 2026-07-08 — Z-Image-Turbo round 3 (`alpha.44.2-local`): 5.15s → **3.12s = TIED with Comfy (3.1s)** — coherent ✓

Warm median 3.12s (best 3.10s), every stage verified visually against the astronaut reference. From 52.1s
(16.8× slower) three days ago to 1.00× — the same arc that took Krea2 past Comfy.

**Attribution first** (phase probes at Verbose): the 5.15s was TE 0.43s / denoise 2.75s (F32 ~344ms/step) /
unpatchify+drain 0.20s / **VAE decode 1.49s** / tails ~0.3s. The round-2 note "VAE is already GPU — no work
there" was wrong in a different way: the shared Flux `VaeDecoder` at 1024² ran **9 overlapping tiles** (2.25×
redundant decode + host blend/normalize loops + per-tile D2H) — and unconditional `[TILEDBG]` full-tensor host
scans on every gen.

| Fix | Δ / result |
|---|---|
| **VAE full-res direct decode** (`VaeDecoder.DecodeTiled`: when the estimated worst im2col fits a 10 GB cap, decode the whole latent in one pass; OOM → fall back to tiles for the session; TILEDBG scans gated behind `HARTSY_VAE_STATS`) | 1490ms → 632ms; SDXL/Flux/Chroma/AuraFlow inherit free (SDXL 46.3 → 37.3s, coherent ✓) |
| **BF16 VAE weights for Z-Image** (extension `ZImageLoader` via the existing `VaePrecisionHelper` — the SDXL policy) + **planned pool-trim** before the big im2col (its 9.2 GB alloc OOM-retried against MEMPOOL_KEEP reservations EVERY gen, ~+470ms) | decode → **416–474ms stable**, im2col 9.2 → 4.6 GB, zero OOM retries |
| **TE prompt-embedding cache** (extension-side, keyed on prompt string; TE weights preloaded+freed only on cache miss — the freed ~8 GB is the VAE headroom) | repeat-prompt TE ~430ms → **0** |
| **`DitStepGraph` for Z-Image** (`ZImageTransformer.ForwardPacked` → capturable `PackedCore`; per-GEN lifecycle — the pipeline's final `FreeActivations` frees the fixed boundary buffers, so cross-gen replay = CUDA 700 (hit live); invalidate at gen end, re-warm+re-capture next gen) | correct + banked; wall ~neutral (GPU-bound) |
| **`HARTSY_DIT_F16` for Z-Image blocks** — see the sandwich-damp story below | steps 344 → **~256ms** (denoise 2.75 → 2.05s) |
| **Device unpatchify** (`dit_unpatchify_f32` kernel + `IBackend.UnpatchifyTokens`, parameterized for Z-Image (ph,pw,c) AND Krea2 (c,ph,pw) inner orders) — end-of-loop tokens → VAE never leave the GPU | unpatchify+drain 280ms → ~0 host (wall −100ms; the rest was GPU tail that now overlaps the VAE phase) |

### The F16 story: sandwich-norm scale damping (new general technique)
Z-Image's attention out-projection and SwiGLU raw outputs genuinely exceed F16's 65504 (traced live: attnProjected
INF in the FIRST refiner block; raw ffnOut ~1.05M at step 6 — late steps grow ~4× past step 1). But both feed
straight into an RmsNorm, and **RMSNorm(c·x) ≡ RMSNorm(x)** — so scaling `attention.out` and `feed_forward.w3`
by **1/64** (folded into the GEMM alpha via `Fp8ScaleFactor`, zero extra kernels, bit-exact post-norm, zero
relative-precision cost) makes the whole block-loop F16-safe with NO F32 detours. 1/16 left exactly ONE element
at INF; the magnitudes were measured with the env-gated `HARTSY_ZIMAGE_F16TRACE` per-tensor probes (kept in-tree).

### Cross-model step-graph owner guard (latent bug found by inspection, fixed + verified live)
The backend step-graph slot is single — with two graph-capturing models alternating (Krea2 ↔ Z-Image under
KEEP_MODELS), model A would happily `StepGraphLaunch` model B's captured graph (A's own signature never changed).
Added `IBackend.StepGraphOwner`; owner mismatch → reset + re-capture (never counted as a CFG flip). Verified live
by the no-KEEP rotation bench below.

### Deploy gotcha (cost one debugging round): a `1.0.0-alpha.44` pin resolves to NUGET.ORG
The extension pin was bumped to `1.0.0-alpha.44` (no `-local`, no matching local pack) — NuGet silently restored
the months-old PUBLIC `HartsyInference 1.0.0-alpha.44` from nuget.org: 59s gens, no phase probes, no error.
Local packs must sort ABOVE any published version (hence `alpha.44.2-local`) and the deployed DLL should be
md5-checked against the local nupkg after every extension rebuild.

### Final numbers (RTX 4090, warm median of 3+, random seeds, all coherent ✓ visually)
| Model | Comfy | round 2 | **round 3** | config |
|---|---|---|---|---|
| **Z-Image-Turbo** | 3.1s | 5.15s | **3.12s (1.01×)** | KEEP_MODELS; 3.13–3.18s no-KEEP |
| Krea2-Turbo (regression) | 6.5s | 4.68s | **4.67s** ✓ | KEEP_MODELS; 6.47s no-KEEP (bar: 6.56) |
| SDXL (free inheritance) | 3.7s | 46.3s | **37.3s** | full-res VAE only; rest unported |

Warm 3.12s profile: TE 0 (cache) · denoise 8×~256ms = 2.05s · unpatchify ~0 · VAE 430ms · rgb 5ms · Swarm tail ~0.4s.

**Remaining to BEAT 3.1s cleanly (~0.2s, all GPU-compute):** fused QKV GEMM (easy for Z-Image — the fused
checkpoint tensor shares ONE fp8 scale, unlike Krea2), w1/w3 FFN fusion (mind the w3 damp), cuBLASLt algo tuning
for M=4160, VAE mid-block attention F16/cuDNN, cross-gen graph persistence (needs the fixed buffers out of the
activation cache). Fleet: port the prompt-embed cache + device rgb + full-res-VAE audit to Flux/Chroma/ERNIE.

## 2026-07-08 — Ideogram4 grind: 39.3s → **19.5s** (Comfy 17.0s; gap 2.31× → 1.15×) — engine `alpha.44.7-local`

Warm median 19.51s @ 1024²/20 steps (V4_DEFAULT_20, structured-JSON prompt), coherent + prompt-faithful,
verified visually every round. Peak 23.7 GB (KEEP_MODELS: BOTH 9.3B DiTs resident). Fresh baseline this morning
was 39.33s (the 07-05 table's 42.3s predates the fleet fixes).

**Attribution first** (probes): TE 1.7s / denoise 32.2s (20×1.61s) / VAE 0.7s / DiT re-preload 4.6s.

| Round | Fix | Result |
|---|---|---|
| 1 | **Step-invariant conditioning restructure**: the [1,L,53248] llmFull (933 MB) + zero negLlm (872 MB) were re-uploaded + re-projected (RmsNorm-53248 + 53248→4608 GEMM) EVERY STEP. Text projection now computed once per prompt over the ~286 text rows only, `[proj | Fill+Concat zeros]`, pinned; uncond text path skipped outright (provably zero); MRoPE cos/sin cached+pinned per gen; image path `Linear(z)+ScatterRowsAfter`; final layer on image rows only. All bit-identical. + TE prompt-embedding cache + posIds cache + `HARTSY_KEEP_MODELS` (evicts DiTs on prompt MISS — TE 8 GB can't coexist) | 39.33 → **25.35s** (steps 1.61 → 0.97s) |
| 2 | **cuDNN fused flash attention at head_dim=256** — cuDNN 9.24 supports D=256 fwd on Ada; was gated to D∈{64,128} and the F32 fallback materialized 1.4–2.3 GB score matrices at the VRAM ceiling (OOM-retry thrash). Added per-D failure isolation (a D-reject no longer kills the engine for other models' D=128) | 25.35 → **24.84s** |
| 3 | **HARTSY_DIT_F16 for Ideogram4Block** (new `dit_rope_f16` rotate-half + `dit_slice_lastdim_f16` kernels; o/w3 damped 1/64 via Fp8ScaleFactor — sandwich RMSNorms cancel exactly, the Z-Image recipe) + **drain-free loop** (removed per-step `z.DataPointer` + FreeActivations = measured 325 ms/step of hidden pipeline drain; kept only for masked inpaint) | 24.84 → **20.45s** (step wall 1156 → ~935ms) |
| 4-5 | **Banded im2col Conv2D** (new `im2col_banded` kernel + band loop: caps any conv's im2col at 1 GB — bit-identical, ldc trick, parity-tested vs CPU): the Flux.2 VAE's 9.2 GB full-res estimate → 1 GB, so **full-res decode engages beside 18.6 GB resident DiTs** (was 3×3 tiles = 2.25× redundant + a VISIBLE SKY SEAM). + adaptive full-res gate (post-trim headroom check; skip-not-disable). + cuBLASLt fp8 workspace 4→64 MB | 20.45 → **19.51s**, VAE 1.48 → 0.64s, seam GONE |

**GPU-bound confirmed** (util median 100% during the loop; step-graph capture would be wall-neutral, skipped).
Step ≈ 935ms = fp8 GEMMs (~158 TFLOP/step at ~250-300 TFLOPS eff) + fused SDPA + F16 glue.
Profile split: attn ~506ms / mlp ~389ms (blocking-profile inflated).

**Remaining menu to beat 17.0s (~2.5s, all GPU-compute):** BSHD strided cuDNN SDPA (skip 4 Permute0213/block
= 272 kernels/step), w1/w3 fusion (needs common-scale fp8 requant at load), cuBLASLt heuristic-cached algo
selection for the M≈8478 fp8 shapes, fused norm+modulate kernels, VAE decode on the second GPU.

**Fleet regression (same session, shared-code changes: banded conv, cuDNN per-D isolation, Lt workspace 64 MB):**
| Model | bar | KEEP config | no-KEEP | note |
|---|---|---|---|---|
| Krea2-Turbo | <6.5s | **4.44s** ✓ (was 4.68) | (below) | coherent ✓ visually |
| Z-Image-Turbo | ≤3.2s | **2.94s** ✓ (was 3.12) | (below) | coherent ✓ visually |
| SDXL (banded-VAE sanity) | — | 36.3s (was 37.3) | — | coherent ✓ visually |

## 2026-07-08 — Qwen-Image round 1: 355s → **40.9s = BEATS ComfyUI (54.8s, 1.34× faster)** — engine `alpha.44.8-local`

Warm median 40.9s @ 1024²/20 steps/cfg 2.5 (dual-pass CFG, Q4_K_M GGUF), coherent + prompt-faithful, visually
verified (fresh same-morning baseline was ~355s at 14–24 s/step, GPU util 36%; the 07-07 table's 192s predates
this env). Steps now ~2.05s (both CFG passes). Qwen-Image-Edit inherits (same pipeline/blocks).

One deploy, five fixes — all the Ideogram/Krea2/Z-Image playbook:
| Fix | What |
|---|---|
| **Device joint RoPE** | `QwenImageRope.ApplyJoint` was a HOST loop reading jointQ/K `DataPointer` — a 2×~50 MB D2H+H2D round trip per block × 60 blocks × 2 passes × 20 steps. Now: cached `[S, headDim]` cos/sin tables (`GetOrBuildJointTables`, one host build per layout) + `WanRopeInterleaved` on the pre-permute layout. Proven equal to the host path by `DeviceRopeTables_Equal_HostApplyJoint` (CPU, 1e-6). Host path kept for batch>1 + as the tests' reference. |
| **cuDNN flash SDPA** | `allowF16: true` on the joint attention (head_dim 128, QK-normed, mask-null — the proven config). Was F32 materialized scores. |
| **Drain-free loop** | CFG combine + Euler fused into ONE in-place device op (`CfgEulerStep(z, cond, uncond, cfgScale, dt)` ≡ `uncond + cfg·(cond−uncond)` Euler); latent device-resident all loop; per-step `FreeActivations` gone (kept on the masked-inpaint/edit-ref host paths). |
| **TE prompt cache** | cond + uncond Qwen2.5-VL hidden states keyed on (tokens, dropIndex); repeat prompts skip the whole TE phase. |
| **KEEP_MODELS** | DiT stays resident across gens; prompt-cache MISS evicts it for the TE (Ideogram pattern). |

**Remaining lever (documented, not yet built): Q4_K → fp8-e4m3 requant at load.** The Q4_K GEMM path still
dequantizes each weight to a TRANSIENT F16 buffer per Linear call (the F16 expansion of a 20B model can't be
cached on 24 GB — the Axis-B pathology, GGUF flavor). Requanting to fp8 at load (dequant via `GgufDequantizer`
→ global amax/448 scale on `Fp8ScaleFactor` → `CastTo(F8E4M3)`, mirroring `DequantNvfp4ToFp8` which shipped
coherent for Ideogram from the same 4-bit information budget) puts Qwen on the packed-fp8 native GEMM path:
est. ~2.05 → ~1.2–1.4 s/step (→ ~30s/gen), at ~20 GB resident + a requant-quality visual gate.

**Regression on `alpha.44.8` (includes the other session's Hunyuan/Kandinsky video SDPA flips):** Z-Image-Turbo
**2.86s** ✓ (bar 3.2), Krea2-Turbo **4.47s** ✓ (bar 6.5), images pristine. (One bench pass showed 74–136s
outliers that snapped back to 2.98/4.47 on final reps — transient GPU contention from the concurrent video
session's verification runs, confirmed clean on re-run with an intruder watchdog.)

## 2026-07-08 — Chroma round 1: 110s → **97.2s** (Comfy 16.6s) — engine `alpha.44.9-local` — + fleet recipe status

Chroma got the full pipeline recipe (T5 prompt cache, KEEP_MODELS w/ evict-on-miss, drain-free CfgEulerStep
loop, previews throttled to every 4th step) PLUS the device modulation-table port: `BuildDoubleBlockTemb` /
`SliceModSlab` / per-block `SliceModRows` / `ApplyContinuousNorm` all read the device-produced mod table via
host `DataPointer` (the Kandinsky temb stream-stall, ×57 blocks/forward) — now `SliceRows`/`Concat`/
`LayerNormNoAffine`/`AffineBroadcastLastDim` device ops (B=1; host fallback kept). Coherent ✓ visually (clean
modulation, no scrambling). Only −12% because **Chroma is GPU-bound on its masked SDPA**: the Chroma text mask
disqualifies the cuDNN fused path (mask-null-only), so attention runs F32-materialized/tiled. **Chroma's real
levers (next round): cuDNN SDPA bias/mask support (also unblocks the video fleet's masked models), F16
activations, fp8-native GEMM audit.**

ERNIE haze verdict: same-seed A/B with banded-conv + full-res VAE disabled = IDENTICAL image → today's shared
changes innocent; the flat/grainy look is ERNIE's standing output (quality audit queued for its grind round —
std-ratio vs ComfyUI per the GroupNormSilu washout history). Ideogram's "haze" is the JSON prompt's explicit
"faint atmospheric haze" — not a defect.

Fresh 44.8 baselines banked for the remaining fleet: SDXL 32.97s (Comfy 3.7), Flux-Dev 72.4s (12.5) — Flux also
has two UNCONDITIONAL full-tensor host stat scans per step (`LogTensorStats` + `LogPerLatentChannelMeanPacked`)
to gate, plus the standard recipe. AuraFlow re-download pending (see incident below). Qwen round-1 recipe is the
template for all.

**Incident (2026-07-08 ~10:24):** a disk-full cleanup deleted `/tmp/*_dl` dirs that turned out to be LIVE
symlinked storage for ~10 checkpoints (AuraFlow/Chroma-fp8/Kontext/OmniGen2/HiDream/Kandinsky-T2I/Boogu-TE +
Krea2/Ideogram TEs). Recovered same-session: TEs re-downloaded (Krea2 verified healed with a coherent gen),
Chroma fp8 re-downloaded (`silveroxides/Chroma1-HD-fp8-scaled`). Still to re-fetch: AuraFlow fp8, Kontext,
OmniGen2, Kandinsky-T2I, HiDream, Boogu flux1 VAE. Memory: `models-in-tmp-symlink-trap`.

## 2026-07-08 — Chroma round 2: 97.2s → **63.2s** (Comfy 16.6s) — engine `alpha.44.10-local` — cuDNN SDPA bias/mask support

The round-1 wall is gone: **the cuDNN fused flash-attention engine now takes an additive mask**, so Chroma's
masked attention (the text-padding `[B,1,S,S]` mask that used to disqualify the fused path → F32 materialized
2 GB score matrix) runs fused. Graph = the cudnn-frontend Bias score-modifier pattern:
`bmm1 → scale(MUL) → bias(ADD, fp32 [B,1,Sq,Skv] broadcast over heads) → unified-SOFTMAX → bmm2` — still hits
the fused engine (workspace 0), proven first in the surviving Python ctypes proto (relL2 2.7e-4 vs numpy with
the exact `-1e30` convention; **2.20 ms/call at Chroma shape** B=1,H=24,S=4608,D=128 vs 62.7 ms materialized).
C#: `CudnnSdpa.Execute(..., biasF32, biasB)` (plan-cache keyed on bias presence), `CudaBackend` gate now admits
F32 `[B,1,Sq,Skv]`/`[1,1,Sq,Skv]` masks on both the F16-native and F32-cast cuDNN routes; Chroma's two block
SDPA call sites pass `allowF16:true` (safe: QK RMS-norm bounds scores; the mask is added to fp32 scores INSIDE
the engine, never rounded through F16). `CudnnSdpaTests` gained masked D∈{64,128} cases (all pass, engaged).

**Chroma1-HD @1024²/20 steps: warm 62.99/63.22/63.45s (median 63.2s, was 97.2), peak VRAM 15.3 GB (score
matrix gone), coherent ✓ visually (sharp astronaut, clean masks, no veil).** Gap vs Comfy 5.9× → 3.8×.
Flagship regression: Krea2-Turbo **4.50s** (<6.5 gate ✓), Z-Image-Turbo **2.95s** (≤3.2 gate ✓), both verified
coherent visually.

**Blast radius (intended):** WanVideoBlock / LtxVideoBlock / LtxVideo2Attention / ZImageBlock already pass
mask+`allowF16:true` — with `HARTSY_SDPA_CUDNN=1` their F32 broadcast masks now ride the fused engine too
(per-head `[B,H,Sq,Skv]` masks and no-allowF16 callers — T5/CLIP/Llama encoders — unchanged). Z-Image is
regression-verified above; **the masked video models should be re-benched/verified next video session** (free
speedup expected, e.g. Wan cross-attn).

**Deploy gotcha (cost one OOM'd bench):** the standard perf flags (`HARTSY_SDPA_CUDNN/FP8_NATIVE/DIT_F16/
KEEP_MODELS/MEMPOOL_KEEP`) live ONLY in the Swarm launcher's env — a relaunch without exporting them silently
reverts the engine to defaults (Chroma then OOM'd allocating the materialized 2 GB score buffer it no longer
needs under cuDNN). Verify `/proc/<swarm-pid>/environ` after every relaunch; deploy memory updated.

**Chroma next levers (round 3):** profile first (wall is no longer SDPA) — expected split: fp8 Linear/transient
dequant (Axis B — the Qwen Q4_K→fp8-requant analog), remaining host glue, VAE. Then F16 activations
(`HARTSY_DIT_F16` opt-in for the Chroma blocks) and the per-gen step graph (Z-Image recipe).

**Addendum — `alpha.44.11-local`, deployment self-sufficiency (user directive: persistence belongs in the
extension logic, not launcher env).** Two changes make a BARE Swarm launch fully functional: (1) the extension's
`OnPreInit` sets the 5 standard perf flags as unset-only in-process defaults (env value, incl. "0", still wins —
kill-switches intact; logged at init); (2) engine `CudaLibraryResolver` probes `$HARTSY_CUDA_LIB_DIR` then
`~/.local/lib/cuda13` for cublas/cublasLt/cudnn before soname search, and `CudaContext.IsAvailable` shares that
probe (its old duplicated bare-soname check silently SKIPPED GPU tests as "passed" without LD_LIBRARY_PATH).
Verified end-to-end with a zero-env launch: flags logged, backend live, cuDNN engaged, coherent Chroma gen.
44.11 = 44.10 + resolver only (no math changes) — the 44.10 bench numbers stand.

**Addendum — `alpha.44.12-local`: the standard performance profile is now the ENGINE DEFAULT (user
directive: every downloaded install must reproduce published times with zero configuration, documented
professionally).** `HartsyInference.Core.Runtime.EnvSwitch` gives the 5 profile features tri-state
semantics (unset → default ON, `0`/`false` = kill-switch): `HARTSY_SDPA_CUDNN` (self-disabling fallback),
`HARTSY_FP8_NATIVE` (hardware-gated: default ON only on SM ≥ 8.9), `HARTSY_DIT_F16` (per-arch code opt-in
unchanged), `HARTSY_KEEP_MODELS` (4 pipelines), `HARTSY_MEMPOOL_KEEP`. Experimental switches keep strict
opt-in. Extension OnPreInit no longer sets anything — single source of truth is the engine, so NuGet/CLI
consumers get identical behavior. Documented in the new **`docs/PERFORMANCE.md`** (profile table, native
library requirements + resolver order, verification log lines, benchmark methodology + scoreboard) with
README + extension README/doc-07 sections pointing at it. Verified on a zero-env launch: engine log
`[Cuda] perf flags: SdpaCudnn=True NativeFp8Gemm=True MempoolKeep=True ...`, warm medians unchanged —
Z-Image-Turbo 3.05s, Chroma1-HD 63.5s, both visually coherent.
