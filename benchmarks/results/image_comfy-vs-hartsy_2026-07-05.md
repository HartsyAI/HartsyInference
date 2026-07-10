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

## 2026-07-08 — Flux family round 1: Flux-Dev 72.4s → **31.0s** (Comfy 12.5s), Flux-Schnell first bench **10.5s** — engine `alpha.44.14-local`

One shared FluxPipeline serves Dev / Schnell / Kontext / Tools, so one recipe round covered the family:

1. **THE dominant fix — streaming bypass.** FluxPipeline unconditionally used the block-streaming sliding
   window (`retainBehind: 0`) whenever the backend had a streaming cache — evicting and re-uploading ALL
   ~12 GB of fp8 blocks EVERY forward (~240 GB of PCIe per 20-step gen) on a card that fits the whole
   model (Krea2, same weight class, always ran resident). Now: if the full block set fits beside the
   activation reserve (same estimate the prefetch chooser uses), eager-preload and skip streaming;
   low-VRAM cards stream exactly as before. Follow-up fix: a resident DiT (KEEP_MODELS) must SKIP the
   availability query — free VRAM no longer counts the weights' own footprint, so warm gens alternated
   resident→streaming→resident (55/37/55s pattern) until short-circuited.
2. **Drain-free device loop** (Chroma/Krea2 pattern): host `SchedulerStepPacked` (per-step velocity D2H +
   latent re-upload) → in-place `CfgEulerStep` on the device-resident latent, incl. true-CFG. Per-step
   `FreeActivations` skipped on this path (it would free the live device latent — the Wan I2V bug class).
   Tools/Kontext (host per-step concats), regional bias, masked inpaint keep the host branch, unchanged.
3. **Per-step host stat scans gated** (`HARTSY_FLUX_STATS=1` opt-in): three unconditional full-tensor host
   reads per step (LogTensorStats ×2 + per-channel means) were forced D2H syncs serializing the loop.
4. **TE prompt cache** (CLIP-pooled + T5, cond + neg, keyed on token ids + EOS position); miss evicts a
   resident DiT first. **KEEP_MODELS residency** on the eager path; previews throttled to every 4th step.
5. **allowF16 on both Flux blocks** (QK RMS-norm) → cuDNN fused flash attention engages fleet-wide for the
   family; regional attention bias rides the fused engine's fp32 mask path.

**Flux-Dev @1024²/20 steps: warm 28.3/31.0/31.0s (median 31.0, was 72.4; contention-checked re-run), peak
17.8 GB stable, coherent ✓ (sharp sunset astronaut). Gap vs Comfy 5.8× → 2.5×.**
**Flux-Schnell @1024²/4 steps: warm 10.5s (first recorded number; fp8 checkpoint downloaded; added to
models.json), coherent ✓.** Kontext checkpoint verified present; its t2i path (no ref) IS the Dev path so
it inherits everything — ref-image mode keeps the pre-round host path (round 2: device Concat).
Flagship regression: Krea2 4.52s ✓, Z-Image 2.99s ✓, both coherent visually.

**Deploy note:** 44.13 collided with a concurrent agent deploy that packed sub-packages but no meta →
NU1603 silently resolved the 44.7 meta (stale engine, old flag format in the log — the documented
detection signature). Re-synthesized meta + repacked as 44.14 (never overwrite a restored version).

**Flux round 2 levers:** DIT_F16 opt-in for the Flux blocks (Krea2Block conversion pattern), profile the
remaining 1.4s/step (fp8 transient dequant / Lt tuning / fused QKV), device Concat for Kontext ref +
Tools control (unlocks drain-free there), then the step CUDA graph. **Flux.2 is a separate round:** its
transformer still runs host-side block glue (DataPointer modulation loops) — needs the Chroma-style
GPU-residency port BEFORE any recipe work.

## 2026-07-08 — Flux.2 round 1: GPU-residency port + recipe — Klein 4B first numbers: cold 46s, **warm 15.1s** @1024²/10 steps — engine `alpha.44.15-local`

Flux.2 never had the residency port (audit: Flux2DoubleBlock cpu=33, Flux2SingleBlock cpu=29 host glue
sites). Ported both blocks + transformer glue to backend ops (the verified Flux.1/Chroma idioms: NormModulate,
RmsNorm QK-norm, pre-permute Concat + device RoPE, Permute0213, SliceRows split, GatedResidualLastDim,
Silu+Mul SwiGLU, device AdaLN-Continuous final layer) — blocks now have ZERO host reads. Pipeline gained
weight staging (TE ⇄ DiT swap — it previously had NONE, which is why 32B Dev was marked "blocked": ~12 GB
Mistral TE + ~18 GB Q4 DiT can't coexist on 24 GB), TE prompt cache, KEEP_MODELS residency, and the
drain-free CfgEulerStep loop (g=1; guidance is embedded). allowF16 on both blocks (QK RMS-norm) → cuDNN
fused attention. Klein 4B verified via Swarm: coherent, sharp (viewed), warm 15.1/15.1s.

**Dev-GGUF blocker (next round):** `Flux2Loader` has no GGUF branch (feeds .gguf to SafeTensorsLoader →
JsonReaderException), and `Flux2CheckpointConverter`'s fused-weight splits do per-element row math that a
Q4_K block format breaks — needs the QwenImageLoader GGUF-bridge pattern + quant-block-aligned slicing.
Checkpoints staged as REAL files: flux2-dev-Q4_K_S.gguf (19.3 GB), Mistral TE (12.3 GB), Klein 4B (7.75 GB).
NOTE: hf_hub_download local_dir SYMLINKS into ~/.cache/huggingface — always cp --dereference + purge the
cache repo, or the /tmp-symlink incident repeats and disk double-counts.

## 2026-07-08 — ERNIE-Image round 1: 49.6s → **~20.3s pipeline (BEATS Comfy 24.0s; clean wall median pending)** — engine `alpha.44.19-local`

Fresh same-evening baseline on 44.17: cold 64.75s, warm **49.62/49.69/49.34s (median 49.62s)**, peak 13.8 GB.
After round 1: pipeline-logged warm gens **20.3/23.8/20.3s** (TE-cache-hit gens = 20.3s; the 23.8s rep paid a
TE re-encode). Contended wall 27.1s — the concurrent video session's Wan S2V work + two surprise Swarm restarts
poisoned the wall clock; the official warm-median wall bench is queued behind a sustained-GPU-quiet gate and
this section will be finalized from it. **Correctness gate passed: seed-777 A/B pre/post deploy — identical
composition/contrast, no haze/washout, only F16-attention-class micro-deltas. Coherent ✓ viewed.**

One deploy, five fixes — all in ERNIE-only files (zero shared-code blast radius), the Qwen/Ideogram playbook:
| Fix | What |
|---|---|
| **Step-invariant mask cache — THE find** | `ErnieImageTransformer.BuildAttentionMask` host-built + H2D-uploaded a **77 MB `[1,1,S,S]` F32 padding mask EVERY forward** (2 CFG passes × 20 steps = 40/gen — the Chroma mask pathology, bigger). Now cached per (nImg, textMax, textLen); RoPE cos/sin tables cached the same way (and pre-sliced ONCE per gen instead of 2 slices × 36 blocks per forward); text projection cached on the textEmbeds reference (the pipeline prompt cache reuses refs, keeping it warm across gens). Cache tensors are host-materialized at store time so FreeActivations can't revert them. |
| **Masked cuDNN flash SDPA** | The block SDPA never passed `allowF16` → F32 materialized scores. ERNIE is QK-RMS-normed with exactly the `[B,1,Sq,Skv]` F32 additive mask the 44.10 fused path supports; one-line `allowF16:true` → `[cuDNN SDPA] fused flash-attention engaged (D=128)` confirmed in log. |
| **Drain-free loop** | Host `scheduler.Step` + `CfgHelper.ApplyCfg` (velocity D2H ×2 + latent re-upload per step) → in-place device `CfgEulerStep(latent, cond, uncond, cfg, Dt(i))`; latent device-resident all loop; masked-inpaint keeps the host branch. |
| **TE prompt cache + KEEP_MODELS** | Cond+uncond Ministral-3B hiddens keyed on (tokens, realLen); repeat prompts skip the whole TE phase (preload+encode+free of ~7.7 GB). DiT stays resident across gens; prompt-cache MISS evicts it for the TE (Qwen pattern). |
| **Device tail** | `UnpatchifyTokens` for the per-forward host unpatchify loop (p=1 → pure transpose; was a full D2H drain per forward), device `ChwF32ToHwcU8` rgb, VAE `PreloadWeights`. |

**Round 2 menu:** profile the ~0.9-1.0s/step split (fp8 GEMM vs glue), `HARTSY_DIT_F16` opt-in for
ErnieImageBlock (GELU-gated FFN 12288 — overflow audit first, sandwich-damp if needed), fused QKV,
wall-vs-pipeline gap attribution (~3s of Swarm/extension overhead seen under contention).

**Boogu unblocked (routing metadata, NOT a core update):** the Swarm core (HEAD 2026-07-04) already ships the
`boogu` detector; the real blocker was stale cached metadata — arch `null` in Swarm's model DB and a sidecar
`"boogu-image"` id from an old extension registration (unknown id → normalized null; Swarm never re-detects
existing entries). Fix = `/API/EditModelMetadata` with `type:"boogu"` per model (base/turbo/edit fp8) — sets
ModelClass from the live registry + resaves sidecar + DB. First Swarm-routed Boogu bench queued.

**Concurrent-session note:** engine versions are a shared namespace — the video session took `44.18-local`
mid-evening; this round shipped as `44.19-local` with a hand-synthesized meta nupkg (the 44.13 lesson: `dotnet
pack` on the sln emits sub-packages only). After any unexplained connection-refused, re-verify pin + DLL md5.

## 2026-07-09 — dual-backend routing trap + fresh ComfyUI baselines (incl. FIRST Boogu Comfy numbers)

**Methodology trap (cost one suite):** the concurrent video session enabled the ComfyUI backend (#1) at
~00:35 — with BOTH backends enabled, Swarm load-balances `/API/GenerateText2Image`, and `bench_t2i.py` does
NOT toggle backends per its doc claim. An entire "Hartsy" suite (ERNIE + Z-Image + Boogu) silently ran on
**ComfyUI**; the tell was Hartsy's Boogu VRAM pre-check failing (`≥16 GB free` — Comfy held the card) while
"results" kept arriving. **Rule: always pin routing per-request with `exactbackendid` (Hartsy = "2");
never rely on which backends happen to be enabled.**

The mis-routed suite is still a clean, GPU-quiet **ComfyUI** dataset (warm medians, RTX 4090, 1024²):
| Model | ComfyUI warm | Note |
|---|---:|---|
| ERNIE-Image (20 st) | **23.93s** | re-validates the documented 24.0s |
| Z-Image-Turbo (8 st) | **3.09s** | re-validates the documented 3.1s |
| **Boogu Turbo (4 st)** | **2.54s** (cold 19.4s) | FIRST Comfy Boogu baseline |
| **Boogu Base (20 st, cfg 4)** | **17.78s** (cold 28.5s) | FIRST Comfy Boogu baseline |

Comfy's Boogu output verified coherent (sharp astronauts, viewed). The Hartsy-pinned suite (engine
`alpha.44.20-local` = ERNIE round 1 + Boogu round 1) is queued behind a GPU-quiet gate; its results replace
the pending entries below when they land.

## 2026-07-09 — FINAL pinned suite (engine `alpha.44.20-local`, `exactbackendid=2`, GPU-quiet): ERNIE **BEATS Comfy 20.0 vs 23.9s**; Boogu first Swarm bench 48.9→**5.05s** Turbo / ~6min→**43.2s** Base

All warm medians of 3 (Base: 2), random seeds, RTX 4090, 1024², routing pinned to the Hartsy backend, Comfy
VRAM flushed via `/API/FreeBackendMemory` before each phase, every output visually verified coherent.

| Model | Hartsy 44.20 | ComfyUI (same night) | Verdict |
|---|---:|---:|---|
| **ERNIE-Image (20 st)** | **20.03s** (19.99/20.03/20.09, peak 13.7 GB) | 23.93s | **1.19× FASTER** (round 1: was 49.62s = 2.07× slower this same evening) |
| **Boogu Turbo (4 st)** | **5.05s** (5.06/5.03/5.05; cold 24.2s) | 2.54s | 2.0× — was **48.9s engine-level** (9.7× round-1 speedup); first Swarm-routed bench ever |
| **Boogu Base (20 st, cfg 4)** | **43.2s** (44.2/42.2; cold 58.0s) | 17.78s | 2.4× — was ~6 min engine-level (~8×) |
| Krea2-Turbo (flagship gate) | **4.52s** ✓ | 6.5s | coherent ✓ viewed |
| Z-Image-Turbo (flagship gate) | **2.98s** ✓ | 3.1s | coherent ✓ viewed |

ERNIE wall 20.0s vs 20.3s pipeline-logged → the Swarm tail is ~0s warm (the earlier 27s walls were pure
contention). Boogu images (Turbo 5s + Base 44s) are sharp/detailed astronauts — the device-rope port, caption
cache and drain-free loop verified on the real CUDA path (bit-equivalence was pre-proven on CPU by
`OmniGen2RopeDeviceTableTests`).

**Boogu round-2 menu (uniform ~2× vs Comfy on both variants → per-forward cost):** (1) `allowF16:true` on
the Single/Double block SDPAs — Boogu is QK-RMS-normed with mask=null, EXACTLY the proven cuDNN fused config,
and was left F32-materialized in round 1; (2) packed-latent loop (kill the per-forward host `PatchifyNCHW`
D2H drain — no device patchify op exists yet; the Z-Image/Krea2 token-space pattern); (3) `HARTSY_DIT_F16`
audit; (4) edit-path embed cache. ERNIE round-2: profile the ~0.9s/step split, DIT_F16 (GELU FFN 12288
overflow audit), fused QKV.

## 2026-07-09 — Boogu round 2: Turbo 5.05→**3.26s**, Base 43.2→**26.5s** — engine `alpha.44.22-local` — cuDNN gate widened to D=120

Three fixes, warm medians pinned (`exactbackendid=2`), coherent ✓ viewed (Turbo base-camp scene, Base sunset
silhouette), flagship regression clean (Krea2 **4.54s** ✓, Z-Image **2.99s** ✓ — the shared gate change is
identical for D=128):

| Fix | What |
|---|---|
| **cuDNN head-dim gate widened** (shared, `CudnnSdpa.ShapeSupported`): `{64,128,256}` → multiples of 8 in [64,128] + 256 (the documented SM80+ flash-fprop envelope). Boogu's D=120 (3360/28, axes [40,40,40]) was silently excluded; per-D rejection isolation (the Ideogram mechanism) makes a wrong guess cost one warning + materialized fallback, never a session kill | `[cuDNN SDPA] fused flash-attention engaged (D=120)` confirmed |
| **allowF16 on both Boogu block SDPAs** (QK-RMS-normed, mask=null — the proven config; left F32-materialized in round 1) | with the gate fix, all Boogu attention now rides the fused engine |
| **Packed-latent loop**: `BooguImageTransformer.ForwardPacked` (shared `ForwardCore`; NCHW `Forward` kept for edit) + pipeline keeps the latent in `[1, imgLen, p²·C]` token space across the loop — patchify once (seed-compatible: noise still seeded NCHW), device unpatchify once before the VAE. Kills the per-forward host `PatchifyNCHW` D2H drain. **Bit-equality vs the NCHW path unit-tested** (`Transformer_ForwardPacked_Matches_ForwardNchw`) | |

| Model | Comfy | round 1 | **round 2** | gap |
|---|---:|---:|---:|---|
| Boogu Turbo (4 st) | 2.54s | 5.05s | **3.26s** (3.30/3.25/3.25) | 2.0× → **1.28×** (from 19× at 48.9s) |
| Boogu Base (20 st, cfg 4) | 17.78s | 43.2s | **26.5s** (26.54/26.53) | 2.4× → **1.49×** |

**Round-3 menu (to beat 2.54/17.8):** HARTSY_DIT_F16 for the Boogu blocks (SwiGLU FFN — overflow audit,
sandwich-damp if needed), fused QKV / w1w3 (needs common-scale fp8 requant), Lt algo tuning at the joint-seq
M, per-gen step graph, edit-path embed cache. Note: D=96/112 models fleet-wide may now also fuse for free —
check any allowF16 caller with an odd head dim.

## 2026-07-09 — SDXL round 1: 33.9s → **3.69s = TIED with ComfyUI (3.7s)** — engine `alpha.44.24-local`

Warm median **3.69s** (3.74/3.69/3.68, random seeds, `exactbackendid=2`, GPU-quiet), cold 45.2 → 17.6s, peak
VRAM 10.0 GB. Fresh same-morning baseline on 44.22: 33.94s (33.94/35.02/32.58) — matches the standing 33.0s.
**9.2× in one round; the worst gap in the image fleet (8.9×) is now parity.** Coherent ✓ viewed (sharp
astronaut, correct prompt); seed-50601 A/B vs the 44.22 baseline image: corr 0.9990, mean|Δ| 0.72/255,
std-ratio 1.0001 (no washout) — batch-2 GEMM micro-drift only. Flagship regression clean + coherent viewed:
Krea2-Turbo **4.50s** ✓ (<6.5), Z-Image-Turbo **2.98s** ✓ (≤3.2). CPU e2e test passes the fused path.

Baseline warm phase split (44.22): TE 4.4s / UNet re-preload 2.1s / denoise 20×~1.39s = 27.5s / VAE 0.37s.
Now: TE 0 (cache) / preload 0 (resident) / denoise 20×~153ms = 3.10s / VAE 0.38s.

**THE find — `Tensor.Reshape` (and `.To()`) on GPU activations are hidden host round-trips.** `Reshape`
touches `DataPointer` on construction, which forces a full D2H sync of the source activation, and the view is
a NEW Tensor object so the consuming op cache-misses and re-UPLOADS the bytes (`H2D_MISS_BIG`). The SDXL
attention stack did this 4-6× per sub-block × ~70 sub-blocks (multi-head split/merge views + seq↔spatial
reshapes) plus 9 `.To(Device)` skip-connection clones per forward — profiled **9,149 multi-MB H2D misses per
gen (~457/step, 6.4s host issue)** with the GPU idling at ~44%. This is the Axis-A host-glue pathology in a
new costume: not host LOOPS, host ROUND-TRIPS from tensor plumbing. **Audit rule: grep DiT/UNet block code
for `.Reshape(`/`.To(` on activations, not just `DataPointer` reads.**

One deploy, both fix classes + the standard recipe:
| Fix | What |
|---|---|
| **Reshape-free attention blocks** | `Transpose2D`/`Permute0213` take explicit dims and read buffers flat, so the views were pure overhead: pass the un-reshaped tensor in, and allocate outputs directly in their final shape (shape is metadata). Zero Reshape left in the UNet path. |
| **Device skip clones** | `UNetBlockHelpers.CloneOnDevice` (unit-`Scale` kernel pass) replaces `hidden.To(hidden.Device)` in UNet + DownBlock (9 D2H+memcpy+H2D round-trips/forward gone). |
| **Batched CFG** | cond+uncond as ONE batch-2 UNet forward (`RunDenoiseLoopFused`); at cfg≤1 (Turbo/Lightning) batch-1 cond-only. Halves per-step op dispatch. |
| **Drain-free device loop** | in-place `CfgEulerStep(latent, cond, uncond, cfg, σ[i+1]−σ[i])` — exactly Euler/epsilon (`EulerDiscreteScheduler.FusedEulerCompatible` gates; v-pred + ddim/dpm++/lcm/tcd keep the host loop). Device `SliceRows` splits the batched prediction. Step-invariant conditioning built once/gen: batched text emb pre-cast to the UNet dtype + **cached ADM embedding** (`UNet.ComputeAdmEmbedding` — the old path re-built the ADM sinusoid on host AND read `pooled.DataPointer` EVERY forward). Latent host-materialized once post-loop (the tiled-VAE fallback slices on host). |
| **TE prompt cache + KEEP_MODELS** | dual-CLIP embeddings keyed on all 4 token streams + EOS + clip-skip (repeat prompts skip the 4.4s TE phase — SDXL's dual-CLIP encode is itself suspiciously slow, likely the same Reshape pathology, round-2 item); UNet stays resident (2.5 GB F16 fits beside VAE + CLIPs). Refiner StepSwap bypasses the cache (needs raw CLIP-G). |
| **allowF16 on UNet SDPA** | F16 checkpoints already ride the F16-native cuDNN gate (`engaged (D=64)` confirmed, Skv=77 cross-attn accepted); the flag covers F32-loaded checkpoints. |
| **Preview throttle** | fused loop emits latent previews every 4th step + final (each is a deliberate D2H). Masked inpaint / ControlNet / IPA / refiner / conditioning schedules keep the reference host loop, unchanged. |

SD1.5 / SDXL-Refiner / SDXL-Inpaint share the UNet blocks → they inherit the Reshape/skip-clone wins free
(their pipelines still run the host loop — porting the fused loop to SD15 is a cheap follow-up).

**Round-2 menu (to BEAT 3.7 decisively):** cold-path TE encode 4.4s (Reshape audit in `ClipTextEncoder`),
step 153ms → profile the GPU split (conv im2col vs GEMM vs attention; cuDNN-conv/winograd is the classic SDXL
lever), VAE 377ms (F16 conv), per-gen step graph if host issue resurfaces, cold load 17.6s.

**Incident logged:** running two SwarmUI instances against the same `Data/` (a second launch while an
incumbent held 7801) corrupted `Users.ldb` (LiteDB "Detected loop in Find" on every `GetNewSession`).
Recovered from `Data/UsersBackups/UsersBackup_2026_27.ldb`; corrupted file preserved as
`Users.ldb.corrupt-2026-07-09-dualinstance`. **Always check for an existing SwarmUI (ss -tlnp | grep 7801)
before launching.**

## 2026-07-09 — SDXL round 2: 3.69s → **2.93s = BEATS ComfyUI (3.7s, 1.26× faster)** — engine `alpha.44.26-local` — cuDNN conv + fleet-wide GEMM/VAE wins

Warm median **2.93s** (2.93/2.94/2.91, pinned, GPU-quiet), steps 153→~125ms, VAE 377→235ms, peak 13.6 GB.
Coherent ✓ viewed; seed-50601 A/B vs the ORIGINAL 44.22 baseline image: corr 0.9991, std-ratio 1.0005 (all
three rounds of changes preserve composition). CPU e2e + new GPU parity tests pass. **Flagships IMPROVED by
the shared changes: Krea2-Turbo 4.48s ✓, Z-Image-Turbo 2.77s ✓ (was 2.98 — free −0.2s), both coherent ✓ viewed.**

Attribution first (PROFILE_SYNC per step): Linear ~81ms / Conv2D ~39 / SDPA ~25 / GroupNormSilu ~14 /
Permute ~10 — GEMM+conv-bound, host-glue class gone. Four changes, three of them FLEET-WIDE:

| Change | What | Δ |
|---|---|---|
| **cuBLASLt bias-epilogue GEMM promoted to standard profile** (`EnableEpilogueFusion` default ON, was strict opt-in) | every biased Linear ran GemmEx + a separate BiasAdd kernel + an output-sized HBM round-trip (~700/step); the Lt path folds it into the GEMM epilogue | −0.16s/gen SDXL; helps every model |
| **cuDNN convolution forward** (`CudnnConv.cs` + `HARTSY_CONV_CUDNN`, default ON, self-disable + im2col fallback — the SDPA pattern; enum values verified vs NVIDIA docs, conv-fwd attrs are 700-705) | F16/BF16 NCHW convs skip the im2col materialization (a kH·kW× input-sized HBM write+read per conv) for tensor-core implicit-GEMM/Winograd engines; ~50 convs/step + the whole VAE | Conv2D −32% serialized; parity-tested vs im2col (`CudnnConvTests`, max|Δ| ≤2e-3, 1×1 bit-exact) |
| **VaeAttention Reshape purge** (shared by SD/SDXL/Flux/video VAE mid-blocks) | 5 view round-trips (~16 MB each) per decode + rank-normalizing views; CPU GroupNorm made rank-agnostic (was hardcoded Shape[2]·Shape[3] — silently wrong for rank-5 video without the old view) | VAE 297→235ms; Z-Image/Krea2 inherit |
| **Cross-attention K/V cache** (TransformerSubBlock, keyed on context tensor ref) | the text conditioning's K/V projections are step-invariant — now computed once per gen instead of ×20 steps (140 GEMMs + 140 permutes/step removed); per-step-slicing callers (legacy loop) behave exactly as before | small (~1ms/step — the GEMMs were tiny; kept for the launch-count win) |

**Round-3 reality check (the sub-2s question): SDXL @1024²/20 steps/CFG-batch-2 is ~12 TFLOP/step ≈ 240
TFLOP/gen. At the 4090's ~165 TFLOPS F16 peak the denoise loop's hard floor is ~1.5s ideal — we measure
2.5s at 100% GPU util (~63% of peak, already efficient). Sub-2s TOTAL is not reachable in F16.** The honest
menu: (1) **fp8 UNet GEMMs** (~330 TFLOPS Ada; requant F16→e4m3 at load with per-tensor amax/448 scales on
`Fp8ScaleFactor` — plumbing exists via `QualityProfileApplier`+`Fp8Executor`; est. → ~2.0-2.2s) — but this
CHANGES OUTPUT (Comfy's 3.7s runs F16 weights), so it's a quality-vs-speed default decision, not a perf fix;
(2) GroupNormSilu grid fix (batch·groups=64 blocks on 128 SMs; ~−0.15s); (3) fused self-attn QKV at load
(~−0.1s); (4) fp8 would also want the CFG-collapse check (the Wan lesson) at cfg 7.

## 2026-07-09 — AuraFlow round 1: 31.4s → **13.93s = TIED with ComfyUI (14.0s)** — engine `alpha.44.28-local`

Warm median **13.93s** (13.96/13.93/13.92, pinned, GPU-quiet), cold 19.0s, peak 13.5 GB. Was 31.4s (2.2×).
Coherent ✓ viewed (sharp astronaut-on-horse). Checkpoint re-downloaded (`calcuis/aura` fp8, 9.66 GB — an
/tmp-incident casualty) into the shared Swarm Models dir. Flagship + SDXL regression clean: Krea2 4.48s ✓,
Z-Image 2.75s ✓, SDXL 2.91s ✓. GPU e2e tests pass per-process (both CFG modes).

The standard recipe, one deploy: Pile-T5 prompt cache + KEEP_MODELS residency; drain-free PACKED loop
(patchify once → token-space latent all loop → in-place `CfgEulerStep` per branch → device
`UnpatchifyTokens` once + throttled previews); transformer glue de-hosted (cached device pos-embed slab,
TWO-SLOT step-invariant text-token cache — dual-pass CFG alternates two contexts and a single slot would
thrash; device [text|image] seq concat + SliceRows split at B=1; device pre-final modulate via
SliceLastDim+AddScalar+AffineBroadcastLastDim); `allowF16` on both block SDPAs (QK-RMS-normed, mask-null,
**D=256 fused confirmed engaged**). `HARTSY_AURAFLOW_PACKED=0` kill-switch.

**Bug found on the way (cost one noise-output deploy): AuraFlow's token layouts are ASYMMETRIC** — patchify
feeds the patch projection channel-outer `(c, py, px)`, but `proj_out` emits spatial-first `(py, px, c)`
(see `Unpatchify`). The legacy loop unpatchifies every forward so it never notices; a token-space Euler
update adds velocity directly onto the latent tokens, so the velocity must be permuted back to the latent
layout (one `Permute0213` with S=P², H=C, D=1) — without it every step adds mismatched layouts → pure noise.
**Lesson for every future packed-loop port: verify the model's patchify-in vs proj-out token layouts MATCH
before doing token-space Euler; and the GPU e2e tests' not-black/not-white assertions PASS ON NOISE — only
a viewed image (or a legacy-vs-packed A/B) is a real gate.**

**Round-2 menu (to beat 14.0s):** batched CFG (needs batch-aware seq concat/split — the single-stream stack
mixes text+image rows per batch element), `HARTSY_DIT_F16` opt-in for the blocks (QK-normed; MLP overflow
audit), profile the 680ms/step split (fp8 GEMM vs glue), fused QKV.

**Known issue flagged (pre-existing, NOT from this session's cleanup): Flux.2 Klein fails to load** on
44.26+ with `Unsupported dtype conversion: U8 → F16` in `Flux2Loader.Load` (the Qwen3-4B TE cast loop hits a
U8 tensor in `qwen_3_4b.safetensors`, dated May 10). The loader needs to skip/handle U8 metadata tensors
(e.g. ComfyUI `scaled_fp8` markers). The 15.1s Klein scoreboard entry is not currently reproducible.

## 2026-07-09 — Flux.2 Klein loader FIXED (U8/comfy-quant TE) + re-bench: 15.1s → **7.50s** — extension-only fix

**The Klein "U8 → F16" load failure is fixed.** Root cause: `qwen_3_4b.safetensors` is a ComfyUI
comfy-quant MIXED checkpoint (fp8 weights + `.weight_scale` scalars, some layers U8-PACKED NVFP4 with
F8-E4M3 block scales + `.weight_scale_2` globals, plus `.comfy_quant` U8 JSON metadata blobs).
`Flux2Loader` ran its own cast-everything-to-F16 loop over the raw dict BEFORE handing it to the encoder —
crashing on the U8 tensors, and (worse, silently) upcasting fp8 weights before their scales were folded.
The engine already handles ALL of this: `LlamaStyleEncoder.LoadWeights` runs
`TextEncoderQuantNormalizer.Normalize` (fp8 scales → `Fp8ScaleFactor` with weights kept PACKED on the
native fp8 GEMM path; NVFP4 dequant via block-scale companions; metadata blobs dropped). Fix = delete the
loader's manual cast loop and pass the raw dict through. **Rule: never pre-cast a text-encoder weight dict
in a loader — the encoder's normalizer owns quant handling.**

Re-bench @1024²/10 steps (pinned, GPU-quiet): warm **7.50s** (7.50/7.56/7.48), cold **10.1s** (was 46s at
first bench), peak 13.2 GB. The 15.1s → 7.50s is the accumulated fleet inheritance since 44.15 (Lt bias
epilogue, cuDNN conv, VAE fixes) + the TE's packed-fp8 GEMMs. Coherent ✓ viewed (sharp on-prompt
lighthouse). No ComfyUI baseline exists for Klein yet (not wired on the Comfy backend) — still a
Hartsy-only number.

### Klein addendum — the "blurry artifacts" were a PARAMETER bug: this is the step-DISTILLED variant (4 steps, not 10)

The oily/over-sharpened texture on Klein outputs (smudged skies, crunchy grain) was over-stepping:
`flux-2-klein-4b` (no "base" in the name) is the step-distilled Klein — official settings are **exactly
4 steps, CFG 1.0-1.5, Euler/Simple**; more steps "breaks the image math" (ComfyUI docs / community guides).
Every bench since the first (44.15's 15.1s, today's 7.50s) ran 10 steps. At the official 4 steps the same
seeds render clean and photographic (viewed A/B, seed 424242 + bench seeds). Base-variant Klein checkpoints
(`flux-2-klein-base-*`) want 20-24 steps + CFG 3.5-5 instead — note our pipeline's Flux.2 loop is g=1
(guidance-embedded); true-CFG for a base checkpoint would need the dual-pass path.

**Corrected scoreboard number: warm 3.45s @1024²/4 steps** (3.44/3.45/3.52, pinned, GPU-quiet), peak
13.0 GB. `benchmarks/swarm_image_bench/models.json` now carries the Klein entry at 4 steps.

## 2026-07-09 — Chroma round 3 OPENED: profile attribution + step-invariant mask cache: 63.2s → **61.1s** — engine `alpha.44.29-local`

Serialized profile (the round-3 "profile first"): per warm gen ≈ **Linear 12.4s** (17.5k calls — real fp8
GEMM work) / **SDPA 6.8s** (2.96ms/call — the F32→F16 CAST route, not zero-cast) / **F32 glue kernels ~6.3s**
(Permute/Gelu/RmsNorm/LNNoAffine/GatedResidual/Rope) / **H2D_MISS_BIG 642/gen ~1.5s**. Serialized GPU ≈ 27s
vs 63s wall → a large host-issue tail over ~60k ops/gen.

Lever 1 landed: the fused-SDPA bias mask ([1,1,4608,4608] F32, 85 MB) was HOST-REBUILT (21M-iteration loop)
+ re-uploaded EVERY forward (40×/gen) — now cached per (mask ref, seqLen), two slots for the alternating
CFG branches (the AuraFlow/ERNIE pattern). 63.2 → **61.1s** (61.09/62.60/60.20), coherent ✓ viewed.

**The remaining round-3 levers are the big ones (next session-scale arc):** (1) `HARTSY_DIT_F16` port for
ChromaDouble/SingleBlocks — halves the ~6.3s F32 glue traffic AND makes all 57 SDPAs zero-cast native-F16
(6.8 → ~4s) AND halves activation alloc churn (overflow audit + sandwich-damp per the Z-Image/Ideogram
recipe); (2) per-gen step CUDA graph (the loop is already drain-free) to erase the ~30s host-issue tail —
the largest single component; (3) Lt algo tuning / fused QKV on the fp8 GEMMs.

## 2026-07-09 — Chroma round 3 EXECUTED: F16 blocks + persistent CFG-pair CUDA graph + context trim: 61.1s → **28.3s** — engine `alpha.44.30–44.32-local`

All three round-3 levers landed in one arc, plus two the profile surfaced along the way. Warm median
**28.35s** (28.35/28.42/28.34, @1024²/20 steps/cfg 4, peak 23.5 GB), Comfy 16.6s → gap 3.7× → **1.7×**.
Coherent verified twice (local 20-step CFG-5 seed-42 BMP + Swarm bench PNG, both viewed).

1. **F16 block loop — exact residual damp, no sandwich norm needed** (`ChromaF16.cs`). Chroma is
   Flux-family: the F16 killer is residual-stream growth, and unlike Z-Image there is no sandwich norm to
   damp against. Instead the WHOLE residual stream rides at 1/32: damp x_embedder + context_embedder and
   every branch-output projection (attn.to_out.0, to_add_out, ff*.net.2, single proj_out). Every branch
   INPUT passes through the scale-invariant no-affine LayerNorm (sees baseline values); the final
   no-affine LN cancels the factor exactly before proj_out — the velocity is unchanged in real arithmetic,
   and 1/32 is a power of two so F16 loses zero relative precision. Weights damp via `Fp8ScaleFactor`
   (GEMM alpha, any dtype); biases need value-scaled copies (the epilogue adds bias after alpha). New
   `dit_layernorm_noaffine_f16` kernel (NVRTC recipe: pass `~/.local/lib/cuda13/include` as the include
   dir for `cuda_fp16.h`). All 57 SDPAs now ride the zero-cast F16 cuDNN fused engine.
2. **Persistent cross-generation step graph of the WHOLE CFG pair** (approximator → cond forward → uncond
   forward), replayed as one `cuGraphLaunch`/step at 6ms host issue. `HARTSY_DIT_GRAPH` became tri-state:
   default ON for Chroma (`DitStepGraph.EnabledDefaultOn`), still opt-in for Z-Image/Krea2 (wall-neutral
   there), `=0` kills everywhere. Chroma's pipeline never calls FreeActivations, so the graph SURVIVES
   across generations — repeat-prompt gens replay all 20 steps with zero re-warm/re-capture (exactly ONE
   capture across the bench's 4 gens). Safety net: `CudaBackend.FreeActivations`/`FreeAllDeviceMemory` now
   reset the graph slot themselves (the VAE full-res OOM fallback calls FreeActivations — was a latent
   poisoning CUDA 700), and the transformer re-warms when it finds the slot externally reset.
3. **Context TRIM replaces the SDPA mask.** Chroma keeps text positions `i <= text_len` — so SLICE the T5
   context to `text_len+1` rows instead of computing 512 padded tokens and masking them out of every
   attention. Exact (dropped tokens can't influence kept outputs; verified: seed-42 composition matches
   the untrimmed run), bench prompts trim to 29/2 kept tokens: −11% joint-seq GEMM, −20% attention score
   work, the 85 MB additive mask (and round 3's two-slot mask cache) disappear entirely, and cuDNN runs
   its mask-free fused engine.
4. **Profile finds:** `ChromaApproximator` ran a HOST RmsNorm loop reading the device hidden's DataPointer
   (5 pipeline drains per table build + a capture invalidator) → `backend.RmsNorm`; `FluxRope.Precompute`
   rebuilt host tables + re-expanded + re-uploaded the GPU cos/sin tables EVERY forward → two-slot rope
   cache keyed (txtSeqLen, grid). ForwardPaired also builds the modulation table ONCE per step (was twice).
5. **THE eviction bug (44.32): `cuDeviceGraphMemTrim`.** Memory allocated by captured allocation nodes
   lives in a per-device GRAPH pool that `cuGraphExecDestroy` does NOT release and `cuMemPoolTrimTo` never
   touches. The destroyed Chroma pair-graph kept its multi-GB working set reserved through model eviction
   and OOM'd the next model's load (Krea2, 88 MB free). `StepGraphReset` now syncs + trims the graph pool
   whenever it destroys a captured graph. This was invisible until now because no default-on graph model
   existed.

**Flagship regression (same session, 44.32):** Z-Image-Turbo **2.76s** ✓ (≤3.2 gate), Krea2-Turbo re-run
after the graph-mem fix (was the OOM victim) — see results json.

**Remaining gap (~1.7×) is GPU math, not host:** replay pair ≈ 1.3s; the fp8 Linears already run ~200
TFLOPS effective (≈ Comfy parity per token) and SDPA is fused. Next levers, in order: **batched CFG**
(B=2 in one forward — Comfy's remaining structural edge; needs the B==1-gated device paths widened),
cuBLASLt algo tuning, F16/BF16 VAE + device tail. Fused QKV is OFF the table for fp8-scaled checkpoints:
q/k/v carry different per-tensor weight_scales — one GEMM alpha can't represent them without lossy requant.

### Round-3 addendum (44.33–44.35): the step-graph driver-cache OOM — diagnosed and fixed

The first Krea2-after-Chroma bench OOM'd (88 MB free at its VAE decode). Instrumented eviction +
a standalone real-pipeline harness pinned it: after a session that captured a step graph, destroying the
graph leaves **~4.5 GB of driver-side lazily-reclaimable cache** that is reported as USED by
`cuMemGetInfo`, is NOT in the graph pool (`cuDeviceGetGraphMemAttribute` reads 0 used/0 reserved), and is
returned by neither `cuMemPoolTrimTo` nor `cuDeviceGraphMemTrim`. Capture-window accounting proved the
graph itself is balanced (7117 allocs = 7117 frees, ~90 GB round-tripped, zero outstanding); the size is
constant regardless of replay count. The driver releases this cache ONLY under **synchronous** allocation
pressure: a probing `cuMemAlloc` of the needed size succeeds and un-hides the memory, while stream-ordered
pool growth (`cuMemAllocAsync`) just fails.

**Fix (44.35):** `CudaMemory.AllocateAsync`'s OOM retry now probe-allocates the requested size with
`cuMemAlloc`, frees the probe, and retries the async alloc — self-healing wherever the pressure appears
(verified in the harness: pool growth 63 MB free → probe → 4.9 GB free → 17×1 GB async allocs succeed).
Plus defense-in-depth: `FreeActivations`/`FreeAllDeviceMemory` reset the step-graph slot (a captured graph's
baked pointers die with the activations — the VAE full-res OOM-fallback path was a latent poisoning
CUDA 700), the transformer re-warms when it finds the slot externally reset, and `StepGraphReset`
graph-mem-trims after destroying a captured graph. A capture-window alloc/free tracker logs one line per
capture — OUTSTANDING > 0 there means a tensor created inside the captured region is never disposed, i.e.
a permanent per-graph leak.

**Final 44.35 deploy verification (fleet bench, sequential Z-Image → Chroma → Krea2 on one backend):**
Z-Image-Turbo **2.76s** ✓ (gate ≤3.2) · Chroma1-HD **28.5s** median (28.54/28.51/28.54, peak 23.5 GB, ONE
graph capture across all gens — cross-gen persistence confirmed) · Krea2-Turbo **4.52s** ✓ (gate <6.5),
loading AFTER the Chroma graph session — the sync-probe reclaim fired at its VAE decode
(92 MB free → 4.8 GB free) exactly where the OOM used to be. All three outputs viewed, coherent.

## 2026-07-10 — Board-clearing round 1: Krea2-Base validated, Comfy baselines for Schnell/Klein, Lumina2+OmniGen2 wired

- **Krea2-Base CFG path VALIDATED** (was validation-pending since the Krea2 bring-up): warm **30.3s**
  median (30.23/30.29/30.33, cold 45.9) @1024²/28 steps/cfg 4.5, peak 21.2 GB, on `44.35-local`. Output
  sharp + on-prompt with correct contrast (no over-guidance) — the standard uncond-anchored CFG anchoring
  at `Krea2Pipeline` is correct. No ComfyUI baseline run yet for Base.
- **ComfyUI baselines recorded** (comfy selfstart backend 1, same 4090, same requests):
  **Flux-Schnell 3.04s** (ours 10.5s → **3.5× slower**, worse than Flux-Dev's 2.5×!) and
  **Flux.2 Klein 4B 1.85s** (ours 3.45s → **1.9×**). Both join Flux-Dev in the Flux-family grind bucket —
  the Chroma round-3 kit (persistent step graph, F16 residual damp, rope/ctx caches) transplants directly.
- **Lumina2 + OmniGen2 wired into the Swarm extension** (were engine-✅ but unreachable): new
  `Lumina2Loader`/`OmniGen2Loader` with LIVE text-conditioning — Lumina2 runs Gemma-2-2B
  (`hidden_states[-2]` tap via EncodeMultiLayer, system-prompt + " <Prompt Start> " template, verified
  coherent @1024/25st/cfg4) and OmniGen2 runs Qwen2.5-VL-3B (ComfyUI `text_encoders/omnigen2.py` template
  parity: full templated sequence, final-norm'd last hidden state, no prefix drop). Side-models
  (`gemma_2_2b_fp16`, `qwen_2.5_vl_fp16` 3B, Flux ae) auto-resolve via the SideModels registry.
  **Trap:** the diffusers-format Lumina2 transformer matches core's `isOmniGen` key predicate
  (`time_caption_embed.*` + `context_refiner.*` — OmniGen2 is Lumina-derived) and got classed `omnigen-2`;
  fixed per-model via `EditModelMetadata type=lumina-2`. The robust discriminator for a future core fix:
  OmniGen2 has `ref_image_patch_embedder.*` / `image_index_embedding`, Lumina2 doesn't.
  Both models are SLOW pre-optimization (Lumina2 650s @1024²/25st — F32 + host-glue; the residency
  backlog applies) — wiring ≠ perf round.

## 2026-07-10 — HiDream-i1 residency round 1: ~29s/step → **~1.4s/step (≈20×)**, 1024-CFG blocked on VRAM

The 29s/step causes, all fixed this round: **(1) MoE host loops** — `ComputeTopKGateWeights` (softmax+top-k
on host, reading the device logits' DataPointer) and the per-expert `AccumulateGatedExpert` host loop =
~5 full pipeline drains per block × 48 blocks × forward. New `dit_moe_topk_gate_f32` +
`dit_row_gated_accum_f32` kernels behind `IBackend.MoeTopKGate`/`RowGatedAccumulate` (host defaults hoisted
— CPU backend unchanged bit-for-bit); the shared expert is now the in-place accumulator. **(2) Host
multi-head reshapes/concats/splits** in both attention paths → the Chroma idiom (flat-layout device Concat,
`Permute0213`, `SliceRows`) for B=1. NOTE: HiDream QK-norm is over the FLAT hidden (not per-head) — norm
first, then dim-explicit permute on the flat tensor. **(3) Host RoPE** → `FluxRope.ApplyGpu` pre-permute
over the full joint sequence (text positions all-zero = identity, bit-identical to the old image-only
rotation). **(4)** rope `Precompute` ran twice per forward (identical args) → signature cache.
**(5)** SDPA now passes `allowF16` (QK-normed → cuDNN fused engine). Pipeline: KEEP_MODELS residency +
quad-encoder (CLIP-L/G+T5+Llama, 49 hidden states) prompt cache with DiT-evict-on-miss staging + planned
pool trims at both handoffs.

**Result (Dev config, 25 st/no-CFG @1024): 50.6s total ≈ 1.4s/step (was ~29s/step)**, composition correct.
Note the local `hidream_i1.safetensors` is now the **Full** fp8 (re-downloaded after the /tmp loss — the
no-CFG render is expectedly soft; Full wants cfg≈5).

**Blocker: 1024²/50-step CFG OOMs on step 1** — ~6.9 GB live mid-forward beside the 17 GB resident fp8
weights (F32 activations; the old host drains were accidentally throttling the queue). Serializing the CFG
pair + pool trims didn't close it. **Next lever = the F16-activation port** (the fleet recipe — halves every
transient) plus an audit of mid-forward live-activation growth; until then the standard bench config
(1024², cfg 5) can't run, so no scoreboard row yet. Comfy comparison pending that.
