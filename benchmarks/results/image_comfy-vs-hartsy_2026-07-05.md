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
