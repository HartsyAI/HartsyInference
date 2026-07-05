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
