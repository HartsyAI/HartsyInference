# Flash-Attention Kernel — Implementation Plan (2026-07-03)

Attention (`ScaledDotProductAttention`) is **~half of GPU time** for Wan-Video DiT (sync-profiled), running at
~5% of the 4090's FLOPS because the non-tiled path **materializes the full `[heads,Sq,Skv]` score matrix** (Wan-1.3B
self-attn ≈ 963 MB → ~4 GB HBM traffic/call). The existing `FlashAttention` kernel is a **naive reference**
(re-reads all K/V per query row) and is *slower* (159s vs 40s). Goal: a real fused flash-attention kernel.

## Build toolchain — VERIFIED WORKING 2026-07-03
`nvrtc_compile.c` was patched to accept include dirs (argv[4..] → `--include-path=`). A WMMA TF32 test kernel now
compiles to PTX with real tensor-core ops (`wmma.mma.sync.aligned.row.col.m16n16k8.f32.tf32.tf32.f32`). Recipe:
```bash
cc -O2 -o native/cuda/nvrtc_compile native/cuda/nvrtc_compile.c -ldl   # rebuild helper (once, after the patch)
TINC="/home/hartsy/Desktop/Swarm/SwarmUI.not too old/dlbackend/ComfyUI/venv/lib/python3.12/site-packages/triton/backends/nvidia/include"
LD_LIBRARY_PATH=~/.local/lib/cuda13 native/cuda/nvrtc_compile in.cu out.ptx compute_80 "$TINC"
```
`$TINC` is a complete CUDA header set (`mma.h`, `cuda_fp16.h`, `cooperative_groups`, `crt/`). TF32 WMMA fragment sizes
are 16×16×8 (`wmma::precision::tf32`); cast fragment elements with `wmma::__float_to_tf32(...)` before `mma_sync`.

## Build reality (CRITICAL)
- **No `nvcc` on this box.** The `native/cuda/*/build.sh` scripts (which call `nvcc -ptx`) are misleading.
- Real recipe: the committed helper `native/cuda/nvrtc_compile` (dlopens `libnvrtc.so`):
  ```bash
  export LD_LIBRARY_PATH=~/.local/lib/cuda13        # libnvrtc.so.13
  cd native/cuda/lm
  ../nvrtc_compile flash_attn_v2_f32.cu flash_attn_v2_f32.ptx compute_80
  cp flash_attn_v2_f32.ptx ../../../src/HartsyInference.Cuda/Ptx/flash_attn_v2_f32.ptx
  ```
  Target `compute_80` (JITs forward to sm_89). nvrtc ships built-in `mma.h`/`cuda_fp16.h` (verify on first compile;
  if missing, add an `--include-path` option to `nvrtc_compile.c`, ~3 lines).
- `HartsyInference.Cuda.csproj` globs `Ptx\*.ptx` into the nupkg — **dropping the new `.ptx` there is the whole
  packaging step**, no csproj edit.
- `TensorCoreGemm` / `hgemm_mma_sm80.ptx` has **NO `.cu` source** (PTX only, "layout unverified") — don't reuse it;
  author fresh WMMA code.
- `cuFuncSetAttribute` is **not yet bound** in `CudaDriverApi.cs` — add the P/Invoke (attrib 8 =
  MAX_DYNAMIC_SHARED_SIZE_BYTES) only if a tile needs >48 KB shared mem.

## Kernel design
Fused, query-tiled, online-softmax kernel using **`nvcuda::wmma` tensor cores, TF32 inputs, F32 accumulate**
(WMMA over hand-written `mma.sync` — nvrtc-friendly, far lower risk; ~10-20% off peak but crushes the memory-bound
baseline). CUTLASS not usable (needs nvcc/headers).

**Precision = the Z-Image-blackout fix:** default path is **TF32-input + F32-accumulate**. TF32 keeps the full F32
*exponent* range, so pre-softmax scores never overflow **even for unbounded fp8 models** — safe for Wan AND Z-Image.
(The earlier F16 blackout was from storing `scoresBuf` in F16 at CudaBackend ~L2208; never do that.) Online softmax
(`m`,`l`,`O_acc`,`S`) stays in **F32 registers/smem** regardless of input dtype. An opt-in F16-*input* variant
(`HARTSY_SDPA_F16`, gated to bounded/RMS-normed callers = Wan) gives the last ~1.5-2×.

**Tiling (specialize D=128):** one block per `(batch, head, query-tile)`. Br=64 query rows, Bc=32 (M1) → 64 (M2)
key cols. Grid `(ceil(Sq/Br), H, B)` = 70×12×1 = 840 blocks for Wan self-attn → good occupancy. Block=128 threads
(4 warps, 16 q-rows/warp for WMMA 16×16). Dataflow: load Q-tile to regs (scaled); loop K/V tiles → smem (double-buffer
in M2); `S=Q·Kᵀ` (WMMA TF32, F32 accum); online-softmax update `m/l`, rescale `O_acc`; `O+=P·V` (WMMA); epilogue
`O/=l`, single write out. HBM traffic ~4 GB → ~55 MB (one read Q/K/V + one write O).
**smem:** M1 Bc=32 single-buffer K+V (32 KB) + Stile[64][64] F32 (16 KB) < 48 KB (no opt-in). M2 Bc=64 double-buffer
→ 64-99 KB → needs `cuFuncSetAttribute`. Transpose between GEMMs = smem round-trip (store S, softmax, reload P), not HBM.

**Scope of the fast path: no-mask MHA, D∈{64,128} only.** Keep the existing materialized + `SdpaTiledF32NoMask` +
naive `FlashAttention` paths as oracle + fallback for masked (Matrix-Game block-causal), GQA (`kvGroup>1`), BF16, other-D.

## Integration
1. `native/cuda/lm/flash_attn_v2_f32.cu` (entry `lm_flash_attn_v2(out,Q,K,V,B,H,Sq,Skv,D,scale,useF16)`); add to
   `lm/build.sh` `KERNELS=()`; build via nvrtc; land `Ptx/flash_attn_v2_f32.ptx`.
2. `CudaKernels.cs`: module load + `GetFunction` (mirror ~L389) + `LaunchFlashAttentionV2` wrapper (mirror
   `LaunchFlashAttention` ~L1230), grid/block/sharedBytes; call `cuFuncSetAttribute` once at load if >48 KB.
3. Dispatch in `CudaBackend.ScaledDotProductAttention` (~L2130): `if (mask is null && query.DType==F32 && D∈{64,128}
   && !EnvFlag("HARTSY_SDPA_V2_OFF")) { FlashAttentionV2(...); return; }`. Keep FORCE_FLASH/FORCE_TILED escape hatches.
4. New `FlashAttentionV2` C# method mirrors `FlashAttention` (~L3260) for CopyToDevice/CacheActivation.

## Validation
Extend `tests/HartsyInference.Cuda.Tests/CudaFlashAttentionTests.cs` vs `AttentionReference` (CPU oracle) + the
materialized path over Wan shapes `(1,12,4480,4480,128)` and cross `(…,4480,512,128)` and D=64. TF32-vs-TF32 ~1e-3 rel;
F16 variant ~2e-2. **Overflow regression:** run F16 variant with large unbounded Q/K, assert finite (proves F32-accum
fixed the blackout). E2E: rerun Wan T2V, view decoded frames (not just stats), confirm `HARTSY_PROFILE_SYNC` shows
SDPA share dropping.

## Milestones
| # | Deliverable | Effort | Risk | Effect |
|---|---|---|---|---|
| M0 | Replace per-head `cublasGemmEx` loops with `cublasGemmStridedBatchedEx` (materialized + tiled paths) | 0.5d | v.low | ~1.05-1.2× Wan self-attn (memory-bound), more for cross/short-seq. Stepping stone. |
| **M1** | Fused TF32 WMMA + F32-accum kernel, Br=64/Bc=32 single-buffer (<48 KB), no-mask MHA, D∈{64,128} + wiring + tests | 3-5d | med | **~3-5× SDPA → ~1.6-2× e2e**. Safe for Z-Image. The milestone that matters. |
| M2 | Bc=64 + double-buffer `cp.async`, `cuFuncSetAttribute` 64-99 KB, tune | 2-3d | med | +~1.3-1.6× kernel |
| M3 | F16-input variant behind `HARTSY_SDPA_F16`, gated to bounded-score callers (Wan), F32 accum retained | 1-2d | low-med | +~1.5-2× Wan; toward PyTorch's ~6.3s |

Net: M1 ≈ halves total GPU time; M1+M2+M3 lands the video DiT in the low-teens of seconds with a safe TF32 default and
Wan-gated F16 opt-in. Related: [`BENCHMARKING.md`](BENCHMARKING.md),
[`../../benchmarks/results/video_comfy-vs-hartsy_2026-07-03.md`](../../benchmarks/results/video_comfy-vs-hartsy_2026-07-03.md).

## Status 2026-07-03: M1 kernel CORRECT, needs M2 tuning
`native/cuda/lm/flash_attn_v2_tf32.cu` written, compiles (WMMA tf32 for both GEMMs), fully wired
(`LaunchFlashAttentionV2Tf32`, `CudaBackend.FlashAttentionV2`, dispatch behind `HARTSY_SDPA_V2=1`,
`cuFuncSetAttribute` 96KB). **Verified NUMERICALLY CORRECT** on real Wan-1.3B T2V (coherent frames, mean 151).
But **SLOWER than baseline: 54.7s vs 23.65s** — the M1 layout keeps the O accumulator + K/V/S in shared memory
(~72 KB/block → only 1 block/SM, kills latency hiding) and does a serial per-row softmax. M2 = the standard FA2
optimizations: keep O in REGISTERS (WMMA accumulator fragments held across the whole K-loop, not smem round-tripped —
the tricky part is the per-row `corr` rescale of fragment elements), shrink smem for ≥2 blocks/SM, `cp.async`
double-buffer K/V, parallelize the softmax. Correctness oracle: the current M1 kernel + the materialized path.

## M2 attempt 2026-07-04: occupancy insufficient — kernel needs deep rework to beat cublas
Shrank tiles BR=64/BC=32→32/16 (72KB→34KB smem, 1→2 blocks/SM). Result: 54.7s→52.2s (~5% only). Still CORRECT.
**Honest conclusion:** my hand-written WMMA flash kernel is ~21.7ms/attn vs ~13ms for the materialized+cublas+F16
path — i.e. SLOWER, and ~35× off the theoretical TF32 optimum. Occupancy is not the bottleneck; the kernel is just
far less efficient than cublas (tiny 16×16 tiles, SERIAL per-row softmax, smem-O round-trip per K-step, per-K-step
`__syncthreads`). Beating cublas+F16-materialized needs a genuinely competitive kernel (parallel warp-reduce softmax,
register-resident O accumulator, larger tiles, cp.async, warp specialization) — a multi-day expert effort, not a
tweak spot. **Pragmatic recommendation: keep the materialized+F16 path (the shipped Wan 2.85×) as the default; leave
this correct M1/M2 kernel as a documented WIP behind `HARTSY_SDPA_V2` (off by default).** The materialized path with
F16 scores (allowF16) is already a good attention path; the bigger wins are elsewhere (per-arch host-loop ports, etc.).
