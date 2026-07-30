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
