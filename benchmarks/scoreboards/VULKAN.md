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

## What this does NOT show yet

- No apples-to-apples full-pipeline number (one DiT block / one LLM decode step) against CUDA — see
  `VulkanLinearProfileMeasurement.Measure_LlmDecodeStep_ResidencyVsDispatchOverhead` for a Vulkan-only
  synthetic-decode-step measurement (2.58 ms/step, 5.0 D2H syncs/step, 10.0 transfer-cache misses/step —
  every one a CPU-loop-default `IBackend` member: `SliceLastDim`/`ApplyRope`/`KvCacheAppend`/`CopyTo`/
  `GluActivate`'s internal `SliceLastDim`). No CUDA equivalent of that synthetic step was run for
  comparison in this pass.
- `SdpaGpuBenchmarks`'s largest shape (B=1,H=24,S=16384,D=128 — video-DiT scale) was intentionally not
  run on Vulkan in this pass: at that size the naive materialized 3-pass SDPA would need to allocate a
  ~25 GB score matrix (`B·H·Sq·Skv·4` bytes), which is the same class of failure as the documented
  Wan-video full-resolution OOM. Not measured here; tracked as a Phase 4 (`sdpa_flash`) correctness
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
