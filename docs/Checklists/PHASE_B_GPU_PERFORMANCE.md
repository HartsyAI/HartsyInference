# Phase B — GPU Performance Optimization

> **Goal**: HartsyInference CUDA backend within 1.0–1.5× of PyTorch + diffusers per kernel; SDXL 1024² @ 20 steps ≤ 5 s/step on RTX 3060.
> **Output**: a peer-reviewable systems paper + a reproducible benchmark harness.
> **Approach**: measure first, profile second, optimize third. Every change validated for correctness *and* significance.

Strategic plan: [`docs/Research/CUDA_PERFORMANCE_PLAN.md`](../Research/CUDA_PERFORMANCE_PLAN.md). Profiling commands: [`docs/Research/PROFILING_METHODOLOGY.md`](../Research/PROFILING_METHODOLOGY.md). Paper skeleton: [`docs/Research/TECH_PAPER_OUTLINE.md`](../Research/TECH_PAPER_OUTLINE.md).

---

## B0 — Planning & Documentation

- [x] [`CUDA_PERFORMANCE_PLAN.md`](../Research/CUDA_PERFORMANCE_PLAN.md) — master plan, methodology, hypotheses
- [x] [`PROFILING_METHODOLOGY.md`](../Research/PROFILING_METHODOLOGY.md) — Nsight + NVTX commands, statistical analysis
- [x] [`PHASE_B_GPU_PERFORMANCE.md`](PHASE_B_GPU_PERFORMANCE.md) — this file
- [x] [`TECH_PAPER_OUTLINE.md`](../Research/TECH_PAPER_OUTLINE.md) — paper skeleton

## B1 — Benchmark Infrastructure

### B1.1 — C# GPU Microbenchmark Project

- [ ] [`benchmarks/HartsyInference.GpuBenchmarks/HartsyInference.GpuBenchmarks.csproj`](../../benchmarks/HartsyInference.GpuBenchmarks/HartsyInference.GpuBenchmarks.csproj) — net8.0+net10.0 BenchmarkDotNet project
- [ ] `BenchmarkConfig.cs` — common config (1 warmup, 5 trials, JSON+Markdown export, `--artifacts` dir)
- [ ] `MatMulGpuBenchmarks.cs` — F16 / F32 / BF16 cuBLAS at all GEMM shapes hit by SDXL / Flux / SD3.5 / Z-Image / Flux2
- [ ] `Conv2DGpuBenchmarks.cs` — VAE + UNet conv shapes, 1×1 + 3×3 + 3×3-stride-2 variants
- [ ] `GroupNormGpuBenchmarks.cs` — UNet/VAE GroupNorm shapes (32-group typical)
- [ ] `LayerNormGpuBenchmarks.cs` — DiT LayerNorm shapes
- [ ] `RmsNormGpuBenchmarks.cs` — Flux/SD3 RmsNorm shapes
- [ ] `SdpaGpuBenchmarks.cs` — parameterized over (Sq, Skv, head_dim, num_heads, dtype). Self + cross-attn shapes from every model.
- [ ] `ElementwiseGpuBenchmarks.cs` — Silu, Gelu, BroadcastAdd, Add, Mul, Scale
- [ ] `MemoryAllocFreeBenchmarks.cs` — `cuMemAlloc` vs `cuMemAllocAsync` vs (future) pool, at common size classes
- [ ] `Program.cs` — entry point, captures hardware/software fingerprint into the artifact dir before BenchmarkDotNet runs

### B1.2 — Python PyTorch Baseline

- [ ] [`benchmarks/python-baseline/requirements.txt`](../../benchmarks/python-baseline/requirements.txt) — pinned per master plan
- [ ] `bench_pytorch_matmul.py` — same shape grid as `MatMulGpuBenchmarks`
- [ ] `bench_pytorch_conv2d.py` — same shape grid, with both `cudnn.benchmark=False` (fair) and `True` (best-case) variants
- [ ] `bench_pytorch_groupnorm.py`, `bench_pytorch_layernorm.py`, `bench_pytorch_rmsnorm.py`
- [ ] `bench_pytorch_sdpa.py` — uses `torch.nn.functional.scaled_dot_product_attention` (which dispatches to FlashAttention when available); also captures separate xformers + naive variants
- [ ] `bench_pytorch_elementwise.py`
- [ ] `bench_pytorch_e2e.py` — diffusers SDXL / Flux Dev FP8 / SD3.5 Medium / Z-Image Turbo with matched seed=42, prompt "A photograph of an astronaut riding a horse", per-step timing
- [ ] `run_all.sh` — orchestrates the above into a single `output_dir/microbench.csv` + `output_dir/e2e_*.csv`
- [ ] `_common.py` — shared utilities (CSV writer, hardware fingerprint, GPU memory polling, statistical config)

### B1.3 — NVTX + In-Process Profiling

- [ ] [`src/HartsyInference.Cuda/Profiling/NvtxRange.cs`](../../src/HartsyInference.Cuda/Profiling/NvtxRange.cs) — IDisposable wrapper around `nvtxRangePushA` / `nvtxRangePop`
- [ ] [`src/HartsyInference.Cuda/Profiling/NvtxApi.cs`](../../src/HartsyInference.Cuda/Profiling/NvtxApi.cs) — P/Invoke for `libnvToolsExt.so.1` / `nvToolsExt64_1.dll`
- [ ] `CudaBackend` — wrap pipeline-phase entry points (`MatMul`, `Linear`, `Conv2D`, `ScaledDotProductAttention`, etc.) with NVTX ranges. Per-op granularity gated behind the `HARTSYINFERENCE_NVTX_DETAILED` env var to avoid 4 000+ ranges/step polluting the steady-state timeline.
- [ ] [`src/HartsyInference.Cuda/Profiling/CudaProfilerControl.cs`](../../src/HartsyInference.Cuda/Profiling/CudaProfilerControl.cs) — wrappers around `cuProfilerStart` / `cuProfilerStop` so `nsys --capture-range=cudaProfilerApi` can scope to the steady-state denoise window only

### B1.4 — Run Scripts + Result Layout

- [ ] [`benchmarks/run_benchmarks.sh`](../../benchmarks/run_benchmarks.sh) — one-command harness:
    1. Capture hardware + software fingerprint
    2. Run C# microbench suite
    3. Run Python baseline
    4. Background-poll `nvidia-smi` for VRAM/util/temp
    5. Join CSVs via `analyze.py`, write `comparison.{md,csv}`
    6. Atomically move staging dir to `benchmarks/results/run_{utc-iso8601}_{gpu-slug}/`
- [ ] [`benchmarks/profile.sh`](../../benchmarks/profile.sh) — Nsight Systems wrapper, runs SDXL once with `nsys profile --trace=cuda,nvtx,osrt,cublas,cudnn`
- [ ] [`benchmarks/profile_kernel.sh`](../../benchmarks/profile_kernel.sh) — Nsight Compute wrapper for top-N kernel deep dive (parameterized by `--kernel-name regex:...`)
- [ ] [`benchmarks/analyze.py`](../../benchmarks/analyze.py) — Welch's t-test + 95 % CI calculation, comparison.md generation
- [ ] [`benchmarks/results/.gitkeep`](../../benchmarks/results/.gitkeep) + [`benchmarks/results/README.md`](../../benchmarks/results/README.md) — committed dir with the result-naming convention documented
- [ ] [`benchmarks/README.md`](../../benchmarks/README.md) — top-level: how to run, where results go, schema reference

### B1.5 — Verification (no perf changes yet)

- [ ] `dotnet build benchmarks/HartsyInference.GpuBenchmarks/HartsyInference.GpuBenchmarks.csproj` succeeds
- [ ] `dotnet run -c Release --project benchmarks/HartsyInference.GpuBenchmarks -- --filter '*MatMulGpu*' --runOncePerIteration` completes (smoke test the harness on whatever GPU is available)
- [ ] `python3 -m pip install -r benchmarks/python-baseline/requirements.txt` succeeds in a fresh venv
- [ ] `python3 benchmarks/python-baseline/bench_pytorch_matmul.py --device cuda --trials 1 --output /tmp/smoke.csv` completes (smoke test the Python harness)
- [ ] `bash benchmarks/run_benchmarks.sh --smoke` (1 trial per kernel, 1 model) completes end-to-end

## B2 — Capture Baseline

For each target device, produce a complete `benchmarks/results/run_baseline_{date}_{gpu}/`. Treat this as the immutable reference for every later "X× faster" claim.

- [ ] **RTX 3060 (SM 8.6) baseline** — dev box; the primary reference
- [ ] **L40S (SM 8.9) baseline** — cloud (Lambda Labs / RunPod); exercises native FP8 GEMM
- [ ] **A100 40 GB (SM 8.0) baseline** — cloud; Ampere reference for paper
- [ ] **H100 80 GB (SM 9.0) baseline** — cloud; Hopper, exercises Tensor Core 4th-gen + WGMMA (relevant if FA3 is tackled)
- [ ] **`baseline_summary.md`** — top-level summary across all four devices, committed under `benchmarks/results/`

## B3 — Profile + Identify Real Bottlenecks

- [ ] Read each `nsys` trace, populate `bottlenecks_per_model.md` with top-5 hot kernels per (device × model) tuple
- [ ] Update the predicted-speedup table in [`CUDA_PERFORMANCE_PLAN.md`](../Research/CUDA_PERFORMANCE_PLAN.md) with measured kernel times
- [ ] Confirm or revise the priority order for B4. If anything beats SDPA as #1, reorder.

## B4 — Optimization Phases

Each subphase is its own deliverable with a benchmarks/results/run_post_{tag}_{gpu}/ directory.

### B4.1 — FlashAttention 2 PTX Kernel (highest predicted impact)

- [ ] [`native/cuda/attention/flash_attention_f16.cu`](../../native/cuda/attention/flash_attention_f16.cu) — FA2 with online softmax, B_r=B_c=64–128, wmma Tensor Core path for head_dim ∈ {64, 128}
- [ ] [`native/cuda/attention/flash_attention_f32.cu`](../../native/cuda/attention/flash_attention_f32.cu) — F32 reference path (slower, used for accuracy validation)
- [ ] [`native/cuda/attention/build.sh`](../../native/cuda/attention/build.sh) — `nvcc -ptx -arch=sm_70 ...` (and sm_80, sm_89 variants)
- [ ] [`src/HartsyInference.Cuda/Ptx/flash_attention_f16.ptx`](../../src/HartsyInference.Cuda/Ptx/flash_attention_f16.ptx) — compiled output (committed)
- [ ] [`src/HartsyInference.Cuda/CudaKernels.cs`](../../src/HartsyInference.Cuda/CudaKernels.cs) — `LaunchFlashAttentionF16` / `LaunchFlashAttentionF32`
- [ ] [`src/HartsyInference.Cuda/CudaBackend.cs`](../../src/HartsyInference.Cuda/CudaBackend.cs) — `ScaledDotProductAttention` dispatches to FA2 when shape fits the kernel; falls back to existing materialize-S path otherwise. Gated by `EnableFlashAttention` flag (default true once validated).
- [ ] Unit tests: [`tests/HartsyInference.Cuda.Tests/FlashAttentionTests.cs`](../../tests/HartsyInference.Cuda.Tests/FlashAttentionTests.cs) — accuracy vs the materialize-S reference (avg_err < 1e-3 in F16, < 1e-5 in F32)
- [ ] Python parity: `tests/python-reference/dump_sdpa_reference.py` — diffs C# FA2 against `F.scaled_dot_product_attention` at the model-relevant shapes
- [ ] Microbench: `SdpaGpuBenchmarks` reruns; speedup table updated
- [ ] E2E: SSIM tests pass; `e2e.csv` shows SDXL / Flux / SD3.5 / Z-Image step-time improvement
- [ ] Deviation entry added to [`PHASE_3_DEVIATIONS.md`](PHASE_3_DEVIATIONS.md) with before/after measurement
- [ ] `benchmarks/results/run_post_b41_{date}_{gpu}/` committed across all devices
- [ ] Welch's *t*-test confirms speedup at α = 0.01 on every benchmarked shape

### B4.2 — cuDNN Conv2D Winograd Path

- [ ] [`src/HartsyInference.Cuda/CudnnApi.cs`](../../src/HartsyInference.Cuda/CudnnApi.cs) — P/Invoke for `cudnnConvolutionForward`, `cudnnGetConvolutionForwardAlgorithm_v7`, descriptor management
- [ ] [`src/HartsyInference.Cuda/Cudnn/CudnnConv2D.cs`](../../src/HartsyInference.Cuda/Cudnn/CudnnConv2D.cs) — wrapper around the cuDNN forward conv path
- [ ] `CudaBackend.Conv2D` — dispatches to cuDNN Winograd when (kernel == 3×3, stride == 1, padding == 1, in_channels >= 32, dilation == 1); falls back to im2col otherwise. Gated by `EnableCudnnConv2D` flag.
- [ ] Algorithm cache — heuristic-driven selection (cuDNN's `IMPLICIT_GEMM` vs `WINOGRAD` vs `WINOGRAD_NONFUSED`) memoized per (input shape, kernel shape) tuple
- [ ] Workspace allocator — cuDNN needs scratch memory; size queried via `cudnnGetConvolutionForwardWorkspaceSize`, allocated lazily, reused across calls
- [ ] Unit tests: accuracy vs im2col path (avg_err < 1e-4)
- [ ] Microbench: `Conv2DGpuBenchmarks`
- [ ] E2E: SDXL UNet + VAE step time
- [ ] Deviation entry, results dir, t-test

### B4.3 — Kernel Fusion

- [ ] PTX kernels:
    - `linear_bias_silu_f16.cu` / `.ptx` — Linear + bias + SiLU in one launch
    - `linear_bias_gelu_f16.cu` / `.ptx`
    - `conv2d_bias_silu_f16.cu` / `.ptx` (only for the small fused-conv shapes; large convs stay on cuDNN path from B4.2)
    - `rmsnorm_modulate_f16.cu` / `.ptx` — RmsNorm + (1+scale) multiply (AdaLN-Zero pattern)
- [ ] `IBackend` additions: `LinearBiasSilu`, `LinearBiasGelu`, `RmsNormModulate` with default fallback to current 2-step
- [ ] CudaBackend implementations
- [ ] CpuBackend reference implementations (so default IBackend impl works on CPU too)
- [ ] Model code adoption — SDXL / Flux / SD3.5 / Z-Image / Lumina2 / Hunyuan / Qwen / etc. blocks call the fused ops
- [ ] Unit tests: fused output bit-identical (within ULP) to unfused
- [ ] Microbench shows reduced kernel-launch count in NVTX timeline
- [ ] E2E speedup measured
- [ ] Deviation entry, results dir, t-test

### B4.4 — Activation Memory Pool

- [ ] [`src/HartsyInference.Cuda/Memory/CudaMemoryPool.cs`](../../src/HartsyInference.Cuda/Memory/CudaMemoryPool.cs) — size-class buckets, free-list per bucket, `Acquire(size)` / `Release(ptr, size)` API
- [ ] `GpuTransferHelper` — routes `AllocateDevice` / `FreeAsync` through the pool
- [ ] Pool size cap (e.g. 25 % of total VRAM); evict-on-full
- [ ] No leaks under stress (1000-step generation finishes with same VRAM as start)
- [ ] Microbench: `MemoryAllocFreeBenchmarks` shows < 5 µs per acquire/release
- [ ] E2E: per-step variance reduced (memory alloc was a source of jitter)
- [ ] Deviation entry, results dir

### B4.5 — CUDA Graphs for the Denoise Step

> **UPDATE 2026-07-04:** the `CudaGraph` wrapper (`src/HartsyInference.Cuda/CudaGraph.cs`, Capture/TryUpdate/Launch) is now **verified working on-GPU** — was untested; `CudaBackend.GraphSmokeTest()` (CLI `hartsyinference-textgen graphtest`) captures a Scale on a stable buffer, replays with changed input → PASS. So the API below (item 1) effectively exists (`CudaGraph`) and the foundation is proven; the async-pool memory model is capture-compatible. The **same device-side-scalar requirement** (item 2) was hit and fully mapped for the LLM decode step — see [LLM_DECODE_PERF_GRIND.md](LLM_DECODE_PERF_GRIND.md) Phase 6 (precompute-table RoPE, device position counter for attention/KV, device token/embed). The denoise step needs the analogous device-side timestep/sigma. Foundation ✅; the device-resident conversion (LLM or denoise) is the remaining build.

- [x] `CudaGraph.Capture()` / `.Launch()` / `.TryUpdate()` API — exists + **verified** (`CudaGraph.cs`, `GraphSmokeTest`)
- [ ] Refactor scheduler step to use device-side scalars (timestep, sigma) so the captured graph is parameter-stable
- [ ] First step runs in capture mode; steps 2..N run via `cuGraphLaunch`
- [ ] Re-capture trigger when shape changes (new resolution, different model)
- [ ] Validation: bit-identical output to non-graph path
- [ ] Microbench: kernel-launch overhead (per-launch CPU time) reduced
- [ ] E2E: speedup proportional to launch count × launch overhead
- [ ] Deviation entry, results dir

### B4.6 — F16 / BF16 Audit + Tensor Core Sweep

- [ ] Find every `DType.F32` promotion in the CudaBackend hot path
- [ ] For each, decide: legitimate (numerical stability needs F32 accumulator) vs vestigial (could stay F16)
- [ ] Promotions that can drop: change to F16 with F32 accumulator inside the kernel
- [ ] Add F16 paths to ops that don't have one yet (audit complete via grep)
- [ ] Tensor Core utilization measured via Nsight Compute on each updated op
- [ ] SSIM regression suite passes on every model
- [ ] Deviation entry, results dir

### B4.7 — Optional: FlashAttention 3 / WGMMA on Hopper

(Stretch goal — only if a Hopper GPU is reliably available and B4.1–B4.6 don't already meet the acceptance criteria)

- [ ] FA3 kernel (`flash_attention_f16_sm90a.cu`) targeting `sm_90a` with TMA + warp specialization
- [ ] Per-device dispatch in `CudaBackend.ScaledDotProductAttention`: SM 9.0+ → FA3, SM 8.0–8.9 → FA2
- [ ] H100 microbench + E2E + deviation entry

## B5 — Final Validation + Tech Paper

- [ ] Re-run full harness across all target devices → `benchmarks/results/run_final_{date}_{gpu}/`
- [ ] [`benchmarks/results/final_report.md`](../../benchmarks/results/final_report.md) — cross-device summary, before/after table per kernel, end-to-end SDXL/Flux/SD3.5/Z-Image numbers
- [ ] [`docs/Research/TECH_PAPER_OUTLINE.md`](../Research/TECH_PAPER_OUTLINE.md) — every section populated with real figures
- [ ] [`paper/`](../../paper/) — generated LaTeX from the outline + figure CSVs
- [ ] Tag commit `phase-b-complete`
- [ ] Deviation entries in [`PHASE_3_DEVIATIONS.md`](PHASE_3_DEVIATIONS.md) for every kernel change

---

## Acceptance Criteria

The phase is complete only when **every** condition holds:

1. ✅ All existing Generation / SSIM / unit tests pass (no numerical regression)
2. ✅ Each microbench within 1.5× of PyTorch on the same shape; top-5 hot kernels within 1.0–1.2×
3. ✅ SDXL 1024² @ 20 steps ≤ 5 s/step on RTX 3060
4. ✅ `bash benchmarks/run_benchmarks.sh` reproduces our results on any CUDA box with the pinned stack
5. ✅ Cross-device coverage: ≥ 3 SM generations measured (Ampere / Ada / Hopper)
6. ✅ Each kernel change has a deviation entry with before/after numbers + accuracy metric
7. ✅ Paper outline fully populated; LaTeX builds; figures + tables sourced from committed CSVs
8. ✅ Welch's t-test (α = 0.01) confirms every claimed speedup

---

## Live Status

| Date (UTC) | Subphase | What changed |
|---|---|---|
| 2026-05-07 | B0 | Plan approved; planning docs written |
