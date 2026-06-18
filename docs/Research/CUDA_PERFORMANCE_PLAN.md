# CUDA Performance Plan — Master Document

> **Phase**: B (GPU performance optimization)
> **Goal**: HartsyInference C# CUDA backend within 1.0–1.5× of PyTorch + diffusers per kernel; end-to-end SDXL 1024² @ 20 steps in ≤ 5 s/step on RTX 3060.
> **Output**: a peer-reviewable systems paper (target: MLSys / EuroSys / SC) describing the methodology, kernels, and measurement framework.
> **Constraint**: every claim is grounded in a CSV under [`benchmarks/results/`](../../benchmarks/results/). No marketing numbers.

This is the master tracking document for Phase B. It supersedes the optimization roadmap in [`CUDA_PERFORMANCE.md`](CUDA_PERFORMANCE.md) (the older doc remains as the historical Phase 0–2 record). Companion docs:

- [`PROFILING_METHODOLOGY.md`](PROFILING_METHODOLOGY.md) — how to profile (Nsight Systems / Compute, NVTX, cuBLAS log)
- [`../Checklists/PHASE_B_GPU_PERFORMANCE.md`](../Checklists/PHASE_B_GPU_PERFORMANCE.md) — checkbox tracking
- [`TECH_PAPER_OUTLINE.md`](TECH_PAPER_OUTLINE.md) — paper skeleton, populated as we go
- [`FLASH_ATTENTION.md`](FLASH_ATTENTION.md) — algorithm reference (existing, unchanged)
- [`BENCHMARKING.md`](BENCHMARKING.md) — earlier procedural notes (existing, superseded by this doc + the run scripts under `benchmarks/`)

### Technique survey docs (2026-06-16 literature sweep)

A four-axis research sweep across algorithmic, quantization, kernel, and memory/serving techniques (2023-2026), each scoped to what a custom C# + PTX + cuBLAS/cuBLASLt engine can actually adopt. These feed the B4 optimization phases below.

- [`STEP_ACCELERATION.md`](STEP_ACCELERATION.md) — algorithmic / step-level: step distillation (LCM/Turbo/DMD2/Hyper-SD/PCM/TCD), training-free feature caching (DeepCache/TeaCache/FBCache), CFG distillation, solver efficiency, video/audio/3D specifics. Highest single-model leverage (~2-10x).
- [`QUANTIZATION_LOW_PRECISION_INFERENCE.md`](QUANTIZATION_LOW_PRECISION_INFERENCE.md) — PTQ method families, GGUF/NF4 weight-only, FP8/cuBLASLt mechanics, SVDQuant/NVFP4/MXFP4, INT8/INT4 GEMM, SageAttention. Ampere-first ranking (INT8 is the only native low-precision compute on the 3060).
- [`DEEP_KERNEL_OPTIMIZATION.md`](DEEP_KERNEL_OPTIMIZATION.md) — `mma.sync`/`ldmatrix`/`cp.async` tensor-core PTX, FA2-vs-FA3, Hopper TMA/wgmma/clusters, CUDA Graphs via Driver API, cuBLASLt epilogue fusion, megakernels, occupancy micro-opts, implicit-GEMM/Winograd conv.
- [`MEMORY_SCHEDULING_SERVING.md`](MEMORY_SCHEDULING_SERVING.md) — stream-ordered + VMM + caching allocators, pinned/async transfer overlap, block-swap weight streaming (the key 12 GB enabler), VAE tiling, multi-stream/CFG batching, feature caching + video AR KV-cache.

---

## Methodology

Peer-review-grade rigor was an explicit user requirement. The framework targets the strictest interpretation of "reproducible systems experiment":

1. **Hardware fingerprinting**: every run records `nvidia-smi -q`, `nvcc --version`, `lscpu`, `uname -a`, kernel revision, driver version, ECC mode, persistence-mode state, and PCIe topology. Pinned in the result CSV.
2. **Software fingerprinting**: `dotnet --info`, all NuGet versions, exact PyTorch/CUDA/diffusers/transformers/xformers/accelerate versions (locked — see § Pinned Software Stack), exact PTX file SHA-256 digests, exact safetensors checkpoint SHA-256.
3. **Multiple trials**: each microbenchmark runs N=5 trials after a 1-trial warmup. Reports report mean, stddev, 95 % confidence interval (Student-*t*, df=4). Outliers (> 3σ) are flagged but not silently discarded.
4. **Cold vs warm**: explicit categorization. "Cold" = first run after process start (includes JIT, first kernel launch, page-pinning costs). "Warm" = post-warmup steady-state. Separate columns in the CSV.
5. **Same shapes, same seeds, same prompts**: C# microbenchmarks and Python baselines drive the same shape grid, the same fixed random seeds (NumPy + PyTorch + .NET RNGs all initialized identically), and the same prompts. End-to-end runs share `seed=42` and the prompt `"A photograph of an astronaut riding a horse"` (matches every existing SSIM test in the repo).
6. **Cloud cross-device coverage**: validation across at minimum {RTX 3060 (SM 8.6, dev box), L40S or RTX 4090 (SM 8.9, Ada — exercises FP8 GEMM path), A100 (SM 8.0, Ampere baseline), H100 (SM 9.0, Hopper — exercises WGMMA + FA3 if we go that far)}. Each device gets its own `benchmarks/results/run_{date}_{gpu}/` directory.
7. **Statistical significance gate**: a "speedup" claim requires (a) the new mean is outside the old 95 % CI, AND (b) a Welch's *t*-test rejects μ_new = μ_old at α = 0.01. This filters out noise-level changes that look like wins.
8. **Numerical correctness gate**: every kernel change must pass an avg_err / max_err threshold against the unfused / unoptimized reference *and* against PyTorch on the same input. If the optimization is bit-exact (e.g. removing dead code), assert exact equality.

---

## Pinned Software Stack

Locked at plan-approval time (2026-05-07). Any later upgrade requires re-running the full baseline and writing a follow-up "ABI shift" deviation entry.

```toml
# Python side — benchmarks/python-baseline/requirements.txt
torch==2.5.1+cu124
torchvision==0.20.1+cu124
diffusers==0.32.1
transformers==4.46.3
accelerate==1.2.1
xformers==0.0.28.post3
safetensors==0.4.5
sentencepiece==0.2.0
numpy==1.26.4
psutil==6.1.0
nvidia-ml-py==12.560.30  # for per-GPU stats inside the script
```

```ini
# CUDA side
CUDA toolkit:  12.4.x  (matches PyTorch 2.5.1 wheel)
Driver:        ≥ 535.x (R535+ exposes cuBLASLt FP8)
nvcc:          12.4.x  (PTX target sm_70 for portability; sm_89 / sm_90 builds for native FP8 / WGMMA)
cuDNN:         9.x      (when wired in B4.2; gated to that phase)
```

```ini
# .NET side
SDK:           .NET 10  (also tested on .NET 8 LTS for matrix coverage)
BenchmarkDotNet: 0.14.0  (already pinned in csproj)
```

**Why these exact versions**: PyTorch 2.5.1 + CUDA 12.4 is what ComfyUI ships in its bundled installer as of late 2025 / early 2026. diffusers 0.32.1 is the latest stable that supports the model set in this repo (Flux, Flux.2, SD3.5, Z-Image, Lumina2, Hunyuan Image, Qwen-Image). xformers 0.0.28.post3 is the matched-ABI build for that PyTorch.

---

## Current Baseline (placeholder — to be filled by B2)

| Hardware | SDXL 1024² | Flux Dev FP8 512² | SD3.5 Med 512² | Z-Image Turbo 512² | Notes |
|---|---|---|---|---|---|
| RTX 3060 12 GB | TBD | TBD | TBD | TBD | Dev box |
| L40S 48 GB (cloud) | TBD | TBD | TBD | TBD | Ada — FP8 native path |
| A100 40 GB (cloud) | TBD | TBD | TBD | TBD | Ampere reference |
| H100 80 GB (cloud) | TBD | TBD | TBD | TBD | Hopper — Tensor Core 4th-gen |

PyTorch reference column added once `benchmarks/python-baseline/bench_pytorch_e2e.py` runs. The 53 s/step number from [`CUDA_PERFORMANCE.md`](CUDA_PERFORMANCE.md) is *stale* (Phase 2 closeout, April 2026 — predates Phase 4 model breadth). Treat as historical only.

---

## Per-Kernel Predicted Speedup Table

Updated as B2 produces real numbers. The "predicted" column is from [`BENCHMARKING.md`](BENCHMARKING.md) gap analysis; the "measured" column is filled after each optimization phase lands.

| Kernel / Subsystem | Current impl | Predicted speedup | Measured (post-fix) | Hypothesis test |
|---|---|---|---|---|
| Self-attention SDPA | Materialize full S matrix via cuBLAS GEMM + softmax (vanilla) | 2–3× (FA2 tiled) | TBD | Memory-bound at large Sq |
| 3×3 Conv2D (stride 1) | im2col + cuBLAS SGEMM | 2–3× (cuDNN Winograd) | TBD | Compute-bound; im2col allocation overhead |
| Linear + bias + activation | 2 separate kernels (Linear, BiasAdd / Add) | 1.3–1.5× (fused) | TBD | Launch overhead + HBM round-trips |
| Conv2D + bias + SiLU | 3 separate kernels | 1.5–2× (fused) | TBD | Same as Linear+bias+act, larger gain |
| RMSNorm + AdaLN modulate | `RmsNorm` then CPU-side `(1+scale)` multiply | 1.5–2× (fused PTX) | TBD | Currently does CPU-side multiply per token |
| Memory alloc/free | `cuMemFreeAsync` per op | 1.2–1.5× (size-class pool) | TBD | ~50 µs per alloc × 4300 launches/step |
| Per-step kernel launch | 4300+ individual launches/step | 1.2–1.5× (CUDA Graph) | TBD | Step is identical structure 1..N |
| FP16 throughout | Mixed F16 / F32 today | 1.5–2× | TBD | Tensor Core throughput |
| Native FP8 GEMM (Ada+) | Cast-to-F16 fallback on Ampere | 1.6–2× on SM 8.9+ | TBD | Already wired, untested |

**Stack target**: conservatively 10–15× → SDXL 1024² @ 20 steps from ~53 s/step to ~3.5–5.3 s/step on RTX 3060 (within 1–2× of PyTorch's ~3 s/step).

---

## Phased Execution

### B0 — Planning & Documentation (this phase)

Deliverables (one-shot, no code):
- [x] [`CUDA_PERFORMANCE_PLAN.md`](CUDA_PERFORMANCE_PLAN.md) — this doc
- [x] [`PROFILING_METHODOLOGY.md`](PROFILING_METHODOLOGY.md) — Nsight + NVTX commands, reproducibility
- [x] [`../Checklists/PHASE_B_GPU_PERFORMANCE.md`](../Checklists/PHASE_B_GPU_PERFORMANCE.md) — checkbox tracking
- [x] [`TECH_PAPER_OUTLINE.md`](TECH_PAPER_OUTLINE.md) — paper skeleton

### B1 — Benchmark Infrastructure (no perf changes)

Build the harness *before* changing kernels. The cardinal rule: measure first.

1. [`benchmarks/HartsyInference.GpuBenchmarks/`](../../benchmarks/HartsyInference.GpuBenchmarks/) — BenchmarkDotNet GPU project
   - `MatMulGpuBenchmarks` — F16/F32/BF16 cuBLAS at GEMM shapes from each model
   - `Conv2DGpuBenchmarks` — VAE + UNet conv shapes
   - `GroupNormGpuBenchmarks`, `LayerNormGpuBenchmarks`, `RmsNormGpuBenchmarks`
   - `SdpaGpuBenchmarks` — parameterized over (Sq, Skv, head_dim, n_heads)
   - `ElementwiseGpuBenchmarks` — Silu, Gelu, BroadcastAdd
   - `MemoryAllocFreeBenchmarks` — `cuMemAlloc` vs `cuMemAllocAsync` vs pool
2. [`benchmarks/python-baseline/`](../../benchmarks/python-baseline/) — pinned PyTorch parity
   - `bench_pytorch_matmul.py`, `bench_pytorch_conv2d.py`, `bench_pytorch_sdpa.py`, etc.
   - `bench_pytorch_e2e.py` — diffusers SDXL / Flux / SD3.5 with matched seeds
   - `requirements.txt` (pinned per § above), `run_all.sh`
3. [`benchmarks/run_benchmarks.sh`](../../benchmarks/run_benchmarks.sh) — master harness
   - Captures hardware + software fingerprint
   - Runs C# + Python halves
   - N=5 trials, joins CSVs, emits `results/run_{utc-iso8601}_{gpu-slug}/comparison.{md,csv}`
   - Pulls peak VRAM via background `nvidia-smi` polling
4. [`benchmarks/profile.sh`](../../benchmarks/profile.sh) — Nsight Systems wrapper around a single SDXL run
5. NVTX annotations in `CudaBackend` — wrap each `IBackend` op with `cuNvtxRangePush/Pop` so Nsight timelines are readable

### B2 — Capture Baseline (no perf changes, dependency on B1)

Run the full harness across cloud devices. Each device produces a numbered, signed `benchmarks/results/run_baseline_{date}_{gpu}/` directory with:
- `hardware.txt`, `software.txt`, `digests.txt` (PTX + checkpoint hashes)
- `microbench.csv` (per-kernel per-shape numbers)
- `e2e.csv` (per-model end-to-end numbers)
- `comparison.md` (human-readable summary, generated)
- `nsys/sdxl_1024_step.qdrep` (one Nsight trace per model)

Accept criteria for "baseline complete": each target device has a directory; `comparison.md` is reviewable.

### B3 — Profile + Identify Real Bottlenecks

Read the Nsight `.qdrep` traces and the per-kernel CSV. Update the predicted-speedup table above with measured numbers. Identify the top-3 hot kernels per model — these become the targets for B4.

Guard against confirmation bias: if the actual hot kernel doesn't match the predicted list (e.g. SDPA isn't actually #1), update the priority order accordingly.

### B4 — Optimization Phases (priority order)

Each B4.x is a self-contained subphase: benchmarks before, implementation, benchmarks after, accuracy validation, deviation entry, results commit.

- **B4.1 — FlashAttention 2 PTX kernel** — biggest single predicted gain. Diffusion is non-causal/fixed-seqlen, so one kernel parameterized by `(seqlen_q, seqlen_kv, head_dim)` covers self + cross attention (drops all mask/varlen machinery). See [`DEEP_KERNEL_OPTIMIZATION.md`](DEEP_KERNEL_OPTIMIZATION.md) §2.
- **B4.2 — Conv2D cuDNN Winograd path** — second-biggest gain on conv-heavy models (SDXL VAE). Survey note: prefer hand-rolled implicit-GEMM (drop im2col materialization) + Winograd F(2x2,3x3) in PTX over a cuDNN runtime dependency (no-native-libs pillar); [`DEEP_KERNEL_OPTIMIZATION.md`](DEEP_KERNEL_OPTIMIZATION.md) §7.
- **B4.3 — Kernel fusion** — Conv2D+bias+act, Linear+bias+act, RMSNorm+modulate. **cuBLASLt epilogue fusion (bias+GELU/ReLU) is the lowest-effort first move** (descriptor P/Invoke, no new PTX); fused AdaLN/modulation is the highest-value diffusion-specific hand-rolled fusion. [`DEEP_KERNEL_OPTIMIZATION.md`](DEEP_KERNEL_OPTIMIZATION.md) §5.
- **B4.4 — Memory pool** — eliminate per-op alloc/free. Survey path: stream-ordered pool (`cuMemAllocAsync` + capped `RELEASE_THRESHOLD`) first, then a size-class caching suballocator; VMM expandable-segments only if fragmentation OOMs appear. [`MEMORY_SCHEDULING_SERVING.md`](MEMORY_SCHEDULING_SERVING.md) §1-3.
- **B4.5 — CUDA Graphs** — capture step, replay. Patch per-step scalars via `cuGraphExecKernelNodeSetParams` (never re-instantiate); bucket by shape; allocate outside capture. [`DEEP_KERNEL_OPTIMIZATION.md`](DEEP_KERNEL_OPTIMIZATION.md) §4.
- **B4.6 — F16/BF16 audit + Tensor Core utilization sweep** — depends on a `mma.sync` + `ldmatrix` + `cp.async` + XOR-swizzle GEMM core (the dependency for everything fast); INT8 IMMA is the only native low-precision compute on the 3060. [`QUANTIZATION_LOW_PRECISION_INFERENCE.md`](QUANTIZATION_LOW_PRECISION_INFERENCE.md).

Survey-informed candidate subphases (new; prioritize against B3 profiling before committing):

- **B4.7 — Block-swap weight streaming** — pinned double-buffer + transfer stream + event-gated prefetch over the existing weight cache. The enabler for unquantized Flux 12B / video models on 12 GB; hides ~all PCIe latency when per-block compute > per-block transfer. Highest impact for the dev box. [`MEMORY_SCHEDULING_SERVING.md`](MEMORY_SCHEDULING_SERVING.md) §4-5.
- **B4.8 — Training-free step/feature acceleration** — guidance-distilled single-pass CFG, TeaCache/FBCache residual caching, LCM/TCD consistency schedulers, DPM-Solver++/UniPC. Pure loop/forward code or loadable weights, often 2-10x, orthogonal to kernel work. [`STEP_ACCELERATION.md`](STEP_ACCELERATION.md).
- **B4.9 — VAE tiling** — spatial (image) + temporal/spatial with causal feature cache (video). Removes the binding high-res/video activation peak (SD-VAE hits ~60 GB at 4k). Mostly host tile loops over existing kernels + a linear-blend pass. [`MEMORY_SCHEDULING_SERVING.md`](MEMORY_SCHEDULING_SERVING.md) §6.
- **B4.10 — Rolling KV-cache for AR world models** — block-causal mask + fixed-window K/V ring buffers for Matrix-Game/Oasis/CausVid; the O(TL)-vs-O(T^2) difference for interactive use. [`MEMORY_SCHEDULING_SERVING.md`](MEMORY_SCHEDULING_SERVING.md) §8.
- **B4.11 — L2 persistence + 128-bit vectorized loads** — cheap Driver-only / address-math micro-opts (~20% where hot data fits; 4x fewer load instructions). [`DEEP_KERNEL_OPTIMIZATION.md`](DEEP_KERNEL_OPTIMIZATION.md) §6.

### B5 — Final Validation + Tech Paper

1. Re-run full harness, populate "after" columns
2. Cross-device validation matrix (RTX 3060 / L40S / A100 / H100)
3. Generate paper from [`TECH_PAPER_OUTLINE.md`](TECH_PAPER_OUTLINE.md)
4. Tag commit, archive `benchmarks/results/run_final_{utc-iso}/`

---

## Acceptance Criteria

The phase is complete only when **every** condition holds:

1. **No regression**: every existing Generation / SSIM / unit test passes (numerical correctness preserved across all phases).
2. **Per-kernel parity with PyTorch**: each microbench shape is within 1.5× of PyTorch's matched call (median across N=5 trials, 95 % CI). Hot kernels (top-5 by total time) within 1.0–1.2×.
3. **End-to-end target met**: SDXL 1024² @ 20 steps ≤ 5 s/step on RTX 3060.
4. **Reproducible**: `bash benchmarks/run_benchmarks.sh` on any CUDA box with the pinned stack produces a `results/run_*/comparison.md` directly comparable to ours.
5. **Cross-device coverage**: at least three SM generations measured (Ampere / Ada / Hopper).
6. **Documented**: every kernel change has a deviation entry in `PHASE_3_DEVIATIONS.md` with before/after numbers + accuracy metric. The paper skeleton has every section populated with real figures.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| FA2 kernel correctness — online softmax is famously easy to get subtly wrong | Cross-validate against PyTorch's `F.scaled_dot_product_attention` at multiple shapes; commit a unit test that diffs against a Python reference dump |
| cuDNN ABI churn between minor releases | Pin to exact cuDNN 9.x version; if cuDNN unavailable on a target device, fall back to im2col (already exists) |
| CUDA Graph capture failures on dynamic shapes (e.g. variable text seq length) | Document supported shape regimes; non-graphable paths fall through to non-graph dispatch |
| Cloud GPU cost overruns during cross-device validation | Spot instances; cache the safetensors on persistent volumes; baseline runs are O(minutes), not hours |
| PyTorch version drift breaking baselines mid-project | Software stack pinned (§ Pinned Software Stack); upgrades require explicit re-baseline + deviation entry |
| RTX 3060 (SM 8.6) doesn't exercise FP8 paths | Cloud Ada GPU (L40S / RTX 4090) for FP8 validation; document Ampere-only result rows clearly |
| Optimizations help one model but regress another | The harness exercises 4 models (SDXL, Flux Dev FP8, SD3.5, Z-Image); regression on any is a blocker |

---

## Hypotheses (to be tested)

These are the falsifiable claims the experiments are designed to confirm or reject. Each gets its own row in the final paper's results table.

- **H1**: A C# CUDA backend can match PyTorch + diffusers per-kernel within 1.5× without writing CUDA C++ (PTX-only path remains viable).
- **H2**: A native FA2 implementation in PTX provides ≥ 2× speedup over the materialize-S baseline on diffusion self-attention shapes (Sq ≥ 1024, Skv ≥ 1024) at FP16.
- **H3**: Kernel fusion (Conv2D+bias+act, Linear+bias+act, RMSNorm+modulate) provides ≥ 1.3× speedup on diffusion ResNet/AdaLN blocks at the per-block level.
- **H4**: CUDA Graphs reduce per-step launch overhead by ≥ 30 % on a 4 000+ launch-per-step workload at 1024² SDXL.
- **H5**: Native FP8 GEMM (cuBLASLt on SM ≥ 8.9) provides ≥ 1.6× speedup over cast-to-F16 on Flux Dev FP8 transformer Linear ops.
- **H6 (negative result, also valuable)**: Memory pooling alone provides < 1.1× speedup on a workload already using `cuMemFreeAsync`. (If true, deprioritize.)

---

## Out of Scope

- **CPU backend optimization** — `HartsyInference.Cpu` perf is tracked in `SIMD_INTRINSICS_DOTNET.md`; not part of this phase.
- **Vulkan backend** — `HartsyInference.Vulkan` is in `PHASE_3_5_VULKAN_BACKEND.md`; not Phase B.
- **Per-block weight streaming for huge models** — covered in `PHASE_4_MODEL_BREADTH.md` follow-ups (Flux.2 Dev, Hunyuan Image 2.1, Qwen-Image at 12 GB).
- **Pure inference compilation** (e.g. via TVM, Triton) — orthogonal track; would compete with the PTX kernels we're authoring.
- **Audio / vision modalities** — Phase 5 / 6.

These exclusions are intentional. The paper's contribution is the *measurement framework + the PTX kernel set*, not a kitchen-sink optimization pass.

---

## Status (live)

| Date (UTC) | Subphase | What changed |
|---|---|---|
| 2026-05-07 | B0 | Planning docs written and approved |
| 2026-06-16 | B0 | Four-axis technique survey added: STEP_ACCELERATION, QUANTIZATION_LOW_PRECISION_INFERENCE, DEEP_KERNEL_OPTIMIZATION, MEMORY_SCHEDULING_SERVING. B4 roadmap extended with survey-informed candidate subphases B4.7-B4.11 (block-swap streaming, step/feature acceleration, VAE tiling, rolling KV-cache, L2 persistence). No code changes. |
