# Profiling Methodology — How To Measure HartsyInference GPU Workloads

> **Purpose**: every measurement that lands in [`benchmarks/results/`](../../benchmarks/results/) is reproducible by someone with a CUDA box and the pinned software stack. This document is the recipe.

This is the operational companion to [`CUDA_PERFORMANCE_PLAN.md`](CUDA_PERFORMANCE_PLAN.md). Read that first for the strategic context (what we measure and why); read this for the exact commands.

---

## 1. Tools Used

| Tool | Purpose | Where it ships from |
|---|---|---|
| **BenchmarkDotNet 0.14.0** | C# microbenchmarks with proper warmup, statistical analysis, and result serialization | NuGet (already in csproj) |
| **NVIDIA Nsight Systems 2024.6+** | Whole-application timeline profiler (kernel timing, NVTX ranges, GPU/CPU correlation) | `apt install nsight-systems` or NVIDIA HPC SDK |
| **NVIDIA Nsight Compute 2024.3+** | Per-kernel deep dive (occupancy, memory throughput, warp stalls) | `apt install nsight-compute-202x` |
| **`nvidia-smi`** | Per-second VRAM, utilization, power, temp polling | NVIDIA driver |
| **`nvprof`** | Legacy per-kernel timing (still useful on older driver/toolkit pairs) | CUDA toolkit (deprecated, use `nsys` instead when available) |
| **`cuBLAS log`** (`CUBLAS_LOGINFO_DBG=1`) | Per-cuBLAS-call algorithm selection log | env var, no install needed |
| **PyTorch profiler** (`torch.profiler`) | Per-op timing for the Python baseline runs | pip (already in `requirements.txt`) |
| **`scipy.stats`** | Welch's *t*-test for significance gates | pip (`scipy==1.14.1`) |

---

## 2. Hardware Fingerprint (captured at every run)

Every `benchmarks/results/run_*/` directory must contain a `hardware.txt` produced by:

```bash
# Captured at the top of run_benchmarks.sh
{
    echo "## hostname"
    hostname
    echo "## uname"
    uname -a
    echo "## kernel"
    cat /proc/version
    echo "## CPU"
    lscpu | grep -E '^(Architecture|CPU\(s\)|Model name|CPU MHz|Cache)'
    echo "## RAM"
    free -h
    echo "## nvidia-smi -q (full)"
    nvidia-smi -q
    echo "## nvidia-smi --query-gpu (machine readable)"
    nvidia-smi --query-gpu=name,driver_version,vbios_version,compute_cap,memory.total,power.limit,clocks.max.sm,clocks.max.mem,persistence_mode,ecc.mode.current --format=csv
    echo "## PCIe topology"
    nvidia-smi topo --matrix 2>/dev/null || true
    echo "## CPU governor"
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "n/a"
} > hardware.txt
```

**Why every field**: GPU clocks, persistence mode, ECC state, CPU governor, and power limit all measurably affect kernel latency. A reviewer reproducing our numbers needs to know whether the test machine was in `performance` governor or default `powersave`.

---

## 3. Software Fingerprint

```bash
# Captured at the top of run_benchmarks.sh
{
    echo "## dotnet"
    dotnet --info
    echo "## CUDA toolkit"
    nvcc --version
    echo "## driver via nvidia-smi"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader
    echo "## python"
    python3 --version
    pip freeze | grep -E '^(torch|diffusers|transformers|accelerate|xformers|safetensors|sentencepiece|numpy|scipy|nvidia-)'
    echo "## git"
    git rev-parse HEAD
    git status --porcelain
    echo "## PTX SHA-256"
    find src/HartsyInference.Cuda/Ptx -name '*.ptx' | sort | xargs sha256sum
    echo "## checkpoint SHA-256 (if present)"
    find Models -name '*.safetensors' 2>/dev/null | sort | xargs -r sha256sum
} > software.txt
```

**Critical**: if `git status` reports any uncommitted files (other than `benchmarks/results/`), the run is invalid for paper purposes — record the diff or commit before measuring.

---

## 4. NVTX Annotations (in code)

For Nsight Systems timelines to be readable, the C# code annotates ranges. The pattern is:

```csharp
// Add to CudaBackend ops
using HartsyInference.Cuda.Profiling;

public unsafe void Linear(Tensor output, Tensor input, Tensor weight, Tensor? bias)
{
    using NvtxRange range = NvtxRange.Push("Linear");
    // ... existing code
}
```

Where `NvtxRange` is a tiny wrapper around `nvtxRangePushA` / `nvtxRangePop` from `libnvToolsExt.so` (P/Invoked, no native dependency beyond the driver). Implementation lives in [`src/HartsyInference.Cuda/Profiling/NvtxRange.cs`](../../src/HartsyInference.Cuda/Profiling/NvtxRange.cs) (added in B1).

**Granularity**: pipeline phases get NVTX ranges (e.g. "TextEncode", "DenoiseStep", "VaeDecode"); per-op ranges are gated behind a `#if NVTX_ENABLED` to avoid 4 000+ ranges/step polluting the timeline (turn on only when zoomed-in profiling is needed).

---

## 5. Microbenchmark Run

```bash
# Single GPU, full microbench suite, 5 trials each, JSON output
dotnet run -c Release --project benchmarks/HartsyInference.GpuBenchmarks -- \
    --filter '*' \
    --warmupCount 1 \
    --iterationCount 5 \
    --exporters json,markdown \
    --artifacts benchmarks/results/run_$(date -u +%Y-%m-%dT%H%M%SZ)_microbench
```

`--warmupCount 1` because BenchmarkDotNet's default warmup is too long for GPU work (it'd run the GEMM 5+ times before counting). `--iterationCount 5` matches the rigor commitment in the master plan.

---

## 6. End-to-End Run

End-to-end timing wraps an actual generation test with a stopwatch:

```bash
# Run a single SDXL test, capture per-step timings
HARTSYINFERENCE_BENCH_OUT=benchmarks/results/run_$(date -u +%Y-%m-%dT%H%M%SZ)_e2e/sdxl.csv \
dotnet test tests/HartsyInference.Diffusion.Tests/HartsyInference.Diffusion.Tests.csproj \
    --filter "FullyQualifiedName~Sdxl_GenerateImage_Gpu" \
    --logger "console;verbosity=detailed" \
    -- --runtime-mode benchmark
```

The pipelines already emit per-step timing via the `onProgress` callback. A small `BenchmarkProgressLogger` (added in B1) consumes that callback and writes the CSV.

---

## 7. Nsight Systems Capture

Single command, captures full timeline including NVTX ranges, CUDA API, kernel timing, and CPU sampling:

```bash
nsys profile \
    --output benchmarks/results/run_$(date -u +%Y-%m-%dT%H%M%SZ)_nsys/sdxl_1024 \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --sample=cpu \
    --capture-range=cudaProfilerApi \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    dotnet test tests/HartsyInference.Diffusion.Tests/HartsyInference.Diffusion.Tests.csproj \
        --filter "FullyQualifiedName~Sdxl_GenerateImage_Gpu" \
        -- --runtime-mode benchmark
```

`--capture-range=cudaProfilerApi` means we wrap the actual measurement window with `cudaProfilerStart()` / `cudaProfilerStop()` — captures the warmup-then-steady-state phase only, not test-runner startup. The C# code triggers these via `cuProfilerStart` / `cuProfilerStop` in `BenchmarkProgressLogger`.

After collection:

```bash
# Convert to readable summary
nsys stats --report cudaapisum,gpukernsum,nvtxsum benchmarks/results/.../sdxl_1024.nsys-rep
```

The `gpukernsum` report sorted by total time gives the top hot kernels — this is the input to B3 (bottleneck identification).

---

## 8. Nsight Compute Per-Kernel Deep Dive

For the top-3 hot kernels identified by Nsight Systems, we rerun under Nsight Compute for the metric set we care about:

```bash
# Profile a single kernel call with full metric set
ncu \
    --set full \
    --launch-skip 100 \
    --launch-count 1 \
    --output benchmarks/results/.../ncu_groupnorm_silu \
    --kernel-name regex:'groupnorm_silu' \
    dotnet test ... --filter "FullyQualifiedName~Sdxl_GenerateImage_Gpu"
```

`--launch-skip 100` skips the first 100 launches (warmup, not the steady-state target). `--launch-count 1` profiles exactly one steady-state call.

Metrics we care about for the paper:
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — SM utilization
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` — HBM throughput vs theoretical max
- `smsp__warps_eligible.avg.per_cycle_active` — warp issue rate
- `l1tex__t_bytes.sum` — L1 traffic
- `lts__t_bytes.sum` — L2 traffic
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` — Tensor Core utilization (key metric for "are we using the hardware?")

---

## 9. Python Parallel Benchmark Run

The Python harness mirrors the C# microbench shape grid. Run order:

```bash
cd benchmarks/python-baseline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh \
    --output ../results/run_$(date -u +%Y-%m-%dT%H%M%SZ)_pytorch_baseline \
    --device cuda \
    --trials 5
```

Output structure mirrors the C# side:
```
run_*_pytorch_baseline/
├── hardware.txt          # same fields as C# side
├── software.txt          # pip freeze + cuda info
├── microbench.csv        # per-op per-shape latency
├── e2e_sdxl.csv          # per-step timing
└── e2e_flux_dev.csv
```

CSV schemas are identical between C# and Python so they merge cleanly.

---

## 10. Cross-Device Cloud Runs

For cloud GPUs (L40S / A100 / H100), the typical workflow:

```bash
# On the cloud instance — Lambda Labs, RunPod, vast.ai, etc.
# Driver and CUDA toolkit must already be installed; check:
nvidia-smi
nvcc --version

git clone <this-repo> hartsyinference && cd hartsyinference
git checkout <commit-sha>          # exact commit being benchmarked

# Install pinned PyTorch
cd benchmarks/python-baseline && python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ../..

# Build + run
dotnet build -c Release
bash benchmarks/run_benchmarks.sh    # full harness
```

Cloud runs commit to a separate device-tagged subdirectory:
```
benchmarks/results/run_2026-05-15T143000Z_l40s/
benchmarks/results/run_2026-05-15T160000Z_a100/
benchmarks/results/run_2026-05-15T180000Z_h100/
```

The slug is derived from `nvidia-smi --query-gpu=name --format=csv,noheader | tr ' ' '-' | tr '[:upper:]' '[:lower:]'` (e.g. `nvidia-l40s`, `nvidia-a100-pcie-40gb`).

---

## 11. Statistical Analysis

After a run, we compute statistics via [`benchmarks/analyze.py`](../../benchmarks/analyze.py):

```python
# Pseudocode of what analyze.py does
import scipy.stats as stats
import pandas as pd

baseline = pd.read_csv("results/run_baseline_a100/microbench.csv")
optimized = pd.read_csv("results/run_post_fa2_a100/microbench.csv")

merged = baseline.merge(optimized, on=["op", "shape", "dtype"], suffixes=("_old", "_new"))

# Per row, run Welch's t-test on the 5 trial latencies (long format)
def is_significant(row):
    t, p = stats.ttest_ind(
        row["latency_us_old_trials"],   # list of 5 latencies
        row["latency_us_new_trials"],
        equal_var=False)
    return p < 0.01 and row["latency_us_new_mean"] < row["latency_us_old_mean"]

merged["significant_speedup"] = merged.apply(is_significant, axis=1)
merged["speedup_x"] = merged["latency_us_old_mean"] / merged["latency_us_new_mean"]
```

Output: `results/run_*_comparison/comparison.csv` with columns:
```
op, shape, dtype, latency_us_old_mean, latency_us_old_ci95, latency_us_new_mean, latency_us_new_ci95, speedup_x, p_value, significant_speedup
```

Plus `comparison.md` rendering of the same data, sorted by speedup descending.

---

## 12. Common Pitfalls

These bite every GPU benchmarker. Document them so reviewers can confirm we avoided them:

1. **CPU governor**: must be `performance` for steady-state numbers. Default `powersave` adds 5–15 % variance.
   ```bash
   sudo cpupower frequency-set -g performance
   ```
2. **GPU clocks**: nvidia-smi can lock clocks for repeatable benchmarks (requires sudo + persistence mode):
   ```bash
   sudo nvidia-smi -pm 1                         # persistence mode on
   sudo nvidia-smi -lgc <base_clock>,<base_clock> # lock graphics clock to base (e.g. 1410 for 3060)
   sudo nvidia-smi -lmc <mem_clock>,<mem_clock>   # lock memory clock
   ```
   Required only on instances where thermals throttle mid-run. On cloud instances with adequate cooling this is usually unnecessary; document the choice in `hardware.txt`.
3. **Warmup**: the first kernel call after process start includes JIT (cuBLAS heuristics), kernel binary loading, and CUDA context setup. Always discard trial 0.
4. **Memory pressure noise**: another process pinning VRAM changes our `cuMemAlloc` latency. The harness kills any other CUDA process before starting (or aborts if one is found).
   ```bash
   nvidia-smi --query-compute-apps=pid --format=csv,noheader
   ```
5. **PCIe traffic**: external host activity (e.g. test result CSV writes) can interfere. The harness writes results to `/tmp/...` then atomically moves to `benchmarks/results/` after the measurement window closes.
6. **GPU thermal throttling**: long-running suites can heat-throttle. The harness samples GPU temperature; if it exceeds 80 °C, sleeps 30 s between trials.
7. **Different CUDA streams**: HartsyInference uses `nonBlocking=false` (per [`TROUBLESHOOTING.md`](../Checklists/TROUBLESHOOTING.md) #4); always benchmark on this stream, not on a fresh one.
8. **Process forking and CUDA context**: after `Parallel.For` etc., the CUDA context state may not match. Benchmarks always run on the main thread.
9. **`nsys --capture-range=cudaProfilerApi`**: without this, the trace includes test-runner startup, which dominates the timeline. With it, only the measurement window is captured.

---

## 13. Reproducibility Checklist

Before committing a result directory, verify:

- [ ] `hardware.txt` exists and shows GPU + driver + CPU
- [ ] `software.txt` exists and includes the git commit SHA + uncommitted-changes status
- [ ] `digests.txt` contains SHA-256 for every `.ptx` and any safetensors used
- [ ] All CSVs use the standard schema (see § 14)
- [ ] `comparison.md` is generated by `analyze.py`, not hand-edited
- [ ] N=5 trials per row; trial values stored in long format (one row per trial) so the t-test is rerunnable
- [ ] No PII or local paths in any committed file (paths absolutized via `realpath --relative-to=$REPO_ROOT`)

---

## 14. CSV Schemas (canonical)

### `microbench.csv`

```
run_id,timestamp_utc,gpu_name,gpu_compute_cap,driver_version,cuda_version,backend,op,shape,dtype,trial,latency_us,throughput_gflops,memory_mb,workspace_mb
```

- `backend` ∈ {`hartsyinference_cuda`, `pytorch`, `pytorch_xformers`}
- `op` ∈ {`matmul`, `conv2d_3x3`, `conv2d_1x1`, `groupnorm`, `layernorm`, `rmsnorm`, `sdpa_self`, `sdpa_cross`, `silu`, `gelu`, `broadcast_add`}
- `shape` is a string `"BxCxHxW"` for conv inputs or `"BxMxN-BxNxK"` for GEMM
- `trial` ∈ {0..4}; trial 0 may be flagged as warmup (see § 12)

### `e2e.csv`

```
run_id,timestamp_utc,gpu_name,backend,model,resolution,steps,seed,trial,total_ms,text_encode_ms,denoise_ms,vae_decode_ms,peak_vram_mb
```

### `comparison.csv`

```
op,shape,dtype,latency_us_old_mean,latency_us_old_ci95,latency_us_old_n,latency_us_new_mean,latency_us_new_ci95,latency_us_new_n,speedup_x,p_value,significant_speedup,delta_pct
```

These schemas are stable for the duration of Phase B. Schema changes require bumping a version field and re-baselining.

---

## 15. References

- NVIDIA Nsight Systems user guide: https://docs.nvidia.com/nsight-systems/UserGuide/
- NVIDIA Nsight Compute kernel profiling: https://docs.nvidia.com/nsight-compute/NsightCompute/index.html
- BenchmarkDotNet: https://benchmarkdotnet.org/
- "How to benchmark GPU code", NVIDIA dev blog: https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
- PyTorch profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- NVTX 3 ranges: https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvtx
- "Benchmarking Crimes" (Heiser, 2010) — list of common errors in systems benchmarks: https://gernot-heiser.org/benchmarking-crimes.html
