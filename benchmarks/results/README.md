# Benchmark Results

This directory holds the raw output of every benchmark run. Each subdirectory is named:

```
run_<utc-iso8601>_<gpu-slug>/
```

Examples:

```
run_2026-05-15T143000Z_nvidia-rtx-3060/
run_2026-05-16T093000Z_nvidia-l40s/
run_2026-05-17T120000Z_nvidia-a100-pcie-40gb/
run_2026-05-18T160000Z_nvidia-h100-80gb-hbm3/
```

## What's in each run directory

| File | Source | Purpose |
|---|---|---|
| `hardware.txt` | `run_benchmarks.sh` | nvidia-smi, lscpu, governor, ECC, etc. |
| `software.txt` | `run_benchmarks.sh` | dotnet, nvcc, driver, pip-freeze, git rev |
| `digests.txt` | `run_benchmarks.sh` | SHA-256 of every PTX file + native CUDA source |
| `microbench_csharp.csv` | BenchmarkDotNet via `HartsyInference.GpuBenchmarks` | per-kernel per-shape latency (C#) |
| `microbench_pytorch.csv` | `python-baseline/run_all.sh` | per-kernel per-shape latency (PyTorch) |
| `e2e_csharp.csv` | (added in B4 once wired) | per-step end-to-end timing (C#) |
| `e2e_pytorch.csv` | `python-baseline/bench_pytorch_e2e.py` | per-step end-to-end timing (PyTorch + diffusers) |
| `comparison.csv` | `analyze.py` | merged microbench, Welch's t-test outcome |
| `comparison.md` | `analyze.py` | human-readable side-by-side, sorted by speedup |
| `nsys_report.qdrep` | `profile.sh` | Nsight Systems trace (when collected) |
| `nsys_stats.txt` | `profile.sh` | top-N kernels by total time, NVTX summary |
| `*.log` | each step | redirected stdout/stderr from each phase of the harness |

## CSV schemas

See `docs/Research/PROFILING_METHODOLOGY.md` § 14 for the canonical column lists. Schemas are stable
for the duration of Phase B; any change requires bumping a version field and re-baselining.

## How to interpret results

1. **Always read `hardware.txt` and `software.txt` first.** A speedup claim that holds on one driver
   version may not hold on another. The fingerprints scope the claim.
2. **Open `comparison.md`** — it lists every (op, shape, dtype) tuple sorted by `speedup_x_csharp_over_pytorch`.
   Values > 1 mean HartsyInference is faster than PyTorch on that shape.
3. **Trust the `significant` column.** A speedup that doesn't survive a Welch's t-test at α = 0.01 is
   noise.
4. **For paper figures**: drive plotting from `comparison.csv` (the joined data); never hand-author
   numbers.

## Naming convention for special runs

| Suffix | Meaning |
|---|---|
| `run_baseline_<gpu>` | The B2 baseline run (committed; immutable) |
| `run_post_<phaseId>_<gpu>` | After-optimization run for B4.x — e.g. `run_post_b41_nvidia-rtx-3060` |
| `run_final_<gpu>` | Final B5 run; populates the paper's headline figures |
| `run_smoke_<gpu>` | Smoke test from `--smoke` flag; not for publication |

## Reproducing a result

```bash
# On any CUDA box with dotnet, python3, nvidia-smi:
git clone <repo> hartsyinference && cd hartsyinference
git checkout <commit-sha-from-software.txt>

# Match Python deps
python3 -m venv benchmarks/python-baseline/.venv
source benchmarks/python-baseline/.venv/bin/activate
pip install -r benchmarks/python-baseline/requirements.txt

# Run (matches the methodology used in this repo)
bash benchmarks/run_benchmarks.sh --py-venv benchmarks/python-baseline/.venv

# Compare
diff <new-run>/comparison.csv <archived-run>/comparison.csv
```

If the diff shows means deviating > 5% on any non-noise shape, investigate (driver change, thermal,
a process pinning VRAM during the run, etc.).
