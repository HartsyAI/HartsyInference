# Cloud GPU Runbook — Capturing Baselines Across SM Generations

> **Purpose**: produce the four `benchmarks/results/run_baseline_*` directories required by Phase B2. One per device, committed to the repo, immutable thereafter.

This runbook is the operational companion to [`docs/Research/PROFILING_METHODOLOGY.md`](../docs/Research/PROFILING_METHODOLOGY.md). The methodology document explains the *what* and *why*; this document is the *how* for cloud rentals.

## Target devices (B2 baseline matrix)

| Device | SM | Memory | Why included |
|---|---|---|---|
| RTX 3060 12 GB | 8.6 | 12 GB GDDR6 | Dev box; the primary "consumer" data point in the paper |
| RTX 4090 24 GB **or** L40S 48 GB | 8.9 | 24 / 48 GB | Ada — exercises native FP8 GEMM via `cublasLtMatmul` |
| A100 40 GB | 8.0 | 40 GB HBM2e | Datacenter Ampere reference |
| H100 80 GB | 9.0 | 80 GB HBM3 | Hopper — Tensor Core 4th-gen + (B4.7 stretch) WGMMA |

Each gets its own `run_baseline_<utc>_<gpu-slug>` directory under `benchmarks/results/`.

## Cloud provider quick-pick

Pick whichever has the device available with hourly rates. Cost-sensitive: spot/preemptible pricing where available. All numbers are approximate as of 2026-05; recheck before renting.

| Provider | RTX 4090 | L40S | A100 40 GB | H100 80 GB | Notes |
|---|---|---|---|---|---|
| **Lambda Labs** | n/a | ~$1.10/hr | ~$1.10/hr | ~$3.00/hr | Pre-built CUDA, easy `lambdalabs/lambda-stack` AMI |
| **RunPod** (community) | ~$0.30/hr | ~$0.70/hr | ~$0.85/hr | ~$2.20/hr | Cheaper, sometimes flaky |
| **vast.ai** | ~$0.20/hr | ~$0.70/hr | ~$0.80/hr | ~$2.00/hr | Cheapest; verifies via DLPerf score |
| **GCP A2/A3** | n/a | n/a | A2 (~$3.00/hr) | A3 (~$11/hr on-demand, less spot) | Best for reproducibility-critical runs (verified hardware) |
| **AWS p4/p5** | n/a | n/a | p4d.24xl (~$3.00/hr) | p5.48xl (very expensive) | Use for paper submission camera-ready runs |

For the paper baseline, **prefer Lambda Labs or RunPod community** — sufficient hardware verification, low enough cost to do multiple full passes per device. Use AWS / GCP only for the final camera-ready figure.

## Per-device runbook (script-friendly steps)

The exact same script runs on every device. The instance type selection is the only variable.

### 0. Pre-flight (once per provider)

```bash
# Verify GPU + CUDA toolkit + driver are sane
nvidia-smi
nvcc --version
# Driver must be ≥ 535.x for cuBLASLt FP8 (Ada+ paths).
# CUDA toolkit must be 12.4.x to match our pinned PyTorch wheel.

# Optional: for repeatability, lock GPU clocks (requires sudo + persistence mode)
# Skip this on shared / cloud instances where it may be denied.
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc <base_clock>,<base_clock>
sudo nvidia-smi -lmc <mem_clock>,<mem_clock>

# Make sure no other CUDA process is pinning VRAM
nvidia-smi --query-compute-apps=pid --format=csv,noheader
```

### 1. Clone + build (one-time)

```bash
git clone https://github.com/<org>/HartsyInference hartsyinference
cd hartsyinference

# Pin to exact commit being benchmarked
git checkout <commit-sha>
git rev-parse HEAD  # capture this; goes into software.txt

# Build everything in Release
dotnet build -c Release
```

### 2. Python venv (one-time per box)

```bash
cd benchmarks/python-baseline
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

# Verify
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

deactivate
cd ../..
```

### 3. Download checkpoints for end-to-end runs (one-time per box)

End-to-end Python baselines need the model weights resident. The microbench layer doesn't need them
(it benchmarks ops on synthetic data). Skip this step if you're only running microbenches.

```bash
# diffusers caches under HF_HOME / DIFFUSERS_CACHE; default is ~/.cache/huggingface
mkdir -p ~/.cache/huggingface
export HF_HOME=~/.cache/huggingface

# Optional but recommended on cloud: pre-download to a persistent volume
# huggingface-cli login  # if any model is gated
# huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0
# huggingface-cli download black-forest-labs/FLUX.1-dev
```

### 4. Run the harness

```bash
# Full baseline pass — typically 20-40 min per device with all 4 e2e models
bash benchmarks/run_benchmarks.sh \
    --py-venv benchmarks/python-baseline/.venv \
    --trials 5 \
    --out-base benchmarks/results
```

The output is committed to `benchmarks/results/run_<utc>_<gpu-slug>/`. **Rename the directory** to
include `_baseline` so it's clearly the immutable reference:

```bash
DIR=$(ls -1dt benchmarks/results/run_2026-* | head -n1)  # most recent
mv "$DIR" "${DIR}_baseline"
```

### 5. Smoke check the result

```bash
NEW=$(ls -1dt benchmarks/results/run_*_baseline | head -n1)

# Required files present?
for f in hardware.txt software.txt digests.txt microbench_csharp.csv microbench_pytorch.csv comparison.md; do
    test -s "$NEW/$f" && echo "OK: $f" || echo "MISSING: $f"
done

# Comparison sanity — should have a row per (op, shape, dtype)
wc -l "$NEW/comparison.csv"
head -5 "$NEW/comparison.md"
```

If any required file is missing, see the troubleshooting section below. If everything is present:

```bash
# Commit the baseline (raw CSVs + reports go in git)
git add "$NEW"
git commit -m "B2: baseline benchmark on $(basename $NEW)"
git push
```

### 6. Tear down

```bash
# On the cloud instance, stop and destroy to avoid extra charges
# (provider-specific; e.g. for Lambda Labs the dashboard has a stop button)
```

## Environment variables that affect runs

| Var | Effect | When to set |
|---|---|---|
| `HF_HOME` | diffusers / HF cache root | Set to a persistent volume on cloud instances to avoid re-downloading on each rental |
| `CUDA_VISIBLE_DEVICES` | restricts visible GPUs | Set to `0` on multi-GPU boxes to ensure single-GPU benchmarks |
| `CUBLASLT_LOG_LEVEL` | cuBLASLt log verbosity | Set to `5` to debug FP8 GEMM dispatch issues; do NOT set during measurement (slows things down) |
| `HARTSYINFERENCE_NVTX_DETAILED` | enables per-op NVTX ranges | Set when running under `nsys` for fine-grained timeline; leave unset for plain timing |
| `TORCH_CUDA_ALLOC_CONF` | torch CUDA allocator tuning | Leave at default — changing this invalidates the baseline |

## Troubleshooting

### "PyTorch baseline failed — see ...pytorch_bench.log"

The most common cause is a missing checkpoint. The e2e scripts skip a model gracefully when its
weights aren't downloaded; check the log for `[sdxl] skipped — checkpoint unavailable`. Either
download the checkpoint and re-run, or pass `--skip-e2e` to keep the microbench data.

### "Test Run Aborted" in the C# part

Unrelated to the actual benchmark — xunit's noise after a return-style skip. Confirm via the BDN
report directory: `ls benchmarks/results/run_*/csharp_bench/results/`. If the JSON files exist,
the run was fine.

### `nvidia-smi --query-gpu=memory.used` shows huge usage between runs

`cuMemFreeAsync` defers the actual free until the stream syncs. The harness syncs at end of run, but
if you Ctrl-C mid-run the deferred memory may stay resident until the process exits. Wait for the
process to fully exit before inspecting VRAM, or run `nvidia-smi --gpu-reset` (requires sudo, root
device only) — and document any reset in the run log.

### Latencies are 2-5× higher than the previous baseline on the same device

Most likely thermal throttling or another process pinning VRAM. Check:

```bash
nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm --format=csv -l 1
```

If the SM clock is below the device's base clock at high temperatures, the run is invalid. Wait
for cooldown and retry, or document the throttling in the run's `hardware.txt`.

### "PTX module load failed" on a brand-new device

The compiled PTX in `src/HartsyInference.Cuda/Ptx/` targets `sm_70` by default (forward-compatible
to all newer architectures). If a device fails this load, check `nvcc --version` matches our
pinned CUDA 12.4 — older toolkits may not generate compatible PTX.

## What "baseline complete" means

A `run_baseline_<gpu>_<utc>` directory is **complete** when:

- [ ] `hardware.txt`, `software.txt`, `digests.txt` populated with non-empty content
- [ ] `microbench_csharp.csv` has > 50 rows (~14 benchmarks × 5 trials × ≥ 5 shapes each, with both F16 + F32)
- [ ] `microbench_pytorch.csv` has > 50 rows (matching shape grid)
- [ ] `comparison.md` exists and is non-empty
- [ ] `e2e_pytorch.csv` exists for at least one of {SDXL, Flux Dev} when checkpoints were available
- [ ] git diff is empty (other than the new directory itself) — no accidental code changes during the run

When all four target devices have a complete baseline, B2 is done. Move to B3 (profile + identify
bottlenecks).

## Cost estimate (rough)

Per device, end-to-end:

| Phase | Time | Cost @ $1/hr | Cost @ $3/hr |
|---|---|---|---|
| Setup (clone, deps, downloads) | 30-60 min | $0.50 | $1.50 |
| Microbench run | 10-20 min | $0.20 | $0.60 |
| E2E run (SDXL + Flux) | 5-15 min | $0.20 | $0.60 |
| Buffer | 30 min | $0.50 | $1.50 |
| **Total** | ~2 hr | **~$1.50** | **~$4.50** |

For 4 devices total: $6 - $20. Budget $30 to allow re-runs if a baseline is invalidated by
environmental noise.
