# HartsyInference Benchmarks

This directory holds the benchmarking infrastructure for [Phase B GPU performance optimization](../docs/Checklists/PHASE_B_GPU_PERFORMANCE.md). Read [`docs/Research/CUDA_PERFORMANCE_PLAN.md`](../docs/Research/CUDA_PERFORMANCE_PLAN.md) and [`docs/Research/PROFILING_METHODOLOGY.md`](../docs/Research/PROFILING_METHODOLOGY.md) before adding new benchmarks — the methodology is non-trivial.

## LLM decode throughput vs llama.cpp (2026-07-04)

Separate from the diffusion/Phase-B harness below. Docs: [`LLM_THROUGHPUT_BENCHMARK.md`](../docs/Checklists/LLM_THROUGHPUT_BENCHMARK.md) (baseline + method) and [`LLM_DECODE_PERF_GRIND.md`](../docs/Checklists/LLM_DECODE_PERF_GRIND.md) (optimization log). Assets:

- **Baseline**: CUDA-compiled `llama-bench` (`~/models/llamacpp/build/bin/`), swept over the 7 SOTA GGUFs in `Models/llm/` with `-ngl 99 -p 512 -n 128 -r 5`. Result: `results/llamacpp_baseline_3060.json`.
- **Tier-1 (engine, direct)**: the existing `samples/HartsyInference.TextGen.Cli` (`gguf cuda <ntok> "<prompt>"`) — loads a GGUF on CUDA, times decode, reports tok/s + D2H sync count. Raw: `results/tier1_engine_3060.txt`.
- **Tier-2 (engine, through Swarm)**: `swarm_llm_bench/swarm_llm_bench.py` drives the live SwarmUI `LLMAssistantSendMessageWS` WebSocket (hartsy-local provider), measuring client-side. Raw: `results/swarm_llm_3060.json`.
- **Outcome**: decode gap **20-54× → 1.94-2.88×** off llama.cpp (Llama-3.2-1B under 2×) via fused quantized GEMV + quantized lm_head + split-K flash-decode attention + vectorized loads.
- **CUDA graph foundation** (last lever): `hartsyinference-textgen graphtest` verifies `CudaGraph` capture/replay works on-GPU.

### Current LLM decode results (RTX 3060, warm, 128-token greedy, tok/s)

| Model | Quant | llama.cpp tg | engine tg | gap |
|---|---|---:|---:|---:|
| Llama-3.2-1B | Q8_0 | 215.9 | ~111.5 | 1.94× |
| Mistral-7B-v0.3 | Q4_K_M | 66.5 | ~30.7 | 2.12× |
| Qwen3-0.6B | Q4_K_M | 354.5 | ~157 | 2.26× |
| Gemma-3-1B | Q4_K_M | 229.8 | ~79.7 | 2.88× |

Prefill (pp512) is not the bottleneck; the remaining decode gap is launch-overhead on small models (next lever: CUDA graphs). Full per-phase log in `LLM_DECODE_PERF_GRIND.md`.

## Diffusion / video e2e vs ComfyUI (2026-07-03)

End-to-end wall-clock through the **SwarmUI API** (the identical request routed to the ComfyUI backend, then the HartsyInference backend, on the same RTX 4090). This is the user-perceived-latency comparison; it complements the in-engine microbench harness. Full write-up + host-vs-compute diagnosis: [`results/video_comfy-vs-hartsy_2026-07-03.md`](results/video_comfy-vs-hartsy_2026-07-03.md).

| Model | Quant | Hartsy warm | Comfy warm | gap |
|---|---|---:|---:|---:|
| Wan 2.1 T2V 1.3B | fp16 | ~23.7 s | 6.28 s | ~3.8× |
| LTX-0.9 2B | fp16 | ~15 s | 2.84 s | ~5.3× |
| Wan 2.2 TI2V-5B | fp16 | ~37.9 s | 4.52 s | ~8.4× |
| Wan 2.1 T2V 14B | fp8 | ~180 s | 30.6 s | ~5.9× |

All outputs are coherent: this is a speed gap, not a correctness gap. Image architectures (Flux/SD3/Ideogram) were device-ported and run much closer; the video DiT blocks are the current frontier (no full flash-attention kernel yet, some F32-only elementwise ops, launch-overhead at small token counts).

## Quick start

```bash
# 1. Set up Python venv (one time; see requirements.txt for pinned versions)
python3 -m venv benchmarks/python-baseline/.venv
source benchmarks/python-baseline/.venv/bin/activate
pip install -r benchmarks/python-baseline/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
deactivate

# 2. Run the full harness (~20-40 min depending on GPU + e2e flag)
bash benchmarks/run_benchmarks.sh --py-venv benchmarks/python-baseline/.venv

# 3. Smoke test (faster, 1 trial of MatMul only) for harness debugging
bash benchmarks/run_benchmarks.sh --smoke --py-venv benchmarks/python-baseline/.venv
```

Result lands in `benchmarks/results/run_<utc>_<gpu>/`. See [`results/README.md`](results/README.md) for what's in there.

## Directory layout

```
benchmarks/
├── README.md                                 ← this file
├── run_benchmarks.sh                         ← end-to-end harness
├── profile.sh                                ← Nsight Systems wrapper
├── analyze.py                                ← Welch's t-test joiner; emits comparison.{csv,md}
├── results/                                  ← committed result directories (raw data for the paper)
│   ├── README.md
│   └── run_*/
├── HartsyInference.Benchmarks/                ← legacy CPU benchmarks (existing; not Phase B)
├── HartsyInference.GpuBenchmarks/             ← C# GPU microbenchmarks via BenchmarkDotNet
│   ├── HartsyInference.GpuBenchmarks.csproj
│   ├── BenchmarkConfig.cs                    ← shared BDN config (1 warmup, 5 trials)
│   ├── BenchmarkFixture.cs                   ← CudaBackend + tensor allocation helpers
│   ├── Program.cs
│   ├── MatMulGpuBenchmarks.cs
│   ├── Conv2DGpuBenchmarks.cs
│   ├── NormGpuBenchmarks.cs
│   ├── SdpaGpuBenchmarks.cs
│   ├── ElementwiseGpuBenchmarks.cs
│   └── MemoryAllocFreeBenchmarks.cs
└── python-baseline/                          ← pinned PyTorch + diffusers parity scripts
    ├── README.md (TBD)
    ├── requirements.txt                      ← PyTorch 2.5.1 + cu124 etc., pinned
    ├── _common.py                            ← timing, fingerprints, CSV writer
    ├── run_all.sh
    ├── bench_pytorch_matmul.py
    ├── bench_pytorch_conv2d.py
    ├── bench_pytorch_norms.py
    ├── bench_pytorch_sdpa.py
    ├── bench_pytorch_elementwise.py
    └── bench_pytorch_e2e.py
```

## Statistical rigor

Every measurement reported in the paper is grounded in a `benchmarks/results/run_*/` directory.
Methodology:

- **Warmup**: 1 invocation discarded.
- **Trials**: N=5 per benchmark.
- **Confidence interval**: 95 % via Student-t (df=4).
- **Significance gate**: a "speedup" is reported only when (a) the new mean is outside the old 95 % CI AND (b) Welch's t-test rejects μ_new = μ_old at α = 0.01.
- **Fingerprinting**: every run captures hardware, software, and PTX/checkpoint digests — see [`results/README.md`](results/README.md).

See [`docs/Research/PROFILING_METHODOLOGY.md`](../docs/Research/PROFILING_METHODOLOGY.md) for the full procedure.

## What lives in each script

| Script | What it does |
|---|---|
| `run_benchmarks.sh` | Top-level harness: fingerprints → dotnet build → C# microbench → Python baselines → analyze.py → atomic move into `results/` |
| `profile.sh` | Wraps Nsight Systems (`nsys profile`) around a single end-to-end test; emits `.qdrep` + `nsys stats` summary |
| `analyze.py` | Joins C# and PyTorch microbench CSVs by (op, shape, dtype); runs Welch's t-test; emits comparison.{md,csv} |
| `python-baseline/run_all.sh` | Runs every PyTorch baseline script in sequence, appending to a single CSV |

## Common operations

```bash
# Smoke test the harness (1 trial each, MatMul only, no e2e)
bash benchmarks/run_benchmarks.sh --smoke

# Skip Python (e.g. to iterate on C# benches without re-running PyTorch)
bash benchmarks/run_benchmarks.sh --skip-python

# Profile a single SDXL run with Nsight Systems
bash benchmarks/profile.sh --test "FullyQualifiedName~Sdxl_GenerateImage_Gpu"

# Run only one C# benchmark class
dotnet run --no-build -c Release --project benchmarks/HartsyInference.GpuBenchmarks -- \
    --filter '*Sdpa*' --warmupCount 1 --iterationCount 5

# Run only one Python baseline (for debugging schema issues)
python3 benchmarks/python-baseline/bench_pytorch_matmul.py --output /tmp/test.csv --trials 1
```

## What does NOT belong here

- **Functional tests** (numerical correctness): those live under `tests/`. Benchmarks measure speed; tests measure math.
- **Per-checkpoint reference dumps**: those live under `tests/python-reference/`.
- **CPU benchmarks**: keep using `benchmarks/HartsyInference.Benchmarks/` (existing, BDN-based).

## When to add a new benchmark

Add a new `*GpuBenchmarks.cs` class when:
1. A new `IBackend` op is introduced
2. A model uses a shape combination not covered by the current grid
3. An optimization phase (B4.x) targets a workload pattern that needs its own microbench

When you add one, also add the matching Python script to `python-baseline/` so the comparison stays apples-to-apples.

## License + reproducibility

All result CSVs in `benchmarks/results/` are checked into the repository. They form the public reproducibility trail for the paper. The harness scripts (this directory) are part of the HartsyInference MIT-licensed source distribution — re-run them on any compatible CUDA box to reproduce or refute our numbers.
