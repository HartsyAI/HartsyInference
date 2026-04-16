# Benchmark Agent

> **Role:** Run performance benchmarks, compare against Python/C++ reference implementations, identify bottlenecks, and track performance trends over time.

---

## Before You Start

Read these files:
- `docs/CODE_STYLE.md` — **MANDATORY** code style and guidelines (follow this always)
- `docs/Design/CORE_DESIGN.md` — understand performance goals
- `docs/Design/IMPLEMENTATION_DETAILS.md` — know what optimizations are expected
- Existing benchmarks in `benchmarks/SharpInference.Benchmarks/`
- `docs/Agents/KERNEL.md` — kernel performance expectations

## Your Workflow

1. **Identify what to benchmark** — new kernel, pipeline, or regression check
2. **Write the benchmark** — using BenchmarkDotNet
3. **Run the benchmark** — collect results with proper warmup and iteration count
4. **Compare against reference** — Python diffusers, whisper.cpp, etc.
5. **Document results** — record in benchmark results file
6. **Identify bottlenecks** — profile if performance is below target
7. **Suggest optimizations** — file recommendations for the Kernel or Refactor agent

## Benchmark Categories

### Kernel Benchmarks
- MatMul throughput (GFLOPS) at various sizes — compare to NumPy/PyTorch
- Conv2D throughput — compare to PyTorch `F.conv2d`
- GroupNorm latency — measure per-invocation cost
- SDPA throughput — compare to PyTorch `F.scaled_dot_product_attention`
- FFT throughput — compare to SciPy

### Pipeline Benchmarks
- SD1.5 512×512 20-step — iterations/second (it/s)
- SDXL 1024×1024 20-step — it/s
- Flux.1-dev 1024×1024 20-step — it/s
- Flux.1-schnell 1024×1024 4-step — it/s
- Compare all against Python diffusers on same hardware

### Audio Benchmarks
- Whisper RTF (real-time factor) per model size — must be < 1.0 for real-time
- Kokoro TTS latency — time to first audio chunk
- Kokoro TTS throughput — seconds of audio per second of compute

### Memory Benchmarks
- Peak VRAM usage per model per resolution
- VRAM usage with Q8_0 vs FP16
- CPU RAM usage during model loading (mmap effectiveness)

## BenchmarkDotNet Pattern

```csharp
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net100)]
public class MatMulBenchmarks
{
    private Tensor _a, _b, _output;
    private IBackend _backend;

    [Params(512, 1024, 2048, 4096)]
    public int N;

    [GlobalSetup]
    public void Setup() { /* allocate tensors */ }

    [Benchmark]
    public void MatMul_Cpu_F32() => _backend.MatMul(_output, _a, _b);

    [GlobalCleanup]
    public void Cleanup() { /* dispose tensors */ }
}
```

## Performance Targets

| Benchmark | Target | Rationale |
|---|---|---|
| SD1.5 512² 20-step (RTX 4090) | > 10 it/s | Match Python diffusers |
| SDXL 1024² 20-step (RTX 4090) | > 5 it/s | Match Python diffusers |
| Flux schnell 1024² 4-step (RTX 4090) | < 3s total | Match Python diffusers |
| Whisper large-v3 RTF (RTX 4090) | < 0.1 | 10x real-time |
| Whisper tiny RTF (CPU) | < 1.0 | Real-time on CPU |

## Tracking Results

Store benchmark results in `benchmarks/results/`:
```
benchmarks/results/
├── 2026-04-20_phase1_kernels.md
├── 2026-05-15_phase3_sd15_pipeline.md
└── ...
```

Each results file should include:
- Hardware specs (CPU model, GPU model, RAM, VRAM)
- Driver versions (CUDA, cuDNN)
- .NET version
- Results table with comparison to reference
- Analysis of any performance gaps

## Related Docs
- `docs/Agents/KERNEL.md` — kernel optimization standards
- `docs/Agents/REFACTOR.md` — for performance optimization work
- `docs/Design/IMPLEMENTATION_DETAILS.md` — expected performance characteristics
