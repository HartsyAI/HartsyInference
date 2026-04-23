# Benchmark Agent

> Run performance benchmarks, compare against references, identify bottlenecks, track trends.

## Extra Reading
- `docs/Design/IMPLEMENTATION_DETAILS.md`
- `docs/Agents/KERNEL.md`
- Existing benchmarks in `benchmarks/SharpInference.Benchmarks/`

## Workflow
1. Identify target (kernel, pipeline, regression check)
2. Write BenchmarkDotNet benchmark
3. Run and compare against Python diffusers, whisper.cpp, etc.
4. Document results in `benchmarks/results/`
5. Profile and suggest optimizations

## Benchmark Categories

**Kernel:** MatMul (GFLOPS), Conv2D, GroupNorm, SDPA, FFT — compare to PyTorch.
**Pipeline:** SD1.5 512² 20-step, SDXL 1024² 20-step, Flux 1024² — it/s vs diffusers.
**Audio:** Whisper RTF (must be < 1.0), Kokoro TTS latency/throughput.
**Memory:** Peak VRAM per model/resolution; Q8_0 vs FP16; mmap effectiveness.

## BenchmarkDotNet Pattern
```csharp
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net100)]
public class MatMulBenchmarks
{
    private Tensor _a, _b, _output;
    private IBackend _backend;
    [Params(512, 1024, 2048, 4096)] public int N;
    [GlobalSetup] public void Setup() { /* allocate */ }
    [Benchmark] public void MatMul_Cpu_F32() => _backend.MatMul(_output, _a, _b);
    [GlobalCleanup] public void Cleanup() { /* dispose */ }
}
```

## Targets

| Benchmark | Target |
|---|---|
| SD1.5 512² 20-step (RTX 4090) | > 10 it/s |
| SDXL 1024² 20-step (RTX 4090) | > 5 it/s |
| Flux schnell 1024² 4-step (RTX 4090) | < 3s total |
| Whisper large-v3 RTF (RTX 4090) | < 0.1 |
| Whisper tiny RTF (CPU) | < 1.0 |

## Results Storage
`benchmarks/results/YYYY-MM-DD_phaseN_component.md` with hardware specs, driver versions, .NET version, results table, gap analysis.
