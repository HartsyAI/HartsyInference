using BenchmarkDotNet.Attributes;
using HartsyInference.Core.Tensors;

namespace HartsyInference.GpuBenchmarks;

/// <summary>Stage-6 probe for the quant/GEMM perf plan: how expensive is the per-op dequant in the
/// low-VRAM <see cref="HartsyInference.Cuda.CudaBackend.QuantizedMatMul"/> path (re-dequant every call,
/// weights stay compressed) versus the cached-F16 path (dequant once, F16-sized VRAM)? The dequant is
/// O(N·K), independent of M, so it dominates at M=1 (LLM decode) but amortizes over the big GEMM at the
/// large M of diffusion (1024–4096 tokens/step). Comparing QuantMatMul vs cached-F16 Linear at both Ms
/// quantifies the overhead and its amortization — the input to the (a) wire-it / (b) leave-load-only
/// decision. Shape: a Flux FFN (K=3072, N=12288).</summary>
[Config(typeof(GpuBenchmarkConfig))]
public class QuantMatMulGpuBenchmarks
{
    private const int K = 3072, N = 12288;   // Flux DiT FFN expand

    private BenchmarkFixture? _fixture;
    private Tensor? _inF16, _wF16, _wQ4K, _outF16;

    [Params(1, 1024, 4096)]
    public int M { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _fixture = new BenchmarkFixture();
        _inF16 = BenchmarkFixture.AllocateF16(new TensorShape(M, K), seed: 1);
        _wF16 = BenchmarkFixture.AllocateF16(new TensorShape(N, K), seed: 2);
        _outF16 = new Tensor(new TensorShape(M, N), DType.F16);
        // Synthetic Q4_K weight ([N,K], N·K divisible by 256). Byte content is irrelevant to timing — the
        // dequant kernel walks the same block layout regardless; we only measure kernel time.
        _wQ4K = new Tensor(new TensorShape(N, K), DType.Q4_K);
        // Pin weights + input on-device so CopyToDevice is a cache hit, not a per-call H2D upload — else the
        // probe measures PCIe bandwidth (the 75 MB F16 weight), not the dequant. Real inference preloads.
        _fixture.Backend.PreloadWeights(new[] { _inF16, _wF16, _wQ4K });
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _inF16?.Dispose(); _wF16?.Dispose(); _wQ4K?.Dispose(); _outF16?.Dispose();
        _fixture?.Dispose();
    }

    /// <summary>Baseline: F16 weight already resident, pure cuBLAS GEMM (no dequant). The cached-F16 path.</summary>
    [Benchmark(Baseline = true)]
    public void Linear_F16()
    {
        _fixture!.Backend.Linear(_outF16!, _inF16!, _wF16!, bias: null);
        _fixture.Sync();
    }

    /// <summary>Low-VRAM path: Q4_K weight stays compressed; dequant→F16 happens every call, then GEMM.
    /// Overhead vs <see cref="Linear_F16"/> at the same M is the per-op dequant cost.</summary>
    [Benchmark]
    public void QuantMatMul_Q4K()
    {
        _fixture!.Backend.QuantizedMatMul(_outF16!, _inF16!, _wQ4K!, bias: null);
        _fixture.Sync();
    }
}
