using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Isolates the Q8_0 fused GEMV (the HeartMuLa decode hot path) vs cuBLAS bf16 at the CSM shapes
/// (N=K=3072, M=1, plus the MLP shapes), weight resident. HeartMuLa Q8 measured ~8× slower than bf16 on the real
/// model — this pins down whether that's the KERNEL (slow at M=1) or the PATH (weight re-upload thrash), by timing
/// the kernel directly with the weight cached resident across iterations. Opt-in: <c>dotnet test --filter Category=Q8Bench</c>.</summary>
[Collection("CudaSerial")]
[Trait("Category", "Q8Bench")]
public sealed unsafe class Q8GemvMicroBench
{
    private readonly ITestOutputHelper _output;
    public Q8GemvMicroBench(ITestOutputHelper output) => _output = output;

    private static uint _rng = 0x9E3779B9u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.1f; }
    private static Tensor RndF32(params int[] dims)
    {
        long[] d = Array.ConvertAll(dims, x => (long)x);
        Tensor t = new(new TensorShape(d), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = Rand();
        return t;
    }

    // (N, K) for one projection: backbone attn q/o (3072x3072), kv (1024x3072), mlp gate/up (8192x3072), down (3072x8192).
    private static readonly (int N, int K, string tag)[] Shapes =
    {
        (3072, 3072, "attn q/o 3072x3072"),
        (1024, 3072, "attn kv  1024x3072"),
        (8192, 3072, "mlp up   8192x3072"),
        (3072, 8192, "mlp down 3072x8192"),
    };

    [Fact]
    public void Q8Gemv_vs_Bf16_M1()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        using CudaBackend cuda = new(0, ptxDir);
        IBackend b = cuda;
        const int warmup = 20, iters = 200;
        _output.WriteLine($"M=1 decode GEMV, weight resident (cached), warmup={warmup} iters={iters}");
        _output.WriteLine($"{"shape",-22}{"bf16 (µs)",12}{"Q8 Linear",12}{"Q8 QMatMul",12}{"Q8/bf16",10}");

        foreach ((int N, int K, string tag) in Shapes)
        {
            using Tensor input = RndF32(1, 1, K);
            using Tensor wF32 = RndF32(N, K);
            using Tensor wBf16 = wF32.CastTo(DType.BF16);
            using Tensor wQ8 = GgufQuantizer.Quantize(wF32, DType.Q8_0);
            using Tensor outBf16 = new(new TensorShape(1, 1, N), DType.F32);
            using Tensor outQ8 = new(new TensorShape(1, 1, N), DType.F32);

            double bf16us = Time(() => b.Linear(outBf16, input, wBf16, null), cuda, warmup, iters);
            double q8LinUs = Time(() => b.Linear(outQ8, input, wQ8, null), cuda, warmup, iters);
            double q8QmmUs = Time(() => b.QuantizedMatMul(outQ8, input, wQ8, null), cuda, warmup, iters);

            _output.WriteLine($"{tag,-22}{bf16us,11:F1}{q8LinUs,11:F1}{q8QmmUs,11:F1}{q8LinUs / bf16us,9:F2}x");
        }
    }

    private static double Time(Action op, CudaBackend cuda, int warmup, int iters)
    {
        for (int i = 0; i < warmup; i++) op();
        cuda.Sync();
        Stopwatch sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) op();
        cuda.Sync();
        sw.Stop();
        return sw.Elapsed.TotalMicroseconds / iters;
    }
}
