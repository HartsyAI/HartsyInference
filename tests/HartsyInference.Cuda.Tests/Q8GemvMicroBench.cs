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

    // (N, K) for one projection. CSM/HeartMuLa: attn q/o 3072x3072, kv 1024x3072, mlp up 8192x3072, down 3072x8192.
    // YuE2's AR is a different aspect ratio (hidden 2048, intermediate 6144) and its `mlp down` — small N, large K —
    // is the under-occupied shape the K-split note at CudaKernels.LaunchMulMatVec* calls out, so it is measured here
    // rather than inferred from the 3072-square case.
    private static readonly (int N, int K, string tag)[] Shapes =
    {
        (3072, 3072, "csm  attn q/o 3072x3072"),
        (1024, 3072, "csm  attn kv  1024x3072"),
        (8192, 3072, "csm  mlp up   8192x3072"),
        (3072, 8192, "csm  mlp down 3072x8192"),
        (2048, 2048, "yue2 attn q/o 2048x2048"),
        (1024, 2048, "yue2 attn kv  1024x2048"),
        (6144, 2048, "yue2 mlp up   6144x2048"),
        (2048, 6144, "yue2 mlp down 2048x6144"),
        // The 4090 has 72 MB of L2, so every shape above is CACHE-RESIDENT when a micro-benchmark re-reads one
        // weight 200x in a loop — which is why they report 1500-2900 GB/s, well past the card's ~1 TB/s of DRAM.
        // Real decode streams a 2.82 GB working set, so nothing is resident. These two exceed L2 and are the only
        // rows here that measure the kernel's actual DRAM bandwidth.
        (16384, 4096, "DRAM 16384x4096 (128MB)"),
        (8192, 8192, "DRAM  8192x8192 (128MB)"),
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
        _output.WriteLine($"{"shape",-24}{"bf16 (µs)",11}{"bf16 GB/s",11}{"Q8 Linear",11}{"Q8 QMatMul",12}{"Q8/bf16",9}");

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

            // At M=1 the BF16 weight read dominates, so bytes/time is the effective bandwidth the kernel achieves.
            double bf16GBs = (double)N * K * 2 / (bf16us * 1e-6) / 1e9;
            _output.WriteLine($"{tag,-24}{bf16us,10:F1}{bf16GBs,10:F0}{q8LinUs,10:F1}{q8QmmUs,11:F1}{q8LinUs / bf16us,8:F2}x");
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
