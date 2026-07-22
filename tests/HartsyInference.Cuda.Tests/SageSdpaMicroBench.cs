using System.Diagnostics;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Quick warm A/B of the SageAttention INT8 SDPA path vs the default F32 dispatch at DiT shapes
/// (INFERENCE_ACCEL_GRIND §H4 gate 2 — preliminary; the committed claim runs through SdpaGpuBenchmarks/BDN).
/// Same-process, 1 warmup + 5 trials per side, Welch's t printed; the quant prologues are inside the Sage
/// timing because every real forward pays them. Run explicitly:
///   dotnet test --filter "FullyQualifiedName~SageSdpaMicroBench" (Category=SageBench, excluded from sweeps)</summary>
[Collection("CudaSerial")]
[Trait("Category", "SageBench")]
public sealed unsafe class SageSdpaMicroBench
{
    private readonly ITestOutputHelper _output;
    public SageSdpaMicroBench(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor RandomF32(TensorShape shape, int seed)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        long n = shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return t;
    }

    [Theory]
    [InlineData(1, 24, 4096, 128)]     // Qwen/Flux-class image self-attn
    [InlineData(1, 24, 12288, 128)]    // video-DiT-class self-attn (Wan/LTX territory, 3060-sized)
    public void SageInt8_Vs_DefaultF32(int b, int h, int s, int d)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (!File.Exists(Path.Combine(PtxDir(), "sage_attn_int8.ptx"))) { _output.WriteLine("SKIPPED: no sage ptx"); return; }

        float scale = 1.0f / MathF.Sqrt(d);
        using Tensor q = RandomF32(new TensorShape(b, h, s, d), 11);
        using Tensor k = RandomF32(new TensorShape(b, h, s, d), 12);
        using Tensor v = RandomF32(new TensorShape(b, h, s, d), 13);
        using Tensor outA = new Tensor(new TensorShape(b, h, s, d), DType.F32);
        using Tensor outB = new Tensor(new TensorShape(b, h, s, d), DType.F32);

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        const int Trials = 5;

        double[] Run(Tensor outT, bool sage)
        {
            Environment.SetEnvironmentVariable("HARTSY_SAGE_ATTN", sage ? "1" : null);
            try
            {
                cuda.ScaledDotProductAttention(outT, q, k, v, null, scale);   // warmup (JIT, pools)
                cuda.Sync();
                double[] times = new double[Trials];
                Stopwatch sw = new Stopwatch();
                for (int t = 0; t < Trials; t++)
                {
                    sw.Restart();
                    cuda.ScaledDotProductAttention(outT, q, k, v, null, scale);
                    cuda.Sync();
                    sw.Stop();
                    times[t] = sw.Elapsed.TotalMilliseconds;
                }
                return times;
            }
            finally
            {
                Environment.SetEnvironmentVariable("HARTSY_SAGE_ATTN", null);
            }
        }

        double[] baseT = Run(outA, sage: false);
        double[] sageT = Run(outB, sage: true);

        static (double mean, double var) Stats(double[] xs)
        {
            double m = xs.Average();
            double v = xs.Length > 1 ? xs.Sum(x => (x - m) * (x - m)) / (xs.Length - 1) : 0;
            return (m, v);
        }
        (double mb, double vb) = Stats(baseT);
        (double ms, double vs) = Stats(sageT);
        double se = Math.Sqrt(vb / Trials + vs / Trials);
        double tStat = se > 0 ? (mb - ms) / se : double.PositiveInfinity;

        _output.WriteLine($"[{b},{h},{s},{d}] default F32: {mb:F2} ms (σ {Math.Sqrt(vb):F2})  [{string.Join(", ", baseT.Select(x => x.ToString("F1")))}]");
        _output.WriteLine($"[{b},{h},{s},{d}] sage INT8 : {ms:F2} ms (σ {Math.Sqrt(vs):F2})  [{string.Join(", ", sageT.Select(x => x.ToString("F1")))}]");
        _output.WriteLine($"speedup {mb / ms:F2}×, Welch t = {tStat:F1} (|t| > ~4.6 ≈ significant at α=0.01, df≈4)");
    }
}
