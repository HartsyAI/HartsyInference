using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Measures achieved throughput of the resident int8-ConvRot <c>Linear</c> at LTX-2.5's real DiT shapes,
/// so a perf pass can tell "the GEMM is at the hardware wall" apart from "the GEMM is leaving throughput on the
/// table". Reports TOPS against the RTX 4090's ~330 TOPS dense INT8 tensor-core peak. Diagnostic, not a gate —
/// it asserts only that the path ran, and prints the numbers for a human to read.</summary>
[Collection("CudaSerial")]
public sealed unsafe class Int8ConvRotGemmThroughputTests
{
    private readonly ITestOutputHelper _output;
    public Int8ConvRotGemmThroughputTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    /// <summary>Builds an int8 weight carrying the per-row scale + ConvRot group size the resident path requires,
    /// exactly as <c>LtxVideo2CheckpointConverter</c> hands them over.</summary>
    private static Tensor Int8Weight(int n, int k, int seed)
    {
        Tensor w = new Tensor(new TensorShape(n, k), DType.I8);
        sbyte* p = (sbyte*)w.DataPointer;
        Random rng = new Random(seed);
        for (long i = 0; i < (long)n * k; i++) p[i] = (sbyte)(rng.Next(-127, 128));
        Tensor scale = new Tensor(new TensorShape(n, 1), DType.F32);
        float* s = (float*)scale.DataPointer;
        for (int i = 0; i < n; i++) s[i] = 0.01f;
        w.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = scale, ConvRotGroupSize = 256 };
        return w;
    }

    private static Tensor Act(int m, int k, DType dtype, int seed)
    {
        Tensor f32 = new Tensor(new TensorShape(m, k), DType.F32);
        float* p = (float*)f32.DataPointer;
        Random rng = new Random(seed);
        for (long i = 0; i < (long)m * k; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        if (dtype == DType.F32) return f32;
        Tensor cast = f32.CastTo(dtype);
        f32.Dispose();
        return cast;
    }

    [Theory]
    // The three shapes that carry LTX-2.5's DiT arithmetic at 768x512x97f (4992 video tokens):
    [InlineData(4992, 16384, 4096, "ffn_up")]
    [InlineData(4992, 4096, 16384, "ffn_down")]
    [InlineData(4992, 4096, 4096, "attn_qkvo")]
    // The same shapes with the CFG pair batched into one forward (m doubles) — sizes that lever.
    [InlineData(9984, 16384, 4096, "ffn_up_cfg2")]
    [InlineData(9984, 4096, 16384, "ffn_down_cfg2")]
    [InlineData(9984, 4096, 4096, "attn_qkvo_cfg2")]
    public void ResidentInt8Linear_AchievedThroughput(int m, int n, int k, string label)
    {
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor w = Int8Weight(n, k, seed: 7);
        using Tensor x = Act(m, k, DType.F16, seed: 11);
        using Tensor y = new Tensor(new TensorShape(m, n), DType.F16);

        cuda.PreloadWeights([w]);
        for (int i = 0; i < 3; i++) cuda.Linear(y, x, w, null);   // warm: JIT, row-scale upload, algo heuristic
        cuda.Sync();

        const int Reps = 20;
        System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < Reps; i++) cuda.Linear(y, x, w, null);
        cuda.Sync();
        sw.Stop();

        double perCallMs = sw.Elapsed.TotalMilliseconds / Reps;
        double flop = 2.0 * m * n * k;
        double tops = flop / (perCallMs * 1e-3) / 1e12;
        _output.WriteLine($"{label,-10} m={m} n={n} k={k}  {perCallMs:F3} ms/call  {tops:F1} TOPS " +
            $"({tops / 330.0 * 100:F0}% of the 4090's ~330 TOPS dense INT8 peak)");
        Assert.True(perCallMs > 0);
    }

    /// <summary>The GELU folded into the int8 dequant epilogue must match a plain Linear followed by the
    /// engine's own <c>Gelu</c> — the two are compared on the same weights and input, so a wrong activation
    /// constant or a mis-plumbed actMode flag fails here rather than as slightly-off video.</summary>
    [Fact]
    public void LinearGelu_FusedEpilogue_MatchesLinearThenGelu()
    {
        const int m = 64, n = 256, k = 512;
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor w = Int8Weight(n, k, seed: 41);
        using Tensor x = Act(m, k, DType.F16, seed: 42);
        cuda.PreloadWeights([w]);

        using Tensor viaFused = new Tensor(new TensorShape(m, n), DType.F16);
        cuda.LinearGelu(viaFused, x, w, null);

        using Tensor viaSeq = new Tensor(new TensorShape(m, n), DType.F16);
        cuda.Linear(viaSeq, x, w, null);
        cuda.Gelu(viaSeq, viaSeq);
        cuda.Sync();

        using Tensor a = viaFused.CastTo(DType.F32);
        using Tensor b = viaSeq.CastTo(DType.F32);
        float* pa = (float*)a.DataPointer;
        float* pb = (float*)b.DataPointer;
        double maxErr = 0;
        for (long i = 0; i < a.ElementCount; i++) maxErr = Math.Max(maxErr, Math.Abs(pa[i] - pb[i]));
        _output.WriteLine($"fused-epilogue GELU vs Linear+Gelu: max abs err {maxErr}");
        Assert.True(maxErr <= 5e-3, $"max abs err {maxErr} > 5e-3");
    }
}
