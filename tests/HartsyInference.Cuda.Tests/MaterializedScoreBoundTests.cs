using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;
using static HartsyInference.Cuda.Tests.CudnnSdpaVariantTests;

namespace HartsyInference.Cuda.Tests;

/// <summary>A head-major F16 or BF16 call past the 2^32 score elements the materialized row addresses, with cuDNN and Sage off so no
/// fused row takes it: it runs query-tiled on F32 widenings and every sampled row matches a float64 oracle, the rows whose 32-bit
/// offsets would wrap included.</summary>
[Collection("CudaSerial")]
public sealed unsafe class MaterializedScoreBoundTests
{
    private readonly ITestOutputHelper _output;
    public MaterializedScoreBoundTests(ITestOutputHelper output) => _output = output;

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData("f16")]
    [InlineData("bf16")]
    public void AHalfPrecisionCallPastTheBound_RunsTiled_AndMatchesTheOracleOnTheRowsThatWouldWrap(string dtypeName)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int s = 66000, d = 64;
        DType dtype = dtypeName == "bf16" ? DType.BF16 : DType.F16;
        double tolerance = dtype == DType.BF16 ? Bf16Tolerance : F16Tolerance;
        long firstWrapped = CudaBackend.MaterializedMaxScores / s + 1;
        float scale = 1f / MathF.Sqrt(d);
        Random rng = new(66000);
        float[] q = Quantize(Gauss(rng, (long)s * d, 1.7f), dtype), k = Quantize(Gauss(rng, (long)s * d, 1.7f), dtype);
        float[] v = Quantize(Gauss(rng, (long)s * d), dtype);
        TensorShape shape = new(1, 1, s, d);
        float[] got;
        string? route;
        KnobStore.Set(EngineKnobs.SdpaCudnn, false);
        KnobStore.Set(EngineKnobs.SageAttn, false);
        try
        {
            using CudaBackend backend = new(0, PtxDir());
            using Tensor qt = Make(q, shape, dtype), kt = Make(k, shape, dtype), vt = Make(v, shape, dtype);
            using Tensor ot = new(shape, dtype);
            backend.ScaledDotProductAttention(ot, qt, kt, vt, null, scale, allowF16: false);
            backend.Sync();
            route = CudaBackend.LastAttentionRoute;
            got = Read(ot);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.SdpaCudnn);
            KnobStore.Clear(EngineKnobs.SageAttn);
        }

        int[] rows = [0, 1, 7, 14, 1000, s / 2, (int)firstWrapped - 1, (int)firstWrapped, s - 400, s - 1];
        double worst = 0;
        foreach (int row in rows)
        {
            double err = RowError(got, q, k, v, row, s, d, scale);
            worst = Math.Max(worst, double.IsNaN(err) ? double.PositiveInfinity : err);
            _output.WriteLine($"row {row}{(row >= firstWrapped ? " (past a 32-bit offset)" : "")}: relL2 {err:E2}");
        }
        _output.WriteLine($"{dtypeName}: route {route}, worst relL2 {worst:E2}");
        Assert.Equal("S.11", route);
        Assert.True(worst <= tolerance, $"worst sampled row relL2 {worst:E2}");
    }

    /// <summary>Relative L2 error of one output row against float64 softmax(scale·qKᵀ)·V over every key; NaN when the row is not finite.</summary>
    private static double RowError(float[] got, float[] q, float[] k, float[] v, int row, int s, int d, float scale)
    {
        double[] p = new double[s];
        double max = double.NegativeInfinity;
        for (int j = 0; j < s; j++)
        {
            double dot = 0;
            for (int e = 0; e < d; e++) dot += (double)q[(long)row * d + e] * k[(long)j * d + e];
            p[j] = dot * scale;
            max = Math.Max(max, p[j]);
        }
        double sum = 0;
        for (int j = 0; j < s; j++) { p[j] = Math.Exp(p[j] - max); sum += p[j]; }
        double err = 0, norm = 0;
        for (int e = 0; e < d; e++)
        {
            double want = 0;
            for (int j = 0; j < s; j++) want += p[j] * v[(long)j * d + e];
            want /= sum;
            float value = got[(long)row * d + e];
            if (!float.IsFinite(value)) return double.NaN;
            err += (value - want) * (value - want);
            norm += want * want;
        }
        return Math.Sqrt(err) / Math.Max(Math.Sqrt(norm), 1e-12);
    }
}
