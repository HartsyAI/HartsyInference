using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>The engine's F16 flash kernel (flash_attn_f16.ptx) at head dim 256 against a double-precision CPU
/// reference. Sequence lengths are not tile multiples, so the zero-filled K/V tail and the unstored query rows are
/// exercised; one case puts the largest scores in the last key tile, so the running max rises mid-row and the O
/// rescale has to be right. Skips cleanly without CUDA or without the PTX.</summary>
[Collection("CudaSerial")]
public sealed unsafe class FlashAttnF16Tests(ITestOutputHelper output)
{
    private const int D = 256;

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(1, 2, 100, 77, false)]
    [InlineData(1, 2, 90, 100, true)]
    public void HeadMajor_MatchesCpuReference(int batch, int heads, int sq, int skv, bool risingMax)
    {
        using CudaBackend? backend = Create();
        if (backend is null) return;
        float[] q = Random(batch * heads * sq * D, 11), k = Random(batch * heads * skv * D, 12), v = Random(batch * heads * skv * D, 13);
        if (risingMax) RaiseLastTileScores(q, k, batch * heads, sq, skv);
        using Tensor qt = F16(q, batch, heads, sq), kt = F16(k, batch, heads, skv), vt = F16(v, batch, heads, skv);
        using Tensor ot = new(new TensorShape(batch, heads, sq, D), DType.F16);
        ((IBackend)backend).ScaledDotProductAttention(ot, qt, kt, vt, null, 1f / 16f, allowF16: true);
        backend.Sync();

        Assert.Equal(1, backend.FlashF16ExecutionCount);
        double[] expected = Reference(q, k, v, batch * heads, sq, skv);
        AssertClose(expected, ToFloat(ot), $"B={batch} H={heads} Sq={sq} Skv={skv} risingMax={risingMax}");
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TokenMajor_MatchesCpuReference(bool rank4)
    {
        using CudaBackend? backend = Create();
        if (backend is null) return;
        const int heads = 3, s = 130;
        float[] q = Random(heads * s * D, 21), k = Random(heads * s * D, 22), v = Random(heads * s * D, 23);
        using Tensor qt = TokenMajor(q, heads, s, rank4), kt = TokenMajor(k, heads, s, rank4), vt = TokenMajor(v, heads, s, rank4);
        using Tensor ot = new(new TensorShape(s, heads * D), DType.F16);
        ((IBackend)backend).ScaledDotProductAttentionTokenMajor(ot, qt, kt, vt, null, heads, D, 1f / 16f, allowF16: true);
        backend.Sync();

        Assert.Equal(1, backend.FlashF16ExecutionCount);
        double[] headMajor = Reference(q, k, v, heads, s, s);
        float[] actual = ToFloat(ot);
        double[] expected = new double[actual.Length];
        for (int h = 0; h < heads; h++)
            for (int i = 0; i < s; i++)
                for (int e = 0; e < D; e++)
                    expected[((long)i * heads + h) * D + e] = headMajor[((long)h * s + i) * D + e];
        AssertClose(expected, actual, $"token-major rank{(rank4 ? 4 : 2)}");
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void KnobOff_LeavesTheCallToCudnn()
    {
        KnobStore.Set(EngineKnobs.FlashF16, false);
        try
        {
            using CudaBackend? backend = Create();
            if (backend is null) return;
            float[] q = Random(2 * 64 * D, 31);
            using Tensor qt = F16(q, 1, 2, 64);
            using Tensor ot = new(new TensorShape(1, 2, 64, D), DType.F16);
            ((IBackend)backend).ScaledDotProductAttention(ot, qt, qt, qt, null, 1f / 16f, allowF16: true);
            backend.Sync();
            Assert.Equal(0, backend.FlashF16ExecutionCount);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.FlashF16);
        }
    }

    /// <summary>Ideogram 4's attention shape, flash kernel against cuDNN. Prints both; asserts only agreement.</summary>
    private CudaBackend? Create()
    {
        if (!CudaContext.IsAvailable()) { output.WriteLine("SKIPPED: CUDA unavailable"); return null; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        if (!File.Exists(Path.Combine(ptxDir, "flash_attn_f16.ptx"))) { output.WriteLine("SKIPPED: flash_attn_f16.ptx absent"); return null; }
        return new CudaBackend(0, ptxDir);
    }

    private static float[] Random(int n, int seed)
    {
        Random rng = new(seed);
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    /// <summary>Points the last key tile along each query row's direction, so those scores dominate and arrive last.</summary>
    private static void RaiseLastTileScores(float[] q, float[] k, int planes, int sq, int skv)
    {
        for (int p = 0; p < planes; p++)
            for (int j = skv - 8; j < skv; j++)
                for (int e = 0; e < D; e++)
                    k[((long)p * skv + j) * D + e] = 3f * q[((long)p * sq + (j % sq)) * D + e];
    }

    private static Tensor F16(float[] values, int batch, int heads, int rows)
    {
        Tensor t = new(new TensorShape(batch, heads, rows, D), DType.F16);
        Fill(t, values);
        return t;
    }

    private static Tensor TokenMajor(float[] headMajor, int heads, int rows, bool rank4)
    {
        Tensor t = rank4 ? new(new TensorShape(1, rows, heads, D), DType.F16) : new(new TensorShape(rows, heads * D), DType.F16);
        float[] tm = new float[headMajor.Length];
        for (int h = 0; h < heads; h++)
            for (int i = 0; i < rows; i++)
                for (int e = 0; e < D; e++)
                    tm[((long)i * heads + h) * D + e] = headMajor[((long)h * rows + i) * D + e];
        Fill(t, tm);
        return t;
    }

    private static void Fill(Tensor t, float[] values)
    {
        Half* p = (Half*)t.DataPointer;
        for (int i = 0; i < values.Length; i++) p[i] = (Half)values[i];
    }

    private static float[] ToFloat(Tensor t)
    {
        Half* p = (Half*)t.DataPointer;
        float[] a = new float[t.ElementCount];
        for (int i = 0; i < a.Length; i++) a[i] = (float)p[i];
        return a;
    }

    /// <summary>softmax(Q·Kᵀ/16)·V per plane in double, from the F16-rounded inputs the kernel actually sees.</summary>
    private static double[] Reference(float[] q, float[] k, float[] v, int planes, int sq, int skv)
    {
        double[] o = new double[(long)planes * sq * D];
        double[] row = new double[skv];
        for (int p = 0; p < planes; p++)
            for (int i = 0; i < sq; i++)
            {
                double max = double.NegativeInfinity;
                for (int j = 0; j < skv; j++)
                {
                    double dot = 0;
                    for (int e = 0; e < D; e++)
                        dot += (double)(float)(Half)q[((long)p * sq + i) * D + e] * (float)(Half)k[((long)p * skv + j) * D + e];
                    row[j] = dot / 16.0;
                    max = Math.Max(max, row[j]);
                }
                double sum = 0;
                for (int j = 0; j < skv; j++) { row[j] = Math.Exp(row[j] - max); sum += row[j]; }
                for (int e = 0; e < D; e++)
                {
                    double acc = 0;
                    for (int j = 0; j < skv; j++) acc += row[j] * (float)(Half)v[((long)p * skv + j) * D + e];
                    o[((long)p * sq + i) * D + e] = acc / sum;
                }
            }
        return o;
    }

    private void AssertClose(double[] expected, float[] actual, string what)
    {
        double maxDiff = 0;
        for (int i = 0; i < expected.Length; i++) maxDiff = Math.Max(maxDiff, Math.Abs(expected[i] - actual[i]));
        output.WriteLine($"{what}: max |ref - flash| = {maxDiff:E3}");
        // F16 output rounding (values up to ~1, half-ulp ~5e-4) plus the F16 P operand of the PV product.
        Assert.True(maxDiff < 3e-3, $"{what}: flash diverges from the reference by {maxDiff:E3}");
    }
}
