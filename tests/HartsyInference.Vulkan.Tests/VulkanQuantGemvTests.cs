using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;
using Xunit;
using Xunit.Abstractions;
using static HartsyInference.Vulkan.Tests.Dsv41TestData;

namespace HartsyInference.Vulkan.Tests;

/// <summary><c>gemv_quant</c> (Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K) against the host dequantizer and a double-precision CPU dot on random
/// blocks, plus the Linear routing around it: decode-sized M takes the GEMV, a longer M the dequantize and GEMM, and a
/// K off the block size is refused by name. Skips without a Vulkan device.</summary>
[Trait("Category", "GpuIntegration")]
public sealed unsafe class VulkanQuantGemvTests(ITestOutputHelper log)
{
    /// <summary>Random GGUF blocks with the half-float scale fields set to small finite values; every other byte random.</summary>
    private static Tensor RandomQuant(DType dtype, int n, int k, int seed)
    {
        Tensor t = new(new TensorShape(n, k), dtype);
        long bytes = dtype.ComputeByteCount((long)n * k);
        byte* p = (byte*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < bytes; i++) p[i] = (byte)rng.Next(256);
        long blocks = (long)n * k / dtype.BlockElementCount;
        int blockBytes = (int)(bytes / blocks);
        for (long b = 0; b < blocks; b++)
        {
            byte* bb = p + b * blockBytes;
            if (dtype == DType.Q8_0 || dtype == DType.Q4_0 || dtype == DType.Q5_0) WriteHalf(bb, 0.002f + 0.01f * (float)rng.NextDouble());
            else if (dtype == DType.Q4_K || dtype == DType.Q5_K)
            {
                WriteHalf(bb, 0.0005f + 0.002f * (float)rng.NextDouble());
                WriteHalf(bb + 2, 0.0005f + 0.002f * (float)rng.NextDouble());
            }
            else WriteHalf(bb + 208, 0.0002f + 0.001f * (float)rng.NextDouble());
        }
        return t;
    }

    private static void WriteHalf(byte* dst, float value)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((Half)value);
        dst[0] = (byte)bits;
        dst[1] = (byte)(bits >> 8);
    }

    /// <summary>out[m, n] in double from the host-dequantized weight, and Σ|w||x| per output for the tolerance.</summary>
    private static (float[] Ref, float[] Mag) Reference(Tensor weight, float[] x, float[]? bias, int n, int k, int m)
    {
        using Tensor w = GgufDequantizer.Dequantize(weight, DType.F32);
        float* wp = (float*)w.DataPointer;
        float[] r = new float[m * n], mag = new float[m * n];
        for (int mi = 0; mi < m; mi++)
        {
            for (int ni = 0; ni < n; ni++)
            {
                double s = 0, a = 0;
                for (int ki = 0; ki < k; ki++)
                {
                    double prod = (double)wp[(long)ni * k + ki] * x[mi * k + ki];
                    s += prod;
                    a += Math.Abs(prod);
                }
                if (bias is not null) { s += bias[ni]; a += Math.Abs(bias[ni]); }
                r[mi * n + ni] = (float)s;
                mag[mi * n + ni] = (float)a;
            }
        }
        return (r, mag);
    }

    private void AssertClose(float[] expected, float[] mag, float[] actual, double relTol, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double err = Math.Abs(expected[i] - actual[i]);
            double bound = relTol * mag[i] + 1e-6;
            worst = Math.Max(worst, err / bound);
            Assert.True(err <= bound, $"{what} [{i}]: ref {expected[i]:R} vk {actual[i]:R} (|w||x| {mag[i]:R})");
        }
        log.WriteLine($"{what}: worst error {worst:F3} of the bound");
    }

    public static IEnumerable<object[]> Cases()
    {
        // type, N, K, M, bias. N odd everywhere; an odd block count per row (Q8_0, Q4_0, Q5_0 at K=32/96, Q6_K at 256) leaves rows unaligned.
        yield return new object[] { "Q8_0", 7, 32, 1, false };
        yield return new object[] { "Q8_0", 33, 96, 3, true };
        yield return new object[] { "Q8_0", 4097, 2048, 1, false };
        yield return new object[] { "Q4_K", 5, 256, 1, false };
        yield return new object[] { "Q4_K", 31, 768, 8, true };
        yield return new object[] { "Q4_K", 4097, 2048, 1, false };
        yield return new object[] { "Q6_K", 3, 256, 1, false };
        yield return new object[] { "Q6_K", 17, 512, 8, true };
        yield return new object[] { "Q6_K", 4097, 2048, 2, false };
        yield return new object[] { "Q4_0", 9, 32, 1, false };
        yield return new object[] { "Q4_0", 4097, 2048, 4, true };
        yield return new object[] { "Q5_0", 11, 96, 1, false };
        yield return new object[] { "Q5_0", 4097, 896, 3, true };
        yield return new object[] { "Q5_K", 3, 256, 1, false };
        yield return new object[] { "Q5_K", 4097, 2048, 8, true };
    }

    private static DType Named(string name) => name switch
    {
        "Q8_0" => DType.Q8_0,
        "Q4_0" => DType.Q4_0,
        "Q5_0" => DType.Q5_0,
        "Q4_K" => DType.Q4_K,
        "Q5_K" => DType.Q5_K,
        _ => DType.Q6_K,
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void Linear_DecodeM_TakesTheGemvAndMatchesHostDequant(string type, int n, int k, int m, bool withBias)
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        log.WriteLine($"device: {vk.Vk.DeviceName}");
        DType dtype = Named(type);
        using Tensor w = RandomQuant(dtype, n, k, seed: n * 31 + k);
        float[] x = Random(m * k, seed: k + m, scale: 1f);
        float[]? bias = withBias ? Random(n, seed: n + 5, scale: 0.5f) : null;
        (float[] expected, float[] mag) = Reference(w, x, bias, n, k, m);

        using Tensor xt = F32(x, 1, m, k);
        using Tensor? bt = bias is null ? null : F32(bias, n);
        using Tensor o = EmptyF32(1, m, n);
        long before = vk.QuantGemvCount;
        vk.Linear(o, xt, w, bt);
        float[] got = ReadF32(o);
        Assert.Equal(before + 1, vk.QuantGemvCount);
        AssertClose(expected, mag, got, 2e-5, $"{type} N={n} K={k} M={m} bias={withBias}");
    }

    [Theory]
    [InlineData("Q8_0")]
    [InlineData("Q4_0")]
    [InlineData("Q5_0")]
    [InlineData("Q4_K")]
    [InlineData("Q5_K")]
    [InlineData("Q6_K")]
    public void Linear_PrefillM_TakesTheDequantizeAndGemm(string type)
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        DType dtype = Named(type);
        const int N = 64, K = 512, M = 24;
        using Tensor w = RandomQuant(dtype, N, K, seed: 7);
        float[] x = Random(M * K, seed: 9);
        (float[] expected, float[] mag) = Reference(w, x, null, N, K, M);
        using Tensor xt = F32(x, 1, M, K);
        using Tensor o = EmptyF32(1, M, N);
        long before = vk.QuantGemvCount;
        vk.Linear(o, xt, w, null);
        float[] got = ReadF32(o);
        Assert.Equal(before, vk.QuantGemvCount);
        // The prefill GEMM computes in F16 with an F32 accumulator (numerics.vkF16Gemm), so the bound is the F16 one.
        AssertClose(expected, mag, got, 2e-3, $"{type} prefill M={M}");
    }

    [Fact]
    public void QuantizedMatMul_F16Activation_MatchesHostDequant()
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        const int N = 129, K = 256;
        using Tensor w = RandomQuant(DType.Q6_K, N, K, seed: 3);
        float[] x = Random(K, seed: 4);
        for (int i = 0; i < x.Length; i++) x[i] = (float)(Half)x[i];
        (float[] expected, float[] mag) = Reference(w, x, null, N, K, 1);
        using Tensor xt = F32(x, 1, 1, K);
        using Tensor xh = xt.CastTo(DType.F16);
        using Tensor o = EmptyF32(1, 1, N);
        vk.QuantizedMatMul(o, xh, w, null);
        AssertClose(expected, mag, ReadF32(o), 2e-3, "QuantizedMatMul F16 activation");
    }

    /// <summary>A LoRA on a packed weight rides as an adjunct; both entry points add it after the GEMV.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Gemv_AppliesTheLowRankAdjunct(bool viaQuantizedMatMul)
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        const int N = 65, K = 256, Rank = 4;
        const float DownValue = 0.01f, UpValue = 0.02f;
        using Tensor w = RandomQuant(DType.Q8_0, N, K, seed: 11);
        using Tensor down = F32(Enumerable.Repeat(DownValue, Rank * K).ToArray(), Rank, K);
        using Tensor up = F32(Enumerable.Repeat(UpValue, N * Rank).ToArray(), N, Rank);
        using Tensor patched = w.WithLowRankAdjunct(new LowRankAdjunct
        {
            Terms = [new LowRankAdjunctTerm { Down = down, Up = up, Scale = 1.0f }],
        });
        float[] x = Random(K, seed: 12);
        (float[] expected, float[] mag) = Reference(w, x, null, N, K, 1);
        float delta = Rank * DownValue * UpValue * x.Sum();
        for (int i = 0; i < N; i++) { expected[i] += delta; mag[i] += Math.Abs(delta); }

        using Tensor xt = F32(x, 1, 1, K);
        using Tensor o = EmptyF32(1, 1, N);
        long before = vk.QuantGemvCount;
        if (viaQuantizedMatMul) vk.QuantizedMatMul(o, xt, patched, null);
        else vk.Linear(o, xt, patched, null);
        float[] got = ReadF32(o);
        Assert.Equal(before + 1, vk.QuantGemvCount);
        AssertClose(expected, mag, got, 2e-5, $"adjunct via {(viaQuantizedMatMul ? "QuantizedMatMul" : "Linear")}");
    }

    [Theory]
    [InlineData("Q8_0", 48, 1)]
    [InlineData("Q4_K", 300, 1)]
    [InlineData("Q6_K", 512 + 32, 1)]
    [InlineData("Q5_0", 40, 1)]
    [InlineData("Q5_K", 128, 1)]
    public void Gemv_KOffTheBlock_IsRefusedByName(string type, int k, int m)
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        ArgumentException ex = Assert.Throws<ArgumentException>(() => vk.DispatchQuantGemv(Named(type), 1, 1, 1, 0, 4, k, m));
        Assert.Contains("not a multiple", ex.Message);
    }

    [Fact]
    public void Gemv_TooManyRows_IsRefused()
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        Assert.Throws<ArgumentOutOfRangeException>(() => vk.DispatchQuantGemv(DType.Q8_0, 1, 1, 1, 0, 4, 32, VulkanBackend.QuantGemvMaxRows + 1));
    }
}
