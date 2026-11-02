using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>The query-tiled F32 kernel against the scalar route on random F32 inputs. Dense cases cover head dims 40, 64, 80, 128
/// and 160, both strided layouts, with and without an additive bias, at 512 and 4096 tokens. Head dim 160 has no scalar kernel
/// (sdpa_flash serves 128 and below), so its baseline is the naive row. Causal and window cases run the scalar FlashAttention
/// route, with head dims 40 and 80 to keep the cooperative-matrix prefill out. A mask case also checks that the mask moves the
/// scalar output, so a pass cannot come from both kernels ignoring it.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanTiledAttentionTests(ITestOutputHelper output)
{
    private static bool VulkanAvailable()
    {
        try { using VulkanInstance i = new(); return i.EnumeratePhysicalDevices().Length > 0; }
        catch { return false; }
    }

    private static Tensor Uniform(TensorShape shape, int seed, float amplitude = 1f)
    {
        Tensor t = new(shape, DType.F32);
        System.Random rng = new(seed);
        Span<float> p = t.AsSpan<float>();
        for (int i = 0; i < p.Length; i++) p[i] = amplitude * (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    /// <summary>Head-major [B, H, S, D] to token-major [B, S, H, D]: the same values in the other layout.</summary>
    private static Tensor TokenMajorCopy(Tensor headMajor, int b, int h, int s, int d)
    {
        Tensor t = new(new TensorShape(b, s, h, d), DType.F32);
        ReadOnlySpan<float> src = headMajor.AsReadOnlySpan<float>();
        Span<float> dst = t.AsSpan<float>();
        for (int bi = 0; bi < b; bi++)
            for (int hi = 0; hi < h; hi++)
                for (int si = 0; si < s; si++)
                    for (int di = 0; di < d; di++)
                        dst[((bi * s + si) * h + hi) * d + di] = src[((bi * h + hi) * s + si) * d + di];
        return t;
    }

    /// <summary>The largest absolute error, and the largest relative error against |b| with a 1e-6 floor.</summary>
    private static (float Abs, float Rel) Errors(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float abs = 0, rel = 0;
        for (int i = 0; i < a.Length; i++)
        {
            float e = MathF.Abs(a[i] - b[i]);
            abs = MathF.Max(abs, e);
            rel = MathF.Max(rel, e / MathF.Max(MathF.Abs(b[i]), 1e-6f));
        }
        return (abs, rel);
    }

    public static IEnumerable<object[]> ParityCases()
    {
        foreach (int d in new[] { 40, 64, 80, 128, 160 })
            foreach (int seq in new[] { 512, 4096 })
                foreach (bool tokenMajor in new[] { false, true })
                    foreach (bool withBias in new[] { false, true })
                        yield return new object[] { d, seq, tokenMajor, withBias };
    }

    [Theory]
    [MemberData(nameof(ParityCases))]
    public void Dense_TiledMatchesTheScalarRoute(int d, int seq, bool tokenMajor, bool withBias)
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        const int b = 1, hq = 4, hkv = 2;
        float scale = 1f / MathF.Sqrt(d);
        using Tensor q = Uniform(new TensorShape(b, hq, seq, d), 1);
        using Tensor k = Uniform(new TensorShape(b, hkv, seq, d), 2);
        using Tensor v = Uniform(new TensorShape(b, hkv, seq, d), 3);
        // A full per-row bias at 512; a key-only bias, broadcast over the query rows (row stride 0), at 4096.
        TensorShape biasShape = seq <= 512 ? new TensorShape(b, hq, seq, seq) : new TensorShape(b, hq, 1, seq);
        using Tensor? bias = withBias ? Uniform(biasShape, 4, 0.5f) : null;

        using Tensor expected = new(q.Shape, DType.F32);
        backend.EnableTiledAttention = false;
        backend.ScaledDotProductAttention(expected, q, k, v, bias, scale);
        string scalarRoute = VulkanBackend.LastAttentionRoute ?? "?";
        Assert.Equal(d == 160 ? "A.3" : "A.2", scalarRoute);

        if (withBias)
        {
            using Tensor plain = new(q.Shape, DType.F32);
            backend.ScaledDotProductAttention(plain, q, k, v, null, scale);
            float separation = Errors(expected.AsReadOnlySpan<float>(), plain.AsReadOnlySpan<float>()).Abs;
            Assert.True(separation > 1e-4f, $"d={d} seq={seq}: the bias barely moves the scalar output ({separation})");
        }

        backend.EnableTiledAttention = true;
        float absErr, relErr;
        string tiledRoute;
        if (!tokenMajor)
        {
            using Tensor actual = new(q.Shape, DType.F32);
            backend.ScaledDotProductAttention(actual, q, k, v, bias, scale);
            tiledRoute = VulkanBackend.LastAttentionRoute ?? "?";
            Assert.Equal("A.2t", tiledRoute);
            (absErr, relErr) = Errors(actual.AsReadOnlySpan<float>(), expected.AsReadOnlySpan<float>());
        }
        else
        {
            // Token-major Q/K/V/O through the strides directly: the dense rows only take head-major tensors.
            using Tensor qT = TokenMajorCopy(q, b, hq, seq, d);
            using Tensor kT = TokenMajorCopy(k, b, hkv, seq, d);
            using Tensor vT = TokenMajorCopy(v, b, hkv, seq, d);
            using Tensor expectedT = TokenMajorCopy(expected, b, hq, seq, d);
            using Tensor actualT = new(qT.Shape, DType.F32);
            backend.DispatchFlashTiled(actualT, qT, kT, vT, bias, new VulkanBackend.TiledCall(b, hq, hkv, seq, seq, d, scale,
                Causal: false, QOffset: 0, Window: 0, VulkanBackend.TokenMajorStrides(hq, seq, d), VulkanBackend.TokenMajorStrides(hkv, seq, d),
                VulkanBackend.TokenMajorStrides(hkv, seq, d), VulkanBackend.TokenMajorStrides(hq, seq, d)));
            tiledRoute = "direct token-major";
            (absErr, relErr) = Errors(actualT.AsReadOnlySpan<float>(), expectedT.AsReadOnlySpan<float>());
        }

        output.WriteLine($"dense d={d} seq={seq} {(tokenMajor ? "token" : "head")}-major bias={withBias}: scalar {scalarRoute}, "
            + $"tiled {tiledRoute}, max abs {absErr:E2}, max rel {relErr:E2}");
        Assert.True(absErr < 1e-4f, $"d={d} seq={seq} tokenMajor={tokenMajor} bias={withBias}: max abs error {absErr}");
    }

    /// <summary>(name, head dim, query heads, KV heads, query rows, key rows, query offset, causal, window). Query and key tile
    /// tails are partial in every case.</summary>
    public static IEnumerable<object[]> MaskCases() =>
    [
        ["causal-window-gqa", 40, 4, 2, 100, 260, 160, true, 64],
        ["causal-ragged-80", 80, 2, 2, 37, 101, 64, true, 0],
        ["dense-ragged-40", 40, 2, 1, 50, 70, 0, false, 0],
    ];

    [Theory]
    [MemberData(nameof(MaskCases))]
    public void CausalAndWindow_TiledMatchesTheScalarRoute(string name, int d, int hq, int hkv, int sq, int skv, int qOffset, bool causal,
        int window)
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        float scale = 1f / MathF.Sqrt(d);
        using Tensor q = Uniform(new TensorShape(1, hq, sq, d), 5);
        using Tensor k = Uniform(new TensorShape(1, hkv, skv, d), 6);
        using Tensor v = Uniform(new TensorShape(1, hkv, skv, d), 7);
        using Tensor expected = new(q.Shape, DType.F32);
        using Tensor unmasked = new(q.Shape, DType.F32);
        using Tensor actual = new(q.Shape, DType.F32);
        backend.FlashAttention(expected, q, k, v, kvLen: skv, kvGroup: hq / hkv, causal: causal, qOffset: qOffset, scale: scale,
            slidingWindow: window);
        if (causal || window > 0)
        {
            backend.FlashAttention(unmasked, q, k, v, kvLen: skv, kvGroup: hq / hkv, causal: false, qOffset: 0, scale: scale);
            float separation = Errors(expected.AsReadOnlySpan<float>(), unmasked.AsReadOnlySpan<float>()).Abs;
            Assert.True(separation > 1e-3f, $"{name}: the mask barely moves the scalar output ({separation})");
        }

        backend.DispatchFlashTiled(actual, q, k, v, null, new VulkanBackend.TiledCall(1, hq, hkv, sq, skv, d, scale, causal, qOffset, window,
            VulkanBackend.HeadMajorStrides(hq, sq, d), VulkanBackend.HeadMajorStrides(hkv, skv, d),
            VulkanBackend.HeadMajorStrides(hkv, skv, d), VulkanBackend.HeadMajorStrides(hq, sq, d)));
        (float absErr, float relErr) = Errors(actual.AsReadOnlySpan<float>(), expected.AsReadOnlySpan<float>());
        output.WriteLine($"{name}: d={d} hq={hq} hkv={hkv} sq={sq} skv={skv} qOffset={qOffset} causal={causal} window={window}, "
            + $"max abs {absErr:E2}, max rel {relErr:E2}");
        Assert.True(absErr < 1e-4f, $"{name}: max abs error {absErr}");
    }
}
