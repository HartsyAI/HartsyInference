using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Vulkan attention against the contract reference (<see cref="AttentionHostReference"/>) on the shapes its kernels
/// used to get wrong: per-head and per-batch masks, grouped K/V on the naive path, F32 rows into an F16 cache, a window
/// without causal, and F16 narrowing without the permission. Every mask case carries a negative control: the reference
/// with every mask block replaced by block 0 must differ, or the case could not see the bug it exists for.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanAttentionCorrectnessTests(ITestOutputHelper output)
{
    private static bool VulkanAvailable()
    {
        try { using VulkanInstance i = new(); return i.EnumeratePhysicalDevices().Length > 0; }
        catch { return false; }
    }

    private static Tensor Filled(TensorShape shape, Func<long, float> f)
    {
        Tensor t = new(shape, DType.F32);
        Span<float> p = t.AsSpan<float>();
        for (int i = 0; i < p.Length; i++) p[i] = f(i);
        return t;
    }

    private static Tensor Random(TensorShape shape, int seed, bool halfRounded)
    {
        Random rng = new(seed);
        return Filled(shape, _ =>
        {
            float x = (float)(rng.NextDouble() * 2 - 1);
            return halfRounded ? (float)(Half)x : x;
        });
    }

    private static float MaxAbs(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float m = 0;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    /// <summary>(name, B, Hq, Hkv, Sq, Skv, D, mask shape, F16 permission). D 160 reaches the naive path; F16 permission with
    /// head dim 64 or 128 reaches cm2 where the device has it, the scalar kernel otherwise.</summary>
    public static IEnumerable<object[]> MaskCases() =>
    [
        ["t5-perhead", 1, 4, 4, 37, 37, 64, new[] { 1, 4, 37, 37 }, false],
        ["swin-batch-head", 2, 3, 3, 16, 16, 32, new[] { 2, 3, 16, 16 }, false],
        ["batch-only", 2, 3, 3, 20, 20, 64, new[] { 2, 1, 20, 20 }, false],
        ["perhead-keyonly", 1, 4, 4, 24, 24, 64, new[] { 1, 4, 1, 24 }, false],
        ["rank3-perhead", 1, 3, 3, 19, 19, 64, new[] { 3, 19, 19 }, false],
        ["naive-d160-batch", 2, 2, 2, 12, 12, 160, new[] { 2, 1, 12, 12 }, false],
        ["naive-d160-gqa", 2, 4, 2, 12, 12, 160, new[] { 2, 1, 12, 12 }, false],
        ["naive-d160-keyonly", 2, 2, 2, 12, 12, 160, new[] { 2, 2, 1, 12 }, false],
        ["cm2-perhead", 1, 4, 4, 70, 70, 64, new[] { 1, 4, 70, 70 }, true],
        ["cm2-batch", 2, 2, 2, 65, 65, 128, new[] { 2, 1, 65, 65 }, true],
        ["cm2-keyonly-perhead", 1, 4, 2, 66, 66, 64, new[] { 1, 4, 1, 66 }, true],
    ];

    [Theory]
    [MemberData(nameof(MaskCases))]
    public void BroadcastMasks_MatchTheReference(string name, int b, int hq, int hkv, int sq, int skv, int d, int[] maskDims, bool allowF16)
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        TensorShape maskShape = maskDims.Length == 4 ? new(maskDims[0], maskDims[1], maskDims[2], maskDims[3]) : new(maskDims[0], maskDims[1], maskDims[2]);
        int rank = maskDims.Length, mq = maskDims[rank - 2], mh = rank >= 3 ? maskDims[rank - 3] : 1;
        using Tensor mask = Filled(maskShape, i =>
        {
            long k = i % skv, q = i / skv % mq, h = i / skv / mq % mh, bb = i / skv / mq / mh;
            return 0.7f * MathF.Sin(bb * 3f + h * 1.3f + q * 0.11f + k * 0.07f);
        });
        using Tensor block0 = Filled(new TensorShape(1, 1, mq, skv), i => mask.AsReadOnlySpan<float>()[(int)i]);
        using Tensor q = Random(new TensorShape(b, hq, sq, d), 1, allowF16), k = Random(new TensorShape(b, hkv, skv, d), 2, allowF16);
        using Tensor v = Random(new TensorShape(b, hkv, skv, d), 3, allowF16);
        using Tensor expected = new(q.Shape, DType.F32), control = new(q.Shape, DType.F32), actual = new(q.Shape, DType.F32);
        float scale = 1f / MathF.Sqrt(d);

        AttentionHostReference.Run(AttentionCall.FromScaledDotProductAttention(expected, q, k, v, mask, scale, allowF16: false));
        AttentionHostReference.Run(AttentionCall.FromScaledDotProductAttention(control, q, k, v, block0, scale, allowF16: false));
        float separation = MaxAbs(expected.AsReadOnlySpan<float>(), control.AsReadOnlySpan<float>());
        Assert.True(separation > 1e-2f, $"{name}: the per-block mask barely moves the output ({separation}); the case cannot see the bug");

        backend.ScaledDotProductAttention(actual, q, k, v, mask, scale, allowF16);
        float error = MaxAbs(actual.AsReadOnlySpan<float>(), expected.AsReadOnlySpan<float>());
        output.WriteLine($"{name}: route {VulkanBackend.LastAttentionRoute}, max error {error:E2}, separation {separation:E2}");
        Assert.True(error < (allowF16 ? 5e-3f : 2e-4f), $"{name}: max error {error} on route {VulkanBackend.LastAttentionRoute}");
    }

    [Fact]
    public void F32RowsIntoAnF16Cache_ConvertAndSaturate()
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        const int heads = 2, maxSeq = 6, d = 5, offset = 2;
        float[] rows = [1.5f, -3.25f, 7e4f, -7e4f, float.NaN];
        using Tensor cache = new(new TensorShape(1, heads, maxSeq, d), DType.F16);
        Span<Half> c = cache.AsSpan<Half>();
        for (int i = 0; i < c.Length; i++) c[i] = (Half)9f;
        using Tensor newKv = Filled(new TensorShape(1, heads, 1, d), i => rows[i % d]);
        backend.KvCacheAppend(cache, newKv, offset);
        ReadOnlySpan<Half> got = cache.AsReadOnlySpan<Half>();
        for (int h = 0; h < heads; h++)
            for (int t = 0; t < maxSeq; t++)
                for (int j = 0; j < d; j++)
                {
                    Half value = got[(h * maxSeq + t) * d + j];
                    if (t != offset) { Assert.Equal((Half)9f, value); continue; }
                    float want = rows[j];
                    if (float.IsNaN(want)) Assert.True(Half.IsNaN(value), $"NaN at head {h} dim {j} became {value}");
                    else Assert.Equal((Half)Math.Clamp(want, -65504f, 65504f), value);
                }
    }

    [Fact]
    public void F16RowsIntoAnF32Cache_AreRefused()
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        using Tensor cache = new(new TensorShape(1, 2, 4, 8), DType.F32);
        using Tensor newKv = new(new TensorShape(1, 2, 1, 8), DType.F16);
        Assert.Throws<NotSupportedException>(() => backend.KvCacheAppend(cache, newKv, 0));
    }

    /// <summary>A window without causal drops only keys too far behind the query, as on CUDA and in the reference.</summary>
    [Fact]
    public void AWindowWithoutCausal_KeepsLaterKeys()
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        const int hq = 4, hkv = 2, sq = 9, skv = 9, d = 64, window = 3;
        using Tensor q = Random(new TensorShape(1, hq, sq, d), 1, false), k = Random(new TensorShape(1, hkv, skv, d), 2, false);
        using Tensor v = Random(new TensorShape(1, hkv, skv, d), 3, false);
        using Tensor expected = new(q.Shape, DType.F32), actual = new(q.Shape, DType.F32);
        AttentionReference.FlashAttention(expected, q, k, v, skv, hq / hkv, causal: false, qOffset: 0, scale: 0.125f, slidingWindow: window);
        backend.FlashAttention(actual, q, k, v, skv, hq / hkv, causal: false, qOffset: 0, scale: 0.125f, slidingWindow: window);
        float error = MaxAbs(actual.AsReadOnlySpan<float>(), expected.AsReadOnlySpan<float>());
        Assert.True(error < 2e-4f, $"max error {error} on route {VulkanBackend.LastAttentionRoute}");
    }

    /// <summary>F32 token-major attention without the F16 permission stays in F32: it must match the reference far closer
    /// than F16 rounding would.</summary>
    [Fact]
    public void TokenMajorWithoutTheF16Permission_StaysF32()
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        const int heads = 4, s = 80, d = 64;
        using Tensor q = Random(new TensorShape(s, heads * d), 1, false), k = Random(new TensorShape(s, heads * d), 2, false);
        using Tensor v = Random(new TensorShape(s, heads * d), 3, false);
        using Tensor expected = new(q.Shape, DType.F32), actual = new(q.Shape, DType.F32);
        AttentionHostReference.Run(AttentionCall.FromTokenMajor(expected, q, k, v, null, heads, heads, d, 0.125f, allowF16: false));
        backend.ScaledDotProductAttentionTokenMajor(actual, q, k, v, null, heads, d, 0.125f, allowF16: false);
        float error = MaxAbs(actual.AsReadOnlySpan<float>(), expected.AsReadOnlySpan<float>());
        output.WriteLine($"route {VulkanBackend.LastAttentionRoute}, max error {error:E2}");
        Assert.True(error < 2e-5f, $"max error {error} on route {VulkanBackend.LastAttentionRoute}");
    }

    /// <summary>Values beyond the F16 range saturate when attention narrows them, as cuDNN's casts do, instead of becoming
    /// infinity and turning the row into NaN.</summary>
    [Fact]
    public void OutOfRangeValues_SaturateWhenNarrowed()
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        const int heads = 2, s = 70, d = 64;
        using Tensor q = Random(new TensorShape(1, heads, s, d), 1, true), k = Random(new TensorShape(1, heads, s, d), 2, true);
        using Tensor v = Filled(new TensorShape(1, heads, s, d), i => i % 7 == 0 ? 1e5f : 0.5f);
        using Tensor actual = new(q.Shape, DType.F32);
        backend.ScaledDotProductAttention(actual, q, k, v, null, 0.125f, allowF16: true);
        ReadOnlySpan<float> o = actual.AsReadOnlySpan<float>();
        for (int i = 0; i < o.Length; i++)
            Assert.True(float.IsFinite(o[i]) && MathF.Abs(o[i]) <= 65504f * 1.001f, $"output[{i}] = {o[i]} on route {VulkanBackend.LastAttentionRoute}");
    }
}
