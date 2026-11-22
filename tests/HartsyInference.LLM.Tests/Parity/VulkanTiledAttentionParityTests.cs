using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>The query-tiled F32 kernel (<c>tiled.f32</c>) against the host reference, on the calls the default now gives
/// it and on the head-dim-256 bucket <c>all</c> selects. Each case checks that the tiled row ran, then the contract's F32
/// bound: relative L2 at most 2e-5. Token counts are chosen off the tile heights (16, 32 and 8 query rows; 32 and 16 keys).</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class VulkanTiledAttentionParityTests(ITestOutputHelper output)
{
    private const double F32Bound = 2e-5;

    /// <summary>The additive bias of one case.</summary>
    public enum Bias { None, PerQuery, PerHead, KeyOnly, WithAnExcludedRow }

    private static Tensor Random(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        Random random = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = random.NextSingle() * 2f - 1f;
        return tensor;
    }

    private static Tensor? MakeBias(Bias kind, int batch, int heads, int queryTokens, int keyTokens)
    {
        if (kind == Bias.None) return null;
        Tensor bias = Random(kind switch
        {
            Bias.PerHead => new TensorShape(batch, heads, queryTokens, keyTokens),
            Bias.KeyOnly => new TensorShape(batch, 1, 1, keyTokens),
            _ => new TensorShape(queryTokens, keyTokens),
        }, 4);
        if (kind != Bias.WithAnExcludedRow) return bias;
        // Row 1 excludes every key; every third key of the other rows is excluded too.
        Span<float> values = bias.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            if (i / keyTokens == 1 || i % 3 == 0) values[i] = float.NegativeInfinity;
        return bias;
    }

    private static double RelativeL2(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        double error = 0, reference = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - (double)expected[i];
            error += d * d;
            reference += expected[i] * (double)expected[i];
        }
        return Math.Sqrt(error / reference);
    }

    private static string Ran(AttentionEvidenceSession evidence) => string.Join(", ",
        evidence.Snapshot().Where(e => e.Kind == AttentionEvidenceKind.Execution).Select(e => e.Implementation));

    [VulkanTheory]
    [InlineData("D 32", VulkanTiledAttention.Auto, 1, 4, 4, 37, 45, 32, Bias.None)]
    [InlineData("D 64", VulkanTiledAttention.Auto, 1, 4, 4, 37, 45, 64, Bias.None)]
    [InlineData("D 64, one query row", VulkanTiledAttention.Auto, 1, 4, 4, 1, 45, 64, Bias.None)]
    [InlineData("D 64, grouped heads, batch 2", VulkanTiledAttention.Auto, 2, 6, 2, 37, 45, 64, Bias.None)]
    [InlineData("D 64, a bias row per query", VulkanTiledAttention.Auto, 1, 4, 4, 37, 45, 64, Bias.PerQuery)]
    [InlineData("D 64, a bias per head, batch 2", VulkanTiledAttention.Auto, 2, 4, 2, 37, 45, 64, Bias.PerHead)]
    [InlineData("D 64, a key-only bias, batch 2", VulkanTiledAttention.Auto, 2, 4, 4, 37, 45, 64, Bias.KeyOnly)]
    [InlineData("D 64, a fully excluded row", VulkanTiledAttention.Auto, 1, 4, 4, 37, 45, 64, Bias.WithAnExcludedRow)]
    [InlineData("D 96", VulkanTiledAttention.Auto, 1, 4, 2, 70, 45, 96, Bias.None)]
    [InlineData("D 128", VulkanTiledAttention.Auto, 1, 4, 4, 70, 45, 128, Bias.None)]
    [InlineData("D 128, a bias per head, grouped heads, batch 2", VulkanTiledAttention.Auto, 2, 4, 2, 70, 45, 128, Bias.PerHead)]
    [InlineData("D 128, a fully excluded row", VulkanTiledAttention.Auto, 1, 4, 4, 70, 45, 128, Bias.WithAnExcludedRow)]
    [InlineData("D 200, selected by all", VulkanTiledAttention.All, 1, 4, 2, 21, 45, 200, Bias.None)]
    [InlineData("D 256, selected by all", VulkanTiledAttention.All, 2, 4, 4, 21, 45, 256, Bias.PerHead)]
    [InlineData("D 256, selected by all, a fully excluded row", VulkanTiledAttention.All, 1, 4, 4, 21, 45, 256, Bias.WithAnExcludedRow)]
    public void TiledRowMatchesTheHostReference(string name, VulkanTiledAttention mode, int batch, int heads, int kvHeads, int queryTokens,
        int keyTokens, int headDim, Bias biasKind)
    {
        float scale = 1f / MathF.Sqrt(headDim);
        using VulkanBackend backend = new() { TiledAttention = mode };
        using Tensor q = Random(new TensorShape(batch, heads, queryTokens, headDim), 1);
        using Tensor k = Random(new TensorShape(batch, kvHeads, keyTokens, headDim), 2);
        using Tensor v = Random(k.Shape, 3);
        using Tensor? bias = MakeBias(biasKind, batch, heads, queryTokens, keyTokens);
        using Tensor actual = new(q.Shape, DType.F32), expected = new(q.Shape, DType.F32);
        AttentionCall reference = AttentionCall.FromScaledDotProductAttention(expected, q, k, v, bias, scale, allowF16: false);
        AttentionHostReference.Run(in reference);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        backend.ScaledDotProductAttention(actual, q, k, v, bias, scale);
        backend.Sync();

        Assert.Equal("tiled.f32", Ran(evidence));
        double error = RelativeL2(actual.AsReadOnlySpan<float>(), expected.AsReadOnlySpan<float>());
        output.WriteLine($"PARITY | {name} | B {batch}, H {heads}/{kvHeads}, {queryTokens} x {keyTokens}, D {headDim} | relative L2 {error:E2}");
        Assert.True(error <= F32Bound, $"{name}: relative L2 {error:E3} is past {F32Bound:E0}");
        if (biasKind != Bias.WithAnExcludedRow) return;
        // Query row 1 of every head of every batch item has no key left: the contract's result is zero, not NaN.
        ReadOnlySpan<float> values = actual.AsReadOnlySpan<float>();
        for (int head = 0; head < batch * heads; head++)
            for (int d = 0; d < headDim; d++)
                Assert.Equal(0f, values[(head * queryTokens + 1) * headDim + d]);
    }

    /// <summary>A token-major call the cooperative-matrix row does not take is permuted into the dense entry, and BF16
    /// operands are widened to F32 first, so both reach the tiled row under the default and the scalar row with the knob off.</summary>
    [VulkanTheory]
    [InlineData(false, VulkanTiledAttention.Auto, "tiled.f32")]
    [InlineData(false, VulkanTiledAttention.Off, "flash.scalar")]
    [InlineData(true, VulkanTiledAttention.Auto, "tiled.f32")]
    [InlineData(true, VulkanTiledAttention.Off, "flash.scalar")]
    public void TokenMajorCallReachesTheSameDenseRow(bool bf16, VulkanTiledAttention mode, string denseRow)
    {
        const int heads = 4, kvHeads = 2, queryTokens = 37, keyTokens = 45, headDim = 64;
        float scale = 1f / MathF.Sqrt(headDim);
        using VulkanBackend backend = new() { TiledAttention = mode };
        using Tensor q32 = Random(new TensorShape(1, queryTokens, heads, headDim), 1);
        using Tensor k32 = Random(new TensorShape(1, keyTokens, kvHeads, headDim), 2);
        using Tensor v32 = Random(k32.Shape, 3);
        DType storage = bf16 ? DType.BF16 : DType.F32;
        using Tensor q = q32.CastTo(storage), k = k32.CastTo(storage), v = v32.CastTo(storage);
        using Tensor actual = new(q.Shape, storage), expected = new(q.Shape, storage);
        AttentionCall reference = AttentionCall.FromTokenMajor(expected, q, k, v, null, heads, kvHeads, headDim, scale, allowF16: false);
        AttentionHostReference.Run(in reference);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        backend.ScaledDotProductAttentionTokenMajor(actual, q, k, v, null, heads, kvHeads, headDim, scale);
        backend.Sync();

        string ran = Ran(evidence);
        Assert.Contains(denseRow, ran.Split(", "));
        Assert.Contains("permute.dense", ran.Split(", "));
        using Tensor actual32 = actual.CastTo(DType.F32), expected32 = expected.CastTo(DType.F32);
        double error = RelativeL2(actual32.AsReadOnlySpan<float>(), expected32.AsReadOnlySpan<float>());
        // BF16 storage rounds the output, so its bound is the contract's for BF16 storage.
        double bound = bf16 ? 8e-3 : F32Bound;
        output.WriteLine($"PARITY | token-major, {storage}, {mode} | ran {ran} | relative L2 {error:E2}");
        Assert.True(error <= bound, $"token-major {storage} under {mode}: relative L2 {error:E3} is past {bound:E0}");
    }
}
