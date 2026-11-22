using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Idol;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Idol;

/// <summary><see cref="IdolAttention.Attend"/> attends through an <see cref="Core.Layers.Attention.AttentionLayer"/> made by
/// <see cref="IdolAttention.Layer"/>. It used to call <c>ScaledDotProductAttention(..., allowF16: true)</c> and permute back itself;
/// that body is kept here as the oracle. On the CPU backend the new path must be bit-identical to it and must take the same row the
/// same number of times with the same permission. (CUDA and Vulkan are compared on a real Idol generation, not here.)</summary>
[Collection("AttentionEvidence")]
public sealed class IdolAttentionTests
{
    private static Tensor Filled(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        Random random = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = random.NextSingle() * 2f - 1f;
        }

        return tensor;
    }

    /// <summary>The deleted body: head-major copies, the legacy dense entry with the F16 permission, the permute back.</summary>
    private static Tensor Former(IBackend backend, Tensor q, Tensor k, Tensor v, int queryTokens, int keyTokens, int heads, int headDim, float scale)
    {
        using Tensor qh = ToHeadMajor(backend, q, queryTokens, heads, headDim);
        using Tensor kh = ToHeadMajor(backend, k, keyTokens, heads, headDim);
        using Tensor vh = ToHeadMajor(backend, v, keyTokens, heads, headDim);
        Tensor att = new(new TensorShape(1, heads, queryTokens, headDim), DType.F32);
        backend.ScaledDotProductAttention(att, qh, kh, vh, null, scale, allowF16: true);
        Tensor flat = new(new TensorShape(1, queryTokens, (long)heads * headDim), DType.F32);
        backend.Permute0213(flat, att, heads, queryTokens, headDim);
        att.Dispose();
        return flat;
    }

    private static Tensor ToHeadMajor(IBackend backend, Tensor tokenMajor, int tokens, int heads, int headDim)
    {
        Tensor o = new(new TensorShape(1, heads, tokens, headDim), DType.F32);
        backend.Permute0213(o, tokenMajor, tokens, heads, headDim);
        return o;
    }

    private static string Rows(AttentionEvidenceEvent[] events) => string.Join("; ", events
        .Where(e => e.Kind == AttentionEvidenceKind.Execution)
        .GroupBy(e => $"{e.Implementation} {e.Route} {e.QueryType}/{e.KvType} h{e.Heads}/{e.KvHeads} q{e.QueryTokens} k{e.KeyCapacity} d{e.QkHeadDim} "
            + $"f16permission={e.Numerics.LegacyAssertedBounded}")
        .OrderBy(g => g.Key, StringComparer.Ordinal).Select(g => $"{g.Sum(e => e.Count)} x {g.Key}"));

    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(5, 5, 4)]
    [InlineData(37, 37, 16)]
    [InlineData(23, 41, 16)]
    [InlineData(64, 7, 3)]
    public void OnTheCpuTheLayerPathIsBitIdenticalToTheFormerBodyAndTakesTheSameRow(int queryTokens, int keyTokens, int heads)
    {
        const int headDim = 8;
        float scale = 1f / MathF.Sqrt(headDim);
        using CpuBackend backend = new();
        using Tensor q1 = Filled(new TensorShape(1, queryTokens, heads, headDim), 1), k1 = Filled(new TensorShape(1, keyTokens, heads, headDim), 2);
        using Tensor v1 = Filled(k1.Shape, 3);
        using Tensor q2 = Filled(q1.Shape, 1), k2 = Filled(k1.Shape, 2), v2 = Filled(v1.Shape, 3);

        string formerRows, layerRows;
        Tensor former, layer;
        using (AttentionEvidenceSession evidence = AttentionEvidence.Begin(64))
        {
            former = Former(backend, q1, k1, v1, queryTokens, keyTokens, heads, headDim, scale);
            formerRows = Rows(evidence.Snapshot());
        }

        using (AttentionEvidenceSession evidence = AttentionEvidence.Begin(64))
        {
            layer = IdolAttention.Attend(backend, IdolAttention.Layer(heads, headDim), q2, k2, v2, queryTokens, keyTokens, heads, headDim);
            layerRows = Rows(evidence.Snapshot());
        }

        using (former)
        using (layer)
        {
            Assert.Equal(former.Shape, layer.Shape);
            ReadOnlySpan<uint> expected = System.Runtime.InteropServices.MemoryMarshal.Cast<float, uint>(former.AsReadOnlySpan<float>());
            ReadOnlySpan<uint> actual = System.Runtime.InteropServices.MemoryMarshal.Cast<float, uint>(layer.AsReadOnlySpan<float>());
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.True(expected[i] == actual[i], $"element {i}: {expected[i]:x8} against {actual[i]:x8}");
            }
        }

        Assert.Equal(formerRows, layerRows);
        Assert.Contains("f16permission=True", layerRows);
    }

    [Fact]
    public void TheLayerCarriesTheF16PermissionAndTheDefaultScale()
    {
        Core.Layers.Attention.AttentionLayer layer = IdolAttention.Layer(16, 64);
        Assert.Equal(16, layer.Config.Heads);
        Assert.Equal(16, layer.Config.KvHeads);
        Assert.Equal(64, layer.Config.HeadDim);
        Assert.True(layer.Config.AllowF16);
        Assert.Equal(1f / MathF.Sqrt(64), layer.Config.EffectiveScale);
    }
}
