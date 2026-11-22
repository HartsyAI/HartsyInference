using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

/// <summary><see cref="Trellis2FlowBlock.Attend"/> attends through a shared <see cref="Core.Layers.Attention.AttentionLayer"/> and merges
/// the heads with <c>MergeHeads</c>. It used to call <c>ScaledDotProductAttention(..., null, scale)</c> and permute back itself; that body is
/// kept here as the oracle. On the CPU backend the new path must be bit-identical to it, take the same row the same number of times
/// without the F16 permission, and dispose q, k and v like the old body did. (CUDA and Vulkan are compared on a real generation, not here.)</summary>
[Collection("AttentionEvidence")]
public sealed class Trellis2FlowBlockAttendTests
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

    /// <summary>The deleted body: the legacy dense entry without a mask and without the F16 permission, the permute back.</summary>
    private static Tensor Former(IBackend backend, Tensor q, Tensor k, Tensor v, int tokens, int heads, int headDim, float scale)
    {
        Tensor att = new(new TensorShape(1, heads, tokens, headDim), DType.F32);
        backend.ScaledDotProductAttention(att, q, k, v, null, scale);
        q.Dispose();
        k.Dispose();
        v.Dispose();
        Tensor flat = new(new TensorShape(1, tokens, (long)heads * headDim), DType.F32);
        backend.Permute0213(flat, att, heads, tokens, headDim);
        att.Dispose();
        return flat;
    }

    private static string Rows(AttentionEvidenceEvent[] events) => string.Join("; ", events
        .Where(e => e.Kind == AttentionEvidenceKind.Execution)
        .GroupBy(e => $"{e.Implementation} {e.Route} {e.QueryType}/{e.KvType} h{e.Heads}/{e.KvHeads} q{e.QueryTokens} k{e.KeyCapacity} d{e.QkHeadDim} "
            + $"f16permission={e.Numerics.LegacyAssertedBounded}")
        .OrderBy(g => g.Key, StringComparer.Ordinal).Select(g => $"{g.Sum(e => e.Count)} x {g.Key}"));

    [Theory]
    [InlineData(1, 1)]
    [InlineData(5, 5)]
    [InlineData(37, 37)]
    [InlineData(23, 6)]
    [InlineData(9, 41)]
    public void OnTheCpuTheLayerPathIsBitIdenticalToTheFormerBodyAndTakesTheSameRow(int queryTokens, int keyTokens)
    {
        Trellis2FlowConfig config = Trellis2TestData.Tiny(dense: false);
        Dictionary<string, Tensor> tensors = Trellis2TestData.RandomWeights(config, 5);
        using Trellis2FlowWeights weights = Trellis2FlowWeights.FromTensors(config, tensors);
        using Trellis2FlowBlock block = new(weights, 0);
        int heads = config.NumHeads, headDim = config.HeadDim;
        float scale = 1f / MathF.Sqrt(headDim);
        using CpuBackend backend = new();
        TensorShape qShape = new(1, heads, queryTokens, headDim), kvShape = new(1, heads, keyTokens, headDim);
        Tensor q1 = Filled(qShape, 1), k1 = Filled(kvShape, 2), v1 = Filled(kvShape, 3);
        Tensor q2 = Filled(qShape, 1), k2 = Filled(kvShape, 2), v2 = Filled(kvShape, 3);

        string formerRows, layerRows;
        Tensor former, layer;
        using (AttentionEvidenceSession evidence = AttentionEvidence.Begin(64))
        {
            former = Former(backend, q1, k1, v1, queryTokens, heads, headDim, scale);
            formerRows = Rows(evidence.Snapshot());
        }

        using (AttentionEvidenceSession evidence = AttentionEvidence.Begin(64))
        {
            layer = block.Attend(backend, q2, k2, v2, queryTokens);
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
        Assert.Contains("f16permission=False", layerRows);
        Trellis2TestData.AssertDisposed(q2);
        Trellis2TestData.AssertDisposed(k2);
        Trellis2TestData.AssertDisposed(v2);
        Trellis2TestData.DisposeAll(tensors.Values);
    }
}
