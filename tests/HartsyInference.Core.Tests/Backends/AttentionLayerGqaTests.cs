using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

[Collection("AttentionEvidence")]
public sealed class AttentionLayerGqaTests
{
    [Fact]
    public void HeadMajorTransformPassesGroupedKvToTheBackendWithoutRepeating()
    {
        const int heads = 4, kvHeads = 2, queryTokens = 2, keyTokens = 3, headDim = 4;
        using CpuBackend backend = new();
        AttentionLayer layer = new(new AttentionLayerConfig { Heads = heads, KvHeads = kvHeads, HeadDim = headDim });
        Tensor query = new(new TensorShape(1, queryTokens, heads, headDim), DType.F32);
        Tensor key = new(new TensorShape(1, keyTokens, kvHeads, headDim), DType.F32);
        Tensor value = new(key.Shape, DType.F32);
        query.AsSpan<float>().Fill(0);
        key.AsSpan<float>().Fill(0);
        for (int token = 0; token < keyTokens; token++)
            for (int head = 0; head < kvHeads; head++)
                value.AsSpan<float>().Slice((token * kvHeads + head) * headDim, headDim).Fill(7 + head * 6);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(32);
        using Tensor output = layer.Attend(backend, query, key, value,
            new TensorShape(1, queryTokens, heads, headDim), default);

        AttentionEvidenceEvent execution = Assert.Single(evidence.Snapshot(),
            e => e.Kind == AttentionEvidenceKind.Execution && e.Implementation == "cpu.simd");
        Assert.Equal(heads, execution.Heads);
        Assert.Equal(kvHeads, execution.KvHeads);
        Assert.DoesNotContain(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Repeat);
        for (int token = 0; token < queryTokens; token++)
            for (int head = 0; head < heads; head++)
                Assert.All(output.AsReadOnlySpan<float>().Slice((token * heads + head) * headDim, headDim).ToArray(),
                    actual => Assert.Equal(7 + head / (heads / kvHeads) * 6, actual, 5));
    }

    /// <summary>The SIMD kernel reading grouped K/V in place computes the bits it computes over repeated K/V.</summary>
    [Fact]
    public void CpuDenseGroupedKvMatchesRepeatedKvBitForBit()
    {
        const int batch = 2, heads = 6, kvHeads = 2, queryTokens = 5, keyTokens = 7, headDim = 12;
        using CpuBackend backend = new();
        using Tensor query = new(new TensorShape(batch, heads, queryTokens, headDim), DType.F32);
        using Tensor key = new(new TensorShape(batch, kvHeads, keyTokens, headDim), DType.F32);
        using Tensor value = new(key.Shape, DType.F32);
        using Tensor mask = new(new TensorShape(batch, heads, queryTokens, keyTokens), DType.F32);
        Random random = new(11);
        foreach (Tensor tensor in (Tensor[])[query, key, value, mask])
        {
            Span<float> span = tensor.AsSpan<float>();
            for (int i = 0; i < span.Length; i++) span[i] = (float)(random.NextDouble() * 2 - 1);
        }
        using Tensor keyRepeated = new(new TensorShape(batch, heads, keyTokens, headDim), DType.F32);
        using Tensor valueRepeated = new(keyRepeated.Shape, DType.F32);
        StridedCopies.RepeatInterleave(backend, keyRepeated, key, 1, heads / kvHeads);
        StridedCopies.RepeatInterleave(backend, valueRepeated, value, 1, heads / kvHeads);
        using Tensor grouped = new(query.Shape, DType.F32);
        using Tensor repeated = new(query.Shape, DType.F32);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(8);
        backend.ScaledDotProductAttention(grouped, query, key, value, mask, 0.3f);
        Assert.Equal("cpu.simd", Assert.Single(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Execution).Implementation);
        backend.ScaledDotProductAttention(repeated, query, keyRepeated, valueRepeated, mask, 0.3f);

        Assert.True(grouped.AsReadOnlySpan<float>().SequenceEqual(repeated.AsReadOnlySpan<float>()));
    }
}
