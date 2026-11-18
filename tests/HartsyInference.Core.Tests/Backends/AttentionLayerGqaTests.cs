using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
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
            e => e.Kind == AttentionEvidenceKind.Execution && e.Implementation == "cpu.host.reference");
        Assert.Equal(heads, execution.Heads);
        Assert.Equal(kvHeads, execution.KvHeads);
        Assert.DoesNotContain(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Repeat);
        for (int token = 0; token < queryTokens; token++)
            for (int head = 0; head < heads; head++)
                Assert.All(output.AsReadOnlySpan<float>().Slice((token * heads + head) * headDim, headDim).ToArray(),
                    actual => Assert.Equal(7 + head / (heads / kvHeads) * 6, actual));
    }
}
