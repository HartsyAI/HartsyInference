using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Tests.Transformer;

/// <summary>A bias as wide as the valid keys is widened on the host to a fixed cache's capacity before the call. That
/// copy is recorded with its bytes; a bias that already fits is not copied and leaves no record.</summary>
[Collection("AttentionSemantics")]
public sealed class AttentionBiasWideningEvidenceTests
{
    private const int Heads = 2, HeadDim = 8, Capacity = 16, Rows = 5;

    private static AttentionEvidenceEvent[] Attend(int biasWidth)
    {
        using CpuBackend backend = new();
        using FixedKvCache cache = new(1, 1, Heads, HeadDim, Capacity, DType.F32);
        AttentionLayer layer = new(new AttentionLayerConfig { Heads = Heads, KvHeads = Heads, HeadDim = HeadDim });
        using Tensor q = new(new TensorShape(1, Heads, Rows, HeadDim), DType.F32);
        using Tensor k = new(q.Shape, DType.F32), v = new(q.Shape, DType.F32);
        using Tensor bias = new(new TensorShape(1, 1, Rows, biasWidth), DType.F32);
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        using Tensor output = layer.AttendOverKvHeadMajor(backend, q, k, v, cache, 0, 0,
            new KvAttentionCall { KvLength = Rows, QueryOffset = 0, Bias = bias }, OperandOwnership.None);
        return evidence.Snapshot();
    }

    [Fact]
    public void ABiasNarrowerThanTheCacheIsWidenedAndRecorded()
    {
        AttentionEvidenceEvent[] events = Attend(biasWidth: Rows);
        AttentionEvidenceEvent widened = Assert.Single(events, e => e.Kind == AttentionEvidenceKind.ExpandMask && e.Site == "attention.kv.bias");
        Assert.Equal((long)Rows * Capacity * sizeof(float), widened.Bytes);
    }

    [Fact]
    public void ABiasAsWideAsTheCacheIsNotCopied()
    {
        AttentionEvidenceEvent[] events = Attend(biasWidth: Capacity);
        Assert.DoesNotContain(events, e => e.Kind == AttentionEvidenceKind.ExpandMask);
    }
}
