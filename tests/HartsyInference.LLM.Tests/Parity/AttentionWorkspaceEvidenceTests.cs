using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>Per-call attention scratch is recorded as <see cref="AttentionEvidenceKind.Workspace"/> evidence with its
/// logical size: the baseline a shared high-water workspace has to bring to zero.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class AttentionWorkspaceEvidenceTests
{
    [CudaFact]
    public void CudaSageRecordsItsSixScratchBuffers()
    {
        const int heads = 2, tokens = 64, headDim = 128;
        using IDisposable policy = KnobProfile.Create("workspace-evidence-sage").With(EngineKnobs.SageAttn, true)
            .With(EngineKnobs.SageF16MinSkv, tokens).Push();
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using Tensor query = Filled(new TensorShape(1, heads, tokens, headDim), DType.F16);
        using Tensor key = Filled(query.Shape, DType.F16);
        using Tensor value = Filled(query.Shape, DType.F16);
        using Tensor output = new(query.Shape, DType.F16);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        backend.ScaledDotProductAttention(output, query, key, value, null, 0.1f);
        backend.Sync();

        Assert.Contains(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Execution && e.Implementation == "sage.int8.f16");
        AttentionEvidenceEvent scratch = Assert.Single(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Workspace);
        Assert.Equal("cuda.sage.scratch", scratch.Site);
        Assert.Equal(6, scratch.Count);
        // K mean, INT8 Q and K with one F32 scale per row, and V transposed to F16 over the padded key axis.
        long expected = (long)heads * (headDim * sizeof(float) + 2 * tokens * (headDim + sizeof(float))
            + headDim * CudaKernels.SageSkvPad(tokens) * sizeof(ushort));
        Assert.Equal(expected, scratch.Bytes);
    }

    [CudaFact]
    public void CudaMaterializedAttentionRecordsItsScoreMatrix()
    {
        const int heads = 2, queryTokens = 3, keyTokens = 5, headDim = 64;
        using IDisposable policy = KnobProfile.Create("workspace-evidence-materialized").With(EngineKnobs.SdpaCudnn, false)
            .With(EngineKnobs.SdpaForceTiled, false).Push();
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using Tensor query = Filled(new TensorShape(1, heads, queryTokens, headDim), DType.F32);
        using Tensor key = Filled(new TensorShape(1, heads, keyTokens, headDim), DType.F32);
        using Tensor value = Filled(key.Shape, DType.F32);
        using Tensor output = new(query.Shape, DType.F32);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        backend.ScaledDotProductAttention(output, query, key, value, null, 0.1f);
        backend.Sync();

        AttentionEvidenceEvent scores = Assert.Single(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Workspace);
        Assert.Equal("cuda.materialized.scores", scores.Site);
        Assert.Equal((long)heads * queryTokens * keyTokens * sizeof(float), scores.Bytes);
    }

    [CudaFact]
    public void CudaSplitDecodeRecordsItsPartialBuffers()
    {
        const int heads = 2, keyTokens = 128, headDim = 64;
        using IDisposable policy = KnobProfile.Create("workspace-evidence-decode").With(EngineKnobs.FlashSplitForce, true).Push();
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using Tensor query = Filled(new TensorShape(1, heads, 1, headDim), DType.F32);
        using Tensor key = Filled(new TensorShape(1, heads, keyTokens, headDim), DType.F32);
        using Tensor value = Filled(key.Shape, DType.F32);
        using Tensor output = new(query.Shape, DType.F32);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        backend.FlashAttention(output, query, key, value, keyTokens, 1, causal: true, qOffset: keyTokens - 1, 0.1f);
        backend.Sync();

        AttentionEvidenceEvent partials = Assert.Single(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Workspace);
        Assert.StartsWith("cuda.flash.", partials.Site);
        Assert.EndsWith(".partials", partials.Site);
        Assert.Equal(3, partials.Count);
        // m, l and acc per (head, split): a whole number of splits, more than one.
        long perSplit = (long)heads * (headDim + 2) * sizeof(float);
        Assert.Equal(0, partials.Bytes % perSplit);
        Assert.True(partials.Bytes / perSplit > 1);
    }

    private static Tensor Filled(TensorShape shape, DType dtype)
    {
        Tensor tensor = new(shape, dtype);
        Random random = new(7);
        if (dtype == DType.F32)
        {
            Span<float> span = tensor.AsSpan<float>();
            for (int i = 0; i < span.Length; i++) span[i] = (float)(random.NextDouble() - 0.5);
        }
        else
        {
            Span<Half> span = tensor.AsSpan<Half>();
            for (int i = 0; i < span.Length; i++) span[i] = (Half)(random.NextDouble() - 0.5);
        }
        return tensor;
    }
}
