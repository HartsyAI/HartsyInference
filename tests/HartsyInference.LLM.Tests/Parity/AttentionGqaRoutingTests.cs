using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>Route-forced GQA controls distinguish native grouped execution from explicit compatibility widening.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class AttentionGqaRoutingTests
{
    private const int Heads = 4, KvHeads = 2, QueryTokens = 2, KeyTokens = 3, HeadDim = 64;

    [CudaFact]
    public void CudaDenseNativeGqaDoesNotRepeatKv()
    {
        using IDisposable policy = Policy(cudnn: true).Push();
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using Tensor query = new(new TensorShape(1, Heads, QueryTokens, HeadDim), DType.F16);
        using Tensor key = new(new TensorShape(1, KvHeads, KeyTokens, HeadDim), DType.F16);
        using Tensor value = new(key.Shape, DType.F16);
        using Tensor output = new(query.Shape, DType.F16);
        Fill(query, 0);
        Fill(key, 0);
        FillValues(value);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        backend.ScaledDotProductAttention(output, query, key, value, null, 1);
        backend.Sync();

        AttentionEvidenceEvent execution = Assert.Single(evidence.Snapshot(),
            e => e.Kind == AttentionEvidenceKind.Execution && e.Implementation == "cudnn.f16");
        Assert.Equal(Heads, execution.Heads);
        Assert.Equal(KvHeads, execution.KvHeads);
        Assert.Equal(AttentionLowering.None, execution.DeclaredLowering);
        Assert.DoesNotContain(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Repeat);
        AssertGroupedValues(output);
    }

    [CudaFact]
    public void CudaDenseCompatibilityWideningIsExplicitAndPreservesGroupedValues()
    {
        using IDisposable policy = Policy(cudnn: false).Push();
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using Tensor query = new(new TensorShape(1, Heads, QueryTokens, HeadDim), DType.F32);
        using Tensor key = new(new TensorShape(1, KvHeads, KeyTokens, HeadDim), DType.F32);
        using Tensor value = new(key.Shape, DType.F32);
        using Tensor output = new(query.Shape, DType.F32);
        Fill(query, 0);
        Fill(key, 0);
        FillValues(value);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        backend.ScaledDotProductAttention(output, query, key, value, null, 1);
        backend.Sync();

        AttentionEvidenceEvent widening = Assert.Single(evidence.Snapshot(),
            e => e.Kind == AttentionEvidenceKind.Execution && e.Implementation == "widen.dense");
        Assert.Equal(KvHeads, widening.KvHeads);
        Assert.Equal(AttentionLowering.WidenKv, widening.DeclaredLowering);
        AttentionEvidenceEvent nested = Assert.Single(evidence.Snapshot(),
            e => e.Kind == AttentionEvidenceKind.Execution && e.Implementation == "materialized");
        Assert.Equal(Heads, nested.KvHeads);
        Assert.Equal(2, evidence.Snapshot().Count(e => e.Kind == AttentionEvidenceKind.Repeat));
        AssertGroupedValues(output);
    }

    [CudaFact]
    public void CudaTokenMajorNativeGqaDoesNotRepeatKv()
    {
        using IDisposable policy = Policy(cudnn: true).Push();
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        using Tensor query = new(new TensorShape(1, QueryTokens, Heads, HeadDim), DType.F16);
        using Tensor key = new(new TensorShape(1, KeyTokens, KvHeads, HeadDim), DType.F16);
        using Tensor value = new(key.Shape, DType.F16);
        using Tensor output = new(query.Shape, DType.F16);
        Fill(query, 0);
        Fill(key, 0);
        FillValues(value);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        backend.ScaledDotProductAttentionTokenMajor(output, query, key, value, null, Heads, KvHeads, HeadDim, 1);
        backend.Sync();

        AttentionEvidenceEvent execution = Assert.Single(evidence.Snapshot(),
            e => e.Kind == AttentionEvidenceKind.Execution && e.Implementation == "cudnn.tokenmajor.gqa");
        Assert.Equal(Heads, execution.Heads);
        Assert.Equal(KvHeads, execution.KvHeads);
        Assert.Equal(AttentionLowering.None, execution.DeclaredLowering);
        Assert.DoesNotContain(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Repeat);
        AssertGroupedValues(output);
    }

    private static KnobProfile Policy(bool cudnn) => KnobProfile.Create("gqa-routing-tests")
        .With(EngineKnobs.SdpaCudnn, cudnn).With(EngineKnobs.SageAttn, false)
        .With(EngineKnobs.FlashF16, false).With(EngineKnobs.SdpaForceTiled, false)
        .With(EngineKnobs.SdpaNoTf32, true);

    private static void FillValues(Tensor value)
    {
        bool tokenMajor = value.Shape[2] == KvHeads;
        for (int token = 0; token < KeyTokens; token++)
            for (int head = 0; head < KvHeads; head++)
                for (int channel = 0; channel < HeadDim; channel++)
                {
                    int offset = tokenMajor ? token * KvHeads + head : head * KeyTokens + token;
                    Write(value, offset * HeadDim + channel, 7 + head * 6);
                }
    }

    private static void AssertGroupedValues(Tensor output)
    {
        bool tokenMajor = output.Shape[2] == Heads;
        for (int token = 0; token < QueryTokens; token++)
            for (int head = 0; head < Heads; head++)
                for (int channel = 0; channel < HeadDim; channel++)
                {
                    int offset = tokenMajor ? token * Heads + head : head * QueryTokens + token;
                    float expected = 7 + head / (Heads / KvHeads) * 6;
                    Assert.InRange(Math.Abs(Read(output, offset * HeadDim + channel) - expected), 0, 2e-5 * expected);
                }
    }

    private static void Fill(Tensor tensor, float value)
    {
        for (int i = 0; i < tensor.ElementCount; i++) Write(tensor, i, value);
    }

    private static void Write(Tensor tensor, int index, float value)
    {
        if (tensor.DType == DType.F32) tensor.AsSpan<float>()[index] = value;
        else tensor.AsSpan<Half>()[index] = (Half)value;
    }

    private static float Read(Tensor tensor, int index) => tensor.DType == DType.F32
        ? tensor.AsReadOnlySpan<float>()[index] : (float)tensor.AsReadOnlySpan<Half>()[index];
}
