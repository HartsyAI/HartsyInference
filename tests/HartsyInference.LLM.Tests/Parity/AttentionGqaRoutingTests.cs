using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;

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

    /// <summary>A grouped call keeps the kernel its widened form has always run on: Sage above its measured crossover and
    /// the F16 flash kernel at head dim 256. Native cuDNN GQA takes the rest. Calibrated routing (PR 08B) owns any change.</summary>
    [CudaTheory]
    [InlineData(false, 128, "sage.int8.f16")]
    [InlineData(true, 128, "sage.int8.f16")]
    [InlineData(false, 256, "flash.f16")]
    [InlineData(true, 256, "flash.f16")]
    public void CudaGroupedCallKeepsItsPreferredEqualHeadKernel(bool tokenMajor, int headDim, string kernel)
    {
        const int tokens = 64;
        using IDisposable policy = KnobProfile.Create("gqa-preferred-kernel-tests")
            .With(EngineKnobs.SdpaCudnn, true).With(EngineKnobs.SageAttn, true).With(EngineKnobs.SageF16MinSkv, tokens)
            .With(EngineKnobs.FlashF16, true).With(EngineKnobs.SdpaForceTiled, false).Push();
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        TensorShape qShape = tokenMajor ? new(1, tokens, Heads, headDim) : new(1, Heads, tokens, headDim);
        TensorShape kvShape = tokenMajor ? new(1, tokens, KvHeads, headDim) : new(1, KvHeads, tokens, headDim);
        using Tensor query = new(qShape, DType.F16);
        using Tensor key = new(kvShape, DType.F16);
        using Tensor value = new(kvShape, DType.F16);
        using Tensor output = new(qShape, DType.F16);
        Fill(query, 0);
        Fill(key, 0);
        Span<Half> values = value.AsSpan<Half>();
        for (int i = 0; i < values.Length; i++)
            values[i] = (Half)(7 + (tokenMajor ? i / headDim % KvHeads : i / (tokens * headDim)) * 6);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        if (tokenMajor) backend.ScaledDotProductAttentionTokenMajor(output, query, key, value, null, Heads, KvHeads, headDim, 1);
        else backend.ScaledDotProductAttention(output, query, key, value, null, 1);
        backend.Sync();

        AttentionEvidenceEvent[] executions = evidence.Snapshot().Where(e => e.Kind == AttentionEvidenceKind.Execution).ToArray();
        Assert.Contains(executions, e => e.Implementation == kernel && e.KvHeads == Heads);
        Assert.Contains(executions, e => e.Implementation == "widen.preferred" && e.KvHeads == KvHeads && e.DeclaredLowering == AttentionLowering.WidenKv);
        Assert.DoesNotContain(executions, e => e.Implementation is "cudnn.f16" or "cudnn.tokenmajor.gqa");
        ReadOnlySpan<Half> result = output.AsReadOnlySpan<Half>();
        for (int i = 0; i < result.Length; i++)
        {
            float expected = 7 + (tokenMajor ? i / headDim % Heads : i / (tokens * headDim)) / (Heads / KvHeads) * 6;
            Assert.InRange(Math.Abs((float)result[i] - expected), 0, 0.02f * expected);
        }
    }

    /// <summary>Every Vulkan dense and token-major row reads grouped K/V in place; the layer no longer widens for them.</summary>
    [VulkanFact]
    public void VulkanDenseGroupedCallDoesNotRepeatKv() => VulkanGroupedCallDoesNotRepeatKv(tokenMajor: false);

    [VulkanFact]
    public void VulkanTokenMajorGroupedCallDoesNotRepeatKv() => VulkanGroupedCallDoesNotRepeatKv(tokenMajor: true);

    private static void VulkanGroupedCallDoesNotRepeatKv(bool tokenMajor)
    {
        using VulkanBackend backend = new();
        TensorShape qShape = tokenMajor ? new(1, QueryTokens, Heads, HeadDim) : new(1, Heads, QueryTokens, HeadDim);
        TensorShape kvShape = tokenMajor ? new(1, KeyTokens, KvHeads, HeadDim) : new(1, KvHeads, KeyTokens, HeadDim);
        using Tensor query = new(qShape, DType.F32);
        using Tensor key = new(kvShape, DType.F32);
        using Tensor value = new(kvShape, DType.F32);
        using Tensor output = new(qShape, DType.F32);
        Fill(query, 0);
        Fill(key, 0);
        FillValues(value);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        if (tokenMajor) backend.ScaledDotProductAttentionTokenMajor(output, query, key, value, null, Heads, KvHeads, HeadDim, 1);
        else backend.ScaledDotProductAttention(output, query, key, value, null, 1);
        backend.Sync();

        AttentionEvidenceEvent[] executions = evidence.Snapshot().Where(e => e.Kind == AttentionEvidenceKind.Execution).ToArray();
        Assert.NotEmpty(executions);
        Assert.All(executions, e => Assert.Equal(KvHeads, e.KvHeads));
        Assert.DoesNotContain(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Repeat);
        AssertGroupedValues(output, 1e-3f);
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

    private static void AssertGroupedValues(Tensor output, float tolerance = 2e-5f)
    {
        bool tokenMajor = output.Shape[2] == Heads;
        for (int token = 0; token < QueryTokens; token++)
            for (int head = 0; head < Heads; head++)
                for (int channel = 0; channel < HeadDim; channel++)
                {
                    int offset = tokenMajor ? token * Heads + head : head * QueryTokens + token;
                    float expected = 7 + head / (Heads / KvHeads) * 6;
                    Assert.InRange(Math.Abs(Read(output, offset * HeadDim + channel) - expected), 0, tolerance * expected);
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
