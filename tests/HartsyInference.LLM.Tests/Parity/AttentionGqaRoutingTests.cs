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
        FillValues(value, tokenMajor: false);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        backend.ScaledDotProductAttention(output, query, key, value, null, 1);
        backend.Sync();

        AttentionEvidenceEvent execution = Assert.Single(evidence.Snapshot(),
            e => e.Kind == AttentionEvidenceKind.Execution && e.Implementation == "cudnn.f16");
        Assert.Equal(Heads, execution.Heads);
        Assert.Equal(KvHeads, execution.KvHeads);
        Assert.Equal(AttentionLowering.None, execution.DeclaredLowering);
        Assert.DoesNotContain(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Repeat);
        AssertGroupedValues(output, tokenMajor: false);
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
        FillValues(value, tokenMajor: false);

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
        AssertGroupedValues(output, tokenMajor: false);
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
        FillValues(value, tokenMajor: true);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        backend.ScaledDotProductAttentionTokenMajor(output, query, key, value, null, Heads, KvHeads, HeadDim, 1);
        backend.Sync();

        AttentionEvidenceEvent execution = Assert.Single(evidence.Snapshot(),
            e => e.Kind == AttentionEvidenceKind.Execution && e.Implementation == "cudnn.tokenmajor.gqa");
        Assert.Equal(Heads, execution.Heads);
        Assert.Equal(KvHeads, execution.KvHeads);
        Assert.Equal(AttentionLowering.None, execution.DeclaredLowering);
        // The row states that it prepares nothing, which is what was recorded.
        AttentionCall call = AttentionCall.FromTokenMajor(output, query, key, value, null, Heads, KvHeads, HeadDim, 1f, allowF16: false);
        AttentionPlan plan = backend.PlanAttention(in call);
        Assert.Equal("cudnn.tokenmajor.gqa", plan.RowId);
        Assert.Equal(0, plan.Preparation!.Value.Total);
        Assert.DoesNotContain(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Cast);
        Assert.DoesNotContain(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Repeat);
        AssertGroupedValues(output, tokenMajor: true);
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
        FillValues(value, tokenMajor);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        if (tokenMajor) backend.ScaledDotProductAttentionTokenMajor(output, query, key, value, null, Heads, KvHeads, HeadDim, 1);
        else backend.ScaledDotProductAttention(output, query, key, value, null, 1);
        backend.Sync();

        AttentionEvidenceEvent[] executions = evidence.Snapshot().Where(e => e.Kind == AttentionEvidenceKind.Execution).ToArray();
        Assert.NotEmpty(executions);
        Assert.All(executions, e => Assert.Equal(KvHeads, e.KvHeads));
        Assert.DoesNotContain(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Repeat);
        AssertGroupedValues(output, tokenMajor, 1e-3f);
    }

    /// <summary>Sage computes QK in INT8, so a grouped call that refuses INT8 is not widened for it: native cuDNN GQA runs.</summary>
    [CudaFact]
    public void CudaGroupedCallThatRefusesInt8IsNotWidenedForSage()
    {
        const int tokens = 64, headDim = 128;
        using IDisposable policy = KnobProfile.Create("gqa-refused-int8").With(EngineKnobs.SdpaCudnn, true)
            .With(EngineKnobs.SageAttn, true).With(EngineKnobs.SageF16MinSkv, tokens).With(EngineKnobs.FlashF16, false).Push();
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        Random random = new(5);
        using Tensor query = RandomTensor(new TensorShape(1, Heads, tokens, headDim), DType.F16, random);
        using Tensor key = RandomTensor(new TensorShape(1, KvHeads, tokens, headDim), DType.F16, random);
        using Tensor value = RandomTensor(key.Shape, DType.F16, random);
        using Tensor output = new(query.Shape, DType.F16);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(output, query, key, value, null, 0.1f, allowF16: false)
            with { Numerics = new AttentionNumerics { Refused = AttentionFormats.Int8 } };

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        backend.Attention(in call);
        backend.Sync();

        AttentionEvidenceEvent execution = Assert.Single(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Execution);
        Assert.Equal("cudnn.f16", execution.Implementation);
        Assert.Equal(KvHeads, execution.KvHeads);
        Assert.DoesNotContain(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Repeat);
    }

    /// <summary>Random operands and a per-head bias, in a batch of two where the entry takes one: each grouped route computes
    /// the host reference's result, so scale, bias and every key's contribution are checked, not only the head mapping.</summary>
    [CudaTheory]
    [InlineData("cudnn.f16", true, false)]
    [InlineData("widen.dense", false, false)]
    [InlineData("permute.dense", false, true)]
    public void CudaGroupedRoutesMatchTheHostReference(string implementation, bool half, bool tokenMajor)
    {
        const int queryTokens = 5, keyTokens = 8;   // cuDNN reads a per-head bias only where the key count is a multiple of 4
        int batch = tokenMajor ? 1 : 2;
        DType dtype = half ? DType.F16 : DType.F32;
        using IDisposable policy = Policy(cudnn: half).Push();
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        TensorShape qShape = tokenMajor ? new(batch, queryTokens, Heads, HeadDim) : new(batch, Heads, queryTokens, HeadDim);
        TensorShape kvShape = tokenMajor ? new(batch, keyTokens, KvHeads, HeadDim) : new(batch, KvHeads, keyTokens, HeadDim);
        Random random = new(23);
        using Tensor query = RandomTensor(qShape, dtype, random);
        using Tensor key = RandomTensor(kvShape, dtype, random);
        using Tensor value = RandomTensor(kvShape, dtype, random);
        using Tensor bias = RandomTensor(new TensorShape(batch, Heads, queryTokens, keyTokens), DType.F32, random);
        using Tensor output = new(qShape, dtype);
        using Tensor expected = new(qShape, dtype);
        const float scale = 0.31f;
        AttentionCall reference = tokenMajor
            ? AttentionCall.FromTokenMajor(expected, query, key, value, bias, Heads, KvHeads, HeadDim, scale, allowF16: false)
            : AttentionCall.FromScaledDotProductAttention(expected, query, key, value, bias, scale, allowF16: false);
        AttentionHostReference.Run(in reference);

        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        if (tokenMajor) backend.ScaledDotProductAttentionTokenMajor(output, query, key, value, bias, Heads, KvHeads, HeadDim, scale);
        else backend.ScaledDotProductAttention(output, query, key, value, bias, scale);
        backend.Sync();

        Assert.Contains(evidence.Snapshot(), e => e.Kind == AttentionEvidenceKind.Execution && e.Implementation == implementation);
        Assert.Equal(half ? 0 : 2, evidence.Snapshot().Count(e => e.Kind == AttentionEvidenceKind.Repeat));
        float tolerance = half ? 5e-3f : 1e-4f;
        for (int i = 0; i < output.ElementCount; i++)
            Assert.InRange(Math.Abs(Read(output, i) - Read(expected, i)), 0, tolerance);
    }

    /// <summary>The dense validator accepts grouped K/V, so the capability flag is what keeps a grouped call off the
    /// equal-head kernels: whatever the knobs, the row planned for one declares grouped support.</summary>
    [CudaTheory]
    [InlineData(true, false)]
    [InlineData(false, false)]
    [InlineData(false, true)]
    public void CudaPlansOnlyGroupedRowsForAGroupedDenseCall(bool cudnn, bool forceTiled)
    {
        using IDisposable policy = KnobProfile.Create("gqa-planning-guard").With(EngineKnobs.SdpaCudnn, cudnn)
            .With(EngineKnobs.SdpaForceTiled, forceTiled).Push();
        using CudaBackend backend = new(0, CudaPtxDirectory.Resolve());
        foreach (DType dtype in (DType[])[DType.F32, DType.F16, DType.BF16])
        {
            using Tensor query = new(new TensorShape(1, Heads, 64, 128), dtype);
            using Tensor key = new(new TensorShape(1, KvHeads, 64, 128), dtype);
            using Tensor value = new(key.Shape, dtype);
            using Tensor output = new(query.Shape, dtype);
            AttentionCall call = AttentionCall.FromScaledDotProductAttention(output, query, key, value, null, 0.1f, allowF16: false);
            AttentionKernelRow? row = backend.PlanAttention(in call).Row;
            Assert.NotNull(row);
            Assert.True(row.Capability.Gqa, $"{dtype}: planned {row.Capability.Id}, which reads K/V with the query's head count.");
        }
    }

    private static KnobProfile Policy(bool cudnn) => KnobProfile.Create("gqa-routing-tests")
        .With(EngineKnobs.SdpaCudnn, cudnn).With(EngineKnobs.SageAttn, false)
        .With(EngineKnobs.FlashF16, false).With(EngineKnobs.SdpaForceTiled, false)
        .With(EngineKnobs.SdpaNoTf32, true);

    private static void FillValues(Tensor value, bool tokenMajor)
    {
        for (int token = 0; token < KeyTokens; token++)
            for (int head = 0; head < KvHeads; head++)
                for (int channel = 0; channel < HeadDim; channel++)
                {
                    int offset = tokenMajor ? token * KvHeads + head : head * KeyTokens + token;
                    Write(value, offset * HeadDim + channel, 7 + head * 6);
                }
    }

    private static void AssertGroupedValues(Tensor output, bool tokenMajor, float tolerance = 2e-5f)
    {
        for (int token = 0; token < QueryTokens; token++)
            for (int head = 0; head < Heads; head++)
                for (int channel = 0; channel < HeadDim; channel++)
                {
                    int offset = tokenMajor ? token * Heads + head : head * QueryTokens + token;
                    float expected = 7 + head / (Heads / KvHeads) * 6;
                    Assert.InRange(Math.Abs(Read(output, offset * HeadDim + channel) - expected), 0, tolerance * expected);
                }
    }

    private static Tensor RandomTensor(TensorShape shape, DType dtype, Random random)
    {
        Tensor tensor = new(shape, dtype);
        for (int i = 0; i < tensor.ElementCount; i++) Write(tensor, i, (float)(random.NextDouble() - 0.5));
        return tensor;
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
