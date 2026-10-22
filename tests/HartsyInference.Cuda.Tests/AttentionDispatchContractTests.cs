using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>Unit coverage for attention fast-path contracts that must be enforced before any CUDA launch.</summary>
[Collection("CudaSerial")]
public sealed class AttentionDispatchContractTests
{
    [Theory]
    [InlineData(92099, false)]
    [InlineData(92100, true)]
    public void CudnnSdpa_RequiresUnifiedSoftmaxRuntime(long version, bool expected)
    {
        Assert.Equal(expected, CudnnRuntime.IsSdpaVersionSupported(version));
    }

    [Theory]
    [InlineData(24, false)]
    [InlineData(32, true)]
    [InlineData(36, false)]
    [InlineData(40, true)]
    [InlineData(160, true)]
    [InlineData(256, true)]
    [InlineData(264, false)]
    public void CudnnSdpa_HeadDims_AreTheProbedEnvelope(long d, bool expected) => Assert.Equal(expected, CudnnSdpa.ShapeSupported(d));

    /// <summary>F32 Sage dispatch requires both the feature opt-in and the explicit unsafe V-narrowing opt-in.</summary>
    [Fact]
    public void SageF32Dispatch_RequiresTwoExplicitOptIns()
    {
        try
        {
            KnobStore.Clear(EngineKnobs.SageAttnExplicit);
            KnobStore.Clear(EngineKnobs.SageUnsafeF32VNarrow);
            Assert.False(CudaBackend.SageF32ValueNarrowingEnabled);

            KnobStore.Set(EngineKnobs.SageAttnExplicit, true);
            Assert.False(CudaBackend.SageF32ValueNarrowingEnabled);

            KnobStore.Set(EngineKnobs.SageUnsafeF32VNarrow, true);
            Assert.True(CudaBackend.SageF32ValueNarrowingEnabled);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.SageAttnExplicit);
            KnobStore.Clear(EngineKnobs.SageUnsafeF32VNarrow);
        }
    }

    [Fact]
    public void SdpaContract_RejectsMismatchedBuffersAndMaskAxes()
    {
        using Tensor query = new(new TensorShape(2, 3, 4, 8), DType.F32);
        using Tensor key = new(new TensorShape(2, 3, 6, 8), DType.F32);
        using Tensor value = new(key.Shape, DType.F32);
        using Tensor output = new(query.Shape, DType.F32);
        using Tensor validMask = new(new TensorShape(1, 3, 4, 6), DType.F32);

        CudaBackend.ValidateScaledDotProductAttentionContract(output, query, key, value, validMask, 0.125f);

        using Tensor shortValue = new(new TensorShape(2, 3, 5, 8), DType.F32);
        Assert.Throws<ArgumentException>(() =>
            CudaBackend.ValidateScaledDotProductAttentionContract(output, query, key, shortValue, validMask, 0.125f));

        using Tensor shortOutput = new(new TensorShape(2, 3, 3, 8), DType.F32);
        Assert.Throws<ArgumentException>(() =>
            CudaBackend.ValidateScaledDotProductAttentionContract(shortOutput, query, key, value, validMask, 0.125f));

        using Tensor invalidMask = new(new TensorShape(2, 2, 4, 6), DType.F32);
        Assert.Throws<ArgumentException>(() =>
            CudaBackend.ValidateScaledDotProductAttentionContract(output, query, key, value, invalidMask, 0.125f));
    }

    /// <summary>The equal-head token-major entry takes one sequence as <c>[S, H·D]</c>, <c>[1, S, H·D]</c> or <c>[1, S, H, D]</c>, the
    /// shapes the grouped entry takes, and refuses a batch or a width that is not H·D.</summary>
    [Fact]
    public void TokenMajorContract_TakesOneSequenceInEachLayout_AndRefusesABatch()
    {
        const int Heads = 2, HeadDim = 8, S = 5, Inner = Heads * HeadDim;
        TensorShape[] layouts = [new(S, Inner), new(1, S, Inner), new(1, S, Heads, HeadDim)];
        foreach (TensorShape operands in layouts)
        {
            foreach (TensorShape output in layouts)
            {
                using Tensor q = new(operands, DType.F16), k = new(operands, DType.F16), v = new(operands, DType.F16), o = new(output, DType.F16);
                CudaBackend.ValidateTokenMajorAttentionContract(o, q, k, v, Heads, HeadDim, 0.25f);
                CudaBackend.ValidateTokenMajorGqaAttentionContract(o, q, k, v, Heads, Heads, HeadDim, 0.25f);
            }
        }

        using Tensor query = new(new TensorShape(1, S, Heads, HeadDim), DType.F16), key = new(query.Shape, DType.F16);
        using Tensor value = new(query.Shape, DType.F16);
        using Tensor batched = new(new TensorShape(2, S, Inner), DType.F16), wide = new(new TensorShape(1, S, Inner + HeadDim), DType.F16);
        using Tensor short3 = new(new TensorShape(1, S - 1, Inner), DType.F16);
        foreach (Tensor output in (Tensor[])[batched, wide, short3])
            Assert.Throws<ArgumentException>(() => CudaBackend.ValidateTokenMajorAttentionContract(output, query, key, value, Heads, HeadDim, 0.25f));
    }

    [Fact]
    public void FlashContract_RejectsInvalidCapacityGqaAndAuxiliaryLengths()
    {
        using Tensor query = new(new TensorShape(1, 4, 3, 8), DType.F32);
        using Tensor key = new(new TensorShape(1, 2, 5, 8), DType.F32);
        using Tensor value = new(key.Shape, DType.F32);
        using Tensor output = new(query.Shape, DType.F32);

        CudaBackend.ValidateFlashAttentionContract(
            output, query, key, value, 5, 2, causal: false, qOffset: 0, 0.125f, 0f, null, 0, null);
        CudaBackend.ValidateFlashAttentionContract(
            output, query, key, value, 0, 2, causal: true, qOffset: 0, 0.125f, 0f, null, 0, null,
            positionOnDevice: true);

        Assert.Throws<ArgumentException>(() => CudaBackend.ValidateFlashAttentionContract(
            output, query, key, value, 5, 1, false, 0, 0.125f, 0f, null, 0, null));
        Assert.Throws<ArgumentOutOfRangeException>(() => CudaBackend.ValidateFlashAttentionContract(
            output, query, key, value, 6, 2, false, 0, 0.125f, 0f, null, 0, null));

        using Tensor shortValue = new(new TensorShape(1, 2, 4, 8), DType.F32);
        Assert.Throws<ArgumentException>(() => CudaBackend.ValidateFlashAttentionContract(
            output, query, key, shortValue, 5, 2, false, 0, 0.125f, 0f, null, 0, null));

        using Tensor shortSink = new(new TensorShape(3), DType.F32);
        Assert.Throws<ArgumentException>(() => CudaBackend.ValidateFlashAttentionContract(
            output, query, key, value, 5, 2, false, 0, 0.125f, 0f, shortSink, 0, null));
    }
}
