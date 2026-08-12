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
    [InlineData(92500, true)]
    public void CudnnSdpa_RequiresUnifiedSoftmaxRuntime(long version, bool expected)
    {
        Assert.Equal(expected, CudnnRuntime.IsSdpaVersionSupported(version));
    }

    /// <summary>F32 Sage dispatch requires both the feature opt-in and the explicit unsafe V-narrowing opt-in.</summary>
    [Fact]
    public void SageF32Dispatch_RequiresTwoExplicitOptIns()
    {
        string? previousSage = Environment.GetEnvironmentVariable("HARTSY_SAGE_ATTN");
        string? previousNarrowing = Environment.GetEnvironmentVariable("HARTSY_SAGE_UNSAFE_F32_V_NARROW");
        try
        {
            Environment.SetEnvironmentVariable("HARTSY_SAGE_ATTN", null);
            Environment.SetEnvironmentVariable("HARTSY_SAGE_UNSAFE_F32_V_NARROW", null);
            Assert.False(CudaBackend.SageF32ValueNarrowingEnabled);

            Environment.SetEnvironmentVariable("HARTSY_SAGE_ATTN", "1");
            Assert.False(CudaBackend.SageF32ValueNarrowingEnabled);

            Environment.SetEnvironmentVariable("HARTSY_SAGE_UNSAFE_F32_V_NARROW", "1");
            Assert.True(CudaBackend.SageF32ValueNarrowingEnabled);
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_SAGE_ATTN", previousSage);
            Environment.SetEnvironmentVariable("HARTSY_SAGE_UNSAFE_F32_V_NARROW", previousNarrowing);
        }
    }

    /// <summary>FlashAttention-v2 accepts only exact F32 MHA shapes with complete query tiles on TF32 hardware.</summary>
    [Fact]
    public void FlashV2Contract_RejectsMixedDtypesAndPartialTiles()
    {
        const float Scale = 0.125f;
        using Tensor query = new Tensor(new TensorShape(1, 2, 32, 64), DType.F32);
        using Tensor key = new Tensor(new TensorShape(1, 2, 17, 64), DType.F32);
        using Tensor value = new Tensor(new TensorShape(1, 2, 17, 64), DType.F32);
        using Tensor output = new Tensor(query.Shape, DType.F32);
        Assert.True(CudaBackend.FlashAttentionV2ContractSatisfied(
            output, query, key, value, null, Scale, tf32Available: true));

        using Tensor tailQuery = new Tensor(new TensorShape(1, 2, 33, 64), DType.F32);
        using Tensor tailOutput = new Tensor(tailQuery.Shape, DType.F32);
        Assert.False(CudaBackend.FlashAttentionV2ContractSatisfied(
            tailOutput, tailQuery, key, value, null, Scale, tf32Available: true));

        using Tensor f16Key = new Tensor(key.Shape, DType.F16);
        Assert.False(CudaBackend.FlashAttentionV2ContractSatisfied(
            output, query, f16Key, value, null, Scale, tf32Available: true));

        using Tensor wrongHeads = new Tensor(new TensorShape(1, 1, 17, 64), DType.F32);
        Assert.False(CudaBackend.FlashAttentionV2ContractSatisfied(
            output, query, wrongHeads, wrongHeads, null, Scale, tf32Available: true));

        using Tensor f16Output = new Tensor(query.Shape, DType.F16);
        Assert.False(CudaBackend.FlashAttentionV2ContractSatisfied(
            f16Output, query, key, value, null, Scale, tf32Available: true));
        Assert.False(CudaBackend.FlashAttentionV2ContractSatisfied(
            output, query, key, value, null, float.NaN, tf32Available: true));
        Assert.False(CudaBackend.FlashAttentionV2ContractSatisfied(
            output, query, key, value, null, Scale, tf32Available: false));
    }

    [Theory]
    [InlineData(65_535, 1, true)]
    [InlineData(1, 65_535, true)]
    [InlineData(0, 1, false)]
    [InlineData(1, 0, false)]
    [InlineData(65_536, 1, false)]
    [InlineData(1, 65_536, false)]
    public void FlashV2GridContract_EnforcesCudaYzLimits(long batch, long heads, bool expected)
    {
        Assert.Equal(expected, CudaBackend.FlashAttentionV2GridDimensionsSupported(batch, heads));
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

    /// <summary>cuDNN plan identity includes exact scale bits, preventing stale scalar reuse by shape.</summary>
    [Fact]
    public void CudnnSdpaPlanKey_DistinguishesScaleBits()
    {
        CudnnSdpa.PlanKey first = new CudnnSdpa.PlanKey(
            1, 4, 256, 256, 64, BitConverter.SingleToInt32Bits(0.125f), false, 1);
        CudnnSdpa.PlanKey second = new CudnnSdpa.PlanKey(
            1, 4, 256, 256, 64, BitConverter.SingleToInt32Bits(4.0f), false, 1);

        Assert.NotEqual(first, second);
    }
}
