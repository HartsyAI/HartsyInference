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

    /// <summary>The Sage V stride: whole 16-byte rows, and never a power of two from 2048 up (those alias in L2).</summary>
    [Theory]
    [InlineData(1, 8)]
    [InlineData(250, 256)]
    [InlineData(1024, 1024)]
    [InlineData(1025, 1032)]
    [InlineData(2040, 2040)]
    [InlineData(2048, 2056)]
    [InlineData(2041, 2056)]
    [InlineData(8192, 8200)]
    [InlineData(17480, 17480)]
    public void SageSkvPad_IsWholeRowsAndNeverALargePowerOfTwo(long skv, long expected) => Assert.Equal(expected, CudaKernels.SageSkvPad(skv));

    /// <summary>INT8 v2's workspace: counters at the head for every shape, regions 256-byte aligned in order, V sized by format.</summary>
    [Fact]
    public void Int8V2Layout_KeepsTheCountersFirstAndSizesEveryRegion()
    {
        Int8AttentionV2Layout seq = new(1, 24, 8192, 8192, Int8AttentionV2Pv.U8SequenceScales);
        Assert.Equal((64, 128, 8192L), (seq.Tiles, seq.Chunks, seq.KvPad));
        Assert.Equal(Int8AttentionV2Layout.CounterSlots * 4L, seq.Q8);
        long[] starts = [seq.Q8, seq.QScale, seq.K8, seq.KScale, seq.V8, seq.VScale, seq.KMean, seq.KPartial, seq.VPartial, seq.Bytes];
        Assert.All(starts, x => Assert.Equal(0, x % 256));
        Assert.Equal(starts.Order(), starts);
        Assert.Equal(24L * 8192 * 128, seq.QScale - seq.Q8);
        Assert.Equal(24L * 8192 * 128, seq.VScale - seq.V8);
        Int8AttentionV2Layout tile = new(2, 3, 300, 333, Int8AttentionV2Pv.U8TileScales);
        Assert.Equal((3, 6, 384L), (tile.Tiles, tile.Chunks, tile.KvPad));
        Assert.Equal(tile.Bytes, tile.VPartial);
        Assert.Equal(6L * 3 * 128 * 4, tile.KMean - tile.VScale);
        Int8AttentionV2Layout f16 = new(1, 2, 100, 100, Int8AttentionV2Pv.F16);
        Assert.Equal(2L * 128 * 128 * 2, f16.VScale - f16.V8);
        Assert.Throws<ArgumentOutOfRangeException>(() => new Int8AttentionV2Layout(1, Int8AttentionV2Layout.CounterSlots + 1, 8, 8,
            Int8AttentionV2Pv.F16));
    }

    /// <summary>The V key permutation is a bijection on every 32-key chunk that puts a lane's two keys of four n-tiles (2t, 2t+1 of
    /// n-tiles 0..3) at physical bytes 4t..4t+3 and 16+4t..16+4t+3, the m16n8k32 A-fragment the lane already holds.</summary>
    [Fact]
    public void Int8V2KeyPermutation_TurnsALanesAccumulatorsIntoItsAFragment()
    {
        for (int chunk = 0; chunk < 4; chunk++)
        {
            int[] physical = [.. Enumerable.Range(32 * chunk, 32).Select(Int8AttentionV2Layout.PhysicalKey)];
            Assert.Equal(Enumerable.Range(32 * chunk, 32), physical.Order());
        }
        for (int t = 0; t < 4; t++)
            for (int n = 0; n < 4; n++)
                for (int e = 0; e < 2; e++)
                {
                    int key = 8 * n + 2 * t + e;
                    Assert.Equal(16 * (n >> 1) + 4 * t + 2 * (n & 1) + e, Int8AttentionV2Layout.PhysicalKey(key));
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

    /// <summary>The token-major entry bounds each tensor at Int32.MaxValue elements, as the head-major entry does, and leaves the score
    /// matrix to its rows: 46341 query and key rows (SD1.5's self-attention at 1728² and up) are taken.</summary>
    [Fact]
    public void TokenMajorContract_TakesAScoreMatrixPastInt32()
    {
        const int Heads = 1, HeadDim = 8, S = 46341;
        Assert.True((long)S * S > int.MaxValue);
        using Tensor q = new(new TensorShape(S, Heads * HeadDim), DType.F16), k = new(q.Shape, DType.F16), v = new(q.Shape, DType.F16);
        using Tensor o = new(q.Shape, DType.F16);

        CudaBackend.ValidateTokenMajorAttentionContract(o, q, k, v, Heads, HeadDim, 0.25f);
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
