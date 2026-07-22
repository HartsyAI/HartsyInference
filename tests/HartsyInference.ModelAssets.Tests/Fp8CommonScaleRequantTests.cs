using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Tests for <see cref="CheckpointConvertUtils.RequantizeToCommonFp8Scale"/> — the load-time
/// common-scale rewrite that unblocks concat-fusion (Q/K/V, FFN w1/w3) of separately-scaled fp8 tensors.
/// The load-bearing claims: (1) decoded values are preserved within E4M3 rounding for normal-range weights,
/// (2) all tensors exit with one identical <see cref="Tensor.Fp8ScaleFactor"/>, (3) the rewrite is in-place
/// (same Tensor objects), (4) equal-scale groups are untouched, (5) power-of-two scale ratios round-trip
/// EXACTLY for values that stay normal (pure exponent shift).</summary>
public sealed unsafe class Fp8CommonScaleRequantTests
{
    private static Tensor MakeFp8(float[] realValues, float scale)
    {
        Tensor f32 = new Tensor(new TensorShape(1, realValues.Length), DType.F32);
        float* p = (float*)f32.DataPointer;
        for (int i = 0; i < realValues.Length; i++) p[i] = realValues[i] / scale;
        Tensor fp8 = f32.CastTo(DType.F8E4M3);
        f32.Dispose();
        fp8.Fp8ScaleFactor = scale;
        return fp8;
    }

    private static float[] Decode(Tensor fp8)
    {
        Tensor f32 = fp8.CastTo(DType.F32);
        float[] result = new float[f32.ElementCount];
        float* p = (float*)f32.DataPointer;
        for (int i = 0; i < result.Length; i++) result[i] = p[i];
        f32.Dispose();
        return result;
    }

    [Fact]
    public void PowerOfTwoScaleRatio_NormalRangeValues_RoundTripExactly()
    {
        // Scales 4.0 and 1.0 (ratio 4 = 2²): rescaling is a pure exponent shift for values whose
        // common-scale encoding stays in E4M3's normal range (|v|/s* ≥ 2⁻⁶ → |v| ≥ 0.0625 at s*=4).
        float[] valuesA = [1.0f, -2.0f, 3.5f, 0.25f, -0.125f];
        float[] valuesB = [0.5f, -1.5f, 2.0f, 0.75f, -0.25f];
        using Tensor a = MakeFp8(valuesA, scale: 4.0f);
        using Tensor b = MakeFp8(valuesB, scale: 1.0f);
        float[] beforeA = Decode(a);
        float[] beforeB = Decode(b);

        float err = CheckpointConvertUtils.RequantizeToCommonFp8Scale(a, b);

        Assert.Equal(4.0f, a.Fp8ScaleFactor);
        Assert.Equal(4.0f, b.Fp8ScaleFactor);
        Assert.Equal(beforeA, Decode(a));
        Assert.Equal(beforeB, Decode(b));
        Assert.Equal(0f, err);
    }

    [Fact]
    public void NonPowerOfTwoRatio_StaysWithinE4M3Rounding()
    {
        // Scales 3.0 and 1.0: a non-power-of-two rescale re-rounds each mantissa once. E4M3 has 3 mantissa
        // bits → half-ulp relative error ≤ 1/16 for normal values; normalized to amax the bound holds too.
        float[] valuesA = [2.5f, -1.75f, 3.0f, 1.0f];
        float[] valuesB = [0.9f, -0.6f, 0.45f, 0.3f];
        using Tensor a = MakeFp8(valuesA, scale: 3.0f);
        using Tensor b = MakeFp8(valuesB, scale: 1.0f);
        float[] beforeB = Decode(b);

        float err = CheckpointConvertUtils.RequantizeToCommonFp8Scale(a, b);

        Assert.Equal(3.0f, a.Fp8ScaleFactor);
        Assert.Equal(3.0f, b.Fp8ScaleFactor);
        Assert.True(err <= 1.0f / 16.0f, $"amax-normalized requant error {err} exceeds the E4M3 half-ulp bound 1/16");

        float[] afterB = Decode(b);
        float amaxB = 0f;
        foreach (float v in beforeB) amaxB = Math.Max(amaxB, Math.Abs(v));
        for (int i = 0; i < beforeB.Length; i++)
            Assert.True(Math.Abs(afterB[i] - beforeB[i]) <= amaxB / 16.0f,
                $"element {i}: {beforeB[i]} → {afterB[i]} moved more than amax/16");
    }

    [Fact]
    public void EqualScales_AreLeftUntouched()
    {
        float[] values = [1.0f, -0.5f, 2.0f];
        using Tensor a = MakeFp8(values, scale: 2.0f);
        using Tensor b = MakeFp8(values, scale: 2.0f);
        byte firstByteA = *(byte*)a.DataPointer;

        float err = CheckpointConvertUtils.RequantizeToCommonFp8Scale(a, b);

        Assert.Equal(0f, err);
        Assert.Equal(2.0f, a.Fp8ScaleFactor);
        Assert.Equal(firstByteA, *(byte*)a.DataPointer);
    }

    [Fact]
    public void ThreeTensorQkvGroup_UnifiesToMaxScale()
    {
        // The actual Q/K/V fusion shape: three projections with distinct scales unify to the max.
        using Tensor q = MakeFp8([1.0f, -2.0f], scale: 1.5f);
        using Tensor k = MakeFp8([0.5f, 0.25f], scale: 6.0f);
        using Tensor v = MakeFp8([3.0f, -1.0f], scale: 3.0f);
        float[] beforeQ = Decode(q);
        float[] beforeK = Decode(k);
        float[] beforeV = Decode(v);

        CheckpointConvertUtils.RequantizeToCommonFp8Scale(q, k, v);

        Assert.Equal(6.0f, q.Fp8ScaleFactor);
        Assert.Equal(6.0f, k.Fp8ScaleFactor);
        Assert.Equal(6.0f, v.Fp8ScaleFactor);

        // K had the max scale — bit-untouched. Q and V re-encoded within E4M3 rounding of their amax.
        Assert.Equal(beforeK, Decode(k));
        float[] afterQ = Decode(q);
        float[] afterV = Decode(v);
        for (int i = 0; i < beforeQ.Length; i++)
        {
            Assert.True(Math.Abs(afterQ[i] - beforeQ[i]) <= 2.0f / 16.0f);
            Assert.True(Math.Abs(afterV[i] - beforeV[i]) <= 3.0f / 16.0f);
        }
    }

    [Fact]
    public void NonFp8Tensor_Throws()
    {
        using Tensor a = MakeFp8([1.0f], scale: 1.0f);
        using Tensor f32 = new Tensor(new TensorShape(1, 1), DType.F32);
        Assert.Throws<ArgumentException>(() => CheckpointConvertUtils.RequantizeToCommonFp8Scale(a, f32));
    }

    [Fact]
    public void SubnormalFlush_IsBoundedByDocumentedThreshold()
    {
        // A tiny weight (0.001 at scale 1) rescaled to s*=8 encodes as 0.000125 in common-scale units —
        // below E4M3's min subnormal (2⁻⁹ ≈ 0.00195) → flushes to zero. The documented behavior: bounded
        // absolute loss (the original tiny magnitude), negligible relative to the tensor amax.
        float[] tiny = [1.0f, 0.001f];
        using Tensor a = MakeFp8([8.0f, 4.0f], scale: 8.0f);
        using Tensor b = MakeFp8(tiny, scale: 1.0f);

        float err = CheckpointConvertUtils.RequantizeToCommonFp8Scale(a, b);

        float[] afterB = Decode(b);
        Assert.Equal(0f, afterB[1]);
        // amax-normalized loss of the flushed element: 0.001 / 1.0 = 0.001 ≪ the 1/16 normal-range bound.
        Assert.True(err <= 1.0f / 16.0f);
    }
}
