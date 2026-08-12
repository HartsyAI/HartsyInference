using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Tests for <see cref="CheckpointConvertUtils.FuseSwiGluPairs"/> +
/// <see cref="CheckpointConvertUtils.ConcatRowsHost"/> — the H3.2 load-time w1/w3 → w13 fusion
/// (INFERENCE_ACCEL_GRIND). Load-bearing claims: (1) fp8 pairs with differing scales are unified then
/// row-concatenated, and the fused tensor DECODES to the originals within the requant bound; (2) F32
/// pairs concatenate byte-exactly; (3) key bookkeeping (w1/w3 removed, w13 added); (4) unfusable pairs
/// (shape mismatch) are left untouched; (5) the fused row order is [w1; w3] — the order
/// <c>Ideogram4Block.ForwardSwiGlu</c>'s slices assume.</summary>
public sealed unsafe class SwiGluFusionTests
{
    private static Tensor MakeFp8Rows(int rows, int cols, float scale, int seed)
    {
        Tensor f32 = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)f32.DataPointer;
        Random rng = new Random(seed);
        for (int i = 0; i < rows * cols; i++) p[i] = (float)((rng.NextDouble() * 2.0 - 1.0)) / scale;
        Tensor fp8 = f32.CastTo(DType.F8E4M3);
        f32.Dispose();
        fp8.Fp8ScaleFactor = scale;
        return fp8;
    }

    private static Tensor MakeF32Rows(int rows, int cols, int seed)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        for (int i = 0; i < rows * cols; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return t;
    }

    private static float[] DecodeAll(Tensor t)
    {
        Tensor f32 = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        float[] result = new float[f32.ElementCount];
        float* p = (float*)f32.DataPointer;
        for (int i = 0; i < result.Length; i++) result[i] = p[i];
        if (!ReferenceEquals(f32, t)) f32.Dispose();
        return result;
    }

    [Fact]
    public void Fp8Pair_DifferentScales_FusesWithinRequantBound()
    {
        const int inner = 8, hidden = 16;
        Dictionary<string, Tensor> weights = new()
        {
            ["layers.0.feed_forward.w1.weight"] = MakeFp8Rows(inner, hidden, scale: 1.0f, seed: 1),
            ["layers.0.feed_forward.w3.weight"] = MakeFp8Rows(inner, hidden, scale: 4.0f, seed: 2),
        };
        float[] w1Before = DecodeAll(weights["layers.0.feed_forward.w1.weight"]);
        float[] w3Before = DecodeAll(weights["layers.0.feed_forward.w3.weight"]);

        (int fused, float worst) = CheckpointConvertUtils.FuseSwiGluPairs(
            weights, ".feed_forward.w1.weight", ".feed_forward.w3.weight", ".feed_forward.w13.weight");

        Assert.Equal(1, fused);
        Assert.True(worst <= 1f / 16f, $"requant error {worst:E2} above bound");
        Assert.False(weights.ContainsKey("layers.0.feed_forward.w1.weight"));
        Assert.False(weights.ContainsKey("layers.0.feed_forward.w3.weight"));
        Tensor w13 = weights["layers.0.feed_forward.w13.weight"];
        Assert.Equal(2 * inner, (int)w13.Shape[0]);
        Assert.Equal(hidden, (int)w13.Shape[1]);

        // Row order [w1; w3], decoded within the documented per-tensor bound (amax-normalized ≤ 1/16;
        // these uniform values stay normal-range, so the practical error is one E4M3 rounding).
        float[] fusedVals = DecodeAll(w13);
        for (int i = 0; i < inner * hidden; i++)
        {
            Assert.True(Math.Abs(fusedVals[i] - w1Before[i]) <= 1f / 16f + 1e-6f,
                $"w1 row drift at {i}: {fusedVals[i]} vs {w1Before[i]}");
            Assert.True(Math.Abs(fusedVals[inner * hidden + i] - w3Before[i]) <= 1f / 16f + 1e-6f,
                $"w3 row drift at {i}: {fusedVals[inner * hidden + i]} vs {w3Before[i]}");
        }
    }

    [Fact]
    public void F32Pair_ConcatenatesByteExactly()
    {
        const int inner = 4, hidden = 8;
        Tensor w1 = MakeF32Rows(inner, hidden, seed: 3);
        Tensor w3 = MakeF32Rows(inner, hidden, seed: 4);
        float[] w1Vals = DecodeAll(w1);
        float[] w3Vals = DecodeAll(w3);
        Dictionary<string, Tensor> weights = new()
        {
            ["layers.7.feed_forward.w1.weight"] = w1,
            ["layers.7.feed_forward.w3.weight"] = w3,
        };

        (int fused, float worst) = CheckpointConvertUtils.FuseSwiGluPairs(
            weights, ".feed_forward.w1.weight", ".feed_forward.w3.weight", ".feed_forward.w13.weight");

        Assert.Equal(1, fused);
        Assert.Equal(0f, worst);
        float[] fusedVals = DecodeAll(weights["layers.7.feed_forward.w13.weight"]);
        for (int i = 0; i < inner * hidden; i++)
        {
            Assert.Equal(w1Vals[i], fusedVals[i]);
            Assert.Equal(w3Vals[i], fusedVals[inner * hidden + i]);
        }
    }

    [Fact]
    public void MismatchedShapes_LeftUnfused()
    {
        Dictionary<string, Tensor> weights = new()
        {
            ["layers.0.feed_forward.w1.weight"] = MakeF32Rows(4, 8, seed: 5),
            ["layers.0.feed_forward.w3.weight"] = MakeF32Rows(4, 12, seed: 6),   // different cols
        };

        (int fused, _) = CheckpointConvertUtils.FuseSwiGluPairs(
            weights, ".feed_forward.w1.weight", ".feed_forward.w3.weight", ".feed_forward.w13.weight");

        Assert.Equal(0, fused);
        Assert.True(weights.ContainsKey("layers.0.feed_forward.w1.weight"));
        Assert.True(weights.ContainsKey("layers.0.feed_forward.w3.weight"));
        Assert.False(weights.ContainsKey("layers.0.feed_forward.w13.weight"));
    }

    /// <summary>Once the int8 companions have been folded onto <see cref="Tensor.QuantInfo"/> their KEYS are gone, so
    /// the comfy_quant key guard no longer sees the pair; concatenating would drop both row scales silently.</summary>
    [Fact]
    public void Int8PairWithQuantInfo_LeftUnfused()
    {
        Tensor rowScale = new Tensor(new TensorShape(4), DType.F32);
        Dictionary<string, Tensor> weights = new()
        {
            ["layers.0.feed_forward.w1.weight"] = new Tensor(new TensorShape(4, 8), DType.I8),
            ["layers.0.feed_forward.w3.weight"] = new Tensor(new TensorShape(4, 8), DType.I8),
        };
        foreach (Tensor t in weights.Values)
            t.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = rowScale };

        (int fused, _) = CheckpointConvertUtils.FuseSwiGluPairs(
            weights, ".feed_forward.w1.weight", ".feed_forward.w3.weight", ".feed_forward.w13.weight");

        Assert.Equal(0, fused);
        Assert.True(weights.ContainsKey("layers.0.feed_forward.w1.weight"));
        Assert.False(weights.ContainsKey("layers.0.feed_forward.w13.weight"));
        rowScale.Dispose();
        foreach (Tensor t in weights.Values) t.Dispose();
    }
}
