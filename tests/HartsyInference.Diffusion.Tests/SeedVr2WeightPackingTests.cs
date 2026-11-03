using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Gguf.Codecs;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>SeedVR2's DiT packing: which lever packs, which weights pack, and that a packed weight decodes back to within
/// Q8_0's rounding. A packed weight that decodes wrong renders as noise, so the round trip is checked value by value.</summary>
public sealed class SeedVr2WeightPackingTests
{
    [Fact]
    public void Choose_PacksOnlyWhenTheLeverIsOn()
    {
        Assert.Equal(DType.Q8_0, SeedVr2WeightPacking.Choose(LeverState.On));
        Assert.Null(SeedVr2WeightPacking.Choose(LeverState.Off));
        Assert.Null(SeedVr2WeightPacking.Choose(LeverState.Auto));
    }

    [Theory]
    [InlineData("blocks.0.attn.proj_qkv.vid.weight", 96, 64, true)]
    [InlineData("txt_in.weight", 2560, 5120, true)]
    [InlineData("vid_out.proj.weight", 64, 2560, true)]
    [InlineData("emb_in.proj_out.weight", 15360, 2560, false)]
    [InlineData("emb_in.proj_in.weight", 2560, 256, false)]
    [InlineData("blocks.0.attn.norm_q.vid.weight", 128, 0, false)]
    [InlineData("txt_in.bias", 2560, 0, false)]
    [InlineData("vid_in.proj.weight", 2560, 70, false)]
    public void IsPackable_TakesLinearsOutsideTheTimestepEmbedding(string key, int rows, int cols, bool expected)
    {
        using Tensor weight = cols == 0 ? new Tensor(new TensorShape(rows), DType.F32) : new Tensor(new TensorShape(rows, cols), DType.F32);
        Assert.Equal(expected, SeedVr2WeightPacking.IsPackable(key, weight, DType.Q8_0));
    }

    [Fact]
    public void PackLinears_RoundTripsWithinQ8Rounding_AndLeavesOtherTensorsAlone()
    {
        const int rows = 24, cols = 96;
        using Tensor linear = new(new TensorShape(rows, cols), DType.F32);
        using Tensor bias = new(new TensorShape(rows), DType.F32);
        Span<float> values = linear.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = MathF.Sin(i * 0.37f) * 0.8f + MathF.Cos(i * 0.11f) * 0.2f;
        bias.AsSpan<float>().Fill(0.5f);
        Dictionary<string, Tensor> weights = new()
        {
            ["blocks.0.mlp.vid.proj_in.weight"] = linear,
            ["blocks.0.mlp.vid.proj_in.bias"] = bias,
            ["emb_in.proj_hid.weight"] = linear,
        };
        List<Tensor> owned = [];

        SeedVr2WeightPacking.PackLinears(weights, DType.Q8_0, owned);

        Tensor packed = Assert.Single(owned);
        Assert.Equal(DType.Q8_0, packed.DType);
        Assert.Same(packed, weights["blocks.0.mlp.vid.proj_in.weight"]);
        Assert.Same(bias, weights["blocks.0.mlp.vid.proj_in.bias"]);
        Assert.Same(linear, weights["emb_in.proj_hid.weight"]);

        float[] decoded = new float[rows * cols];
        unsafe
        {
            fixed (float* dst = decoded)
            {
                GgufCodecRegistry.Get(DType.Q8_0).DequantizeToF32((byte*)packed.DataPointer, dst, rows * cols);
            }
        }
        // Q8_0 keeps one fp16 scale per 32 weights and rounds each to the nearest step (a step is the block's max / 127), so the
        // error is about half a step. The bound is a full global step: loose enough for the scale's rounding, far below the
        // error a wrong layout or a misread block produces, which is on the order of the weight itself.
        float maxAbs = values.ToArray().Max(MathF.Abs);
        float worst = 0;
        for (int i = 0; i < decoded.Length; i++)
            worst = MathF.Max(worst, MathF.Abs(decoded[i] - values[i]));
        Assert.True(worst <= maxAbs / 127f, $"worst Q8_0 error {worst} exceeds one step of {maxAbs / 127f}");
    }
}
