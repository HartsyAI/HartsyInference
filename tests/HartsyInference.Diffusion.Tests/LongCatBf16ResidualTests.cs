using HartsyInference.Core.Numerics;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Tests.Shared;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The BF16 residual rounding of LongCat's blocks: the helper rounds to the nearest even BF16 value, and a block rounds its residual sums
/// only on a BF16 or INT8 release (the reference's x is BF16 there), never on the F32 route, and not when the knob is off.</summary>
[Collection(GlobalKnobStore.Name)]
public sealed class LongCatBf16ResidualTests
{
    [Fact]
    public void RoundToBf16IsTheNearestEvenBf16Value()
    {
        // 1 + 2^-8 lies exactly between the BF16 values 1.0 and 1 + 2^-7; the even mantissa is 1.0.
        float[] values = [1.0f, 1.00390625f, 3.14159f, -2.71828f, 1e-3f, 65504f];
        using CpuBackend cpu = new();
        using Tensor input = new(new TensorShape(values.Length), DType.F32);
        values.AsSpan().CopyTo(input.AsSpan<float>());
        using Tensor rounded = LongCatOps.RoundToBf16(cpu, input);

        ReadOnlySpan<float> got = rounded.AsReadOnlySpan<float>();
        for (int i = 0; i < values.Length; i++)
            Assert.Equal(BFloat16Bits.ToFloat(BFloat16Bits.FromFloat(values[i])), got[i]);
        Assert.Equal(1f, got[1]);
    }

    [Fact]
    public void ABlockRoundsItsResidualOnlyOnAReducedPrecisionRelease()
    {
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(LongCatTestData.Tiny, seed: 5);
        List<Tensor> owned = [];

        LongCatBlock f32 = new(LongCatTestData.Tiny);
        f32.LoadWeights(weights, "blocks.0", owned);
        Assert.False(f32.Bf16Residual);

        Tensor qkv = weights["blocks.0.attn.qkv.weight"];
        weights["blocks.0.attn.qkv.weight"] = qkv.CastTo(DType.BF16);
        LongCatBlock bf16;
        using (KnobProfile.Create("bf16-residual-on").With(EngineKnobs.LongcatBf16Residual, true).Push())
        {
            bf16 = new LongCatBlock(LongCatTestData.Tiny);
            bf16.LoadWeights(weights, "blocks.0", owned);
        }
        Assert.True(bf16.Bf16Residual);
        LongCatBlock defaults = new(LongCatTestData.Tiny);
        defaults.LoadWeights(weights, "blocks.0", owned);
        Assert.False(defaults.Bf16Residual);

        using (KnobProfile.Create("bf16-residual-off").With(EngineKnobs.LongcatBf16Residual, false).Push())
        {
            LongCatBlock off = new(LongCatTestData.Tiny);
            off.LoadWeights(weights, "blocks.0", owned);
            Assert.False(off.Bf16Residual);
        }

        foreach (Tensor t in owned) t.Dispose();
        foreach (Tensor t in weights.Values) t.Dispose();
    }
}
