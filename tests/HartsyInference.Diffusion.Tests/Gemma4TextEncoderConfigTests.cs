using HartsyInference.Diffusion.Models.TextEncoders;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Guards the two pieces of Gemma 4 geometry that are easy to get plausibly wrong and impossible to
/// notice afterwards: which layers are global, and the global layers' partial-rotary frequency table.</summary>
public sealed class Gemma4TextEncoderConfigTests
{
    private static readonly Gemma4TextEncoderConfig Config = Gemma4TextEncoderConfig.Gemma4_12B;

    [Fact]
    public void LayerTypes_EverySixthLayerIsGlobal()
    {
        int globalCount = 0;
        for (int i = 0; i < Config.NumLayers; i++)
        {
            bool expected = i % 6 == 5;
            Assert.Equal(expected, Config.IsGlobalLayer(i));
            if (expected) globalCount++;
        }
        Assert.Equal(8, globalCount);
        Assert.False(Config.IsGlobalLayer(0));
        Assert.True(Config.IsGlobalLayer(5));
        Assert.True(Config.IsGlobalLayer(47));
    }

    [Fact]
    public void LayerGeometry_DiffersBetweenSlidingAndGlobal()
    {
        Assert.Equal(256, Config.HeadDimFor(0));
        Assert.Equal(8, Config.KvHeadsFor(0));
        Assert.False(Config.KEqualsVFor(0));

        Assert.Equal(512, Config.HeadDimFor(5));
        Assert.Equal(1, Config.KvHeadsFor(5));
        Assert.True(Config.KEqualsVFor(5));
    }

    [Fact]
    public void GlobalInverseFrequencies_Are64RealPairsPaddedWith192Zeros()
    {
        double[] inv = Config.BuildInverseFrequencies(5);
        Assert.Equal(256, inv.Length);
        for (int k = 0; k < 64; k++)
        {
            // The exponent denominator is the FULL global head dim (512), not the rotary width (128).
            double expected = 1.0 / System.Math.Pow(1_000_000.0, 2.0 * k / 512.0);
            Assert.Equal(expected, inv[k], 12);
        }
        for (int k = 64; k < 256; k++) Assert.Equal(0.0, inv[k]);
    }

    [Fact]
    public void SlidingInverseFrequencies_RotateEveryPair()
    {
        double[] inv = Config.BuildInverseFrequencies(0);
        Assert.Equal(128, inv.Length);
        for (int k = 0; k < 128; k++)
        {
            double expected = 1.0 / System.Math.Pow(10_000.0, 2.0 * k / 256.0);
            Assert.Equal(expected, inv[k], 12);
            Assert.NotEqual(0.0, inv[k]);
        }
    }

    [Fact]
    public void EmbeddingScale_IsSqrtOfHiddenSize()
    {
        Assert.Equal(System.MathF.Sqrt(3840f), Config.EmbeddingScale, 5);
    }

    [Fact]
    public void UnsupportedGemma3nMechanisms_AreRejectedNotIgnored()
    {
        Assert.Throws<NotSupportedException>(() =>
            new Gemma4TextEncoder(Config with { HiddenSizePerLayerInput = 256 }));
        Assert.Throws<NotSupportedException>(() =>
            new Gemma4TextEncoder(Config with { NumKvSharedLayers = 18 }));
    }
}
