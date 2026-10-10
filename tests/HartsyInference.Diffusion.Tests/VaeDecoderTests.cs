using HartsyInference.Diffusion.Models.Vae;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tests for VAE config presets: the latent-channel and scaling-factor golden values that a wrong preset
/// would silently mis-decode with.</summary>
public sealed class VaeDecoderTests
{
    private const float Tolerance = 1e-5f;

    [Fact]
    public void VaeConfig_Sd15_HasCorrectValues()
    {
        VaeConfig config = VaeConfig.Sd15;

        Assert.Equal(4, config.LatentChannels);
        Assert.Equal(0.18215f, config.ScalingFactor);
        Assert.Null(config.ShiftFactor);
        Assert.True(config.UsePostQuantConv);
        Assert.True(config.UseQuantConv);
        Assert.Equal(512, config.SampleSize);
        Assert.Equal([128, 256, 512, 512], config.BlockOutChannels);
    }

    [Fact]
    public void VaeConfig_Sd3_Has16LatentChannels()
    {
        VaeConfig config = VaeConfig.Sd3;

        Assert.Equal(16, config.LatentChannels);
        Assert.Equal(1.5305f, config.ScalingFactor);
        Assert.NotNull(config.ShiftFactor);
        Assert.InRange(config.ShiftFactor!.Value, 0.0609f - Tolerance, 0.0609f + Tolerance);
        Assert.False(config.UsePostQuantConv);
        Assert.False(config.UseQuantConv);
    }

    [Fact]
    public void VaeConfig_Flux_HasCorrectScaling()
    {
        VaeConfig config = VaeConfig.Flux;

        Assert.Equal(16, config.LatentChannels);
        Assert.InRange(config.ScalingFactor, 0.3611f - Tolerance, 0.3611f + Tolerance);
        Assert.NotNull(config.ShiftFactor);
        Assert.InRange(config.ShiftFactor!.Value, 0.1159f - Tolerance, 0.1159f + Tolerance);
        Assert.False(config.UsePostQuantConv);
    }
}
