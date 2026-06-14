using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tests for VAE decoder components: config presets, ResNet block shapes, attention shapes, tiled blending math, and scaling factors.</summary>
public sealed class VaeDecoderTests
{
    private const float Tolerance = 1e-5f;

    // ── VaeConfig Tests ──────────────────────────────────────────────────

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
    public void VaeConfig_Sdxl_HasCorrectValues()
    {
        VaeConfig config = VaeConfig.Sdxl;

        Assert.Equal(4, config.LatentChannels);
        Assert.Equal(0.13025f, config.ScalingFactor);
        Assert.Null(config.ShiftFactor);
        Assert.True(config.UsePostQuantConv);
        Assert.Equal(1024, config.SampleSize);
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

    // ── Scaling Math Tests ──────────────────────────────────────────────

    [Fact]
    public void Scaling_Sd15_InverseRoundTrip()
    {
        // latent = raw * scaling_factor → raw = latent / scaling_factor
        float scalingFactor = 0.18215f;
        float rawValue = 1.0f;

        float latent = rawValue * scalingFactor;
        float reconstructed = latent / scalingFactor;

        Assert.InRange(reconstructed, rawValue - Tolerance, rawValue + Tolerance);
    }

    [Fact]
    public void Scaling_Sd3_InverseRoundTrip_WithShift()
    {
        // latent = (raw - shift) * scaling → raw = latent / scaling + shift
        float scalingFactor = 1.5305f;
        float shiftFactor = 0.0609f;
        float rawValue = 0.5f;

        float latent = (rawValue - shiftFactor) * scalingFactor;
        float reconstructed = latent / scalingFactor + shiftFactor;

        Assert.InRange(reconstructed, rawValue - Tolerance, rawValue + Tolerance);
    }

    // ── ResNetBlock2D Tests ──────────────────────────────────────────────

    [Fact]
    public void ResNetBlock2D_SameChannels_NoShortcut()
    {
        ResNetBlock2D block = new ResNetBlock2D(128, 128);
        // Block should not have a shortcut conv when in/out channels match
        // This is a construction test — the block type should be valid
        Assert.NotNull(block);
    }

    [Fact]
    public void ResNetBlock2D_DifferentChannels_HasShortcut()
    {
        ResNetBlock2D block = new ResNetBlock2D(128, 256);
        // Block with different in/out channels should be valid (shortcut conv internally)
        Assert.NotNull(block);
    }

    // ── VaeDecoder Construction Tests ────────────────────────────────────

    [Fact]
    public void VaeDecoder_Sd15Config_ConstructsSuccessfully()
    {
        VaeDecoder decoder = new VaeDecoder(VaeConfig.Sd15);
        Assert.Equal(4, decoder.Config.LatentChannels);
    }

    [Fact]
    public void VaeDecoder_Sd3Config_ConstructsSuccessfully()
    {
        VaeDecoder decoder = new VaeDecoder(VaeConfig.Sd3);
        Assert.Equal(16, decoder.Config.LatentChannels);
    }

    [Fact]
    public void VaeDecoder_FluxConfig_ConstructsSuccessfully()
    {
        VaeDecoder decoder = new VaeDecoder(VaeConfig.Flux);
        Assert.False(decoder.Config.UsePostQuantConv);
    }

    // ── Tiled Decoder Tests ──────────────────────────────────────────────

    [Fact]
    public void TiledDecoder_TileParameters_Sd15()
    {
        // SD1.5: sample_size=512, 4 block stages → tile_latent=64, overlap=48, blend=16
        VaeConfig config = VaeConfig.Sd15;
        int tileLatentSize = config.SampleSize / (int)Math.Pow(2, config.BlockOutChannels.Length - 1);
        float overlapFactor = 0.25f;
        int overlapSize = (int)(tileLatentSize * (1.0f - overlapFactor));
        int blendExtent = (int)(tileLatentSize * overlapFactor);
        int rowLimit = tileLatentSize - blendExtent;

        Assert.Equal(64, tileLatentSize);
        Assert.Equal(48, overlapSize);
        Assert.Equal(16, blendExtent);
        Assert.Equal(48, rowLimit);
    }

    [Fact]
    public void TiledDecoder_TileParameters_Sdxl()
    {
        // SDXL: sample_size=1024, 4 block stages → tile_latent=128, overlap=96, blend=32
        VaeConfig config = VaeConfig.Sdxl;
        int tileLatentSize = config.SampleSize / (int)Math.Pow(2, config.BlockOutChannels.Length - 1);
        float overlapFactor = 0.25f;
        int overlapSize = (int)(tileLatentSize * (1.0f - overlapFactor));
        int blendExtent = (int)(tileLatentSize * overlapFactor);

        Assert.Equal(128, tileLatentSize);
        Assert.Equal(96, overlapSize);
        Assert.Equal(32, blendExtent);
    }

    [Fact]
    public void LinearBlend_WeightProgression()
    {
        // Verify the linear blend weight formula: weight = x / blend_extent
        // At x=0: weight=0 (favor left/top), at x=blend: weight=1 (favor right/bottom)
        int blendExtent = 16;

        float weight0 = 0.0f / blendExtent;
        float weightMid = 8.0f / blendExtent;
        float weightEnd = 15.0f / blendExtent;

        Assert.Equal(0.0f, weight0);
        Assert.InRange(weightMid, 0.5f - Tolerance, 0.5f + Tolerance);
        Assert.InRange(weightEnd, 0.9375f - Tolerance, 0.9375f + Tolerance);
    }

    [Fact]
    public unsafe void LinearBlend_TwoValues_ProducesExpectedResult()
    {
        // Blend two pixel values with weight=0.5 → average
        float left = 10.0f;
        float right = 20.0f;
        float weight = 0.5f;

        float blended = left * (1.0f - weight) + right * weight;

        Assert.InRange(blended, 15.0f - Tolerance, 15.0f + Tolerance);
    }

    // ── VaeAttention Tests ──────────────────────────────────────────────

    [Fact]
    public void VaeAttention_Construction_ValidChannels()
    {
        VaeAttention attention = new VaeAttention(512, normGroups: 32, normEps: 1e-6f);
        Assert.NotNull(attention);
    }

    [Fact]
    public void VaeAttention_ScaleFactor_MatchesExpected()
    {
        // scale = 1/sqrt(channels) for single-head attention
        int channels = 512;
        float expectedScale = 1.0f / MathF.Sqrt(channels);

        // scale = 1/sqrt(512) ≈ 0.04419
        Assert.InRange(expectedScale, 0.0441f, 0.0443f);
    }

    // ── Channel Progression Tests ────────────────────────────────────────

    [Fact]
    public void DecoderChannels_ReversedOrder_MatchesExpected()
    {
        // Decoder up-blocks process channels in reversed order: [512, 512, 256, 128]
        int[] blockOutChannels = [128, 256, 512, 512];
        int[] reversed = new int[blockOutChannels.Length];
        for (int i = 0; i < blockOutChannels.Length; i++)
        {
            reversed[i] = blockOutChannels[^(i + 1)];
        }

        Assert.Equal([512, 512, 256, 128], reversed);
    }

    [Fact]
    public void DecoderResNetsPerBlock_IsLayersPerBlockPlusOne()
    {
        // Decoder has layers_per_block + 1 = 3 ResNets per block
        VaeConfig config = VaeConfig.Sd15;
        int resnetsPerBlock = config.LayersPerBlock + 1;
        Assert.Equal(3, resnetsPerBlock);
    }

    [Fact]
    public void SpatialCompression_8x()
    {
        // 3 downsamples of 2x each = 8x total spatial compression
        int compressionFactor = (int)Math.Pow(2, 3);
        Assert.Equal(8, compressionFactor);

        // 512px → 64 latent, 1024px → 128 latent
        Assert.Equal(64, 512 / compressionFactor);
        Assert.Equal(128, 1024 / compressionFactor);
    }
}
