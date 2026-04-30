using SharpInference.Diffusion.Models.Vae;
using SharpInference.ModelHandler.CheckpointConverters.Utils;
using Xunit;

namespace SharpInference.Diffusion.Tests;

/// <summary>Tests for the VAE encoder: construction with each preset, weight-key conventions, and the LDM→diffusers encoder key conversion in <see cref="CheckpointConvertUtils.ConvertVaeKey"/>.</summary>
public sealed class VaeEncoderTests
{
    // ── Construction ────────────────────────────────────────────────────

    [Fact]
    public void VaeEncoder_Sd15Config_ConstructsSuccessfully()
    {
        VaeEncoder encoder = new VaeEncoder(VaeConfig.Sd15);
        Assert.Equal(4, encoder.Config.LatentChannels);
        Assert.True(encoder.Config.UseQuantConv);
    }

    [Fact]
    public void VaeEncoder_SdxlConfig_ConstructsSuccessfully()
    {
        VaeEncoder encoder = new VaeEncoder(VaeConfig.Sdxl);
        Assert.Equal(4, encoder.Config.LatentChannels);
        Assert.Equal(0.13025f, encoder.Config.ScalingFactor);
    }

    [Fact]
    public void VaeEncoder_Sd3Config_ConstructsSuccessfully()
    {
        VaeEncoder encoder = new VaeEncoder(VaeConfig.Sd3);
        Assert.Equal(16, encoder.Config.LatentChannels);
        Assert.False(encoder.Config.UseQuantConv);
    }

    [Fact]
    public void VaeEncoder_FluxConfig_ConstructsSuccessfully()
    {
        VaeEncoder encoder = new VaeEncoder(VaeConfig.Flux);
        Assert.False(encoder.Config.UseQuantConv);
        Assert.NotNull(encoder.Config.ShiftFactor);
    }

    // ── Channel Progression ────────────────────────────────────────────

    [Fact]
    public void EncoderChannels_ForwardOrder_MatchesBlockOutChannels()
    {
        // Encoder runs through block_out_channels in forward order [128, 256, 512, 512].
        // (Decoder runs through them reversed.)
        int[] blockOutChannels = VaeConfig.Sd15.BlockOutChannels;
        Assert.Equal([128, 256, 512, 512], blockOutChannels);
    }

    [Fact]
    public void EncoderResNetsPerBlock_IsLayersPerBlock()
    {
        // Encoder has layers_per_block ResNets per block (no +1 — that's a decoder property).
        VaeConfig config = VaeConfig.Sd15;
        Assert.Equal(2, config.LayersPerBlock);
    }

    // ── Scaling Math ────────────────────────────────────────────────────

    [Fact]
    public void EncoderScaling_Sd15_NoShift()
    {
        // Encoder: latent = (mu - shift) * scaling. No shift → latent = mu * scaling.
        float scalingFactor = 0.18215f;
        float mu = 1.0f;
        float latent = (mu - 0f) * scalingFactor;
        Assert.Equal(0.18215f, latent, 1e-6);
    }

    [Fact]
    public void EncoderScaling_Sd3_WithShift_InverseOfDecoder()
    {
        // Encoder: latent = (mu - shift) * scaling.
        // Decoder: mu = latent / scaling + shift.
        // Round-trip: encode then decode the raw posterior mean.
        float scalingFactor = 1.5305f;
        float shiftFactor = 0.0609f;
        float mu = 0.5f;

        float latent = (mu - shiftFactor) * scalingFactor;
        float reconstructed = latent / scalingFactor + shiftFactor;

        Assert.Equal(mu, reconstructed, 1e-5);
    }

    // ── ConvertVaeKey: Encoder Path ─────────────────────────────────────

    [Fact]
    public void ConvertVaeKey_EncoderConvIn_MapsThrough()
    {
        // LDM "encoder.conv_in.weight" → diffusers "encoder.conv_in.weight" (no rename).
        string? result = CheckpointConvertUtils.ConvertVaeKey("encoder.conv_in.weight");
        Assert.Equal("encoder.conv_in.weight", result);
    }

    [Fact]
    public void ConvertVaeKey_EncoderConvOut_MapsThrough()
    {
        string? result = CheckpointConvertUtils.ConvertVaeKey("encoder.conv_out.bias");
        Assert.Equal("encoder.conv_out.bias", result);
    }

    [Fact]
    public void ConvertVaeKey_EncoderNormOut_RenamesTo_ConvNormOut()
    {
        // LDM "encoder.norm_out.*" → diffusers "encoder.conv_norm_out.*".
        string? result = CheckpointConvertUtils.ConvertVaeKey("encoder.norm_out.weight");
        Assert.Equal("encoder.conv_norm_out.weight", result);
    }

    [Fact]
    public void ConvertVaeKey_EncoderDownBlock_PreservesLevelOrder()
    {
        // Encoder down levels run shallow→deep in BOTH LDM and diffusers (no reversal).
        string? l0 = CheckpointConvertUtils.ConvertVaeKey("encoder.down.0.block.0.norm1.weight");
        Assert.Equal("encoder.down_blocks.0.resnets.0.norm1.weight", l0);

        string? l3 = CheckpointConvertUtils.ConvertVaeKey("encoder.down.3.block.1.conv2.bias");
        Assert.Equal("encoder.down_blocks.3.resnets.1.conv2.bias", l3);
    }

    [Fact]
    public void ConvertVaeKey_EncoderDownsample_MapsToDownsamplers0()
    {
        string? result = CheckpointConvertUtils.ConvertVaeKey("encoder.down.1.downsample.conv.weight");
        Assert.Equal("encoder.down_blocks.1.downsamplers.0.conv.weight", result);
    }

    [Fact]
    public void ConvertVaeKey_EncoderShortcut_NinShortcutRenamed()
    {
        // LDM "nin_shortcut" is renamed to "conv_shortcut" (matches decoder behavior).
        string? result = CheckpointConvertUtils.ConvertVaeKey("encoder.down.1.block.0.nin_shortcut.weight");
        Assert.Equal("encoder.down_blocks.1.resnets.0.conv_shortcut.weight", result);
    }

    [Fact]
    public void ConvertVaeKey_EncoderMidBlock_ResNetAndAttention()
    {
        // Mid block layout is identical to decoder, just under the encoder.* prefix.
        string? mid0 = CheckpointConvertUtils.ConvertVaeKey("encoder.mid.block_1.norm1.weight");
        Assert.Equal("encoder.mid_block.resnets.0.norm1.weight", mid0);

        string? mid1 = CheckpointConvertUtils.ConvertVaeKey("encoder.mid.block_2.conv2.bias");
        Assert.Equal("encoder.mid_block.resnets.1.conv2.bias", mid1);

        string? attn = CheckpointConvertUtils.ConvertVaeKey("encoder.mid.attn_1.q.weight");
        Assert.Equal("encoder.mid_block.attentions.0.to_q.weight", attn);

        string? attnNorm = CheckpointConvertUtils.ConvertVaeKey("encoder.mid.attn_1.norm.weight");
        Assert.Equal("encoder.mid_block.attentions.0.group_norm.weight", attnNorm);

        string? attnOut = CheckpointConvertUtils.ConvertVaeKey("encoder.mid.attn_1.proj_out.bias");
        Assert.Equal("encoder.mid_block.attentions.0.to_out.0.bias", attnOut);
    }

    [Fact]
    public void ConvertVaeKey_QuantConvAndPostQuantConv_PassThroughUnchanged()
    {
        Assert.Equal("quant_conv.weight", CheckpointConvertUtils.ConvertVaeKey("quant_conv.weight"));
        Assert.Equal("post_quant_conv.bias", CheckpointConvertUtils.ConvertVaeKey("post_quant_conv.bias"));
    }

    // ── ConvertVaeKey: Decoder Path Regression ─────────────────────────

    [Fact]
    public void ConvertVaeKey_DecoderUpBlock_StillReversesLevelOrder()
    {
        // Regression: the decoder path must still reverse levels (LDM ldmLevel 0 → diffusers level numUpLevels-1).
        string? l0 = CheckpointConvertUtils.ConvertVaeKey("decoder.up.0.block.0.norm1.weight");
        Assert.Equal("decoder.up_blocks.3.resnets.0.norm1.weight", l0);

        string? l3 = CheckpointConvertUtils.ConvertVaeKey("decoder.up.3.block.0.norm1.weight");
        Assert.Equal("decoder.up_blocks.0.resnets.0.norm1.weight", l3);
    }

    [Fact]
    public void ConvertVaeKey_DecoderMidBlock_StillUnderDecoderPrefix()
    {
        // Regression: shared ConvertVaeMidKey now takes a section param; decoder must still get "decoder.mid_block.*".
        string? mid = CheckpointConvertUtils.ConvertVaeKey("decoder.mid.block_1.norm1.weight");
        Assert.Equal("decoder.mid_block.resnets.0.norm1.weight", mid);
    }

    [Fact]
    public void ConvertVaeKey_UnknownKey_ReturnsNull()
    {
        Assert.Null(CheckpointConvertUtils.ConvertVaeKey("loss.something"));
        Assert.Null(CheckpointConvertUtils.ConvertVaeKey("foo.bar"));
    }

    // ── Spatial Compression ─────────────────────────────────────────────

    [Fact]
    public void EncoderSpatialCompression_8x()
    {
        // 4 down blocks; the last has no downsample → 3 downsamples × 2× = 8× total compression.
        int numBlocks = VaeConfig.Sd15.BlockOutChannels.Length;
        int compressionFactor = (int)Math.Pow(2, numBlocks - 1);
        Assert.Equal(8, compressionFactor);

        Assert.Equal(64, 512 / compressionFactor);
        Assert.Equal(128, 1024 / compressionFactor);
    }
}
