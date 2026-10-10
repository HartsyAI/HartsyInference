using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tests for the VAE encoder: construction with the reference preset, and the LDM→diffusers encoder key conversion in <see cref="CheckpointConvertUtils.ConvertVaeKey"/>.</summary>
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

    // ── ConvertVaeKey: Encoder Path ─────────────────────────────────────

    [Fact]
    public void ConvertVaeKey_PassThroughKeys_AreUnchanged()
    {
        Assert.Equal("encoder.conv_in.weight", CheckpointConvertUtils.ConvertVaeKey("encoder.conv_in.weight"));
        Assert.Equal("quant_conv.weight", CheckpointConvertUtils.ConvertVaeKey("quant_conv.weight"));
        Assert.Equal("post_quant_conv.bias", CheckpointConvertUtils.ConvertVaeKey("post_quant_conv.bias"));
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
}
