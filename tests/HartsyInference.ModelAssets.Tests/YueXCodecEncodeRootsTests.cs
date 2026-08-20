using HartsyInference.ModelAssets.CheckpointConverters;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Asset-free tests for the x-codec encode-branch key set and the stale-export probe that decides whether a
/// repacked <c>xcodec.safetensors</c> has to be rebuilt. Exports written before the encode roots were kept are
/// decode-only; serving one silently would leave reference-audio (ICL) prompting permanently broken on every existing
/// install, so the probe is what makes that self-healing.</summary>
public sealed class YueXCodecEncodeRootsTests
{
    // The four roots the decode path drops and the encode path needs (spec §8).
    private static readonly string[] _encodeOnlyKeys =
    [
        "semantic_model.encoder.layers.0.attention.q_proj.weight",
        "semantic_model.feature_extractor.conv_layers.0.conv.weight",
        "encoder_semantic.conv.conv.weight",
        "encoder_semantic.conv_blocks.0.res_units.1.conv2.weight",
        "encoder.block.0.weight_v",
        "encoder.block.6.bias",
        "fc_prior.weight",
        "fc_prior.bias",
    ];

    // Kept in BOTH directions — the decode path must stay byte-identical.
    private static readonly string[] _decodeKeys =
    [
        "quantizer.vq.layers.0._codebook.embed",
        "quantizer.vq.layers.11._codebook.embed",
        "fc_post2.weight",
        "fc_post2.bias",
        "decoder_2.model.0.weight_v",
        "decoder_2.model.5.block.3.weight_g",
    ];

    // Dropped in BOTH directions — training extras and the semantic-reconstruction head.
    private static readonly string[] _alwaysDroppedKeys =
    [
        "decoder_semantic.conv.conv.weight",
        "decoder_semantic_2.model.0.weight",
        "fc_post1.weight",
        "fc_post_a.weight",
        "fc_post_s.weight",
        "discriminator.discriminators.0.convs.0.weight",
    ];

    [Fact]
    public void MapXCodecKey_KeepsEncodeRoots_OnlyWhenForEncode()
    {
        foreach (string key in _encodeOnlyKeys)
        {
            Assert.Null(YueCheckpointConverter.MapXCodecKey(key));
            Assert.Equal(key, YueCheckpointConverter.MapXCodecKey(key, forEncode: true));
        }
    }

    [Fact]
    public void MapXCodecKey_DecodePathIsUnchangedByForEncode()
    {
        foreach (string key in _decodeKeys)
        {
            string? decode = YueCheckpointConverter.MapXCodecKey(key);
            Assert.NotNull(decode);
            Assert.Equal(decode, YueCheckpointConverter.MapXCodecKey(key, forEncode: true));
        }
        // decoder_2 -> decoder renaming survives the encode key set.
        Assert.Equal("decoder.model.0.weight_v",
            YueCheckpointConverter.MapXCodecKey("decoder_2.model.0.weight_v", forEncode: true));
    }

    [Fact]
    public void MapXCodecKey_StillDropsTrainingExtras_WhenForEncode()
    {
        foreach (string key in _alwaysDroppedKeys)
        {
            Assert.Null(YueCheckpointConverter.MapXCodecKey(key));
            Assert.Null(YueCheckpointConverter.MapXCodecKey(key, forEncode: true));
        }
    }

    [Fact]
    public void MapXCodecKey_EncoderDotDoesNotMatchEncoderSemantic()
    {
        // "encoder." and "encoder_semantic." are independent switches — a prefix test that conflated them would
        // silently drop the RepCodec branch.
        Assert.Null(YueCheckpointConverter.MapXCodecKey("encoder_semantic.conv.conv.weight"));
        Assert.Null(YueCheckpointConverter.MapXCodecKey("encoder.block.0.weight_v"));
    }

    [Fact]
    public void ExportHasEncodeRoots_FalseForADecodeOnlyExport()
    {
        Assert.False(YueCheckpointConverter.XCodecExportHasEncodeRoots(_decodeKeys));
    }

    [Fact]
    public void ExportHasEncodeRoots_TrueWhenFcPriorSurvives()
    {
        List<string> keys = [.. _decodeKeys, .. _encodeOnlyKeys];
        Assert.True(YueCheckpointConverter.XCodecExportHasEncodeRoots(keys));
    }

    [Fact]
    public void ExportHasEncodeRoots_SeesThroughTheWrapperPrefix()
    {
        // A repack preserves the SOURCE spelling, which may still carry `codec_model.`; a literal key comparison
        // would answer false and rewrite a perfectly good export on every load.
        Assert.True(YueCheckpointConverter.XCodecExportHasEncodeRoots(["codec_model.fc_prior.weight"]));
        Assert.True(YueCheckpointConverter.XCodecExportHasEncodeRoots(["generator.fc_prior.weight"]));
        Assert.False(YueCheckpointConverter.XCodecExportHasEncodeRoots(["codec_model.fc_post2.weight"]));
    }

    [Fact]
    public void ExportHasEncodeRoots_FcPriorBiasAloneIsNotEnough()
    {
        // The probe is specifically fc_prior.weight — the tensor XCodec.CanEncode keys off.
        Assert.False(YueCheckpointConverter.XCodecExportHasEncodeRoots(["fc_prior.bias"]));
        Assert.Equal("fc_prior.weight", YueCheckpointConverter.XCodecEncodeProbeKey);
    }

    [Fact]
    public void ExportHasEncodeRoots_EmptyIsFalse()
    {
        Assert.False(YueCheckpointConverter.XCodecExportHasEncodeRoots([]));
    }
}
