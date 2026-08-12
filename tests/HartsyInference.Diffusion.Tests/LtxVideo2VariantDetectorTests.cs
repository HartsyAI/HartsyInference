using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Covers the config fields that actually differ between shipped LTX-2 generations. The JSON fragments
/// mirror the real <c>__metadata__["config"]</c> of the 2.3 and 2.5 checkpoints (verified 2026-08-12 by byte-range
/// reads of their safetensors headers).</summary>
public sealed class LtxVideo2VariantDetectorTests
{
    private const string Ltx25ConfigJson = """
        {"transformer":{"num_layers":48,"num_attention_heads":32,"attention_head_dim":128,"in_channels":128,
        "out_channels":128,"cross_attention_dim":4096,"caption_channels":3840,"audio_num_attention_heads":32,
        "audio_attention_head_dim":64,"audio_cross_attention_dim":2048,"audio_out_channels":128,
        "timestep_scale_multiplier":1000,"av_ca_timestep_scale_multiplier":1000.0,
        "positional_embedding_theta":10000.0,"positional_embedding_max_pos":[20,2048,2048],
        "audio_positional_embedding_max_pos":[20],"rope_type":"split","norm_eps":1e-06,
        "cross_attention_adaln":true,"ff_bias":false,"use_keyframes_abs_pos_embedding":true},
        "scheduler":{"_class_name":"RectifiedFlowScheduler"}}
        """;

    private const string Ltx23ConfigJson = """
        {"transformer":{"num_layers":48,"num_attention_heads":32,"attention_head_dim":128,"in_channels":128,
        "out_channels":128,"cross_attention_dim":4096,"caption_channels":3840,"audio_num_attention_heads":32,
        "audio_attention_head_dim":64,"audio_cross_attention_dim":2048,"audio_out_channels":128,
        "timestep_scale_multiplier":1000,"av_ca_timestep_scale_multiplier":1000.0,
        "positional_embedding_theta":10000.0,"positional_embedding_max_pos":[20,2048,2048],
        "audio_positional_embedding_max_pos":[20],"rope_type":"split","norm_eps":1e-06,
        "cross_attention_adaln":true},"scheduler":{"_class_name":"RectifiedFlowScheduler"}}
        """;

    private static Func<string, bool> Keys(params string[] present)
    {
        HashSet<string> set = new(present, StringComparer.Ordinal);
        return set.Contains;
    }

    [Fact]
    public void Ltx25MetadataAndKeysAgree()
    {
        LtxVideo2Config config = LtxVideo2VariantDetector.Detect(
            new Dictionary<string, string> { ["model_version"] = "2.5.0", ["config"] = Ltx25ConfigJson },
            Keys(LtxVideo2VariantDetector.KeyframesEmbeddingKey));

        Assert.False(config.FfBias);
        Assert.True(config.UseKeyframesAbsPosEmbedding);
        Assert.Equal(48, config.NumLayers);
        Assert.Equal(LtxVideo2Rope.RopeType.Split, config.RopeType);
        // Both generations ship 1000 despite the reference configurator defaulting this to 1.
        Assert.Equal(1000, config.CrossAttnTimestepScaleMultiplier);
    }

    [Fact]
    public void Ltx23MetadataYieldsV23Behavior()
    {
        LtxVideo2Config config = LtxVideo2VariantDetector.Detect(
            new Dictionary<string, string> { ["model_version"] = "2.3.0", ["config"] = Ltx23ConfigJson },
            Keys(LtxVideo2VariantDetector.VideoFfnBiasKey));

        Assert.True(config.FfBias);
        Assert.False(config.UseKeyframesAbsPosEmbedding);
        Assert.Equal(LtxVideo2Config.V23.NumLayers, config.NumLayers);
        Assert.Equal(LtxVideo2Config.V23.CrossAttentionDim, config.CrossAttentionDim);
    }

    [Fact]
    public void KeyProbesCarryStrippedRepack()
    {
        LtxVideo2Config config = LtxVideo2VariantDetector.Detect(
            metadata: null,
            Keys(LtxVideo2VariantDetector.KeyframesEmbeddingKey));

        Assert.True(config.UseKeyframesAbsPosEmbedding);
        Assert.False(config.FfBias);
    }

    [Fact]
    public void StrippedRepackOf23KeepsBias()
    {
        LtxVideo2Config config = LtxVideo2VariantDetector.Detect(
            metadata: null,
            Keys(LtxVideo2VariantDetector.VideoFfnBiasKey));

        Assert.True(config.FfBias);
        Assert.False(config.UseKeyframesAbsPosEmbedding);
    }

    [Fact]
    public void KeyPresenceOverridesMetadataForKeyframes()
    {
        // ComfyUI runs its key probe after merging the metadata config, so the weights win. A repack that edited
        // metadata but kept the parameter must still get the embedding applied.
        LtxVideo2Config config = LtxVideo2VariantDetector.Detect(
            new Dictionary<string, string> { ["config"] = Ltx23ConfigJson },
            Keys(LtxVideo2VariantDetector.KeyframesEmbeddingKey));

        Assert.True(config.UseKeyframesAbsPosEmbedding);
    }

    [Fact]
    public void MetadataClaimingKeyframesWithoutTheWeightIsRejected()
    {
        LtxVideo2Config config = LtxVideo2VariantDetector.Detect(
            new Dictionary<string, string> { ["config"] = Ltx25ConfigJson },
            Keys(/* no keyframes tensor */));

        Assert.False(config.UseKeyframesAbsPosEmbedding);
    }

    [Fact]
    public void RepackWithMetadataButNoLtxConfigStillProbesTheFfnBias()
    {
        // Community fp8/GGUF repacks are written by conversion scripts that keep a `format` entry and drop the LTX
        // config. "Metadata exists" must not be read as "the architecture was declared" — assuming 2.3's biased FFN
        // here would make the transformer reject the repack outright.
        LtxVideo2Config config = LtxVideo2VariantDetector.Detect(
            new Dictionary<string, string> { ["format"] = "pt" },
            Keys(LtxVideo2VariantDetector.KeyframesEmbeddingKey));

        Assert.False(config.FfBias);
        Assert.True(config.UseKeyframesAbsPosEmbedding);
    }

    [Fact]
    public void RepackWithMetadataButNoLtxConfigKeepsBiasWhenPresent()
    {
        LtxVideo2Config config = LtxVideo2VariantDetector.Detect(
            new Dictionary<string, string> { ["format"] = "pt" },
            Keys(LtxVideo2VariantDetector.VideoFfnBiasKey));

        Assert.True(config.FfBias);
        Assert.False(config.UseKeyframesAbsPosEmbedding);
    }

    [Fact]
    public void MalformedMetadataFallsBackInsteadOfThrowing()
    {
        LtxVideo2Config config = LtxVideo2VariantDetector.Detect(
            new Dictionary<string, string> { ["config"] = "{not json" },
            Keys(LtxVideo2VariantDetector.KeyframesEmbeddingKey));

        Assert.True(config.UseKeyframesAbsPosEmbedding);
        Assert.False(config.FfBias);
        Assert.Equal(LtxVideo2Config.V23.NumLayers, config.NumLayers);
    }

    [Fact]
    public void DistilledSigmasAreNotSharedBetweenConfigs()
    {
        float[] first = LtxVideo2Config.V25Distilled.FixedSigmas!;
        first[0] = -1f;

        Assert.Equal(1.0f, LtxVideo2Config.V25Distilled.FixedSigmas![0]);
    }

    [Fact]
    public void NoMetadataAndNoKeysIsV23()
    {
        LtxVideo2Config config = LtxVideo2VariantDetector.Detect(null, Keys());

        Assert.False(config.UseKeyframesAbsPosEmbedding);
        Assert.Equal(LtxVideo2Config.V23.NumLayers, config.NumLayers);
        Assert.Equal(LtxVideo2Config.V23.RopeType, config.RopeType);
    }

    [Fact]
    public void DimensionOverridesAreHonored()
    {
        LtxVideo2Config config = LtxVideo2VariantDetector.Detect(
            new Dictionary<string, string>
            {
                ["config"] = """{"transformer":{"num_layers":32,"num_attention_heads":24,"rope_type":"interleaved"}}""",
            },
            Keys());

        Assert.Equal(32, config.NumLayers);
        Assert.Equal(24, config.NumHeads);
        Assert.Equal(LtxVideo2Rope.RopeType.Interleaved, config.RopeType);
    }

    [Fact]
    public void DistilledIsNotInferableFromTheCheckpoint()
    {
        // The dev and distilled 2.5 transformers share model_version, config and tensor keys, so nothing here may
        // set a fixed schedule — that choice belongs to the caller.
        LtxVideo2Config config = LtxVideo2VariantDetector.Detect(
            new Dictionary<string, string> { ["model_version"] = "2.5.0", ["config"] = Ltx25ConfigJson },
            Keys(LtxVideo2VariantDetector.KeyframesEmbeddingKey));

        Assert.Null(config.FixedSigmas);
    }
}
