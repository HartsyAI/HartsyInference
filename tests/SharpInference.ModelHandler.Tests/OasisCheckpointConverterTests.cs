using Xunit;
using SharpInference.ModelHandler.CheckpointConverters;

namespace SharpInference.ModelHandler.Tests;

/// <summary>Key-mapping tests for the Oasis loader: wrapper-prefix strip and recomputed RoPE buffer drop.</summary>
public class OasisCheckpointConverterTests
{
    [Theory]
    [InlineData("blocks.0.s_attn.to_qkv.weight", "blocks.0.s_attn.to_qkv.weight")]
    [InlineData("model.blocks.0.s_attn.to_qkv.weight", "blocks.0.s_attn.to_qkv.weight")]
    [InlineData("module.external_cond.weight", "external_cond.weight")]
    [InlineData("encoder.3.attn.qkv.bias", "encoder.3.attn.qkv.bias")]
    [InlineData("final_layer.adaLN_modulation.1.weight", "final_layer.adaLN_modulation.1.weight")]
    public void MapKey_StripsWrapperPrefixes(string key, string expected)
    {
        Assert.Equal(expected, OasisCheckpointConverter.MapKey(key));
    }

    [Theory]
    [InlineData("encoder.0.attn.rotary_freqs")]
    [InlineData("model.decoder.2.attn.rotary_freqs")]
    public void MapKey_DropsRecomputedRopeBuffers(string key)
    {
        Assert.Null(OasisCheckpointConverter.MapKey(key));
    }
}
