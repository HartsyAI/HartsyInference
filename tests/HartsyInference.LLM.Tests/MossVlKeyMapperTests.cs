using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Gguf.KeyMappers;

namespace HartsyInference.LLM.Tests;

/// <summary>The MOSS-VL GGUF mapper maps its own dialect and leaves plain llama-family files to the llama mapper.</summary>
public sealed class MossVlKeyMapperTests
{
    [Fact]
    public void MapsTextAndCrossAttentionNamesToCheckpointNames()
    {
        MossVlKeyMapper mapper = new();

        Assert.Equal("model.language_model.embed_tokens.weight", mapper.MapKey("token_embd.weight"));
        Assert.Equal("model.language_model.layers.4.self_attn.q_proj.weight", mapper.MapKey("blk.4.attn_q.weight"));
        Assert.Equal("model.language_model.layers.2.cross_attn.k_norm.weight", mapper.MapKey("blk.2.cross_attn_k_norm.weight"));
        Assert.Equal("model.language_model.layers.2.cross_attn_attn_gate", mapper.MapKey("blk.2.cross_gate_attn.weight"));
        Assert.Equal("model.language_model.layers.46.cross_attn_mlp_gate", mapper.MapKey("blk.46.cross_gate_mlp.weight"));
        Assert.Equal("model.separator_token", mapper.MapKey("mm.separator.weight"));
        Assert.Null(mapper.MapKey("blk.0.unknown.weight"));
    }

    [Fact]
    public void ResolvesTheDeclaredArchitectureToTheMossVlMapper()
    {
        Assert.IsType<MossVlKeyMapper>(GgufKeyMapperRegistry.GetByArchitecture("moss_vl"));
    }

    [Fact]
    public void PlainLlamaFamilyFilesAreNotClaimedByTheMossVlMapper()
    {
        string[] llamaNames =
        [
            "token_embd.weight", "output_norm.weight", "output.weight",
            "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_norm.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_norm.weight",
        ];

        Assert.False(new MossVlKeyMapper().MatchesByKeys(llamaNames));
        Assert.IsNotType<MossVlKeyMapper>(GgufKeyMapperRegistry.DetectByKeys(llamaNames));
    }
}
