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

    [Theory]
    [InlineData("v.patch_embd.weight", "model.visual.patch_embed.proj.weight")]
    [InlineData("v.patch_embd.bias", "model.visual.patch_embed.proj.bias")]
    [InlineData("v.pos_emb.weight", "model.visual.pos_embed.weight")]
    [InlineData("v.blk.0.ln1.weight", "model.visual.blocks.0.norm1.weight")]
    [InlineData("v.blk.26.ln2.bias", "model.visual.blocks.26.norm2.bias")]
    [InlineData("v.blk.3.wqkv.weight", "model.visual.blocks.3.attn.qkv.weight")]
    [InlineData("v.blk.3.wqkv.bias", "model.visual.blocks.3.attn.qkv.bias")]
    [InlineData("v.blk.3.wo.weight", "model.visual.blocks.3.attn.proj.weight")]
    [InlineData("v.blk.12.ffn_up.weight", "model.visual.blocks.12.mlp.linear_fc1.weight")]
    [InlineData("v.blk.12.ffn_down.bias", "model.visual.blocks.12.mlp.linear_fc2.bias")]
    [InlineData("mm.ln0.weight", "model.visual.merger.norms.0.weight")]
    [InlineData("mm.ln3.bias", "model.visual.merger.norms.3.bias")]
    [InlineData("mm.fc1.weight", "model.visual.merger.linear_fc1.weight")]
    [InlineData("mm.fc2.bias", "model.visual.merger.linear_fc2.bias")]
    [InlineData("v.blk.0.unknown.weight", "v.blk.0.unknown.weight")]
    public void MapsVisionNamesToCheckpointNames(string gguf, string expected) =>
        Assert.Equal(expected, new MossVlKeyMapper().MapKey(gguf));

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
