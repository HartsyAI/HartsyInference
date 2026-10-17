using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Gguf.KeyMappers;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>llama.cpp's <c>dory</c> tensor dialect (ggml-org/llama.cpp#30306) onto the names the looped transformer loads.</summary>
public sealed class DoryKeyMapperTests
{
    [Fact]
    public void Registry_ResolvesDoryByName()
    {
        IGgufKeyMapper? mapper = GgufKeyMapperRegistry.GetByArchitecture("dory");
        Assert.NotNull(mapper);
        Assert.IsType<DoryKeyMapper>(mapper);
    }

    [Theory]
    [InlineData("token_embd.weight", "model.embed_tokens.weight")]
    [InlineData("output.weight", "lm_head.weight")]
    [InlineData("output_scale.weight", "lm_head.logit_scale.weight")]
    [InlineData("blk.3.residual_alpha.weight", "model.layers.3.residual_alpha.weight")]
    [InlineData("blk.3.attn_q.weight", "model.layers.3.self_attn.q_proj.weight")]
    [InlineData("blk.3.attn_k.weight", "model.layers.3.self_attn.k_proj.weight")]
    [InlineData("blk.3.attn_v.weight", "model.layers.3.self_attn.v_proj.weight")]
    [InlineData("blk.3.attn_output.weight", "model.layers.3.self_attn.o_proj.weight")]
    [InlineData("blk.3.attn_gate.weight", "model.layers.3.self_attn.gate_proj.weight")]
    [InlineData("blk.3.attn_qk_scale.weight", "model.layers.3.self_attn.qk_scale.weight")]
    [InlineData("blk.3.attn_gate_scale.weight", "model.layers.3.self_attn.gate_scale.weight")]
    [InlineData("blk.4.ffn_gate.weight", "model.layers.4.mlp.gate_proj.weight")]
    [InlineData("blk.4.ffn_up.weight", "model.layers.4.mlp.up_proj.weight")]
    [InlineData("blk.4.ffn_down.weight", "model.layers.4.mlp.down_proj.weight")]
    [InlineData("blk.4.ffn_gate_scale.weight", "model.layers.4.mlp.gate_scale.weight")]
    [InlineData("blk.4.ffn_up_scale.weight", "model.layers.4.mlp.up_scale.weight")]
    public void MapKey_TranslatesEveryDoryTensor(string gguf, string expected)
    {
        Assert.Equal(expected, new DoryKeyMapper().MapKey(gguf));
    }

    [Fact]
    public void MatchesByKeys_NeedsTheResidualAlpha()
    {
        DoryKeyMapper mapper = new();
        Assert.True(mapper.MatchesByKeys(["token_embd.weight", "blk.0.residual_alpha.weight"]));
        Assert.False(mapper.MatchesByKeys(["token_embd.weight", "blk.0.attn_q.weight"]));
        // The heuristic fallback must pick Dory over the llama family for a file whose metadata names no architecture.
        Assert.IsType<DoryKeyMapper>(GgufKeyMapperRegistry.DetectByKeys(["token_embd.weight", "blk.0.attn_q.weight", "blk.0.residual_alpha.weight"]));
    }
}
