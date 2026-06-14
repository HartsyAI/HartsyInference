using Xunit;
using HartsyInference.ModelHandler.CheckpointConverters;
using B = HartsyInference.ModelHandler.CheckpointConverters.LanceCheckpointConverter.LanceBucket;

namespace HartsyInference.ModelHandler.Tests;

/// <summary>Tests the pure key-routing of <see cref="LanceCheckpointConverter.RouteKey"/> — the backbone-prefix strip, MoT <c>_moe_gen</c> sibling preservation, ViT bucketing, and dropping of T2I-unused keys. No checkpoint files needed.</summary>
public class LanceCheckpointConverterTests
{
    [Theory]
    // Backbone: language_model.model. prefix stripped to bare keys LanceTransformer expects.
    [InlineData("language_model.model.embed_tokens.weight", B.Transformer, "embed_tokens.weight")]
    [InlineData("language_model.model.layers.0.self_attn.q_proj.weight", B.Transformer, "layers.0.self_attn.q_proj.weight")]
    [InlineData("language_model.model.norm.weight", B.Transformer, "norm.weight")]
    [InlineData("language_model.model.norm_moe_gen.weight", B.Transformer, "norm_moe_gen.weight")]
    // MoT gen-path sibling weights preserved verbatim (just prefix-stripped).
    [InlineData("language_model.model.layers.7.self_attn.q_proj_moe_gen.weight", B.Transformer, "layers.7.self_attn.q_proj_moe_gen.weight")]
    [InlineData("language_model.model.layers.7.mlp_moe_gen.gate_proj.weight", B.Transformer, "layers.7.mlp_moe_gen.gate_proj.weight")]
    [InlineData("language_model.model.layers.3.input_layernorm_moe_gen.weight", B.Transformer, "layers.3.input_layernorm_moe_gen.weight")]
    // Top-level generation heads pass through unchanged.
    [InlineData("vae_in.weight", B.Transformer, "vae_in.weight")]
    [InlineData("vae_out.bias", B.Transformer, "vae_out.bias")]
    [InlineData("time_embedder.mlp.0.weight", B.Transformer, "time_embedder.mlp.0.weight")]
    // ViT / connector → editing bucket.
    [InlineData("vit.blocks.0.attn.qkv.weight", B.Vit, "vit.blocks.0.attn.qkv.weight")]
    [InlineData("connector.fc1.weight", B.Vit, "connector.fc1.weight")]
    public void RouteKey_MapsExpectedBuckets(string key, B bucket, string mapped)
    {
        (LanceCheckpointConverter.LanceBucket b, string? m) = LanceCheckpointConverter.RouteKey(key);
        Assert.Equal(bucket, b);
        Assert.Equal(mapped, m);
    }

    [Theory]
    [InlineData("language_model.lm_head.weight")]   // tied head, unused
    [InlineData("pos_embed_3d.pos_embed")]          // recomputed
    [InlineData("task_embed.weight")]
    [InlineData("modality_embed.weight")]
    public void RouteKey_DropsUnusedKeys(string key)
    {
        (LanceCheckpointConverter.LanceBucket b, string? m) = LanceCheckpointConverter.RouteKey(key);
        Assert.Equal(B.Drop, b);
        Assert.Null(m);
    }
}
