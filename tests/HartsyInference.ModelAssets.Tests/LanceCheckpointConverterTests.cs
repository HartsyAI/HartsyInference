using Xunit;
using HartsyInference.ModelAssets.CheckpointConverters;
using B = HartsyInference.ModelAssets.CheckpointConverters.LanceCheckpointConverter.LanceBucket;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Tests the pure key-routing of <see cref="LanceCheckpointConverter.RouteKey"/> — the backbone-prefix strip, MoT <c>_moe_gen</c> sibling preservation, ViT bucketing, and dropping of T2I-unused keys. No checkpoint files needed.</summary>
public class LanceCheckpointConverterTests
{
    [Theory]
    // Backbone: language_model.model. prefix stripped to bare keys LanceTransformer expects.
    [InlineData("language_model.model.embed_tokens.weight", B.Transformer, "embed_tokens.weight")]
    [InlineData("language_model.model.layers.0.self_attn.q_proj.weight", B.Transformer, "layers.0.self_attn.q_proj.weight")]
    // MoT gen-path sibling weights preserved verbatim (just prefix-stripped).
    [InlineData("language_model.model.layers.7.self_attn.q_proj_moe_gen.weight", B.Transformer, "layers.7.self_attn.q_proj_moe_gen.weight")]
    // QK-norm weights (present in the real checkpoint) preserved.
    // Top-level generation heads pass through unchanged (real names: vae2llm/llm2vae/latent_pos_embed).
    [InlineData("vae2llm.weight", B.Transformer, "vae2llm.weight")]
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
    [InlineData("language_model.lm_head.weight")]   // understanding-only head, unused by generation
    [InlineData("task_embed.weight")]
    public void RouteKey_DropsUnusedKeys(string key)
    {
        (LanceCheckpointConverter.LanceBucket b, string? m) = LanceCheckpointConverter.RouteKey(key);
        Assert.Equal(B.Drop, b);
        Assert.Null(m);
    }
}
