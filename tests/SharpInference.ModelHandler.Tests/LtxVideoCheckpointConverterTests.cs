using Xunit;
using SharpInference.ModelHandler.CheckpointConverters;
using B = SharpInference.ModelHandler.CheckpointConverters.LtxVideoCheckpointConverter.LtxBucket;

namespace SharpInference.ModelHandler.Tests;

/// <summary>Tests the pure key routing of <see cref="LtxVideoCheckpointConverter.RouteKey"/> — single-file prefix
/// bucketing (<c>model.diffusion_model.</c> / <c>vae.</c>), the transformer renames, the ordered VAE flat-list
/// regrouping (original <c>up_blocks.0..9</c> → diffusers <c>mid_block</c>/<c>up_blocks.{i}.{conv_in,upsamplers,resnets}</c>),
/// and per-channel-stats handling. No checkpoint files needed.</summary>
public class LtxVideoCheckpointConverterTests
{
    [Theory]
    // Transformer: prefix strip + original→diffusers renames.
    [InlineData("model.diffusion_model.patchify_proj.weight", B.Transformer, "proj_in.weight")]
    [InlineData("model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.weight", B.Transformer, "time_embed.emb.timestep_embedder.linear_1.weight")]
    [InlineData("model.diffusion_model.adaln_single.linear.weight", B.Transformer, "time_embed.linear.weight")]
    [InlineData("model.diffusion_model.transformer_blocks.0.attn1.q_norm.weight", B.Transformer, "transformer_blocks.0.attn1.norm_q.weight")]
    [InlineData("model.diffusion_model.transformer_blocks.27.attn2.k_norm.weight", B.Transformer, "transformer_blocks.27.attn2.norm_k.weight")]
    [InlineData("model.diffusion_model.transformer_blocks.0.attn1.to_q.weight", B.Transformer, "transformer_blocks.0.attn1.to_q.weight")]
    [InlineData("model.diffusion_model.caption_projection.linear_1.weight", B.Transformer, "caption_projection.linear_1.weight")]
    [InlineData("model.diffusion_model.scale_shift_table", B.Transformer, "scale_shift_table")]
    [InlineData("model.diffusion_model.proj_out.weight", B.Transformer, "proj_out.weight")]
    // Diffusers folder shards: bare transformer keys pass through.
    [InlineData("transformer_blocks.0.ff.net.0.proj.weight", B.Transformer, "transformer_blocks.0.ff.net.0.proj.weight")]
    public void RouteKey_Transformer(string key, B bucket, string mapped)
    {
        (B b, string? m) = LtxVideoCheckpointConverter.RouteKey(key, vaeOriginalNaming: true);
        Assert.Equal(bucket, b);
        Assert.Equal(mapped, m);
    }

    [Theory]
    // VAE flat decoder list → diffusers grouping (LtxVideoVaeDecoder's key contract).
    [InlineData("vae.decoder.conv_in.conv.weight", B.Vae, "decoder.conv_in.conv.weight")]
    [InlineData("vae.decoder.up_blocks.0.res_blocks.0.conv1.conv.weight", B.Vae, "decoder.mid_block.resnets.0.conv1.conv.weight")]
    [InlineData("vae.decoder.up_blocks.1.res_blocks.2.norm2.weight", B.Vae, "decoder.up_blocks.0.resnets.2.norm2.weight")]
    [InlineData("vae.decoder.up_blocks.2.conv.conv.weight", B.Vae, "decoder.up_blocks.1.upsamplers.0.conv.conv.weight")]
    [InlineData("vae.decoder.up_blocks.3.res_blocks.0.conv_shortcut.weight", B.Vae, "decoder.up_blocks.1.resnets.0.conv_shortcut.conv.weight")]
    [InlineData("vae.decoder.up_blocks.4.conv1.conv.weight", B.Vae, "decoder.up_blocks.2.conv_in.conv1.conv.weight")]
    [InlineData("vae.decoder.up_blocks.5.conv.conv.weight", B.Vae, "decoder.up_blocks.2.upsamplers.0.conv.conv.weight")]
    [InlineData("vae.decoder.up_blocks.6.res_blocks.1.conv2.conv.bias", B.Vae, "decoder.up_blocks.2.resnets.1.conv2.conv.bias")]
    [InlineData("vae.decoder.up_blocks.7.conv1.conv.weight", B.Vae, "decoder.up_blocks.3.conv_in.conv1.conv.weight")]
    [InlineData("vae.decoder.up_blocks.8.conv.conv.weight", B.Vae, "decoder.up_blocks.3.upsamplers.0.conv.conv.weight")]
    [InlineData("vae.decoder.up_blocks.9.res_blocks.3.conv1.conv.weight", B.Vae, "decoder.up_blocks.3.resnets.3.conv1.conv.weight")]
    [InlineData("vae.decoder.conv_out.conv.weight", B.Vae, "decoder.conv_out.conv.weight")]
    // Encoder keys carried for a future encoder.
    [InlineData("vae.encoder.down_blocks.1.conv.conv.weight", B.Vae, "encoder.down_blocks.0.downsamplers.0.conv.conv.weight")]
    [InlineData("vae.encoder.down_blocks.9.res_blocks.0.conv1.conv.weight", B.Vae, "encoder.mid_block.resnets.0.conv1.conv.weight")]
    // Per-channel statistics: means/stds kept under diffusers names, the rest dropped.
    [InlineData("vae.per_channel_statistics.mean-of-means", B.Vae, "latents_mean")]
    [InlineData("vae.per_channel_statistics.std-of-means", B.Vae, "latents_std")]
    public void RouteKey_Vae(string key, B bucket, string mapped)
    {
        (B b, string? m) = LtxVideoCheckpointConverter.RouteKey(key, vaeOriginalNaming: true);
        Assert.Equal(bucket, b);
        Assert.Equal(mapped, m);
    }

    [Theory]
    [InlineData("vae.per_channel_statistics.channel")]
    [InlineData("vae.per_channel_statistics.mean-of-stds")]
    [InlineData("scaled_fp8")]
    public void RouteKey_DropsMetadata(string key)
    {
        (B b, string? m) = LtxVideoCheckpointConverter.RouteKey(key, vaeOriginalNaming: true);
        Assert.Equal(B.Drop, b);
        Assert.Null(m);
    }

    [Theory]
    // Already-diffusers VAE keys must NOT be regrouped (the rename table would corrupt them).
    [InlineData("decoder.up_blocks.0.resnets.0.conv1.conv.weight")]
    [InlineData("decoder.mid_block.resnets.0.norm1.weight")]
    [InlineData("latents_mean")]
    public void RouteKey_DiffusersVae_PassesThrough(string key)
    {
        (B b, string? m) = LtxVideoCheckpointConverter.RouteKey(key, vaeOriginalNaming: false);
        Assert.Equal(B.Vae, b);
        Assert.Equal(key, m);
    }

    [Fact]
    public void IsOriginalVaeNaming_DetectsBothFormats()
    {
        Assert.True(LtxVideoCheckpointConverter.IsOriginalVaeNaming(["vae.decoder.up_blocks.0.res_blocks.0.conv1.conv.weight"]));
        Assert.True(LtxVideoCheckpointConverter.IsOriginalVaeNaming(["vae.per_channel_statistics.mean-of-means"]));
        Assert.False(LtxVideoCheckpointConverter.IsOriginalVaeNaming(["decoder.up_blocks.0.resnets.0.conv1.conv.weight"]));
    }
}
