using Xunit;
using HartsyInference.ModelHandler.CheckpointConverters;
using B = HartsyInference.ModelHandler.CheckpointConverters.LtxVideo2CheckpointConverter.Ltx2Bucket;

namespace HartsyInference.ModelHandler.Tests;

/// <summary>Tests the pure key routing of <see cref="LtxVideo2CheckpointConverter.RouteKey"/>: per-component prefix
/// bucketing for the LTX-2.3 single file — DiT (<c>model.diffusion_model.*</c>, prefix stripped), the text
/// connectors (kept prefixed), the video/audio VAEs (<c>vae.</c>/<c>audio_vae.</c> stripped), the vocoder
/// (<c>vocoder.</c> kept), and the optional Gemma text tower. No checkpoint files needed.</summary>
public class LtxVideo2CheckpointConverterTests
{
    [Theory]
    // DiT: prefix strip + the patchify rename; q_norm/k_norm are NOT renamed (the attention reads them verbatim).
    [InlineData("model.diffusion_model.patchify_proj.weight", B.Transformer, "proj_in.weight")]
    [InlineData("model.diffusion_model.audio_patchify_proj.weight", B.Transformer, "audio_proj_in.weight")]
    [InlineData("model.diffusion_model.time_embed.linear.weight", B.Transformer, "time_embed.linear.weight")]
    [InlineData("model.diffusion_model.transformer_blocks.0.attn1.q_norm.weight", B.Transformer, "transformer_blocks.0.attn1.q_norm.weight")]
    [InlineData("model.diffusion_model.transformer_blocks.47.audio_to_video_attn.to_q.weight", B.Transformer, "transformer_blocks.47.audio_to_video_attn.to_q.weight")]
    [InlineData("model.diffusion_model.av_cross_attn_video_a2v_gate.linear.weight", B.Transformer, "av_cross_attn_video_a2v_gate.linear.weight")]
    [InlineData("model.diffusion_model.scale_shift_table", B.Transformer, "scale_shift_table")]
    [InlineData("model.diffusion_model.audio_proj_out.weight", B.Transformer, "audio_proj_out.weight")]
    public void RouteKey_Transformer(string key, B bucket, string mapped)
    {
        (B b, string? m) = LtxVideo2CheckpointConverter.RouteKey(key);
        Assert.Equal(bucket, b);
        Assert.Equal(mapped, m);
    }

    [Theory]
    // Connectors keep their full key (the connector LoadWeights reads these strings verbatim).
    [InlineData("model.diffusion_model.video_embeddings_connector.learnable_registers", "model.diffusion_model.video_embeddings_connector.learnable_registers")]
    [InlineData("model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight", "model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight")]
    [InlineData("text_embedding_projection.video_aggregate_embed.weight", "text_embedding_projection.video_aggregate_embed.weight")]
    public void RouteKey_Connectors(string key, string mapped)
    {
        (B b, string? m) = LtxVideo2CheckpointConverter.RouteKey(key);
        Assert.Equal(B.Connectors, b);
        Assert.Equal(mapped, m);
    }

    [Theory]
    // Original single-file Lightricks VAE (flat up_blocks.0..6) → the diffusers grouping the decoder reads.
    // Real keys verified from the ltx-2 checkpoint header.
    [InlineData("vae.decoder.up_blocks.0.res_blocks.0.conv1.conv.weight", B.Vae, "decoder.mid_block.resnets.0.conv1.conv.weight")]
    [InlineData("vae.decoder.up_blocks.6.res_blocks.4.conv2.conv.bias", B.Vae, "decoder.up_blocks.2.resnets.4.conv2.conv.bias")]
    [InlineData("vae.decoder.up_blocks.1.conv.conv.weight", B.Vae, "decoder.up_blocks.0.upsamplers.0.conv.conv.weight")]
    [InlineData("vae.decoder.up_blocks.3.conv.conv.weight", B.Vae, "decoder.up_blocks.1.upsamplers.0.conv.conv.weight")]
    [InlineData("vae.decoder.up_blocks.5.conv.conv.weight", B.Vae, "decoder.up_blocks.2.upsamplers.0.conv.conv.weight")]
    [InlineData("vae.decoder.up_blocks.4.res_blocks.2.conv1.conv.weight", B.Vae, "decoder.up_blocks.1.resnets.2.conv1.conv.weight")]
    // 22B/2.3: 9 up_blocks (4 up-stages). Index-parity regroup must handle up_blocks.7 (upsample) and .8 (resnets).
    [InlineData("vae.decoder.up_blocks.0.res_blocks.1.conv2.conv.bias", B.Vae, "decoder.mid_block.resnets.1.conv2.conv.bias")]
    [InlineData("vae.decoder.up_blocks.7.conv.conv.weight", B.Vae, "decoder.up_blocks.3.upsamplers.0.conv.conv.weight")]
    [InlineData("vae.decoder.up_blocks.8.res_blocks.3.conv2.conv.bias", B.Vae, "decoder.up_blocks.3.resnets.3.conv2.conv.bias")]
    // Already-diffusers folder-shard keys must pass through unchanged (not re-grouped).
    [InlineData("vae.decoder.up_blocks.2.resnets.0.conv1.conv.weight", B.Vae, "decoder.up_blocks.2.resnets.0.conv1.conv.weight")]
    [InlineData("vae.decoder.mid_block.resnets.0.conv1.conv.weight", B.Vae, "decoder.mid_block.resnets.0.conv1.conv.weight")]
    [InlineData("vae.per_channel_statistics.mean-of-means", B.Vae, "latents_mean")]
    [InlineData("vae.per_channel_statistics.std-of-means", B.Vae, "latents_std")]
    [InlineData("vae.decoder.conv_in.conv.weight", B.Vae, "decoder.conv_in.conv.weight")]
    [InlineData("vae.latents_mean", B.Vae, "latents_mean")]
    [InlineData("audio_vae.decoder.conv_out.conv.weight", B.AudioVae, "decoder.conv_out.conv.weight")]
    [InlineData("audio_vae.latents_std", B.AudioVae, "latents_std")]
    // Vocoder keeps its prefix (the vocoder reads vocoder.vocoder.* / vocoder.bwe_generator.* / vocoder.mel_stft.*).
    [InlineData("vocoder.vocoder.conv_pre.weight", B.Vocoder, "vocoder.vocoder.conv_pre.weight")]
    [InlineData("vocoder.mel_stft.mel_basis", B.Vocoder, "vocoder.mel_stft.mel_basis")]
    // Gemma text tower (if bundled): prefix stripped.
    [InlineData("text_encoder.model.layers.0.self_attn.q_proj.weight", B.TextEncoder, "model.layers.0.self_attn.q_proj.weight")]
    public void RouteKey_AudioVisualComponents(string key, B bucket, string mapped)
    {
        (B b, string? m) = LtxVideo2CheckpointConverter.RouteKey(key);
        Assert.Equal(bucket, b);
        Assert.Equal(mapped, m);
    }

    [Fact]
    public void Convert_RoutesEachComponentToItsBucket()
    {
        Dictionary<string, Core.Tensors.Tensor> w = new()
        {
            ["model.diffusion_model.patchify_proj.weight"] = Stub(),
            ["model.diffusion_model.video_embeddings_connector.learnable_registers"] = Stub(),
            ["text_embedding_projection.video_aggregate_embed.weight"] = Stub(),
            ["vae.decoder.conv_in.conv.weight"] = Stub(),
            ["audio_vae.decoder.conv_in.conv.weight"] = Stub(),
            ["vocoder.vocoder.conv_pre.weight"] = Stub(),
        };
        LtxVideo2CheckpointConverter.ConvertedWeights c = LtxVideo2CheckpointConverter.Convert(w);
        Assert.True(c.Transformer.ContainsKey("proj_in.weight"));
        Assert.True(c.Connectors.ContainsKey("model.diffusion_model.video_embeddings_connector.learnable_registers"));
        Assert.True(c.Connectors.ContainsKey("text_embedding_projection.video_aggregate_embed.weight"));
        Assert.True(c.Vae.ContainsKey("decoder.conv_in.conv.weight"));
        Assert.True(c.AudioVae.ContainsKey("decoder.conv_in.conv.weight"));
        Assert.True(c.Vocoder.ContainsKey("vocoder.vocoder.conv_pre.weight"));
    }

    private static Core.Tensors.Tensor Stub() =>
        new Core.Tensors.Tensor(new Core.Tensors.TensorShape(1), Core.Tensors.DType.F32);
}
