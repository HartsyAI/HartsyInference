using System.Text;
using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using B = HartsyInference.ModelAssets.CheckpointConverters.MiniMaxH3CheckpointConverter.MiniMaxH3Bucket;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Tests the pure key routing of <see cref="MiniMaxH3CheckpointConverter.RouteKey"/> and the int8-convrot
/// rejection. Every "real header" key below was read from the actual 535-tensor DiT header
/// (<c>MiniMaxH3/FL2VA/transformer/minimax_h3_fl2va_bf16.safetensors</c>) and the 902-tensor text-encoder header
/// (<c>.../text_encoder/qwen3vl_32b_minimax_h3_bf16.safetensors</c>); no checkpoint files are needed to run them.</summary>
public class MiniMaxH3CheckpointConverterTests
{
    private const string QuantDescriptorJson = "{\"format\": \"int8_tensorwise\", \"convrot\": true, \"convrot_groupsize\": 256}";

    [Theory]
    // The FL2VA transformer file is already flat: every DiT key routes through byte-identical.
    [InlineData("video_patch_proj.weight")]
    [InlineData("video_patch_proj.bias")]
    [InlineData("audio_patch_proj.weight")]
    [InlineData("audio_patch_proj.bias")]
    [InlineData("condition_proj.weight")]
    [InlineData("condition_proj.bias")]
    [InlineData("blocks.0.attn.qkv_proj.weight")]
    [InlineData("blocks.0.attn.q_norm.weight")]
    [InlineData("blocks.0.attn.k_norm.weight")]
    [InlineData("blocks.49.attn.out_proj.weight")]
    [InlineData("blocks.0.mlp.fc1.weight")]
    [InlineData("blocks.49.mlp.fc2.weight")]
    [InlineData("blocks.0.norm1.weight")]
    [InlineData("blocks.49.norm2.weight")]
    [InlineData("blocks.0.adaln_proj.linear.weight")]
    [InlineData("blocks.49.adaln_proj.linear.bias")]
    [InlineData("token_refiner.blocks.0.attn.qkv_proj.weight")]
    [InlineData("token_refiner.blocks.0.attn.q_norm.weight")]
    [InlineData("token_refiner.blocks.1.attn.k_norm.weight")]
    [InlineData("token_refiner.blocks.1.attn.out_proj.weight")]
    [InlineData("token_refiner.blocks.0.mlp.fc1.weight")]
    [InlineData("token_refiner.blocks.1.mlp.fc2.weight")]
    [InlineData("token_refiner.blocks.0.norm1.weight")]
    [InlineData("token_refiner.blocks.1.norm2.weight")]
    [InlineData("token_refiner.final_norm.weight")]
    [InlineData("final_layer.norm.weight")]
    [InlineData("final_layer.adaln_proj.linear.weight")]
    [InlineData("final_layer.adaln_proj.linear.bias")]
    [InlineData("final_layer.video_out.weight")]
    [InlineData("final_layer.video_out.bias")]
    [InlineData("final_layer.audio_out.weight")]
    [InlineData("final_layer.audio_out.bias")]
    [InlineData("time_embedder.proj_in.weight")]
    [InlineData("time_embedder.proj_in.bias")]
    [InlineData("time_embedder.proj_out.weight")]
    [InlineData("time_embedder.proj_out.bias")]
    [InlineData("rope.inv_freq")]
    public void RouteKey_RealDitHeaderKeysPassThroughUnchanged(string key)
    {
        (B bucket, string? mapped) = MiniMaxH3CheckpointConverter.RouteKey(key);
        Assert.Equal(B.Transformer, bucket);
        Assert.Equal(key, mapped);
    }

    [Fact]
    public void RouteKey_PrunedCurveTableStaysWithTheTransformer()
    {
        (B bucket, string? mapped) = MiniMaxH3CheckpointConverter.RouteKey("adaln_t_table");
        Assert.Equal(B.Transformer, bucket);
        Assert.Equal("adaln_t_table", mapped);
    }

    [Theory]
    // The text-encoder file is flat too, and carries no model.norm / lm_head — it is truncated at layer 50.
    [InlineData("model.embed_tokens.weight")]
    [InlineData("model.layers.0.self_attn.q_proj.weight")]
    [InlineData("model.layers.49.self_attn.k_norm.weight")]
    [InlineData("model.layers.49.mlp.down_proj.weight")]
    [InlineData("model.layers.0.input_layernorm.weight")]
    [InlineData("model.layers.49.post_attention_layernorm.weight")]
    [InlineData("visual.patch_embed.proj.weight")]
    [InlineData("visual.pos_embed.weight")]
    [InlineData("visual.blocks.26.attn.qkv.weight")]
    [InlineData("visual.merger.linear_fc2.weight")]
    [InlineData("visual.deepstack_merger_list.2.linear_fc1.bias")]
    public void RouteKey_RealTextEncoderHeaderKeysRouteToTextEncoder(string key)
    {
        (B bucket, string? mapped) = MiniMaxH3CheckpointConverter.RouteKey(key);
        Assert.Equal(B.TextEncoder, bucket);
        Assert.Equal(key, mapped);
    }

    [Theory]
    // Bundled-file prefixes (ComfyUI convention) are stripped; "audio_vae." must win over "vae." and neither may
    // capture the DiT's own audio_patch_proj / video_patch_proj keys.
    [InlineData("model.diffusion_model.blocks.0.attn.qkv_proj.weight", B.Transformer, "blocks.0.attn.qkv_proj.weight")]
    [InlineData("diffusion_model.rope.inv_freq", B.Transformer, "rope.inv_freq")]
    [InlineData("vae.decoder.conv_in.weight", B.VideoVae, "decoder.conv_in.weight")]
    [InlineData("video_vae.decoder.conv_in.weight", B.VideoVae, "decoder.conv_in.weight")]
    [InlineData("audio_vae.decoder.conv_in.weight", B.AudioVae, "decoder.conv_in.weight")]
    [InlineData("text_encoders.qwen3vl_32b.transformer.model.layers.0.mlp.up_proj.weight", B.TextEncoder, "model.layers.0.mlp.up_proj.weight")]
    [InlineData("text_encoder.visual.pos_embed.weight", B.TextEncoder, "visual.pos_embed.weight")]
    public void RouteKey_BundledPrefixesAreStripped(string key, B bucket, string mapped)
    {
        (B b, string? m) = MiniMaxH3CheckpointConverter.RouteKey(key);
        Assert.Equal(bucket, b);
        Assert.Equal(mapped, m);
    }

    [Theory]
    [InlineData("blocks.0.attn.qkv_proj.comfy_quant")]
    [InlineData("blocks.0.attn.qkv_proj.weight_scale")]
    [InlineData("blocks.49.mlp.fc2.comfy_quant")]
    [InlineData("blocks.49.mlp.fc2.weight_scale")]
    [InlineData("blocks.12.attn.out_proj.weight_scale")]
    [InlineData("blocks.12.mlp.fc1.comfy_quant")]
    public void RouteKey_Int8ConvrotCompanionsAreIsolated(string key)
    {
        Assert.Equal(B.Int8Quant, MiniMaxH3CheckpointConverter.RouteKey(key).Bucket);
    }

    [Fact]
    public void Convert_ThrowsNamingInt8Convrot()
    {
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>
        {
            ["video_patch_proj.weight"] = new Tensor(new TensorShape(2, 2), DType.F32),
            ["audio_patch_proj.weight"] = new Tensor(new TensorShape(2, 2), DType.F32),
            ["blocks.0.attn.qkv_proj.weight"] = new Tensor(new TensorShape(4, 4), DType.I8),
            ["blocks.0.attn.qkv_proj.weight_scale"] = new Tensor(new TensorShape(4, 1), DType.F32),
            ["blocks.0.attn.qkv_proj.comfy_quant"] = Descriptor(),
            ["adaln_t_table"] = new Tensor(new TensorShape(1025, 8), DType.F32),
        };
        try
        {
            NotSupportedException ex = Assert.Throws<NotSupportedException>(
                () => MiniMaxH3CheckpointConverter.Convert(weights));
            Assert.Contains("convrot", ex.Message, StringComparison.Ordinal);
            Assert.Contains("int8_tensorwise", ex.Message, StringComparison.Ordinal);
            Assert.Contains("convrot_groupsize", ex.Message, StringComparison.Ordinal);
            Assert.Contains("blocks.0.attn.qkv_proj.comfy_quant", ex.Message, StringComparison.Ordinal);
        }
        finally
        {
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }

    [Fact]
    public void Convert_CastsBf16ToF32AndKeepsF32()
    {
        Tensor bf16 = new Tensor(new TensorShape(2), DType.BF16);
        Span<ushort> raw = bf16.AsSpan<ushort>();
        raw[0] = 0x3F80;
        raw[1] = 0xC000;
        Tensor f32 = new Tensor(new TensorShape(1), DType.F32);
        f32.AsSpan<float>()[0] = 7f;
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>
        {
            ["blocks.0.norm1.weight"] = bf16,
            ["rope.inv_freq"] = f32,
        };

        MiniMaxH3CheckpointConverter.ConvertedWeights converted = MiniMaxH3CheckpointConverter.Convert(weights);
        try
        {
            Tensor cast = converted.Transformer["blocks.0.norm1.weight"];
            Assert.Equal(DType.F32, cast.DType);
            Assert.Equal(1f, cast.AsSpan<float>()[0]);
            Assert.Equal(-2f, cast.AsSpan<float>()[1]);
            Assert.Same(f32, converted.Transformer["rope.inv_freq"]);

            MiniMaxH3CheckpointConverter.ConvertedWeights raw32 = MiniMaxH3CheckpointConverter.Convert(weights, castToF32: false);
            Assert.Same(bf16, raw32.Transformer["blocks.0.norm1.weight"]);
        }
        finally
        {
            converted.Transformer["blocks.0.norm1.weight"].Dispose();
            bf16.Dispose();
            f32.Dispose();
        }
    }

    [Fact]
    public void Convert_SplitsComponentsAndDropsFp8Markers()
    {
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>
        {
            ["model.diffusion_model.video_patch_proj.weight"] = new Tensor(new TensorShape(2, 2), DType.F32),
            ["model.diffusion_model.audio_patch_proj.weight"] = new Tensor(new TensorShape(2, 2), DType.F32),
            ["vae.decoder.conv_in.weight"] = new Tensor(new TensorShape(2, 2), DType.F32),
            ["audio_vae.decoder.conv_in.weight"] = new Tensor(new TensorShape(2, 2), DType.F32),
            ["text_encoders.qwen3vl_32b.transformer.model.embed_tokens.weight"] = new Tensor(new TensorShape(2, 2), DType.F32),
            ["scaled_fp8"] = new Tensor(new TensorShape(1), DType.F32),
        };
        try
        {
            Assert.True(MiniMaxH3CheckpointConverter.IsMiniMaxH3(weights));
            MiniMaxH3CheckpointConverter.ConvertedWeights converted = MiniMaxH3CheckpointConverter.Convert(weights);
            Assert.Equal(2, converted.Transformer.Count);
            Assert.True(converted.Transformer.ContainsKey("video_patch_proj.weight"));
            Assert.Equal(["decoder.conv_in.weight"], converted.VideoVae.Keys);
            Assert.Equal(["decoder.conv_in.weight"], converted.AudioVae.Keys);
            Assert.Equal(["model.embed_tokens.weight"], converted.TextEncoder.Keys);
        }
        finally
        {
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }

    [Fact]
    public void IsMiniMaxH3_RejectsAnUnrelatedCheckpoint()
    {
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>
        {
            ["model.diffusion_model.patchify_proj.weight"] = new Tensor(new TensorShape(2, 2), DType.F32),
        };
        try
        {
            Assert.False(MiniMaxH3CheckpointConverter.IsMiniMaxH3(weights));
        }
        finally
        {
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }

    /// <summary>Routes every key of the real DiT header and asserts none is dropped, renamed, or misbucketed. Skips
    /// when the checkpoint is absent.
    /// <code>MINIMAX_H3_DIT=/path/minimax_h3_fl2va_bf16.safetensors dotnet test --filter MiniMaxH3CheckpointConverter</code></summary>
    [Fact]
    [Trait("Category", "Integration")]
    public void RouteKey_CoversEveryKeyOfTheRealDitHeader()
    {
        string? path = Environment.GetEnvironmentVariable("MINIMAX_H3_DIT");
        if (path is null || !File.Exists(path)) return;

        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(path);
        Dictionary<string, Tensor> all = loader.GetAllTensors();
        Assert.NotEmpty(all);
        Assert.True(MiniMaxH3CheckpointConverter.IsMiniMaxH3(all));
        foreach (string key in all.Keys)
        {
            (B bucket, string? mapped) = MiniMaxH3CheckpointConverter.RouteKey(key);
            Assert.Equal(B.Transformer, bucket);
            Assert.Equal(key, mapped);
        }
        MiniMaxH3CheckpointConverter.ConvertedWeights converted = MiniMaxH3CheckpointConverter.Convert(all, castToF32: false);
        Assert.Equal(all.Count, converted.Transformer.Count);
        Assert.Empty(converted.VideoVae);
        Assert.Empty(converted.AudioVae);
        Assert.Empty(converted.TextEncoder);
    }

    private static Tensor Descriptor()
    {
        byte[] json = Encoding.ASCII.GetBytes(QuantDescriptorJson);
        Tensor t = new Tensor(new TensorShape(json.Length), DType.U8);
        json.CopyTo(t.AsSpan<byte>());
        return t;
    }
}
