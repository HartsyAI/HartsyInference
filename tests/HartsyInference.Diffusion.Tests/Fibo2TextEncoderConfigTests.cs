using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Fibo 2.0 text-encoder shape inference and the encoder-to-denoiser checks. Tensors are shape-only, so no weights are allocated.</summary>
public sealed class Fibo2TextEncoderConfigTests
{
    /// <summary>A tiny Qwen3-style layout: hidden 16, 4 query heads, 2 key/value heads, head dim 8, FFN 24, vocab 32.</summary>
    private static Dictionary<string, Tensor> TinyEncoder(int layers, string prefix = "")
    {
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal)
        {
            [prefix + "embed_tokens.weight"] = Shape(32, 16),
            [prefix + "norm.weight"] = Shape(16),
        };
        for (int i = 0; i < layers; i++)
        {
            string block = $"{prefix}layers.{i}.";
            weights[block + "self_attn.q_proj.weight"] = Shape(32, 16);
            weights[block + "self_attn.k_proj.weight"] = Shape(16, 16);
            weights[block + "self_attn.q_norm.weight"] = Shape(8);
            weights[block + "mlp.gate_proj.weight"] = Shape(24, 16);
        }
        return weights;
    }

    [Theory]
    [InlineData("")]
    [InlineData("model.")]
    public void FromWeights_ReadsTheStructureFromShapes(string prefix)
    {
        LlamaStyleEncoderConfig config = Fibo2TextEncoderConfig.FromWeights(TinyEncoder(layers: 3, prefix));

        Assert.Equal(16, config.HiddenSize);
        Assert.Equal(3, config.NumLayers);
        Assert.Equal(4, config.NumQueryHeads);
        Assert.Equal(2, config.NumKvHeads);
        Assert.Equal(8, config.HeadDim);
        Assert.Equal(24, config.IntermediateSize);
        Assert.Equal(32, config.VocabSize);
        Assert.Equal(LlamaStyleEncoderConfig.Qwen3_VL_4B.RopeTheta, config.RopeTheta);
    }

    [Fact]
    public void FromWeights_MissingBlockKeyIsRejected()
    {
        Dictionary<string, Tensor> weights = TinyEncoder(layers: 2);
        weights.Remove("layers.0.self_attn.q_norm.weight");

        Assert.Throws<KeyNotFoundException>(() => Fibo2TextEncoderConfig.FromWeights(weights));
    }

    [Fact]
    public void ValidateFor_QwenVl4bPresetMatchesTheFiboCheckpoint()
    {
        Fibo2TextEncoderConfig.ValidateFor(LlamaStyleEncoderConfig.Qwen3_VL_4B, Fibo2Config.Checkpoint);
    }

    [Fact]
    public void ValidateFor_CapFeatWidthMustBeThreeTapsOfTheEncoderHidden()
    {
        Fibo2Config mismatched = Fibo2Config.Checkpoint with { CapFeatDim = 4096 };

        InvalidDataException error = Assert.Throws<InvalidDataException>(
            () => Fibo2TextEncoderConfig.ValidateFor(LlamaStyleEncoderConfig.Qwen3_VL_4B, mismatched));

        Assert.Contains("cap_feat_dim", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ValidateFor_RejectsAnEncoderThatCannotReachTheHighestTap()
    {
        LlamaStyleEncoderConfig shallow = LlamaStyleEncoderConfig.Qwen3_VL_4B with { NumLayers = Fibo2TextEncoderConfig.RequiredLayers - 1 };

        Assert.Throws<InvalidDataException>(() => Fibo2TextEncoderConfig.ValidateFor(shallow, Fibo2Config.Checkpoint));
    }

    [Fact]
    public void RequiredLayers_IsTheHighestTapLayer()
    {
        Assert.Equal(35, Fibo2TextEncoderConfig.RequiredLayers);
    }

    private static Tensor Shape(params long[] dims) => new(new TensorShape(dims), DType.F32);
}
