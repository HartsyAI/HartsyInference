using HartsyInference.LLM.Models.Lfm3;
using Xunit;

namespace HartsyInference.LLM.Tests.Lfm3;

/// <summary>The LFM3 config reader against the LiquidAI/LFM3-1.7B config.json and the layouts it must refuse.</summary>
public sealed class Lfm3ConfigTests
{
    private const string Json = """
        {
          "model_type": "lfm3", "bos_token_id": 124894, "eos_token_id": 124900, "conv_L_cache": 7, "conv_bias": false,
          "head_dim": 128, "hidden_size": 2048, "intermediate_size": 7680, "max_position_embeddings": 262144,
          "norm_eps": 1e-05, "num_attention_heads": 16, "num_hidden_layers": 6, "num_key_value_heads": 4,
          "layer_types": ["conv", "conv", "full_attention", "conv", "sliding_attention", "conv"],
          "rope_parameters": {
            "full_attention": { "partial_rotary_factor": 0.5, "rope_theta": 1000000.0, "rope_type": "default" },
            "sliding_attention": { "rope_theta": 10000, "rope_type": "default" }
          },
          "sliding_window": 256, "tie_word_embeddings": true, "use_sandwich_norm": true, "vocab_size": 128000
        }
        """;

    [Fact]
    public void ReadsReleasedLayout()
    {
        Lfm3Config c = Lfm3Config.FromJson(Json);
        Assert.Equal(6, c.NumLayers);
        Assert.Equal([Lfm3LayerKind.Conv, Lfm3LayerKind.Conv, Lfm3LayerKind.FullAttention, Lfm3LayerKind.Conv,
            Lfm3LayerKind.SlidingAttention, Lfm3LayerKind.Conv], c.Layers);
        Assert.Equal(7, c.ConvKernel);
        Assert.True(c.SandwichNorm);
        Assert.Equal(256, c.SlidingWindow);
        Assert.Equal(124900, c.EosTokenId);
        Assert.Equal(32, c.RopeOf(2).RotatedPairs(c.HeadDim));
        Assert.Equal(1_000_000f, c.RopeOf(2).Theta);
        Assert.Equal(64, c.RopeOf(4).RotatedPairs(c.HeadDim));
        Assert.Equal(10_000f, c.RopeOf(4).Theta);
        Assert.Throws<InvalidOperationException>(() => c.RopeOf(0));
    }

    [Theory]
    [InlineData("\"model_type\": \"lfm3\"", "\"model_type\": \"llama\"", typeof(NotSupportedException))]
    [InlineData("\"conv_bias\": false", "\"conv_bias\": true", typeof(NotSupportedException))]
    [InlineData("\"rope_type\": \"default\" },\n    \"sliding", "\"rope_type\": \"yarn\" },\n    \"sliding", typeof(NotSupportedException))]
    [InlineData("\"conv\", \"conv\", \"full_attention\"", "\"conv\", \"full_attention\"", typeof(InvalidDataException))]
    [InlineData("\"sliding_window\": 256", "\"sliding_window\": 0", typeof(InvalidDataException))]
    [InlineData("\"num_key_value_heads\": 4", "\"num_key_value_heads\": 5", typeof(InvalidDataException))]
    [InlineData("\"conv\", \"sliding_attention\"", "\"conv\", \"state_space\"", typeof(NotSupportedException))]
    public void RefusesUnsupportedLayouts(string from, string to, Type expected)
    {
        string json = Json.Replace(from, to);
        Assert.NotEqual(Json, json);
        Assert.Throws(expected, () => Lfm3Config.FromJson(json));
    }
}
