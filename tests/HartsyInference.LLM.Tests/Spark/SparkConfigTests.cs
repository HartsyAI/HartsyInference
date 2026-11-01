using HartsyInference.LLM.Models.Spark;
using Xunit;

namespace HartsyInference.LLM.Tests.Spark;

/// <summary>The Spark-X config reader against the XHToken/Spark-X2.6-4B layout and the variants it must refuse.</summary>
public sealed class SparkConfigTests
{
    private const string Json = """
        {
          "model_type": "spark2_5", "bos_token_id": 0, "eos_token_id": 1, "head_dim": 256, "hidden_size": 2560, "intermediate_size": 10240,
          "hidden_act": "gelu", "gate_attn_act_mode": "sigmoid", "headwise_attn_output_gate": true, "attention_bias": false, "mlp_bias": false,
          "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
          "max_position_embeddings": 1048576, "num_attention_heads": 16, "num_hidden_layers": 4, "num_key_value_heads": 4,
          "rms_norm_eps": 1e-06, "sliding_window": 512, "tie_word_embeddings": true, "vocab_size": 131072,
          "rope_parameters": {
            "full_attention": { "partial_rotary_factor": 0.25, "rope_theta": 5000000 },
            "sliding_attention": { "partial_rotary_factor": 1.0, "rope_theta": 10000 }
          }
        }
        """;

    [Fact]
    public void ReadsReleasedLayout()
    {
        SparkConfig c = SparkConfig.FromJson(Json);
        Assert.Equal([true, true, true, false], c.SlidingLayers);
        Assert.Equal(512, c.SlidingWindow);
        Assert.Equal(64, c.RopeOf(3).RotaryDim(c.HeadDim));
        Assert.Equal(5_000_000f, c.RopeOf(3).Theta);
        Assert.Equal(256, c.RopeOf(0).RotaryDim(c.HeadDim));
        Assert.Equal(10_000f, c.RopeOf(0).Theta);
        Assert.Equal(1, c.EosTokenId);
    }

    [Theory]
    [InlineData("\"model_type\": \"spark2_5\"", "\"model_type\": \"llama\"", typeof(NotSupportedException))]
    [InlineData("\"hidden_act\": \"gelu\"", "\"hidden_act\": \"silu\"", typeof(NotSupportedException))]
    [InlineData("\"gate_attn_act_mode\": \"sigmoid\"", "\"gate_attn_act_mode\": \"silu\"", typeof(NotSupportedException))]
    [InlineData("\"headwise_attn_output_gate\": true", "\"headwise_attn_output_gate\": false", typeof(NotSupportedException))]
    [InlineData("\"attention_bias\": false", "\"attention_bias\": true", typeof(NotSupportedException))]
    [InlineData("\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", \"full_attention\"", "\"sliding_attention\", \"state_space\", \"sliding_attention\", \"full_attention\"", typeof(NotSupportedException))]
    [InlineData("\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", \"full_attention\"", "\"sliding_attention\", \"full_attention\"", typeof(InvalidDataException))]
    [InlineData("\"sliding_window\": 512", "\"sliding_window\": 0", typeof(InvalidDataException))]
    [InlineData("\"num_key_value_heads\": 4", "\"num_key_value_heads\": 5", typeof(InvalidDataException))]
    public void RefusesUnsupportedLayouts(string from, string to, Type expected)
    {
        string json = Json.Replace(from, to);
        Assert.NotEqual(Json, json);
        Assert.Throws(expected, () => SparkConfig.FromJson(json));
    }
}
