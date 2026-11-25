using System.Text.Json;
using HartsyInference.Core.Rope;
using HartsyInference.LLM.Ssm;
using Xunit;

namespace HartsyInference.LLM.Tests.Ssm;

/// <summary>The RoPE scaling a Qwen3.5 text config declares: both scopes are read, and a YaRN config without its original context length is refused
/// (HF's <c>_compute_yarn_parameters</c> reads it with no fallback).</summary>
public sealed class SenseNovaLoopRopeConfigTests
{
    private static Qwen35HfConfig Parse(string ropeFields) =>
        Qwen35HfConfig.FromTextConfig(JsonDocument.Parse($$"""
            { "hidden_size": 64, "num_hidden_layers": 4, "vocab_size": 80, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
              "full_attention_interval": 4, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 8, "linear_value_head_dim": 8,
              "linear_num_key_heads": 2, "linear_num_value_heads": 4, "max_position_embeddings": 256, {{ropeFields}} }
            """).RootElement);

    [Fact]
    public void RopeScalingScope_IsReadFromEitherKey()
    {
        const string yarn = """ "type": "yarn", "rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 128, "rope_theta": 10000.0 """;
        Assert.Equal(RopeScalingType.Yarn, Parse($"\"rope_parameters\": {{ {yarn} }}").RopeScaling.Type);
        Assert.Equal(RopeScalingType.Yarn, Parse($"\"rope_scaling\": {{ {yarn} }}").RopeScaling.Type);
        Assert.Equal(RopeScalingType.None, Parse("\"rope_parameters\": { \"rope_type\": \"default\", \"rope_theta\": 10000.0 }").RopeScaling.Type);
    }

    [Fact]
    public void YarnWithoutOriginalContextLength_IsRefused()
    {
        string json = "\"rope_parameters\": { \"rope_type\": \"yarn\", \"factor\": 2.0, \"rope_theta\": 10000.0 }";
        Assert.Throws<InvalidDataException>(() => Parse(json));
    }

    [Fact]
    public void UnknownRopeType_IsRefused()
    {
        Assert.Throws<InvalidDataException>(() => Parse("\"rope_parameters\": { \"rope_type\": \"linear\", \"factor\": 2.0 }"));
    }
}
