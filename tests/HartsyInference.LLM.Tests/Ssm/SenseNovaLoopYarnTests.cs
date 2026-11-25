using System.Text.Json;
using HartsyInference.Core.Rope;
using HartsyInference.LLM.Ssm;
using Xunit;

namespace HartsyInference.LLM.Tests.Ssm;

/// <summary>The YaRN frequencies and attention scaling at SenseNova-Loop-9B's real rope parameters (64 rotary channels, theta 1e7, factor 2,
/// original 262144) against what transformers' <c>ROPE_INIT_FUNCTIONS["yarn"]</c> returns for the shipped config
/// (<c>tools/sensenova_loop/yarn_reference.py</c>, transformers 5.24.0). The tiny fixtures only exercise two frequency pairs.</summary>
public sealed class SenseNovaLoopYarnTests
{
    private const string RealRopeConfig = """
        { "hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 248320, "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
          "layer_types": [], "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
          "linear_num_key_heads": 16, "linear_num_value_heads": 32, "full_attention_interval": 4, "max_position_embeddings": 524288,
          "rope_parameters": { "mrope_interleaved": true, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25, "rope_theta": 10000000,
            "rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 262144 } }
        """;

    [Fact]
    public void RealConfig_MatchesTransformers()
    {
        using JsonDocument reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "SenseNovaLoop", "yarn_real.json")));
        double[] want = reference.RootElement.GetProperty("inv_freq").EnumerateArray().Select(e => e.GetDouble()).ToArray();
        double wantScaling = reference.RootElement.GetProperty("attention_scaling").GetDouble();

        string json = RealRopeConfig.Replace("\"layer_types\": [],", "");
        Qwen35HfConfig cfg = Qwen35HfConfig.FromTextConfig(JsonDocument.Parse(json).RootElement);
        Assert.Equal(RopeScalingType.Yarn, cfg.RopeScaling.Type);
        Assert.Equal(64, cfg.RotaryDim);
        (double[] got, double scaling) = RopeFrequencyBuilder.Build(cfg.RotaryDim, cfg.RopeTheta, cfg.RopeScaling, 1);

        Assert.Equal(want.Length, got.Length);
        double worst = 0;
        for (int i = 0; i < want.Length; i++) worst = Math.Max(worst, Math.Abs(got[i] - want[i]) / want[i]);
        Assert.True(worst < 1e-6, $"max relative difference of the inverse frequencies {worst}");
        Assert.Equal(wantScaling, scaling, 9);
        // The blend is not a no-op: the slowest frequencies are halved.
        Assert.True(got[^1] < RopeFrequencyBuilder.Build(cfg.RotaryDim, cfg.RopeTheta, RopeScaling.None, 1).InvFreq[^1] * 0.75);
    }
}
