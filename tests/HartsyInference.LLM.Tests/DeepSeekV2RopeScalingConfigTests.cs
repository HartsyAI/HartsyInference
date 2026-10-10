using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>The truncated YaRN correction range is DeepSeek-V2 behaviour, so GgufConfigFactory sets it for deepseek2 only.</summary>
public sealed class DeepSeekV2RopeScalingConfigTests
{
    private static GgufMetadata Metadata(string arch)
    {
        GgufMetadata m = new();
        m.Add("general.architecture", arch);
        m.Add($"{arch}.block_count", 2u);
        m.Add($"{arch}.embedding_length", 16u);
        m.Add($"{arch}.feed_forward_length", 32u);
        m.Add($"{arch}.attention.head_count", 4u);
        m.Add($"{arch}.attention.head_count_kv", 4u);
        m.Add($"{arch}.attention.layer_norm_rms_epsilon", 1e-6f);
        m.Add($"{arch}.rope.freq_base", 10000f);
        m.Add($"{arch}.rope.scaling.type", "yarn");
        m.Add($"{arch}.rope.scaling.factor", 40f);
        m.Add($"{arch}.rope.scaling.original_context_length", 4096u);
        if (arch == "deepseek2")
        {
            m.Add("deepseek2.attention.kv_lora_rank", 8u);
            m.Add("deepseek2.attention.key_length", 12u);
            m.Add("deepseek2.attention.value_length", 8u);
            m.Add("deepseek2.rope.dimension_count", 4u);
        }
        return m;
    }

    private static Dictionary<string, Tensor> Weights() => new()
    {
        ["model.embed_tokens.weight"] = new Tensor(new TensorShape(32, 16), DType.F32),
    };

    [Fact]
    public void DeepSeekV2_SetsTruncatedYarnCorrectionRange()
    {
        TransformerConfig config = GgufConfigFactory.FromGguf(Metadata("deepseek2"), Weights());
        Assert.True(config.RopeScaling.TruncateYarnCorrectionRange);
    }

    [Fact]
    public void OtherYarnArchitectures_KeepFractionalCorrectionRange()
    {
        TransformerConfig config = GgufConfigFactory.FromGguf(Metadata("llama"), Weights());
        Assert.False(config.RopeScaling.TruncateYarnCorrectionRange);
    }
}
