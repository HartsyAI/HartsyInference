using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>DeepSeek-V2-Lite's GGUF omits <c>deepseek2.expert_weights_norm</c>, because its HF config has
/// <c>norm_topk_prob=false</c>. An absent key must therefore mean no top-k renormalization, while an explicit key
/// (DeepSeek-V3 writes <c>true</c>) still decides.</summary>
public sealed class DeepSeekV2GgufRoutingTests
{
    private static GgufMetadata Metadata(bool? expertWeightsNorm)
    {
        GgufMetadata m = new();
        m.Add("general.architecture", "deepseek2");
        m.Add("deepseek2.block_count", 2u);
        m.Add("deepseek2.embedding_length", 16u);
        m.Add("deepseek2.feed_forward_length", 32u);
        m.Add("deepseek2.attention.head_count", 4u);
        m.Add("deepseek2.attention.head_count_kv", 4u);
        m.Add("deepseek2.attention.kv_lora_rank", 8u);
        m.Add("deepseek2.attention.key_length", 12u);
        m.Add("deepseek2.attention.value_length", 8u);
        m.Add("deepseek2.rope.dimension_count", 4u);
        m.Add("deepseek2.rope.freq_base", 10000f);
        m.Add("deepseek2.attention.layer_norm_rms_epsilon", 1e-6f);
        m.Add("deepseek2.expert_count", 8u);
        m.Add("deepseek2.expert_used_count", 2u);
        m.Add("deepseek2.expert_feed_forward_length", 8u);
        m.Add("deepseek2.expert_shared_count", 2u);
        m.Add("deepseek2.leading_dense_block_count", 1u);
        m.Add("deepseek2.expert_weights_scale", 1f);
        if (expertWeightsNorm is bool norm) m.Add("deepseek2.expert_weights_norm", norm);
        return m;
    }

    private static Dictionary<string, Tensor> Weights() => new()
    {
        ["model.embed_tokens.weight"] = new Tensor(new TensorShape(32, 16), DType.F32),
    };

    [Fact]
    public void AbsentExpertWeightsNorm_DoesNotRenormalizeTopK()
    {
        TransformerConfig config = GgufConfigFactory.FromGguf(Metadata(expertWeightsNorm: null), Weights());
        Assert.False(config.Moe!.NormTopKProb);
    }

    [Fact]
    public void ExplicitExpertWeightsNormTrue_RenormalizesTopK()
    {
        TransformerConfig config = GgufConfigFactory.FromGguf(Metadata(expertWeightsNorm: true), Weights());
        Assert.True(config.Moe!.NormTopKProb);
    }
}
