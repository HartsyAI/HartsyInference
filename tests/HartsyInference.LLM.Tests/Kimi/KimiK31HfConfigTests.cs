using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Kimi;
using Xunit;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>The Kimi K3.1 config reader against the published checkpoint's values, and the loader's refusal of MXFP4-packed experts.</summary>
public sealed class KimiK31HfConfigTests
{
    // The published Kimi-K3.1-Pro config.json, reduced to the fields the reader uses (values unchanged).
    private const string PublishedConfig = """
    {
      "model_type": "kimi_k31",
      "quantization_config": { "format": "mxfp4-pack-quantized" },
      "text_config": {
        "model_type": "kimi_linear",
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "attn_res_block_size": 12,
        "dsa_config": { "index_num_heads": 32, "index_head_dim": 128, "index_topk": 2048 },
        "first_k_dense_replace": 1,
        "hidden_size": 7168,
        "intermediate_size": 33792,
        "kv_lora_rank": 512,
        "linear_attn_config": {
          "full_attn_layers": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93],
          "gate_lower_bound": -5.0,
          "head_dim": 128,
          "num_heads": 96,
          "short_conv_kernel_size": 4
        },
        "moe_intermediate_size": 3072,
        "num_attention_heads": 96,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_hidden_layers": 93,
        "num_key_value_heads": 96,
        "num_shared_experts": 2,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "rms_norm_eps": 1e-05,
        "routed_expert_hidden_size": 3584,
        "v_head_dim": 128,
        "vocab_size": 163840
      }
    }
    """;

    [Fact]
    public void Parse_ReadsThePublishedDimensions()
    {
        KimiK31Config c = KimiK31HfConfig.Parse(PublishedConfig);

        Assert.Equal(7168, c.Hidden);
        Assert.Equal(163840, c.Vocab);
        Assert.Equal(93, c.Layers);
        Assert.Equal(1, c.DenseLayers);
        Assert.Equal(33792, c.DenseIntermediate);
        Assert.Equal(96, c.Heads);
        Assert.Equal(128, c.KdaHeadDim);
        Assert.Equal(4, c.ConvKernel);
        Assert.Equal(-5f, c.GateLowerBound);
        Assert.Equal(24, c.FullAttentionLayers.Count);
        Assert.Contains(3, c.FullAttentionLayers);
        Assert.Contains(92, c.FullAttentionLayers);
        Assert.DoesNotContain(4, c.FullAttentionLayers);
        Assert.Equal(1536, c.QLoraRank);
        Assert.Equal(512, c.KvLoraRank);
        Assert.Equal(128, c.QkNopeDim);
        Assert.Equal(64, c.QkRopeDim);
        Assert.Equal(128, c.VDim);
        Assert.Equal(32, c.IndexHeads);
        Assert.Equal(128, c.IndexHeadDim);
        Assert.Equal(2048, c.IndexTopK);
        Assert.Equal(896, c.Experts);
        Assert.Equal(16, c.ExpertsPerToken);
        Assert.Equal(3072, c.MoeIntermediate);
        Assert.Equal(2, c.SharedExperts);
        Assert.Equal(3584, c.LatentSize);
        Assert.Equal(4f, c.SituBeta);
        Assert.Equal(25f, c.SituLinearBeta);
        Assert.Equal(12, c.AttnResBlock);
    }

    [Fact]
    public void Parse_RoutedExpertsAreMxfp4()
    {
        Assert.True(KimiK31HfConfig.RoutedExpertsAreMxfp4(PublishedConfig));
    }

    [Fact]
    public void Parse_RefusesAnotherModelType()
    {
        string other = PublishedConfig.Replace("\"kimi_k31\"", "\"deepseek_v3\"", StringComparison.Ordinal);
        Assert.Throws<InvalidDataException>(() => KimiK31HfConfig.Parse(other));
    }

    [Fact]
    public void Parse_RefusesANonSitu_Activation()
    {
        string gelu = PublishedConfig.Replace("\"situ\"", "\"gelu\"", StringComparison.Ordinal);
        Assert.Throws<InvalidDataException>(() => KimiK31HfConfig.Parse(gelu));
    }

    [Fact]
    public void Load_RefusesPackedExpertsBeforeReadingAnything()
    {
        KimiK31Config c = KimiK31HfConfig.Parse(PublishedConfig);
        RecordingSource source = new();
        NotSupportedException error = Assert.Throws<NotSupportedException>(() => KimiK31WeightLoader.Load(c, source));
        Assert.Contains("MXFP4-packed", error.Message, StringComparison.Ordinal);
        Assert.Equal(0, source.ReadCount);
    }

    /// <summary>A source that claims every packed expert exists and counts any read; the loader must refuse before reading.</summary>
    private sealed class RecordingSource : IKimiK31WeightSource
    {
        public int ReadCount { get; private set; }

        public bool Has(string name) => name.EndsWith(".w1.weight_packed", StringComparison.Ordinal);

        public Tensor Read(string name)
        {
            ReadCount++;
            throw new KeyNotFoundException(name);
        }
    }
}
