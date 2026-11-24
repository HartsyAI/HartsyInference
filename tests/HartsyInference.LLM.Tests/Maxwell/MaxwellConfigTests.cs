using System.Text.Json.Nodes;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.Maxwell;
using Xunit;

namespace HartsyInference.LLM.Tests.Maxwell;

/// <summary>The release's own <c>config.json</c> (commit 5dd7c6f of maxwell-model/Maxwell-V1-Flash) through <see cref="MaxwellConfig.Parse"/>, and the
/// variants it refuses.</summary>
public sealed class MaxwellConfigTests
{
    /// <summary>The real config with every field the parser reads; <paramref name="edit"/> changes it before it is serialized.</summary>
    internal static string Json(Action<JsonObject>? edit = null)
    {
        JsonObject root = JsonNode.Parse($$"""
            {
              "attention_variant": "block_sparse_gqa", "hidden_act": "silu", "hidden_size": 2048, "intermediate_size": 6144,
              "max_position_embeddings": 524288, "model_type": "maxwell_moe", "moe_intermediate_size": 512,
              "shared_expert_intermediate_size": 1024, "num_attention_heads": 32, "num_experts": 256, "num_experts_per_tok": 12,
              "num_hidden_layers": 45, "num_key_value_heads": 4, "attention_head_size": 128, "rms_norm_eps": 1e-06,
              "rope_theta": 20000000, "tie_word_embeddings": false, "vocab_size": 200704, "mlp_only_layers": [0, 1, 2],
              "qk_layernorm": true, "enable_hyper_connections": true, "num_residual_streams": 4, "mhc_sinkhorn_iterations": 20,
              "moe_router_enable_expert_bias": true, "router_score_function": "sigmoid", "mtp_num_layers": 1,
              "block_sparse_gqa_indexer_n_heads": 4, "block_sparse_gqa_indexer_head_dim": 128, "block_sparse_gqa_block_size": 128,
              "block_sparse_gqa_topk_blocks": 16, "block_sparse_gqa_total_support_blocks": 32, "block_sparse_gqa_group_reduction": "per_group_max",
              "block_sparse_gqa_prefix_window_size": 1024, "block_sparse_gqa_local_window_size": 1024, "block_sparse_gqa_num_remote_blocks": 16,
              "block_sparse_gqa_layer_freq": [{{string.Join(",", Enumerable.Repeat(1, 45))}}]
            }
            """)!.AsObject();
        edit?.Invoke(root);
        return root.ToJsonString();
    }

    [Fact]
    public void ReleaseConfig_ParsesToTheDocumentedGeometry()
    {
        MaxwellConfig config = MaxwellConfig.Parse(Json());

        Assert.Equal((2048, 45, 1, 32, 4, 128), (config.HiddenSize, config.NumHiddenLayers, config.MtpLayers, config.NumAttentionHeads,
            config.NumKeyValueHeads, config.HeadDim));
        Assert.Equal((256, 12, 512, 1024, 6144), (config.NumExperts, config.NumExpertsPerTok, config.MoeIntermediateSize,
            config.SharedExpertIntermediateSize, config.IntermediateSize));
        Assert.Equal((4, 20), (config.HcMult, config.HcSinkhornIters));
        Assert.Equal((4, 128, 128, 8, 8, 16), (config.Sparse.IndexerGroups, config.Sparse.IndexerHeadDim, config.Sparse.BlockSize,
            config.Sparse.PrefixBlocks, config.Sparse.LocalBlocks, config.Sparse.RemoteBlocks));
        Assert.True(config.IsDenseLayer(2));
        Assert.False(config.IsDenseLayer(3));
        Assert.Equal(46, config.TotalLayers);
        Assert.Equal(0, config.Fp8Block);
    }

    [Fact]
    public void DenseCausalLimit_IsTheFirstPositionTheIndexerCanChange()
    {
        // 8 prefix + 8 local + 16 remote blocks of 128 tokens cover every causal block below position 4096.
        Assert.Equal(4096, MaxwellConfig.Parse(Json()).Sparse.DenseCausalLimit);
    }

    [Fact]
    public void Fp8Config_ReportsItsScaleBlock()
    {
        string json = Json(r => r["quantization_config"] = JsonNode.Parse(
            """{"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}"""));

        Assert.Equal(128, MaxwellConfig.Parse(json).Fp8Block);
    }

    [Theory]
    [InlineData("model_type", "\"qwen3_moe\"", "model_type")]
    [InlineData("attention_variant", "\"dense\"", "attention_variant")]
    [InlineData("enable_hyper_connections", "false", "hyper-connections")]
    [InlineData("router_score_function", "\"softmax\"", "router")]
    [InlineData("use_shared_expert_gate", "true", "gated shared expert")]
    [InlineData("tie_word_embeddings", "true", "tied")]
    [InlineData("moe_router_dtype", "\"bf16\"", "router dtype")]
    [InlineData("block_sparse_gqa_group_reduction", "\"mean\"", "per_group_max")]
    [InlineData("block_sparse_gqa_total_support_blocks", "30", "total_support_blocks")]
    [InlineData("block_sparse_gqa_prefix_window_size", "1000", "whole blocks")]
    [InlineData("num_residual_streams", "9", "num_residual_streams")]
    public void UnsupportedVariant_IsRefusedByName(string field, string value, string expected)
    {
        string json = Json(r => r[field] = JsonNode.Parse(value));

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => MaxwellConfig.Parse(json));
        Assert.Contains(expected, error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void LayersWithoutBlockSparseAttention_AreRefused()
    {
        string json = Json(r => r["block_sparse_gqa_layer_freq"] = JsonNode.Parse($"[0,{string.Join(",", Enumerable.Repeat(1, 44))}]"));

        Assert.Contains("block-sparse", Assert.Throws<HartsyInferenceException>(() => MaxwellConfig.Parse(json)).Message,
            StringComparison.Ordinal);
    }

    [Fact]
    public void MissingField_NamesTheField()
    {
        string json = Json(r => r.Remove("moe_intermediate_size"));

        Assert.Contains("moe_intermediate_size", Assert.Throws<HartsyInferenceException>(() => MaxwellConfig.Parse(json)).Message,
            StringComparison.Ordinal);
    }

    [Fact]
    public void NonSquareFp8Block_IsRefused()
    {
        string json = Json(r => r["quantization_config"] = JsonNode.Parse(
            """{"quant_method": "fp8", "weight_block_size": [128, 64]}"""));

        Assert.Throws<HartsyInferenceException>(() => MaxwellConfig.Parse(json));
    }
}
