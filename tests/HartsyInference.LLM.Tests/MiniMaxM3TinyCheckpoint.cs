using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.LLM.Tests;

/// <summary>The tiny random MiniMax-M3 fixture written as the release's directory: <c>language_model.</c> names, <c>w1/w2/w3</c> experts, separate dense gate/up, <c>index_*</c> projections and a
/// <c>text_config</c> config.json. The checkpoint stores the centered Gemma norm weight, so nothing is baked in.</summary>
internal static class MiniMaxM3TinyCheckpoint
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "MiniMaxM3");

    /// <summary>Writes <c>config.json</c> and <c>model.safetensors</c> into <paramref name="directory"/>.</summary>
    public static void Write(string directory)
    {
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, "config.json"), ConfigJson());
        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(Dir, "minimax_m3_tiny.safetensors"));
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "minimax_m3_tiny_config.json")));
        SafeTensorsWriter.Save(Path.Combine(directory, "model.safetensors"), LayoutTensors(loader, document.RootElement));
    }

    /// <summary>The tiny fixture's <c>config.json</c> in the release's <c>text_config</c> layout.</summary>
    public static string ConfigJson()
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "minimax_m3_tiny_config.json")));
        JsonElement c = document.RootElement;
        int layers = c.GetProperty("num_hidden_layers").GetInt32();
        int denseLayers = c.GetProperty("mlp_layer_types").EnumerateArray().Count(e => e.GetString() == "dense");
        int[] ones = [.. Enumerable.Repeat(1, layers)];
        Dictionary<string, object> text = new()
        {
            ["hidden_size"] = c.GetProperty("hidden_size").GetInt32(), ["intermediate_size"] = c.GetProperty("intermediate_size").GetInt32(),
            ["dense_intermediate_size"] = c.GetProperty("dense_intermediate_size").GetInt32(), ["shared_intermediate_size"] = c.GetProperty("shared_intermediate_size").GetInt32(),
            ["num_hidden_layers"] = layers, ["num_attention_heads"] = c.GetProperty("num_attention_heads").GetInt32(),
            ["num_key_value_heads"] = c.GetProperty("num_key_value_heads").GetInt32(), ["head_dim"] = c.GetProperty("head_dim").GetInt32(),
            ["vocab_size"] = c.GetProperty("vocab_size").GetInt32(), ["max_position_embeddings"] = 4096, ["rms_norm_eps"] = c.GetProperty("rms_norm_eps").GetDouble(),
            ["use_gemma_norm"] = true, ["attention_output_gate"] = false, ["use_qk_norm"] = true, ["qk_norm_type"] = "per_head",
            ["rope_theta"] = c.GetProperty("rope_parameters").GetProperty("rope_theta").GetDouble(),
            ["partial_rotary_factor"] = c.GetProperty("rope_parameters").GetProperty("partial_rotary_factor").GetDouble(),
            ["hidden_act"] = "swigluoai", ["tie_word_embeddings"] = false, ["num_local_experts"] = c.GetProperty("num_local_experts").GetInt32(),
            ["num_experts_per_tok"] = c.GetProperty("num_experts_per_tok").GetInt32(), ["scoring_func"] = "sigmoid", ["use_routing_bias"] = true,
            ["moe_layer_freq"] = Enumerable.Range(0, layers).Select(i => i < denseLayers ? 0 : 1).ToArray(),
            ["swiglu_alpha"] = c.GetProperty("swiglu_alpha").GetDouble(), ["swiglu_limit"] = c.GetProperty("swiglu_limit").GetDouble(),
            ["routed_scaling_factor"] = c.GetProperty("routed_scaling_factor").GetDouble(),
            ["sparse_attention_config"] = new Dictionary<string, object>
            {
                ["use_sparse_attention"] = true, ["sparse_index_dim"] = c.GetProperty("index_head_dim").GetInt32(),
                ["sparse_num_index_heads"] = c.GetProperty("index_n_heads").GetInt32(), ["sparse_topk_blocks"] = c.GetProperty("index_topk_blocks").GetInt32(),
                ["sparse_block_size"] = c.GetProperty("index_block_size").GetInt32(), ["sparse_score_type"] = "max", ["sparse_init_block"] = 0,
                ["sparse_local_block"] = c.GetProperty("index_local_blocks").GetInt32(), ["sparse_attention_freq"] = ones, ["sparse_disable_index_value"] = ones,
            },
        };
        return JsonSerializer.Serialize(new Dictionary<string, object> { ["model_type"] = "minimax_m3_vl", ["text_config"] = text });
    }

    private static Dictionary<string, Tensor> LayoutTensors(SafeTensorsLoader source, JsonElement c)
    {
        int layers = c.GetProperty("num_hidden_layers").GetInt32();
        int denseLayers = c.GetProperty("mlp_layer_types").EnumerateArray().Count(e => e.GetString() == "dense");
        int moeInter = c.GetProperty("intermediate_size").GetInt32();
        int denseInter = c.GetProperty("dense_intermediate_size").GetInt32();
        int sharedInter = c.GetProperty("shared_intermediate_size").GetInt32();
        Dictionary<string, Tensor> tensors = new()
        {
            ["language_model.model.embed_tokens.weight"] = source.GetTensor("model.embed_tokens.weight"),
            ["language_model.model.norm.weight"] = source.GetTensor("model.norm.weight"),
            ["language_model.lm_head.weight"] = source.GetTensor("lm_head.weight"),
        };
        for (int i = 0; i < layers; i++)
        {
            string hf = $"model.layers.{i}", rel = $"language_model.model.layers.{i}";
            foreach (string norm in (string[])["input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"])
                tensors[$"{rel}.{norm}.weight"] = source.GetTensor($"{hf}.{norm}.weight");
            foreach (string projection in (string[])["q_proj", "k_proj", "v_proj", "o_proj"])
                tensors[$"{rel}.self_attn.{projection}.weight"] = source.GetTensor($"{hf}.self_attn.{projection}.weight");
            foreach (string part in (string[])["q_proj", "k_proj", "q_norm", "k_norm"])
                tensors[$"{rel}.self_attn.index_{part}.weight"] = source.GetTensor($"{hf}.self_attn.indexer.{part}.weight");
            if (i < denseLayers)
            {
                SplitGateUp(tensors, $"{rel}.mlp", source.GetTensor($"{hf}.mlp.gate_up_proj.weight"), denseInter);
                tensors[$"{rel}.mlp.down_proj.weight"] = source.GetTensor($"{hf}.mlp.down_proj.weight");
                continue;
            }
            tensors[$"{rel}.block_sparse_moe.gate.weight"] = source.GetTensor($"{hf}.mlp.gate.weight");
            tensors[$"{rel}.block_sparse_moe.e_score_correction_bias"] = source.GetTensor($"{hf}.mlp.gate.e_score_correction_bias");
            SplitGateUp(tensors, $"{rel}.block_sparse_moe.shared_experts", source.GetTensor($"{hf}.mlp.shared_experts.gate_up_proj.weight"), sharedInter);
            tensors[$"{rel}.block_sparse_moe.shared_experts.down_proj.weight"] = source.GetTensor($"{hf}.mlp.shared_experts.down_proj.weight");
            Tensor gateUp = source.GetTensor($"{hf}.mlp.experts.gate_up_proj");
            Tensor down = source.GetTensor($"{hf}.mlp.experts.down_proj");
            int experts = (int)gateUp.Shape[0];
            Tensor gateUpRows = gateUp.Reshape(new TensorShape(gateUp.Shape[0] * gateUp.Shape[1], gateUp.Shape[2]));
            Tensor downRows = down.Reshape(new TensorShape(down.Shape[0] * down.Shape[1], down.Shape[2]));
            for (int e = 0; e < experts; e++)
            {
                string expert = $"{rel}.block_sparse_moe.experts.{e}";
                tensors[$"{expert}.w1.weight"] = gateUpRows.SliceRows(e * 2L * moeInter, moeInter);
                tensors[$"{expert}.w3.weight"] = gateUpRows.SliceRows(e * 2L * moeInter + moeInter, moeInter);
                tensors[$"{expert}.w2.weight"] = downRows.SliceRows(e * down.Shape[1], down.Shape[1]);
            }
        }
        return tensors;
    }

    private static void SplitGateUp(Dictionary<string, Tensor> tensors, string prefix, Tensor fused, int inter)
    {
        tensors[$"{prefix}.gate_proj.weight"] = fused.SliceRows(0, inter);
        tensors[$"{prefix}.up_proj.weight"] = fused.SliceRows(inter, inter);
    }
}
