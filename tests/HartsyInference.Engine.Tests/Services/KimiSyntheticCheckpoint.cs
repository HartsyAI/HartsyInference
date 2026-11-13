using System.Globalization;
using System.Text;
using System.Text.Json;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>Writes a tiny Kimi K3.1 checkpoint directory under the published tensor names, with seeded random F32 weights: <c>config.json</c>, one
/// safetensors shard, its index, and the byte-level tokenizer and chat template the Text path needs. Its layers are KDA, MLA, KDA, MLA with a
/// dense first layer and latent MoE after it. The vocabulary covers the ids of the shared synthetic tokenizer.</summary>
internal static class KimiSyntheticCheckpoint
{
    // Covers the merged id 258 ("hi") of the shared tokenizer.
    internal const int Hidden = 16, Vocab = Ernie5SyntheticCheckpoint.EosId + 2, Layers = 4, DenseLayers = 1, DenseIntermediate = 24, Heads = 2, KdaHeadDim = 8;
    internal const int ConvKernel = 4, QLora = 8, KvLora = 6, Nope = 4, Rope = 4, VDim = 4, IndexHeads = 2, IndexHeadDim = 8, TopK = 16;
    internal const int Experts = 4, ExpertsPerTok = 2, MoeInter = 8, Shared = 1, Latent = 8, Block = 2, KdaForgetRank = 128;
    internal const int BosId = Ernie5SyntheticCheckpoint.BosId, EosId = Ernie5SyntheticCheckpoint.EosId;

    // 0-based layers that use MLA with the DSA indexer.
    private static readonly int[] MlaLayers = [1, 3];

    /// <summary>Writes the checkpoint into <paramref name="directory"/>; <paramref name="withVision"/> adds a vision config and vision tensors.</summary>
    internal static void Write(string directory, int seed = 7, bool withVision = false)
    {
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, "config.json"), ConfigJson(withVision));

        List<(string Name, int[] Shape)> tensors = TensorNames(withVision);
        Random rng = new(seed);
        WriteShard(Path.Combine(directory, "model.safetensors"), tensors, rng);

        Dictionary<string, object> weightMap = new(StringComparer.Ordinal);
        foreach ((string name, _) in tensors) weightMap[name] = "model.safetensors";
        File.WriteAllText(Path.Combine(directory, "model.safetensors.index.json"),
            JsonSerializer.Serialize(new Dictionary<string, object> { ["weight_map"] = weightMap }));

        File.WriteAllText(Path.Combine(directory, "tokenizer.json"), Ernie5SyntheticCheckpoint.TokenizerJson());
        File.WriteAllText(Path.Combine(directory, "tokenizer_config.json"), "{\"bos_token\": \"<s>\", \"eos_token\": \"</s>\"}");
        File.WriteAllText(Path.Combine(directory, "chat_template.jinja"), "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}");
    }

    internal static string ConfigJson(bool withVision = false)
    {
        int[] oneBased = [.. MlaLayers.Select(i => i + 1)];
        string vision = withVision ? "\"vision_config\": {\"hidden_size\": 8, \"patch_size\": 14}, " : string.Empty;
        return "{"
            + "\"model_type\": \"kimi_k31\", " + vision
            + $"\"bos_token_id\": {BosId}, \"eos_token_id\": {EosId}, "
            + "\"text_config\": {"
            + "\"model_type\": \"kimi_linear\", \"hidden_act\": \"situ\", \"activation_situ_beta\": 4.0, \"activation_situ_linear_beta\": 25.0,"
            + $"\"attn_res_block_size\": {Block}, \"dsa_config\": {{\"index_num_heads\": {IndexHeads}, \"index_head_dim\": {IndexHeadDim}, \"index_topk\": {TopK}}},"
            + $"\"first_k_dense_replace\": {DenseLayers}, \"hidden_size\": {Hidden}, \"intermediate_size\": {DenseIntermediate}, \"kv_lora_rank\": {KvLora},"
            + $"\"linear_attn_config\": {{\"full_attn_layers\": [{string.Join(", ", oneBased)}], \"gate_lower_bound\": -5.0, \"head_dim\": {KdaHeadDim}, \"num_heads\": {Heads}, \"short_conv_kernel_size\": {ConvKernel}}},"
            + $"\"moe_intermediate_size\": {MoeInter}, \"num_attention_heads\": {Heads}, \"num_experts\": {Experts}, \"num_experts_per_token\": {ExpertsPerTok},"
            + $"\"num_hidden_layers\": {Layers}, \"num_key_value_heads\": {Heads}, \"num_shared_experts\": {Shared}, \"q_lora_rank\": {QLora},"
            + $"\"qk_nope_head_dim\": {Nope}, \"qk_rope_head_dim\": {Rope}, \"rms_norm_eps\": 1e-5, \"routed_expert_hidden_size\": {Latent},"
            + $"\"v_head_dim\": {VDim}, \"vocab_size\": {Vocab.ToString(CultureInfo.InvariantCulture)}"
            + "}}";
    }

    internal static List<(string Name, int[] Shape)> TensorNames(bool withVision = false)
    {
        const string root = "language_model.model.";
        int keyDim = Heads * KdaHeadDim, qk = Nope + Rope;
        List<(string, int[])> t = new()
        {
            ($"{root}embed_tokens.weight", [Vocab, Hidden]),
            ($"{root}norm.weight", [Hidden]),
            ("language_model.lm_head.weight", [Vocab, Hidden]),
            ($"{root}output_attn_res_proj.weight", [1, Hidden]),
            ($"{root}output_attn_res_norm.weight", [Hidden]),
        };
        for (int i = 0; i < Layers; i++)
        {
            string p = $"{root}layers.{i}.";
            t.Add((p + "input_layernorm.weight", [Hidden]));
            t.Add((p + "post_attention_layernorm.weight", [Hidden]));
            t.Add((p + "self_attention_res_proj.weight", [1, Hidden]));
            t.Add((p + "self_attention_res_norm.weight", [Hidden]));
            t.Add((p + "mlp_res_proj.weight", [1, Hidden]));
            t.Add((p + "mlp_res_norm.weight", [Hidden]));
            string a = p + "self_attn.";
            if (MlaLayers.Contains(i))
            {
                t.Add((a + "kv_a_proj_with_mqa.weight", [KvLora + Rope, Hidden]));
                t.Add((a + "q_a_proj.weight", [QLora, Hidden]));
                t.Add((a + "q_b_proj.weight", [Heads * qk, QLora]));
                t.Add((a + "kv_b_proj.weight", [Heads * (Nope + VDim), KvLora]));
                t.Add((a + "o_proj.weight", [Hidden, Heads * VDim]));
                t.Add((a + "g_proj.weight", [Heads * VDim, Hidden]));
                t.Add((a + "q_a_layernorm.weight", [QLora]));
                t.Add((a + "kv_a_layernorm.weight", [KvLora]));
                t.Add((a + "indexer.wq_b.weight", [IndexHeads * IndexHeadDim, QLora]));
                t.Add((a + "indexer.wk.weight", [IndexHeadDim, Hidden]));
                t.Add((a + "indexer.k_norm.weight", [IndexHeadDim]));
                t.Add((a + "indexer.k_norm.bias", [IndexHeadDim]));
                t.Add((a + "indexer.weights_proj.weight", [IndexHeads, Hidden]));
            }
            else
            {
                foreach (string proj in new[] { "q_proj", "k_proj", "v_proj" }) t.Add((a + proj + ".weight", [keyDim, Hidden]));
                foreach (string conv in new[] { "q_conv1d", "k_conv1d", "v_conv1d" }) t.Add((a + conv + ".weight", [keyDim, 1, ConvKernel]));
                t.Add((a + "A_log", [KdaHeadDim]));
                t.Add((a + "dt_bias", [keyDim]));
                t.Add((a + "f_a_proj.weight", [KdaForgetRank, Hidden]));
                t.Add((a + "f_b_proj.weight", [keyDim, KdaForgetRank]));
                t.Add((a + "b_proj.weight", [Heads, Hidden]));
                t.Add((a + "g_proj.weight", [keyDim, Hidden]));
                t.Add((a + "o_norm.weight", [KdaHeadDim]));
                t.Add((a + "o_proj.weight", [Hidden, keyDim]));
            }
            if (i < DenseLayers)
            {
                t.Add((p + "mlp.gate_proj.weight", [DenseIntermediate, Hidden]));
                t.Add((p + "mlp.up_proj.weight", [DenseIntermediate, Hidden]));
                t.Add((p + "mlp.down_proj.weight", [Hidden, DenseIntermediate]));
            }
            else
            {
                string m = p + "block_sparse_moe.";
                t.Add((m + "gate.weight", [Experts, Hidden]));
                t.Add((m + "gate.e_score_correction_bias", [Experts]));
                t.Add((m + "routed_expert_down_proj.weight", [Latent, Hidden]));
                t.Add((m + "routed_expert_norm.weight", [Latent]));
                t.Add((m + "routed_expert_up_proj.weight", [Hidden, Latent]));
                t.Add((m + "shared_experts.gate_proj.weight", [Shared * MoeInter, Hidden]));
                t.Add((m + "shared_experts.up_proj.weight", [Shared * MoeInter, Hidden]));
                t.Add((m + "shared_experts.down_proj.weight", [Hidden, Shared * MoeInter]));
                for (int e = 0; e < Experts; e++)
                {
                    t.Add(($"{m}experts.{e}.w1.weight", [MoeInter, Latent]));
                    t.Add(($"{m}experts.{e}.w3.weight", [MoeInter, Latent]));
                    t.Add(($"{m}experts.{e}.w2.weight", [Latent, MoeInter]));
                }
            }
        }
        if (withVision)
        {
            t.Add(("vision_tower.encoder.weight", [4, 4]));
            t.Add(("mm_projector.proj.weight", [4, 4]));
        }
        return t.Select(x => (x.Item1, x.Item2)).ToList();
    }

    // A safetensors shard: an 8-byte little-endian header length, the JSON header of {name: {dtype, shape, data_offsets}}, then the F32 data.
    private static void WriteShard(string path, List<(string Name, int[] Shape)> tensors, Random rng)
    {
        Dictionary<string, object> header = new(StringComparer.Ordinal);
        long offset = 0;
        List<float[]> data = new();
        foreach ((string name, int[] shape) in tensors)
        {
            long count = 1;
            foreach (int d in shape) count *= d;
            float[] values = new float[count];
            for (int i = 0; i < values.Length; i++) values[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
            if (name.EndsWith("norm.weight", StringComparison.Ordinal)) Array.Fill(values, 1f);
            // The stored A_log is [KdaHeadDim] with zero padding past the heads, as the checkpoint stores it.
            if (name.EndsWith("A_log", StringComparison.Ordinal)) for (int i = Heads; i < values.Length; i++) values[i] = 0f;
            long bytes = count * 4;
            header[name] = new Dictionary<string, object>
            {
                ["dtype"] = "F32",
                ["shape"] = shape,
                ["data_offsets"] = new long[] { offset, offset + bytes },
            };
            offset += bytes;
            data.Add(values);
        }
        byte[] headerBytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        using FileStream file = File.Create(path);
        using BinaryWriter writer = new(file);
        writer.Write((long)headerBytes.Length);
        writer.Write(headerBytes);
        foreach (float[] values in data)
            foreach (float v in values) writer.Write(v);
    }
}
