using System.Globalization;
using System.Text;
using HartsyInference.LLM.Ernie5;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>Writes a tiny ERNIE-5.5 checkpoint directory (config, F32 safetensors with every key the model maps, byte-level tokenizer and chat
/// template) with seeded random weights. Layer 0 is an HCA attention with a dense FFN; layer 1 is a DSA attention with a latent MoE.</summary>
internal static class Ernie5SyntheticCheckpoint
{
    internal const int Vocab = 264, Hidden = 16, Hc = 2, Layers = 2, Heads = 2, HeadDim = 8, QLora = 8, OGroups = 1, OLoraRank = 8;
    internal const int MlaHeads = 2, MlaQLora = 8, MlaKvLora = 8, MlaNope = 4, MlaRope = 4, MlaV = 4, IndexHeads = 2, IndexHeadDim = 128;
    internal const int Dense = 12, MoeInter = 8, Latent = 6, Experts = 4, SharedExperts = 1, ExpertsPerTok = 2;
    internal const int BosId = 256, EosId = 257;

    /// <summary>The config as a checkpoint ships it; <paramref name="overrides"/> replaces or adds top-level fields (raw JSON values).</summary>
    internal static string ConfigJson(IReadOnlyDictionary<string, string>? overrides = null)
    {
        Dictionary<string, string> fields = new(StringComparer.Ordinal)
        {
            ["model_type"] = "\"ernie5_5_moe\"",
            ["vocab_size"] = Vocab.ToString(CultureInfo.InvariantCulture),
            ["hidden_size"] = Hidden.ToString(CultureInfo.InvariantCulture),
            ["num_hidden_layers"] = Layers.ToString(CultureInfo.InvariantCulture),
            ["num_nextn_predict_layers"] = "0",
            ["num_attention_heads"] = Heads.ToString(CultureInfo.InvariantCulture),
            ["num_key_value_heads"] = "1",
            ["head_dim"] = HeadDim.ToString(CultureInfo.InvariantCulture),
            ["qk_nope_head_dim"] = "4",
            ["qk_rope_head_dim"] = "4",
            ["v_head_dim"] = HeadDim.ToString(CultureInfo.InvariantCulture),
            ["q_lora_rank"] = QLora.ToString(CultureInfo.InvariantCulture),
            ["kv_lora_rank"] = "8",
            ["o_lora_rank"] = OLoraRank.ToString(CultureInfo.InvariantCulture),
            ["o_groups"] = OGroups.ToString(CultureInfo.InvariantCulture),
            ["hybrid_mla_num_attention_heads"] = MlaHeads.ToString(CultureInfo.InvariantCulture),
            ["hybrid_mla_q_lora_rank"] = MlaQLora.ToString(CultureInfo.InvariantCulture),
            ["hybrid_mla_kv_lora_rank"] = MlaKvLora.ToString(CultureInfo.InvariantCulture),
            ["hybrid_mla_qk_nope_head_dim"] = MlaNope.ToString(CultureInfo.InvariantCulture),
            ["hybrid_mla_qk_rope_head_dim"] = MlaRope.ToString(CultureInfo.InvariantCulture),
            ["hybrid_mla_v_head_dim"] = MlaV.ToString(CultureInfo.InvariantCulture),
            ["intermediate_size"] = Dense.ToString(CultureInfo.InvariantCulture),
            ["moe_intermediate_size"] = MoeInter.ToString(CultureInfo.InvariantCulture),
            ["moe_latent_size"] = Latent.ToString(CultureInfo.InvariantCulture),
            ["n_routed_experts"] = Experts.ToString(CultureInfo.InvariantCulture),
            ["n_shared_experts"] = SharedExperts.ToString(CultureInfo.InvariantCulture),
            ["num_experts_per_tok"] = ExpertsPerTok.ToString(CultureInfo.InvariantCulture),
            ["first_k_dense_replace"] = "1",
            ["routed_scaling_factor"] = "1.0",
            ["routed_scaling_factor_learnable"] = "false",
            ["moe_split_feature_routing"] = "false",
            ["scoring_func"] = "\"sigmoid\"",
            ["norm_topk_prob"] = "true",
            ["n_group"] = "1",
            ["topk_group"] = "1",
            ["moe_shared_expert_gate"] = "false",
            ["latent_moe_use_norm"] = "true",
            ["hidden_act"] = "\"situ\"",
            ["activation_situ_beta"] = "4.0",
            ["activation_situ_linear_beta"] = "25.0",
            ["compress_ratios"] = "[2, -2]",
            ["experimental_attention_variant"] = "\"dsv4_hybrid\"",
            ["use_gated_attn"] = "false",
            ["use_vha_attention"] = "false",
            ["rope_interleave"] = "true",
            ["enable_hyper_connections"] = "true",
            ["hc_mult"] = Hc.ToString(CultureInfo.InvariantCulture),
            ["hc_sinkhorn_iters"] = "3",
            ["hc_eps"] = "1e-6",
            ["index_n_heads"] = IndexHeads.ToString(CultureInfo.InvariantCulture),
            ["index_head_dim"] = IndexHeadDim.ToString(CultureInfo.InvariantCulture),
            ["index_topk"] = "4",
            ["sliding_window"] = "4",
            ["rope_theta"] = "10000.0",
            ["compress_rope_theta"] = "10000.0",
            ["rms_norm_eps"] = "1e-6",
            ["max_position_embeddings"] = "64",
            ["tie_word_embeddings"] = "false",
            ["bos_token_id"] = BosId.ToString(CultureInfo.InvariantCulture),
            ["eos_token_id"] = EosId.ToString(CultureInfo.InvariantCulture),
        };
        if (overrides is not null)
            foreach (KeyValuePair<string, string> kv in overrides) fields[kv.Key] = kv.Value;
        return "{" + string.Join(",\n", fields.Select(kv => $"\"{kv.Key}\": {kv.Value}")) + "}";
    }

    /// <summary>Writes a complete checkpoint into <paramref name="directory"/> (created if absent) with weights from <paramref name="seed"/>.</summary>
    internal static void Write(string directory, int seed = 7)
    {
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, "config.json"), ConfigJson());
        File.WriteAllText(Path.Combine(directory, "tokenizer.json"), TokenizerJson());
        File.WriteAllText(Path.Combine(directory, "tokenizer_config.json"), "{\"bos_token\": \"<s>\", \"eos_token\": \"</s>\"}");
        File.WriteAllText(Path.Combine(directory, "chat_template.jinja"), "{% for message in messages %}{{ message['content'] }}{% endfor %}");
        Ernie5Config config = Ernie5Config.Parse(ConfigJson());
        WriteSafetensors(Path.Combine(directory, "model.safetensors"), config, seed);
    }

    /// <summary>A byte-level BPE tokenizer: ids 0..255 are the bytes, then the begin and end specials.</summary>
    internal static string TokenizerJson()
    {
        Dictionary<int, char> map = ByteToUnicode();
        StringBuilder vocab = new();
        for (int b = 0; b < 256; b++)
        {
            if (b > 0) vocab.Append(',');
            vocab.Append(System.Text.Json.JsonSerializer.Serialize(map[b].ToString())).Append(':').Append(b);
        }
        // one merge, so the tokenizer is a BPE and not a bare alphabet: "h" + "i" -> "hi" (id 258)
        vocab.Append(",\"hi\":258");
        return "{\"version\":\"1.0\",\"model\":{\"type\":\"BPE\",\"vocab\":{" + vocab + "},\"merges\":[\"h i\"]},"
            + "\"added_tokens\":["
            + $"{{\"id\":{BosId},\"content\":\"<s>\",\"special\":true}},{{\"id\":{EosId},\"content\":\"</s>\",\"special\":true}}]}}";
    }

    // The GPT-2 byte-level alphabet: printable bytes map to themselves, the rest to code points above 255.
    private static Dictionary<int, char> ByteToUnicode()
    {
        List<int> bytes = [];
        for (int b = 33; b <= 126; b++) bytes.Add(b);
        for (int b = 161; b <= 172; b++) bytes.Add(b);
        for (int b = 174; b <= 255; b++) bytes.Add(b);
        List<int> codes = [.. bytes];
        int extra = 0;
        for (int b = 0; b < 256; b++)
        {
            if (bytes.Contains(b)) continue;
            bytes.Add(b);
            codes.Add(256 + extra++);
        }
        Dictionary<int, char> map = new();
        for (int i = 0; i < bytes.Count; i++) map[bytes[i]] = (char)codes[i];
        return map;
    }

    private static void WriteSafetensors(string path, Ernie5Config config, int seed)
    {
        Random rng = new(seed);
        List<(string Name, int[] Shape, float[] Values)> tensors = [];
        foreach (string key in Ernie5WeightMap.ExpectedKeys(config))
        {
            int[] shape = ShapeOf(key, config);
            int count = shape.Aggregate(1, (a, n) => a * n);
            tensors.Add((key, shape, ValuesOf(key, count, rng)));
        }
        tensors.Sort(static (a, b) => string.CompareOrdinal(a.Name, b.Name));

        StringBuilder header = new("{");
        long offset = 0;
        for (int i = 0; i < tensors.Count; i++)
        {
            (string name, int[] shape, float[] values) = tensors[i];
            long bytes = (long)values.Length * sizeof(float);
            if (i > 0) header.Append(',');
            header.Append($"\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{string.Join(",", shape)}],\"data_offsets\":[{offset},{offset + bytes}]}}");
            offset += bytes;
        }
        header.Append('}');
        // the header is padded with spaces to an 8-byte boundary, as the format recommends
        while (header.Length % 8 != 0) header.Append(' ');
        byte[] headerBytes = Encoding.UTF8.GetBytes(header.ToString());

        using FileStream stream = File.Create(path);
        using BinaryWriter writer = new(stream);
        writer.Write((long)headerBytes.Length);
        writer.Write(headerBytes);
        foreach ((string _, int[] _, float[] values) in tensors)
            foreach (float v in values) writer.Write(v);
    }

    private static float[] ValuesOf(string key, int count, Random rng)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
        {
            if (key.EndsWith("norm.weight", StringComparison.Ordinal)) values[i] = 1f;
            else if (key.EndsWith("norm.bias", StringComparison.Ordinal) || key.EndsWith(".bias", StringComparison.Ordinal)
                || key.EndsWith("attn_sink", StringComparison.Ordinal) || key.EndsWith("softmax_offset", StringComparison.Ordinal)
                || key.EndsWith("e_score_correction_bias", StringComparison.Ordinal) || key.EndsWith("hc_head_base", StringComparison.Ordinal)) values[i] = 0f;
            else if (key.EndsWith("hc_head_scale", StringComparison.Ordinal)) values[i] = 1f;
            else if (key.Contains("_hyper_connection.alpha_", StringComparison.Ordinal)) values[i] = 0.05f;
            else values[i] = (float)((rng.NextDouble() * 2 - 1) * 0.05);
        }
        return values;
    }

    private static int[] ShapeOf(string key, Ernie5Config c)
    {
        int hc = c.HcMult, mixes = (2 + hc) * hc, flat = hc * c.HiddenSize, h = c.HiddenSize;
        if (key == Ernie5WeightNames.Embedding || key == Ernie5WeightNames.LmHead) return [c.VocabSize, h];
        if (key == Ernie5WeightNames.FinalNorm) return [h];
        if (key == Ernie5WeightNames.HeadFn) return [flat, hc];
        if (key == Ernie5WeightNames.HeadBase) return [hc];
        if (key == Ernie5WeightNames.HeadScale) return [1];
        if (key.Contains("_hyper_connection.mapping_proj.weight", StringComparison.Ordinal)) return [flat, mixes];
        if (key.Contains("_hyper_connection.bias", StringComparison.Ordinal)) return [mixes];
        if (key.Contains("_hyper_connection.alpha_", StringComparison.Ordinal)) return [1];
        if (key.EndsWith("input_layernorm.weight", StringComparison.Ordinal) || key.EndsWith("post_attention_layernorm.weight", StringComparison.Ordinal))
            return [h];

        int layer = LayerOf(key);
        string leaf = key[(key.IndexOf('.', "model.layers.".Length) + 1)..];
        if (leaf.StartsWith("self_attn.", StringComparison.Ordinal))
            return c.IsDsaLayer(layer) ? DsaShape(leaf, c) : HcaShape(leaf, c);
        if (leaf.StartsWith("mlp.", StringComparison.Ordinal) && layer < c.FirstKDenseReplace)
            return leaf.StartsWith("mlp.down_proj", StringComparison.Ordinal) ? [h, Dense] : [Dense, h];
        return MoeShape(leaf, c);
    }

    private static int[] HcaShape(string leaf, Ernie5Config c)
    {
        int grouped = c.OGroups * c.OLoraRank, groupDim = c.NumAttentionHeads * c.HeadDim / c.OGroups, ratio = c.CompressRatios[0];
        return leaf switch
        {
            "self_attn.linear_q_down_proj.weight" => [c.QLoraRank, c.HiddenSize],
            "self_attn.q_layernorm.weight" => [c.QLoraRank],
            "self_attn.linear_q_up_proj.weight" => [c.NumAttentionHeads * c.HeadDim, c.QLoraRank],
            "self_attn.linear_kv_proj.weight" => [c.HeadDim, c.HiddenSize],
            "self_attn.kv_layernorm.weight" => [c.HeadDim],
            "self_attn.linear_o_group_proj" => [grouped * groupDim],
            "self_attn.o_proj.weight" => [c.HiddenSize, grouped],
            "self_attn.core_attention.attn_sink" => [c.NumAttentionHeads],
            "self_attn.core_attention.compressor.linear_wkv.weight" => [c.HeadDim, c.HiddenSize],
            "self_attn.core_attention.compressor.linear_wgate.weight" => [c.HeadDim, c.HiddenSize],
            "self_attn.core_attention.compressor.ape" => [ratio * c.HeadDim],
            "self_attn.core_attention.compressor.norm.weight" => [c.HeadDim],
            _ => throw new InvalidOperationException($"No HCA shape for '{leaf}'."),
        };
    }

    private static int[] DsaShape(string leaf, Ernie5Config c)
    {
        int heads = c.MlaHeads, qk = c.MlaQkNopeHeadDim + c.MlaQkRopeHeadDim, latent = c.MlaKvLoraRank + c.MlaQkRopeHeadDim;
        return leaf switch
        {
            "self_attn.q_a_proj.weight" => [c.MlaQLoraRank, c.HiddenSize],
            "self_attn.q_a_layernorm.weight" => [c.MlaQLoraRank],
            "self_attn.q_b_proj.weight" => [heads * qk, c.MlaQLoraRank],
            "self_attn.kv_a_proj_with_mqa.weight" => [latent, c.HiddenSize],
            "self_attn.kv_a_layernorm.weight" => [c.MlaKvLoraRank],
            "self_attn.kv_b_proj.weight" => [heads * (c.MlaQkNopeHeadDim + c.MlaVHeadDim), c.MlaKvLoraRank],
            "self_attn.o_proj.weight" => [c.HiddenSize, heads * c.MlaVHeadDim],
            "self_attn.core_attention.softmax_offset" => [heads],
            "self_attn.core_attention.indexer.wq_b.weight" => [c.IndexNHeads * c.IndexHeadDim, c.MlaQLoraRank],
            "self_attn.core_attention.indexer.wk.weight" => [c.IndexHeadDim, c.HiddenSize],
            "self_attn.core_attention.indexer.k_norm.weight" => [c.IndexHeadDim],
            "self_attn.core_attention.indexer.k_norm.bias" => [c.IndexHeadDim],
            "self_attn.core_attention.indexer.weights_proj.weight" => [c.IndexNHeads, c.HiddenSize],
            _ => throw new InvalidOperationException($"No DSA shape for '{leaf}'."),
        };
    }

    private static int[] MoeShape(string leaf, Ernie5Config c)
    {
        int shared = c.NSharedExperts * c.MoeIntermediateSize, latent = c.MoeLatentSize;
        if (leaf == "block_sparse_moe.gate.weight") return [c.NRoutedExperts, c.HiddenSize];
        if (leaf == "block_sparse_moe.e_score_correction_bias") return [c.NRoutedExperts];
        if (leaf == "block_sparse_moe.fc1_latent_proj.weight") return [latent, c.HiddenSize];
        if (leaf == "block_sparse_moe.latent_norm.weight") return [latent];
        if (leaf == "block_sparse_moe.fc2_latent_proj.weight") return [c.HiddenSize, latent];
        if (leaf.StartsWith("block_sparse_moe.shared_experts.", StringComparison.Ordinal))
            return leaf.Contains(".w2.", StringComparison.Ordinal) ? [c.HiddenSize, shared] : [shared, c.HiddenSize];
        if (leaf.StartsWith("block_sparse_moe.experts.", StringComparison.Ordinal))
            return leaf.Contains(".w2.", StringComparison.Ordinal) ? [latent, c.MoeIntermediateSize] : [c.MoeIntermediateSize, latent];
        throw new InvalidOperationException($"No MoE shape for '{leaf}'.");
    }

    private static int LayerOf(string key)
    {
        const string prefix = "model.layers.";
        int end = key.IndexOf('.', prefix.Length);
        return int.Parse(key.AsSpan(prefix.Length, end - prefix.Length), NumberStyles.None, CultureInfo.InvariantCulture);
    }
}
