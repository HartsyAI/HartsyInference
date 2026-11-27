using System.Text.Json;

namespace HartsyInference.LLM.Kimi;

/// <summary>Reads a Kimi K3.1 <c>config.json</c> into <see cref="KimiK31Config"/>. Strict: a missing field, a wrong model type or an
/// activation the engine does not run fails here, by name, before any weight is read.</summary>
public static class KimiK31HfConfig
{
    /// <summary>Parses the checkpoint's <c>config.json</c> text.</summary>
    /// <exception cref="InvalidDataException">The config is not a Kimi K3.1 text decoder the engine can build.</exception>
    public static KimiK31Config Parse(string json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        JsonElement root = doc.RootElement;
        RequireString(root, "model_type", "kimi_k31");
        JsonElement t = Object(root, "text_config");
        RequireString(t, "model_type", "kimi_linear");
        RequireString(t, "hidden_act", "situ");
        JsonElement lin = Object(t, "linear_attn_config");
        JsonElement dsa = Object(t, "dsa_config");

        int heads = Int(t, "num_attention_heads");
        int kdaHeads = Int(lin, "num_heads");
        if (kdaHeads != heads) throw new InvalidDataException($"linear_attn_config.num_heads ({kdaHeads}) must equal num_attention_heads ({heads}).");
        if (Int(t, "num_key_value_heads") != heads) throw new InvalidDataException("The decoder expects one key/value head per attention head.");

        return new KimiK31Config
        {
            Hidden = Int(t, "hidden_size"),
            Vocab = Int(t, "vocab_size"),
            Layers = Int(t, "num_hidden_layers"),
            DenseLayers = Int(t, "first_k_dense_replace"),
            DenseIntermediate = Int(t, "intermediate_size"),
            Heads = heads,
            KdaHeadDim = Int(lin, "head_dim"),
            // The forget projection's rank is not in config.json; the loader checks f_a_proj's shape against it.
            KdaForgetRank = 128,
            ConvKernel = Int(lin, "short_conv_kernel_size"),
            GateLowerBound = Float(lin, "gate_lower_bound"),
            FullAttentionLayers = OneBased(lin, "full_attn_layers"),
            QLoraRank = Int(t, "q_lora_rank"),
            KvLoraRank = Int(t, "kv_lora_rank"),
            QkNopeDim = Int(t, "qk_nope_head_dim"),
            QkRopeDim = Int(t, "qk_rope_head_dim"),
            VDim = Int(t, "v_head_dim"),
            IndexHeads = Int(dsa, "index_num_heads"),
            IndexHeadDim = Int(dsa, "index_head_dim"),
            IndexTopK = Int(dsa, "index_topk"),
            Experts = Int(t, "num_experts"),
            ExpertsPerToken = Int(t, "num_experts_per_token"),
            MoeIntermediate = Int(t, "moe_intermediate_size"),
            SharedExperts = Int(t, "num_shared_experts"),
            LatentSize = Int(t, "routed_expert_hidden_size"),
            SituBeta = Float(t, "activation_situ_beta"),
            SituLinearBeta = Float(t, "activation_situ_linear_beta"),
            AttnResBlock = Int(t, "attn_res_block_size"),
            RmsNormEps = Float(t, "rms_norm_eps"),
        };
    }

    /// <summary>Whether the checkpoint's routed experts are MXFP4-packed (<c>quantization_config.format</c>).</summary>
    public static bool RoutedExpertsAreMxfp4(string json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        return doc.RootElement.TryGetProperty("quantization_config", out JsonElement q)
            && q.TryGetProperty("format", out JsonElement f) && f.GetString() == "mxfp4-pack-quantized";
    }

    private static JsonElement Object(JsonElement e, string name) =>
        e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Object
            ? v : throw new InvalidDataException($"config.json is missing the object '{name}'.");

    private static int Int(JsonElement e, string name) =>
        e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number && v.TryGetInt32(out int i)
            ? i : throw new InvalidDataException($"config.json is missing the integer '{name}'.");

    private static float Float(JsonElement e, string name) =>
        e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number
            ? (float)v.GetDouble() : throw new InvalidDataException($"config.json is missing the number '{name}'.");

    private static void RequireString(JsonElement e, string name, string expected)
    {
        string? actual = e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
        if (actual != expected) throw new InvalidDataException($"config.json '{name}' is '{actual ?? "missing"}'; the Kimi K3.1 decoder needs '{expected}'.");
    }

    // The checkpoint lists full-attention layers 1-based; the decoder indexes layers from 0.
    private static HashSet<int> OneBased(JsonElement e, string name)
    {
        if (!e.TryGetProperty(name, out JsonElement arr) || arr.ValueKind != JsonValueKind.Array)
            throw new InvalidDataException($"config.json is missing the array '{name}'.");
        HashSet<int> set = new();
        foreach (JsonElement v in arr.EnumerateArray()) set.Add(v.GetInt32() - 1);
        return set;
    }
}
