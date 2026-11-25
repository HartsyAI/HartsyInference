using System.Text.Json;

namespace HartsyInference.LLM.Glm5Next;

/// <summary>Reads a <c>glm5_next</c> <c>config.json</c> into <see cref="Glm5NextConfig"/>. Strict: a missing field, a wrong model type or a
/// variant the engine does not run fails here, by name, before any weight is read.</summary>
public static class Glm5NextHfConfig
{
    /// <summary>Parses the checkpoint's <c>config.json</c> text.</summary>
    /// <exception cref="InvalidDataException">The config is not a <c>glm5_next</c> decoder the engine can build.</exception>
    public static Glm5NextConfig Parse(string json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        JsonElement root = doc.RootElement;
        RequireString(root, "model_type", "glm5_next");
        JsonElement t = Object(root, "text_config");
        RequireString(t, "model_type", "glm5_next_text");
        RequireString(t, "hidden_act", "silu");
        RequireString(t, "scoring_func", "sigmoid");
        RequireString(t, "topk_method", "noaux_tc");
        JsonElement lin = Object(t, "linear_attn_config");
        bool hasVision = root.TryGetProperty("vision_config", out _);

        int layers = Int(t, "num_hidden_layers");
        int heads = Int(t, "num_attention_heads");
        if (Int(t, "num_key_value_heads") != heads) throw new InvalidDataException("The decoder expects one key/value head per attention head.");
        if (Int(lin, "num_heads") != heads) throw new InvalidDataException("linear_attn_config.num_heads must equal num_attention_heads.");
        if (Int(t, "qk_rope_head_dim") != 0) throw new InvalidDataException("The DSA layers are NoPE: qk_rope_head_dim must be 0.");
        if (!Bool(t, "mla_use_nope")) throw new InvalidDataException("mla_use_nope must be true.");
        if (!Bool(t, "index_kpool_compress")) throw new InvalidDataException("The indexer runs on compressed key pools: index_kpool_compress must be true.");
        if (Int(t, "n_group") != 1 || Int(t, "topk_group") != 1) throw new InvalidDataException("Grouped expert routing is not used by this model: n_group and topk_group must be 1.");
        int topK = Int(t, "index_topk"), pool = Int(t, "index_kpool");
        if (topK < 1 || pool < 1 || topK % pool != 0) throw new InvalidDataException($"index_topk ({topK}) must be a positive multiple of index_kpool ({pool}).");
        int qkNope = Int(t, "qk_nope_head_dim");
        if (Int(t, "qk_head_dim") != qkNope) throw new InvalidDataException("qk_head_dim must equal qk_nope_head_dim for a NoPE layer.");

        Glm5NextAttentionKind[] kinds = LayerKinds(t, lin, layers);
        JsonElement indexerTypes = Array(t, "indexer_types");
        if (indexerTypes.GetArrayLength() != layers) throw new InvalidDataException("indexer_types must name every decoder layer.");
        foreach (JsonElement v in indexerTypes.EnumerateArray())
            if (v.GetString() != "full") throw new InvalidDataException("Shared indexer layers (indexer_types 'shared') are not run by this engine; no released checkpoint uses them.");
        int mtpLayers = Int(t, "num_nextn_predict_layers");
        if (mtpLayers < 0) throw new InvalidDataException("num_nextn_predict_layers must not be negative.");

        int[] eos = Array(t, "eos_token_id").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        return new Glm5NextConfig
        {
            Hidden = Int(t, "hidden_size"),
            Vocab = Int(t, "vocab_size"),
            Layers = layers,
            MtpLayers = mtpLayers,
            Heads = heads,
            LayerKinds = kinds,
            SparseMlp = MlpKinds(t, layers),
            DenseIntermediate = Int(t, "intermediate_size"),
            SwigluLimit = Float(t, "swiglu_limit"),
            KdaHeadDim = Int(lin, "head_dim"),
            ConvKernel = Int(lin, "short_conv_kernel_size"),
            GateLowerBound = Float(lin, "gate_lower_bound"),
            QLoraRank = Int(t, "q_lora_rank"),
            KvLoraRank = Int(t, "kv_lora_rank"),
            QkDim = qkNope,
            VDim = Int(t, "v_head_dim"),
            IndexHeads = Int(t, "index_n_heads"),
            IndexHeadDim = Int(t, "index_head_dim"),
            IndexTopK = topK,
            IndexPool = pool,
            IndexAlwaysSelectTail = Bool(t, "index_kpool_always_select_tail"),
            Experts = Int(t, "n_routed_experts"),
            ExpertsPerToken = Int(t, "num_experts_per_tok"),
            MoeIntermediate = Int(t, "moe_intermediate_size"),
            SharedExperts = Int(t, "n_shared_experts"),
            RoutedScalingFactor = Float(t, "routed_scaling_factor"),
            NormTopKProb = Bool(t, "norm_topk_prob"),
            HcMult = Int(t, "hc_mult"),
            HcEps = Float(t, "hc_eps"),
            HcSinkhornIters = Int(t, "hc_sinkhorn_iters"),
            RmsNormEps = Float(t, "rms_norm_eps"),
            EosTokenIds = eos,
            PadTokenId = Int(t, "pad_token_id"),
            Vision = hasVision ? ParseVision(root.GetProperty("vision_config")) : null,
            // A vision tower needs its placeholder ids; text-only configs have none.
            ImageTokenId = hasVision ? Int(root, "image_token_id") : IntOr(root, "image_token_id"),
            VideoTokenId = hasVision ? Int(root, "video_token_id") : IntOr(root, "video_token_id"),
        };
    }

    /// <summary>Parses only the <c>vision_config</c> object of a <c>config.json</c> (a vision-tower test or tool needs no text config).</summary>
    /// <exception cref="InvalidDataException">The config has no vision tower or it is not a <c>glm5_next_vision</c> the engine can build.</exception>
    public static Glm5NextVisionConfig ParseVision(string json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        return doc.RootElement.TryGetProperty("vision_config", out JsonElement v) ? ParseVision(v) : throw new InvalidDataException("config.json has no vision_config.");
    }

    private static Glm5NextVisionConfig ParseVision(JsonElement v)
    {
        RequireString(v, "model_type", "glm5_next_vision");
        RequireString(v, "hidden_act", "silu");
        Glm5NextVisionConfig config = new()
        {
            Depth = Int(v, "depth"),
            Hidden = Int(v, "hidden_size"),
            Heads = Int(v, "num_heads"),
            Intermediate = Int(v, "intermediate_size"),
            PatchSize = Int(v, "patch_size"),
            TemporalPatchSize = Int(v, "temporal_patch_size"),
            SpatialMergeSize = Int(v, "spatial_merge_size"),
            OutHidden = Int(v, "out_hidden_size"),
            MergerIntermediate = Int(v, "projection_intermediate_size"),
            RmsNormEps = Float(v, "rms_norm_eps"),
            SwigluLimit = Float(v, "swiglu_limit"),
        };
        config.Validate();
        return config;
    }

    // layer_types spells DSA layers deepseek_sparse_attention (transformers 5.16-5.18), indexed_attention (5.19+) or full_attention; each
    // must agree with linear_attn_config's kda_layers and full_attn_layers when those are present.
    private static Glm5NextAttentionKind[] LayerKinds(JsonElement t, JsonElement lin, int layers)
    {
        JsonElement types = Array(t, "layer_types");
        if (types.GetArrayLength() != layers) throw new InvalidDataException("layer_types must name every decoder layer.");
        Glm5NextAttentionKind[] kinds = new Glm5NextAttentionKind[layers];
        int i = 0;
        foreach (JsonElement e in types.EnumerateArray())
        {
            string? s = e.GetString();
            kinds[i++] = s switch
            {
                "linear_attention" => Glm5NextAttentionKind.Kda,
                // transformers' configuration_glm5_next.py (52bac5d0, L166-168) maps a 'full_attention' layer to 'indexed_attention', so it is the DSA spelling too.
                "deepseek_sparse_attention" or "indexed_attention" or "full_attention" => Glm5NextAttentionKind.Dsa,
                _ => throw new InvalidDataException($"layer_types has the unknown layer type '{s}'."),
            };
        }
        CheckSchedule(lin, "kda_layers", kinds, Glm5NextAttentionKind.Kda);
        CheckSchedule(lin, "full_attn_layers", kinds, Glm5NextAttentionKind.Dsa);
        return kinds;
    }

    private static void CheckSchedule(JsonElement lin, string name, Glm5NextAttentionKind[] kinds, Glm5NextAttentionKind expected)
    {
        if (!lin.TryGetProperty(name, out JsonElement arr) || arr.ValueKind != JsonValueKind.Array) return;
        HashSet<int> listed = arr.EnumerateArray().Select(e => e.GetInt32()).ToHashSet();
        for (int l = 0; l < kinds.Length; l++)
            if ((kinds[l] == expected) != listed.Contains(l))
                throw new InvalidDataException($"linear_attn_config.{name} disagrees with layer_types at layer {l}.");
    }

    private static bool[] MlpKinds(JsonElement t, int layers)
    {
        JsonElement types = Array(t, "mlp_layer_types");
        if (types.GetArrayLength() != layers) throw new InvalidDataException("mlp_layer_types must name every decoder layer.");
        bool[] sparse = new bool[layers];
        int i = 0;
        foreach (JsonElement e in types.EnumerateArray())
        {
            string? s = e.GetString();
            if (s != "dense" && s != "sparse") throw new InvalidDataException($"mlp_layer_types has the unknown kind '{s}'.");
            sparse[i++] = s == "sparse";
        }
        return sparse;
    }

    private static JsonElement Object(JsonElement e, string name) =>
        e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Object
            ? v : throw new InvalidDataException($"config.json is missing the object '{name}'.");

    private static JsonElement Array(JsonElement e, string name) =>
        e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Array
            ? v : throw new InvalidDataException($"config.json is missing the array '{name}'.");

    private static int Int(JsonElement e, string name) =>
        e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number && v.TryGetInt32(out int i)
            ? i : throw new InvalidDataException($"config.json is missing the integer '{name}'.");

    private static int IntOr(JsonElement e, string name) =>
        e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number && v.TryGetInt32(out int i) ? i : 0;

    private static float Float(JsonElement e, string name) =>
        e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number
            ? (float)v.GetDouble() : throw new InvalidDataException($"config.json is missing the number '{name}'.");

    private static bool Bool(JsonElement e, string name) =>
        e.TryGetProperty(name, out JsonElement v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False
            ? v.GetBoolean() : throw new InvalidDataException($"config.json is missing the boolean '{name}'.");

    private static void RequireString(JsonElement e, string name, string expected)
    {
        string? actual = e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
        if (actual != expected) throw new InvalidDataException($"config.json '{name}' is '{actual ?? "missing"}'; the glm5_next decoder needs '{expected}'.");
    }
}
