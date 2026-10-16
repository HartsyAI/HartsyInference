using System.Text.Json;

namespace HartsyInference.LLM.Ssm;

/// <summary>The <c>text_config</c> of a HuggingFace Qwen3.5 checkpoint, as far as the text trunk needs it: the dense
/// tier, and the MoE tier (<c>qwen3_5_moe_text</c>) when <see cref="NumExperts"/> is set.</summary>
public sealed record Qwen35HfConfig
{
    public required int HiddenSize { get; init; }

    public required int NumLayers { get; init; }

    public required int VocabSize { get; init; }

    public required int NumHeads { get; init; }

    public required int NumKvHeads { get; init; }

    public required int HeadDim { get; init; }

    /// <summary>Every this-many-th layer is full attention; 0 when <see cref="LinearLayers"/> names each layer instead.</summary>
    public required int FullAttentionInterval { get; init; }

    public required int LinearConvKernel { get; init; }

    public required int LinearKeyHeadDim { get; init; }

    public required int LinearValueHeadDim { get; init; }

    public required int LinearNumKeyHeads { get; init; }

    public required int LinearNumValueHeads { get; init; }

    public float PartialRotaryFactor { get; init; } = 0.25f;

    public float RopeTheta { get; init; } = 10_000_000f;

    public float RmsNormEps { get; init; } = 1e-6f;

    /// <summary>Per-layer attention kind from <c>layer_types</c> (true = Gated DeltaNet); null derives it from <see cref="FullAttentionInterval"/>.</summary>
    public IReadOnlyList<bool>? LinearLayers { get; init; }

    /// <summary>Routed experts per MoE layer; 0 for the dense tier.</summary>
    public int NumExperts { get; init; }

    public int NumExpertsPerTok { get; init; }

    public int MoeIntermediateSize { get; init; }

    public int SharedExpertIntermediateSize { get; init; }

    /// <summary>Renormalizes the selected experts' softmax weights to sum to one.</summary>
    public bool NormTopKProb { get; init; } = true;

    /// <summary>True when the output head reuses the token embedding instead of a separate <c>lm_head</c>.</summary>
    public bool TieWordEmbeddings { get; init; } = true;

    public int MaxPositionEmbeddings { get; init; } = 262_144;

    /// <summary>True for the MoE tier.</summary>
    public bool IsMoe => NumExperts > 0;

    /// <summary>Width of the partial rotary embedding.</summary>
    public int RotaryDim => (int)(HeadDim * PartialRotaryFactor);

    /// <summary>True when layer <paramref name="layer"/> is a Gated DeltaNet (linear-attention) layer.</summary>
    public bool IsLinearLayer(int layer) => LinearLayers is not null ? LinearLayers[layer] : (layer + 1) % FullAttentionInterval != 0;

    /// <summary>Reads a checkpoint's <c>config.json</c> (a conditional-generation wrapper with a nested <c>text_config</c>, or a bare text config).</summary>
    public static Qwen35HfConfig FromJson(string json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        JsonElement root = doc.RootElement;
        return FromTextConfig(root.TryGetProperty("text_config", out JsonElement nested) ? nested : root);
    }

    /// <summary>Reads a parsed <c>text_config</c> object.</summary>
    public static Qwen35HfConfig FromTextConfig(JsonElement t)
    {
        float theta = 10_000_000f, partial = 0.25f;
        if (t.TryGetProperty("rope_parameters", out JsonElement rope))
        {
            if (rope.TryGetProperty("rope_theta", out JsonElement th)) theta = (float)th.GetDouble();
            if (rope.TryGetProperty("partial_rotary_factor", out JsonElement pf)) partial = (float)pf.GetDouble();
        }
        else
        {
            if (t.TryGetProperty("rope_theta", out JsonElement th)) theta = (float)th.GetDouble();
            if (t.TryGetProperty("partial_rotary_factor", out JsonElement pf2)) partial = (float)pf2.GetDouble();
        }
        int layers = t.GetProperty("num_hidden_layers").GetInt32();
        bool[]? linear = null;
        if (t.TryGetProperty("layer_types", out JsonElement types) && types.ValueKind == JsonValueKind.Array)
        {
            if (types.GetArrayLength() != layers)
            {
                throw new InvalidDataException($"layer_types names {types.GetArrayLength()} layers; num_hidden_layers is {layers}.");
            }
            linear = new bool[layers];
            int i = 0;
            foreach (JsonElement type in types.EnumerateArray())
            {
                linear[i++] = type.GetString() switch
                {
                    "linear_attention" => true,
                    "full_attention" => false,
                    string other => throw new InvalidDataException($"Unknown Qwen3.5 layer type '{other}'."),
                    null => throw new InvalidDataException("layer_types holds a null entry."),
                };
            }
        }
        int interval = t.TryGetProperty("full_attention_interval", out JsonElement fai) ? fai.GetInt32() : 0;
        if (linear is null && interval <= 0)
        {
            throw new InvalidDataException("A Qwen3.5 config needs layer_types or full_attention_interval.");
        }
        return new Qwen35HfConfig
        {
            HiddenSize = t.GetProperty("hidden_size").GetInt32(), NumLayers = layers,
            VocabSize = t.GetProperty("vocab_size").GetInt32(), NumHeads = t.GetProperty("num_attention_heads").GetInt32(),
            NumKvHeads = t.GetProperty("num_key_value_heads").GetInt32(), HeadDim = t.GetProperty("head_dim").GetInt32(),
            FullAttentionInterval = interval, LinearLayers = linear,
            LinearConvKernel = t.GetProperty("linear_conv_kernel_dim").GetInt32(),
            LinearKeyHeadDim = t.GetProperty("linear_key_head_dim").GetInt32(),
            LinearValueHeadDim = t.GetProperty("linear_value_head_dim").GetInt32(),
            LinearNumKeyHeads = t.GetProperty("linear_num_key_heads").GetInt32(),
            LinearNumValueHeads = t.GetProperty("linear_num_value_heads").GetInt32(),
            PartialRotaryFactor = partial, RopeTheta = theta,
            RmsNormEps = t.TryGetProperty("rms_norm_eps", out JsonElement eps) ? (float)eps.GetDouble() : 1e-6f,
            NumExperts = Int(t, "num_experts", 0), NumExpertsPerTok = Int(t, "num_experts_per_tok", 0),
            MoeIntermediateSize = Int(t, "moe_intermediate_size", 0),
            SharedExpertIntermediateSize = Int(t, "shared_expert_intermediate_size", 0),
            NormTopKProb = !t.TryGetProperty("norm_topk_prob", out JsonElement norm) || norm.GetBoolean(),
            TieWordEmbeddings = !t.TryGetProperty("tie_word_embeddings", out JsonElement tie) || tie.GetBoolean(),
            MaxPositionEmbeddings = Int(t, "max_position_embeddings", 262_144),
        };
    }

    private static int Int(JsonElement t, string name, int fallback) =>
        t.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;
}
