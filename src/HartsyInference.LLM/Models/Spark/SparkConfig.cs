using System.Text.Json;
using HartsyInference.LLM.Models.Shared;

namespace HartsyInference.LLM.Models.Spark;

/// <summary>A HuggingFace Spark-X text config (<c>spark2_5</c>): dense GELU-gated decoder layers with fused QKV, a head-wise sigmoid output gate, and sliding or full attention per layer.</summary>
public sealed record SparkConfig
{
    public const string ModelType = "spark2_5";

    public required int HiddenSize { get; init; }

    public required int IntermediateSize { get; init; }

    public required int VocabSize { get; init; }

    public required int NumHeads { get; init; }

    public required int NumKvHeads { get; init; }

    public required int HeadDim { get; init; }

    public required float NormEps { get; init; }

    public required int SlidingWindow { get; init; }

    public required bool TieWordEmbeddings { get; init; }

    public required int MaxPositionEmbeddings { get; init; }

    /// <summary>True per layer for sliding attention, false for full attention.</summary>
    public required IReadOnlyList<bool> SlidingLayers { get; init; }

    public required LayerRopeParameters FullRope { get; init; }

    public required LayerRopeParameters SlidingRope { get; init; }

    public int BosTokenId { get; init; }

    public int EosTokenId { get; init; }

    public int NumLayers => SlidingLayers.Count;

    /// <summary>RoPE of attention layer <paramref name="layer"/>.</summary>
    public LayerRopeParameters RopeOf(int layer) => SlidingLayers[layer] ? SlidingRope : FullRope;

    /// <summary>Reads a checkpoint's <c>config.json</c>.</summary>
    /// <exception cref="NotSupportedException">The config is not Spark-X or uses a feature the engine does not run.</exception>
    /// <exception cref="InvalidDataException">The config is inconsistent.</exception>
    public static SparkConfig FromJson(string json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        JsonElement t = doc.RootElement;
        string modelType = t.TryGetProperty("model_type", out JsonElement mt) ? mt.GetString() ?? "" : "";
        if (!string.Equals(modelType, ModelType, StringComparison.Ordinal))
        {
            throw new NotSupportedException($"config.json model_type is '{modelType}', not {ModelType}.");
        }
        RequireString(t, "hidden_act", "gelu");
        if (!Bool(t, "headwise_attn_output_gate", true))
        {
            throw new NotSupportedException("Spark-X without headwise_attn_output_gate is not supported.");
        }
        RequireString(t, "gate_attn_act_mode", "sigmoid");
        if (Bool(t, "attention_bias", false) || Bool(t, "mlp_bias", false))
        {
            throw new NotSupportedException("Spark-X attention_bias and mlp_bias are not supported; the released checkpoints have none.");
        }
        int layerCount = t.GetProperty("num_hidden_layers").GetInt32();
        JsonElement types = t.GetProperty("layer_types");
        if (types.GetArrayLength() != layerCount)
        {
            throw new InvalidDataException($"layer_types names {types.GetArrayLength()} layers; num_hidden_layers is {layerCount}.");
        }
        List<bool> sliding = [];
        foreach (JsonElement type in types.EnumerateArray())
        {
            sliding.Add(type.GetString() switch
            {
                "sliding_attention" => true,
                "full_attention" => false,
                string other => throw new NotSupportedException($"Spark-X layer type '{other}' is not supported."),
                null => throw new InvalidDataException("layer_types holds a null entry."),
            });
        }
        int heads = t.GetProperty("num_attention_heads").GetInt32();
        int kvHeads = t.GetProperty("num_key_value_heads").GetInt32();
        if (heads % kvHeads != 0)
        {
            throw new InvalidDataException($"num_attention_heads {heads} is not a multiple of num_key_value_heads {kvHeads}.");
        }
        int window = t.TryGetProperty("sliding_window", out JsonElement sw) && sw.ValueKind == JsonValueKind.Number ? sw.GetInt32() : 0;
        if (sliding.Contains(true) && window <= 0)
        {
            throw new InvalidDataException("A sliding_attention layer needs a positive sliding_window.");
        }
        JsonElement rope = t.GetProperty("rope_parameters");
        return new SparkConfig
        {
            HiddenSize = t.GetProperty("hidden_size").GetInt32(),
            IntermediateSize = t.GetProperty("intermediate_size").GetInt32(),
            VocabSize = t.GetProperty("vocab_size").GetInt32(),
            NumHeads = heads,
            NumKvHeads = kvHeads,
            HeadDim = t.GetProperty("head_dim").GetInt32(),
            NormEps = (float)t.GetProperty("rms_norm_eps").GetDouble(),
            SlidingWindow = window,
            TieWordEmbeddings = Bool(t, "tie_word_embeddings", true),
            MaxPositionEmbeddings = t.GetProperty("max_position_embeddings").GetInt32(),
            SlidingLayers = sliding,
            FullRope = ReadRope(rope, sliding, "full_attention", false),
            SlidingRope = ReadRope(rope, sliding, "sliding_attention", true),
            BosTokenId = Int(t, "bos_token_id"),
            EosTokenId = Int(t, "eos_token_id"),
        };
    }

    private static LayerRopeParameters ReadRope(JsonElement rope, List<bool> sliding, string type, bool slidingType)
    {
        if (!sliding.Contains(slidingType))
        {
            return default;
        }
        if (!rope.TryGetProperty(type, out JsonElement scope))
        {
            throw new InvalidDataException($"rope_parameters has no '{type}' entry for a model that has {type} layers.");
        }
        float partial = scope.TryGetProperty("partial_rotary_factor", out JsonElement pf) ? (float)pf.GetDouble() : 1f;
        return new LayerRopeParameters((float)scope.GetProperty("rope_theta").GetDouble(), partial);
    }

    private static void RequireString(JsonElement e, string name, string expected)
    {
        string actual = e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.String ? v.GetString() ?? "" : expected;
        if (!string.Equals(actual, expected, StringComparison.Ordinal))
        {
            throw new NotSupportedException($"Spark-X {name} '{actual}' is not supported; only '{expected}' is.");
        }
    }

    private static bool Bool(JsonElement e, string name, bool fallback) =>
        e.TryGetProperty(name, out JsonElement v) && (v.ValueKind is JsonValueKind.True or JsonValueKind.False) ? v.GetBoolean() : fallback;

    /// <summary>A token id: a number, the first entry of a list (HF configs list several end ids), or 0 when absent.</summary>
    /// <exception cref="InvalidDataException">The field is neither a number nor a list that starts with one.</exception>
    private static int Int(JsonElement e, string name)
    {
        if (!e.TryGetProperty(name, out JsonElement v) || v.ValueKind == JsonValueKind.Null) return 0;
        if (v.ValueKind == JsonValueKind.Number) return v.GetInt32();
        if (v.ValueKind == JsonValueKind.Array && v.GetArrayLength() > 0 && v[0].ValueKind == JsonValueKind.Number) return v[0].GetInt32();
        throw new InvalidDataException($"config.json {name} must be a number or a list of numbers.");
    }
}
