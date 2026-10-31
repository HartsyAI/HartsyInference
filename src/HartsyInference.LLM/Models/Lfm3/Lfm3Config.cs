using System.Text.Json;

namespace HartsyInference.LLM.Models.Lfm3;

/// <summary>The kind of token mixer a LFM3 decoder layer runs.</summary>
public enum Lfm3LayerKind
{
    /// <summary>Gated short convolution with a carried window.</summary>
    Conv,

    /// <summary>Causal attention over the whole sequence.</summary>
    FullAttention,

    /// <summary>Causal attention limited to the last <see cref="Lfm3Config.SlidingWindow"/> tokens.</summary>
    SlidingAttention,
}

/// <summary>RoPE of one attention layer type: the base and the fraction of the head dimension that rotates.</summary>
public readonly record struct Lfm3RopeParameters(float Theta, float PartialRotaryFactor)
{
    /// <summary>Rotated frequency pairs; the remaining pairs have zero frequency and leave their dimensions unrotated.</summary>
    public int RotatedPairs(int headDim) => (int)(PartialRotaryFactor * headDim) / 2;
}

/// <summary>A HuggingFace LFM3 text config (<c>lfm3</c>): a stack of gated short-conv, full-attention and sliding-attention layers with sandwich norms.</summary>
public sealed record Lfm3Config
{
    public const string ModelType = "lfm3";

    public required int HiddenSize { get; init; }

    public required int IntermediateSize { get; init; }

    public required int VocabSize { get; init; }

    public required int NumHeads { get; init; }

    public required int NumKvHeads { get; init; }

    public required int HeadDim { get; init; }

    public required float NormEps { get; init; }

    /// <summary>Taps of the depthwise convolution, which is also the length of the carried window plus one.</summary>
    public required int ConvKernel { get; init; }

    public required bool ConvBias { get; init; }

    public required int SlidingWindow { get; init; }

    public required bool SandwichNorm { get; init; }

    public required bool TieWordEmbeddings { get; init; }

    public required int MaxPositionEmbeddings { get; init; }

    public required IReadOnlyList<Lfm3LayerKind> Layers { get; init; }

    /// <summary>RoPE of full-attention layers; sliding layers read <see cref="SlidingRope"/>.</summary>
    public required Lfm3RopeParameters FullRope { get; init; }

    public required Lfm3RopeParameters SlidingRope { get; init; }

    public int BosTokenId { get; init; }

    public int EosTokenId { get; init; }

    public int NumLayers => Layers.Count;

    /// <summary>RoPE of attention layer <paramref name="layer"/>.</summary>
    /// <exception cref="InvalidOperationException">The layer is a conv layer.</exception>
    public Lfm3RopeParameters RopeOf(int layer) => Layers[layer] switch
    {
        Lfm3LayerKind.FullAttention => FullRope,
        Lfm3LayerKind.SlidingAttention => SlidingRope,
        _ => throw new InvalidOperationException($"Layer {layer} is a conv layer and has no RoPE."),
    };

    /// <summary>Reads a checkpoint's <c>config.json</c>.</summary>
    /// <exception cref="NotSupportedException">The config is not LFM3 or uses a feature the engine does not run.</exception>
    /// <exception cref="InvalidDataException">The config is inconsistent.</exception>
    public static Lfm3Config FromJson(string json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        JsonElement t = doc.RootElement;
        string modelType = t.TryGetProperty("model_type", out JsonElement mt) ? mt.GetString() ?? "" : "";
        if (!string.Equals(modelType, ModelType, StringComparison.Ordinal))
        {
            throw new NotSupportedException($"config.json model_type is '{modelType}', not {ModelType}.");
        }
        int layerCount = t.GetProperty("num_hidden_layers").GetInt32();
        JsonElement types = t.GetProperty("layer_types");
        if (types.GetArrayLength() != layerCount)
        {
            throw new InvalidDataException($"layer_types names {types.GetArrayLength()} layers; num_hidden_layers is {layerCount}.");
        }
        List<Lfm3LayerKind> layers = [];
        foreach (JsonElement type in types.EnumerateArray())
        {
            layers.Add(type.GetString() switch
            {
                "conv" => Lfm3LayerKind.Conv,
                "full_attention" => Lfm3LayerKind.FullAttention,
                "sliding_attention" => Lfm3LayerKind.SlidingAttention,
                string other => throw new NotSupportedException($"LFM3 layer type '{other}' is not supported."),
                null => throw new InvalidDataException("layer_types holds a null entry."),
            });
        }
        int hidden = t.GetProperty("hidden_size").GetInt32();
        int heads = t.GetProperty("num_attention_heads").GetInt32();
        int kvHeads = t.GetProperty("num_key_value_heads").GetInt32();
        int headDim = t.TryGetProperty("head_dim", out JsonElement hd) ? hd.GetInt32() : hidden / heads;
        if (heads % kvHeads != 0)
        {
            throw new InvalidDataException($"num_attention_heads {heads} is not a multiple of num_key_value_heads {kvHeads}.");
        }
        int window = t.TryGetProperty("sliding_window", out JsonElement sw) && sw.ValueKind == JsonValueKind.Number ? sw.GetInt32() : 0;
        if (layers.Contains(Lfm3LayerKind.SlidingAttention) && window <= 0)
        {
            throw new InvalidDataException("A sliding_attention layer needs a positive sliding_window.");
        }
        JsonElement rope = t.GetProperty("rope_parameters");
        Lfm3Config config = new()
        {
            HiddenSize = hidden,
            IntermediateSize = t.GetProperty("intermediate_size").GetInt32(),
            VocabSize = t.GetProperty("vocab_size").GetInt32(),
            NumHeads = heads,
            NumKvHeads = kvHeads,
            HeadDim = headDim,
            NormEps = (float)t.GetProperty("norm_eps").GetDouble(),
            ConvKernel = t.GetProperty("conv_L_cache").GetInt32(),
            ConvBias = Bool(t, "conv_bias", false),
            SlidingWindow = window,
            SandwichNorm = Bool(t, "use_sandwich_norm", false),
            TieWordEmbeddings = Bool(t, "tie_word_embeddings", true),
            MaxPositionEmbeddings = t.GetProperty("max_position_embeddings").GetInt32(),
            Layers = layers,
            FullRope = ReadRope(rope, layers, "full_attention"),
            SlidingRope = ReadRope(rope, layers, "sliding_attention"),
            BosTokenId = Int(t, "bos_token_id"),
            EosTokenId = Int(t, "eos_token_id"),
        };
        if (config.ConvBias)
        {
            throw new NotSupportedException("LFM3 conv_bias is not supported; the released checkpoints have no conv or projection bias.");
        }
        return config;
    }

    private static Lfm3RopeParameters ReadRope(JsonElement rope, List<Lfm3LayerKind> layers, string type)
    {
        Lfm3LayerKind kind = type == "full_attention" ? Lfm3LayerKind.FullAttention : Lfm3LayerKind.SlidingAttention;
        if (!layers.Contains(kind))
        {
            return default;
        }
        if (!rope.TryGetProperty(type, out JsonElement scope))
        {
            throw new InvalidDataException($"rope_parameters has no '{type}' entry for a model that has {type} layers.");
        }
        string ropeType = scope.TryGetProperty("rope_type", out JsonElement rt) ? rt.GetString() ?? "default" : "default";
        if (ropeType != "default")
        {
            throw new NotSupportedException($"LFM3 rope_type '{ropeType}' is not supported.");
        }
        float partial = scope.TryGetProperty("partial_rotary_factor", out JsonElement pf) ? (float)pf.GetDouble() : 1f;
        return new Lfm3RopeParameters((float)scope.GetProperty("rope_theta").GetDouble(), partial);
    }

    private static bool Bool(JsonElement e, string name, bool fallback) =>
        e.TryGetProperty(name, out JsonElement v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False ? v.GetBoolean() : fallback;

    private static int Int(JsonElement e, string name) =>
        e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : 0;
}
