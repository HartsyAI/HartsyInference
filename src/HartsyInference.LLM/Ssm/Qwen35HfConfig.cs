using System.Text.Json;
using HartsyInference.Core.Tensors;

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

    /// <summary>Frequency pairs per multimodal RoPE axis (temporal, height, width); null when the config does not say, in
    /// which case the model takes the family's <c>[11, 11, 10]</c>.</summary>
    public IReadOnlyList<int>? MropeSection { get; init; }

    /// <summary>Whether the M-RoPE axes interleave across frequency pairs (Qwen3.5) rather than sitting in sections.</summary>
    public bool MropeInterleaved { get; init; } = true;

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
        List<JsonElement> mropeScopes = [];
        foreach (string key in (string[])["rope_parameters", "rope_scaling"])
        {
            if (t.TryGetProperty(key, out JsonElement scope) && scope.ValueKind == JsonValueKind.Object) mropeScopes.Add(scope);
        }
        mropeScopes.Add(t);
        if (t.TryGetProperty("rope_parameters", out JsonElement parameters))
        {
            if (parameters.TryGetProperty("rope_theta", out JsonElement th)) theta = (float)th.GetDouble();
            if (parameters.TryGetProperty("partial_rotary_factor", out JsonElement pf)) partial = (float)pf.GetDouble();
        }
        else
        {
            if (t.TryGetProperty("rope_theta", out JsonElement th)) theta = (float)th.GetDouble();
            if (t.TryGetProperty("partial_rotary_factor", out JsonElement pf2)) partial = (float)pf2.GetDouble();
        }
        int[]? mropeSection = null;
        bool mropeInterleaved = true;
        foreach (JsonElement scope in mropeScopes)
        {
            if (mropeSection is null && scope.TryGetProperty("mrope_section", out JsonElement section) && section.ValueKind == JsonValueKind.Array)
            {
                mropeSection = new int[section.GetArrayLength()];
                int at = 0;
                foreach (JsonElement pairs in section.EnumerateArray()) mropeSection[at++] = pairs.GetInt32();
            }
            if (scope.TryGetProperty("mrope_interleaved", out JsonElement interleaved) && interleaved.ValueKind is JsonValueKind.True or JsonValueKind.False)
                mropeInterleaved = interleaved.GetBoolean();
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
            PartialRotaryFactor = partial, RopeTheta = theta, MropeSection = mropeSection, MropeInterleaved = mropeInterleaved,
            RmsNormEps = t.TryGetProperty("rms_norm_eps", out JsonElement eps) ? (float)eps.GetDouble() : 1e-6f,
            NumExperts = Int(t, "num_experts", 0), NumExpertsPerTok = Int(t, "num_experts_per_tok", 0),
            MoeIntermediateSize = Int(t, "moe_intermediate_size", 0),
            SharedExpertIntermediateSize = Int(t, "shared_expert_intermediate_size", 0),
            NormTopKProb = !t.TryGetProperty("norm_topk_prob", out JsonElement norm) || norm.GetBoolean(),
            TieWordEmbeddings = !t.TryGetProperty("tie_word_embeddings", out JsonElement tie) || tie.GetBoolean(),
            MaxPositionEmbeddings = Int(t, "max_position_embeddings", 262_144),
        };
    }

    /// <summary>The HF key roots a dense Qwen3.5 text trunk can sit under, most specific first:
    /// <c>Qwen3_5ForConditionalGeneration</c>, <c>Qwen3_5Model</c>, and a bare text model.</summary>
    private static readonly string[] LanguageModelPrefixes = ["model.language_model", "language_model", "model"];

    /// <summary>The root a checkpoint's text trunk sits under (the prefix whose <c>embed_tokens.weight</c> it carries).</summary>
    /// <exception cref="InvalidDataException">No known root holds a token embedding.</exception>
    public static string LanguageModelPrefix(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        foreach (string prefix in LanguageModelPrefixes)
        {
            if (weights.ContainsKey($"{prefix}.embed_tokens.weight"))
                return prefix;
        }
        throw new InvalidDataException("No Qwen3.5 text trunk found: none of model.language_model, language_model or model holds embed_tokens.weight.");
    }

    /// <summary>Derives a dense Qwen3.5 text config from the weight shapes under <paramref name="prefix"/>, for a
    /// single-file checkpoint shipped without its <c>config.json</c> (ComfyUI's repackaged text encoders). Layer kinds
    /// come from which attention each layer carries; head counts from the projection rows over the per-head widths of
    /// the Q/K norm and the gated norm. What shapes cannot carry takes the family's constants (partial rotary 0.25,
    /// rope theta 1e7, RMS eps 1e-6, equal key and value widths in the linear layers), which every Qwen3.5 size
    /// shares (ComfyUI <c>comfy/text_encoders/qwen35.py</c>). MoE checkpoints need their <c>config.json</c>.</summary>
    public static Qwen35HfConfig FromWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        ArgumentNullException.ThrowIfNull(weights);
        Tensor embed = Require(weights, $"{prefix}.embed_tokens.weight");
        int layers = 0;
        while (weights.ContainsKey($"{prefix}.layers.{layers}.input_layernorm.weight")) layers++;
        if (layers == 0)
            throw new InvalidDataException($"No Qwen3.5 layers under '{prefix}'.");
        if (weights.ContainsKey($"{prefix}.layers.0.mlp.gate.weight"))
            throw new InvalidDataException("This is a Qwen3.5-MoE checkpoint; its config.json is needed to load it.");
        bool[] linear = new bool[layers];
        int fullLayer = -1, linearLayer = -1;
        for (int i = 0; i < layers; i++)
        {
            linear[i] = weights.ContainsKey($"{prefix}.layers.{i}.linear_attn.in_proj_qkv.weight");
            if (linear[i] && linearLayer < 0) linearLayer = i;
            if (!linear[i] && fullLayer < 0) fullLayer = i;
        }
        if (fullLayer < 0 || linearLayer < 0)
            throw new InvalidDataException("A Qwen3.5 trunk has both full-attention and Gated DeltaNet layers; this checkpoint lacks one kind.");
        string sa = $"{prefix}.layers.{fullLayer}.self_attn", la = $"{prefix}.layers.{linearLayer}.linear_attn";
        int headDim = (int)Require(weights, $"{sa}.q_norm.weight").ElementCount;
        int valueHeads = (int)Require(weights, $"{la}.in_proj_b.weight").Shape[0];
        int valueDim = (int)Require(weights, $"{la}.norm.weight").ElementCount;
        int qkvRows = (int)Require(weights, $"{la}.in_proj_qkv.weight").Shape[0];
        int keyTotal = (qkvRows - valueHeads * valueDim) / 2;
        if (keyTotal <= 0 || keyTotal % valueDim != 0)
            throw new InvalidDataException($"Qwen3.5 in_proj_qkv has {qkvRows} rows, which does not split into keys and {valueHeads}x{valueDim} values.");
        Tensor conv = Require(weights, $"{la}.conv1d.weight");
        return new Qwen35HfConfig
        {
            HiddenSize = (int)embed.Shape[1],
            VocabSize = (int)embed.Shape[0],
            NumLayers = layers,
            NumHeads = (int)(Require(weights, $"{sa}.q_proj.weight").Shape[0] / (2 * headDim)),
            NumKvHeads = (int)(Require(weights, $"{sa}.k_proj.weight").Shape[0] / headDim),
            HeadDim = headDim,
            FullAttentionInterval = 0,
            LinearLayers = linear,
            LinearConvKernel = (int)conv.Shape[conv.Shape.Rank - 1],
            LinearKeyHeadDim = valueDim,
            LinearValueHeadDim = valueDim,
            LinearNumKeyHeads = keyTotal / valueDim,
            LinearNumValueHeads = valueHeads,
            TieWordEmbeddings = !weights.ContainsKey("lm_head.weight"),
        };
    }

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string key) =>
        weights.TryGetValue(key, out Tensor? tensor)
            ? tensor : throw new InvalidDataException($"The Qwen3.5 checkpoint has no tensor '{key}'.");

    private static int Int(JsonElement t, string name, int fallback) =>
        t.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;
}
