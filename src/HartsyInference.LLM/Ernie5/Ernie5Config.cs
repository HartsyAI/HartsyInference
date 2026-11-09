using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Ernie5;

/// <summary>
/// The ERNIE-5.5 MoE text config (<c>model_type: "ernie5_5_moe"</c>), read from a real <c>config.json</c>. It names the
/// architecture's dimensions and the features the runtime does not implement yet; the engine refuses to load the
/// checkpoint and names each one (<see cref="RequireRunnable"/>) rather than run it with a feature silently dropped.
/// </summary>
public sealed record Ernie5Config
{
    /// <summary>The <c>model_type</c> value this reader accepts.</summary>
    public const string ModelType = "ernie5_5_moe";

    /// <summary>Vocabulary size (<c>vocab_size</c>).</summary>
    public required int VocabSize { get; init; }

    /// <summary>Residual stream width (<c>hidden_size</c>).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Decoder layers (<c>num_hidden_layers</c>), excluding the MTP layer.</summary>
    public required int NumHiddenLayers { get; init; }

    /// <summary>Multi-token-prediction layers (<c>num_nextn_predict_layers</c>); speculative-draft heads, not on the main path.</summary>
    public int NumNextnPredictLayers { get; init; }

    /// <summary>Query heads of the main attention (<c>num_attention_heads</c>).</summary>
    public required int NumAttentionHeads { get; init; }

    /// <summary>KV heads of the main attention (<c>num_key_value_heads</c>): one shared KV head (MQA).</summary>
    public required int NumKeyValueHeads { get; init; }

    /// <summary>Per-head width of the main attention (<c>head_dim</c>).</summary>
    public required int HeadDim { get; init; }

    /// <summary>Per-head no-position Q/K width (<c>qk_nope_head_dim</c>).</summary>
    public required int QkNopeHeadDim { get; init; }

    /// <summary>Per-head rotary Q/K width (<c>qk_rope_head_dim</c>).</summary>
    public required int QkRopeHeadDim { get; init; }

    /// <summary>Per-head value width (<c>v_head_dim</c>).</summary>
    public required int VHeadDim { get; init; }

    /// <summary>Low-rank query projection rank (<c>q_lora_rank</c>).</summary>
    public required int QLoraRank { get; init; }

    /// <summary>Low-rank compressed KV rank (<c>kv_lora_rank</c>).</summary>
    public required int KvLoraRank { get; init; }

    /// <summary>Low-rank output projection rank (<c>o_lora_rank</c>).</summary>
    public required int OLoraRank { get; init; }

    /// <summary>Groups of the output projection (<c>o_groups</c>); 1 means ungrouped.</summary>
    public int OGroups { get; init; } = 1;

    /// <summary>Dense FFN width (<c>intermediate_size</c>), used by the first <see cref="FirstKDenseReplace"/> layers.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Routed-expert FFN width (<c>moe_intermediate_size</c>).</summary>
    public required int MoeIntermediateSize { get; init; }

    /// <summary>Width the routed experts run at (<c>moe_latent_size</c>); 0 runs them at <see cref="HiddenSize"/>.</summary>
    public int MoeLatentSize { get; init; }

    /// <summary>Routed experts per MoE layer (<c>n_routed_experts</c>).</summary>
    public required int NRoutedExperts { get; init; }

    /// <summary>Always-on shared experts (<c>n_shared_experts</c>).</summary>
    public int NSharedExperts { get; init; }

    /// <summary>Routed experts selected per token (<c>num_experts_per_tok</c>).</summary>
    public required int NumExpertsPerTok { get; init; }

    /// <summary>Leading dense layers before the first MoE layer (<c>first_k_dense_replace</c>).</summary>
    public int FirstKDenseReplace { get; init; }

    /// <summary>Multiplier on the routed-expert output (<c>routed_scaling_factor</c>).</summary>
    public double RoutedScalingFactor { get; init; } = 1.0;

    /// <summary>Activation name of the FFN (<c>hidden_act</c>); <c>"situ"</c> is the SiTU-GLU gate.</summary>
    public required string HiddenAct { get; init; }

    /// <summary>Per-layer compression ratio, one entry per decoder and MTP layer (<c>compress_ratios</c>); <c>-2</c> marks an indexer layer.</summary>
    public required int[] CompressRatios { get; init; }

    /// <summary>Attention variant tag (<c>experimental_attention_variant</c>), e.g. <c>"dsv4_hybrid"</c>.</summary>
    public string? ExperimentalAttentionVariant { get; init; }

    /// <summary>Whether the residual stream is split into <see cref="HcMult"/> hyper-connection streams (<c>enable_hyper_connections</c>).</summary>
    public bool EnableHyperConnections { get; init; }

    /// <summary>Hyper-connection expansion rate (<c>hc_mult</c>).</summary>
    public int HcMult { get; init; } = 1;

    /// <summary>Sinkhorn iterations of the hyper-connection mixing matrix (<c>hc_sinkhorn_iters</c>).</summary>
    public int HcSinkhornIters { get; init; }

    /// <summary>Hyper-connection epsilon (<c>hc_eps</c>).</summary>
    public double HcEps { get; init; }

    /// <summary>Indexer head count (<c>index_n_heads</c>).</summary>
    public int IndexNHeads { get; init; }

    /// <summary>Indexer head width (<c>index_head_dim</c>).</summary>
    public int IndexHeadDim { get; init; }

    /// <summary>Compressed positions the indexer keeps per query (<c>index_topk</c>).</summary>
    public int IndexTopk { get; init; }

    /// <summary>Sliding-window length (<c>sliding_window</c>).</summary>
    public int SlidingWindow { get; init; }

    /// <summary>RoPE base of the uncompressed attention (<c>rope_theta</c>).</summary>
    public double RopeTheta { get; init; }

    /// <summary>RoPE base of the compressed path (<c>compress_rope_theta</c>).</summary>
    public double CompressRopeTheta { get; init; }

    /// <summary>RMSNorm epsilon (<c>rms_norm_eps</c>).</summary>
    public double RmsNormEps { get; init; }

    /// <summary>Maximum context length (<c>max_position_embeddings</c>).</summary>
    public int MaxPositionEmbeddings { get; init; }

    /// <summary>Quantization method from <c>quantization_config.quant_method</c>, or <c>null</c> for an unquantized checkpoint.</summary>
    public string? QuantMethod { get; init; }

    /// <summary>
    /// The features of this config the engine does not implement, one plain-language entry each. Empty means nothing is
    /// refused on feature grounds (no ERNIE runtime exists yet, so the loader still refuses the checkpoint).
    /// </summary>
    public IReadOnlyList<string> UnimplementedFeatures()
    {
        List<string> missing = [];
        if (string.Equals(HiddenAct, "situ", StringComparison.Ordinal))
            missing.Add("SiTU-GLU activation (hidden_act 'situ')");
        if (CompressRatios.Any(static ratio => ratio != 0) || ExperimentalAttentionVariant is not null)
            missing.Add("hybrid compressed attention (compress_ratios, DSA indexer)");
        if (EnableHyperConnections)
            missing.Add($"manifold-constrained hyper-connections (hc_mult {HcMult})");
        if (MoeLatentSize > 0)
            missing.Add($"latent MoE (moe_latent_size {MoeLatentSize})");
        if (OGroups > 1)
            missing.Add($"grouped output projection (o_groups {OGroups})");
        if (NumNextnPredictLayers > 0)
            missing.Add($"multi-token prediction (num_nextn_predict_layers {NumNextnPredictLayers})");
        return missing;
    }

    /// <summary>Parses a <c>config.json</c> document.</summary>
    /// <exception cref="HartsyInferenceException">The document is not an <c>ernie5_5_moe</c> config, or a required field is missing or malformed.</exception>
    public static Ernie5Config Parse(string json)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        HfConfigFields r = new(root, "config", "ERNIE-5.5");

        string modelType = r.OptionalString("model_type") ?? string.Empty;
        if (!string.Equals(modelType, ModelType, StringComparison.Ordinal))
            throw new HartsyInferenceException($"ERNIE-5.5 config: model_type is '{modelType}', expected '{ModelType}'.");

        string? quantMethod = null;
        if (root.TryGetProperty("quantization_config", out JsonElement quant) && quant.ValueKind == JsonValueKind.Object
            && quant.TryGetProperty("quant_method", out JsonElement method) && method.ValueKind == JsonValueKind.String)
            quantMethod = method.GetString();

        return new Ernie5Config
        {
            VocabSize = r.Int("vocab_size"),
            HiddenSize = r.Int("hidden_size"),
            NumHiddenLayers = r.Int("num_hidden_layers"),
            NumNextnPredictLayers = r.Int("num_nextn_predict_layers", 0),
            NumAttentionHeads = r.Int("num_attention_heads"),
            NumKeyValueHeads = r.Int("num_key_value_heads"),
            HeadDim = r.Int("head_dim"),
            QkNopeHeadDim = r.Int("qk_nope_head_dim"),
            QkRopeHeadDim = r.Int("qk_rope_head_dim"),
            VHeadDim = r.Int("v_head_dim"),
            QLoraRank = r.Int("q_lora_rank"),
            KvLoraRank = r.Int("kv_lora_rank"),
            OLoraRank = r.Int("o_lora_rank"),
            OGroups = r.Int("o_groups", 1),
            IntermediateSize = r.Int("intermediate_size"),
            MoeIntermediateSize = r.Int("moe_intermediate_size"),
            MoeLatentSize = r.Int("moe_latent_size", 0),
            NRoutedExperts = r.Int("n_routed_experts"),
            NSharedExperts = r.Int("n_shared_experts", 0),
            NumExpertsPerTok = r.Int("num_experts_per_tok"),
            FirstKDenseReplace = r.Int("first_k_dense_replace", 0),
            RoutedScalingFactor = r.Double("routed_scaling_factor", 1.0),
            HiddenAct = r.OptionalString("hidden_act")
                ?? throw new HartsyInferenceException("ERNIE-5.5 config: 'config.hidden_act' must be a string."),
            CompressRatios = r.Ints("compress_ratios"),
            ExperimentalAttentionVariant = r.OptionalString("experimental_attention_variant"),
            EnableHyperConnections = r.Bool("enable_hyper_connections", false),
            HcMult = r.Int("hc_mult", 1),
            HcSinkhornIters = r.Int("hc_sinkhorn_iters", 0),
            HcEps = r.Double("hc_eps", 0.0),
            IndexNHeads = r.Int("index_n_heads", 0),
            IndexHeadDim = r.Int("index_head_dim", 0),
            IndexTopk = r.Int("index_topk", 0),
            SlidingWindow = r.Int("sliding_window", 0),
            RopeTheta = r.Double("rope_theta"),
            CompressRopeTheta = r.Double("compress_rope_theta", 0.0),
            RmsNormEps = r.Double("rms_norm_eps", 1e-6),
            MaxPositionEmbeddings = r.Int("max_position_embeddings"),
            QuantMethod = quantMethod,
        };
    }

    /// <summary>Reads and parses the <c>config.json</c> at <paramref name="path"/>.</summary>
    public static Ernie5Config Load(string path) => Parse(File.ReadAllText(path));

    /// <summary>
    /// Refuses an ERNIE-5.5 checkpoint, naming every feature of its config the engine does not implement. Always throws:
    /// no ERNIE runtime is wired yet, so the checkpoint cannot run even when its feature list is empty.
    /// </summary>
    /// <exception cref="HartsyInferenceException">Always: the checkpoint cannot be loaded.</exception>
    public static void RequireRunnable(string root, string configPath)
    {
        IReadOnlyList<string> missing = Load(configPath).UnimplementedFeatures();
        string features = missing.Count == 0 ? "no ERNIE-5.5 runtime is wired" : string.Join("; ", missing);
        throw new HartsyInferenceException(
            $"'{root}' is an ERNIE-5.5 checkpoint (model_type '{ModelType}') that the engine cannot run yet; not implemented: {features}.");
    }
}
