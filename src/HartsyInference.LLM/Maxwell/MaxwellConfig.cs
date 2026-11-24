using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Maxwell;

/// <summary>The text decoder of Maxwell-V1.0-Flash (<c>model_type maxwell_moe</c>): mHC residual streams, block-sparse GQA attention with a
/// per-layer indexer, dense layers then sigmoid-routed MoE with a shared expert, and optional multi-token-prediction layers.</summary>
public sealed record MaxwellConfig
{
    /// <summary>The checkpoint's <c>model_type</c>.</summary>
    public const string ModelType = "maxwell_moe";

    /// <summary>Vocabulary size.</summary>
    public required int VocabSize { get; init; }

    /// <summary>Residual width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Decoder layers, MTP layers excluded.</summary>
    public required int NumHiddenLayers { get; init; }

    /// <summary>Multi-token-prediction layers that follow the decoder layers.</summary>
    public int MtpLayers { get; init; }

    /// <summary>Query heads.</summary>
    public required int NumAttentionHeads { get; init; }

    /// <summary>Key/value heads, which are also the indexer's groups.</summary>
    public required int NumKeyValueHeads { get; init; }

    /// <summary>Per-head width of queries, keys and values.</summary>
    public required int HeadDim { get; init; }

    /// <summary>Width of a dense layer's FFN.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Width of one routed expert.</summary>
    public required int MoeIntermediateSize { get; init; }

    /// <summary>Width of the always-on shared expert.</summary>
    public required int SharedExpertIntermediateSize { get; init; }

    /// <summary>Routed experts per MoE layer.</summary>
    public required int NumExperts { get; init; }

    /// <summary>Routed experts a token uses.</summary>
    public required int NumExpertsPerTok { get; init; }

    /// <summary>Layers with a dense FFN instead of MoE.</summary>
    public required IReadOnlySet<int> DenseLayers { get; init; }

    /// <summary>RMS-norm epsilon.</summary>
    public required double RmsNormEps { get; init; }

    /// <summary>RoPE base; the rotation covers the whole head.</summary>
    public required double RopeTheta { get; init; }

    /// <summary>The config's <c>max_position_embeddings</c>; the model card's table says 262,144, so see <see cref="SupportedContext"/>.</summary>
    public required int MaxPositionEmbeddings { get; init; }

    /// <summary>Residual streams.</summary>
    public required int HcMult { get; init; }

    /// <summary>Sinkhorn rounds on the comb matrix.</summary>
    public required int HcSinkhornIters { get; init; }

    /// <summary>The block-sparse attention geometry.</summary>
    public required MaxwellSparseConfig Sparse { get; init; }

    /// <summary>Block edge of the FP8 scale grids, or 0 when the checkpoint is not block-FP8.</summary>
    public int Fp8Block { get; init; }

    /// <summary>The context the model card states as supported: 262,144 tokens.</summary>
    public const int SupportedContext = 262_144;

    /// <summary>The epsilon of every hyper-connection formula (the reference's <c>self.eps</c>) and of its read-out head.</summary>
    public const double HcEpsilon = 1e-6;

    /// <summary>Total layers including MTP layers; MTP layer <c>k</c> is layer <c>NumHiddenLayers + k</c>.</summary>
    public int TotalLayers => NumHiddenLayers + MtpLayers;

    /// <summary>Whether layer <paramref name="layer"/> has a dense FFN.</summary>
    public bool IsDenseLayer(int layer) => DenseLayers.Contains(layer);

    /// <summary>Parses a <c>config.json</c> document.</summary>
    /// <exception cref="HartsyInferenceException">The document is not a Maxwell config the engine runs, or a field is missing.</exception>
    public static MaxwellConfig Parse(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        HfConfigFields r = new(root, "config", "Maxwell");
        string modelType = r.OptionalString("model_type") ?? string.Empty;
        if (!string.Equals(modelType, ModelType, StringComparison.Ordinal))
            throw new HartsyInferenceException($"Maxwell config: model_type is '{modelType}', expected '{ModelType}'.");
        MaxwellConfig config = new()
        {
            VocabSize = r.Int("vocab_size"),
            HiddenSize = r.Int("hidden_size"),
            NumHiddenLayers = r.Int("num_hidden_layers"),
            MtpLayers = r.Int("mtp_num_layers", 0),
            NumAttentionHeads = r.Int("num_attention_heads"),
            NumKeyValueHeads = r.Int("num_key_value_heads"),
            HeadDim = r.Int("attention_head_size"),
            IntermediateSize = r.Int("intermediate_size"),
            MoeIntermediateSize = r.Int("moe_intermediate_size"),
            SharedExpertIntermediateSize = r.Int("shared_expert_intermediate_size"),
            NumExperts = r.Int("num_experts"),
            NumExpertsPerTok = r.Int("num_experts_per_tok"),
            DenseLayers = new HashSet<int>(r.Ints("mlp_only_layers")),
            RmsNormEps = r.Double("rms_norm_eps"),
            RopeTheta = r.Double("rope_theta"),
            MaxPositionEmbeddings = r.Int("max_position_embeddings"),
            HcMult = r.Int("num_residual_streams"),
            HcSinkhornIters = r.Int("mhc_sinkhorn_iterations"),
            Sparse = MaxwellSparseConfig.Parse(r),
            Fp8Block = ReadFp8Block(root),
        };
        config.RequireSupported(r);
        return config;
    }

    // The features the engine does not run are named, so a variant fails here and not in a forward.
    private void RequireSupported(HfConfigFields r)
    {
        List<string> unsupported = [];
        if (r.OptionalString("attention_variant") != "block_sparse_gqa") unsupported.Add("attention_variant other than block_sparse_gqa");
        if (r.OptionalString("hidden_act") != "silu") unsupported.Add("hidden_act other than silu");
        if (!r.Bool("enable_hyper_connections", false)) unsupported.Add("a model without hyper-connections");
        if (!r.Bool("qk_layernorm", false)) unsupported.Add("attention without q/k RMS norm");
        if (r.OptionalString("router_score_function") != "sigmoid") unsupported.Add("a router other than sigmoid");
        if (!r.Bool("moe_router_enable_expert_bias", false)) unsupported.Add("a router without expert_bias");
        if (r.Bool("use_shared_expert_gate", false)) unsupported.Add("a gated shared expert");
        if (r.Bool("tie_word_embeddings", false)) unsupported.Add("tied embeddings");
        if (r.OptionalString("moe_router_dtype") is { } routerDtype && routerDtype != "fp32") unsupported.Add($"router dtype {routerDtype}");
        if (r.Ints("block_sparse_gqa_layer_freq") is { } freq && (freq.Length != NumHiddenLayers || freq.Any(static f => f != 1)))
            unsupported.Add("layers without block-sparse attention");
        if (unsupported.Count != 0)
            throw new HartsyInferenceException($"Maxwell config: unsupported {string.Join(", ", unsupported)}.");
        if (NumAttentionHeads % NumKeyValueHeads != 0 || Sparse.IndexerGroups != NumKeyValueHeads)
            throw new HartsyInferenceException("Maxwell config: the indexer needs one group per key/value head, and the heads must divide evenly.");
        if (HcMult < 1 || HcMult > 8) throw new HartsyInferenceException($"Maxwell config: num_residual_streams {HcMult} is outside 1..8.");
        if (Sparse.IndexerHeadDim != HeadDim)
            throw new HartsyInferenceException("Maxwell config: the indexer shares the attention's RoPE, so its head width must equal attention_head_size.");
        // The hyper-connections fix their own epsilon at 1e-6 and share this one in the engine's head read-out.
        if (RmsNormEps != HcEpsilon) throw new HartsyInferenceException($"Maxwell config: rms_norm_eps {RmsNormEps} must be {HcEpsilon}.");
        if (!DenseLayers.OrderBy(static l => l).SequenceEqual(Enumerable.Range(0, DenseLayers.Count)))
            throw new HartsyInferenceException("Maxwell config: mlp_only_layers must be the first layers of the model.");
    }

    private static int ReadFp8Block(JsonElement root)
    {
        if (!root.TryGetProperty("quantization_config", out JsonElement quant) || quant.ValueKind != JsonValueKind.Object) return 0;
        HfConfigFields q = new(quant, "config.quantization_config", "Maxwell");
        if (q.OptionalString("quant_method") != "fp8") throw new HartsyInferenceException("Maxwell config: only fp8 quantization is wired.");
        int[] block = q.Ints("weight_block_size");
        if (block.Length != 2 || block[0] != block[1])
            throw new HartsyInferenceException($"Maxwell config: weight_block_size [{string.Join(", ", block)}] must be a square block.");
        return block[0];
    }
}
