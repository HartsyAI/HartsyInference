using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Ernie5;

/// <summary>
/// The ERNIE-5.5 MoE text config (<c>model_type: "ernie5_5_moe"</c>), read from a real <c>config.json</c>. It holds every field
/// <see cref="Ernie5Model"/> reads and names the values the runtime does not implement (<see cref="UnimplementedFeatures"/>), so a
/// checkpoint is refused by name rather than run with a feature silently dropped.
/// </summary>
public sealed record Ernie5Config
{
    /// <summary>The <c>model_type</c> value this reader accepts.</summary>
    public const string ModelType = "ernie5_5_moe";

    /// <summary>The <c>compress_ratios</c> entry of a DSA layer.</summary>
    public const int DsaRatio = -2;

    /// <summary>Vocabulary size (<c>vocab_size</c>).</summary>
    public required int VocabSize { get; init; }

    /// <summary>Residual stream width (<c>hidden_size</c>).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Decoder layers (<c>num_hidden_layers</c>), excluding the MTP layer.</summary>
    public required int NumHiddenLayers { get; init; }

    /// <summary>Multi-token-prediction layers (<c>num_nextn_predict_layers</c>), numbered after the decoder layers; speculative-draft heads
    /// off the main path, loaded only with <see cref="Ernie5LoadOptions.Mtp"/>.</summary>
    public int NumNextnPredictLayers { get; init; }

    /// <summary>Query heads of the HCA attention (<c>num_attention_heads</c>).</summary>
    public required int NumAttentionHeads { get; init; }

    /// <summary>KV heads of the HCA attention (<c>num_key_value_heads</c>): one shared KV row (MQA).</summary>
    public required int NumKeyValueHeads { get; init; }

    /// <summary>Per-head width of the HCA attention (<c>head_dim</c>).</summary>
    public required int HeadDim { get; init; }

    /// <summary>Per-head no-position Q/K width of the HCA attention (<c>qk_nope_head_dim</c>).</summary>
    public required int QkNopeHeadDim { get; init; }

    /// <summary>Per-head rotary Q/K width of the HCA attention (<c>qk_rope_head_dim</c>).</summary>
    public required int QkRopeHeadDim { get; init; }

    /// <summary>Per-head value width (<c>v_head_dim</c>).</summary>
    public required int VHeadDim { get; init; }

    /// <summary>Low-rank query projection rank of the HCA attention (<c>q_lora_rank</c>).</summary>
    public required int QLoraRank { get; init; }

    /// <summary>Low-rank compressed KV rank (<c>kv_lora_rank</c>).</summary>
    public required int KvLoraRank { get; init; }

    /// <summary>Low-rank output projection rank (<c>o_lora_rank</c>).</summary>
    public required int OLoraRank { get; init; }

    /// <summary>Groups of the output projection (<c>o_groups</c>); 1 means ungrouped.</summary>
    public int OGroups { get; init; } = 1;

    /// <summary>Query heads of the DSA attention (<c>hybrid_mla_num_attention_heads</c>).</summary>
    public required int MlaHeads { get; init; }

    /// <summary>Query latent width of the DSA attention (<c>hybrid_mla_q_lora_rank</c>).</summary>
    public required int MlaQLoraRank { get; init; }

    /// <summary>Key/value latent width of the DSA attention (<c>hybrid_mla_kv_lora_rank</c>).</summary>
    public required int MlaKvLoraRank { get; init; }

    /// <summary>Per-head no-position width of the DSA attention (<c>hybrid_mla_qk_nope_head_dim</c>).</summary>
    public required int MlaQkNopeHeadDim { get; init; }

    /// <summary>Per-head rotary width of the DSA attention (<c>hybrid_mla_qk_rope_head_dim</c>).</summary>
    public required int MlaQkRopeHeadDim { get; init; }

    /// <summary>Per-head value width of the DSA attention (<c>hybrid_mla_v_head_dim</c>).</summary>
    public required int MlaVHeadDim { get; init; }

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

    /// <summary>Multiplier on the routed-expert output (<c>routed_scaling_factor</c>); unused when
    /// <see cref="RoutedScalingFactorLearnable"/>.</summary>
    public double RoutedScalingFactor { get; init; } = 1.0;

    /// <summary>Per-expert learned routed scale from the checkpoint (<c>routed_scaling_factor_learnable</c>).</summary>
    public bool RoutedScalingFactorLearnable { get; init; }

    /// <summary>Two router matrices whose sigmoid scores are summed (<c>moe_split_feature_routing</c>).</summary>
    public bool MoeSplitFeatureRouting { get; init; }

    /// <summary>Router scoring (<c>scoring_func</c>).</summary>
    public string ScoringFunc { get; init; } = "sigmoid";

    /// <summary>Renormalizes the selected routing weights (<c>norm_topk_prob</c>).</summary>
    public bool NormTopkProb { get; init; } = true;

    /// <summary>Router expert groups (<c>n_group</c>); 1 means no group limit.</summary>
    public int NGroup { get; init; } = 1;

    /// <summary>Groups kept per token (<c>topk_group</c>).</summary>
    public int TopkGroup { get; init; } = 1;

    /// <summary>Sigmoid gate on the shared expert (<c>moe_shared_expert_gate</c>).</summary>
    public bool MoeSharedExpertGate { get; init; }

    /// <summary>RMS norm on the latent expert output (<c>latent_moe_use_norm</c>).</summary>
    public bool LatentMoeUseNorm { get; init; } = true;

    /// <summary>Activation name of the FFN (<c>hidden_act</c>); <c>"situ"</c> is the SiTU-GLU gate.</summary>
    public required string HiddenAct { get; init; }

    /// <summary>SiTU gate beta (<c>activation_situ_beta</c>).</summary>
    public float ActivationSituBeta { get; init; } = 4f;

    /// <summary>SiTU value soft-cap beta (<c>activation_situ_linear_beta</c>).</summary>
    public float ActivationSituLinearBeta { get; init; } = 25f;

    /// <summary>Per-layer compression ratio, one entry per decoder and MTP layer (<c>compress_ratios</c>); <see cref="DsaRatio"/> marks a
    /// DSA layer.</summary>
    public required int[] CompressRatios { get; init; }

    /// <summary>Attention variant tag (<c>experimental_attention_variant</c>), e.g. <c>"dsv4_hybrid"</c>.</summary>
    public string? ExperimentalAttentionVariant { get; init; }

    /// <summary>Sigmoid output gate on the attention (<c>use_gated_attn</c>).</summary>
    public bool UseGatedAttn { get; init; }

    /// <summary>Activation of the attention output gate (<c>gated_attn_act</c>).</summary>
    public string? GatedAttnAct { get; init; }

    /// <summary>Low-rank head mix on the attention output (<c>use_vha_attention</c>).</summary>
    public bool UseVhaAttention { get; init; }

    /// <summary>Query/key norm form (<c>qk_norm_type</c>); <c>"per_head"</c> normalizes each HCA query head.</summary>
    public string? QkNormType { get; init; }

    /// <summary>Whether the rotated slice pairs adjacent values (<c>rope_interleave</c>).</summary>
    public bool RopeInterleave { get; init; } = true;

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

    /// <summary>RoPE base of the DSA attention and its indexer (<c>rope_theta</c>).</summary>
    public double RopeTheta { get; init; }

    /// <summary>RoPE base of the HCA attention (<c>compress_rope_theta</c>).</summary>
    public double CompressRopeTheta { get; init; }

    /// <summary>Scaling type of the HCA rope (<c>rope_scaling.type</c>); null is plain rope.</summary>
    public string? RopeScalingType { get; init; }

    /// <summary>YaRN parameters of the HCA rope (<c>rope_scaling</c>); null when <see cref="RopeScalingType"/> is not <c>"yarn"</c>.</summary>
    public DeepSeekV41RopeScaling? HcaRopeScaling { get; init; }

    /// <summary>RMSNorm epsilon (<c>rms_norm_eps</c>).</summary>
    public double RmsNormEps { get; init; }

    /// <summary>Maximum context length (<c>max_position_embeddings</c>).</summary>
    public int MaxPositionEmbeddings { get; init; }

    /// <summary>Learned logit transform after the LM head (<c>multimax</c>); <c>"lm_head"</c> reads its ranges and gains from the
    /// checkpoint.</summary>
    public string? Multimax { get; init; }

    /// <summary>Whether the LM head shares the embedding (<c>tie_word_embeddings</c>).</summary>
    public bool TieWordEmbeddings { get; init; }

    /// <summary>Begin-of-sequence id (<c>bos_token_id</c>).</summary>
    public int? BosTokenId { get; init; }

    /// <summary>End-of-sequence id (<c>eos_token_id</c>).</summary>
    public int? EosTokenId { get; init; }

    /// <summary>Quantization method from <c>quantization_config.quant_method</c>, or <c>null</c> for an unquantized checkpoint.</summary>
    public string? QuantMethod { get; init; }

    /// <summary>FP8 format (<c>quantization_config.fmt</c>).</summary>
    public string? QuantFormat { get; init; }

    /// <summary>FP8 scale block (<c>quantization_config.weight_block_size</c>); empty when absent.</summary>
    public int[] QuantBlock { get; init; } = [];

    /// <summary>Decoder layers plus multi-token-prediction layers; <see cref="CompressRatios"/> has one entry for each.</summary>
    public int TotalLayers => NumHiddenLayers + NumNextnPredictLayers;

    /// <summary>Whether layer <paramref name="layer"/> is a DSA layer; every other layer is HCA.</summary>
    public bool IsDsaLayer(int layer) => CompressRatios[layer] == DsaRatio;

    /// <summary>Whether layer <paramref name="layer"/> is a multi-token-prediction layer, numbered after the decoder layers.</summary>
    public bool IsMtpLayer(int layer) => layer >= NumHiddenLayers && layer < TotalLayers;

    /// <summary>
    /// The values of this config the engine does not implement, one plain-language entry each, for the decoder layers and, when
    /// <paramref name="mtp"/>, the multi-token-prediction layers too. Empty means the host reference model runs every feature the config names.
    /// </summary>
    public IReadOnlyList<string> UnimplementedFeatures(bool mtp = false)
    {
        List<string> missing = [];
        int layers = mtp ? TotalLayers : NumHiddenLayers;
        if (!string.Equals(HiddenAct, "situ", StringComparison.Ordinal))
            missing.Add($"FFN activation '{HiddenAct}' (only 'situ' is wired)");
        if (CompressRatios.Length < layers)
            missing.Add($"compress_ratios with {CompressRatios.Length} entries for {layers} {(mtp ? "decoder and MTP" : "decoder")} layers");
        for (int i = 0; i < Math.Min(layers, CompressRatios.Length); i++)
        {
            int ratio = CompressRatios[i];
            if (ratio != DsaRatio && (ratio < 2 || ratio == 4))
                missing.Add($"layer {i} attention with compress ratio {ratio} (only HCA ratios >= 2 other than 4, and DSA -2, are wired)");
        }
        if (ExperimentalAttentionVariant is not null && !string.Equals(ExperimentalAttentionVariant, "dsv4_hybrid", StringComparison.Ordinal))
            missing.Add($"attention variant '{ExperimentalAttentionVariant}'");
        if (NumKeyValueHeads != 1)
            missing.Add($"HCA attention with {NumKeyValueHeads} KV heads (one shared KV row is wired)");
        if (QkNopeHeadDim + QkRopeHeadDim != HeadDim)
            missing.Add($"HCA head width {HeadDim} that is not qk_nope_head_dim + qk_rope_head_dim");
        if (QkNormType is not null && !string.Equals(QkNormType, "per_head", StringComparison.Ordinal))
            missing.Add($"query norm '{QkNormType}'");
        if (!RopeInterleave)
            missing.Add("non-interleaved rope (rope_interleave false)");
        if (RopeScalingType is not null && HcaRopeScaling is null)
            missing.Add($"HCA rope scaling '{RopeScalingType}' (only yarn is wired)");
        if (UseGatedAttn && !string.Equals(GatedAttnAct ?? "sigmoid", "sigmoid", StringComparison.Ordinal))
            missing.Add($"attention gate activation '{GatedAttnAct}'");
        if (!EnableHyperConnections || HcMult < 2)
            missing.Add("a residual stream without hyper-connections (only mHC is wired)");
        if (MoeLatentSize <= 0)
            missing.Add("routed experts at full width (only latent MoE is wired)");
        if (!string.Equals(ScoringFunc, "sigmoid", StringComparison.Ordinal))
            missing.Add($"router scoring '{ScoringFunc}'");
        if (NGroup > 1 || TopkGroup > 1)
            missing.Add($"group-limited routing (n_group {NGroup}, topk_group {TopkGroup})");
        if (MoeSharedExpertGate)
            missing.Add("a gated shared expert (moe_shared_expert_gate)");
        if (NSharedExperts > 1)
            missing.Add($"{NSharedExperts} shared experts (one is wired)");
        if (Multimax is not null && !string.Equals(Multimax, "lm_head", StringComparison.Ordinal))
            missing.Add($"multimax '{Multimax}'");
        if (TieWordEmbeddings)
            missing.Add("a LM head tied to the embedding");
        if (QuantMethod is not null && (!string.Equals(QuantMethod, "fp8", StringComparison.Ordinal)
            || !string.Equals(QuantFormat ?? "e4m3", "e4m3", StringComparison.Ordinal) || QuantBlock is not [128, 128]))
            missing.Add($"quantization '{QuantMethod}' {QuantFormat} [{string.Join(", ", QuantBlock)}] (only fp8 e4m3 128x128 blocks are wired)");
        return missing;
    }

    /// <summary>Parses a <c>config.json</c> document.</summary>
    /// <exception cref="HartsyInferenceException">The document is not an <c>ernie5_5_moe</c> config, or a required field is missing or
    /// malformed.</exception>
    public static Ernie5Config Parse(string json)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        HfConfigFields r = new(root, "config", "ERNIE-5.5");

        string modelType = r.OptionalString("model_type") ?? string.Empty;
        if (!string.Equals(modelType, ModelType, StringComparison.Ordinal))
            throw new HartsyInferenceException($"ERNIE-5.5 config: model_type is '{modelType}', expected '{ModelType}'.");

        string? quantMethod = null, quantFormat = null;
        int[] quantBlock = [];
        if (root.TryGetProperty("quantization_config", out JsonElement quant) && quant.ValueKind == JsonValueKind.Object)
        {
            HfConfigFields q = new(quant, "config.quantization_config", "ERNIE-5.5");
            quantMethod = q.OptionalString("quant_method");
            quantFormat = q.OptionalString("fmt");
            if (quant.TryGetProperty("weight_block_size", out JsonElement block) && block.ValueKind == JsonValueKind.Array)
                quantBlock = q.Ints("weight_block_size");
        }

        string? scalingType = null;
        DeepSeekV41RopeScaling? yarn = null;
        if (root.TryGetProperty("rope_scaling", out JsonElement scaling) && scaling.ValueKind == JsonValueKind.Object)
        {
            HfConfigFields s = r.Object("rope_scaling");
            scalingType = s.OptionalString("type") ?? s.OptionalString("rope_type");
            if (string.Equals(scalingType, "yarn", StringComparison.Ordinal))
                yarn = new DeepSeekV41RopeScaling(s.Double("factor"), s.Double("beta_fast", 32.0), s.Double("beta_slow", 1.0),
                    s.Int("original_max_position_embeddings"));
        }

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
            MlaHeads = r.Int("hybrid_mla_num_attention_heads"),
            MlaQLoraRank = r.Int("hybrid_mla_q_lora_rank"),
            MlaKvLoraRank = r.Int("hybrid_mla_kv_lora_rank"),
            MlaQkNopeHeadDim = r.Int("hybrid_mla_qk_nope_head_dim"),
            MlaQkRopeHeadDim = r.Int("hybrid_mla_qk_rope_head_dim"),
            MlaVHeadDim = r.Int("hybrid_mla_v_head_dim"),
            IntermediateSize = r.Int("intermediate_size"),
            MoeIntermediateSize = r.Int("moe_intermediate_size"),
            MoeLatentSize = r.Int("moe_latent_size", 0),
            NRoutedExperts = r.Int("n_routed_experts"),
            NSharedExperts = r.Int("n_shared_experts", 0),
            NumExpertsPerTok = r.Int("num_experts_per_tok"),
            FirstKDenseReplace = r.Int("first_k_dense_replace", 0),
            RoutedScalingFactor = r.Double("routed_scaling_factor", 1.0),
            RoutedScalingFactorLearnable = r.Bool("routed_scaling_factor_learnable", false),
            MoeSplitFeatureRouting = r.Bool("moe_split_feature_routing", false),
            // the reference router is sigmoid-only (moe.py ernie5_5_routing), so an absent field means sigmoid
            ScoringFunc = r.OptionalString("scoring_func") ?? "sigmoid",
            NormTopkProb = r.Bool("norm_topk_prob", true),
            NGroup = r.Int("n_group", 1),
            TopkGroup = r.Int("topk_group", 1),
            MoeSharedExpertGate = r.Bool("moe_shared_expert_gate", false),
            LatentMoeUseNorm = r.Bool("latent_moe_use_norm", true),
            HiddenAct = r.OptionalString("hidden_act")
                ?? throw new HartsyInferenceException("ERNIE-5.5 config: 'config.hidden_act' must be a string."),
            ActivationSituBeta = (float)r.Double("activation_situ_beta", 4.0),
            ActivationSituLinearBeta = (float)r.Double("activation_situ_linear_beta", 25.0),
            CompressRatios = r.Ints("compress_ratios"),
            ExperimentalAttentionVariant = r.OptionalString("experimental_attention_variant"),
            UseGatedAttn = r.Bool("use_gated_attn", false),
            GatedAttnAct = r.OptionalString("gated_attn_act"),
            UseVhaAttention = r.Bool("use_vha_attention", false),
            QkNormType = r.OptionalString("qk_norm_type"),
            RopeInterleave = r.Bool("rope_interleave", true),
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
            RopeScalingType = scalingType,
            HcaRopeScaling = yarn,
            RmsNormEps = r.Double("rms_norm_eps", 1e-6),
            MaxPositionEmbeddings = r.Int("max_position_embeddings"),
            Multimax = r.OptionalString("multimax"),
            TieWordEmbeddings = r.Bool("tie_word_embeddings", false),
            BosTokenId = root.TryGetProperty("bos_token_id", out JsonElement bos) && bos.ValueKind == JsonValueKind.Number ? bos.GetInt32() : null,
            EosTokenId = root.TryGetProperty("eos_token_id", out JsonElement eos) && eos.ValueKind == JsonValueKind.Number ? eos.GetInt32() : null,
            QuantMethod = quantMethod,
            QuantFormat = quantFormat,
            QuantBlock = quantBlock,
        };
    }

    /// <summary>Reads and parses the <c>config.json</c> at <paramref name="path"/>.</summary>
    public static Ernie5Config Load(string path) => Parse(File.ReadAllText(path));

    /// <summary>Refuses an ERNIE-5.5 checkpoint the Text path cannot run, naming each config value the engine does not implement. The Text
    /// path skips the multi-token-prediction layers, so their values are not checked here; <see cref="Ernie5Model"/> checks them when loaded
    /// with <see cref="Ernie5LoadOptions.Mtp"/>.</summary>
    /// <exception cref="HartsyInferenceException">The checkpoint has a feature the Text path does not run.</exception>
    public static void RequireRunnable(string root, string configPath)
    {
        Ernie5Config config = Load(configPath);
        IReadOnlyList<string> missing = config.UnimplementedFeatures();
        if (missing.Count > 0)
            throw new HartsyInferenceException(
                $"'{root}' is an ERNIE-5.5 checkpoint (model_type '{ModelType}') with features the Text path does not run: "
                + $"{string.Join("; ", missing)}.");
    }
}
