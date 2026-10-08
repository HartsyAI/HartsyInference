using System.Globalization;
using System.Text.Json;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>DeepSeek-V4.1 hyperparameters parsed from a checkpoint's <c>config.json</c>; pure data with no model class behind it.</summary>
/// <remarks>Reads both the official nested layout (<c>text_config</c> and <c>vision_config</c>) and the flat layout the MLX conversion uses. Layers are numbered backbone first, then the draft layers, so <c>compress_ratios</c> carries <c>num_hidden_layers + num_nextn_predict_layers</c> entries.</remarks>
public sealed record DeepSeekV41Config
{
    /// <summary>The <c>model_type</c> value a DeepSeek-V4.1 config declares.</summary>
    public const string ModelType = "deepseek_v41";

    /// <summary>Vocabulary size.</summary>
    public required int VocabSize { get; init; }

    /// <summary>Residual stream width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Width of each routed and shared expert's hidden layer.</summary>
    public required int MoeIntermediateSize { get; init; }

    /// <summary>Backbone layer count (the draft layers are extra).</summary>
    public required int NumHiddenLayers { get; init; }

    /// <summary>Attention head count.</summary>
    public required int NumAttentionHeads { get; init; }

    /// <summary>KV head count (1: the latent is shared by every head).</summary>
    public required int NumKeyValueHeads { get; init; }

    /// <summary>Per-head width.</summary>
    public required int HeadDim { get; init; }

    /// <summary>Rotary slice of the head width.</summary>
    public required int QkRopeHeadDim { get; init; }

    /// <summary>Query low-rank width.</summary>
    public required int QLoraRank { get; init; }

    /// <summary>Output low-rank width.</summary>
    public required int OLoraRank { get; init; }

    /// <summary>Output projection group count.</summary>
    public required int OGroups { get; init; }

    /// <summary>SwiGLU clamp applied to the gate and up projections.</summary>
    public required double SwigluLimit { get; init; }

    /// <summary>RMSNorm epsilon (1e-20: it must never be clamped up).</summary>
    public required double RmsNormEps { get; init; }

    /// <summary>Longest supported context.</summary>
    public required int MaxPositionEmbeddings { get; init; }

    /// <summary>Rotary base for uncompressed attention.</summary>
    public required double RopeTheta { get; init; }

    /// <summary>Rotary base for the compressed-KV path.</summary>
    public required double CompressRopeTheta { get; init; }

    /// <summary>YaRN extension, or null when the config carries none (the MLX conversion drops it).</summary>
    public DeepSeekV41RopeScaling? RopeScaling { get; init; }

    /// <summary>Routed experts per layer.</summary>
    public required int NRoutedExperts { get; init; }

    /// <summary>Shared experts per layer.</summary>
    public required int NSharedExperts { get; init; }

    /// <summary>Routed experts chosen per token.</summary>
    public required int NumExpertsPerTok { get; init; }

    /// <summary>Router score function name.</summary>
    public required string ScoringFunc { get; init; }

    /// <summary>Router top-k method name.</summary>
    public required string TopkMethod { get; init; }

    /// <summary>Whether selected router weights are renormalised.</summary>
    public required bool NormTopkProb { get; init; }

    /// <summary>Multiplier applied to the routed output.</summary>
    public required double RoutedScalingFactor { get; init; }

    /// <summary>Sliding-window attention span.</summary>
    public required int SlidingWindow { get; init; }

    /// <summary>Layers that publish compressed KV for later layers.</summary>
    public required IReadOnlyList<int> KvSourceLayerIds { get; init; }

    /// <summary>Layers that publish sparse-attention indices for later layers.</summary>
    public required IReadOnlyList<int> IndexSourceLayerIds { get; init; }

    /// <summary>Indexer head count.</summary>
    public required int IndexNHeads { get; init; }

    /// <summary>Indexer head width.</summary>
    public required int IndexHeadDim { get; init; }

    /// <summary>Compressed positions the indexer selects per query.</summary>
    public required int IndexTopk { get; init; }

    /// <summary>Layer whose candidates the later index layers refine.</summary>
    public required int CandidateSourceLayerId { get; init; }

    /// <summary>Candidate blocks kept by the source layer.</summary>
    public required int CandidateTopkBlocks { get; init; }

    /// <summary>Positions per candidate block.</summary>
    public required int CandidateBlockSize { get; init; }

    /// <summary>Hyper-connection stream count.</summary>
    public required int HcMult { get; init; }

    /// <summary>Sinkhorn iterations of the hyper-connection mixer.</summary>
    public required int HcSinkhornIters { get; init; }

    /// <summary>Sinkhorn epsilon.</summary>
    public required double HcEps { get; init; }

    /// <summary>Layers that carry an Engram table.</summary>
    public required IReadOnlyList<int> EngramLayerIds { get; init; }

    /// <summary>Rows of each Engram table, parallel to <see cref="EngramLayerIds"/>.</summary>
    public required IReadOnlyList<long> EngramNumEmbeddings { get; init; }

    /// <summary>Longest n-gram hashed into Engram.</summary>
    public required int EngramMaxNgramSize { get; init; }

    /// <summary>Hash vocabulary size.</summary>
    public required int EngramVocabSize { get; init; }

    /// <summary>Engram head count.</summary>
    public required int EngramNHeads { get; init; }

    /// <summary>Engram head width.</summary>
    public required int EngramHeadDim { get; init; }

    /// <summary>Token id Engram treats as padding.</summary>
    public required int EngramPadTokenId { get; init; }

    /// <summary>Entries of the compressed-vocabulary map.</summary>
    public required int EngramCompressedVocabSize { get; init; }

    /// <summary>Draft layer count (<c>mtp.0</c> onward).</summary>
    public required int NumNextnPredictLayers { get; init; }

    /// <summary>Tokens the draft proposes per block.</summary>
    public required int DsparkBlockSize { get; init; }

    /// <summary>Placeholder token id the draft conditions on.</summary>
    public required int DsparkNoiseTokenId { get; init; }

    /// <summary>Backbone layers whose hidden states feed the draft.</summary>
    public required IReadOnlyList<int> DsparkTargetLayerIds { get; init; }

    /// <summary>Rank of the draft's Markov head.</summary>
    public required int DsparkMarkovRank { get; init; }

    /// <summary>Routed experts per draft layer.</summary>
    public required int DsparkNRoutedExperts { get; init; }

    /// <summary>Routed experts chosen per token in a draft layer.</summary>
    public required int DsparkNumExpertsPerTok { get; init; }

    /// <summary>Per-layer compression ratio for every backbone then draft layer.</summary>
    public required IReadOnlyList<int> CompressRatios { get; init; }

    /// <summary>Token id of the image placeholder.</summary>
    public int? ImageTokenId { get; init; }

    /// <summary>Vision tower dimensions, or null for a text-only config.</summary>
    public DeepSeekV41VisionConfig? Vision { get; init; }

    /// <summary>Beginning-of-sequence token id.</summary>
    public int? BosTokenId { get; init; }

    /// <summary>End-of-sequence token id.</summary>
    public int? EosTokenId { get; init; }

    /// <summary>The per-layer plan inputs, backbone layers then draft layers.</summary>
    public IReadOnlyList<DeepSeekV41LayerPlan> LayerPlans { get; init; } = [];

    /// <summary>Backbone plus draft layers: the length <c>compress_ratios</c> must have.</summary>
    public int TotalLayerCount => NumHiddenLayers + NumNextnPredictLayers;

    /// <summary>Parses <paramref name="json"/> and validates its cross-field invariants.</summary>
    /// <exception cref="HartsyInferenceException">A field is missing or malformed, or an invariant fails; every invariant failure is listed at once.</exception>
    public static DeepSeekV41Config Parse(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(json);
        }
        catch (JsonException exception)
        {
            throw new HartsyInferenceException($"DeepSeek-V4.1 config is not valid JSON: {exception.Message}", exception);
        }
        using (document)
        {
            return FromElement(document.RootElement);
        }
    }

    /// <summary>Reads and parses the <c>config.json</c> at <paramref name="path"/>.</summary>
    public static DeepSeekV41Config Load(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        return Parse(File.ReadAllText(path));
    }

    private static DeepSeekV41Config FromElement(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object)
            throw new HartsyInferenceException("DeepSeek-V4.1 config must be a JSON object.");
        DeepSeekV41ConfigReader top = new(root, "config");
        string modelType = top.String("model_type");
        if (!string.Equals(modelType, ModelType, StringComparison.Ordinal))
            throw new HartsyInferenceException($"Config model_type is '{modelType}', expected '{ModelType}'.");

        DeepSeekV41ConfigReader text = top.Object("text_config") ?? top;
        DeepSeekV41Config config = new()
        {
            VocabSize = text.Int("vocab_size"),
            HiddenSize = text.Int("hidden_size"),
            MoeIntermediateSize = text.Int("moe_intermediate_size"),
            NumHiddenLayers = text.Int("num_hidden_layers"),
            NumAttentionHeads = text.Int("num_attention_heads"),
            NumKeyValueHeads = text.Int("num_key_value_heads"),
            HeadDim = text.Int("head_dim"),
            QkRopeHeadDim = text.Int("qk_rope_head_dim"),
            QLoraRank = text.Int("q_lora_rank"),
            OLoraRank = text.Int("o_lora_rank"),
            OGroups = text.Int("o_groups"),
            SwigluLimit = text.Double("swiglu_limit"),
            RmsNormEps = text.Double("rms_norm_eps"),
            MaxPositionEmbeddings = text.Int("max_position_embeddings"),
            RopeTheta = text.Double("rope_theta"),
            CompressRopeTheta = text.Double("compress_rope_theta"),
            RopeScaling = ReadRopeScaling(text),
            NRoutedExperts = text.Int("n_routed_experts"),
            NSharedExperts = text.Int("n_shared_experts"),
            NumExpertsPerTok = text.Int("num_experts_per_tok"),
            ScoringFunc = text.String("scoring_func"),
            TopkMethod = text.String("topk_method"),
            NormTopkProb = text.Bool("norm_topk_prob", fallback: true),
            RoutedScalingFactor = text.Double("routed_scaling_factor"),
            SlidingWindow = text.Int("sliding_window"),
            KvSourceLayerIds = text.IntArray("kv_source_layer_ids"),
            IndexSourceLayerIds = text.IntArray("index_source_layer_ids"),
            IndexNHeads = text.Int("index_n_heads"),
            IndexHeadDim = text.Int("index_head_dim"),
            IndexTopk = text.Int("index_topk"),
            CandidateSourceLayerId = text.Int("candidate_source_layer_id"),
            CandidateTopkBlocks = text.Int("candidate_topk_blocks"),
            CandidateBlockSize = text.Int("candidate_block_size"),
            HcMult = text.Int("hc_mult"),
            HcSinkhornIters = text.Int("hc_sinkhorn_iters"),
            HcEps = text.Double("hc_eps"),
            EngramLayerIds = text.IntArray("engram_layer_ids"),
            EngramNumEmbeddings = text.LongArray("engram_num_embeddings"),
            EngramMaxNgramSize = text.Int("engram_max_ngram_size"),
            EngramVocabSize = text.Int("engram_vocab_size"),
            EngramNHeads = text.Int("engram_n_heads"),
            EngramHeadDim = text.Int("engram_head_dim"),
            EngramPadTokenId = text.Int("engram_pad_token_id"),
            EngramCompressedVocabSize = text.Int("engram_compressed_vocab_size"),
            NumNextnPredictLayers = text.Int("num_nextn_predict_layers"),
            DsparkBlockSize = text.Int("dspark_block_size"),
            DsparkNoiseTokenId = text.Int("dspark_noise_token_id"),
            DsparkTargetLayerIds = text.IntArray("dspark_target_layer_ids"),
            DsparkMarkovRank = text.Int("dspark_markov_rank"),
            DsparkNRoutedExperts = text.Int("dspark_n_routed_experts"),
            DsparkNumExpertsPerTok = text.Int("dspark_num_experts_per_tok"),
            CompressRatios = text.IntArray("compress_ratios"),
            ImageTokenId = top.Has("image_token_id") ? top.Int("image_token_id") : null,
            Vision = ReadVision(top),
            BosTokenId = top.Has("bos_token_id") ? top.Int("bos_token_id") : null,
            EosTokenId = top.Has("eos_token_id") ? top.Int("eos_token_id") : null,
        };
        config.Validate();
        return config with { LayerPlans = config.BuildLayerPlans() };
    }

    private static DeepSeekV41RopeScaling? ReadRopeScaling(DeepSeekV41ConfigReader text)
    {
        DeepSeekV41ConfigReader? scaling = text.Object("rope_scaling");
        return scaling is null ? null : new DeepSeekV41RopeScaling(scaling.Value.Double("factor"),
            scaling.Value.Double("beta_fast"), scaling.Value.Double("beta_slow"), scaling.Value.Int("original_max_position_embeddings"));
    }

    private static DeepSeekV41VisionConfig? ReadVision(DeepSeekV41ConfigReader top)
    {
        if (top.Object("vision_config") is { } nested)
        {
            return new DeepSeekV41VisionConfig(nested.Int("num_hidden_layers"), nested.Int("hidden_size"),
                nested.Int("num_attention_heads"), nested.Int("intermediate_size"), nested.Int("patch_size"),
                nested.Int("downsample_ratio"), nested.Double("rope_theta", DeepSeekV41VisionConfig.DefaultRopeTheta));
        }
        if (!top.Has("vision_num_layers"))
            return null;
        return new DeepSeekV41VisionConfig(top.Int("vision_num_layers"), top.Int("vision_hidden_size"),
            top.Int("vision_num_heads"), top.Int("vision_intermediate_size"), top.Int("vision_patch_size"),
            top.Int("vision_downsample_ratio"), top.Double("vision_rope_theta", DeepSeekV41VisionConfig.DefaultRopeTheta));
    }

    private void Validate()
    {
        List<string> problems = new List<string>();
        if (CompressRatios.Count != TotalLayerCount)
        {
            problems.Add(string.Create(CultureInfo.InvariantCulture,
                $"compress_ratios has {CompressRatios.Count} entries, expected num_hidden_layers + num_nextn_predict_layers = {TotalLayerCount}"));
        }
        if (CompressRatios.Any(static ratio => ratio < 0))
            problems.Add("compress_ratios contains a negative entry");
        if (EngramLayerIds.Count != EngramNumEmbeddings.Count)
        {
            problems.Add(string.Create(CultureInfo.InvariantCulture,
                $"engram_layer_ids has {EngramLayerIds.Count} entries but engram_num_embeddings has {EngramNumEmbeddings.Count}"));
        }
        RequireBackbone(problems, "engram_layer_ids", EngramLayerIds);
        RequireBackbone(problems, "kv_source_layer_ids", KvSourceLayerIds);
        RequireBackbone(problems, "index_source_layer_ids", IndexSourceLayerIds);
        RequireBackbone(problems, "dspark_target_layer_ids", DsparkTargetLayerIds);
        if (NumExpertsPerTok > NRoutedExperts)
            problems.Add($"num_experts_per_tok {NumExpertsPerTok} exceeds n_routed_experts {NRoutedExperts}");
        if (DsparkNumExpertsPerTok > DsparkNRoutedExperts)
            problems.Add($"dspark_num_experts_per_tok {DsparkNumExpertsPerTok} exceeds dspark_n_routed_experts {DsparkNRoutedExperts}");
        if (problems.Count > 0)
            throw new HartsyInferenceException("DeepSeek-V4.1 config is inconsistent: " + string.Join("; ", problems) + ".");
    }

    private void RequireBackbone(List<string> problems, string field, IReadOnlyList<int> ids)
    {
        foreach (int id in ids)
        {
            if (id < 0 || id >= NumHiddenLayers)
                problems.Add($"{field} entry {id} is outside the {NumHiddenLayers} backbone layers");
        }
    }

    private IReadOnlyList<DeepSeekV41LayerPlan> BuildLayerPlans()
    {
        DeepSeekV41LayerPlan[] plans = new DeepSeekV41LayerPlan[TotalLayerCount];
        for (int layer = 0; layer < plans.Length; layer++)
        {
            bool draft = layer >= NumHiddenLayers;
            int slot = draft ? -1 : EngramLayerIds.ToList().IndexOf(layer);
            plans[layer] = new DeepSeekV41LayerPlan(layer, draft, CompressRatios[layer],
                !draft && KvSourceLayerIds.Contains(layer), !draft && IndexSourceLayerIds.Contains(layer),
                slot >= 0 ? slot : null);
        }
        return plans;
    }
}
