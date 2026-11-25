using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>
/// The Tencent Hy4 Flash config (<c>model_type: "hy_v4_flash"</c>), read from a real <c>config.json</c> with the legacy names the shipped
/// files use (the transformers reference accepts them as aliases: <c>deepseek_sparse_attention</c>, <c>compressed_indexer_*</c>, <c>swa_*</c>,
/// <c>routed_expert_hidden_size</c>). Reference: transformers PR 50903 at <c>eb1d5db</c>, <c>configuration_hy_v4_flash.py</c>.
/// </summary>
public sealed record Hy4FlashConfig
{
    /// <summary>The <c>model_type</c> value this reader accepts.</summary>
    public const string ModelType = "hy_v4_flash";

    /// <summary>Vocabulary size (<c>text_config.vocab_size</c>).</summary>
    public required int VocabSize { get; init; }

    /// <summary>Residual stream width (<c>hidden_size</c>).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Decoder layers (<c>num_hidden_layers</c>).</summary>
    public required int NumHiddenLayers { get; init; }

    /// <summary>Attention kind of every layer.</summary>
    public required IReadOnlyList<Hy4FlashLayerKind> LayerKinds { get; init; }

    /// <summary>Whether each layer's FFN is a routed MoE (<c>mlp_layer_types</c>); false is the dense SwiGLU.</summary>
    public required IReadOnlyList<bool> SparseFfn { get; init; }

    /// <summary>Query heads of an indexed layer (<c>num_attention_heads</c>).</summary>
    public required int NumAttentionHeads { get; init; }

    /// <summary>Query heads of a sliding layer (<c>swa_num_attention_heads</c>).</summary>
    public required int SlidingAttentionHeads { get; init; }

    /// <summary>Query latent width (<c>q_lora_rank</c>).</summary>
    public required int QLoraRank { get; init; }

    /// <summary>Key/value latent width (<c>kv_lora_rank</c>).</summary>
    public required int KvLoraRank { get; init; }

    /// <summary>Per-head no-position query/key width (<c>qk_nope_head_dim</c>).</summary>
    public required int QkNopeHeadDim { get; init; }

    /// <summary>Per-head rotated query/key width (<c>qk_rope_head_dim</c>).</summary>
    public required int QkRopeHeadDim { get; init; }

    /// <summary>Per-head value width (<c>v_head_dim</c>).</summary>
    public required int VHeadDim { get; init; }

    /// <summary>Keys a sliding layer attends, its own position included (<c>sliding_window</c>).</summary>
    public required int SlidingWindow { get; init; }

    /// <summary>Per-head no-position query/key width of a sliding layer (<c>swa_qk_nope_head_dim</c>); must equal <see cref="QkNopeHeadDim"/>.</summary>
    public int SlidingQkNopeHeadDim { get; init; }

    /// <summary>Per-head rotated width of a sliding layer (<c>swa_qk_rope_head_dim</c>); must equal <see cref="QkRopeHeadDim"/>.</summary>
    public int SlidingQkRopeHeadDim { get; init; }

    /// <summary>Per-head value width of a sliding layer (<c>swa_v_head_dim</c>); must equal <see cref="VHeadDim"/>.</summary>
    public int SlidingVHeadDim { get; init; }

    /// <summary>Initial per-head sink logit (<c>learnable_sink_init</c>).</summary>
    public double LearnableSinkInit { get; init; }

    /// <summary>Dense FFN width (<c>intermediate_size</c>).</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Routed and shared expert FFN width (<c>moe_intermediate_size</c>).</summary>
    public required int MoeIntermediateSize { get; init; }

    /// <summary>Width the routed experts run at (<c>routed_expert_hidden_size</c>).</summary>
    public required int MoeHiddenSize { get; init; }

    /// <summary>Routed experts per MoE layer (<c>n_routed_experts</c>).</summary>
    public required int NRoutedExperts { get; init; }

    /// <summary>Shared experts (<c>n_shared_experts</c>).</summary>
    public required int NSharedExperts { get; init; }

    /// <summary>Routed experts per token (<c>num_experts_per_tok</c>).</summary>
    public required int NumExpertsPerTok { get; init; }

    /// <summary>Router expert groups (<c>n_group</c>).</summary>
    public int NGroup { get; init; } = 1;

    /// <summary>Router groups kept (<c>topk_group</c>).</summary>
    public int TopkGroup { get; init; } = 1;

    /// <summary>Renormalizes the picked routing weights (<c>norm_topk_prob</c>).</summary>
    public bool NormTopkProb { get; init; } = true;

    /// <summary>Multiplier on the routed output (<c>routed_scaling_factor</c>).</summary>
    public double RoutedScalingFactor { get; init; } = 1.0;

    /// <summary>Clamp bound of the expert SwiGLU (<c>swiglu_limit</c>).</summary>
    public double SwigluLimit { get; init; } = 10.0;

    /// <summary>Tokens pooled into one indexer group (<c>compressed_indexer_compress_ratio</c>).</summary>
    public required int IndexKpool { get; init; }

    /// <summary>Keys an indexed layer may attend once its context is longer (<c>index_topk</c>).</summary>
    public required int IndexTopk { get; init; }

    /// <summary>Indexer heads (<c>index_n_heads</c>).</summary>
    public required int IndexNHeads { get; init; }

    /// <summary>Indexer head width (<c>index_head_dim</c>).</summary>
    public required int IndexHeadDim { get; init; }

    /// <summary>Residual stream copies (<c>hc_mult</c>).</summary>
    public required int HcMult { get; init; }

    /// <summary>Gate epsilon of the hyper-connections (<c>hc_eps</c>).</summary>
    public required double HcEps { get; init; }

    /// <summary>Multiplier on the post gate (<c>hc_magnitude</c>).</summary>
    public required double HcMagnitude { get; init; }

    /// <summary>RoPE base of an indexed layer.</summary>
    public required double IndexedRopeTheta { get; init; }

    /// <summary>RoPE base of a sliding layer.</summary>
    public required double SlidingRopeTheta { get; init; }

    /// <summary>Time, height and width pair counts of the interleaved M-RoPE, summing to half the rotated width.</summary>
    public required IReadOnlyList<int> MropeSection { get; init; }

    /// <summary>Engram layer ids as the config lists them; the decoder layer index is the id minus one.</summary>
    public required IReadOnlyList<int> EngramLayerIds { get; init; }

    /// <summary>Longest n-gram (<c>engram_max_ngram_size</c>); orders 2 up to this are hashed.</summary>
    public int EngramMaxNgramSize { get; init; } = 3;

    /// <summary>Hash heads per n-gram order (<c>engram_n_head_per_ngram</c>).</summary>
    public int EngramHeadsPerNgram { get; init; } = 8;

    /// <summary>Embedding width per n-gram order, summed over its heads (<c>engram_n_embed_per_ngram</c>).</summary>
    public int EngramEmbedPerNgram { get; init; } = 768;

    /// <summary>Token id that reads as padding (<c>engram_pad_id</c>).</summary>
    public int EngramPadId { get; init; } = 120002;

    /// <summary>Seed of the per-layer hash multiplier draw (<c>engram_seed</c>).</summary>
    public int EngramSeed { get; init; }

    /// <summary>Whether looked-up rows take an FP8 round trip (<c>engram_fp8_embedding</c>); true is the reference default and the base config's value, false in the instruct ones.</summary>
    public bool EngramFp8Embedding { get; init; } = true;

    /// <summary>Length of the frozen token map (<c>engram_lookup_table_size</c>, default 120,818 as in the reference).</summary>
    public int EngramLookupTableSize { get; init; } = 120818;

    /// <summary>Hash moduli per Engram layer id: one array of <see cref="EngramHeadsPerNgram"/> primes per n-gram order.</summary>
    public required IReadOnlyDictionary<int, long[][]> EngramHeadVocabSizes { get; init; }

    /// <summary>Norm epsilon (<c>rms_norm_eps</c>).</summary>
    public required double RmsNormEps { get; init; }

    /// <summary>Positions the model was trained for (<c>max_position_embeddings</c>).</summary>
    public required int MaxPositionEmbeddings { get; init; }

    /// <summary>Whether the head shares the embedding (<c>tie_word_embeddings</c>).</summary>
    public bool TieWordEmbeddings { get; init; }

    /// <summary>Beginning-of-sequence id (<c>text_config.bos_token_id</c>).</summary>
    public int? BosTokenId { get; init; }

    /// <summary>End-of-sequence id (<c>text_config.eos_token_id</c>).</summary>
    public int? EosTokenId { get; init; }

    /// <summary>Quantisation method of the text weights (<c>text_config.quantization_config.quant_method</c>), or null for an unquantised checkpoint.</summary>
    public string? QuantMethod { get; init; }

    /// <summary>Quantisation format (<c>fmt</c>).</summary>
    public string? QuantFormat { get; init; }

    /// <summary>Block edges of the weight scales (<c>weight_block_size</c>), empty when unquantised.</summary>
    public IReadOnlyList<int> QuantBlock { get; init; } = [];

    /// <summary>Draft layers of the speculative drafter (<c>dfly_num_hidden_layers</c>), or 0 when the checkpoint has none.</summary>
    public int DflyNumHiddenLayers { get; init; }

    /// <summary>Vision tower and projector.</summary>
    public required Hy4FlashVisionConfig Vision { get; init; }

    /// <summary>Image placeholder token id (<c>image_token_id</c>).</summary>
    public required int ImageTokenId { get; init; }

    /// <summary>Id before an image's tokens (<c>image_start_token_id</c>).</summary>
    public required int ImageStartTokenId { get; init; }

    /// <summary>Id after an image's tokens (<c>image_end_token_id</c>).</summary>
    public required int ImageEndTokenId { get; init; }

    /// <summary>Id that closes each merged image row (<c>image_newline_token_id</c>).</summary>
    public required int ImageNewlineTokenId { get; init; }

    /// <summary>Video placeholder token id (<c>video_token_id</c>).</summary>
    public required int VideoTokenId { get; init; }

    /// <summary>Per-head query/key width.</summary>
    public int QkHeadDim => QkNopeHeadDim + QkRopeHeadDim;

    /// <summary>Whether the weights are block-scaled FP8.</summary>
    public bool IsFp8 => string.Equals(QuantMethod, "fp8", StringComparison.Ordinal);

    /// <summary>Query heads of <paramref name="layer"/>.</summary>
    public int HeadsFor(int layer) => LayerKinds[layer] == Hy4FlashLayerKind.Indexed ? NumAttentionHeads : SlidingAttentionHeads;

    /// <summary>Whether decoder layer <paramref name="layer"/> carries an Engram memory.</summary>
    public bool HasEngram(int layer) => EngramLayerIds.Contains(layer + 1);

    /// <summary>Hash heads per Engram layer: n-gram orders 2 up to the maximum, each with its own heads.</summary>
    public int EngramHashHeads => (EngramMaxNgramSize - 1) * EngramHeadsPerNgram;

    /// <summary>Rows of the flat embedding table of Engram layer id <paramref name="layerId"/>: the sum of its hash moduli.</summary>
    public long EngramTableRows(int layerId) => EngramHeadVocabSizes[layerId].SelectMany(static order => order).Sum();

    /// <summary>Parses a <c>config.json</c> document.</summary>
    /// <exception cref="HartsyInferenceException">The document is not a <c>hy_v4_flash</c> config, a required field is missing, or the shape is inconsistent.</exception>
    public static Hy4FlashConfig Parse(string json)
    {
        try
        {
            return ParseCore(json);
        }
        catch (Exception ex) when (ex is InvalidOperationException or FormatException or KeyNotFoundException or JsonException or ArgumentException)
        {
            throw new HartsyInferenceException($"Hy4 Flash config: {ex.Message}", ex);
        }
    }

    private static Hy4FlashConfig ParseCore(string json)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        HfConfigFields top = new(root, "config", "Hy4 Flash");
        string modelType = top.OptionalString("model_type") ?? string.Empty;
        if (!string.Equals(modelType, ModelType, StringComparison.Ordinal))
            throw new HartsyInferenceException($"Hy4 Flash config: model_type is '{modelType}', expected '{ModelType}'.");
        HfConfigFields t = top.Object("text_config");
        JsonElement text = root.GetProperty("text_config");
        HfConfigFields v = top.Object("vision_config");

        int layers = t.Int("num_hidden_layers");
        int heads = t.Int("num_attention_heads");
        int nope = t.Int("qk_nope_head_dim"), rope = t.Int("qk_rope_head_dim");
        Hy4FlashLayerKind[] kinds = ReadLayerKinds(text, layers);
        bool[] sparse = ReadSparseFfn(text, layers);
        (double indexedTheta, double slidingTheta, int[] section) = ReadRope(text);

        string? quantMethod = null, quantFormat = null;
        int[] block = [];
        if (text.TryGetProperty("quantization_config", out JsonElement quant) && quant.ValueKind == JsonValueKind.Object)
        {
            HfConfigFields q = t.Object("quantization_config");
            quantMethod = q.OptionalString("quant_method");
            quantFormat = q.OptionalString("fmt");
            if (quant.TryGetProperty("weight_block_size", out JsonElement b) && b.ValueKind == JsonValueKind.Array) block = q.Ints("weight_block_size");
        }

        Hy4FlashConfig config = new()
        {
            VocabSize = t.Int("vocab_size"),
            HiddenSize = t.Int("hidden_size"),
            NumHiddenLayers = layers,
            LayerKinds = kinds,
            SparseFfn = sparse,
            NumAttentionHeads = heads,
            SlidingAttentionHeads = t.Int("swa_num_attention_heads", 64),
            QLoraRank = t.Int("q_lora_rank"),
            KvLoraRank = t.Int("kv_lora_rank"),
            QkNopeHeadDim = nope,
            QkRopeHeadDim = rope,
            VHeadDim = t.Int("v_head_dim"),
            SlidingQkNopeHeadDim = t.Int("swa_qk_nope_head_dim", nope),
            SlidingQkRopeHeadDim = t.Int("swa_qk_rope_head_dim", rope),
            SlidingVHeadDim = t.Int("swa_v_head_dim", t.Int("v_head_dim")),
            SlidingWindow = t.Int("sliding_window", 512),
            LearnableSinkInit = t.Double("learnable_sink_init", 0.0),
            IntermediateSize = t.Int("intermediate_size"),
            MoeIntermediateSize = t.Int("moe_intermediate_size"),
            MoeHiddenSize = t.Int("routed_expert_hidden_size", t.Int("moe_hidden_size", t.Int("hidden_size"))),
            NRoutedExperts = t.Int("n_routed_experts"),
            NSharedExperts = t.Int("n_shared_experts"),
            NumExpertsPerTok = t.Int("num_experts_per_tok"),
            NGroup = t.Int("n_group", 1),
            TopkGroup = t.Int("topk_group", 1),
            NormTopkProb = t.Bool("norm_topk_prob", true),
            RoutedScalingFactor = t.Double("routed_scaling_factor", 1.0),
            SwigluLimit = t.Double("swiglu_limit", 10.0),
            IndexKpool = t.Int("compressed_indexer_compress_ratio", t.Int("index_kpool", 4)),
            IndexTopk = t.Int("index_topk"),
            IndexNHeads = t.Int("index_n_heads"),
            IndexHeadDim = t.Int("index_head_dim"),
            HcMult = t.Int("hc_mult"),
            HcEps = t.Double("hc_eps"),
            HcMagnitude = t.Double("hc_magnitude", 2.0),
            IndexedRopeTheta = indexedTheta,
            SlidingRopeTheta = slidingTheta,
            MropeSection = section,
            EngramLayerIds = text.TryGetProperty("engram_layer_ids", out JsonElement ids) && ids.ValueKind == JsonValueKind.Array
                ? [.. ids.EnumerateArray().Select(static e => e.GetInt32())] : [],
            EngramMaxNgramSize = t.Int("engram_max_ngram_size", 3),
            EngramHeadsPerNgram = t.Int("engram_n_head_per_ngram", 8),
            EngramEmbedPerNgram = t.Int("engram_n_embed_per_ngram", 768),
            EngramPadId = t.Int("engram_pad_id", 120002),
            EngramSeed = t.Int("engram_seed", 0),
            EngramFp8Embedding = t.Bool("engram_fp8_embedding", true),
            EngramLookupTableSize = t.Int("engram_lookup_table_size", 120818),
            EngramHeadVocabSizes = ReadEngramVocab(text),
            RmsNormEps = t.Double("rms_norm_eps", 1e-5),
            MaxPositionEmbeddings = t.Int("max_position_embeddings"),
            TieWordEmbeddings = t.Bool("tie_word_embeddings", false),
            BosTokenId = OptionalInt(text, "bos_token_id"),
            EosTokenId = OptionalInt(text, "eos_token_id"),
            QuantMethod = quantMethod,
            QuantFormat = quantFormat,
            QuantBlock = block,
            DflyNumHiddenLayers = top.Int("dfly_num_hidden_layers", 0),
            Vision = new Hy4FlashVisionConfig
            {
                HiddenSize = v.Int("hidden_size"),
                IntermediateSize = v.Int("intermediate_size"),
                NumHiddenLayers = v.Int("num_hidden_layers"),
                NumAttentionHeads = v.Int("num_attention_heads"),
                OutHiddenSize = v.Int("out_hidden_size"),
                PatchSize = v.Int("patch_size"),
                SpatialMergeSize = v.Int("spatial_merge_size"),
                MaxImageSize = v.Int("max_image_size"),
                RmsNormEps = v.Double("rms_norm_eps", 1e-5),
            },
            ImageTokenId = top.Int("image_token_id"),
            ImageStartTokenId = top.Int("image_start_token_id"),
            ImageEndTokenId = top.Int("image_end_token_id"),
            ImageNewlineTokenId = top.Int("image_newline_token_id"),
            VideoTokenId = top.Int("video_token_id"),
        };
        config.Validate();
        return config;
    }

    /// <summary>Reads and parses the <c>config.json</c> at <paramref name="path"/>.</summary>
    public static Hy4FlashConfig Load(string path) => Parse(File.ReadAllText(path));

    /// <summary>Fails fast on a config the reference itself would refuse (<c>validate_architecture</c>) or whose parts disagree.</summary>
    /// <exception cref="HartsyInferenceException">A rule is broken.</exception>
    public void Validate()
    {
        if (IndexKpool < 1) throw Bad($"index_kpool must be positive, got {IndexKpool}.");
        if (IndexTopk % IndexKpool != 0) throw Bad($"index_topk ({IndexTopk}) must be divisible by index_kpool ({IndexKpool}).");
        if (MropeSection.Count != 3 || MropeSection.Sum() != QkRopeHeadDim / 2)
            throw Bad($"mrope_section [{string.Join(", ", MropeSection)}] must hold three entries summing to {QkRopeHeadDim / 2}.");
        if (LayerKinds.Count != NumHiddenLayers || SparseFfn.Count != NumHiddenLayers)
            throw Bad("layer_types and mlp_layer_types must each list num_hidden_layers entries.");
        if (HcMult < 1) throw Bad($"hc_mult must be positive, got {HcMult}.");
        if (NRoutedExperts < 1 || NumExpertsPerTok < 1 || NumExpertsPerTok > NRoutedExperts)
            throw Bad($"num_experts_per_tok ({NumExpertsPerTok}) must be between 1 and n_routed_experts ({NRoutedExperts}).");
        if (NGroup < 1 || TopkGroup < 1 || TopkGroup > NGroup || NRoutedExperts % NGroup != 0)
            throw Bad($"n_group {NGroup} and topk_group {TopkGroup} must divide the {NRoutedExperts} experts with topk_group at most n_group.");
        if (QkRopeHeadDim % 2 != 0) throw Bad($"qk_rope_head_dim ({QkRopeHeadDim}) must be even.");
        if (Vision.MaxImageSize % Vision.PatchSize != 0)
            throw Bad($"vision max_image_size ({Vision.MaxImageSize}) must be a multiple of the patch size ({Vision.PatchSize}).");
        if (TieWordEmbeddings) throw Bad("tied word embeddings are not read by this reference (the LM head is a separate tensor).");
        if (SlidingQkNopeHeadDim != QkNopeHeadDim || SlidingQkRopeHeadDim != QkRopeHeadDim || SlidingVHeadDim != VHeadDim)
            throw Bad("sliding layers with head widths different from the indexed layers' are not supported (only the head count differs in every shipped file).");
        if (EngramLayerIds.Count > 0)
        {
            if (EngramPadId >= EngramLookupTableSize)
                throw Bad($"engram_lookup_table_size ({EngramLookupTableSize}) must cover engram_pad_id ({EngramPadId}).");
            int orders = EngramMaxNgramSize - 1;
            foreach (int id in EngramLayerIds)
            {
                if (id < 1 || id > NumHiddenLayers) throw Bad($"engram_layer_ids entry {id} is outside 1..{NumHiddenLayers}.");
                if (!EngramHeadVocabSizes.TryGetValue(id, out long[][]? sizes) || sizes.Length != orders
                    || sizes.Any(o => o.Length != EngramHeadsPerNgram || o.Any(m => m <= 0)))
                    throw Bad($"engram_layer_head_vocab_sizes['{id}'] must hold {orders} lists of {EngramHeadsPerNgram} sizes.");
            }
            if (EngramEmbedPerNgram % EngramHeadsPerNgram != 0) throw Bad("engram_n_embed_per_ngram must divide by the head count.");
        }
        if (IsFp8 && (!string.Equals(QuantFormat ?? "e4m3", "e4m3", StringComparison.Ordinal) || QuantBlock is not [128, 128]))
            throw Bad($"quantization '{QuantMethod}' {QuantFormat} [{string.Join(", ", QuantBlock)}]: only fp8 e4m3 128x128 blocks are known.");
    }

    private static HartsyInferenceException Bad(string message) => new($"Hy4 Flash config: {message}");

    private static int? OptionalInt(JsonElement o, string name) =>
        o.TryGetProperty(name, out JsonElement e) && e.ValueKind == JsonValueKind.Number ? e.GetInt32() : null;

    // The shipped file says "deepseek_sparse_attention"; the reference renames it to "indexed_attention". With no list the reference derives
    // two indexed layers, then one per four.
    private static Hy4FlashLayerKind[] ReadLayerKinds(JsonElement text, int layers)
    {
        Hy4FlashLayerKind[] kinds = new Hy4FlashLayerKind[layers];
        if (!text.TryGetProperty("layer_types", out JsonElement list) || list.ValueKind != JsonValueKind.Array)
        {
            for (int i = 0; i < layers; i++) kinds[i] = i < 2 || (i - 1) % 4 == 0 ? Hy4FlashLayerKind.Indexed : Hy4FlashLayerKind.Sliding;
            return kinds;
        }
        if (list.GetArrayLength() != layers) throw Bad($"layer_types lists {list.GetArrayLength()} entries for {layers} layers.");
        int n = 0;
        foreach (JsonElement e in list.EnumerateArray())
            kinds[n++] = e.GetString() switch
            {
                "deepseek_sparse_attention" or "indexed_attention" => Hy4FlashLayerKind.Indexed,
                "sliding_attention" => Hy4FlashLayerKind.Sliding,
                string other => throw Bad($"unknown layer type '{other}'."),
                null => throw Bad("layer_types holds a null entry."),
            };
        return kinds;
    }

    private static bool[] ReadSparseFfn(JsonElement text, int layers)
    {
        bool[] sparse = new bool[layers];
        if (!text.TryGetProperty("mlp_layer_types", out JsonElement list) || list.ValueKind != JsonValueKind.Array)
        {
            for (int i = 1; i < layers; i++) sparse[i] = true;
            return sparse;
        }
        if (list.GetArrayLength() != layers) throw Bad($"mlp_layer_types lists {list.GetArrayLength()} entries for {layers} layers.");
        int n = 0;
        foreach (JsonElement e in list.EnumerateArray())
            sparse[n++] = e.GetString() switch
            {
                "sparse" => true,
                "dense" => false,
                string other => throw Bad($"unknown mlp layer type '{other}'."),
                null => throw Bad("mlp_layer_types holds a null entry."),
            };
        return sparse;
    }

    // rope_parameters is keyed by layer type, the indexed one under its legacy name. The reference allows a section per type; the engine takes
    // one shared section, which is what the shipped file carries, and refuses a file where the two differ.
    private static (double Indexed, double Sliding, int[] Section) ReadRope(JsonElement text)
    {
        double indexed = 10_000_000.0, sliding = 10_000.0;
        int[]? indexedSection = null, slidingSection = null;
        if (text.TryGetProperty("rope_parameters", out JsonElement parameters) && parameters.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty p in parameters.EnumerateObject())
            {
                bool isIndexed = p.Name is "deepseek_sparse_attention" or "indexed_attention";
                if ((!isIndexed && p.Name != "sliding_attention") || p.Value.ValueKind != JsonValueKind.Object) continue;
                if (p.Value.TryGetProperty("rope_theta", out JsonElement theta) && theta.ValueKind == JsonValueKind.Number)
                {
                    if (isIndexed) indexed = theta.GetDouble();
                    else sliding = theta.GetDouble();
                }
                if (p.Value.TryGetProperty("mrope_section", out JsonElement s) && s.ValueKind == JsonValueKind.Array)
                {
                    int[] read = [.. s.EnumerateArray().Select(static e => e.GetInt32())];
                    if (isIndexed) indexedSection = read;
                    else slidingSection = read;
                }
            }
        }
        int[] section = indexedSection ?? slidingSection ?? [12, 10, 10];
        if (!(indexedSection ?? section).AsSpan().SequenceEqual(slidingSection ?? section))
            throw Bad("the indexed and sliding layers carry different mrope_section values; one shared section is supported.");
        return (indexed, sliding, section);
    }

    private static Dictionary<int, long[][]> ReadEngramVocab(JsonElement text)
    {
        Dictionary<int, long[][]> result = [];
        if (!text.TryGetProperty("engram_layer_head_vocab_sizes", out JsonElement map) || map.ValueKind != JsonValueKind.Object) return result;
        foreach (JsonProperty p in map.EnumerateObject())
        {
            if (!int.TryParse(p.Name, System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int id))
                throw Bad($"engram_layer_head_vocab_sizes key '{p.Name}' is not a layer id.");
            result[id] = [.. p.Value.EnumerateArray().Select(static order => order.EnumerateArray().Select(static e => e.GetInt64()).ToArray())];
        }
        return result;
    }
}
