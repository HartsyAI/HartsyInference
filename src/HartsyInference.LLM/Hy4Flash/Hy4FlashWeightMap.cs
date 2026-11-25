using System.Globalization;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>
/// Maps every Hy4 Flash checkpoint key to the parameter it feeds and the shape the config implies, read from the real indexes of
/// <c>tencent/Hy4-flash</c>, <c>-FP8</c> and <c>-base</c> and the transformers reference (<c>conversion_mapping.py</c> gives the renames).
/// </summary>
public static class Hy4FlashWeightMap
{
    /// <summary>Suffix of a block-scale companion: <c>X.weight_scale_inv</c> for a linear, <c>experts.X_scale_inv</c> for a fused stack.</summary>
    public const string ScaleSuffix = "_scale_inv";

    /// <summary>Prefix of the speculative drafter's tensors.</summary>
    public const string DrafterPrefix = "model.dfly_layers.";

    /// <summary>Prefix of the vision tower and projector.</summary>
    public const string VisionPrefix = "vit.";

    private const string Fp8Dtype = "F8_E4M3";
    private const int BlockEdge = 128;
    private static readonly string[] F32Only = ["F32"];
    private static readonly string[] I32Only = ["I32"];
    private static readonly string[] ScaleDtypes = ["BF16", "F32"];

    /// <summary>Prefix of decoder layer <paramref name="layer"/>, without a trailing dot.</summary>
    public static string Layer(int layer) => string.Create(CultureInfo.InvariantCulture, $"model.layers.{layer}");

    /// <summary>The scale-grid key of FP8 weight <paramref name="weightKey"/>.</summary>
    public static string ScaleKey(string weightKey) => weightKey + ScaleSuffix;

    /// <summary>Every parameter the model reads, scale companions and the drafter excluded.</summary>
    public static IReadOnlyList<Hy4FlashExpectedTensor> Expected(Hy4FlashConfig c, bool includeVision = true)
    {
        ArgumentNullException.ThrowIfNull(c);
        List<Hy4FlashExpectedTensor> list = [];
        long h = c.HiddenSize, hc = c.HcMult;
        list.Add(new("model.embed_tokens.weight", [c.VocabSize, h]));
        list.Add(new("model.norm.weight", [h]));
        list.Add(new("lm_head.weight", [c.VocabSize, h]));
        list.Add(new("model.hc_head.hc_head_fn", [hc, hc * h], F32Only));
        list.Add(new("model.hc_head.hc_head_base", [hc], F32Only));
        list.Add(new("model.hc_head.hc_head_scale", [1], F32Only));
        for (int layer = 0; layer < c.NumHiddenLayers; layer++) AddLayer(c, layer, list);
        if (includeVision) AddVision(c.Vision, list);
        return list;
    }

    /// <summary>Classifies, shape-checks and dtype-checks <paramref name="tensors"/> (a header inventory) against <paramref name="config"/>.</summary>
    public static Hy4FlashWeightCoverage Check(Hy4FlashConfig config, IEnumerable<Hy4FlashTensorInfo> tensors, bool includeVision = true)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(tensors);
        Dictionary<string, Hy4FlashExpectedTensor> expected = Expected(config, includeVision)
            .ToDictionary(static e => e.Key, static e => e, StringComparer.Ordinal);
        Dictionary<string, Hy4FlashTensorInfo> all = tensors.ToDictionary(static t => t.Name, StringComparer.Ordinal);
        HashSet<string> seen = new(StringComparer.Ordinal);
        List<string> unmapped = [], shapes = [], scales = [], dtypes = [];
        int mapped = 0, companions = 0, drafter = 0;
        foreach (Hy4FlashTensorInfo t in all.Values)
        {
            if (t.Name.StartsWith(DrafterPrefix, StringComparison.Ordinal))
            {
                drafter++;
                continue;
            }
            bool companion = t.Name.EndsWith(ScaleSuffix, StringComparison.Ordinal);
            string weight = companion ? t.Name[..^ScaleSuffix.Length] : t.Name;
            if (!expected.TryGetValue(weight, out Hy4FlashExpectedTensor entry))
            {
                unmapped.Add(t.Name);
                continue;
            }
            if (companion)
            {
                companions++;
                CheckScale(t, all.TryGetValue(weight, out Hy4FlashTensorInfo w) ? w : null, entry.Shape, scales);
                continue;
            }
            mapped++;
            seen.Add(t.Name);
            if (!t.Shape.AsSpan().SequenceEqual(entry.Shape))
                shapes.Add($"{t.Name} is [{string.Join(", ", t.Shape)}], expected [{string.Join(", ", entry.Shape)}]");
            if (entry.Dtypes is { } allowed && !allowed.Contains(t.DType, StringComparer.Ordinal))
                dtypes.Add($"{t.Name} is {t.DType}, expected {string.Join(" or ", allowed)}");
            if (string.Equals(t.DType, Fp8Dtype, StringComparison.Ordinal) && !all.ContainsKey(ScaleKey(t.Name)))
                scales.Add($"{t.Name} is FP8 with no scale grid");
        }
        List<string> missing = [.. expected.Keys.Where(k => !seen.Contains(k))];
        return new Hy4FlashWeightCoverage(new CheckpointKeyCoverage("Hy4 Flash", mapped, companions, drafter, unmapped, missing), shapes, scales, dtypes);
    }

    /// <summary>Which storage class a checkpoint key belongs to.</summary>
    public static Hy4FlashWeightClass Classify(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        if (key.StartsWith(DrafterPrefix, StringComparison.Ordinal)) return Hy4FlashWeightClass.Drafter;
        if (key.StartsWith(VisionPrefix, StringComparison.Ordinal)) return Hy4FlashWeightClass.Vision;
        if (key.Contains(".engram_layer.multi_head_embedding.", StringComparison.Ordinal)) return Hy4FlashWeightClass.EngramTable;
        if (key.Contains(".mlp.experts.", StringComparison.Ordinal)) return Hy4FlashWeightClass.RoutedExperts;
        return Hy4FlashWeightClass.PerToken;
    }

    /// <summary>Bytes per storage class over <paramref name="tensors"/>.</summary>
    public static IReadOnlyDictionary<Hy4FlashWeightClass, long> Summarize(IEnumerable<Hy4FlashTensorInfo> tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        Dictionary<Hy4FlashWeightClass, long> totals = [];
        foreach (Hy4FlashTensorInfo t in tensors)
        {
            Hy4FlashWeightClass kind = Classify(t.Name);
            totals[kind] = totals.GetValueOrDefault(kind) + t.Bytes;
        }
        return totals;
    }

    private static void CheckScale(Hy4FlashTensorInfo scale, Hy4FlashTensorInfo? weight, long[] weightShape, List<string> problems)
    {
        if (weight is not { } w)
        {
            problems.Add($"{scale.Name} has no weight beside it");
            return;
        }
        if (!string.Equals(w.DType, Fp8Dtype, StringComparison.Ordinal))
        {
            problems.Add($"{scale.Name} sits beside a weight that is {w.DType}, not FP8");
            return;
        }
        if (!ScaleDtypes.Contains(scale.DType, StringComparer.Ordinal))
            problems.Add($"{scale.Name} is {scale.DType}, expected BF16 or F32");
        long[] grid = new long[weightShape.Length];
        for (int i = 0; i < grid.Length; i++) grid[i] = i < grid.Length - 2 ? weightShape[i] : (weightShape[i] + BlockEdge - 1) / BlockEdge;
        if (!scale.Shape.AsSpan().SequenceEqual(grid))
            problems.Add($"{scale.Name} is [{string.Join(", ", scale.Shape)}], expected [{string.Join(", ", grid)}]");
    }

    private static void AddLayer(Hy4FlashConfig c, int layer, List<Hy4FlashExpectedTensor> list)
    {
        string p = Layer(layer);
        long h = c.HiddenSize, hc = c.HcMult, heads = c.HeadsFor(layer);
        long qk = c.QkHeadDim, v = c.VHeadDim, nope = c.QkNopeHeadDim, kvr = c.KvLoraRank, qr = c.QLoraRank;
        list.Add(new($"{p}.input_layernorm.weight", [h]));
        list.Add(new($"{p}.post_attention_layernorm.weight", [h]));
        foreach (string module in (string[])["hc_attn_layer", "hc_mlp_layer"])
        {
            list.Add(new($"{p}.{module}.hc_pre.hc_fn", [2 * hc, hc * h], F32Only));
            list.Add(new($"{p}.{module}.hc_pre.hc_base", [2 * hc], F32Only));
            list.Add(new($"{p}.{module}.hc_pre.hc_scale", [2], F32Only));
        }
        string a = $"{p}.self_attn";
        list.Add(new($"{a}.q_a_proj.weight", [qr, h]));
        list.Add(new($"{a}.q_a_layernorm.weight", [qr]));
        list.Add(new($"{a}.q_b_proj.weight", [heads * qk, qr]));
        list.Add(new($"{a}.kv_a_proj_with_mqa.weight", [kvr + c.QkRopeHeadDim, h]));
        list.Add(new($"{a}.kv_a_layernorm.weight", [kvr]));
        list.Add(new($"{a}.kv_b_proj.weight", [heads * (nope + v), kvr]));
        list.Add(new($"{a}.o_proj.weight", [h, heads * v]));
        list.Add(new($"{a}.linear_gate.weight", [heads * v, h]));
        list.Add(new($"{a}.learnable_sink_param", [heads], F32Only));
        list.Add(new($"{a}.post_proj_norm.weight", [h]));
        if (c.LayerKinds[layer] == Hy4FlashLayerKind.Indexed)
        {
            string i = $"{a}.compressed_indexer";
            long nh = c.IndexNHeads, hd = c.IndexHeadDim;
            list.Add(new($"{i}.query_projection.weight", [nh * hd, qr]));
            list.Add(new($"{i}.head_weight_projection.weight", [nh, h]));
            list.Add(new($"{i}.token_key_projection.weight", [hd, h]));
            list.Add(new($"{i}.token_key_norm.weight", [hd]));
            list.Add(new($"{i}.key_value_projection.weight", [hd, hd], F32Only));
            list.Add(new($"{i}.key_gate_projection.weight", [hd, hd], F32Only));
            list.Add(new($"{i}.key_norm.weight", [hd], F32Only));
            list.Add(new($"{i}.intra_group_ape", [c.IndexKpool, hd], F32Only));
        }
        if (c.SparseFfn[layer]) AddMoe(c, p, list);
        else
        {
            long ff = c.IntermediateSize;
            list.Add(new($"{p}.mlp.gate_proj.weight", [ff, h]));
            list.Add(new($"{p}.mlp.up_proj.weight", [ff, h]));
            list.Add(new($"{p}.mlp.down_proj.weight", [h, ff]));
            list.Add(new($"{p}.mlp.post_fc2_norm.weight", [h]));
        }
        if (c.HasEngram(layer)) AddEngram(c, layer, list);
    }

    private static void AddMoe(Hy4FlashConfig c, string p, List<Hy4FlashExpectedTensor> list)
    {
        long h = c.HiddenSize, e = c.NRoutedExperts, mi = c.MoeIntermediateSize, mh = c.MoeHiddenSize, shared = mi * c.NSharedExperts;
        list.Add(new($"{p}.mlp.gate.weight", [e, h]));
        list.Add(new($"{p}.mlp.gate.e_score_correction_bias", [e], F32Only));
        list.Add(new($"{p}.mlp.routed_expert_down_proj.weight", [mh, h]));
        list.Add(new($"{p}.mlp.routed_expert_up_proj.weight", [h, mh]));
        list.Add(new($"{p}.mlp.experts.gate_up_proj", [e, 2 * mi, mh]));
        list.Add(new($"{p}.mlp.experts.down_proj", [e, mh, mi]));
        list.Add(new($"{p}.mlp.shared_experts.gate_proj.weight", [shared, h]));
        list.Add(new($"{p}.mlp.shared_experts.up_proj.weight", [shared, h]));
        list.Add(new($"{p}.mlp.shared_experts.down_proj.weight", [h, shared]));
        list.Add(new($"{p}.mlp.combine_output_norm.weight", [h]));
    }

    private static void AddEngram(Hy4FlashConfig c, int layer, List<Hy4FlashExpectedTensor> list)
    {
        string e = $"{Layer(layer)}.engram_layer";
        long h = c.HiddenSize, hc = c.HcMult, width = (long)(c.EngramMaxNgramSize - 1) * c.EngramEmbedPerNgram;
        long headDim = c.EngramEmbedPerNgram / c.EngramHeadsPerNgram;
        list.Add(new($"{e}.lookup_table", [c.EngramLookupTableSize], I32Only));
        list.Add(new($"{e}.multi_head_embedding.embedding.weight", [c.EngramTableRows(layer + 1), headDim]));
        for (int head = 0; head < c.EngramHashHeads; head++)
            list.Add(new(string.Create(CultureInfo.InvariantCulture, $"{e}.lookup_per_head_norms.{head}.weight"), [headDim]));
        list.Add(new($"{e}.value_proj.weight", [h, width]));
        list.Add(new($"{e}.value_proj.bias", [h]));
        list.Add(new($"{e}.unified_pre_gamma", [h]));
        for (int i = 0; i < hc; i++)
        {
            list.Add(new(string.Create(CultureInfo.InvariantCulture, $"{e}.key_projs.{i}.weight"), [h, width]));
            list.Add(new(string.Create(CultureInfo.InvariantCulture, $"{e}.key_projs.{i}.bias"), [h]));
            list.Add(new(string.Create(CultureInfo.InvariantCulture, $"{e}.norm_key.{i}.weight"), [h]));
            list.Add(new(string.Create(CultureInfo.InvariantCulture, $"{e}.norm_query.{i}.weight"), [h]));
            list.Add(new(string.Create(CultureInfo.InvariantCulture, $"{e}.unified_per_hc_scales.{i}"), [h]));
        }
    }

    private static void AddVision(Hy4FlashVisionConfig v, List<Hy4FlashExpectedTensor> list)
    {
        long d = v.HiddenSize, ff = v.IntermediateSize, m = v.SpatialMergeSize, merged = d * m * m;
        list.Add(new("vit.embeddings.patch_embedding.weight", [d, 3, v.PatchSize, v.PatchSize]));
        list.Add(new("vit.embeddings.patch_embedding.bias", [d]));
        list.Add(new("vit.embeddings.position_embedding.weight", [v.PositionRows, d]));
        for (int i = 0; i < v.NumHiddenLayers; i++)
        {
            string p = string.Create(CultureInfo.InvariantCulture, $"vit.layers.{i}");
            foreach (string norm in (string[])["input_layernorm", "post_attention_layernorm"])
            {
                list.Add(new($"{p}.{norm}.weight", [d]));
                list.Add(new($"{p}.{norm}.bias", [d]));
            }
            foreach (string proj in (string[])["q_proj", "k_proj", "v_proj", "o_proj"])
            {
                list.Add(new($"{p}.self_attn.{proj}.weight", [d, d]));
                list.Add(new($"{p}.self_attn.{proj}.bias", [d]));
            }
            list.Add(new($"{p}.mlp.dense_h_to_4h.weight", [ff, d]));
            list.Add(new($"{p}.mlp.dense_h_to_4h.bias", [ff]));
            list.Add(new($"{p}.mlp.dense_4h_to_h.weight", [d, ff]));
            list.Add(new($"{p}.mlp.dense_4h_to_h.bias", [d]));
        }
        list.Add(new("vit.perceive.norm.weight", [d]));
        list.Add(new("vit.perceive.proj.0.weight", [merged, d, m, m]));
        list.Add(new("vit.perceive.proj.0.bias", [merged]));
        list.Add(new("vit.perceive.proj.2.weight", [v.OutHiddenSize, merged, 1, 1]));
        list.Add(new("vit.perceive.proj.2.bias", [v.OutHiddenSize]));
        list.Add(new("vit.perceive.image_newline", [v.OutHiddenSize]));
    }
}
