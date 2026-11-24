namespace HartsyInference.LLM.Glm5Next;

/// <summary>Checkpoint key names of a <c>glm5_next</c> release, read from the real <c>model.safetensors.index.json</c> (BF16 and FP8) and
/// the transformers conversion mapping that renames them (<c>conversion_mapping.py</c>, <c>glm5_next</c>). The engine reads the checkpoint's
/// own names; nothing is renamed on disk.</summary>
public static class Glm5NextWeightNames
{
    /// <summary>Suffix that turns an FP8 weight's key into its F32 128x128 block-scale companion.</summary>
    public const string ScaleSuffix = "_scale_inv";

    /// <summary>Prefix of every language-model tensor.</summary>
    public const string Root = "model.language_model.";

    /// <summary>Prefix of every vision-tower tensor.</summary>
    public const string Visual = "model.visual.";

    /// <summary>Token embedding, <c>[vocab, hidden]</c>.</summary>
    public const string Embedding = Root + "embed_tokens.weight";

    /// <summary>Final RMS norm gain, <c>[hidden]</c>.</summary>
    public const string FinalNorm = Root + "norm.weight";

    /// <summary>LM head, <c>[vocab, hidden]</c>.</summary>
    public const string LmHead = "lm_head.weight";

    /// <summary>Prefix of decoder (or MTP) layer <paramref name="layer"/>, with a trailing dot.</summary>
    public static string Layer(int layer) => $"{Root}layers.{layer}.";

    /// <summary>Key of routed expert <paramref name="expert"/>'s <paramref name="projection"/> (<c>gate_proj</c>, <c>up_proj</c>, <c>down_proj</c>).</summary>
    public static string Expert(int layer, int expert, string projection) => $"{Layer(layer)}mlp.experts.{expert}.{projection}.weight";

    /// <summary>The block-scale companion of FP8 weight key <paramref name="weightKey"/>.</summary>
    public static string ScaleOf(string weightKey) => weightKey + ScaleSuffix;

    /// <summary>The expert projections, gate first: the reference concatenates <c>gate_proj</c> then <c>up_proj</c> into its fused <c>gate_up_proj</c>.</summary>
    public static readonly string[] ExpertProjections = ["gate_proj", "up_proj", "down_proj"];

    /// <summary>KDA leaves under <c>self_attn.</c>. The reference concatenates <c>q/k/v_conv1d</c> in that order into one conv, and moves
    /// <c>f_a_proj</c>, <c>f_b_proj</c>, <c>dt_bias</c> and <c>A_log</c> under a <c>forget_gate</c> module.</summary>
    public static readonly string[] KdaLeaves =
    [
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight",
        "f_a_proj.weight", "f_b_proj.weight", "dt_bias", "A_log", "b_proj.weight", "g_a_proj.weight", "g_b_proj.weight", "o_norm.weight",
        "o_proj.weight",
    ];

    /// <summary>MLA leaves under <c>self_attn.</c> of a DSA layer.</summary>
    public static readonly string[] MlaLeaves =
    [
        "q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight", "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight", "kv_b_proj.weight",
        "o_proj.weight",
    ];

    /// <summary>Indexer leaves under <c>self_attn.indexer.</c>.</summary>
    public static readonly string[] IndexerLeaves =
    [
        "wq_b.weight", "wk.weight", "k_norm.weight", "k_norm.bias", "weights_proj.weight", "index_kpool_compress_gate", "index_kpool_compress_ape",
    ];

    /// <summary>Hyper-connection tensors at the layer root: projection, base and scale of the attention and feed-forward connections.</summary>
    public static readonly string[] HcLeaves = ["hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"];

    /// <summary>MTP-only leaves at the layer root. The reference ignores them (<c>_keys_to_ignore_on_load_unexpected</c>).</summary>
    public static readonly string[] MtpLeaves = ["enorm.weight", "hnorm.weight", "eh_proj.weight", "shared_head.norm.weight"];

    /// <summary>Every key of decoder layer <paramref name="layer"/>, or of an MTP layer (own leaves, DSA, MoE, no hyper-connections).</summary>
    public static IEnumerable<string> LayerKeys(Glm5NextConfig c, int layer)
    {
        string p = Layer(layer);
        yield return p + "input_layernorm.weight";
        yield return p + "post_attention_layernorm.weight";
        if (!c.IsMtpLayer(layer))
            foreach (string leaf in HcLeaves) yield return p + leaf;
        else
            foreach (string leaf in MtpLeaves) yield return p + leaf;

        if (c.IsDsa(layer))
        {
            foreach (string leaf in MlaLeaves) yield return p + "self_attn." + leaf;
            foreach (string leaf in IndexerLeaves) yield return p + "self_attn.indexer." + leaf;
        }
        else
        {
            foreach (string leaf in KdaLeaves) yield return p + "self_attn." + leaf;
        }

        if (!c.IsSparse(layer))
        {
            foreach (string proj in ExpertProjections) yield return $"{p}mlp.{proj}.weight";
            yield break;
        }
        yield return p + "mlp.gate.weight";
        yield return p + "mlp.gate.e_score_correction_bias";
        foreach (string proj in ExpertProjections) yield return $"{p}mlp.shared_experts.{proj}.weight";
        for (int e = 0; e < c.Experts; e++)
            foreach (string proj in ExpertProjections) yield return Expert(layer, e, proj);
    }

    /// <summary>Every key of the whole checkpoint for <paramref name="c"/>: embedding, decoder layers, MTP layers, final norm, LM head and, when
    /// <paramref name="includeVision"/> and the config has a tower, the vision tower.</summary>
    public static IEnumerable<string> AllKeys(Glm5NextConfig c, bool includeVision)
    {
        yield return Embedding;
        yield return FinalNorm;
        yield return LmHead;
        for (int l = 0; l < c.Layers + c.MtpLayers; l++)
            foreach (string k in LayerKeys(c, l)) yield return k;
        if (includeVision && c.Vision is { } v)
            foreach (string k in VisionKeys(v)) yield return k;
    }

    /// <summary>Every key of the vision tower.</summary>
    public static IEnumerable<string> VisionKeys(Glm5NextVisionConfig v)
    {
        yield return Visual + "patch_embed.proj.weight";
        yield return Visual + "patch_embed.proj.bias";
        yield return Visual + "post_layernorm.weight";
        yield return Visual + "downsample.weight";
        yield return Visual + "downsample.bias";
        foreach (string leaf in new[] { "proj.weight", "gate_proj.weight", "up_proj.weight", "down_proj.weight", "post_projection_norm.weight", "post_projection_norm.bias" })
            yield return Visual + "merger." + leaf;
        for (int b = 0; b < v.Depth; b++)
        {
            string p = $"{Visual}blocks.{b}.";
            foreach (string leaf in new[] { "norm1.weight", "norm2.weight", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias", "attn.q_norm.weight", "attn.k_norm.weight" })
                yield return p + leaf;
            foreach (string proj in new[] { "gate_proj", "up_proj", "down_proj" })
            {
                yield return $"{p}mlp.{proj}.weight";
                yield return $"{p}mlp.{proj}.bias";
            }
        }
    }

    /// <summary>Whether <paramref name="key"/> is an MTP-layer tensor or a <c>shared_head</c> tensor, which the reference leaves unloaded.</summary>
    public static bool IsMtpKey(Glm5NextConfig c, string key)
    {
        if (!key.StartsWith(Root + "layers.", StringComparison.Ordinal)) return false;
        int start = (Root + "layers.").Length, dot = key.IndexOf('.', start);
        return dot > start && int.TryParse(key.AsSpan(start, dot - start), out int layer) && c.IsMtpLayer(layer);
    }

    /// <summary>Whether the FP8 checkpoint stores <paramref name="weightKey"/> as E4M3 with a <see cref="ScaleSuffix"/> companion. Quantized:
    /// dense and shared FFN projections, routed experts, and the DSA q/kv down-projections, <c>q_b_proj</c> and <c>o_proj</c>. Everything else (KDA
    /// projections, <c>kv_b_proj</c>, indexer, router, embedding, LM head, vision, hyper-connections) stays BF16 or F32.</summary>
    public static bool IsFp8Weight(Glm5NextConfig c, string weightKey)
    {
        if (!weightKey.StartsWith(Root + "layers.", StringComparison.Ordinal) || !weightKey.EndsWith(".weight", StringComparison.Ordinal)) return false;
        int start = (Root + "layers.").Length, dot = weightKey.IndexOf('.', start);
        if (dot <= start || !int.TryParse(weightKey.AsSpan(start, dot - start), out int layer)) return false;
        string leaf = weightKey[(dot + 1)..];
        if (leaf.StartsWith("mlp.", StringComparison.Ordinal))
            return leaf.StartsWith("mlp.experts.", StringComparison.Ordinal) || leaf.StartsWith("mlp.shared_experts.", StringComparison.Ordinal)
                || (!c.IsSparse(layer) && leaf is "mlp.gate_proj.weight" or "mlp.up_proj.weight" or "mlp.down_proj.weight");
        return c.IsDsa(layer) && leaf is "self_attn.q_a_proj.weight" or "self_attn.q_b_proj.weight" or "self_attn.kv_a_proj_with_mqa.weight"
            or "self_attn.o_proj.weight";
    }
}
