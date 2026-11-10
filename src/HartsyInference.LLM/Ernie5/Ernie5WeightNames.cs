using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Ernie5;

/// <summary>ERNIE-5.5 checkpoint key names, read from the real <c>model.safetensors.index.json</c> (vLLM-ERNIE <c>ernie5_5_mapper</c> gives the
/// module each one feeds).</summary>
public static class Ernie5WeightNames
{
    /// <summary>Suffix of an FP8 weight's block-scale companion.</summary>
    public const string ScaleSuffix = "_scale_inv";

    /// <summary>Token embedding, BF16 <c>[vocab, hidden]</c>.</summary>
    public const string Embedding = "model.embed_tokens.weight";

    /// <summary>Final RMS norm gain, <c>[hidden]</c>.</summary>
    public const string FinalNorm = "model.norm.weight";

    /// <summary>LM head, BF16 <c>[vocab, hidden]</c>.</summary>
    public const string LmHead = "lm_head.weight";

    /// <summary>Multimax thresholds <c>r0..r3</c>, <c>[4]</c> (vLLM <c>multimax_ranges</c>).</summary>
    public const string MultimaxRanges = "model.lm_head.multimax_ranges";

    /// <summary>Multimax gains <c>t0..t3</c>, <c>[4]</c> (vLLM <c>multimax_ts</c>).</summary>
    public const string MultimaxGains = "model.lm_head.multimax_ts";

    /// <summary>mHC head read-out projection, F32 stored <c>[hc * hidden, hc]</c> (input-major).</summary>
    public const string HeadFn = "model.mhc_contract.hc_head_fn";

    /// <summary>mHC head gate bias, F32 <c>[hc]</c>.</summary>
    public const string HeadBase = "model.mhc_contract.hc_head_base";

    /// <summary>mHC head gate scale, F32 <c>[1]</c>.</summary>
    public const string HeadScale = "model.mhc_contract.hc_head_scale";

    /// <summary>Prefix of decoder layer <paramref name="layer"/>, without a trailing dot.</summary>
    public static string Layer(int layer) => $"model.layers.{layer}";

    /// <summary>RMS norm before the attention.</summary>
    public const string AttnNorm = "input_layernorm.weight";

    /// <summary>RMS norm before the FFN.</summary>
    public const string FfnNorm = "post_attention_layernorm.weight";

    /// <summary>Hyper-connection module around the attention.</summary>
    public const string AttnHc = "self_attention_hyper_connection";

    /// <summary>Hyper-connection module around the FFN.</summary>
    public const string FfnHc = "mlp_hyper_connection";

    /// <summary>Leaves of one hyper-connection module: the mapper (stored input-major), its bias, and the pre, post and comb scales.</summary>
    public static readonly string[] HcLeaves = ["mapping_proj.weight", "bias", "alpha_pre", "alpha_post", "alpha_res"];

    /// <summary>HCA attention leaves under <c>self_attn.</c>, gate and VHA excluded.</summary>
    public static readonly string[] HcaLeaves =
    [
        "linear_q_down_proj.weight", "q_layernorm.weight", "linear_q_up_proj.weight", "linear_kv_proj.weight", "kv_layernorm.weight",
        "linear_o_group_proj", "o_proj.weight", "core_attention.attn_sink", "core_attention.compressor.linear_wkv.weight",
        "core_attention.compressor.linear_wgate.weight", "core_attention.compressor.ape", "core_attention.compressor.norm.weight",
    ];

    /// <summary>DSA attention leaves under <c>self_attn.</c>, gate and VHA excluded.</summary>
    public static readonly string[] DsaLeaves =
    [
        "q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight", "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight", "kv_b_proj.weight",
        "o_proj.weight", "core_attention.softmax_offset", "core_attention.indexer.wq_b.weight", "core_attention.indexer.wk.weight",
        "core_attention.indexer.k_norm.weight", "core_attention.indexer.k_norm.bias", "core_attention.indexer.weights_proj.weight",
    ];

    /// <summary>Attention output gate, under <c>self_attn.</c>.</summary>
    public const string AttnGate = "gate_proj.weight";

    /// <summary>VHA low-rank factors, under <c>self_attn.</c>.</summary>
    public static readonly string[] VhaLeaves = ["vha_postmix_U", "vha_postmix_V"];

    /// <summary>Dense FFN leaves under <c>mlp.</c>: gate, up, down.</summary>
    public static readonly string[] DenseFfnLeaves = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"];

    /// <summary>The key of an attention leaf of layer <paramref name="layer"/>.</summary>
    public static string Attn(int layer, string leaf) => $"{Layer(layer)}.self_attn.{leaf}";

    /// <summary>The key of a dense FFN leaf of layer <paramref name="layer"/>.</summary>
    public static string DenseFfn(int layer, string leaf) => $"{Layer(layer)}.mlp.{leaf}";

    /// <summary>The key of a hyper-connection leaf of decoder layer <paramref name="layer"/>.</summary>
    public static string Hc(int layer, string module, string leaf) => $"{Layer(layer)}.{module}.{leaf}";

    /// <summary>The key of a hyper-connection leaf of layer <paramref name="layer"/>; an MTP layer keeps its hyper-connections under
    /// <see cref="MtpBlock"/>, everything else of its decoder block at the layer root.</summary>
    public static string Hc(Ernie5Config config, int layer, string module, string leaf) =>
        config.IsMtpLayer(layer) ? $"{Layer(layer)}.{MtpBlock}.{module}.{leaf}" : Hc(layer, module, leaf);

    /// <summary>Sub-prefix of an MTP layer's decoder-block hyper-connections (vLLM <c>ernie5_5_mapper</c> drops it).</summary>
    public const string MtpBlock = "transformer_layer";

    /// <summary>RMS norm of an MTP layer's token embedding, <c>[hidden]</c>.</summary>
    public const string MtpEmbedNorm = "enorm.weight";

    /// <summary>RMS norm of an MTP layer's previous hidden stream, <c>[hidden]</c>.</summary>
    public const string MtpHiddenNorm = "hnorm.weight";

    /// <summary>Projection of an MTP layer's normed embedding, <c>[hidden, hidden]</c>.</summary>
    public const string MtpEmbedProj = "e_proj.weight";

    /// <summary>Projection of an MTP layer's normed previous stream, <c>[hidden, hidden]</c>.</summary>
    public const string MtpHiddenProj = "h_proj.weight";

    /// <summary>An MTP layer's own mHC head read-out projection, F32 stored <c>[hc * hidden, hc]</c> (input-major) like <see cref="HeadFn"/>.</summary>
    public const string MtpHeadFn = "hc_head_fn";

    /// <summary>An MTP layer's mHC head gate bias, F32 <c>[hc]</c>.</summary>
    public const string MtpHeadBase = "hc_head_base";

    /// <summary>An MTP layer's mHC head gate scale, F32 <c>[1]</c>.</summary>
    public const string MtpHeadScale = "hc_head_scale";

    /// <summary>An MTP layer's final RMS norm before the shared LM head, <c>[hidden]</c>.</summary>
    public const string MtpFinalNorm = "shared_head.norm.weight";

    /// <summary>Leaves an MTP layer has outside its decoder block (vLLM <c>_MTP_LAYER_OWN</c>); the LM head and embedding are shared.</summary>
    public static readonly string[] MtpLeaves =
        [MtpEmbedNorm, MtpHiddenNorm, MtpEmbedProj, MtpHiddenProj, MtpHeadFn, MtpHeadBase, MtpHeadScale, MtpFinalNorm];

    /// <summary>The key of an MTP-own leaf of layer <paramref name="layer"/>.</summary>
    public static string Mtp(int layer, string leaf) => $"{Layer(layer)}.{leaf}";

    /// <summary>Every key of the latent MoE block of layer <paramref name="layer"/>: the checkpoint's own name (<c>block_sparse_moe.*</c>,
    /// <c>w1</c>/<c>w3</c>/<c>w2</c> experts) and the engine name <see cref="MoeFeedForward"/> reads it under, the load-time remap MiniMax
    /// uses too. The latent norm keeps the engine's Muse Spark name wherever <see cref="MoeConfig.LatentNorm"/> places it.</summary>
    public static IEnumerable<(string Checkpoint, string Engine)> MoeKeys(Ernie5Config config, int layer)
    {
        string block = $"{Layer(layer)}.block_sparse_moe", mlp = $"{Layer(layer)}.mlp";
        (string Ernie, string Engine)[] projs = [("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")];
        yield return ($"{block}.gate.weight", $"{mlp}.gate.weight");
        if (config.MoeSplitFeatureRouting) yield return ($"{block}.gate.weight_1", $"{mlp}.gate.weight_1");
        yield return ($"{block}.e_score_correction_bias", $"{mlp}.gate.e_score_correction_bias");
        if (config.RoutedScalingFactorLearnable)
            yield return ($"{block}.gate.routed_scaling_factor_param", $"{mlp}.gate.routed_scaling_factor_param");
        yield return ($"{block}.fc1_latent_proj.weight", $"{mlp}.pre_expert_proj.weight");
        if (config.LatentMoeUseNorm) yield return ($"{block}.latent_norm.weight", $"{mlp}.pre_expert_norm.weight");
        yield return ($"{block}.fc2_latent_proj.weight", $"{mlp}.post_expert_proj.weight");
        if (config.NSharedExperts > 0)
            foreach ((string ernie, string engine) in projs)
                yield return ($"{block}.shared_experts.{ernie}.weight", $"{mlp}.shared_expert.{engine}.weight");
        for (int e = 0; e < config.NRoutedExperts; e++)
            foreach ((string ernie, string engine) in projs)
                yield return ($"{block}.experts.{e}.{ernie}.weight", $"{mlp}.experts.{e}.{engine}.weight");
    }
}
