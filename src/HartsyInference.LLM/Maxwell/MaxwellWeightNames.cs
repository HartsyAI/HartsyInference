namespace HartsyInference.LLM.Maxwell;

/// <summary>Maxwell-V1.0-Flash checkpoint key names, read from the release's <c>model.safetensors.index.json</c> and safetensors headers.</summary>
public static class MaxwellWeightNames
{
    /// <summary>Suffix of an FP8 weight's block-scale companion.</summary>
    public const string ScaleSuffix = "_scale_inv";

    /// <summary>Token embedding, BF16 <c>[vocab, hidden]</c>.</summary>
    public const string Embedding = "model.embed_tokens.weight";

    /// <summary>Final RMS norm gain, <c>[hidden]</c>.</summary>
    public const string FinalNorm = "model.norm.weight";

    /// <summary>LM head, BF16 <c>[vocab, hidden]</c>, not tied to the embedding.</summary>
    public const string LmHead = "lm_head.weight";

    /// <summary>The main model's stream read-out head prefix.</summary>
    public const string Head = "model.hc_head";

    /// <summary>Leaves of a stream read-out head: projection <c>[hc, hc * hidden]</c>, gate bias <c>[hc]</c>, gate scale <c>[1]</c>.</summary>
    public static readonly string[] HeadLeaves = ["hc_head_fn", "hc_head_base", "hc_head_scale"];

    /// <summary>RMS norm before the attention.</summary>
    public const string AttnNorm = "input_layernorm.weight";

    /// <summary>RMS norm before the FFN.</summary>
    public const string FfnNorm = "post_attention_layernorm.weight";

    /// <summary>Hyper-connection module around the attention.</summary>
    public const string AttnHc = "self_attn_hyper_connection";

    /// <summary>Hyper-connection module around the FFN.</summary>
    public const string FfnHc = "mlp_hyper_connection";

    /// <summary>Leaves of one hyper-connection module: the mapper <c>[(2 + hc) * hc, hc * hidden]</c> (already the engine's layout), its bias, and
    /// the pre, post and comb scales.</summary>
    public static readonly string[] HcLeaves = ["mapping_proj.weight", "bias", "alpha_pre", "alpha_post", "alpha_res"];

    /// <summary>Attention leaves under <c>self_attn.</c> of every decoder layer.</summary>
    public static readonly string[] AttnLeaves =
    [
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "q_layernorm.weight", "k_layernorm.weight",
    ];

    /// <summary>Indexer leaves under <c>self_attn.indexer.</c>; the MTP layer's dense attention has none.</summary>
    public static readonly string[] IndexerLeaves =
        ["linear_q_proj.weight", "linear_k_proj.weight", "q_layernorm.weight", "k_layernorm.weight"];

    /// <summary>Projections of a dense FFN, the shared expert and each routed expert: gate (<c>w1</c>), up (<c>w3</c>), down (<c>w2</c>).</summary>
    public static readonly string[] FfnLeaves = ["w1.weight", "w3.weight", "w2.weight"];

    /// <summary>Leaves of one multi-token-prediction layer outside its decoder layer, each under <c>mtp_layers.k.</c>.</summary>
    public static readonly string[] MtpLeaves =
    [
        "enorm.weight", "hnorm.weight", "e_proj.weight", "h_proj.weight", "final_layernorm.weight", "hc_head.hc_head_fn",
        "hc_head.hc_head_base", "hc_head.hc_head_scale",
    ];

    /// <summary>Router weight under a layer prefix, <c>[experts, hidden]</c>.</summary>
    public const string Gate = "block_sparse_moe.gate.weight";

    /// <summary>Router selection bias under a layer prefix, F32 <c>[experts]</c>; it must not be rounded to BF16.</summary>
    public const string ExpertBias = "block_sparse_moe.expert_bias";

    /// <summary>Prefix of decoder layer <paramref name="layer"/>, without a trailing dot.</summary>
    public static string Layer(int layer) => $"model.layers.{layer}";

    /// <summary>Prefix of MTP layer <paramref name="mtp"/> (0-based), without a trailing dot.</summary>
    public static string MtpLayer(int mtp) => $"mtp_layers.{mtp}";

    /// <summary>Prefix of the decoder layer inside MTP layer <paramref name="mtp"/>, without a trailing dot.</summary>
    public static string MtpDecoder(int mtp) => $"{MtpLayer(mtp)}.decoder_layer";

    /// <summary>A key under <paramref name="prefix"/>.</summary>
    public static string Key(string prefix, string leaf) => $"{prefix}.{leaf}";

    /// <summary>Hyper-connection key of <paramref name="module"/> under <paramref name="prefix"/>.</summary>
    public static string Hc(string prefix, string module, string leaf) => $"{prefix}.{module}.{leaf}";

    /// <summary>Key of routed expert <paramref name="expert"/>'s <paramref name="leaf"/> (one of <see cref="FfnLeaves"/>) under <paramref name="prefix"/>.</summary>
    public static string Expert(string prefix, int expert, string leaf) => $"{prefix}.block_sparse_moe.experts.{expert}.{leaf}";

    /// <summary>Key of the shared expert's <paramref name="leaf"/> under <paramref name="prefix"/>.</summary>
    public static string SharedExpert(string prefix, string leaf) => $"{prefix}.block_sparse_moe.shared_expert.{leaf}";

    /// <summary>Key of a dense FFN's <paramref name="leaf"/> under <paramref name="prefix"/>.</summary>
    public static string DenseFfn(string prefix, string leaf) => $"{prefix}.block_sparse_moe.{leaf}";
}
