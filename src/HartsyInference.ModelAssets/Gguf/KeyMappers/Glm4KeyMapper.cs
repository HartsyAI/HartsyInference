namespace HartsyInference.ModelAssets.Gguf.KeyMappers;

/// <summary>GGUF mapper for the GLM-4 text decoder (<c>glm4</c> — GLM-4-9B/32B-0414, the LLM_ARCH_GLM4 lineage, NOT
/// the older <c>chatglm</c> arch). GLM-4 uses Gemma-style <b>sandwich norms</b> (a post-attention and a post-FFN
/// RMSNorm applied to each sublayer output before its residual add) but, unlike Gemma, carries Q/K/V projection
/// <i>biases</i> and a <i>fused</i> gate+up FFN (<c>ffn_up</c> holds <c>[gate | up]</c> of width 2·ffn, split
/// downstream in <see cref="HartsyInference.LLM.Generation.GgufLanguageModel"/>). The attention output has no bias,
/// the LM head is untied, RoPE is partial (rope.dimension_count &lt; head_dim). The norm tensor names map onto the
/// same sandwich slots the <see cref="HartsyInference.LLM.Transformer.GenericTransformer"/> loader expects:
/// <c>ffn_norm</c> is the <i>pre</i>-FFN norm (Gemma's <c>pre_feedforward_layernorm</c>), and <c>post_ffw_norm</c>
/// is the post-FFN norm.</summary>
public sealed class Glm4KeyMapper : IGgufKeyMapper
{
    public string Architecture => "glm4";

    public IReadOnlyCollection<string> Architectures => ["glm4"];

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        // GLM-4 is uniquely the sandwich-norm arch that also carries an attn_q.bias (Gemma has no QKV bias).
        bool sandwich = false, qBias = false;
        foreach (string name in tensorNames)
        {
            if (name.EndsWith(".post_ffw_norm.weight", StringComparison.Ordinal)) sandwich = true;
            if (name.EndsWith(".attn_q.bias", StringComparison.Ordinal)) qBias = true;
            if (sandwich && qBias) return true;
        }
        return false;
    }

    public string? MapKey(string ggufKey)
    {
        if (ggufKey.Equals("token_embd.weight", StringComparison.Ordinal)) return "model.embed_tokens.weight";
        if (ggufKey.Equals("output_norm.weight", StringComparison.Ordinal)) return "model.norm.weight";
        if (ggufKey.Equals("output.weight", StringComparison.Ordinal)) return "lm_head.weight";

        if (ggufKey.StartsWith("blk.", StringComparison.Ordinal))
        {
            int dotAfterIndex = ggufKey.IndexOf('.', 4);
            if (dotAfterIndex < 0) return null;
            string blockIdx = ggufKey.Substring(4, dotAfterIndex - 4);
            string suffix = ggufKey.Substring(dotAfterIndex + 1);

            string? mappedSuffix = suffix switch
            {
                "attn_norm.weight" => "input_layernorm.weight",
                "attn_q.weight" => "self_attn.q_proj.weight",
                "attn_q.bias" => "self_attn.q_proj.bias",
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_k.bias" => "self_attn.k_proj.bias",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_v.bias" => "self_attn.v_proj.bias",
                "attn_output.weight" => "self_attn.o_proj.weight",
                "post_attention_norm.weight" => "post_attention_layernorm.weight",
                "ffn_norm.weight" => "pre_feedforward_layernorm.weight",
                "post_ffw_norm.weight" => "post_feedforward_layernorm.weight",
                "ffn_up.weight" => "mlp.gate_up_proj.weight",   // fused [gate|up] → split downstream
                "ffn_down.weight" => "mlp.down_proj.weight",
                _ => null,
            };
            return mappedSuffix is null ? null : $"model.layers.{blockIdx}.{mappedSuffix}";
        }
        return null;
    }
}
