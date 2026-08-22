namespace HartsyInference.ModelAssets.Gguf.KeyMappers;

/// <summary>GGUF mapper for the Gemma family (<c>gemma</c>, <c>gemma2</c>, <c>gemma3</c> text models). Shares the llama.cpp <c>blk.N.attn_*/ffn_*</c> dialect but adds Gemma's sandwich norms (<c>post_attention_norm</c>, <c>post_ffw_norm</c>) and, for Gemma-3, per-head Q/K norms — remapped to the HF names the <see cref="HartsyInference.LLM.Transformer.GenericTransformer"/> loader expects. Gemma ties the LM head, so there is no <c>output.weight</c>. The architectural quirks (GeGLU, (1+w)-norm, embedding scale, dual-RoPE, logit soft-cap) are config flags set from the arch string, not naming differences, so this mapper is purely the mechanical key rewrite.</summary>
public sealed class GemmaKeyMapper : IGgufKeyMapper
{
    public string Architecture => "gemma";

    // gemma/gemma2/gemma3 share this exact tensor dialect; the config factory distinguishes them by arch string.
    public IReadOnlyCollection<string> Architectures => ["gemma", "gemma2", "gemma3"];

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        // Sandwich norms are unique to Gemma-2/3 among the GGUF decoders we map, so they identify the family
        // when metadata is missing. (Gemma-1 has no metadata-less dumps in practice.)
        foreach (string name in tensorNames)
            if (name.EndsWith(".post_ffw_norm.weight", StringComparison.Ordinal)
                || name.EndsWith(".post_attention_norm.weight", StringComparison.Ordinal))
                return true;
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
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_output.weight" => "self_attn.o_proj.weight",
                "attn_q_norm.weight" => "self_attn.q_norm.weight",
                "attn_k_norm.weight" => "self_attn.k_norm.weight",
                "post_attention_norm.weight" => "post_attention_layernorm.weight",
                "ffn_norm.weight" => "pre_feedforward_layernorm.weight",
                "post_ffw_norm.weight" => "post_feedforward_layernorm.weight",
                "ffn_gate.weight" => "mlp.gate_proj.weight",
                "ffn_up.weight" => "mlp.up_proj.weight",
                "ffn_down.weight" => "mlp.down_proj.weight",
                _ => null,
            };
            return mappedSuffix is null ? null : $"model.layers.{blockIdx}.{mappedSuffix}";
        }
        return null;
    }
}
