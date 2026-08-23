namespace HartsyInference.ModelAssets.Gguf.KeyMappers;

/// <summary>GGUF mapper for Microsoft Phi-3 / Phi-3.5 (<c>phi3</c>). Two tensors are <i>fused</i> in this layout and are split downstream (<see cref="HartsyInference.LLM.Generation.GgufLanguageModel"/>) into the separate projections the <see cref="HartsyInference.LLM.Transformer.GenericTransformer"/> expects: <c>attn_qkv</c> → q/k/v, and <c>ffn_up</c> (a fused gate+up) → gate/up. The split is a contiguous row-byte copy, so it works directly on the quantized weights (no dequant). Phi-3 also ships the LongRope per-dimension factor tables as the <c>rope_factors_long/short</c> tensors, mapped through for the config factory.</summary>
public sealed class PhiKeyMapper : IGgufKeyMapper
{
    public string Architecture => "phi3";

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        foreach (string name in tensorNames)
            if (name.EndsWith(".attn_qkv.weight", StringComparison.Ordinal)) return true;
        return false;
    }

    public string? MapKey(string ggufKey)
    {
        if (ggufKey.Equals("token_embd.weight", StringComparison.Ordinal)) return "model.embed_tokens.weight";
        if (ggufKey.Equals("output_norm.weight", StringComparison.Ordinal)) return "model.norm.weight";
        if (ggufKey.Equals("output.weight", StringComparison.Ordinal)) return "lm_head.weight";
        if (ggufKey.Equals("rope_factors_long.weight", StringComparison.Ordinal)) return "model.rope_factors_long.weight";
        if (ggufKey.Equals("rope_factors_short.weight", StringComparison.Ordinal)) return "model.rope_factors_short.weight";

        if (ggufKey.StartsWith("blk.", StringComparison.Ordinal))
        {
            int dotAfterIndex = ggufKey.IndexOf('.', 4);
            if (dotAfterIndex < 0) return null;
            string blockIdx = ggufKey.Substring(4, dotAfterIndex - 4);
            string suffix = ggufKey.Substring(dotAfterIndex + 1);

            string? mappedSuffix = suffix switch
            {
                "attn_norm.weight" => "input_layernorm.weight",
                "attn_qkv.weight" => "self_attn.qkv_proj.weight",      // fused → split into q/k/v downstream
                "attn_output.weight" => "self_attn.o_proj.weight",
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                "ffn_up.weight" => "mlp.gate_up_proj.weight",          // fused gate+up → split downstream
                "ffn_down.weight" => "mlp.down_proj.weight",
                _ => null,
            };
            return mappedSuffix is null ? null : $"model.layers.{blockIdx}.{mappedSuffix}";
        }
        return null;
    }
}
