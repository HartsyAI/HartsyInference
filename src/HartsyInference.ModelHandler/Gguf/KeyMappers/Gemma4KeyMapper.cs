namespace HartsyInference.ModelHandler.Gguf.KeyMappers;

/// <summary>GGUF mapper for Gemma-4 (<c>gemma4</c>). Shares the Gemma-2/3 sandwich-norm dialect (see
/// <see cref="GemmaKeyMapper"/>) plus three additions unique to Gemma-4: per-layer embeddings (PLE, a
/// Gemma-3n-lineage mechanism — a top-level per-layer token table/projection/norm plus each layer's own
/// gate/proj/post-norm), an optional learned per-layer output scale, and — on the 26B-A4B MoE checkpoint only —
/// a routed-expert branch with its own pre/post norms that runs IN PARALLEL WITH the dense FFN (summed, not
/// routed-instead-of). See <see cref="HartsyInference.LLM.Transformer.GgufConfigFactory"/>'s <c>isGemma4</c>
/// branch and <see cref="HartsyInference.LLM.Transformer.GenericTransformer"/>'s PLE/dual-branch-MoE handling
/// for how these tensors are consumed.</summary>
public sealed class Gemma4KeyMapper : IGgufKeyMapper
{
    public string Architecture => "gemma4";

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        foreach (string name in tensorNames)
            if (name.Equals("per_layer_token_embd.weight", StringComparison.Ordinal))
                return true;
        return false;
    }

    public string? MapKey(string ggufKey)
    {
        if (ggufKey.Equals("token_embd.weight", StringComparison.Ordinal)) return "model.embed_tokens.weight";
        if (ggufKey.Equals("output_norm.weight", StringComparison.Ordinal)) return "model.norm.weight";
        if (ggufKey.Equals("output.weight", StringComparison.Ordinal)) return "lm_head.weight";
        if (ggufKey.Equals("rope_freqs.weight", StringComparison.Ordinal)) return "model.rope_freqs.weight";

        // Top-level PLE tensors (not per-layer despite the mechanism feeding every layer).
        if (ggufKey.Equals("per_layer_token_embd.weight", StringComparison.Ordinal)) return "model.per_layer_token_embd.weight";
        if (ggufKey.Equals("per_layer_model_proj.weight", StringComparison.Ordinal)) return "model.per_layer_model_proj.weight";
        if (ggufKey.Equals("per_layer_proj_norm.weight", StringComparison.Ordinal)) return "model.per_layer_proj_norm.weight";

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
                // MoE (26B-A4B only): router + stacked experts (split downstream, same convention as LlamaKeyMapper),
                // plus the two extra pre/post norm pairs and the router's per-channel input scale unique to the
                // parallel dense+MoE-branch pattern.
                "ffn_gate_inp.weight" => "mlp.gate.weight",
                "ffn_gate_inp.scale" => "mlp.gate_inp_scale.weight",
                "ffn_gate_exps.weight" => "mlp.gate_exps.weight",
                "ffn_up_exps.weight" => "mlp.up_exps.weight",
                "ffn_down_exps.weight" => "mlp.down_exps.weight",
                "pre_ffw_norm_2.weight" => "mlp.ffn_pre_norm_2.weight",
                "post_ffw_norm_1.weight" => "mlp.ffn_post_norm_1.weight",
                "post_ffw_norm_2.weight" => "mlp.ffn_post_norm_2.weight",
                // Optional learned per-layer output scale (TENSOR_NOT_REQUIRED upstream — absent on most checkpoints).
                "layer_output_scale.weight" => "layer_out_scale.weight",
                // PLE per-layer tensors.
                "inp_gate.weight" => "per_layer_inp_gate.weight",
                "proj.weight" => "per_layer_proj.weight",
                "post_norm.weight" => "per_layer_post_norm.weight",
                _ => null,
            };
            return mappedSuffix is null ? null : $"model.layers.{blockIdx}.{mappedSuffix}";
        }
        return null;
    }
}
