namespace HartsyInference.ModelHandler.Gguf.KeyMappers;

/// <summary>GGUF mapper for Llama-3.2-Vision (<c>mllama</c>). The text decoder is standard Llama (so the dense
/// self-attention / FFN keys map exactly as the llama family), but mllama interleaves <b>gated cross-attention</b>
/// layers (at <c>[3,8,13,18,23,28,33,38]</c> for 11B) whose tensors carry the <c>cross_attn_*</c> dialect and two
/// learned scalar gates. Those are remapped to the <c>cross_attn.*</c> + <c>cross_attn_attn_gate</c>/<c>_mlp_gate</c>
/// names the <see cref="HartsyInference.LLM.Multimodal.MllamaCrossAttentionLayer"/> expects. The vision tower
/// (<c>v.*</c>) and projector (<c>mm.*</c>) live in the companion mmproj GGUF and are loaded by the vision path.</summary>
public sealed class MllamaKeyMapper : IGgufKeyMapper
{
    public string Architecture => "mllama";
    public IReadOnlyCollection<string> Architectures => ["mllama"];

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        foreach (string name in tensorNames)
            if (name.Contains(".cross_attn_q_proj.", StringComparison.Ordinal)) return true;
        return false;
    }

    public string? MapKey(string ggufKey)
    {
        if (ggufKey.Equals("token_embd.weight", StringComparison.Ordinal)) return "model.embed_tokens.weight";
        if (ggufKey.Equals("output_norm.weight", StringComparison.Ordinal)) return "model.norm.weight";
        if (ggufKey.Equals("output.weight", StringComparison.Ordinal)) return "lm_head.weight";
        if (ggufKey.Equals("rope_freqs.weight", StringComparison.Ordinal)) return "model.rope_freqs.weight";
        // The companion mmproj GGUF is also arch=mllama: pass its vision tower (`v.*`) + projector (`mm.*`)
        // tensors through verbatim for the MllamaVisionEncoder.
        if (ggufKey.StartsWith("v.", StringComparison.Ordinal) || ggufKey.StartsWith("mm.", StringComparison.Ordinal))
            return ggufKey;

        if (ggufKey.StartsWith("blk.", StringComparison.Ordinal))
        {
            int dot = ggufKey.IndexOf('.', 4);
            if (dot < 0) return null;
            string idx = ggufKey.Substring(4, dot - 4);
            string suffix = ggufKey.Substring(dot + 1);

            string? mapped = suffix switch
            {
                // Standard Llama text self-attention + FFN.
                "attn_norm.weight" => "input_layernorm.weight",
                "attn_q.weight" => "self_attn.q_proj.weight",
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_output.weight" => "self_attn.o_proj.weight",
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                "ffn_gate.weight" => "mlp.gate_proj.weight",
                "ffn_up.weight" => "mlp.up_proj.weight",
                "ffn_down.weight" => "mlp.down_proj.weight",
                // Gated cross-attention layers (vision → text).
                "cross_attn_q_proj.weight" => "cross_attn.q_proj.weight",
                "cross_attn_k_proj.weight" => "cross_attn.k_proj.weight",
                "cross_attn_v_proj.weight" => "cross_attn.v_proj.weight",
                "cross_attn_o_proj.weight" => "cross_attn.o_proj.weight",
                "cross_attn_q_norm.weight" => "cross_attn.q_norm.weight",
                "cross_attn_k_norm.weight" => "cross_attn.k_norm.weight",
                "cross_attn_attn_gate" => "cross_attn_attn_gate",
                "cross_attn_mlp_gate" => "cross_attn_mlp_gate",
                _ => null,
            };
            return mapped is null ? null : $"model.layers.{idx}.{mapped}";
        }
        return null;
    }
}
