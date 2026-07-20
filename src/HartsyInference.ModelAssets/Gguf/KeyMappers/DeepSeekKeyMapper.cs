namespace HartsyInference.ModelAssets.Gguf.KeyMappers;

/// <summary>GGUF mapper for the DeepSeek-V2/V3 family (<c>deepseek2</c> — also covers Kimi-K2, a V3-architecture
/// model). These use Multi-head Latent Attention (compressed KV latent + shared RoPE key, up-projected per head)
/// and fine-grained MoE (many small routed experts + a shared expert, a leading dense layer). The MLA tensors
/// (<c>attn_kv_a_mqa</c>, <c>attn_kv_a_norm</c>, <c>attn_kv_b</c>, and either a direct <c>attn_q</c> or the
/// compressed <c>attn_q_a</c>/<c>attn_q_b</c> pair) and the stacked/​shared expert tensors are remapped to the
/// HF-style names the <see cref="HartsyInference.LLM.Transformer.GenericTransformer"/> MLA + MoE paths expect;
/// the stacked <c>*_exps</c> are split downstream (GgufLanguageModel).</summary>
public sealed class DeepSeekKeyMapper : IGgufKeyMapper
{
    public string Architecture => "deepseek2";

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        foreach (string name in tensorNames)
            if (name.EndsWith(".attn_kv_a_mqa.weight", StringComparison.Ordinal)) return true;
        return false;
    }

    public string? MapKey(string ggufKey)
    {
        if (ggufKey.Equals("token_embd.weight", StringComparison.Ordinal)) return "model.embed_tokens.weight";
        if (ggufKey.Equals("output_norm.weight", StringComparison.Ordinal)) return "model.norm.weight";
        if (ggufKey.Equals("output.weight", StringComparison.Ordinal)) return "lm_head.weight";

        if (ggufKey.StartsWith("blk.", StringComparison.Ordinal))
        {
            int dot = ggufKey.IndexOf('.', 4);
            if (dot < 0) return null;
            string idx = ggufKey.Substring(4, dot - 4);
            string suffix = ggufKey.Substring(dot + 1);

            string? mapped = suffix switch
            {
                "attn_norm.weight" => "input_layernorm.weight",
                "attn_q.weight" => "self_attn.q_proj.weight",                 // direct query (V2-Lite)
                "attn_q_a.weight" => "self_attn.q_a_proj.weight",             // compressed query (larger V2/V3)
                "attn_q_a_norm.weight" => "self_attn.q_a_norm.weight",
                "attn_q_b.weight" => "self_attn.q_b_proj.weight",
                "attn_kv_a_mqa.weight" => "self_attn.kv_a_proj.weight",       // → latent[kv_lora] + k_rope
                "attn_kv_a_norm.weight" => "self_attn.kv_a_norm.weight",
                "attn_kv_b.weight" => "self_attn.kv_b_proj.weight",           // latent → per-head k_nope + v
                "attn_output.weight" => "self_attn.o_proj.weight",
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                "ffn_gate.weight" => "mlp.gate_proj.weight",                  // dense (leading) layers
                "ffn_up.weight" => "mlp.up_proj.weight",
                "ffn_down.weight" => "mlp.down_proj.weight",
                "ffn_gate_inp.weight" => "mlp.gate.weight",                   // router
                "exp_probs_b.bias" => "mlp.gate.e_score_correction_bias",     // V3/Kimi node-limited router bias
                "ffn_gate_exps.weight" => "mlp.gate_exps.weight",            // stacked → split
                "ffn_up_exps.weight" => "mlp.up_exps.weight",
                "ffn_down_exps.weight" => "mlp.down_exps.weight",
                "ffn_gate_shexp.weight" => "mlp.shared_expert.gate_proj.weight",
                "ffn_up_shexp.weight" => "mlp.shared_expert.up_proj.weight",
                "ffn_down_shexp.weight" => "mlp.shared_expert.down_proj.weight",
                _ => null,
            };
            return mapped is null ? null : $"model.layers.{idx}.{mapped}";
        }
        return null;
    }
}
