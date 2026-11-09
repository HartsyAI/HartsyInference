namespace HartsyInference.ModelAssets.Gguf.KeyMappers;

/// <summary>GGUF mapper for MOSS-VL-Realtime (<c>moss_vl</c>). The text decoder is the llama-family dense decoder with gated
/// cross-attention layers at <c>[2, 6, …, 46]</c>. Its GGUF names use the <c>cross_attn_*</c> dialect plus <c>cross_gate_attn</c> /
/// <c>cross_gate_mlp</c> for the two gates. They are remapped to the Hugging Face names the text decoder reads
/// (<c>model.language_model.*</c>, as in the safetensors checkpoint), so the same loader serves both sources.
/// The vision blocks (<c>v.*</c>) and merger (<c>mm.*</c>) pass through verbatim, except the separator, which takes its
/// checkpoint name <c>model.separator_token</c>.</summary>
public sealed class MossVlKeyMapper : IGgufKeyMapper
{
    private const string Text = "model.language_model";

    public string Architecture => "moss_vl";
    public IReadOnlyCollection<string> Architectures => ["moss_vl"];

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        foreach (string name in tensorNames)
            if (name.Contains(".cross_gate_attn", StringComparison.Ordinal)) return true;
        return false;
    }

    public string? MapKey(string ggufKey)
    {
        if (ggufKey.Equals("token_embd.weight", StringComparison.Ordinal)) return $"{Text}.embed_tokens.weight";
        if (ggufKey.Equals("output_norm.weight", StringComparison.Ordinal)) return $"{Text}.norm.weight";
        if (ggufKey.Equals("output.weight", StringComparison.Ordinal)) return "lm_head.weight";
        if (ggufKey.Equals("mm.separator.weight", StringComparison.Ordinal)) return "model.separator_token";
        if (ggufKey.StartsWith("v.", StringComparison.Ordinal) || ggufKey.StartsWith("mm.", StringComparison.Ordinal))
            return ggufKey;

        if (!ggufKey.StartsWith("blk.", StringComparison.Ordinal)) return null;
        int dot = ggufKey.IndexOf('.', 4);
        if (dot < 0) return null;
        string idx = ggufKey.Substring(4, dot - 4);
        string suffix = ggufKey.Substring(dot + 1);

        string? mapped = suffix switch
        {
            "attn_norm.weight" => "input_layernorm.weight",
            "attn_q.weight" => "self_attn.q_proj.weight",
            "attn_k.weight" => "self_attn.k_proj.weight",
            "attn_v.weight" => "self_attn.v_proj.weight",
            "attn_output.weight" => "self_attn.o_proj.weight",
            "attn_q_norm.weight" => "self_attn.q_norm.weight",
            "attn_k_norm.weight" => "self_attn.k_norm.weight",
            "ffn_norm.weight" => "post_attention_layernorm.weight",
            "ffn_gate.weight" => "mlp.gate_proj.weight",
            "ffn_up.weight" => "mlp.up_proj.weight",
            "ffn_down.weight" => "mlp.down_proj.weight",
            "cross_attn_q.weight" => "cross_attn.q_proj.weight",
            "cross_attn_k.weight" => "cross_attn.k_proj.weight",
            "cross_attn_v.weight" => "cross_attn.v_proj.weight",
            "cross_attn_output.weight" => "cross_attn.o_proj.weight",
            "cross_attn_q_norm.weight" => "cross_attn.q_norm.weight",
            "cross_attn_k_norm.weight" => "cross_attn.k_norm.weight",
            "cross_gate_attn.weight" => "cross_attn_attn_gate",
            "cross_gate_mlp.weight" => "cross_attn_mlp_gate",
            _ => null,
        };
        return mapped is null ? null : $"{Text}.layers.{idx}.{mapped}";
    }
}
