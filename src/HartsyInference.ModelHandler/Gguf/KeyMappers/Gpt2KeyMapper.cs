namespace HartsyInference.ModelHandler.Gguf.KeyMappers;

/// <summary>GGUF mapper for the GPT-2 lineage (<c>gpt2</c>). This dialect differs from the llama family in three
/// ways: a <i>fused</i> <c>attn_qkv</c> (weight + bias) split downstream into q/k/v
/// (<see cref="HartsyInference.LLM.Generation.GgufLanguageModel"/>), learned <i>absolute</i> position embeddings
/// (<c>position_embd</c>, added to the token embeddings — GPT-2 uses no RoPE), and LayerNorm + projection
/// <i>biases</i> on every block (attention output and the non-gated GELU FFN). Reusable for StarCoder-1 and the
/// GPT-NeoX/BLOOM word-embedding lineage once their config presets land.</summary>
public sealed class Gpt2KeyMapper : IGgufKeyMapper
{
    public string Architecture => "gpt2";

    // BLOOM shares the exact GPT-2 tensor dialect (fused attn_qkv + LayerNorm/FFN biases), differing only in
    // ALiBi-vs-abs-pos (config) and an extra word-embedding LayerNorm (token_embd_norm, mapped below).
    public IReadOnlyCollection<string> Architectures => ["gpt2", "bloom"];

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        bool hasQkv = false, hasPos = false;
        foreach (string name in tensorNames)
        {
            if (name.EndsWith(".attn_qkv.weight", StringComparison.Ordinal)) hasQkv = true;
            if (name.Equals("position_embd.weight", StringComparison.Ordinal)) hasPos = true;
            if (hasQkv && hasPos) return true;
        }
        return false;
    }

    public string? MapKey(string ggufKey)
    {
        switch (ggufKey)
        {
            case "token_embd.weight": return "model.embed_tokens.weight";
            case "position_embd.weight": return "model.embed_positions.weight";   // GPT-2 absolute position table
            case "token_embd_norm.weight": return "model.embed_norm.weight";      // BLOOM word-embedding LayerNorm
            case "token_embd_norm.bias": return "model.embed_norm.bias";
            case "output_norm.weight": return "model.norm.weight";
            case "output_norm.bias": return "model.norm.bias";
            case "output.weight": return "lm_head.weight";
        }

        if (ggufKey.StartsWith("blk.", StringComparison.Ordinal))
        {
            int dotAfterIndex = ggufKey.IndexOf('.', 4);
            if (dotAfterIndex < 0) return null;
            string blockIdx = ggufKey.Substring(4, dotAfterIndex - 4);
            string suffix = ggufKey.Substring(dotAfterIndex + 1);

            string? mappedSuffix = suffix switch
            {
                "attn_norm.weight" => "input_layernorm.weight",
                "attn_norm.bias" => "input_layernorm.bias",
                "attn_qkv.weight" => "self_attn.qkv_proj.weight",   // fused → split into q/k/v downstream
                "attn_qkv.bias" => "self_attn.qkv_proj.bias",       // fused bias → split alongside
                "attn_output.weight" => "self_attn.o_proj.weight",
                "attn_output.bias" => "self_attn.o_proj.bias",
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                "ffn_norm.bias" => "post_attention_layernorm.bias",
                "ffn_up.weight" => "mlp.up_proj.weight",            // non-gated FFN (fc1)
                "ffn_up.bias" => "mlp.up_proj.bias",
                "ffn_down.weight" => "mlp.down_proj.weight",        // non-gated FFN (fc2)
                "ffn_down.bias" => "mlp.down_proj.bias",
                _ => null,
            };
            return mappedSuffix is null ? null : $"model.layers.{blockIdx}.{mappedSuffix}";
        }
        return null;
    }
}
