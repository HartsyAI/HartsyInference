namespace HartsyInference.ModelHandler.Gguf.KeyMappers;

/// <summary>Llama-family GGUF mapper for **text encoders** (Qwen3-4B, Mistral-Small-3, Ministral3B, Qwen2.5-VL, etc.) used as conditioning encoders in Z-Image / Flux.2 / ERNIE-Image.
///
/// <para>llama.cpp's GGUF naming is its own dialect — keys like <c>blk.{i}.attn_q.weight</c>, <c>blk.{i}.ffn_gate.weight</c>, <c>token_embd.weight</c>, <c>output_norm.weight</c>. <see cref="HartsyInference.Diffusion.Models.TextEncoders.LlamaStyleEncoder"/> expects HuggingFace transformers naming: <c>model.layers.{i}.self_attn.q_proj.weight</c>, <c>model.layers.{i}.mlp.gate_proj.weight</c>, <c>model.embed_tokens.weight</c>, <c>model.norm.weight</c>. We rewrite at the GGUF level.</para>
///
/// <para>The remap is mostly mechanical string-replacement following llama.cpp's <c>convert_hf_to_gguf.py</c> in reverse:</para>
/// <list type="bullet">
/// <item><c>token_embd.weight</c> → <c>model.embed_tokens.weight</c></item>
/// <item><c>output_norm.weight</c> → <c>model.norm.weight</c></item>
/// <item><c>output.weight</c> → <c>lm_head.weight</c> (LM head; ignored by feature-extractor encoders)</item>
/// <item><c>blk.{i}.attn_norm.weight</c> → <c>model.layers.{i}.input_layernorm.weight</c></item>
/// <item><c>blk.{i}.attn_q.weight</c> → <c>model.layers.{i}.self_attn.q_proj.weight</c></item>
/// <item><c>blk.{i}.attn_k.weight</c> → <c>model.layers.{i}.self_attn.k_proj.weight</c></item>
/// <item><c>blk.{i}.attn_v.weight</c> → <c>model.layers.{i}.self_attn.v_proj.weight</c></item>
/// <item><c>blk.{i}.attn_output.weight</c> → <c>model.layers.{i}.self_attn.o_proj.weight</c></item>
/// <item><c>blk.{i}.attn_q_norm.weight</c> → <c>model.layers.{i}.self_attn.q_norm.weight</c></item>
/// <item><c>blk.{i}.attn_k_norm.weight</c> → <c>model.layers.{i}.self_attn.k_norm.weight</c></item>
/// <item><c>blk.{i}.ffn_norm.weight</c> → <c>model.layers.{i}.post_attention_layernorm.weight</c></item>
/// <item><c>blk.{i}.ffn_gate.weight</c> → <c>model.layers.{i}.mlp.gate_proj.weight</c></item>
/// <item><c>blk.{i}.ffn_up.weight</c> → <c>model.layers.{i}.mlp.up_proj.weight</c></item>
/// <item><c>blk.{i}.ffn_down.weight</c> → <c>model.layers.{i}.mlp.down_proj.weight</c></item>
/// </list></summary>
public sealed class LlamaKeyMapper : IGgufKeyMapper
{
    public string Architecture => "llama";

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        bool hasBlk = false;
        bool hasTokenEmbd = false;
        foreach (string name in tensorNames)
        {
            if (name.StartsWith("blk.", StringComparison.Ordinal)) hasBlk = true;
            if (name.Equals("token_embd.weight", StringComparison.Ordinal)) hasTokenEmbd = true;
            if (hasBlk && hasTokenEmbd) return true;
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
                "attn_norm.bias" => "input_layernorm.bias",
                "attn_q.weight" => "self_attn.q_proj.weight",
                "attn_q.bias" => "self_attn.q_proj.bias",
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_k.bias" => "self_attn.k_proj.bias",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_v.bias" => "self_attn.v_proj.bias",
                "attn_output.weight" => "self_attn.o_proj.weight",
                "attn_output.bias" => "self_attn.o_proj.bias",
                "attn_q_norm.weight" => "self_attn.q_norm.weight",
                "attn_k_norm.weight" => "self_attn.k_norm.weight",
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                "ffn_norm.bias" => "post_attention_layernorm.bias",
                "ffn_gate.weight" => "mlp.gate_proj.weight",
                "ffn_up.weight" => "mlp.up_proj.weight",
                "ffn_down.weight" => "mlp.down_proj.weight",
                _ => null,
            };
            if (mappedSuffix is null) return null;
            return $"model.layers.{blockIdx}.{mappedSuffix}";
        }

        if (ggufKey.StartsWith("rope_", StringComparison.Ordinal)) return null;
        if (ggufKey.StartsWith("position_embd.", StringComparison.Ordinal)) return null;

        return null;
    }
}
