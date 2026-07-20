namespace HartsyInference.ModelAssets.Gguf.KeyMappers;

/// <summary>GGUF mapper for the llama.cpp **dense-decoder family** — <c>llama</c>, <c>qwen2</c> and <c>qwen3</c> all share one tensor-naming dialect, so one mapper covers them all (declared via <see cref="Architectures"/>). Serves both standalone decoder LLMs (<see cref="HartsyInference.LLM.Transformer.GenericTransformer"/>) and the same models used as conditioning **text encoders** (Qwen3-4B, Mistral-Small-3, Ministral3B, Qwen2.5-VL, etc.) in Z-Image / Flux.2 / ERNIE-Image. Architecture-specific differences (QKV bias on Qwen2, q/k-norm on Qwen3, tied embeddings) are not naming differences — they surface only as tensor *presence* and are detected structurally downstream, so the mechanical remap below is identical for the whole family.
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

    // llama.cpp's convert_hf_to_gguf.py emits an identical blk.N.attn_*/ffn_* dialect across these decoders, so
    // the single remap below is exact for all of them. The MoE members (olmoe, qwen2moe, qwen3moe, and Mixtral
    // under the plain "llama" arch) add the stacked-expert + router tensors handled below; the dense members
    // simply never carry those keys. Registering each explicitly resolves it by name (no heuristic-fallback warning).
    public IReadOnlyCollection<string> Architectures => ["llama", "qwen2", "qwen3", "olmo2", "olmoe", "qwen2moe", "qwen3moe", "granite", "granitemoe", "cohere2", "command-r", "stablelm", "internlm2", "nemotron", "starcoder2", "exaone", "gpt-oss", "gptoss", "minicpm"];

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
        if (ggufKey.Equals("output_norm.bias", StringComparison.Ordinal)) return "model.norm.bias";   // LayerNorm bias (StableLM/GPT-2-lineage)
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
                "attn_q_norm.bias" => "self_attn.q_norm.bias",   // StableLM qk-LayerNorm bias
                "attn_k_norm.weight" => "self_attn.k_norm.weight",
                "attn_k_norm.bias" => "self_attn.k_norm.bias",
                "attn_sinks.weight" => "self_attn.sinks.weight",   // GPT-OSS per-head attention sinks
                "attn_sinks" => "self_attn.sinks.weight",
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                "ffn_norm.bias" => "post_attention_layernorm.bias",
                // OLMo-2 post-norm: norm applied to the attention / FFN output (mapped to the sandwich-norm slots).
                "post_attention_norm.weight" => "post_attention_layernorm.weight",
                "post_ffw_norm.weight" => "post_feedforward_layernorm.weight",
                "ffn_gate.weight" => "mlp.gate_proj.weight",
                "ffn_up.weight" => "mlp.up_proj.weight",
                "ffn_down.weight" => "mlp.down_proj.weight",
                // FFN biases (StarCoder2 / GPT-2-lineage non-gated MLP). Llama/Qwen/Gemma have none.
                "ffn_gate.bias" => "mlp.gate_proj.bias",
                "ffn_up.bias" => "mlp.up_proj.bias",
                "ffn_down.bias" => "mlp.down_proj.bias",
                // MoE: the router and the stacked per-expert tensors. The *_exps tensors are 3D [E, ·, ·] and are
                // split into per-expert 2D projections downstream (GgufLanguageModel). Shared-expert tensors
                // (Qwen-MoE) map to the shared_expert.* names the MoE block expects.
                "ffn_gate_inp.weight" => "mlp.gate.weight",                    // router
                "ffn_gate_exps.weight" => "mlp.gate_exps.weight",             // stacked → split
                "ffn_up_exps.weight" => "mlp.up_exps.weight",
                "ffn_down_exps.weight" => "mlp.down_exps.weight",
                "ffn_gate_shexp.weight" => "mlp.shared_expert.gate_proj.weight",
                "ffn_up_shexp.weight" => "mlp.shared_expert.up_proj.weight",
                "ffn_down_shexp.weight" => "mlp.shared_expert.down_proj.weight",
                "ffn_gate_inp_shexp.weight" => "mlp.shared_expert_gate.weight",
                _ => null,
            };
            if (mappedSuffix is null) return null;
            return $"model.layers.{blockIdx}.{mappedSuffix}";
        }

        // rope_freqs.weight is the precomputed per-frequency multiplier llama.cpp bakes from Llama-3 rope
        // scaling; keep it (the LLM applies it via RopeFrequencyBuilder). Other rope_* tensors are unused.
        if (ggufKey.Equals("rope_freqs.weight", StringComparison.Ordinal)) return "model.rope_freqs.weight";
        if (ggufKey.StartsWith("rope_", StringComparison.Ordinal)) return null;
        if (ggufKey.StartsWith("position_embd.", StringComparison.Ordinal)) return null;

        return null;
    }
}
