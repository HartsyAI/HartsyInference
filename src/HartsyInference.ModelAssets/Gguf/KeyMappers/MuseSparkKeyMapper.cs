namespace HartsyInference.ModelAssets.Gguf.KeyMappers;

/// <summary>Maps llama.cpp's <c>muse-spark</c> GGUF tensors (Meta Muse Spark Open) onto the HF-style names the transformer loads:
/// the llama dialect plus the attention output gate, the precomputed residual-gate <c>alpha</c>/<c>beta</c>, the latent MoE
/// projections and norms, and the selection-only router bias.</summary>
/// <remarks>llama.cpp's converter (ggml-org/llama.cpp#30616) folds the centered norms' <c>+1</c>, the router's output multiplier and
/// the Q scale into the stored weights and resolves each residual gate to its two coefficient vectors, so the mapped tensors keep
/// the meaning the transformer expects without further conversion.</remarks>
public sealed class MuseSparkKeyMapper : IGgufKeyMapper
{
    private readonly LlamaKeyMapper _llama = new();

    public string Architecture => "muse-spark";

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        foreach (string name in tensorNames)
        {
            if (name.EndsWith(".attn_resid_alpha.weight", StringComparison.Ordinal)) return true;
        }
        return false;
    }

    public string? MapKey(string ggufKey)
    {
        if (!ggufKey.StartsWith("blk.", StringComparison.Ordinal)) return _llama.MapKey(ggufKey);
        int dotAfterIndex = ggufKey.IndexOf('.', 4);
        if (dotAfterIndex < 0) return null;
        string blockIdx = ggufKey.Substring(4, dotAfterIndex - 4);
        string suffix = ggufKey.Substring(dotAfterIndex + 1);
        string? mappedSuffix = suffix switch
        {
            "attn_gate.weight" => "self_attn.gate_proj.weight",
            "attn_resid_alpha.weight" => "post_attention_residual_gate.alpha",
            "attn_resid_beta.weight" => "post_attention_residual_gate.beta",
            "ffn_norm.weight" => "pre_feedforward_layernorm.weight",
            "post_ffw_norm.weight" => "post_feedforward_layernorm.weight",
            "ffn_resid_alpha.weight" => "post_feedforward_residual_gate.alpha",
            "ffn_resid_beta.weight" => "post_feedforward_residual_gate.beta",
            "exp_probs_b.bias" => "mlp.gate.e_score_correction_bias",
            "ffn_latent_down.weight" => "mlp.pre_expert_proj.weight",
            "ffn_latent_up.weight" => "mlp.post_expert_proj.weight",
            "ffn_pre_exp_norm.weight" => "mlp.pre_expert_norm.weight",
            "ffn_post_exp_norm.weight" => "mlp.post_expert_norm.weight",
            _ => null,
        };
        return mappedSuffix is null ? _llama.MapKey(ggufKey) : $"model.layers.{blockIdx}.{mappedSuffix}";
    }
}
