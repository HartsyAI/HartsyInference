namespace HartsyInference.ModelAssets.Gguf.KeyMappers;

/// <summary>Maps llama.cpp's <c>dory</c> GGUF tensors (NVIDIA Nemotron Dory, ggml-org/llama.cpp#30306) onto the HF-style names the looped
/// transformer loads: the llama dialect for the projections, plus the attention output gate, the nGPT scale vectors (per-layer residual
/// alpha, query/key scale, gate scale, MLP gate/up scales) and the per-vocabulary logit scale.</summary>
/// <remarks>Each physical layer carries either the attention tensors or the MLP tensors, never both; the converter appends <c>.weight</c> to
/// the scale vectors, which the HF export stores without a suffix.</remarks>
public sealed class DoryKeyMapper : IGgufKeyMapper
{
    private readonly LlamaKeyMapper _llama = new();

    public string Architecture => "dory";

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        foreach (string name in tensorNames)
        {
            if (name.EndsWith(".residual_alpha.weight", StringComparison.Ordinal)) return true;
        }
        return false;
    }

    public string? MapKey(string ggufKey)
    {
        if (ggufKey.Equals("output_scale.weight", StringComparison.Ordinal)) return "lm_head.logit_scale.weight";
        if (!ggufKey.StartsWith("blk.", StringComparison.Ordinal)) return _llama.MapKey(ggufKey);
        int dotAfterIndex = ggufKey.IndexOf('.', 4);
        if (dotAfterIndex < 0) return null;
        string blockIdx = ggufKey.Substring(4, dotAfterIndex - 4);
        string suffix = ggufKey.Substring(dotAfterIndex + 1);
        string? mappedSuffix = suffix switch
        {
            "residual_alpha.weight" => "residual_alpha.weight",
            "attn_gate.weight" => "self_attn.gate_proj.weight",
            "attn_qk_scale.weight" => "self_attn.qk_scale.weight",
            "attn_gate_scale.weight" => "self_attn.gate_scale.weight",
            "ffn_gate_scale.weight" => "mlp.gate_scale.weight",
            "ffn_up_scale.weight" => "mlp.up_scale.weight",
            _ => null,
        };
        return mappedSuffix is null ? _llama.MapKey(ggufKey) : $"model.layers.{blockIdx}.{mappedSuffix}";
    }
}
