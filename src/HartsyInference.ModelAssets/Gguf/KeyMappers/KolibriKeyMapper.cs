namespace HartsyInference.ModelAssets.Gguf.KeyMappers;

/// <summary>Maps Kolibri-1's Qwen3-MoE GGUF tensors, including its sandwich norms and selection-only router bias.</summary>
public sealed class KolibriKeyMapper : IGgufKeyMapper
{
    private readonly LlamaKeyMapper _llama = new();

    public string Architecture => "kolibri1";

    public IReadOnlyCollection<string> Architectures => [Architecture];

    public bool MatchesByKeys(IEnumerable<string> tensorNames) => false;

    public string? MapKey(string ggufKey)
    {
        if (!ggufKey.StartsWith("blk.", StringComparison.Ordinal))
        {
            return _llama.MapKey(ggufKey);
        }
        int dotAfterIndex = ggufKey.IndexOf('.', 4);
        if (dotAfterIndex < 0)
        {
            return null;
        }
        string blockIdx = ggufKey.Substring(4, dotAfterIndex - 4);
        string suffix = ggufKey.Substring(dotAfterIndex + 1);
        string? mappedSuffix = suffix switch
        {
            "post_attention_norm.weight" => "post_attention_layernorm.weight",
            "ffn_norm.weight" => "pre_feedforward_layernorm.weight",
            "post_ffw_norm.weight" => "post_feedforward_layernorm.weight",
            "exp_probs_b.bias" => "mlp.gate.e_score_correction_bias",
            _ => null,
        };
        return mappedSuffix is null ? _llama.MapKey(ggufKey) : $"model.layers.{blockIdx}.{mappedSuffix}";
    }
}
