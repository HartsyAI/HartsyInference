namespace HartsyInference.LLM.Glm5Next;

/// <summary>The token mixer of one GLM-5.3-Flash decoder layer.</summary>
public enum Glm5NextAttentionKind
{
    /// <summary>Kimi Delta Attention: short causal conv, per-channel forget gate, delta-rule recurrence (<c>linear_attention</c>).</summary>
    Kda,

    /// <summary>NoPE multi-head latent attention over the keys a compressed-pool indexer selects (<c>deepseek_sparse_attention</c>, later spelled <c>indexed_attention</c>).</summary>
    Dsa,
}
