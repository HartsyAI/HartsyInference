namespace HartsyInference.LLM.Looped;

/// <summary>What one physical layer of a looped normalized-GPT decoder computes: Dory's layers are single sublayers, each either an
/// attention mixer or a gated MLP, never both (<c>hybrid_layer_pattern</c> characters <c>S</c>, <c>*</c> and <c>-</c>).</summary>
public enum LoopedLayerKind
{
    /// <summary>Sliding-window causal attention (<c>S</c>): each query sees at most <see cref="LoopedTransformerConfig.SlidingWindow"/> keys.</summary>
    SlidingAttention,

    /// <summary>Full causal attention (<c>*</c>).</summary>
    FullAttention,

    /// <summary>Gated SwiGLU MLP (<c>-</c>); carries no key/value cache.</summary>
    Mlp,
}
