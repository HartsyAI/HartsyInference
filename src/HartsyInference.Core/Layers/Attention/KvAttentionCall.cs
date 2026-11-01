using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Layers.Attention;

/// <summary>The per-call inputs of the key/value attention stages: positions and the per-layer constants of the mask and logits.</summary>
public readonly record struct KvAttentionCall
{
    /// <summary>Keys attended: <c>posStart + t</c> on the host; a placeholder (0) on a device-position call.</summary>
    public int KvLength { get; init; }

    /// <summary>Absolute position of query row 0.</summary>
    public int QueryOffset { get; init; }

    /// <summary>False is causal.</summary>
    public bool Bidirectional { get; init; }

    /// <summary>FlashAttention convention: 0 none, N = the query's own key and the N-1 before it.</summary>
    public int SlidingWindow { get; init; }

    /// <summary>Attention-logit soft-cap; 0 none.</summary>
    public float Softcap { get; init; }

    /// <summary>Per-head sink logits.</summary>
    public Tensor? Sinks { get; init; }

    /// <summary>Per-head ALiBi slopes.</summary>
    public Tensor? AlibiSlopes { get; init; }

    /// <summary>Additive F32 bias broadcastable to <c>[1, heads, Sq, KvLength]</c>, where −∞ removes a key (MiniMax's block selection). The caller keeps ownership.</summary>
    public Tensor? Bias { get; init; }
}
