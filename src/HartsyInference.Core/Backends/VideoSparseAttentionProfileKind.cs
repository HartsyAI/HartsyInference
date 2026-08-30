namespace HartsyInference.Core.Backends;

/// <summary>Versioned routing semantics for MiniMax-H3 video sparse attention artifacts.</summary>
public enum VideoSparseAttentionProfileKind
{
    /// <summary>Kijai/ComfySOL 64-token routing: quantized centroids, strict threshold, sinks, and forced neighbours.</summary>
    ComfySol64V1 = 1,

    /// <summary>FastVideo's published 64-token pooled-routing and top-k semantics.</summary>
    FastVideoVsa64V1 = 2,
}
