namespace HartsyInference.Core.Moe;

/// <summary>
/// The sequence state a layer keeps between tokens. A layer can own several kinds at once (DeepSeek-V4.1 source layers keep a
/// sliding-window ring and compressed latents). KV cache is one kind, not the abstraction.
/// </summary>
[Flags]
public enum SequenceStateKind
{
    /// <summary>No persistent state (stateless layer).</summary>
    None = 0,

    /// <summary>Standard per-layer K and V tensors.</summary>
    StandardKv = 1 << 0,

    /// <summary>Multi-head latent attention: a compressed latent instead of full K/V.</summary>
    MlaLatent = 1 << 1,

    /// <summary>Compressed KV blocks selected by an indexer (DeepSeek-V4.1 compressed sparse attention).</summary>
    CompressedKv = 1 << 2,

    /// <summary>Sliding-window KV: only the last W positions persist.</summary>
    SlidingWindowKv = 1 << 3,

    /// <summary>Recurrent or SSM state (Gated DeltaNet, Mamba-style blocks).</summary>
    RecurrentState = 1 << 4,

    /// <summary>Recomputed from bounded replay rather than stored.</summary>
    Reconstructable = 1 << 5,
}
