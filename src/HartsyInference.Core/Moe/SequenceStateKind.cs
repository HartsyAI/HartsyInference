namespace HartsyInference.Core.Moe;

/// <summary>The sequence state a layer keeps between tokens. KV cache is one kind, not the abstraction.</summary>
public enum SequenceStateKind
{
    /// <summary>No persistent state (stateless layer).</summary>
    None,

    /// <summary>Standard per-layer K and V tensors.</summary>
    StandardKv,

    /// <summary>Multi-head latent attention: a compressed latent instead of full K/V.</summary>
    MlaLatent,

    /// <summary>Compressed KV blocks selected by an indexer (DeepSeek-V4.1 compressed sparse attention).</summary>
    CompressedKv,

    /// <summary>Sliding-window KV: only the last W positions persist.</summary>
    SlidingWindowKv,

    /// <summary>Recurrent or SSM state (Gated DeltaNet, Mamba-style blocks).</summary>
    RecurrentState,

    /// <summary>Recomputed from bounded replay rather than stored.</summary>
    Reconstructable,
}
