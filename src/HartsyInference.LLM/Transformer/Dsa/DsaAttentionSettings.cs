using HartsyInference.Core.Backends.Rotary;

namespace HartsyInference.LLM.Transformer.Dsa;

/// <summary>Shape of one DSA attention layer: low-rank (MLA) queries and key/value latent, a sink, and a sparse key set of an indexer
/// top-k plus a sliding window.</summary>
public sealed record DsaAttentionSettings
{
    /// <summary>Hidden width.</summary>
    public required int Dim { get; init; }

    /// <summary>Attention heads.</summary>
    public required int Heads { get; init; }

    /// <summary>Query latent width (<c>q_lora_rank</c>).</summary>
    public required int QLoraRank { get; init; }

    /// <summary>Key/value latent width (<c>kv_lora_rank</c>).</summary>
    public required int KvLoraRank { get; init; }

    /// <summary>Per-head query/key width without position (<c>qk_nope_head_dim</c>).</summary>
    public required int QkNopeDim { get; init; }

    /// <summary>Per-head rotated query/key width, shared by every head on the key side (<c>qk_rope_head_dim</c>).</summary>
    public required int QkRopeDim { get; init; }

    /// <summary>Per-head value width (<c>v_head_dim</c>).</summary>
    public required int VDim { get; init; }

    /// <summary>Sliding window: the newest keys every query attends besides the indexer's picks.</summary>
    public required int Window { get; init; }

    /// <summary>Epsilon of the two latent RMS norms.</summary>
    public required float NormEps { get; init; }

    /// <summary>The indexer that picks older keys.</summary>
    public required DsaIndexerSettings Indexer { get; init; }

    /// <summary>Pairing of the main query/key rotation; ERNIE 5.5's is interleaved (not NeoX) on the trailing <see cref="QkRopeDim"/> values.</summary>
    public RotaryPairing RopePairing { get; init; } = RotaryPairing.Interleaved;

    /// <summary>Per-head query/key width.</summary>
    public int QkHeadDim => QkNopeDim + QkRopeDim;

    /// <summary>One cached latent row: the normalized key/value latent followed by the rotated shared key.</summary>
    public int LatentDim => KvLoraRank + QkRopeDim;

    /// <summary>Softmax scale <c>QkHeadDim^-0.5</c>; the absorbed form keeps it, since absorption leaves every dot product unchanged.</summary>
    public float Scale => 1f / MathF.Sqrt(QkHeadDim);

    /// <summary>Fails fast on a shape the layer cannot run.</summary>
    public void Validate()
    {
        if (Dim < 1 || Heads < 1 || QLoraRank < 1 || KvLoraRank < 1 || QkNopeDim < 1 || VDim < 1 || Window < 1)
            throw new ArgumentException("DSA widths, heads and window must be positive.");
        if (QkRopeDim < 2 || (QkRopeDim & 1) != 0) throw new ArgumentException("QkRopeDim must be even and positive.");
        ArgumentNullException.ThrowIfNull(Indexer);
        Indexer.Validate();
    }
}
