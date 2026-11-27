namespace HartsyInference.LLM.Hy4Flash;

/// <summary>Shape of a Hy4 Flash compressed indexer (<c>compressed_indexer_*</c>, <c>index_*</c>).</summary>
public sealed record Hy4FlashIndexerSettings
{
    /// <summary>Indexer heads (<c>index_n_heads</c>).</summary>
    public required int Heads { get; init; }

    /// <summary>Width of each query and key (<c>index_head_dim</c>).</summary>
    public required int HeadDim { get; init; }

    /// <summary>Trailing channels that are rotated (<c>qk_rope_head_dim</c>).</summary>
    public required int RopeDim { get; init; }

    /// <summary>Tokens pooled into one group key (<c>index_kpool</c>).</summary>
    public required int Kpool { get; init; }

    /// <summary>Keys an indexed layer attends once its context is longer; attention is dense up to this many (<c>index_topk</c>).</summary>
    public required int TopK { get; init; }

    /// <summary>Epsilon of the key norms (<c>rms_norm_eps</c>).</summary>
    public float NormEps { get; init; } = 1e-5f;

    /// <summary>Complete groups recalled per query, besides its own group.</summary>
    public int RecalledGroups => TopK / Kpool - 1;

    /// <summary>Fails fast on a shape the indexer cannot run.</summary>
    public void Validate()
    {
        if (Heads < 1 || HeadDim < 2 || Kpool < 1 || TopK < Kpool) throw new ArgumentException("Indexer Heads, HeadDim, Kpool and TopK must be positive, TopK at least Kpool.");
        if (TopK % Kpool != 0) throw new ArgumentException("TopK must divide by Kpool.");
        if (RopeDim < 2 || (RopeDim & 1) != 0 || RopeDim > HeadDim) throw new ArgumentException("RopeDim must be even and fit in HeadDim.");
    }
}
