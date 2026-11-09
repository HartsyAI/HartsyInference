namespace HartsyInference.LLM.Transformer.Compression;

/// <summary>Per-layer attention kind from one <c>compress_ratios</c> entry: 0 is standard, <c>R &gt;= 2</c> is compressed KV with one latent
/// per R tokens, <see cref="DsaRatio"/> is low-rank (MLA) attention over a DSA indexer's sparse set (<see cref="Dsa.DsaAttention"/>).</summary>
public readonly record struct LayerAttentionKind(int CompressRatio)
{
    /// <summary>The <c>compress_ratios</c> entry that marks a DSA layer.</summary>
    public const int DsaRatio = -2;

    /// <summary>Standard attention with no compressed KV.</summary>
    public static LayerAttentionKind Standard => new(0);

    /// <summary>Low-rank attention over an indexer top-k plus a sliding window.</summary>
    public static LayerAttentionKind Dsa => new(DsaRatio);

    /// <summary>Whether the layer is standard attention.</summary>
    public bool IsStandard => CompressRatio == 0;

    /// <summary>Whether the layer pools its KV into compressed latents.</summary>
    public bool IsCompressed => CompressRatio > 0;

    /// <summary>Whether the layer is DSA sparse attention.</summary>
    public bool IsDsa => CompressRatio == DsaRatio;

    /// <summary>Reads one <c>compress_ratios</c> entry; a ratio with no implemented kind is refused by name.</summary>
    public static LayerAttentionKind FromCompressRatio(int ratio)
    {
        if (ratio == 0) return Standard;
        if (ratio == DsaRatio) return Dsa;
        if (ratio < 2)
            throw new NotSupportedException(
                $"compress_ratio {ratio} is not a supported attention kind; expected 0 (standard), {DsaRatio} (DSA) or at least 2 (compressed KV).");
        return new(ratio);
    }
}
