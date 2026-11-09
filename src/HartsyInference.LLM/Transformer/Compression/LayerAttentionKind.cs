namespace HartsyInference.LLM.Transformer.Compression;

/// <summary>Per-layer attention kind from one <c>compress_ratios</c> entry: 0 is standard, 1 is a compressed KV cache with one latent per
/// token and no pooling (DeepSeek V4.1), <c>R &gt;= 2</c> pools R tokens into each latent (DeepSeek V4.1, ERNIE HCA), and
/// <see cref="DsaRatio"/> is low-rank (MLA) attention over a DSA indexer's sparse set (<see cref="Dsa.DsaAttention"/>).</summary>
/// <remarks>Values come only from <see cref="FromCompressRatio"/>, <see cref="Standard"/> or <see cref="Dsa"/>, so an unvalidated ratio cannot be built.</remarks>
public readonly record struct LayerAttentionKind
{
    private LayerAttentionKind(int compressRatio) => CompressRatio = compressRatio;

    /// <summary>The <c>compress_ratios</c> entry that marks a DSA layer.</summary>
    public const int DsaRatio = -2;

    /// <summary>Standard attention with no compressed KV.</summary>
    public static LayerAttentionKind Standard => default;

    /// <summary>Low-rank attention over an indexer top-k plus a sliding window.</summary>
    public static LayerAttentionKind Dsa => new(DsaRatio);

    /// <summary>The <c>compress_ratios</c> entry: tokens per compressed latent, 0 for standard attention, <see cref="DsaRatio"/> for DSA.</summary>
    public int CompressRatio { get; }

    /// <summary>Whether the layer is standard attention.</summary>
    public bool IsStandard => CompressRatio == 0;

    /// <summary>Whether the layer keeps a compressed KV cache (ratio 1 or more).</summary>
    public bool IsCompressed => CompressRatio > 0;

    /// <summary>Whether the layer pools several tokens into each latent (ratio 2 or more), and so needs a <see cref="KvCompressorState"/>.</summary>
    public bool Pools => CompressRatio > 1;

    /// <summary>Whether the layer is DSA sparse attention.</summary>
    public bool IsDsa => CompressRatio == DsaRatio;

    /// <summary>Reads one <c>compress_ratios</c> entry; a ratio with no implemented kind is refused by name.</summary>
    public static LayerAttentionKind FromCompressRatio(int ratio)
    {
        if (ratio == 0) return Standard;
        if (ratio == DsaRatio) return Dsa;
        if (ratio < 0)
            throw new NotSupportedException(
                $"compress_ratio {ratio} is not a supported attention kind; expected 0 (standard), {DsaRatio} (DSA) or positive (compressed KV).");
        return new(ratio);
    }
}
