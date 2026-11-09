namespace HartsyInference.LLM.Transformer.Compression;

/// <summary>Per-layer attention kind from one <c>compress_ratios</c> entry: 0 is standard, 1 is a compressed KV cache with one latent per
/// token and no pooling (DeepSeek V4.1), <c>R &gt;= 2</c> pools R tokens into each latent (DeepSeek V4.1, ERNIE HCA).</summary>
/// <remarks>Values come only from <see cref="FromCompressRatio"/> or <see cref="Standard"/>, so an unvalidated ratio cannot be built.</remarks>
public readonly record struct LayerAttentionKind
{
    private LayerAttentionKind(int compressRatio) => CompressRatio = compressRatio;

    /// <summary>Standard attention with no compressed KV.</summary>
    public static LayerAttentionKind Standard => default;

    /// <summary>Tokens per compressed latent; 0 for standard attention.</summary>
    public int CompressRatio { get; }

    /// <summary>Whether the layer keeps a compressed KV cache (ratio 1 or more).</summary>
    public bool IsCompressed => CompressRatio > 0;

    /// <summary>Whether the layer pools several tokens into each latent (ratio 2 or more), and so needs a <see cref="KvCompressorState"/>.</summary>
    public bool Pools => CompressRatio > 1;

    /// <summary>Reads one <c>compress_ratios</c> entry. Ratio -2 (DSA, sparse indexer layers) is refused until that path lands.</summary>
    public static LayerAttentionKind FromCompressRatio(int ratio)
    {
        if (ratio == 0) return Standard;
        if (ratio == -2) throw new NotSupportedException("compress_ratio -2 marks a DSA sparse-attention layer, which is not supported yet.");
        if (ratio < 0)
            throw new ArgumentOutOfRangeException(nameof(ratio), ratio, "compress_ratio must be 0 (standard), -2 (DSA, unsupported) or positive.");
        return new(ratio);
    }
}
