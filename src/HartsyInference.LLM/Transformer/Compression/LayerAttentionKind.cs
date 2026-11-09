namespace HartsyInference.LLM.Transformer.Compression;

/// <summary>Per-layer attention kind from one <c>compress_ratios</c> entry: 0 is standard, <c>R &gt;= 2</c> is compressed KV with one latent per R tokens.</summary>
public readonly record struct LayerAttentionKind(int CompressRatio)
{
    /// <summary>Standard attention with no compressed KV.</summary>
    public static LayerAttentionKind Standard => new(0);

    /// <summary>Whether the layer pools its KV into compressed latents.</summary>
    public bool IsCompressed => CompressRatio > 0;

    /// <summary>Reads one <c>compress_ratios</c> entry. Ratio -2 (DSA, sparse indexer layers) is refused until that path lands.</summary>
    public static LayerAttentionKind FromCompressRatio(int ratio)
    {
        if (ratio == 0) return Standard;
        if (ratio == -2) throw new NotSupportedException("compress_ratio -2 marks a DSA sparse-attention layer, which is not supported yet.");
        if (ratio < 2)
            throw new ArgumentOutOfRangeException(nameof(ratio), ratio, "compress_ratio must be 0 (standard), -2 (DSA, unsupported) or at least 2.");
        return new(ratio);
    }
}
