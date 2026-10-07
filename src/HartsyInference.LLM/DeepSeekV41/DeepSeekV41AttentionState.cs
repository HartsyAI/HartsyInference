namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>One attention layer's cache for one sequence: the sliding-window ring and, on a source layer, the compressed latents, partial group and index keys.</summary>
/// <remarks>Latents are stored already rotated and through their quantize-dequantize round trip, so F32 rows are exactly what the quantized cache would decode to.</remarks>
public sealed class DeepSeekV41AttentionState
{
    /// <summary>Ring of the last <c>Window</c> latents, slot <c>position % Window</c>, <c>[Window, HeadDim]</c>.</summary>
    public float[] Window { get; }

    /// <summary>Compressed latents, <c>[rows, HeadDim]</c>; null unless this layer compresses its own KV.</summary>
    public float[]? CompressKv { get; }

    /// <summary>Partial pooling group; null unless this layer pools (ratio above 1).</summary>
    public DeepSeekV41CompressorState? Compressor { get; }

    /// <summary>Index keys, <c>[rows, IndexHeadDim]</c>; null unless this layer owns them.</summary>
    public float[]? IndexKeys { get; }

    /// <summary>Creates empty state sized for sequences up to <paramref name="maxTokens"/>.</summary>
    public DeepSeekV41AttentionState(DeepSeekV41AttentionSettings settings, int maxTokens)
    {
        ArgumentNullException.ThrowIfNull(settings);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxTokens, 1);
        Window = new float[settings.Window * settings.HeadDim];
        if (settings.CompressRatio <= 0 || !settings.IsKvSource) return;
        int rows = (maxTokens + settings.CompressRatio - 1) / settings.CompressRatio;
        CompressKv = new float[rows * settings.HeadDim];
        if (settings.CompressRatio > 1) Compressor = new DeepSeekV41CompressorState(settings.CompressRatio, settings.HeadDim);
        if (settings.IsIndexSource) IndexKeys = new float[rows * settings.IndexHeadDim];
    }

    /// <summary>Forgets everything, ready for a new sequence.</summary>
    public void Reset()
    {
        Array.Clear(Window);
        if (CompressKv is not null) Array.Clear(CompressKv);
        if (IndexKeys is not null) Array.Clear(IndexKeys);
        Compressor?.Reset();
    }
}
