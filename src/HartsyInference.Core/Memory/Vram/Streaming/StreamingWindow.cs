namespace HartsyInference.Core.Memory.Vram.Streaming;

/// <summary>The device bytes of a streaming window: <c>prefetchAhead + 1</c> streamed blocks, the block in flight and the ones prefetched
/// ahead. <see cref="BlockStreamingController.Prime"/> uploads the first of them right after the placement, so a caller's headroom has to
/// include these bytes; the widest block bounds them, which is an upper bound for any streamed set.</summary>
public static class StreamingWindow
{
    /// <summary>The window's bytes for <paramref name="denoiser"/> at <paramref name="prefetchAhead"/>, sized at its largest block.</summary>
    public static long Bytes(IStreamableDenoiser denoiser, int prefetchAhead)
    {
        ArgumentNullException.ThrowIfNull(denoiser);
        ArgumentOutOfRangeException.ThrowIfNegative(prefetchAhead);
        long largestBlock = 0;
        for (int i = 0; i < denoiser.BlockCount; i++)
            largestBlock = Math.Max(largestBlock, denoiser.GetBlock(i).EstimatedWeightBytes);
        return (prefetchAhead + 1) * largestBlock;
    }
}
