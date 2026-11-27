namespace HartsyInference.Diffusion.Streaming;

/// <summary>What a <see cref="RollingTokenWindow"/> keeps as chunks are appended: the first chunks as a permanent sink, and a
/// sliding history bounded by chunk count, by units (frames), or both.</summary>
public sealed record TokenWindowPolicy
{
    /// <summary>Leading chunks kept for the whole stream as an anchor. Zero keeps no sink.</summary>
    public int SinkChunks { get; init; }

    /// <summary>Most chunks the sliding history holds, oldest dropped first; null leaves the chunk count unbounded.</summary>
    public int? MaxHistoryChunks { get; init; }

    /// <summary>Most units (frames) the sliding history holds; the oldest chunk is cut to its newest units when it only partly
    /// fits. Null leaves the unit count unbounded.</summary>
    public int? MaxHistoryUnits { get; init; }

    /// <summary>Keeps every chunk.</summary>
    public static TokenWindowPolicy Unbounded { get; } = new();

    /// <summary>The window of a chunk-causal sampler that attends to <paramref name="cachedChunks"/> earlier chunks: the
    /// newest ones, or, with <paramref name="sink"/>, chunk 0 plus the newest <c>cachedChunks − 1</c>. A count of zero or less
    /// keeps every chunk.</summary>
    public static TokenWindowPolicy RecentChunks(int cachedChunks, bool sink)
    {
        if (cachedChunks <= 0)
            return Unbounded;
        return sink
            ? new TokenWindowPolicy { SinkChunks = 1, MaxHistoryChunks = cachedChunks - 1 }
            : new TokenWindowPolicy { MaxHistoryChunks = cachedChunks };
    }

    /// <summary>A sliding history of the newest <paramref name="units"/> units and no sink.</summary>
    public static TokenWindowPolicy RecentUnits(int units)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(units);
        return new TokenWindowPolicy { MaxHistoryUnits = units };
    }
}
