namespace HartsyInference.ModelAssets.SafeTensors;

/// <summary>How a <see cref="ShardedSafeTensorSet"/> opens and reads its shards.</summary>
public sealed record ShardSetOptions
{
    /// <summary>Largest header accepted per shard.</summary>
    public long MaxHeaderBytes { get; init; } = SafeTensorHeaderReader.DefaultMaxHeaderBytes;

    /// <summary>File names of shards never mapped and read only through <see cref="ShardedSafeTensorSet.GetByteSource"/>.</summary>
    public IReadOnlyCollection<string> PreadOnlyShards { get; init; } = [];

    /// <summary>Whether the summed tensor bytes must equal the index's <c>metadata.total_size</c> when it declares one.</summary>
    public bool RequireTotalSizeMatch { get; init; } = true;

    /// <summary>A byte total the tensors must sum to regardless of what the index claims, or null for no pin.</summary>
    public long? ExpectedTotalBytes { get; init; }

    /// <summary>Whether a mapped shard is advised <c>MADV_RANDOM</c> to suppress readahead.</summary>
    public bool AdviseRandom { get; init; } = true;
}
