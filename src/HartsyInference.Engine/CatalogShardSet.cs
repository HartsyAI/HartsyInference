namespace HartsyInference.Engine;

/// <summary>A revision-pinned multi-file download: a Hugging Face repo at one commit, fetched into one folder under the models root.</summary>
public sealed record CatalogShardSet
{
    /// <summary>Hugging Face repo id.</summary>
    public required string Repo { get; init; }

    /// <summary>Commit sha the files are pinned to (a branch name is not a pin).</summary>
    public required string Revision { get; init; }

    /// <summary>Destination folder relative to the models root.</summary>
    public required string TargetSubdir { get; init; }

    /// <summary>The safetensors index that names every shard, or null when <see cref="Files"/> lists them.</summary>
    public string? IndexFile { get; init; }

    /// <summary>Number of files that make up the set (shards for a safetensors index).</summary>
    public required int FileCount { get; init; }

    /// <summary>Sum of the set's file sizes on the hub, in bytes.</summary>
    public required long TotalBytes { get; init; }

    /// <summary>Explicit repo paths for a set with no index (GGUF files); empty when <see cref="IndexFile"/> drives it.</summary>
    public IReadOnlyList<string> Files { get; init; } = [];
}
