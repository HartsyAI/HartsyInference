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

    /// <summary>Small files that make the folder loadable beyond the weights (config, tokenizer, processor JSON), fetched from the same
    /// repo and revision into the same folder. They are not counted in <see cref="FileCount"/> or <see cref="TotalBytes"/>.</summary>
    public IReadOnlyList<string> SupportFiles { get; init; } = [];

    /// <summary>SHA-256 of files by repo path (the hub's LFS object ids for shards, hashes of the pinned revision for the rest). A file with
    /// no entry downloads unverified, which is only acceptable for a small file at a pinned revision.</summary>
    public IReadOnlyDictionary<string, string> Sha256ByFile { get; init; } = new Dictionary<string, string>();

    /// <summary>Every repo path the set downloads, in order: the <see cref="IndexFile"/>, the weight files, then
    /// <see cref="SupportFiles"/>. Weight files are <see cref="Files"/> when listed, otherwise the Hugging Face shard names
    /// <c>model-00001-of-00005.safetensors</c> that an index set uses (<see cref="FileCount"/> of them).</summary>
    public IReadOnlyList<string> AllFiles()
    {
        List<string> all = [];
        if (IndexFile is not null) all.Add(IndexFile);
        if (Files.Count > 0) all.AddRange(Files);
        else if (IndexFile is not null)
            all.AddRange(Enumerable.Range(1, FileCount).Select(i => $"model-{i:D5}-of-{FileCount:D5}.safetensors"));
        all.AddRange(SupportFiles);
        return all;
    }
}
