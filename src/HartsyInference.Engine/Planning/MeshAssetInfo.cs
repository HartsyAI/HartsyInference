namespace HartsyInference.Engine.Planning;

/// <summary>One file a 3D model needs, and whether it is on disk where the loader will find it.</summary>
public sealed record MeshAssetInfo
{
    /// <summary>What the file is (<c>structure flow</c>, <c>encoder (gated)</c>).</summary>
    public required string Role { get; init; }

    /// <summary>Where the catalog puts the file, relative to the models root; never an absolute path.</summary>
    public required string RelativePath { get; init; }

    /// <summary>False when only an optional feature (the texture, the matting model) needs the file.</summary>
    public required bool Required { get; init; }

    /// <summary>Whether the loader will find the file: at its catalog path, under a legacy name, or in the folder the loader also searches.</summary>
    public required bool Present { get; init; }

    /// <summary>The size of the file on disk; null while it is absent (the catalog records hashes, not sizes).</summary>
    public long? SizeBytes { get; init; }

    /// <summary>The repository is gated.</summary>
    public bool Gated { get; init; }

    /// <summary>Why the engine cannot fetch this file when it is absent, or null when it can.</summary>
    public MeshAssetBlockerKind? Blocker { get; init; }

    /// <summary>What the user must do when <see cref="Blocker"/> is set.</summary>
    public string? Instructions { get; init; }
}
