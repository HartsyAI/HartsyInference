namespace HartsyInference.Cli.Infra;

/// <summary>A model found on disk under the models root: display label, load path, and whether it's a directory or single file.</summary>
public sealed record DiscoveredModel
{
    /// <summary>Display label — the path relative to the models root (e.g. "Stable-Diffusion/SDXL/sd_xl_base_1.0.safetensors").</summary>
    public required string Label { get; init; }

    /// <summary>Absolute path passed to the resolver / handler as the selected model.</summary>
    public required string Path { get; init; }

    /// <summary>True when <see cref="Path"/> is a directory (a multi-file model), false for a single checkpoint file.</summary>
    public required bool IsDirectory { get; init; }

    /// <summary>Size in bytes for a file model, or 0 for a directory (not summed).</summary>
    public long SizeBytes { get; init; }
}
