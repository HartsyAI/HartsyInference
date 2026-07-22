namespace HartsyInference.Engine;

/// <summary>One selectable model in the catalog: its CLI id, modality, backing architecture, and maturity.</summary>
public sealed record CatalogEntry
{
    /// <summary>Short CLI selector passed to <c>--model</c> (e.g. "sdxl", "qwen3", "whisper-large-v3").</summary>
    public required string Id { get; init; }

    /// <summary>Which command family this model belongs to.</summary>
    public required Modality Modality { get; init; }

    /// <summary>Human-readable model name for tables and menus.</summary>
    public required string DisplayName { get; init; }

    /// <summary>Underlying architecture family (e.g. "UNet", "MMDiT", "Qwen2", "flow-matching DiT").</summary>
    public required string Architecture { get; init; }

    /// <summary>Real-weight validation maturity.</summary>
    public required ModelStatus Status { get; init; }

    /// <summary>Whether the CLI can actually drive this model today. The engine supports every catalogued architecture,
    /// but the CLI's per-modality handlers currently wire only a subset; the rest are reachable via the SwarmUI
    /// extension or the library API. This keeps <c>hartsy list</c> honest about what <c>hartsy &lt;cmd&gt;</c> can run.</summary>
    public bool CliDrivable { get; init; }

    /// <summary>HuggingFace repo id used by <c>hartsy pull</c> when no local copy is given, when known.</summary>
    public string? HuggingFaceRepo { get; init; }

    /// <summary>The complete set of files this model needs to run (transformer + text encoder + VAE + …). When the
    /// model is selected but not present, the CLI offers to download exactly these into their target folders. Empty
    /// when no preset download is defined.</summary>
    public IReadOnlyList<ModelAsset> Assets { get; init; } = Array.Empty<ModelAsset>();
}
