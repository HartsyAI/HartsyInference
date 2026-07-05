namespace HartsyInference.Cli.Infra;

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

    /// <summary>HuggingFace repo id used by <c>hartsy pull</c> when no local copy is given, when known.</summary>
    public string? HuggingFaceRepo { get; init; }
}
