namespace HartsyInference.Cli.Dispatch;

/// <summary>The in-memory result of one generation, produced by a handler and consumed by the CLI for display and
/// by <c>ArtifactWriter</c> for persistence. Handlers encode file bytes themselves so the writer stays generic.</summary>
public sealed class GeneratedArtifact
{
    /// <summary>The output form (drives writing and preview).</summary>
    public required ArtifactKind Kind { get; init; }

    /// <summary>Text payload for <see cref="ArtifactKind.Text"/>, or a human-readable summary for other kinds.</summary>
    public string? Text { get; init; }

    /// <summary>Encoded file bytes ready to write (PNG/WAV/GLB/…); null for pure-text results.</summary>
    public byte[]? FileBytes { get; init; }

    /// <summary>Suggested file extension without the dot (e.g. "png", "wav", "glb", "txt").</summary>
    public string Extension { get; init; } = "txt";

    /// <summary>Free-form metadata surfaced to the user (seed, tok/s, dimensions, stop reason).</summary>
    public Dictionary<string, string> Meta { get; } = new(StringComparer.OrdinalIgnoreCase);
}
