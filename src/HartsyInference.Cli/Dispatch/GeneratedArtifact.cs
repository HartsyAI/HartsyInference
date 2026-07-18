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

    /// <summary>True when <see cref="Text"/> was already emitted live via the progress sink (e.g. streamed LLM
    /// tokens), so the presenter should not reprint it.</summary>
    public bool Streamed { get; init; }

    /// <summary>Raw RGB24 pixels (row-major, top-to-bottom) for an inline terminal preview; null when there is nothing
    /// to show. For video/world results this is the first frame.</summary>
    public byte[]? PreviewRgb { get; init; }

    /// <summary>Pixel width of <see cref="PreviewRgb"/>.</summary>
    public int PreviewWidth { get; init; }

    /// <summary>Pixel height of <see cref="PreviewRgb"/>.</summary>
    public int PreviewHeight { get; init; }

    /// <summary>Free-form metadata surfaced to the user (seed, tok/s, dimensions, stop reason).</summary>
    public Dictionary<string, string> Meta { get; } = new(StringComparer.OrdinalIgnoreCase);
}
