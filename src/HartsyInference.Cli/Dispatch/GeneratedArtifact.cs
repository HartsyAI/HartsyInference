namespace HartsyInference.Cli.Dispatch;

/// <summary>The in-memory result of one generation, consumed by the CLI for display and by <c>ArtifactWriter</c> for persistence.</summary>
/// <remarks>Handlers encode file bytes themselves so the writer stays generic.</remarks>
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

    /// <summary>True when <see cref="Text"/> was already streamed live, so the presenter should not reprint it.</summary>
    public bool Streamed { get; init; }

    /// <summary>True when the handler already wrote its own file(s) to disk (no single <see cref="FileBytes"/> payload).</summary>
    /// <remarks><c>ArtifactWriter</c> must skip these: without this flag, its "text with no bytes" fallback would still
    /// write <see cref="Text"/> (the human-readable summary) as a bogus file stamped with <see cref="Extension"/> —
    /// e.g. a "stems-0001.wav" that is actually a text file, not audio.</remarks>
    public bool SelfWritten { get; init; }

    /// <summary>Raw RGB24 pixels (row-major, top-to-bottom) for an inline terminal preview; null when there is nothing to show.</summary>
    /// <remarks>For video/world results, this is the first frame.</remarks>
    public byte[]? PreviewRgb { get; init; }

    /// <summary>Pixel width of <see cref="PreviewRgb"/>.</summary>
    public int PreviewWidth { get; init; }

    /// <summary>Pixel height of <see cref="PreviewRgb"/>.</summary>
    public int PreviewHeight { get; init; }

    /// <summary>Free-form metadata surfaced to the user (seed, tok/s, dimensions, stop reason).</summary>
    public Dictionary<string, string> Meta { get; } = new(StringComparer.OrdinalIgnoreCase);
}
