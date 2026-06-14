namespace HartsyInference.Core.Pipelines;

/// <summary>Result of a speech-to-text transcription.</summary>
public sealed class TranscriptionResult
{
    /// <summary>The transcribed text.</summary>
    public required string Text { get; init; }

    /// <summary>Detected language code (e.g., "en", "es").</summary>
    public string? Language { get; init; }

    /// <summary>Individual segments with timestamps.</summary>
    public IReadOnlyList<TranscriptionSegment>? Segments { get; init; }
}
