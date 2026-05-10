namespace SharpInference.Core.Pipelines;

/// <summary>A segment of transcribed audio with timing information.</summary>
public sealed class TranscriptionSegment
{
    /// <summary>Transcribed text for this segment.</summary>
    public required string Text { get; init; }

    /// <summary>Start time in the audio.</summary>
    public TimeSpan Start { get; init; }

    /// <summary>End time in the audio.</summary>
    public TimeSpan End { get; init; }
}
