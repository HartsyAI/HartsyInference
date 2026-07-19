namespace HartsyInference.Engine.Requests;

/// <summary>One word (or token) of a transcript with its time span and optional speaker attribution.</summary>
public sealed record WordSegment
{
    /// <summary>The word text.</summary>
    public required string Word { get; init; }

    /// <summary>Start time in seconds.</summary>
    public required double Start { get; init; }

    /// <summary>End time in seconds.</summary>
    public required double End { get; init; }

    /// <summary>Zero-based speaker index when diarization ran; null otherwise.</summary>
    public int? Speaker { get; init; }
}
