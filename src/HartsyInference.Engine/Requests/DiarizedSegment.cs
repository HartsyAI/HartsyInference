namespace HartsyInference.Engine.Requests;

/// <summary>One contiguous span attributed to a single speaker by diarization.</summary>
public sealed record DiarizedSegment
{
    /// <summary>Zero-based speaker index.</summary>
    public required int Speaker { get; init; }

    /// <summary>Start time in seconds.</summary>
    public required double Start { get; init; }

    /// <summary>End time in seconds.</summary>
    public required double End { get; init; }

    /// <summary>The transcript text for this span, when available.</summary>
    public string? Text { get; init; }
}
