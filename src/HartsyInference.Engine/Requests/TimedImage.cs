namespace HartsyInference.Engine.Requests;

/// <summary>One video frame with its time in the clip or stream.</summary>
public sealed record TimedImage
{
    /// <summary>The frame.</summary>
    public required ImageData Image { get; init; }

    /// <summary>Seconds from the start of the clip or session.</summary>
    public required double TimestampSeconds { get; init; }
}
