namespace HartsyInference.Engine.Audio;

/// <summary>One timestamped span of a transcript. Granularity is whatever the model natively emits — for Whisper
/// that is a phrase/utterance chunk delimited by <c>&lt;|t|&gt;</c> tokens, never a single word.</summary>
internal sealed record SttSegment
{
    /// <summary>The span's transcript text.</summary>
    internal required string Text { get; init; }

    /// <summary>Start time in seconds.</summary>
    internal required double Start { get; init; }

    /// <summary>End time in seconds.</summary>
    internal required double End { get; init; }
}
