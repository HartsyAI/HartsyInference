namespace HartsyInference.Audio.Pipelines;

/// <summary>One timestamped chunk of a Whisper transcript, delimited by a pair of native <c>&lt;|t|&gt;</c> tokens.
/// Granularity is whatever the model emitted — an utterance/phrase span, NOT a single word: true word-level
/// alignment needs cross-attention DTW, which <c>WhisperDecoder</c> does not expose (it runs fused SDPA and never
/// returns attention weights). Times are the raw token times in seconds (0.02 s grid, 0..30 s within the chunk).</summary>
public sealed record WhisperSegment
{
    /// <summary>Decoded text between the opening and closing timestamp tokens, trimmed.</summary>
    public required string Text { get; init; }

    /// <summary>Opening timestamp token time in seconds.</summary>
    public required double Start { get; init; }

    /// <summary>Closing timestamp token time in seconds; the clip length when the model never closed the span.</summary>
    public required double End { get; init; }
}
