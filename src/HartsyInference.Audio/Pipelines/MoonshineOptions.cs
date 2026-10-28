namespace HartsyInference.Audio.Pipelines;

/// <summary>Inference knobs for a single Moonshine transcription. Moonshine has no
/// language / task selectors (it's English-only and only transcribes) so this is
/// a much thinner surface than <c>WhisperOptions</c>.</summary>
public sealed record MoonshineOptions
{
    /// <summary>Maximum new tokens, capped at the decoder's positions. 0 derives it from the clip length (6.5 per second).</summary>
    public int MaxNewTokens { get; init; }

    /// <summary>Greedy only in v1.</summary>
    public float Temperature { get; init; } = 0f;
}
