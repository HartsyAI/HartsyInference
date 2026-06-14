namespace HartsyInference.Audio.Pipelines;

/// <summary>Inference knobs for a single Moonshine transcription. Moonshine has no
/// language / task selectors (it's English-only and only transcribes) so this is
/// a much thinner surface than <c>WhisperOptions</c>.</summary>
public sealed record MoonshineOptions
{
    /// <summary>Maximum new tokens to generate before forcing EOS. Moonshine's MaxTextPositions
    /// is 194, leaving plenty of headroom even for the longest viable clip.</summary>
    public int MaxNewTokens { get; init; } = 180;

    /// <summary>Greedy only in v1.</summary>
    public float Temperature { get; init; } = 0f;
}
