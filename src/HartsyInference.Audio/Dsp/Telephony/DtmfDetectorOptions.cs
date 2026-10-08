namespace HartsyInference.Audio.Dsp.Telephony;

/// <summary>Thresholds for <see cref="DtmfDetector"/>. The defaults follow the usual telephony receiver limits
/// (ITU-T Q.24 style) and were tuned against synthetic keys, noise and speech; see the voice-agent design document.</summary>
public sealed record DtmfDetectorOptions
{
    /// <summary>Shortest tone reported, in milliseconds. The standard asks receivers to accept 40 ms.</summary>
    public int MinToneMs { get; init; } = 40;

    /// <summary>Absence that ends a held tone, in milliseconds; a dropout shorter than this does not split a key and the
    /// same key is not reported again until it has passed.</summary>
    public int MinPauseMs { get; init; } = 40;

    /// <summary>Smallest amplitude (±1 scale) of each of the two components.</summary>
    public float MinToneAmplitude { get; init; } = 0.002f;

    /// <summary>Share of the block's energy the two components must carry. Pure tones carry nearly all of it; speech
    /// carries a few percent.</summary>
    public float MinTonalFraction { get; init; } = 0.6f;

    /// <summary>How far the strongest frequency of a group must stand above the next one, in dB.</summary>
    public float GroupDominanceDb { get; init; } = 6f;

    /// <summary>Largest amount the high-group tone may be weaker than the low-group one, in dB (forward twist).</summary>
    public float MaxForwardTwistDb { get; init; } = 8f;

    /// <summary>Largest amount the high-group tone may be stronger than the low-group one, in dB (reverse twist).</summary>
    public float MaxReverseTwistDb { get; init; } = 4f;

    /// <summary>Largest second-harmonic energy of the low-group tone relative to the tone. Speech has strong harmonics; a
    /// key does not.</summary>
    public float MaxSecondHarmonicRatioLow { get; init; } = 0.25f;

    /// <summary>Largest second-harmonic energy of the high-group tone relative to the tone.</summary>
    public float MaxSecondHarmonicRatioHigh { get; init; } = 0.1f;

    internal void Validate()
    {
        Require(MinToneMs is >= 20 and <= 500, nameof(MinToneMs), "must be 20-500 ms");
        Require(MinPauseMs is >= 10 and <= 500, nameof(MinPauseMs), "must be 10-500 ms");
        Require(MinToneAmplitude > 0f, nameof(MinToneAmplitude), "must be positive");
        Require(MinTonalFraction is > 0f and <= 1f, nameof(MinTonalFraction), "must be in (0, 1]");
        Require(GroupDominanceDb >= 0f, nameof(GroupDominanceDb), "must be non-negative");
        Require(MaxForwardTwistDb >= 0f && MaxReverseTwistDb >= 0f, nameof(MaxForwardTwistDb), "twist limits must be non-negative");
        Require(MaxSecondHarmonicRatioLow > 0f && MaxSecondHarmonicRatioHigh > 0f, nameof(MaxSecondHarmonicRatioLow), "harmonic limits must be positive");
    }

    private static void Require(bool condition, string name, string rule)
    {
        if (!condition)
        {
            throw new ArgumentOutOfRangeException(name, $"{name} {rule}.");
        }
    }
}
