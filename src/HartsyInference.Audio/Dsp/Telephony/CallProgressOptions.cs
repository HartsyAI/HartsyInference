namespace HartsyInference.Audio.Dsp.Telephony;

/// <summary>Thresholds for <see cref="CallProgressClassifier"/>.</summary>
public sealed record CallProgressOptions
{
    /// <summary>Block RMS (dBFS) below which a block is silence. Real lines have a noise floor; raise this for a noisy
    /// trunk.</summary>
    public float SilenceDbfs { get; init; } = -50f;

    /// <summary>Speech probability (from the host's VAD) at or above which a non-tone block counts as speech.</summary>
    public float SpeechProbability { get; init; } = 0.5f;

    /// <summary>Far-end speech, without a reply from the local side, after which a machine greeting is suspected.</summary>
    public int LongGreetingMs { get; init; } = 6_000;

    /// <summary>Pause that ends a far-end utterance. Pauses shorter than this are inside the utterance.</summary>
    public int UtteranceGapMs { get; init; } = 1_200;

    /// <summary>Non-speech sound after which hold music is suspected.</summary>
    public int HoldMusicMs { get; init; } = 4_000;

    /// <summary>Silence after which <see cref="CallProgressKind.PromptSilence"/> is raised.</summary>
    public int PromptSilenceMs { get; init; } = 3_000;

    internal void Validate()
    {
        Require(SilenceDbfs is >= -100f and <= -20f, nameof(SilenceDbfs), "must be -100 to -20 dBFS");
        Require(SpeechProbability is > 0f and <= 1f, nameof(SpeechProbability), "must be in (0, 1]");
        Require(LongGreetingMs >= 1_000, nameof(LongGreetingMs), "must be at least 1000 ms");
        Require(UtteranceGapMs >= 200, nameof(UtteranceGapMs), "must be at least 200 ms");
        Require(HoldMusicMs >= 1_000, nameof(HoldMusicMs), "must be at least 1000 ms");
        Require(PromptSilenceMs >= 1_000, nameof(PromptSilenceMs), "must be at least 1000 ms");
    }

    private static void Require(bool condition, string name, string rule)
    {
        if (!condition)
        {
            throw new ArgumentOutOfRangeException(name, $"{name} {rule}.");
        }
    }
}
