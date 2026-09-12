namespace HartsyInference.Engine.Audio;

/// <summary>Generated audio: mono (<see cref="Right"/> null) or stereo.</summary>
internal readonly record struct MusicAudio
{
    /// <summary>Left (or mono) channel samples.</summary>
    public float[] Left { get; init; }

    /// <summary>Right channel samples, or null for mono.</summary>
    public float[]? Right { get; init; }

    /// <summary>Model-specific facts about how the take came out, merged into <c>AudioResult.Meta</c>. A log line is
    /// invisible over HTTP, so anything a caller would act on (a truncated song, a clamped duration) belongs here.</summary>
    public IReadOnlyDictionary<string, string>? Meta { get; init; }

    /// <summary>Wraps a mono waveform.</summary>
    public static MusicAudio Mono(float[] samples) => new MusicAudio { Left = samples, Right = null };

    /// <summary>Wraps a stereo pair.</summary>
    public static MusicAudio Stereo(float[] left, float[] right) => new MusicAudio { Left = left, Right = right };
}
