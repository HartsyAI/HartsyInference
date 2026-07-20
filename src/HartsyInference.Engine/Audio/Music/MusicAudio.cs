namespace HartsyInference.Engine.Audio;

/// <summary>Generated audio: mono (<see cref="Right"/> null) or stereo.</summary>
internal readonly record struct MusicAudio
{
    /// <summary>Left (or mono) channel samples.</summary>
    public float[] Left { get; init; }

    /// <summary>Right channel samples, or null for mono.</summary>
    public float[]? Right { get; init; }

    /// <summary>Wraps a mono waveform.</summary>
    public static MusicAudio Mono(float[] samples) => new MusicAudio { Left = samples, Right = null };

    /// <summary>Wraps a stereo pair.</summary>
    public static MusicAudio Stereo(float[] left, float[] right) => new MusicAudio { Left = left, Right = right };
}
