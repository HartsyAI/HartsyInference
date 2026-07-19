namespace HartsyInference.Engine.Requests;

/// <summary>The result of a speech / music / voice-conversion generation: encoded audio bytes plus its format,
/// duration, and sample rate.</summary>
public sealed record AudioResult
{
    /// <summary>Encoded audio container bytes (WAV by default).</summary>
    public required byte[] Data { get; init; }

    /// <summary>Container/codec of <see cref="Data"/> (e.g. "wav").</summary>
    public string Format { get; init; } = "wav";

    /// <summary>Duration of the audio in seconds.</summary>
    public double DurationSeconds { get; init; }

    /// <summary>Sample rate of the rendered audio in Hz.</summary>
    public int SampleRate { get; init; }

    /// <summary>Free-form metadata surfaced to the caller (model, seed, timing).</summary>
    public IReadOnlyDictionary<string, string> Meta { get; init; } = new Dictionary<string, string>();
}
