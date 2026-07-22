namespace HartsyInference.Engine.Requests;

/// <summary>The result of a stem separation: each named stem (drums/bass/other/vocals/…) as its own encoded clip.</summary>
public sealed record StemsResult
{
    /// <summary>Stem name → encoded audio bytes, in source order.</summary>
    public required IReadOnlyDictionary<string, byte[]> Stems { get; init; }

    /// <summary>Container/codec of each stem (e.g. "wav").</summary>
    public string Format { get; init; } = "wav";

    /// <summary>Sample rate of the stems in Hz.</summary>
    public int SampleRate { get; init; }
}
