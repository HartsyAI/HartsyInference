namespace HartsyInference.Engine.Requests;

/// <summary>Stem-separation request (Demucs): split a mix into its component stems.</summary>
public sealed record FxSeparateRequest
{
    /// <summary>The mixed audio to separate.</summary>
    public required AudioClip Audio { get; init; }

    /// <summary>Model/variant selector (e.g. "htdemucs", "htdemucs_6s", "htdemucs_ft"); null uses the default.</summary>
    public string? Model { get; init; }

    /// <summary>Demucs "shift trick" repetitions: run the model on randomly time-shifted copies and average. Upstream documents this as improving SDR by up to 0.2 points and making prediction <c>shifts</c> times slower. 0 (default) disables it.</summary>
    public int Shifts { get; init; }

    /// <summary>Fractional overlap between successive segments in <c>apply_model</c>. Upstream default 0.25; its README notes this "can probably be reduced to 0.1 to improve a bit speed". Null keeps the checkpoint's configured value.</summary>
    public double? Overlap { get; init; }

    /// <summary>Segment length in seconds. Lower it to cut peak memory. Upstream recommends a minimum of 10 s for the older models, and documents that <b>Hybrid Transformer models support at most 7.8 s</b> — the pipeline clamps to the checkpoint's trained segment. Null keeps the configured value.</summary>
    public double? Segment { get; init; }

    /// <summary>Deterministic seed for the <see cref="Shifts"/> offsets, so a shifted run is reproducible.</summary>
    public int Seed { get; init; }
}
